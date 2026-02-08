import datetime
import torch
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import logging, os, datetime, sys
from preference_datasets import get_batch_iterator, tokenize_batch_element, get_collate_fn
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
import os
import re

from MemTracker import MemTracker
import math


def compute_update_dominance_stats(
    losses: torch.Tensor,            # (B,) per-sample loss (after flip mixing, before winsor最好)
    z: torch.Tensor | None = None,   # (B,) = beta * logits, optional
    base_q: float = 0.85,            # 用于定义 tail: loss > quantile(loss, base_q)
    topk_fracs=(0.01, 0.05, 0.10),   # Top-k energy ratios
    gate_scale=None,
    energy=None,
    tail_ref=None,
    eps: float = 1e-12,
):
    """
    Returns a dict of dominance statistics (Python floats).
    - If z is provided: energy proxy uses DPO gate |sigmoid(z)-1|^2
    - If z is None: fallback energy proxy uses normalized loss magnitude
    """
    assert losses.dim() == 1
    x = losses.detach().float()
    B = x.numel()

    out = {}
    out["B"] = float(B)

    # ---- Energy proxy e_i ----
    if z is not None:
        zz = z.detach().float()
        p = torch.sigmoid(zz)
        gate = (p - 1.0).abs()                 # |dℓ/dz| for -logσ(z)
        if gate_scale is not None:
            gate = gate * gate_scale
        if energy is not None:
            e = energy.detach().float()
        else:
            e = gate.pow(2)                        # energy proxy
        out["z_mean"] = float(zz.mean().item())
        out["z_neg_frac"] = float((zz < 0).float().mean().item())
        out["z_neg_mag"] = float(F.relu(-zz).mean().item())
        out["gate_mean"] = float(gate.mean().item())
        out["gate_p95"]  = float(torch.quantile(gate, 0.95).item())
    else:
        # fallback: 用 loss 相对中位数的正偏离作为“难样本能量”
        med = torch.median(x)
        mag = F.relu(x - med)
        scale = mag.mean().clamp_min(eps)
        if energy is not None:
            e = energy.detach().float()
        else:
            e = (mag / scale).pow(2)
        out["loss_med"] = float(med.item())

    e_sum = e.sum().clamp_min(eps)
    p_e = e / e_sum

    # ---- Concentration metrics ----
    # HHI = sum p_i^2 ; ESS = 1/HHI
    hhi = (p_e * p_e).sum().clamp_min(eps)
    ess = 1.0 / hhi
    out["HHI"] = float(hhi.item())
    out["ESS"] = float(ess.item())
    out["ESS_frac"] = float((ess / max(B, 1)).item())  # 归一化到[0,1]，更好看

    # ---- Top-k energy ratios ----
    e_sorted, _ = torch.sort(e, descending=True)
    csum = torch.cumsum(e_sorted, dim=0)
    for frac in topk_fracs:
        k = max(1, int(round(frac * B)))
        rk = (csum[k - 1] / e_sum).clamp(0, 1)
        out[f"R{int(frac*100)}"] = float(rk.item())    # e.g. R10, R5, R1

    # ---- Tail energy share defined by loss quantile ----
    q = max(0.0, min(1.0, base_q))
    ref = tail_ref.detach().float() if tail_ref is not None else x
    T = torch.quantile(ref, q)
    mask = ref > T

    out["tail_T"] = float(T.item())
    out["tail_out_frac"] = float(mask.float().mean().item())
    out["tail_energy_share"] = float((e[mask].sum() / e_sum).item()) if mask.any() else 0.0

    # ---- Optional: sanity stats about losses ----
    out["loss_mean"] = float(x.mean().item())
    out["loss_p95"]  = float(torch.quantile(x, 0.95).item())
    out["loss_p99"]  = float(torch.quantile(x, 0.99).item())

    return out

def sparsemax_1d(z: torch.Tensor) -> torch.Tensor:
    """
    Sparsemax over a 1D vector z of shape (B,).
    Returns p of shape (B,), p>=0, sum p = 1, and p is sparse.
    """
    assert z.dim() == 1
    z = z - z.max()  # stability

    z_sorted, _ = torch.sort(z, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=0)
    k = torch.arange(1, z.numel() + 1, device=z.device, dtype=z.dtype)

    # support: 1 + k*z_k > sum_{j<=k} z_j
    support = 1 + k * z_sorted > z_cumsum
    k_z = support.sum().clamp(min=1)

    tau = (z_cumsum[k_z - 1] - 1) / k_z.to(z.dtype)
    p = torch.clamp(z - tau, min=0)
    return p

def budgeted_flip_prob_sparsemax(
    z: torch.Tensor,
    rho: float = 0.1,
    tau_select: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    z: (B,) = beta * logits (detach/float ok)
    rho: budget fraction, e.g. 0.1 means sum(pi) = 0.1*B
    tau_select: selection temperature (smaller => more aggressive/sparser)
    returns: flip_prob pi in [0,1], shape (B,), sum(pi)=rho*B (up to clamp effects)
    """
    assert z.dim() == 1
    B = z.numel()
    budget = rho * B

    # Two-direction losses in z-space
    L_wl = -F.logsigmoid(z)     # as-labeled
    L_lw = -F.logsigmoid(-z)    # flipped

    # Flip gain: how much loss decreases if we flip
    g = (L_wl - L_lw)  # big positive => flipping helps a lot

    # Scores for selecting flip mass (only g>0 should matter)
    # You can clamp g at 0 to avoid wasting budget on non-beneficial flips.
    scores = (torch.clamp(g, min=0.0) / max(tau_select, 1e-6))

    # If everything is zero, no flip
    if (scores <= 0).all():
        return torch.zeros_like(z)

    # Sparse distribution over samples
    a = sparsemax_1d(scores.float()).to(z.dtype)  # sum(a)=1, many zeros

    # Allocate budget mass over a
    pi = budget * a  # sum(pi)=budget
    pi = torch.clamp(pi, 0.0, 1.0)

    # If clamping caused sum(pi) < budget (rare when rho<=0.1), renormalize remaining mass
    s = pi.sum().item()
    if s + 1e-6 < budget:
        remaining = budget - s
        free = (pi < 1.0)
        if free.any():
            a_free = a[free]
            denom = a_free.sum().clamp_min(eps)
            pi[free] = torch.clamp(pi[free] + remaining * (a_free / denom), 0.0, 1.0)

    return pi

def budgeted_winsor_cap_all(
    losses: torch.Tensor,
    rho_w: float = 0.1,
    base_q: float = 0.9,
    gamma: float = 1.0,      # gamma>1 更偏向极端点
    eps: float = 1e-12,
):
    assert losses.dim() == 1
    B = losses.numel()
    budget = rho_w * B

    x = losses.detach().float()
    T = torch.quantile(x, max(0.0, min(1.0, base_q)))
    mask = x > T
    m = int(mask.sum().item())

    if m == 0 or budget <= 0:
        r = torch.zeros_like(losses)
        return losses.mean(), r, T.to(losses.dtype), 0.0, losses

    s = torch.relu(x - T)                      # severity
    s = (s ** gamma)                           # 可选：强调极端
    s_mask = s[mask]
    denom = s_mask.sum().clamp_min(eps)

    r = torch.zeros_like(losses)
    r[mask] = (budget * (s_mask / denom)).clamp(0.0, 1.0)

    # 如果 clamp 到 1 导致预算没花完，可选二次分配（先不写也行，影响不大）
    T_t = T.to(losses.dtype)
    loss_win = (1.0 - r) * losses + r * T_t
    active_frac = float(mask.float().mean().item())
    return loss_win.mean(), r.detach(), T_t.detach(), active_frac, loss_win


def rho_w_from_neg_margin(
    z: torch.Tensor,             # (B,) = beta * logits
    rho_w_max: float = 0.20,      # only knob to tune
    rho_floor: float = 0.02,      # fixed constant, not tuned
    eps: float = 1e-12,
) -> float:
    """
    Adaptive winsor budget with a fixed floor.

    p = E[sigmoid(-z / s)], where s = E[|z|] is a self-normalized scale.
    rho_w = rho_floor + (rho_w_max - rho_floor) * p
    """
    with torch.no_grad():
        z = z.detach().float()
        s = z.abs().mean().clamp_min(eps)      # self-normalized scale
        p = torch.sigmoid(-z / s).mean()       # in (0,1)
        rho_w = rho_floor + (rho_w_max - rho_floor) * p
        # safety clamp (handles mis-specified bounds)
        lo = min(rho_floor, rho_w_max)
        hi = max(rho_floor, rho_w_max)
        return float(rho_w.clamp(lo, hi).item())

def merge_and_sample(batches: List[Dict[str, Union[List]]], 
                     batch_size: int) -> Dict[str, Union[List]]:
    merged = {}
    # 先把所有 batch 合并
    for batch in batches:
        for key, value in batch.items():
            if key not in merged:
                merged[key] = []
            merged[key].extend(value)

    # 从合并后的数据里随机抽样
    total_size = len(next(iter(merged.values())))
    indices = random.sample(range(total_size), batch_size)

    sampled = {}
    for key, value in merged.items():
        sampled[key] = [value[i] for i in indices]
    return sampled

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    use_reward: bool = False,
                    reward_preference_probability: torch.FloatTensor | None = None,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False,
                    loss_name2: str = "",
                    simpo_gamma_beta_ratio: float = 0.5,
                    rdpo_epsilon: float = 0.1,
                    flip_prob: torch.Tensor | None = None) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0
    if loss_name2  == 'simpo':
        logits = pi_logratios - simpo_gamma_beta_ratio
        rank0_print(f"use simpo loss, gamma/beta ratio={simpo_gamma_beta_ratio}")
    else:
        logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    z = beta * logits

    if use_reward:
        p_theta = torch.sigmoid(beta * logits)
        losses = - (reward_preference_probability * torch.log(p_theta+1e-12) + (1-reward_preference_probability)*torch.log(1-p_theta+1e-12))
    else:
        if ipo:
            losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        elif loss_name2 == 'rdpo':
            L_wl, L_lw = - F.logsigmoid(beta * logits), - F.logsigmoid(-beta * logits)
            losses = ((1 - rdpo_epsilon) * L_wl - rdpo_epsilon * L_lw) / (1 - 2 * rdpo_epsilon)
            rank0_print(f"use rdpo loss, epsilon={rdpo_epsilon}")
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            # if label_smoothing == 0 --> losses = -F.logsigmoid(beta * logits)
            L_wl = -F.logsigmoid(z)
            L_lw = -F.logsigmoid(-z)
            if flip_prob is not None:
                # 建议 flip_prob 不反传（在外面 no_grad/detach）
                losses = (1.0 - flip_prob) * L_wl + flip_prob * L_lw
            else:
                losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    


    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    target_device = None
    if batch.get("image_grid_thw", None) is not None:
        grid = batch["image_grid_thw"]
        concatenated_grid = torch.cat((grid, grid), dim=0)
        concatenated_batch["image_grid_thw"] = concatenated_grid
        target_device = concatenated_grid.device  # 目标设备
    if batch.get("pixel_values", None) is not None:
        # print('torch.cat(batch["pixel_values"], dim=0)', batch["pixel_values"])
        tmp = torch.cat(batch["pixel_values"], dim=0)
        concatenated_batch["pixel_values"] = torch.cat((tmp, tmp,), dim=0).to(target_device)
        # concatenated_batch["pixel_values"] = torch.cat((tmp, tmp,), dim=0)
    
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, helpful_reward_model=None, harmless_reward_model=None):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        #新增
        self.best_gamma = 0.0


        self.mem = MemTracker(enable=True, rank=self.rank)
        
        
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        if "VL" in tokenizer_name_or_path:
            min_pixels = 128*28*28
            max_pixels = 256*28*28
            self.tokenizer = transformers.AutoProcessor.from_pretrained(tokenizer_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
            # if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_bos_token = False

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs), noise_rate=config.noise_rate, harmless_rate=config.harmless_rate)
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(
            **data_iterator_kwargs, 
            split='eval' if not any(item in config.datasets for item in ["hh", "hh_helpful", "hh_harmless"]) else 'test',
            n_examples=config.n_eval_examples, 
            batch_size=config.eval_batch_size, 
            silent=rank != 0, 
            cache_dir=get_local_dir(config.local_dirs), 
            harmless_rate=config.harmless_rate)
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded test data iterator')
        rank0_print(f'Loaded test data iterator')
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

        self.cur_step = 0
        self.num_steps = int(26752 / config.batch_size)
        self.start_rate = 0.3

        self.rho_f = 0.15
        self.rho_max = 0.5
        self.base_q = 0.7

        self.rho_f_tau_select = 1.0   

        

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        from torch.cuda.amp import autocast
        batch['prompt_input_ids'] = batch['prompt_input_ids'].to(device=self.rank)
        batch['prompt_attention_mask'] = batch['prompt_attention_mask'].to(device=self.rank)
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            with autocast(dtype=torch.float16):
                policy_output = self.policy.generate(
                    batch['prompt_input_ids'],
                    attention_mask=batch.get('prompt_attention_mask', None),
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    synced_gpus=True
                )

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        if "VL" in self.config['model']['name_or_path']:
            inputs = {
                    "input_ids": concatenated_batch['concatenated_input_ids'], "attention_mask": concatenated_batch['concatenated_attention_mask'], "pixel_values": concatenated_batch["pixel_values"], "image_grid_thw": concatenated_batch["image_grid_thw"]
                }
            all_logits = model(**inputs).logits.to(torch.float32)
        else:
            all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        if self.config.loss.name2 == 'simpo':
            all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=True)
        else:
            all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps


    # preference probability from reward
    def get_preference_from_reward_model(
        self, 
        prompt_list: list[str], 
        chosen_list: list[str], 
        rejected_list: list[str],
        label_type_list: list[str]):
        chosen_pairs, rejected_pairs,  = [], []
        for index in range(len(prompt_list)):
            # chosen_pairs.append((f"\n\nHuman: {prompt_list[index]} \n\nAssistant:", chosen_list[index]))
            # rejected_pairs.append((f"\n\nHuman: {prompt_list[index]} \n\nAssistant:", rejected_list[index]))
            chosen_pairs.append((prompt_list[index], chosen_list[index]))
            rejected_pairs.append((prompt_list[index], rejected_list[index]))

        def score_pairs(pairs: List[Tuple[str, str]],
                max_length: int = 1024,
                is_helpful: bool = True):
            if is_helpful:
                reward_tokenizer = self.helpful_reward_tokenizer
                reward_model = self.helpful_reward_model
            else:
                reward_tokenizer = self.harmless_reward_tokenizer
                reward_model = self.harmless_reward_model
            with torch.no_grad():
                qs, as_ = zip(*pairs)
                inputs = reward_tokenizer(
                    list(qs),
                    list(as_),                 # 作为 pair 输入
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                logits = reward_model(**inputs).logits.squeeze(-1)
            batch_scores = logits.detach()
            return batch_scores
        helpful_chosen_score = score_pairs(chosen_pairs, 1024, True)
        helpful_rejected_score = score_pairs(rejected_pairs, 1024, True)
        harmless_chosen_score = score_pairs(chosen_pairs, 1024, False)
        harmless_rejected_score = score_pairs(rejected_pairs, 1024, False)

        score_difference = helpful_chosen_score - helpful_rejected_score
        
        harmless_score_difference = harmless_chosen_score - harmless_rejected_score
        rank0_print("score_difference", score_difference.tolist())
        rank0_print("label_type_list", label_type_list)
        rank0_print("harmless_score_difference", harmless_score_difference.tolist())
        for index, label_type in enumerate(label_type_list):
            if label_type == "harmless":
                score_difference[index] = harmless_score_difference[index]
        rank0_print("score_difference", score_difference.tolist())

        return torch.sigmoid(self.config.reward_beta * score_difference)

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True, use_reward=False):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""
        metrics = {}

        
        train_test = 'train' if train else 'eval'
        rho_w_use = 0.15
        
        if loss_config.name in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            # add reward probability
            if use_reward:
                reward_preference_probability = self.get_preference_from_reward_model(
                    batch["prompt"], 
                    batch["chosen_response_only"], 
                    batch["rejected_response_only"],
                    batch["label_type"])
                rank0_print("reward_probability", reward_preference_probability)
            else:
                reward_preference_probability = None
            
            with torch.no_grad():
                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = reference_chosen_logps - reference_rejected_logps
                logits = pi_logratios - ref_logratios
                z_stat = (loss_config.beta * logits).detach().float()   # (B,)
                z = z_stat
                p = torch.sigmoid(z)

                loss_raw = -F.logsigmoid(z)

                tail_ref_fixed = loss_raw.detach().float()
            # 计算 DPO 的 tail 状态
            dom_raw = compute_update_dominance_stats(
                losses=loss_raw,
                z=z_stat,                 # 关键：用 z 的 gate 能量 proxy
                base_q=self.base_q,
                topk_fracs=(0.01, 0.05, 0.10),
                tail_ref=tail_ref_fixed,
            )

            for k, v in dom_raw.items():
                metrics[f"dom/{train_test}_row_{k}"] = [v]

            # Label Flip
            flip_prob = None
            # if train:
            cur_step = self.cur_step
            num_steps = self.num_steps

            enable_flip = False
            if (cur_step is not None) and (num_steps is not None):
                enable_flip = (cur_step >= self.start_rate * num_steps)
                

            if enable_flip:
                rank0_print("STARTFUCKINGFLIP")
                with torch.no_grad():
                    # 复现 logits / z（与 preference_loss 一致）
                    pi_logratios = policy_chosen_logps - policy_rejected_logps
                    ref_logratios = reference_chosen_logps - reference_rejected_logps

                    logits = pi_logratios - ref_logratios

                    z = (loss_config.beta * logits).detach().float()
                    
                    rho_w_use = rho_w_from_neg_margin(
                        z=z,
                        rho_w_max=self.rho_max,
                    )
                    metrics[f'epip/{train_test}_neg_frac'] = [float((z < 0).float().mean().item())]
                    metrics[f'epip/{train_test}_neg_mag']  = [float(torch.relu(-z).mean().item())]
                    metrics[f'winsor/{train_test}_rho_w']  = [rho_w_use]

                    # ✅ 预算式稀疏 flip 概率（sum pi = rho*B）
                    flip_prob = budgeted_flip_prob_sparsemax(
                        z=z,
                        rho=self.rho_f,
                        tau_select=self.rho_f_tau_select,     # 选择温度：小->更硬更稀疏，建议扫 0.5/1/2
                    ).to(policy_chosen_logps.dtype).detach()
                    
            with torch.no_grad():
                if flip_prob is None:
                    flip_prob = torch.zeros_like(z)
                L_wl = -F.logsigmoid(z)
                L_lw = -F.logsigmoid(-z)
                loss_mix = (1.0 - flip_prob) * L_wl + flip_prob * L_lw  # ✅ flip 后的 per-sample loss
            with torch.no_grad():
                gate_flip = (p - 1.0 + flip_prob).abs()         # ✅ matches dℓ_mix/dz
                # 这里建议你让 compute_update_dominance_stats 支持 “外部能量 e”
                # 但如果你暂时不想改函数，可以先把 gate 通过 gate_scale 传进去不太好。
                # 更推荐：在 compute_update_dominance_stats 里新增参数 energy=e（见下）

            dom_flip = compute_update_dominance_stats(  # 你加一个新函数/或扩展原函数
                losses=loss_mix,
                energy=gate_flip.pow(2),
                base_q=self.base_q,
                topk_fracs=(0.01,0.05,0.10),
                # tail_ref=loss_mix,   # ✅ tail 定义在 flip 后
                tail_ref=tail_ref_fixed, 
            )
            for k,v in dom_flip.items():
                metrics[f"dom/{train_test}_flip_{k}"] = [v]
                        
            if loss_config.name == 'dpo' or loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': 'ipo' in loss_config.name or 'ipo' in loss_config.name2, 'use_reward': use_reward, 'reward_preference_probability': reward_preference_probability, "loss_name2": loss_config.name2, "rdpo_epsilon": self.config.loss.rdpo_epsilon, "simpo_gamma_beta_ratio": self.config.loss.simpo_gamma_beta_ratio, "flip_prob": flip_prob}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
            
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
        if loss_config.mode_loss == "DrDPO":
            return - loss_config.mode_weight * torch.log(torch.mean(torch.exp( - losses / loss_config.mode_weight))), metrics
        else:
            loss, r_w, T_w, out_frac, loss_win_vec = budgeted_winsor_cap_all(
                losses=losses,
                rho_w=rho_w_use,      # 预算强度（可做成动态）
                base_q=self.base_q,     # 仍然是按 loss 排序定义 outlier
            )


            loss_mean, r_w, T_w, out_frac, loss_win_vec = budgeted_winsor_cap_all(
                losses=loss_mix,      # ✅ winsor 应该作用在 flip 后 loss 上
                rho_w=rho_w_use,
                base_q=self.base_q,
            )

            # ===== Stage 2 dominance: apply (1-r) scaling to gate =====
            with torch.no_grad():
                w = (1.0 - r_w).to(z.dtype)
                gate_win = gate_flip * w                       # ✅ winsor rescales gradient
                e_win = gate_win.pow(2)

            dom_win = compute_update_dominance_stats(
                losses=loss_win_vec,
                energy=e_win,
                base_q=self.base_q,
                topk_fracs=(0.01,0.05,0.10),
                # tail_ref=loss_mix,   # ✅ tail 集合固定为 flip 后、winsor 前
                tail_ref=tail_ref_fixed,     # ✅ fixed
            )
            for k,v in dom_win.items():
                metrics[f"dom/{train_test}_win_{k}"] = [v]

            metrics[f'winsor/{train_test}_T'] = [float(T_w.item())]
            metrics[f'winsor/{train_test}_out_frac'] = [out_frac]
            metrics[f'winsor/{train_test}_r_mean'] = [float(r_w.float().mean().item())]
            metrics[f'winsor/{train_test}_r_max']  = [float(r_w.float().max().item())]
            metrics[f'winsor/{train_test}_r_eq1_frac'] = [float((r_w >= 0.999).float().mean().item())]
            metrics[f'winsor/{train_test}_r_p95'] = [float(torch.quantile(r_w.float(), 0.95).item())]
            
            return loss, metrics

    def build_logger(self):
        log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(log_dir, f"gamma_search_{ts}.log")

        logging.basicConfig(
            level=logging.INFO,                       # 全局最低等级
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),    # 打到屏幕
                logging.FileHandler(log_file)         # 也写文件
            ]
        )
        return logging.getLogger("gamma-search") 


    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        batch_collector = []
        interval_for_shapo = self.config.interval_for_shapo
        collate_fn = get_collate_fn(self.tokenizer)
        def do_eval_batch():
            rank0_print(f'Running evaluation after {self.example_counter} train examples')
            self.policy.eval()
            # 验证集    helpful 和 harmless 比例和训练集一样
            all_eval_metrics = defaultdict(list)
            for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                with torch.no_grad():
                    _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                for k, v in eval_metrics.items():
                    all_eval_metrics[k].extend(v)
            model_name = "Pythia"
            if "Llama" in self.config.model.name_or_path:
                model_name = "Llama"
            elif "llama" in self.config.model.name_or_path:
                model_name = "llama"
            elif "Qwen" in self.config.model.name_or_path:
                model_name = "Qwen"
            log_save_path = f"/home/y/yangyh/ljl/ShaPO/{model_name}Log"
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)
            log_file_name = self.config.loss.name2
            log_file_path = f"{log_save_path}/reward_{log_file_name}_{self.config.reward_beta}_{self.config.harmless_rate}_{'same_steps' if self.config.same_steps else 'more_steps'}.log"
            mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
            if self.rank == 0:
                with open(log_file_path, "a", encoding="utf-8") as f:  # "a" 表示追加写入
                    f.write(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}' + "\n")
        
        def train_log_write(metrics):
            model_name = "Pythia"
            if "Llama" in self.config.model.name_or_path:
                model_name = "Llama"
            elif "llama" in self.config.model.name_or_path:
                model_name = "llama"
            elif "Qwen" in self.config.model.name_or_path:
                model_name = "Qwen"
            
            log_save_path = f"/home/y/yangyh/ljl/ShaPO/{model_name}TrainLog"
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)
            log_file_name = self.config.loss.name2
            log_file_path = f"{log_save_path}/reward_{log_file_name}_{self.config.reward_beta}_{self.config.harmless_rate}_{'same_steps' if self.config.same_steps else 'more_steps'}.log"
            if self.rank == 0:
                with open(log_file_path, "a", encoding="utf-8") as f:  # "a" 表示追加写入
                    f.write(f'train after {self.example_counter}: {formatted_dict(metrics)}' + "\n")
        
        for batch in self.train_iterator:
            has_dpo = False
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                
                do_eval_batch()
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()
            log = self.build_logger()
            start_time = time.time()
            batch_metrics = defaultdict(list)


            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                with self.mem.region("get loss"):
                    loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                rank0_print(f"*** fucking loss ***: {loss}    microbatch_idx: {microbatch_idx}   rank:{self.rank}")
                

                with self.mem.region("loss backward"):
                    (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)
            
            with self.mem.region("optim_step"):
                grad_norm = self.clip_gradient()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size
            rank0_print("self.batch_counter DPO", self.batch_counter)
            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')
                train_log_write(mean_train_metrics)
                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

            self.cur_step += 1
            #### END TRAINING ####

        # 训练完成后再评测一次
        do_eval_batch()
        # 训练完成后输出文本结果
        if self.config.if_output:
            for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                policy_output_decoded, reference_output_decoded = self.get_batch_samples(local_eval_batch)
                # rank0_print("policy_output_decoded", policy_output_decoded)
                eval_batch_output = []
                for index in range(len(policy_output_decoded)):
                    eval_batch_output.append({
                        "prompt": eval_batch["prompt"][index],
                        "chosen": eval_batch["chosen_response_only"][index],
                        "rejected": eval_batch["rejected_response_only"][index],
                        "label_type": eval_batch["label_type"][index],
                        "policy_output": policy_output_decoded[index],
                        "reference_output": reference_output_decoded[index] if reference_output_decoded is not None else ""
                    })
                if self.rank == 0:
                    model_name = "Pythia"
                    if "Llama" in self.config.model.name_or_path:
                        model_name = "Llama"
                    elif "Qwen" in self.config.model.name_or_path:
                        model_name = "Qwen"
                    log_save_path = f"/home/y/yangyh/ljl/ShaPO/{model_name}TrainAfterInferenceOutputs"
                    if not os.path.exists(log_save_path):
                        os.makedirs(log_save_path)
                    file_path = f"{log_save_path}/reward_{self.config.loss.name2}_{self.config.reward_beta}_{self.config.harmless_rate}_{'same_steps' if self.config.same_steps else 'more_steps'}.jsonl"  
                    # 以追加模式循环写入
                    with open(file_path, 'a', encoding='utf-8') as f:
                        for item in eval_batch_output:
                            # 将每个字典转换为 JSON 字符串并写入文件，末尾加换行符
                            json_line = json.dumps(item, ensure_ascii=False) + '\n'
                            f.write(json_line)

        
    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, helpful_reward_model=None, harmless_reward_model= None):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, helpful_reward_model, harmless_reward_model)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class})
        print(f"当前正在启动rank {self.rank}")
        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=True,
            sync_module_states=True
        )
        


        print(f'Rank {self.rank} Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo'}:
            print(f'Rank {self.rank} Sharding reference / rewards model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
            # self.helpful_reward_model = FSDP(helpful_reward_model, **shared_fsdp_kwargs)
            # self.harmless_reward_model = FSDP(harmless_reward_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier(device_ids=[self.rank])
        print(f"Rank {self.rank} finished barrier sync.")

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        # save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
        #     optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        # if self.rank == 0:
        #     self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        # del optimizer_state_dict
        # dist.barrier()

        # if self.rank == 0:
        #     scheduler_state_dict = self.scheduler.state_dict()
        #     self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        # dist.barrier()
        

class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1, helpful_reward_model=None, harmless_reward_model= None):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, helpful_reward_model, harmless_reward_model)
        
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)
            rank0_print('Sharding reward model...')
            self.helpful_reward_model = tp.tensor_parallel(helpful_reward_model, sharded=False)
            self.harmless_reward_model = tp.tensor_parallel(harmless_reward_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        