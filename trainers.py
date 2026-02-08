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
from preference_datasets import get_batch_iterator, tokenize_batch_element
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
                    # new
                    mode_loss: str = "",
                    mode_weight: float = 1.0) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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


    if ipo:
        base_losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    elif loss_name2 == 'rdpo':
        L_wl, L_lw = - F.logsigmoid(beta * logits), - F.logsigmoid(-beta * logits)
        base_losses = ((1 - rdpo_epsilon) * L_wl - rdpo_epsilon * L_lw) / (1 - 2 * rdpo_epsilon)
        rank0_print(f"use rdpo loss, epsilon={rdpo_epsilon}")
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        # if label_smoothing == 0 --> losses = -F.logsigmoid(beta * logits)
        base_losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    if use_reward:
        if reward_preference_probability is None:
            raise ValueError("use_reward=True but reward_preference_probability is None")
        p_theta = torch.sigmoid(beta * logits)
        losses = - (reward_preference_probability * torch.log(p_theta+1e-12)
                    + (1-reward_preference_probability)*torch.log(1-p_theta+1e-12))
    else:
        losses = base_losses

    if mode_loss == "DrDPO":
        # w = mode_weight
        # # 权重 ∝ exp(-base_loss / w)
        # # 建议 detach base_losses 做权重，符合“先分配权重再矫正”的两阶段语义，也更稳定
        # weights = torch.softmax((-base_losses.detach() / w), dim=0)  # (B,)
        # loss_agg = torch.sum(weights * losses)  # 标量
        
        # 原始方案
        w = mode_weight
        loss_agg = -w * torch.log(torch.mean(torch.exp(-losses / w)))
    else:
        loss_agg = losses.mean()

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return loss_agg, chosen_rewards, rejected_rewards, losses


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

        self.sharpo_radius = config.sharpo_radius
        
        
        self.sharpo_trials = config.sharpo_trials

        self.mem = MemTracker(enable=True, rank=self.rank)
        
        
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        if "VL" in tokenizer_name_or_path:
            min_pixels = 128*28*28
            max_pixels = 256*28*28
            self.tokenizer = transformers.AutoProcessor.from_pretrained(tokenizer_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
        # self.test_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs), harmless_rate=config.harmless_rate)
        # self.test_batches = list(self.test_iterator)
        rank0_print(f'Loaded test data iterator')
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')






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

    

    def get_preference_from_reward_model(
        self, 
        prompt_list: list[str], 
        chosen_list: list[str], 
        rejected_list: list[str],
        label_type_list: list[str]):

        reward_model = self.harmless_reward_model
        reward_tokenizer = self.harmless_reward_tokenizer

        def build_text(prompt: str, answer: str) -> str:
            return f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{answer}"

        def score_texts(texts: List[str], max_length: int = 2048) -> torch.Tensor:
            inputs = reward_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # device_map="auto" 时必须对齐 device
            device = reward_model.get_input_embeddings().weight.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = reward_model(**inputs)
                scores = out.end_scores.squeeze(-1)  # [batch]
            return scores.detach()
        
        
        chosen_texts = [build_text(p, c) for p, c in zip(prompt_list, chosen_list)]
        rejected_texts = [build_text(p, r) for p, r in zip(prompt_list, rejected_list)]

        chosen_score = score_texts(chosen_texts, max_length=2048)
        rejected_score = score_texts(rejected_texts, max_length=2048)

        score_difference = -(chosen_score - rejected_score)
        rank0_print("score_difference", score_difference.tolist())

        return torch.sigmoid(self.config.reward_beta * score_difference)


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True, use_reward=False):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""
        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo','sharpo'}:
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

            if loss_config.name == 'dpo' or loss_config.name=='sharpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False, 'use_reward': use_reward, 'reward_preference_probability': reward_preference_probability, "loss_name2": loss_config.name2, "rdpo_epsilon": self.config.loss.rdpo_epsilon, "simpo_gamma_beta_ratio": self.config.loss.simpo_gamma_beta_ratio, "mode_loss": loss_config.mode_loss, "mode_weight": loss_config.mode_weight}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'ipo': True, "loss_name2": loss_config.name2}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')


            loss, chosen_rewards, rejected_rewards, losses_per_example = preference_loss(
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

            policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

            all_devices_losses = all_gather_if_needed(losses_per_example.detach(), self.rank, self.world_size)
            metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

            return loss, metrics
        elif loss_config.name == 'sft':
            if "VL" in self.config['model']['name_or_path']:
                chosen_inputs = {
                    "input_ids": batch['chosen_input_ids'], "attention_mask": batch['chosen_attention_mask'], "pixel_values": torch.cat(batch["pixel_values"], dim=0).to(self.rank), "image_grid_thw": batch["image_grid_thw"]
                }
                print(chosen_inputs["pixel_values"])
                print(chosen_inputs["image_grid_thw"])
                policy_chosen_logits = self.policy(**chosen_inputs).logits.to(torch.float32)
                policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)
            else:
                policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
                policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

            policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

            all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
            metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

            return losses.mean(), metrics
        
            

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

        if self.config.loss.name in {'dpo', 'ipo','sharpo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        interval_for_shapo = self.config.interval_for_shapo
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
            elif "Qwen" in self.config.model.name_or_path:
                model_name = "Qwen"
            log_save_path = f"/xxx/{model_name}Log"
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)
            log_file_name = self.config.loss.name2
            log_file_path = f"{log_save_path}/reward_{log_file_name}_{self.config.reward_beta}_{self.config.harmless_rate}_{'same_steps' if self.config.same_steps else 'more_steps'}_{self.config.probe_top_k}.log"
            mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
            if self.rank == 0:
                with open(log_file_path, "a", encoding="utf-8") as f:  # "a" 表示追加写入
                    f.write(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}' + "\n")
        
        def train_log_write(metrics):
            model_name = "Pythia"
            if "Llama" in self.config.model.name_or_path:
                model_name = "Llama"
            elif "Qwen" in self.config.model.name_or_path:
                model_name = "Qwen"
            
            log_save_path = f"/xxx/{model_name}TrainLog"
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)
            log_file_name = self.config.loss.name2
            log_file_path = f"{log_save_path}/reward_{log_file_name}_{self.config.reward_beta}_{self.config.harmless_rate}_{'same_steps' if self.config.same_steps else 'more_steps'}.log"
            if self.rank == 0:
                with open(log_file_path, "a", encoding="utf-8") as f:  # "a" 表示追加写入
                    f.write(f'train after {self.example_counter}: {formatted_dict(metrics)}' + "\n")
        
        for batch in self.train_iterator:


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
                    loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True, use_reward=self.config.use_reward)
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
            
            #### END TRAINING ####

        # 训练完成后再评测一次
        do_eval_batch()
        
        
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
            if helpful_reward_model:
                self.helpful_reward_model = FSDP(helpful_reward_model, **shared_fsdp_kwargs)
            if harmless_reward_model:
                self.harmless_reward_model = FSDP(harmless_reward_model, **shared_fsdp_kwargs)
        
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
        