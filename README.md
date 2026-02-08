# wDPO: Winsorized Direct Preference Optimization

**wDPO** is a robust variant of Direct Preference Optimization (DPO) for preference data with noise.
It targets two common issues in real datasets.
Some pairs are mislabeled and the preference direction is flipped.
Some pairs are ambiguous and produce a high-loss tail that can destabilize training.

wDPO adds two lightweight, batch-level steps on top of standard DPO.
First, it applies **sparse flip-aware loss mixing**.
It assigns nonzero flip weights only to a small set of strongly inconsistent pairs.
Second, it applies **soft winsorization** to the loss tail.
It caps only the largest per-sample losses toward a quantile threshold.
The cap strength is adaptive and uses only batch signals.
wDPO needs no extra reward model and does not filter data.

- 🔧 **Two-stage**: sparse flip-aware mixing + tail loss capping
- 📦 **No extra models**: uses only in-batch signals
- 🧩 **Drop-in**: works with standard DPO-style training stacks
- 🧱 **Backbones**: validated on Pythia-2.8B, LLaMA-3.2-3B, LLaMA-3-8B, and Qwen2.5-7B

---

## 📦 Installation

### 1) Create a clean Conda environment

```bash
conda create -n shapo python=3.10 -y
conda activate shapo
```

### 2) Install PyTorch (choose your CUDA / CPU build)

> Replace the command below to match your system: https://pytorch.org/get-started/locally/

```bash
# Example for CUDA 12.x (adjust if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install project dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Project Layout

```
.
├── README.md
├── requirements.txt
├── train.py
├── train_wDPO.py
├── trainers.py
├── trainers_wDPO.py
├── preference_datasets.py
├── configs/
├── datasets/                 # place/prep datasets
└── judeg_eval/               # code for eval
```

---

## 🚀 Quick Start

> Below are the **exact run commands** you provided, organized by task.  
> Tip: Use `CUDA_VISIBLE_DEVICES=...` or FSDP configs as needed on multi-GPU nodes.

### 1) Qwen3-8B-Base SFT (supervised fine-tuning)

Baseline SFT run (Hydra-style arguments).

```bash
python train.py   model=llama38b   datasets=[pku_30k_harmless]   loss=sft   exp_name=LLaMA3-8B-SFT   gradient_accumulation_steps=2   batch_size=64   eval_batch_size=32   trainer=FSDPTrainer   sample_during_eval=false   model.fsdp_policy_mp=bfloat16
```

### 2) LLaMA3-8B wDPO\*\*

Token-level ShaPO combines DPO with SAM-style perturbations on the identified subspace.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u train_wDPO.py model=llama38b datasets=[pku_30k_harmless] loss=dpo loss.beta=0.1 exp_name=llama38_DrShaPO_pku_30k_harmless gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/y/yangyh/ljl/ShaPO/.cache/yangyh/llama38_pku_30k_harmless_sft_2026-01-01_23-20-03_192292/LATEST/policy.pt loss.name2=dpo warmup_steps=10 max_grad_norm=10 n_eval_examples=256 eval_every=2048 reward_beta=10 harmless_rate=0.2 interval_for_shapo=5 same_steps=true if_output=false if_save=true
```

---

## 📚 Datasets

- **PKU-30K** (helpful & harmless preference pairs)

> Prepare or symlink them into `./data/` or update your config paths accordingly.

---

## ⚙️ Common Tips

- 🧮 **Precision**: `bfloat16` works well with FSDP; adjust for your hardware.
- 🧵 **FSDP**: Ensure PyTorch build and NCCL are compatible; set `NCCL_P2P_DISABLE=1` if you hit P2P issues.
- 💾 **Checkpoints**: `model.archive` should point to your SFT checkpoint (`.pt`) before launching ShaPO runs.
- 🧪 **Eval cadence**: Tune `eval_every` and `n_eval_examples` for your compute budget.

---
