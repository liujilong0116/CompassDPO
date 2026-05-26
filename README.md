# CompassDPO: Dynamics-Controlled Direct Preference Optimization

**CompassDPO** is a dynamics-controlled variant of Direct Preference Optimization (DPO) for robust preference alignment.
It targets a training-dynamics failure mode of DPO: under imperfect preference supervision, a small subset of high-influence samples can distort mini-batch updates in direction or magnitude.

CompassDPO controls mini-batch updates along two axes.
First, **directional control** applies sparse, budgeted loss mixing after a warm-up period to reduce update components that conflict with the emerging preference direction.
Second, **magnitude control** applies adaptive soft winsorization to the high-loss tail, limiting tail-dominated update magnitude while preserving gradients from hard preference pairs.

CompassDPO operates within the standard DPO framework.
It uses only training-time signals already available in DPO and requires no external reward model, relabeling, or data reconstruction.

- 🧭 **Dynamics control**: directional control + magnitude control
- 📦 **No extra models**: uses only in-batch training signals
- 🧩 **Drop-in**: works with standard DPO-style training stacks
- 🧱 **Backbones**: validated on Pythia-2.8B, LLaMA-3.2-3B, LLaMA-3-8B, and Qwen2.5-7B

---

## 📦 Installation

### 1) Create a clean Conda environment

```bash
conda create -n compassdpo python=3.10 -y
conda activate compassdpo
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
├── train_CompassDPO.py
├── trainers.py
├── trainers_CompassDPO.py
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

Baseline SFT run

```bash
python train.py   model=llama38b   datasets=[pku_30k_harmless]   loss=sft   exp_name=LLaMA3-8B-SFT   gradient_accumulation_steps=2   batch_size=64   eval_batch_size=32   trainer=FSDPTrainer   sample_during_eval=false   model.fsdp_policy_mp=bfloat16
```

### 2) LLaMA3-8B wDPO

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u train_CompassDPO.py model=llama38b datasets=[pku_30k_harmless] loss=dpo loss.beta=0.1 exp_name=llama38_wDPO_pku_30k_harmless gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/xxx/llama38_pku_30k_harmless_sft_2026-01-01_23-20-03_192292/LATEST/policy.pt loss.name2=dpo warmup_steps=10 max_grad_norm=10 n_eval_examples=256 eval_every=2048 reward_beta=10 harmless_rate=0.2 same_steps=true if_output=false if_save=true
```

---

## 📚 Datasets

- **PKU-30K** (helpful & harmless preference pairs)

> Prepare or symlink them into `./data/` or update your config paths accordingly.

---

## ⚙️ Common Tips

- 🧮 **Precision**: `bfloat16` works well with FSDP; adjust for your hardware.
- 🧵 **FSDP**: Ensure PyTorch build and NCCL are compatible; set `NCCL_P2P_DISABLE=1` if you hit P2P issues.
- 💾 **Checkpoints**: `model.archive` should point to your SFT checkpoint (`.pt`) before launching wDPO runs.
- 🧪 **Eval cadence**: Tune `eval_every` and `n_eval_examples` for your compute budget.

---
