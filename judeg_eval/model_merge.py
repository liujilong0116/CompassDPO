import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/xxx/models/EleutherAI/pythia-2.8b"
output_dir = "/xxx/ShaPO/JudgeEval/Pythia_rDPO_30n_pku_30k_harmless"
pt_file = "/xxx/ShaPO/.cache/yangyh/pythia28_rDPO_pku_30k_harmless_30n_2026-01-27_22-46-22_085870/LATEST/policy.pt"

# 加载模型结构
llm = AutoModelForCausalLM.from_pretrained(model_path)

# 加载你训练好的参数
state_dict = torch.load(pt_file, map_location="cpu")
llm.load_state_dict(state_dict["state"], strict=False)

# 保存为 HuggingFace 可直接加载的格式
llm.save_pretrained(output_dir)

# 如果需要 tokenizer 也一并保存（通常需要）
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(output_dir)
