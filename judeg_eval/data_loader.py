from datasets import load_dataset
import json
import random

dataset_folder = "./datasets"

def get_do_not_answer():
    dataset_name = "LibrAI/do-not-answer"
    ds = load_dataset(f"{dataset_folder}/{dataset_name}", split="train")

    return [item["question"] for item in ds]

def get_pku_30k_test():
    dataset_name = "pku-30k"
    dataset = load_dataset(f"{dataset_folder}/{dataset_name}")["test"]
    ds = dataset.shuffle(seed=42)
    return [item["prompt"] for item in ds]

def get_harm_bench():
    dataset_name = "walledai/HarmBench"
    ds = load_dataset(f"{dataset_folder}/{dataset_name}", name='standard', split="train")
    return [item["prompt"] for item in ds]

def get_hh_rlhf_safety():
    jsonl_path = f"{dataset_folder}/hh-rlhf-safety.jsonl"

    query_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query_list.append(obj["prompt"])

    rng = random.Random(42)
    rng.shuffle(query_list)

    return query_list

def get_salad_bench():
    dataset_name = "walledai/SaladBench"
    dataset = load_dataset(f"{dataset_folder}/{dataset_name}", name="prompts", split="base")
    ds = dataset.shuffle(seed=46)
    return [item["prompt"] for item in ds]