import argparse
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os
import csv

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict

from data_loader import get_pku_30k_test, get_do_not_answer, get_harm_bench, get_hh_rlhf_safety, get_salad_bench

def parse_arguments():
    
    model_name = "Pythia_GBSR_pku_30k_harmless"
    # dataset / benchmark list: 
    # pku_30k
    # do_not_answer
    # harm_bench
    # hh_rlhf_safety
    # salad_bench
    data_name = "pku_30k"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,default=f"Pythia_GBSR_pku_30k_harmless", help="模型路径")
    parser.add_argument('--model', type=str,default=f"/xxx/ShaPO/JudgeEval/", help="模型路径")
    parser.add_argument('--data_name', type=str, default=data_name, help="PKU-SafeRLHF-30K 镜像路径")
    parser.add_argument('--data_num',type = int,default=2000)
    parser.add_argument('--round_name', type=str, default="round0", help="子目录名，例如 round0")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default=f"/xxx/ShaPO/JudgeEval/InferOutput/", help="输出目录 (脚本内部拼文件名)")
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--max_token',type = int,default=256)
    parser.add_argument('--seed',type = int,default=42)
    parser.add_argument('--range',type = int,default=2000)

    args = parser.parse_args()

    # ✅ 默认值（如果用户没传，就跑 pku_30k）
    if args.data_name is None:
        args.data_name = ["pku_30k"]
    else:
        args.data_name = args.data_name.split(",")

    args.model = args.model + args.model_name
    args.output_dir = args.output_dir + args.model_name
    return args

def generate_with_retry(llm, prompts, sampling_params):
    def _run(ps):
        outs = llm.generate(ps, sampling_params)
        texts = []
        for o in outs:
            if hasattr(o, "outputs") and o.outputs:
                texts.append(o.outputs[0].text)
            else:
                texts.append("")
        return [t.replace("</s>", "").strip() for t in texts]

    results = _run(prompts)
    retry_idxs = [i for i, r in enumerate(results) if r == ""]
    if retry_idxs:
        retry_prompts = [prompts[i] for i in retry_idxs]
        retry_outs = _run(retry_prompts)
        for idx_i, rid in enumerate(retry_idxs):
            alt = retry_outs[idx_i]
            results[rid] = alt if alt != "" else "<no_output>"
    return results

def main():
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(model=args.model, tensor_parallel_size=args.world_size)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_token, stop=["\n"])

    for data_name in args.data_name:
        if data_name == "pku_30k":
            query_list = get_pku_30k_test()
        elif data_name == "do_not_answer":
            query_list = get_do_not_answer()
        elif data_name == "harm_bench":
            query_list = get_harm_bench()
        elif "hh_rlhf_safety" in data_name:
            query_list = get_hh_rlhf_safety()
        elif "salad_bench" in data_name:
            query_list = get_salad_bench()
        else:
            query_list = []

        print("Loaded {} samples from split test".format(len(query_list)))
        
        # 准备 prompt / 人类偏好 / meta
        prompts = []
        for query in query_list:
            prompts.append(f"\n\nHuman: {query}\n\nAssistant:")

        uniq_prompts = list(set(prompts))
        if "all" not in data_name:
            uniq_prompts = uniq_prompts[:args.data_num]

        # 生成模型回答
        print("Generate model responses ...")
        model_outs = generate_with_retry(llm, uniq_prompts, sampling_params)

        # 拼文件名
        # 例如：T_{temperature}_frac{data_frac}_{round}_{split}.jsonl
        # dir,base = os.path.split(args.output_dir)
        out_path = args.output_dir + f'_{data_name}.csv'

        # out_path.parent.mkdir(parents=True,exist_ok=True)
        written = 0
        with open(out_path, "w", encoding="utf-8", newline="") as fw:
            writer = csv.DictWriter(
                fw,
                fieldnames=["idx", "query", "response"],
            )
            writer.writeheader()
            for i, p in enumerate(uniq_prompts):
                model_r = model_outs[i]

                clean_p = p.replace("\n\nHuman: ", "").replace("\n\nAssistant:", "")
                writer.writerow(
                    {
                        "idx": i,
                        "query": clean_p,
                        "response": model_r,
                    }
                )

        print(f"Wrote {written} records to {out_path}")

if __name__ == "__main__":
    main()
