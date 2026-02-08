import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from qwen_vl_utils import process_vision_info


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    # dataset.save_to_disk("/root/autodl-tmp/huggingface/dataset")
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('/mnt/scratch/y/yangyh/datasets/datasets--stanfordnlp--SHP', split=split, cache_dir=cache_dir)
    # dataset.save_to_disk("/root/autodl-tmp/huggingface/dataset")
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_spa(split="train", silent=False, cache_dir=None):
    print(f'Loading SPA_VL dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('/mnt/scratch/y/yangyh/datasets/spavl', split="train", cache_dir=cache_dir)
    print('done')

    all_num = len(dataset)

    if split == "train":
        dataset = dataset.select(range(10000))
    elif split == "test":
        dataset = dataset.select(range(10000, 10512))
        

    data = defaultdict(lambda: defaultdict(list))
    for index, item in tqdm.tqdm(enumerate(dataset), desc='Processing SPA_VL', disable=silent):
        prompt = item["question"] + f"{index:05d}"
        responses = [item["chosen"], item["rejected"]]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['image'] = item["image"]
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = item["chosen"]
    return data
    

def get_hh(split: str, silent: bool = False, cache_dir: str = None, data_type: str = "all") -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    # dataset = datasets.load_dataset('/mnt/scratch/y/yangyh/datasets/hh', split=split, cache_dir=cache_dir)
    

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response
    data = defaultdict(lambda: defaultdict(list))
    
    def process_sub_dataset(sub_dataset, label_type="helpful"):
        for row in tqdm.tqdm(sub_dataset, desc='Processing HH', disable=silent):
            prompt, chosen, rejected = split_prompt_and_responses(row)
            responses = [chosen, rejected]
            n_responses = len(data[prompt]['responses'])
            data[prompt]['pairs'].append((n_responses, n_responses + 1))
            data[prompt]['responses'].extend(responses)
            data[prompt]['sft_target'] = chosen
            data[prompt]['label_type'].append(label_type)
    if data_type in ["all", "harmless"]:
        dataset = datasets.load_dataset('/mnt/scratch/y/yangyh/datasets/hh', split=split, data_dir="harmless-base",  cache_dir=cache_dir)
        process_sub_dataset(dataset, label_type="harmless")
    if data_type in ["all", "helpful"]:
        dataset = datasets.load_dataset('/mnt/scratch/y/yangyh/datasets/hh', split=split, data_dir="helpful-base",  cache_dir=cache_dir)
        process_sub_dataset(dataset, label_type="helpful")
        dataset = datasets.load_dataset('/mnt/scratch/y/yangyh/datasets/hh', split=split, data_dir="helpful-online",  cache_dir=cache_dir)
        process_sub_dataset(dataset, label_type="helpful")
        dataset = datasets.load_dataset('/mnt/scratch/y/yangyh/datasets/hh', split=split, data_dir="helpful-rejection-sampled",  cache_dir=cache_dir)
        process_sub_dataset(dataset, label_type="helpful")
    return data

def get_hh_pairwise_noise(split: str, silent: bool = False, cache_dir: str = None, noise_rate=0.) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    cnt_noise = 0
    for row in tqdm.tqdm(dataset, desc='Processing HH FROM NOISE {}'.format(noise_rate), disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)

        # Modify responses and sft_target for 20% of the training data
        if split == 'train' and random.random() < noise_rate:
            responses = [rejected, chosen]
            sft_target = rejected
            cnt_noise += 1
        else:
            responses = [chosen, rejected]
            sft_target = chosen

        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = sft_target
    print("{} NOISE RATIO:\t, len(dataset) is {}\t len(noise) is {}".format(split, len(dataset), cnt_noise), cnt_noise / len(dataset))
    return data

def get_pku_10k(split: str, silent: bool = False, cache_dir: str = None, label_type="helpful"):
    # \n\nHuman: <prompt>\n\nAssistant:
    print(f'Loading PKU_10K dataset ({split} split) from Huggingface...')
    TARGET_TEST_SIZE = 512
    # 1. 加载数据并打乱
    dataset = datasets.load_dataset("/mnt/scratch/y/yangyh/datasets/pkusaferlhf", cache_dir=cache_dir)["train"]
    dataset = dataset.shuffle(seed=42)

    # 2. 建立 prompt -> index_list 映射
    prompt_to_indices = defaultdict(list)
    for i, row in enumerate(dataset):
        prompt_to_indices[row["prompt"]].append(i)

    # 3. 按每个 prompt 的样本数排序（优先使用短的）
    prompts_sorted = sorted(prompt_to_indices.items(), key=lambda x: len(x[1]))

    # 4. 依次累加，直到测试集样本数 >= 512
    test_indices = []
    for prompt, idx_list in prompts_sorted:
        if len(test_indices) + len(idx_list) > TARGET_TEST_SIZE:
            break
        test_indices.extend(idx_list)

    # 5. 其余作为训练集
    all_indices = set(range(len(dataset)))
    train_indices = sorted(list(all_indices - set(test_indices)))
    test_indices = sorted(test_indices)

    if split == "train":
        dataset = dataset.select(train_indices)
    else:
        dataset = dataset.select(test_indices)
    
    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing PKU_10K', disable=silent):
        prompt = f"\n\nHuman: {row['prompt']}\n\nAssistant:"
        data_response = [row['response_0'], row['response_1']]
        if label_type == "helpful":
            chosen= data_response[row['better_response_id']]
            rejected = data_response[1 - row['better_response_id']]
        elif label_type == "harmless":
            chosen= data_response[row['safer_response_id']]
            rejected = data_response[1 - row['safer_response_id']]
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['label_type'].append(label_type)
        data[prompt]['sft_target'] = chosen
    return data

def get_pku_30k(split: str, silent: bool = False, cache_dir: str = None, label_type="helpful", noise=0):
    # \n\nHuman: <prompt>\n\nAssistant:
    print(f'Loading PKU_10K dataset ({split} split) from Huggingface...')
    if split == "train":
        dataset = datasets.load_dataset("/mnt/scratch/y/yangyh/datasets/pku-30k", cache_dir=cache_dir)["train"]
    else:
        dataset = datasets.load_dataset("/mnt/scratch/y/yangyh/datasets/pku-30k", cache_dir=cache_dir)["test"]
    dataset = dataset.shuffle(seed=42)

    n = len(dataset)
    k = int(n * noise)
    idx_list = list(range(n))
    rng = random.Random(42)
    rng.shuffle(idx_list) 
    noise_list = idx_list[:k]
    data = defaultdict(lambda: defaultdict(list))


    count = 0


    for index, row in tqdm.tqdm(enumerate(dataset), desc='Processing PKU_30K', disable=silent):
        prompt = f"\n\nHuman: {row['prompt']}\n\nAssistant:"
        if index not in noise_list:
            data_response = [row['response_0'], row['response_1']]
        else:
            data_response = [row['response_1'], row['response_0']]
        if label_type == "helpful":
            chosen= data_response[row['better_response_id']]
            rejected = data_response[1 - row['better_response_id']]
        elif label_type == "harmless":
            chosen= data_response[row['safer_response_id']]
            rejected = data_response[1 - row['safer_response_id']]
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['label_type'].append(label_type)
        data[prompt]['sft_target'] = chosen
        count += 1
        if count > 2048:
            break
    return data

def get_pku_10k_mixed(split: str, silent: bool = False, cache_dir: str = None, harmless_rate=0.):
    print(f'Loading PKU_10K dataset mixed from Huggingface...')
    dataset = datasets.load_dataset("/mnt/scratch/y/yangyh/datasets/pkusaferlhf", cache_dir=cache_dir)["train"]
    dataset = dataset.shuffle(seed=42)
    total_size = len(dataset)
    train_data_proportion = 0.8
    eval_data_proportion = 0.1
    test_data_proportion = 0.1

    # train_size = int(train_data_proportion * total_size)
    # eval_size = int(eval_data_proportion * total_size)
    # if split == "train":
    #     dataset = dataset.select(range(train_size))
    # elif split == "eval":
    #     dataset = dataset.select(range(train_size, train_size + eval_size))
    # else:
    #     dataset = dataset.select(range(train_size + eval_size, total_size))
    #     harmless_rate = 0.5
    train_size = 9728
    # train_size = 64
    if split == "train":
        dataset = dataset.select(range(train_size))
    elif split == "eval":
        dataset = dataset.select(range(train_size, total_size))
        harmless_rate = 0.5
    
    data = defaultdict(lambda: defaultdict(list))
    dataset_len = len(dataset)
    for index, row in tqdm.tqdm(enumerate(dataset), desc='Processing PKU_10K', disable=silent):
        prompt = f"\n\nHuman: {row['prompt']}\n\nAssistant:"
        data_response = [row['response_0'], row['response_1']]
        if index >= harmless_rate * dataset_len:
            chosen= data_response[row['better_response_id']]
            rejected = data_response[1 - row['better_response_id']]
            data[prompt]['label_type'].append("helpful")
        else:
            chosen= data_response[row['safer_response_id']]
            rejected = data_response[1 - row['safer_response_id']]
            data[prompt]['label_type'].append("harmless")
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
    return data


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, noise_rate=0., harmless_rate=0.):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir, data_type="all")
    elif name == 'hh_harmless':
        data = get_hh(split, silent=silent, cache_dir=cache_dir, data_type="harmless")
    elif name == 'hh_helpful':
        data = get_hh(split, silent=silent, cache_dir=cache_dir, data_type="helpful")
    elif name == 'hh_pairwise_noise':
        data = get_hh_pairwise_noise(split, silent=silent, cache_dir=cache_dir, noise_rate=noise_rate)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name == "spa":
        data = get_spa(split, silent=silent, cache_dir=cache_dir)
    elif name == "pku_helpful":
        data = get_pku_10k(split, silent=silent, cache_dir=cache_dir, label_type="helpful")
    elif name == "pku_harmless":
        data = get_pku_10k(split, silent=silent, cache_dir=cache_dir, label_type="harmless")
    elif name == "pku_30k_helpful":
        data = get_pku_30k(split, silent=silent, cache_dir=cache_dir, label_type="helpful")
    elif name == "pku_30k_harmless":
        data = get_pku_30k(split, silent=silent, cache_dir=cache_dir, label_type="harmless")
    elif name == "pku_30k_harmless_30n":
        data = get_pku_30k(split, silent=silent, cache_dir=cache_dir, label_type="harmless", noise=0.3)
    elif name == "pku_mixed":
        data = get_pku_10k_mixed(split, silent=silent, cache_dir=cache_dir, harmless_rate=harmless_rate)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    # assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
    #     f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

def tokenize_batch_vl(batch, tokenizer) -> Dict:

    prompt_messages = [[
            {'role': 'user', 'content': [{'type': 'image', 'image': item[3]}, {'type': 'text', 'text': item[0]}]}
        ] for item in batch]
    chosen_massages = [[
            {'role': 'user', 'content': [{'type': 'image', 'image': item[3]}, {'type': 'text', 'text': item[0]}]}, 
            {'role': 'assistant', 'content': [{'type': 'text', 'text': item[1]}]}
        ] for item in batch]
    rejected_messages = [[
            {'role': 'user', 'content': [{'type': 'image', 'image': item[3]}, {'type': 'text', 'text': item[0]}]}, 
            {'role': 'assistant', 'content': [{'type': 'text', 'text': item[2]}]}
        ] for item in batch]
    
    prompt_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in prompt_messages]
    chosen_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in chosen_massages]
    rejected_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in rejected_messages]
    
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    prompt_inputs = tokenizer(
        text=prompt_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    chosen_inputs = tokenizer(
        text=chosen_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    rejected_inputs = tokenizer(
        text=rejected_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    chosen_inputs_id_lists = chosen_inputs['input_ids'].tolist()
    chosen_labels_list = []
    for ids_list in chosen_inputs_id_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        chosen_labels_list.append(label_ids)
    chosen_labels_ids = torch.tensor(chosen_labels_list, dtype=torch.int64)

    rejected_inputs_id_lists = rejected_inputs['input_ids'].tolist()
    rejected_labels_list = []
    for ids_list in rejected_inputs_id_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        rejected_labels_list.append(label_ids)
    rejected_labels_ids = torch.tensor(rejected_labels_list, dtype=torch.int64)

    
    def split_pixel_values(
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor
    ) -> List[torch.Tensor]:
        chunks = []
        start = 0
        for thw in image_grid_thw.tolist():
            _, h, w = thw
            n = h * w
            chunks.append(pixel_values[start:start+n])
            start += n
        return chunks
    pixel_values_list = split_pixel_values(prompt_inputs["pixel_values"], prompt_inputs["image_grid_thw"])

    return {
        "prompt": prompt_texts,
        "chosen": chosen_texts,
        "rejected": rejected_texts,
        "prompt_input_ids": prompt_inputs["input_ids"],
        "prompt_attention_mask": prompt_inputs["attention_mask"],
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "chosen_labels": chosen_labels_ids,
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"],
        "rejected_labels": rejected_labels_ids,
        "pixel_values": pixel_values_list,
        "image_grid_thw": prompt_inputs["image_grid_thw"],
        # "chosen_inputs": chosen_inputs
    }

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       noise_rate=0.,
                       harmless_rate=0.) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    print(f"fucking {split} {n_epochs} {n_examples}")
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        # print("permutation_seeds", permutation_seeds)
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, noise_rate=noise_rate, harmless_rate=harmless_rate).items():
                if name == 'spa':
                    # prompt 在构建的时候加入了 5 位的 index，需要去掉
                    # 相比于纯文本数据集多了一个 image
                    flat_data.append((prompt[:-5], data['responses'], data['pairs'], data['sft_target'], data["image"], truncation_mode))
                else:
                    flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode, data["label_type"]))
                    # if "mixed" in name:
                    #     flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode, data["label_type"]))
                    # else:
                    #     flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        if name == "spa":
            for prompt, responses, pairs, sft_target, image, truncation_mode in flat_data:
                if done:
                    break
                batch.append([prompt, responses[0], responses[1], image])
                example_idx += 1
                if len(batch) == batch_size:
                    result_tmp = tokenize_batch_vl(batch, tokenizer)
                    yield result_tmp
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []

        else:
            for prompt, responses, pairs, sft_target, truncation_mode, label_type in flat_data:
                if done:
                    break
                if sft_mode:
                    batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}

                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'Finished generating {n_examples} examples on {split} split')
                            done = True
                        # print(batch)
                        batch = []
                else:
                    for index, p in enumerate(pairs):
                        if done:
                            break
                        batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                        if len(label_type) > 0:
                            batch_element["label_type"] = label_type[index]
                        batch.append(batch_element)
                        example_idx += 1
                        if len(batch) == batch_size:
                            yield collate_fn(batch)
                            if n_examples is not None and example_idx >= n_examples:
                                if not silent:
                                    print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                                done = True
                            batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True