import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "../Checkpoint/"
os.environ["HF_HOME"] = "../Checkpoint/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "../Checkpoint/"
os.environ["TRANSFORMERS_CACHE"] = "../Checkpoint/"

import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

import json 
import argparse
import time

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--sampling-alg", type=str, default='maskgit_plus')
    parser.add_argument("--origin", action="store_true")
    parser.add_argument("--skip", type=float, default=0.2)
    parser.add_argument("--select", type=float, default=0.3)
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()

    model_path = "Dream-org/Dream-v0-Instruct-7B"

    from models import DreamModel
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()

    with open('prompts.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    messages = [[{"role": "user", "content": question} for question in data["questions"]]]

    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = prompt_ids.input_ids.to(device="cuda")
    attention_mask = prompt_ids.attention_mask.to(device="cuda")
    
    if args.origin:
        print("Use Original Model!")
        SparseD_param = None
    else:
        print("Use SparseD version!")
        SparseD_param = {'skip': args.skip, 'select': args.select, 'block_size': args.block_size}

    import time
    start_time = time.time()
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.seq_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=0.05,
        top_p=0.95,
        alg=args.sampling_alg,
        alg_temp=0.,
        SparseD_param=SparseD_param
    )
    end_time = time.time()
    for b in range(len(messages)):
        print()
        print(f"----Question {b+1}: {messages[b][0]['content']}")
        sequence = output.sequences[b]
        print(tokenizer.decode(sequence[len(input_ids[0]):]).split('<|endoftext|>')[0])
    
    print(f"Running Time: {end_time - start_time:.4f}")