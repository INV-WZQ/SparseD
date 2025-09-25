import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "../Checkpoint/"
os.environ["HF_HOME"] = "../Checkpoint/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "../Checkpoint/"
os.environ["TRANSFORMERS_CACHE"] = "../Checkpoint/"

import torch
from transformers import AutoModel, AutoTokenizer

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--sampling-alg", type=str, default='low_confidence')

    parser.add_argument("--origin", action="store_true")

    parser.add_argument("--skip", type=float, default=0.2)
    parser.add_argument("--select", type=float, default=0.3)
    parser.add_argument("--block_size", type=int, default=128)
    
    parser.add_argument("--prompt", type=str, default="short_context")
    args = parser.parse_args()

    model_path = args.model_path

    from models import LLaDAModelLM, generate
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()

    with open('prompts.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    question=data["questions"][args.prompt]
    messages = [{"role": "user", "content": question}]

    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = prompt_ids.input_ids.to(device="cuda")
    
    if args.origin:
        print("Use Original Model!")
        SparseD_param = None
    else:
        print("Use SparseD version!")
        SparseD_param = {
            'skip': args.skip, 
            'select': args.select, 
            'block_size': args.block_size,
            'new_generation': args.seq_len,
            'whole_steps': args.steps
        }

    import time
    start_time = time.time()
    output = generate(
        model, input_ids, 
        steps=args.steps, 
        gen_length=args.seq_len, 
        block_length=args.block_length, 
        temperature=0, 
        remasking=args.sampling_alg, 
        SparseD_param=SparseD_param
    )
    end_time = time.time()

    answer = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"----Question of length {args.prompt}: {messages[0]['content']}")
    print(answer)
    print(f"Running Time: {end_time - start_time:.4f}")