import os
import json
import random
import re
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests
from vllm import LLM, SamplingParams
from utils.re_utils import extract_questions
from transformers import AutoTokenizer
from datetime import datetime
from utils.post_process_response import post_process_gpt3_response, post_process_response
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# random.seed(42)


# def encode_prompt(prompt_instructions, classification=False):
#     """Encode multiple prompt instructions into a single string."""
#     if classification:
#         prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
#     else:
#         prompt = "Come up with a series of tasks:\n"
#     for idx, instruction in enumerate(prompt_instructions):
#         instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
#         prompt += f"{idx+1}. {instruction}\n"
#     prompt += f"{len(prompt_instructions) + 1}."
#     return prompt

def encode_prompt(tokenizer, prompt_instructions: list[str], prompt_json: dict):
    """Encode multiple prompt instructions into a single string."""
    system_prompt = prompt_json["system"]
    user_prompt = prompt_json["user"]
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        user_prompt += f"{idx+1}. {instruction}\n"
    user_prompt += f"{len(prompt_instructions) + 1}."
    if tokenizer:
        item = []
        if system_prompt:
            item.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )
        if user_prompt:
            item.append(
                {
                    "role": "user",
                    "content": user_prompt
                }
            )
        prompt = tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f'{system_prompt}\n{user_prompt}'
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--max_new_instructions_per_request",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="prompt json file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="dir tag"
    )
    parser.add_argument(
        "--load_machine_generated_instructions_dir",
        type=str,
        help="The path to the machine generated instructions."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    
    args.batch_dir = os.path.join(args.batch_dir, args.tag, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if args.load_machine_generated_instructions_dir:
        args.batch_dir = args.load_machine_generated_instructions_dir
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # question generation prompt
    with open(args.prompt, "r") as fin:
        prompt_json = json.load(fin)

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    if not args.api_key:
        llm = LLM(
            model=args.engine,
            enforce_eager=True,
            tensor_parallel_size=4,
            enable_prefix_caching=True,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.engine, trust_remote_code=True)
    else:
        llm = None
        tokenizer = None
    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, 
                    similarities=None,
                    n=2)
                # sample human instructions from the pool
                prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                # prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                prompt = encode_prompt(
                    tokenizer=tokenizer,
                    prompt_instructions=prompt_instructions,
                    prompt_json=prompt_json
                )
                batch_inputs.append(prompt)
            if args.api_key:
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=batch_inputs,
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=2,
                    stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization,
                )
            else:
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=2,
                    max_tokens=1024,
                    # stop=["\n\n", "\n16", "16.", "16 ."],
                    stop=[],
                    logprobs=1,
                    n=1,
                    best_of=1,
                )
                results = llm.generate(batch_inputs, sampling_params)
                results = [{"response": r.outputs[0].text} for r in results]

            instructions = []
            all_metadata = []
            for result in results:
                # new_instructions = post_process_gpt3_response(result["response"])
                new_instructions = post_process_response(result["response"])
                new_instructions = new_instructions[:args.max_new_instructions_per_request]
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)
            for inst, metadata in zip(instructions, all_metadata):
                with Pool(4) as p:
                    rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
                
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                if max(rouge_scores) > 0.7:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                        all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                machine_instructions.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    # "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")
                progress_bar.update(1)
            request_idx += 1


def test_sample_examples():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.engine, trust_remote_code=True)
    with open(args.prompt, "r") as fin:
        prompt_json = json.load(fin)
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    machine_instructions = []
    batch_inputs = []
    for _ in range(args.request_batch_size):
        # sample machine instructions from the pool
        prompt_instructions = sample_machine_instructions(
            machine_instructions, 
            similarities=None,
            n=2)
        # sample human instructions from the pool
        prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
        random.shuffle(prompt_instructions)
        # prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
        prompt = encode_prompt(
                    tokenizer=tokenizer,
                    prompt_instructions=prompt_instructions,
                    prompt_json=prompt_json
                )
        batch_inputs.append(prompt)
if __name__ == "__main__":
    main()
    # test_sample_examples()
