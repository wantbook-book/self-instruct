from pathlib import Path
import json
def parse_to_seed_format(data_path: Path, output_path: Path):
    with open(data_path, "r") as f:
        data = f.readlines()
    instructions = []
    for line in data:
        item = json.loads(line)
        instructions.append({
            "id": item["idx"],
            "instruction": item["problem"],
            "name": "",
            "instances": [],
            "is_classification": False
        })
    with open(output_path, "w") as f:
        for instruction in instructions:
            f.write(json.dumps(instruction) + "\n")
        
    return instructions

def parse_to_train_format(data_path, output_path):
    with open(data_path, "r") as f:
        data = f.readlines()
    instructions = []
    for i, line in enumerate(data):
        item = json.loads(line)
        instructions.append({
            "idx": i,
            "problem": item["instruction"],
            "answer": "",
            "solution": "",
        })
    with open(output_path, "w") as f:
        for instruction in instructions:
            f.write(json.dumps(instruction) + "\n")
        
    return instructions

if __name__ == '__main__':
    src_path = Path(__file__).parent.parent.parent
    # data_path = '/pubshare/fwk/code/MCGEP/dataset/math/0_2_0_8_train_with_idx_sample_500.jsonl'
    # output_path = src_path / 'data/math/0_2_0_8_train_with_idx_sample_500_seed.jsonl'

    # parse_to_seed_format(data_path, output_path)

    data_path = src_path / 'data/llama3_2_3b_generations/0208_500seed_gen7500/2025-04-22_15-19-31/machine_generated_instructions.jsonl'
    output_path = src_path / 'data/llama3_2_3b_generations/0208_500seed_gen7500/2025-04-22_15-19-31/0208_500seed_gen7500.jsonl'
    parse_to_train_format(data_path, output_path)
