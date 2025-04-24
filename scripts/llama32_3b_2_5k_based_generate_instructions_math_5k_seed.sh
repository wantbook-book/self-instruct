batch_dir=data/llama3_2_3b-math2_5k_random_gt_bs16/

TOKENIZERS_PARALLELISM=false python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 8000 \
    --seed_tasks_path data/math/train_with_idx_2_3_seed.jsonl \
    --engine "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-3b-math2_5k_random_gt_bs16" \
    --request_batch_size 5 \
    --num_prompt_instructions 8 \
    --prompt /pubshare/fwk/code/self-instruct/self_instruct/prompts/question_generation.json \
    --tag "ft3b_5k_seed_gen8k" \
    --max_new_instructions_per_request 4

