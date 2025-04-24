batch_dir=data/llama3_2_3b_generations/
TOKENIZERS_PARALLELISM=false python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 7500 \
    --seed_tasks_path data/math/train_filtered_correct_0_2_0_8_seed.jsonl \
    --engine "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-3b-filtered_correct_0_2_0_8_gen_math_7_5k_random_bon_maj_bs16/global_step100_hf" \
    --request_batch_size 5 \
    --num_prompt_instructions 8 \
    --prompt /pubshare/fwk/code/self-instruct/self_instruct/prompts/question_generation.json \
    --tag "train_filtered_correct_0_2_0_8_seed_gen_7500_step100" \
    --load_machine_generated_instructions_dir data/llama3_2_3b_generations/train_filtered_correct_0_2_0_8_seed_gen_7500_step100/2025-03-17_14-16-48/ \
    --max_new_instructions_per_request 4
