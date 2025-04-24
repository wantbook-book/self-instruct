batch_dir=data/llama3_2_3b_generations/
TOKENIZERS_PARALLELISM=false python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 7500 \
    --seed_tasks_path data/math/0_2_0_8_train_with_idx_sample_500_seed.jsonl \
    --engine "/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct" \
    --request_batch_size 5 \
    --num_prompt_instructions 8 \
    --prompt /pubshare/fwk/code/self-instruct/self_instruct/prompts/question_generation.json \
    --tag "0208_500seed_gen7500" \
    --max_new_instructions_per_request 4

