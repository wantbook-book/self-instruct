batch_dir=data/llama3_2_1b_generations/

TOKENIZERS_PARALLELISM=false python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 5000 \
    --seed_tasks_path data/math/train_with_idx_2_3_seed.jsonl \
    --engine "/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct" \
    --request_batch_size 5 \
    --num_prompt_instructions 8 \
    --prompt /pubshare/fwk/code/self-instruct/self_instruct/prompts/question_generation.json \
    --tag "train_2_3_seed_gen_5000"