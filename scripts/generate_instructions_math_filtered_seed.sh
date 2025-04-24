batch_dir=data/llama3_2_3b_generations/

TOKENIZERS_PARALLELISM=false python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 7500 \
    --seed_tasks_path data/math/train_filtered_correct_0_2_0_8_seed.jsonl \
    --engine "/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct" \
    --request_batch_size 5 \
    --num_prompt_instructions 8 \
    --prompt /pubshare/fwk/code/self-instruct/self_instruct/prompts/question_generation.json \
    --tag "train_filtered_correct_0_2_0_8_seed_gen_7500"
