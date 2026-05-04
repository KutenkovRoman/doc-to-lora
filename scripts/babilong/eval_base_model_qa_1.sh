#!/bin/bash

# Models:
# google/gemma-2-2b-it
# Qwen/Qwen3-4B-Instruct-2507
# mistralai/Mistral-7B-Instruct-v0.2

for K in 0; do  #1 2 4 8 16 32 64 128
    clear
    echo "Generating for length ${K}k"

    export CUDA_VISIBLE_DEVICES=0
    export WANDB_MODE=disabled
    conda run --live-stream -n D2L python run_eval.py \
        --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
        --datasets babilong_qa_1 --split train_${K}k \
        --eval_batch_size_gen 1 \
        --output_dir ./eval_results/qwen3-4b-instruct_base \
        --run_name babilong_qa_1_len_${K}k
done
