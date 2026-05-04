#!/bin/bash

# Checkpoint paths:
# ./trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin
# ./trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin
# ./trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin
# ./trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin

# --output_dir ./eval_results/qwen3-4b-instruct_d2l \
# --output_dir ./eval_results/gemma-2-2b-it_d2l \

for K in 64; do
clear
echo "Generating for length ${K}k"

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
conda run --live-stream -n D2L python run_eval.py \
    --checkpoint_path ./trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
    --datasets babilong_qa_1 --split test_${K}k \
    --max_ctx_chunk_len 512 \
    --eval_batch_size_gen 1 \
    --output_dir ./eval_results/gemma-2-2b-it_d2l \
    --run_name babilong_qa_1_len_${K}k
done
