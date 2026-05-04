#!/bin/bash

# Checkpoint paths:
# ./train_outputs/runs/...

for K in 0 1 2 4; do  # 0 1 2 4 8 16 32 64 128
    clear
    echo "Generating for length ${K}k"

    export CUDA_VISIBLE_DEVICES=1
    export WANDB_MODE=disabled
    conda run --live-stream -n D2L python run_eval.py \
        --checkpoint_path ./train_outputs/runs/babilong_qa_1_nochunk_epochs_66/checkpoint-15000/pytorch_model.bin \
        --datasets babilong_qa_1 --split test_${K}k \
        --max_ctx_chunk_len 4096 \
        --eval_batch_size_gen 1 \
        --output_dir ./eval_results/gemma-2-2b-it_sft \
        --run_name babilong_qa_1_len_${K}k_nochunk_epochs_66
done
