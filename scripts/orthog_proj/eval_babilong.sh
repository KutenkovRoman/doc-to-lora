#!/bin/bash

# "babilong_72_epochs/checkpoint-8000"

for CHECKPOINT in "babilong_72_epochs/checkpoint-8352" "babilong_32_epochs/checkpoint-3712"; do
    for K in 0 1 2 4 8; do
    clear
    echo "Generating for ${CHECKPOINT} length ${K}k"

    export CUDA_VISIBLE_DEVICES=2
    export WANDB_MODE=disabled
    conda run --live-stream -n D2L python run_eval.py \
        --checkpoint_path "./train_outputs/runs/${CHECKPOINT}/pytorch_model.bin" \
        --datasets babilong_qa_1 --split test_${K}k \
        --max_ctx_chunk_len 512 \
        --eval_batch_size_gen 1 \
        --output_dir "./eval_results/babilong_gemma/${CHECKPOINT}" \
        --run_name len_${K}k
    done
done
