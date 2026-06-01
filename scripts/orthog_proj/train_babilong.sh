#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
export WANDB_API_KEY=${WANDB_API_KEY:-local-c69001806d746ec255ca0d6a45f3aed923afb9ba}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-doc-to-lora}

# set --use_bias=False

conda run --live-stream -n D2L python train.py \
    configs/babilong_exp/babilong_qa_1.yaml \
    --model_name_or_path=google/gemma-2-2b-it \
    --num_train_epochs=72 \
    --gradient_accumulation_steps=16 \
    --num_blocks=8 \
    --num_self_attn_per_block=0 \
    --eval_strategy=no \
    --num_pre_head_layers=1 \
    --logging_steps=10 \
    --save_steps=1000 \
    --learning_rate=4e-5 \
    --neftune_noise_alpha=0 \
    --per_rank_gen=True \
    --use_bias=False \
    --per_layer_processing=True \
    --gen_lora_l1_reg_coef=1.5 \
    --max_packed_inp_len=1024 \
    --max_packed_ctx_len=4096 \
    --dataloader_num_workers=0 \
    --dataloader_prefetch_factor=None \
    --ctx_encoder_type=early_exit \
    --n_latent_queries=208 \
    --use_kl_loss=False \
    --max_ctx_chunk_len=512 \
    --min_ctx_chunk_len=32 \
    --num_chunk_probs='{"1":"0.5", "2":"0.125", "3":"0.0625", "4":"0.0625", "5":"0.0625", "6":"0.0625", "7":"0.0625", "8":"0.0625"}' \
    --use_per_ctx_average_loss=True \
    --torch_empty_cache_steps=10 \
    "$@"
