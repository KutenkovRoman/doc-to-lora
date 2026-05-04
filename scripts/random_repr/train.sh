#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
export WANDB_API_KEY=${WANDB_API_KEY:-local-c69001806d746ec255ca0d6a45f3aed923afb9ba}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-doc-to-lora}

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

# changed --per_rank_gen=True
# changed --ctx_encoder_type=early_exit -> per_layer_activations
# changed --max_packed_ctx_len=4096
# changed --eval_on_start=True

conda run --live-stream -n D2L python train.py \
    configs/niah_exp/ctx_magic_number_32_256.yaml \
    --model_name_or_path=google/gemma-2-2b-it \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --per_device_eval_batch_size=16 \
    --num_blocks=8 \
    --num_self_attn_per_block=0 \
    --num_pre_head_layers=1 \
    --eval_steps=100 \
    --logging_steps=10 \
    --save_steps=1000 \
    --learning_rate=4e-5 \
    --neftune_noise_alpha=0 \
    --per_rank_gen=False \
    --per_layer_processing=True \
    --gen_lora_l1_reg_coef=1.5 \
    --max_packed_inp_len=4096 \
    --max_packed_ctx_len=4096 \
    --dataloader_num_workers=0 \
    --dataloader_prefetch_factor=None \
    --ctx_encoder_type=early_exit \
    --n_latent_queries=208 \
    --use_kl_loss=False \
    --eval_on_start=True \
    --max_ctx_chunk_len=512 \
    --min_ctx_chunk_len=25 \
    --num_chunk_probs='{"1":"0.5", "2":"0.125", "3":"0.0625", "4":"0.0625", "5":"0.0625", "6":"0.0625", "7":"0.0625", "8":"0.0625"}' \
    --max_val_samples_per_ds=100 \
    --seed=42 \
    --use_per_ctx_average_loss=True \
    --torch_empty_cache_steps=10 \
    "$@"
