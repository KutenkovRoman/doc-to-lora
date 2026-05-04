#!/bin/bash
set -euo pipefail

#--num_train_epochs= \
#--use_flash_attn= \

export CUDA_VISIBLE_DEVICES=0
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
export WANDB_API_KEY=${WANDB_API_KEY:-local-c69001806d746ec255ca0d6a45f3aed923afb9ba}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-doc-to-lora}

conda run --live-stream -n D2L python train.py \
    configs/babilong_exp/babilong_qa_1.yaml \
    --from_pretrained_checkpoint=trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
    --model_name_or_path=google/gemma-2-2b-it \
    --use_flash_attn=False \
    --chunk_lora_merge_mode=attnpool \
    --ctx_encoder_type=per_layer_activations \
    --n_latent_queries=8 \
    --num_blocks=9 \
    --num_self_attn_per_block=0 \
    --eval_strategy=no \
    --max_qas_len=512 \
    --gen_lora_l1_reg_coef=0.1 \
    --max_ctx_chunk_len=512 \
    --max_qas_per_sample=1 \
    --per_rank_gen=True \
    --per_layer_processing=True \
    --max_steps=8200 \
    --gradient_accumulation_steps=16 \
    --learning_rate=2e-5 \
    --warmup_ratio=0.1 \
    --max_packed_inp_len=1024 \
    --max_packed_ctx_len=2048 \
    --use_per_ctx_average_loss=True \
    --use_kl_loss=True \
    --quantize_ctx_encoder=True \
    --max_ctx_chunk_len=512 \
    --min_ctx_chunk_len=32 \
    --num_chunk_probs='{"1":"0.5", "2":"0.125", "3":"0.0625", "4":"0.0625", "5":"0.0625", "6":"0.0625", "7":"0.0625", "8":"0.0625"}' \
    --logging_steps=10 \
    "$@"
