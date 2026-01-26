#!/usr/bin/env bash
set -euo pipefail

QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"

INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
TRAIN_JSONL="${TRAIN_JSONL:?set TRAIN_JSONL}"
VAL_JSONL="${VAL_JSONL:-}"

BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_BIAS="${LORA_BIAS:-none}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

VAL_ARGS=""
if [ -n "${VAL_JSONL}" ]; then
  VAL_ARGS="--val_jsonl ${VAL_JSONL}"
fi

python "${QWEN_DIR}/finetuning/sft_12hz_lora.py" \
  --init_model_path "${INIT_MODEL_PATH}" \
  --output_model_path "${OUTPUT_DIR}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_epochs "${EPOCHS}" \
  --speaker_name "${SPEAKER_NAME}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --attn_implementation "${ATTN_IMPL}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_bias "${LORA_BIAS}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  ${VAL_ARGS}
