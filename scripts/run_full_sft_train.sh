#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"
PYTHON="${PYTHON:-python3}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:?set INIT_MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
TRAIN_JSONL="${TRAIN_JSONL:?set TRAIN_JSONL}"
VAL_JSONL="${VAL_JSONL:-}"

args=(
  --qwen-dir "${QWEN_DIR}"
  --init-model-path "${INIT_MODEL_PATH}"
  --output-model-path "${OUTPUT_DIR}"
  --train-jsonl "${TRAIN_JSONL}"
  --batch-size "${BATCH_SIZE:-2}"
  --learning-rate "${LR:-2e-6}"
  --num-epochs "${EPOCHS:-3}"
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-4}"
  --mixed-precision "${MIXED_PRECISION:-bf16}"
  --attention "${ATTN_IMPL:-flash_attention_2}"
  --speaker-name "${SPEAKER_NAME:-speaker}"
  --speaker-id "${SPEAKER_ID:-3000}"
  --speaker-reference-index "${SPEAKER_REFERENCE_INDEX:-0}"
  --seed "${TRAIN_SEED:-42}"
  --save-every "${SAVE_EVERY:-1}"
  --eval-every "${EVAL_EVERY:-1}"
)
if [[ -n "${VAL_JSONL}" ]]; then
  args+=(--val-jsonl "${VAL_JSONL}")
fi
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  args+=(--eval-batch-size "${EVAL_BATCH_SIZE}")
fi

"${PYTHON}" "${SCRIPT_DIR}/train_full_sft.py" "${args[@]}"
