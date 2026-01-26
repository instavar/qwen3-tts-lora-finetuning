#!/usr/bin/env bash
set -euo pipefail

QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
ADAPTER_DIR="${ADAPTER_DIR:?set ADAPTER_DIR}"
OUT_WAV="${OUT_WAV:-output.wav}"
TEXT="${TEXT:-On a quiet morning, the streets were nearly empty.}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

python "${QWEN_DIR}/finetuning/infer_lora_custom_voice.py" \
  --base_model_path "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_DIR}" \
  --speaker_name "${SPEAKER_NAME}" \
  --text "${TEXT}" \
  --language auto \
  --attn_implementation "${ATTN_IMPL}" \
  --output_wav "${OUT_WAV}"
