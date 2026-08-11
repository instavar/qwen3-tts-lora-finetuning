#!/usr/bin/env bash
# Generate one audio sample per checkpoint for side-by-side listening comparison.
# Usage:
#   QWEN_DIR=./Qwen3-TTS \
#   BASE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
#   ADAPTER_ROOT=./output \
#   bash scripts/run_infer_epochs.sh
set -euo pipefail

QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"
PYTHON="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
ADAPTER_ROOT="${ADAPTER_ROOT:?set ADAPTER_ROOT}"
OUT_DIR="${OUT_DIR:-./epoch_samples_$(date +%Y%m%d_%H%M%S)}"
TEXT="${TEXT:-On a quiet morning, the streets were nearly empty. A delivery truck rolled past the corner shop and then turned toward the river.}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker}"
LORA_SCALE="${LORA_SCALE:-0.3}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

mkdir -p "${OUT_DIR}"

echo "Generating samples from checkpoints in ${ADAPTER_ROOT}..."
echo "Output dir: ${OUT_DIR}"
echo "LoRA scale: ${LORA_SCALE}"
echo ""

for adapter in "${ADAPTER_ROOT}"/checkpoint-epoch-*; do
  [ -d "${adapter}" ] || continue
  epoch="$(basename "${adapter}" | sed 's/checkpoint-epoch-//')"
  out_wav="${OUT_DIR}/epoch_${epoch}.wav"

  echo "Epoch ${epoch}..."
  "${PYTHON}" "${QWEN_DIR}/finetuning/infer_lora_custom_voice.py" \
    --base_model_path "${BASE_MODEL}" \
    --adapter_path "${adapter}" \
    --speaker_name "${SPEAKER_NAME}" \
    --text "${TEXT}" \
    --language auto \
    --lora_scale "${LORA_SCALE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --output_wav "${out_wav}" \
    2>&1 | tail -1

  echo "  -> ${out_wav}"
done

echo ""
echo "Done. Listen to samples in: ${OUT_DIR}"
echo "Tip: best checkpoint is rarely the last one. Compare epoch 8-12 range first."
