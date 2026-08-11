#!/usr/bin/env bash
# Generate the same text from multiple checkpoints for A/B listening comparison.
# Outputs numbered WAV files side-by-side for quick evaluation.
#
# Usage:
#   QWEN_DIR=./Qwen3-TTS \
#   BASE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
#   CHECKPOINTS="output/checkpoint-epoch-8 output/checkpoint-epoch-9 output/checkpoint-epoch-10" \
#   bash scripts/compare_checkpoints.sh
set -euo pipefail

QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"
PYTHON="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
CHECKPOINTS="${CHECKPOINTS:?set CHECKPOINTS (space-separated adapter dirs)}"
OUT_DIR="${OUT_DIR:-./checkpoint_comparison_$(date +%Y%m%d_%H%M%S)}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
SEED="${SEED:-42}"

# Multiple test sentences for thorough comparison
TEXTS=(
  "On a quiet morning, the streets were nearly empty."
  "The experiment produced unexpected results that challenged our initial hypothesis."
  "She paused for a moment, then continued with renewed confidence."
)

SCALES="${SCALES:-0.2 0.3 0.35 0.5}"

mkdir -p "${OUT_DIR}"

echo "Comparing checkpoints with fixed seed ${SEED}"
echo "Output dir: ${OUT_DIR}"
echo ""

for ckpt in ${CHECKPOINTS}; do
  ckpt_name="$(basename "${ckpt}")"
  for scale in ${SCALES}; do
    for i in "${!TEXTS[@]}"; do
      text="${TEXTS[$i]}"
      out_wav="${OUT_DIR}/${ckpt_name}_scale${scale}_text${i}.wav"

      "${PYTHON}" "${QWEN_DIR}/finetuning/infer_lora_custom_voice.py" \
        --base_model_path "${BASE_MODEL}" \
        --adapter_path "${ckpt}" \
        --speaker_name "${SPEAKER_NAME}" \
        --text "${text}" \
        --language auto \
        --lora_scale "${scale}" \
        --seed "${SEED}" \
        --attn_implementation "${ATTN_IMPL}" \
        --output_wav "${out_wav}" \
        2>/dev/null

      echo "  ${ckpt_name} | scale=${scale} | text${i} -> $(basename "${out_wav}")"
    done
  done
done

echo ""
echo "Generated $(find "${OUT_DIR}" -name '*.wav' | wc -l) samples in ${OUT_DIR}"
echo "Listen and compare: best checkpoint × best scale = your production config."
