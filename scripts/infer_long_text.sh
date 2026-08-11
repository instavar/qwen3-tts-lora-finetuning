#!/usr/bin/env bash
# Generate audio for long text by chunking into sentences and concatenating.
# Fixes random seed per chunk to prevent timbre shift across boundaries.
#
# Usage:
#   QWEN_DIR=./Qwen3-TTS \
#   BASE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
#   ADAPTER_DIR=./output/checkpoint-epoch-10 \
#   TEXT_FILE=./long_script.txt \
#   bash scripts/infer_long_text.sh
set -euo pipefail

QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"
PYTHON="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
ADAPTER_DIR="${ADAPTER_DIR:?set ADAPTER_DIR}"
TEXT_FILE="${TEXT_FILE:?set TEXT_FILE (path to text file, one sentence per line)}"
OUT_WAV="${OUT_WAV:-./output_long.wav}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker}"
LORA_SCALE="${LORA_SCALE:-0.3}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
BASE_SEED="${BASE_SEED:-42}"

TMPDIR="$(mktemp -d)"
trap "rm -rf ${TMPDIR}" EXIT

echo "Generating long-text audio from ${TEXT_FILE}..."
echo "Using fixed seed base: ${BASE_SEED} (incremented per chunk)"

chunk_idx=0
file_list=""

while IFS= read -r line || [ -n "$line" ]; do
  # Skip empty lines
  [ -z "$(echo "$line" | tr -d '[:space:]')" ] && continue

  chunk_wav="${TMPDIR}/chunk_$(printf '%04d' ${chunk_idx}).wav"
  seed=$((BASE_SEED + chunk_idx))

  "${PYTHON}" "${QWEN_DIR}/finetuning/infer_lora_custom_voice.py" \
    --base_model_path "${BASE_MODEL}" \
    --adapter_path "${ADAPTER_DIR}" \
    --speaker_name "${SPEAKER_NAME}" \
    --text "${line}" \
    --language auto \
    --lora_scale "${LORA_SCALE}" \
    --seed "${seed}" \
    --attn_implementation "${ATTN_IMPL}" \
    --output_wav "${chunk_wav}" \
    2>/dev/null

  file_list="${file_list}file '${chunk_wav}'\n"
  chunk_idx=$((chunk_idx + 1))
done < "${TEXT_FILE}"

# Concatenate all chunks
echo -e "${file_list}" > "${TMPDIR}/concat_list.txt"
ffmpeg -y -f concat -safe 0 -i "${TMPDIR}/concat_list.txt" -c copy "${OUT_WAV}" -loglevel error

echo "Generated ${chunk_idx} chunks -> ${OUT_WAV}"
echo "Note: each chunk uses seed ${BASE_SEED}+N to maintain timbre consistency."
