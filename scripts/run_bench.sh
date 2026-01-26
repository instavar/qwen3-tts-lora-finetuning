#!/usr/bin/env bash
set -euo pipefail

QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
ADAPTER_DIR="${ADAPTER_DIR:?set ADAPTER_DIR}"

python "${QWEN_DIR}/finetuning/bench_lora_step.py" \
  --base_model_path "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_DIR}"
