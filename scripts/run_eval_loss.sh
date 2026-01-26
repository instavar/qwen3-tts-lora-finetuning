#!/usr/bin/env bash
set -euo pipefail

QWEN_DIR="${QWEN_DIR:?set QWEN_DIR}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
TEST_JSONL="${TEST_JSONL:?set TEST_JSONL}"

python "${QWEN_DIR}/finetuning/eval_sft_12hz.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --test_jsonl "${TEST_JSONL}"
