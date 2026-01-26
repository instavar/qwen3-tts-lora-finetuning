#!/usr/bin/env bash
set -euo pipefail
QWEN_DIR="${QWEN_DIR:?set QWEN_DIR to your Qwen3-TTS clone}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${QWEN_DIR}"

git apply "${SCRIPT_DIR}/../patches/0001-qwen3-tts-lora.patch"

echo "Applied patches to ${QWEN_DIR}"
