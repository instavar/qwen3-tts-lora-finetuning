# Qwen3‑TTS LoRA Fine‑Tuning (Companion Repo)

This repo provides **LoRA fine‑tuning scripts and patches** for Qwen3‑TTS without forking the upstream project.
It is designed to be used alongside the official Qwen3‑TTS repo.

- Upstream repo: https://github.com/QwenLM/Qwen3-TTS
- Tested upstream commit: `0c6a7cbb6c8421a46332f8c2434c7825c4c855ef`

## Why a companion repo

- No fork drift
- Small, auditable changes
- Easy to rebase when upstream updates

## Quick start

```bash
mkdir -p ~/voice-models
cd ~/voice-models

# 1) Clone upstream

git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
git checkout 0c6a7cbb6c8421a46332f8c2434c7825c4c855ef
cd ..

# 2) Clone this repo

git clone https://github.com/cheeweijie/qwen3-tts-lora-finetuning.git

# 3) Apply patches to upstream

QWEN_DIR=~/voice-models/Qwen3-TTS \
  bash qwen3-tts-lora-finetuning/scripts/apply_patches.sh
```

## Environment setup (example)

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts peft
pip install -U flash-attn --no-build-isolation
```

## Scripts

All scripts expect `QWEN_DIR` pointing to the upstream clone.

### Train (LoRA)

```bash
QWEN_DIR=~/voice-models/Qwen3-TTS \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
VAL_JSONL=/path/to/val_with_codes.jsonl \
OUTPUT_DIR=/path/to/output_dir \
BATCH_SIZE=4 \
EPOCHS=3 \
bash scripts/run_lora_train.sh
```

### Eval loss

```bash
QWEN_DIR=~/voice-models/Qwen3-TTS \
CHECKPOINT_DIR=/path/to/checkpoint-epoch-10 \
TEST_JSONL=/path/to/test_with_codes.jsonl \
bash scripts/run_eval_loss.sh
```

### Inference

```bash
QWEN_DIR=~/voice-models/Qwen3-TTS \
BASE_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-Base \
ADAPTER_DIR=/path/to/checkpoint-epoch-10 \
OUT_WAV=./sample.wav \
TEXT="On a quiet morning, the streets were nearly empty." \
bash scripts/run_lora_infer.sh
```

## What the patch adds

- LoRA training script: `finetuning/sft_12hz_lora.py`
- LoRA inference helper: `finetuning/infer_lora_custom_voice.py`
- Eval‑only loss: `finetuning/eval_sft_12hz.py`
- Simple step benchmark: `finetuning/bench_lora_step.py`
- Validation support added to `finetuning/sft_12hz.py`

## License

Apache‑2.0
