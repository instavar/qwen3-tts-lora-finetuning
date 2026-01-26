# Qwen3‑TTS LoRA Fine‑Tuning (Companion Repo)

**Qwen3‑TTS LoRA fine‑tuning** tools for custom voice adaptation, designed as a companion repo to the official Qwen3‑TTS project.
This repo provides **LoRA fine‑tuning scripts and patches** without forking upstream.

- Upstream repo: https://github.com/QwenLM/Qwen3-TTS
- Tested upstream commit: `0c6a7cbb6c8421a46332f8c2434c7825c4c855ef`

## Why a companion repo

- No fork drift
- Small, auditable changes
- Easy to rebase when upstream updates

## Quick start

```bash
# 1) Clone upstream (choose any working directory)
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
git checkout 0c6a7cbb6c8421a46332f8c2434c7825c4c855ef
cd ..

# 2) Clone this repo next to it (or anywhere you like)
git clone https://github.com/cheeweijie/qwen3-tts-lora-finetuning.git

# 3) Apply patches to upstream (set QWEN_DIR to your upstream clone)
QWEN_DIR=/path/to/Qwen3-TTS \
  bash /path/to/qwen3-tts-lora-finetuning/scripts/apply_patches.sh
```

## Environment setup (example)

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts peft
pip install -U flash-attn --no-build-isolation
```

## Scripts

All scripts expect `QWEN_DIR` pointing to the upstream clone. If you keep both repos side‑by‑side, you can do:

```bash
QWEN_DIR=../Qwen3-TTS bash scripts/run_lora_train.sh
```

### Train (LoRA)

```bash
QWEN_DIR=/path/to/Qwen3-TTS \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
VAL_JSONL=/path/to/val_with_codes.jsonl \
OUTPUT_DIR=/path/to/output_dir \
BATCH_SIZE=4 \
EPOCHS=3 \
bash scripts/run_lora_train.sh
```

### Eval loss

```bash
QWEN_DIR=/path/to/Qwen3-TTS \
CHECKPOINT_DIR=/path/to/checkpoint-epoch-10 \
TEST_JSONL=/path/to/test_with_codes.jsonl \
bash scripts/run_eval_loss.sh
```

### Inference

```bash
QWEN_DIR=/path/to/Qwen3-TTS \
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
