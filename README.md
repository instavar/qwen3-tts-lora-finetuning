# Qwen3-TTS LoRA Fine-Tuning (Companion Repo)

**LoRA fine-tuning tools for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — custom voice adaptation without forking upstream.

We used this repo to fine-tune Qwen3-TTS 1.7B on IMDA NSC FEMALE\_01 (Singaporean English) for production voice cloning. The pitfalls, fixes, and recommendations below come from that experience.

- Upstream repo: https://github.com/QwenLM/Qwen3-TTS
- Tested upstream commit: `0c6a7cbb6c8421a46332f8c2434c7825c4c855ef`
- Blog deep-dive: [LoRA Finetuning Qwen3-TTS for Custom Voices](https://instavar.com/blog/ai-production-stack/LoRA_Finetuning_Qwen3_TTS_Custom_Voices)
- Decision tree (9 models): [Which TTS Model Should You Use?](https://instavar.com/blog/ai-production-stack/TTS_Model_Decision_Tree_2026)

## Why a companion repo

- No fork drift — patches apply on top of upstream
- Small, auditable changes
- Easy to rebase when upstream updates

## Known pitfalls

These are model-inherent bugs and edge cases from our fine-tuning runs. Environment setup issues are excluded.

| # | Pitfall | Symptom | Fix | Upstream |
|---|---------|---------|-----|----------|
| 1 | **Double label-shift bug** in `sft_12hz.py` | Speech progressively accelerates each epoch | Replace with `F.cross_entropy()` | PR #178 |
| 2 | **Missing `text_projection` call** (line 93) | Hard crash on 0.6B; silent wrong embeddings on 1.7B | Add `model.talker.text_projection()` | PR #188 |
| 3 | **Default LR too high** (2e-5) | Pure noise, infinite generation (no EOS) | Use **2e-6** | Issue #39 |
| 4 | **Audio not at 24 kHz** | Crash deep in training, no early warning | `bash scripts/resample_to_24k.sh <dir>` | PR #233 |
| 5 | **LoRA scale 1.0 at inference** | Over-steered, forced output | Use **0.3-0.35** | — |
| 6 | **EOS token failures** (~0.5%) | Infinite token generation, hangs | Explicit `eos_token_id` list + `max_new_tokens` | — |
| 7 | **Cold-start decoder distortion** | First inference produces corrupted frames | Prepend silence tokens as warm-up, trim | #219 |
| 8 | **Timbre shift across chunks** | Voice changes in long-text generation | Fix seed per chunk; reuse speaker embedding | — |
| 9 | **Val crash on small val sets** | `RuntimeError: zero-dimensional tensor` | Guard for empty loss tensor in eval | — |
| 10 | **Inference segfaults** | Crashes mid-epoch sweep | Batch inference defensively | — |
| 11 | **Overfitting after epoch 10** | Val loss plateaus, train loss drops | Stop at epoch 10 for single-speaker | — |

The **double label-shift bug (#1)** is the most impactful — it affects every run on the official script.

## Quick start

```bash
# 1) Clone upstream
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
git checkout 0c6a7cbb6c8421a46332f8c2434c7825c4c855ef
cd ..

# 2) Clone this repo
git clone https://github.com/cheeweijie/qwen3-tts-lora-finetuning.git

# 3) Apply patches
QWEN_DIR=./Qwen3-TTS bash qwen3-tts-lora-finetuning/scripts/apply_patches.sh
```

## Environment setup

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts peft
pip install -U flash-attn --no-build-isolation
```

## Data preparation

**Critical: resample to 24 kHz first.** The codec pipeline asserts 24 kHz — other sample rates crash deep in training.

```bash
# Resample all WAVs in your dataset to 24kHz mono
bash scripts/resample_to_24k.sh /path/to/audio_dir

# Then generate audio codes (upstream script)
python ${QWEN_DIR}/finetuning/prepare_data.py \
  --input_dir /path/to/audio_dir \
  --output_jsonl /path/to/train_with_codes.jsonl
```

## Scripts

All scripts expect `QWEN_DIR` pointing to the upstream clone.

### Train (LoRA)

```bash
QWEN_DIR=./Qwen3-TTS \
TRAIN_JSONL=./train_with_codes.jsonl \
VAL_JSONL=./val_with_codes.jsonl \
OUTPUT_DIR=./output \
LR=2e-6 \
EPOCHS=10 \
bash scripts/run_lora_train.sh
```

> **Note:** Default LR is `2e-6` (not the upstream `2e-5`). The higher rate causes noise and EOS failures.

### Eval loss

```bash
QWEN_DIR=./Qwen3-TTS \
CHECKPOINT_DIR=./output/checkpoint-epoch-10 \
TEST_JSONL=./test_with_codes.jsonl \
bash scripts/run_eval_loss.sh
```

### Inference

```bash
QWEN_DIR=./Qwen3-TTS \
BASE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
ADAPTER_DIR=./output/checkpoint-epoch-10 \
LORA_SCALE=0.3 \
TEXT="On a quiet morning, the streets were nearly empty." \
OUT_WAV=./sample.wav \
bash scripts/run_lora_infer.sh
```

> **Tip:** Always sweep LoRA scale. Test at 0.2, 0.3, 0.35, 0.5 before committing. Scale 1.0 is almost always wrong.

### Benchmark step timing

```bash
QWEN_DIR=./Qwen3-TTS \
BASE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
ADAPTER_DIR=./output/checkpoint-epoch-10 \
bash scripts/run_bench.sh
```

## What the patch adds

- `finetuning/sft_12hz_lora.py` — LoRA training script
- `finetuning/infer_lora_custom_voice.py` — LoRA inference helper
- `finetuning/eval_sft_12hz.py` — Eval-only loss computation
- `finetuning/bench_lora_step.py` — Step timing benchmark
- Validation support added to `finetuning/sft_12hz.py`

## Upstream PR tracker

| PR | What it fixes | Status |
|----|--------------|--------|
| #178 | Double label-shift in `sft_12hz.py` | Open |
| #188 | Missing `text_projection` call | Merged (`680d4e9`) |
| #233 | Auto-resample to 24 kHz | Open |
| #259 | `chunked_decode` truncation at ~24s | Open |

When these PRs merge upstream, the corresponding patch hunks become unnecessary.

## Recommended configuration

Based on our IMDA NSC FEMALE\_01 runs:

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 2e-6 | 2e-5 causes noise (validated across multiple runs) |
| Epochs | 10 | Val loss plateaus after; further training overfits |
| LoRA rank | 16 | Sufficient for single-speaker |
| LoRA alpha | 32 | 2x rank (default) |
| Batch size | 4 | Fits 24 GB GPU with gradient accumulation |
| LoRA scale (inference) | 0.3-0.35 | Scale 1.0 over-steers; sweep first |
| Sample rate | 24 kHz | Non-negotiable — codec enforces this |

## Alternatives

This repo provides the **LoRA fine-tuning path** with production-validated pitfalls. Depending on your needs, other options may be a better fit:

| Approach | Repo | Best for |
|----------|------|----------|
| **LoRA fine-tuning** (this repo) | [instavar/qwen3-tts-lora-finetuning](https://github.com/instavar/qwen3-tts-lora-finetuning) | Fast iteration, adapter-based voice adaptation, production deployment with scale control |
| **Full SFT** (official) | [QwenLM/Qwen3-TTS/finetuning](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) | Maximum quality when you can afford full-weight updates. Note: upstream `sft_12hz.py` has known bugs (see pitfalls #1-#2 above) |
| **Full SFT + WebUI** | [mozi1924/Qwen3-TTS-EasyFinetuning](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning) | Automated preprocessing + Gradio interface. Good for users who want a GUI workflow. Does not include LoRA support or the upstream bug fixes |
| **ComfyUI integration** | [DarioFT/ComfyUI-Qwen3-TTS](https://github.com/DarioFT/ComfyUI-Qwen3-TTS) | Fine-tuning and inference within ComfyUI node workflows |
| **Audiobook pipeline + LoRA** | [Finrandojin/alexandria-audiobook](https://github.com/Finrandojin/alexandria-audiobook) | LoRA training embedded in a Gradio audiobook workflow with per-line style control |

If you need full SFT with a friendlier interface and don't need LoRA, `mozi1924/Qwen3-TTS-EasyFinetuning` is worth evaluating. If you need LoRA with documented pitfalls and inference-time scale control, that's what this repo provides.


Apache-2.0


## License
