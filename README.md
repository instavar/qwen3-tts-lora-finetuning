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

- `finetuning/sft_12hz_lora.py` — LoRA training with label-shift fix (PR #178), text_projection fix (PR #188), LR default 2e-6
- `finetuning/infer_lora_custom_voice.py` — Inference with scale control, EOS cap, seed fixing
- `finetuning/eval_sft_12hz.py` — Eval-only loss computation
- `finetuning/bench_lora_step.py` — Step timing benchmark
- Validation support added to `finetuning/sft_12hz.py`

## Utility scripts

### Executable Instavar Voice lifecycle

[`instavar-voice-backend.json`](instavar-voice-backend.json) binds this
repository's LoRA and PyTorch declarations to a real five-stage backend. The
wrapper audits train, validation, and test manifests, runs the existing LoRA
launcher, archives one explicitly selected adapter, reloads it in a fresh
process, executes the frozen generation plan, and copies the byte-identical
adapter archive into the package stage.

Validate the recipe with the pinned evaluator before a GPU run:

```bash
python /path/to/instavar-voice-evaluation/main.py \
  validate-backend instavar-voice-backend.json
```

The required environment names and purposes live in the backend file. Use a
new empty work directory for every attempt. `SELECTED_ADAPTER_NAME` must be one
child directory created under the training output, such as
`checkpoint-epoch-3`. Preflight also requires the experiment's upstream and
Instavar revisions to match the active checkouts. It verifies every patched
Qwen file against a temporary Git index containing pinned upstream plus
`patches/0001-qwen3-tts-lora.patch`, and rejects unrelated dirty paths. A passed
lifecycle proves that the declared commands and artifacts completed without
mutation. It does not prove perceptual improvement.

### Frozen multi-prompt evaluation

Run every Qwen row from an Instavar Voice generation plan while loading the
base model and adapter once:

```bash
python scripts/run_evaluation_suite.py \
  --qwen-dir /path/to/Qwen3-TTS \
  --base-model /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --adapter /path/to/checkpoint-epoch-10 \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id qwen3-epoch10 \
  --runtime-id pytorch \
  --output-dir evaluation/qwen3-epoch10
```

The runner records one observation for every planned attempt, including
failures, and writes audio under the plan's expected path. It does not run ASR,
speaker similarity, or human listening and therefore does not make a quality
claim.

The executable lifecycle passes `--allow-invalid-output` so invalid generations
remain evidence instead of aborting before packaging. It then uses evaluator
revision `3af85259470914e044bf95808ab76ff417107de1` to create
`generation-attempt-receipt.json` and the runtime-bound
`objective-observations.json`. Timing and memory from the raw generation file
must not be used for a version 1.1 comparison before that binding step.

For an exact cross-runtime experiment, also pass `--artifact-set-id` and
`--artifact-set-sha256` together. The runner rejects partial or malformed
bindings. Generate and live-verify the corresponding runtime artifact manifest
with evaluator revision `3af85259470914e044bf95808ab76ff417107de1` before
using `compare-runtimes`. Converted artifacts remain `derived`, not exact.

| Script | Purpose |
|--------|----------|
| `scripts/run_lora_train.sh` | Training launcher with validated config |
| `scripts/run_lora_infer.sh` | Single-sentence inference |
| `scripts/run_eval_loss.sh` | Eval loss on test set |
| `scripts/run_bench.sh` | Step timing benchmark |
| `scripts/run_infer_epochs.sh` | One sample per checkpoint for listening comparison |
| `scripts/compare_checkpoints.sh` | A/B comparison: checkpoints x scales x sentences |
| `scripts/infer_long_text.sh` | Chunked long-text with seed-fixed timbre consistency |
| `scripts/resample_to_24k.sh` | Resample audio dir to 24kHz before codec prep |
| `scripts/apply_patches.sh` | Apply patches to upstream Qwen3-TTS clone |

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


## License

Apache-2.0

## Instavar Voice conformance

[`instavar-voice-capabilities.json`](instavar-voice-capabilities.json) declares the adaptation, runtime, evaluation, and rights boundaries supported by this companion. CI validates it against the pinned public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation). See [`INSTAVAR_VOICE_CONFORMANCE.md`](INSTAVAR_VOICE_CONFORMANCE.md) for the evidence interpretation rules and local validation command.

The pinned evaluator provides schema 1.3 frozen speaker-reference assignments,
the optional schema 1.4 SpeechBrain ECAPA execution path, and the optional
schema 1.5 local faster-whisper ASR path. Version 0.20 also distinguishes
generation-plan-bound ASR reference text from observation-declared strings.
Version 0.21 adds plan-bound category strata so pronunciation, local-context,
and long-form proxy regressions remain visible instead of disappearing into one
candidate mean.
Version 0.22 carries frozen lexical anchors and accepted ASR forms into the
generation plan, reports hit, miss, coverage, and matched deltas, and rejects
candidate-specific alias drift. Phrase hits remain recognition evidence, not
pronunciation or accent judgments.
This companion bundles neither model
weights nor optional extractor dependencies and runs neither learned metric
automatically. Run them explicitly after generation with trusted, content-addressed
models, frozen decoding, and a preregistered reference plan where applicable.
Runtime-bound observations, same-recording smoke scores, or human-recording ASR
alone are not TTS-quality evidence.
