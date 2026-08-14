# Qwen3-TTS LoRA Fine-Tuning (Companion Repo)

**LoRA fine-tuning tools for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** with an experimental, separately declared full-SFT lifecycle for custom voice adaptation.

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

### Executable Instavar Voice lifecycles

[`instavar-voice-backend.json`](instavar-voice-backend.json) binds this
repository's LoRA and PyTorch declarations to a real five-stage backend. The
wrapper audits train, validation, and test manifests, runs the existing LoRA
launcher, archives one explicitly selected adapter, reloads it in a fresh
process, executes the frozen generation plan, and copies the byte-identical
adapter archive into the package stage. Both registered lifecycles also publish
their final research package under a mode-bound content-addressed name to the
preflighted external retention directory and write a persistence receipt.

Validate the recipe with the pinned evaluator before a GPU run:

```bash
python /path/to/instavar-voice-evaluation/main.py \
  validate-backend instavar-voice-backend.json
python /path/to/instavar-voice-evaluation/main.py \
  validate-backend instavar-voice-backend-full-sft.json
python /path/to/instavar-voice-evaluation/main.py \
  validate-backend-registry instavar-voice-backend-registry.json
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

Set `PERSISTED_PACKAGE_ROOT` to an existing directory outside the lifecycle
work directory, companion checkout, Qwen checkout, and any local base-model
input tree. Preflight verifies fsynced no-overwrite hard-link publication and
records the resolved path, filesystem device, and directory inode. The package
stage rechecks that identity, reuses only a byte-identical existing object, and
writes `package/persisted-package.json`. LoRA and full-SFT names are separated,
so equal package bytes cannot collapse the two adaptation modes into one object.
This contract does not prove a real retained model package, remote backup,
restore, access control, distribution rights, or defense against every
adversarial filesystem race.

### Full SFT lifecycle

[`instavar-voice-backend-full-sft.json`](instavar-voice-backend-full-sft.json)
declares a second five-stage backend. It uses
[`scripts/train_full_sft.py`](scripts/train_full_sft.py), not the official
`sft_12hz.py`, because the official trainer at upstream revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e` still lacks the complete
text-projection and label-alignment corrections documented in this repo.

The full-SFT trainer:

- optimizes all model parameters without PEFT;
- computes codec-0 loss with one explicit target shift;
- applies `text_projection` when the model exposes it;
- derives the saved custom-speaker row from the explicit deterministic
  `SPEAKER_REFERENCE_INDEX` training record, defaulting to row zero;
- seeds the single-process training path through `TRAIN_SEED`, defaulting to
  42, and records both seed and reference-row index in checkpoint metadata;
- aborts on non-finite loss and rejects reuse of a speaker ID already assigned
  to another name;
- writes the canonical speaker row into a copied checkpoint state dict, leaving
  the live model unchanged when training continues to another epoch;
- saves the model and processor together for fresh-process reload;
- copies and content-addresses the speech tokenizer required by fresh reload;
- saves a nested Accelerate state for trusted, same-contract, single-process
  resume at the next epoch boundary, including model, optimizer, scaler, and RNG
  state;
- hashes the train and validation manifests plus all training controls into the
  resume contract, content-addresses every resume-state file, and rejects drift;
- content-addresses every regular file in a local base-model directory and
  records per-epoch loss, validation loss, elapsed time, and peak CUDA memory;
- supports explicit `TRAIN_ROW_LIMIT` and `VALIDATION_ROW_LIMIT` bounds for
  qualification runs without changing or partially copying the source manifests;
- rejects multi-process execution until distributed save and reload behavior
  has reproduced evidence.

The lifecycle requires a clean pinned Qwen checkout. Do not apply the LoRA
patch to that checkout. `SELECTED_CHECKPOINT_NAME` selects one child produced
under the full-SFT output, such as `checkpoint-epoch-2`. The registry chooses
the LoRA or full-SFT recipe from the experiment manifest's `adaptation_mode`.
The recipe fixes its declared runtime to CUDA, bfloat16, and FlashAttention 2
so observed evidence cannot silently drift away from the capability manifest.

Run it through the pinned evaluator rather than invoking stages by hand:

```bash
python /path/to/instavar-voice-evaluation/main.py \
  run-registered-lifecycle \
  instavar-voice-backend-registry.json \
  /path/to/full-sft-experiment.json \
  --work-dir /path/to/new-empty-work-dir
```

This path remains `experimental`. The
[bounded RTX 3090 Ti qualification](reports/full-sft-bounded-gpu-2026-08-14.md)
records one optimization step, an interrupted epoch-boundary resume for one
more step, a fresh-process reload, and ten frozen generation rows with complete
objective metric coverage. Two pronunciation rows failed the preregistered
content gate, and no blind listening was performed. This is execution evidence,
not convergence, quality, full-corpus, package-retention, or production evidence.
Plan substantial free disk space: each retained 0.6B checkpoint used about 7.4
GB because the inference model, speech tokenizer, optimizer, and RNG state are
kept together until packaging separates resume state from inference artifacts.
The recorded RNG seed improves reproducibility but does not promise
bit-identical CUDA kernels or outputs across hardware and dependency versions.

Resume only from a checkpoint created by this trainer in a trusted local
environment:

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-epoch-0 \
TRUST_RESUME_STATE=1 \
EPOCHS=3 \
bash scripts/run_full_sft_train.sh
```

`EPOCHS` is the total target, not the number of additional epochs. The trainer
restores through Accelerate only after rebuilding the same model, optimizer, and
dataloaders, and starts at `completed_epochs`. It rejects changed manifests,
hyperparameters, speaker controls, symbolic state roots, modified state files,
and already-complete targets. The explicit trust flag is required because
optimizer state may use PyTorch serialization. Do not resume an untrusted or
downloaded state directory. Resume is supported only between completed epochs
in one process and the same dependency environment; mid-epoch, distributed,
and cross-version equivalence remain unverified.

Future full-SFT checkpoints also write a metadata-bound `trainer-state.json`
with deterministic epoch-boundary progress. Together with the single model,
optimizer, scheduler, and random-state files under `resume-state`, this exposes
the five independent file roles required by Instavar Voice evaluator 0.45.
`evaluator_full_sft_artifact_paths(...)` rechecks the live metadata-bound bytes
and rejects missing or ambiguous state and cross-role hardlinks.

The trainer creates an explicit constant `LambdaLR`, registers it with
Accelerate, and advances it only on optimizer updates. The schedule leaves the
configured learning rate unchanged while ensuring that a real checkpoint, not
only a synthetic fixture, contains the independently addressable scheduler
state required by evaluator 0.45. Its type and step interval are part of the
training contract.

On resume, the trainer derives the canonical initial speaker embedding before
restoring the checkpoint. Building its one-row DataLoader iterator consumes CPU
Torch RNG, so doing it after restoration would introduce resume-only RNG drift.
The checkpoint load is therefore the final RNG boundary before resumed epoch
work begins.

Older Accelerate checkpoints without `trainer-state.json` remain resumable
under their original metadata contract, but they are not eligible for the 0.45
claim tier. The bounded GPU continuation predates schema 1.1 live-conditioning
receipts and is not upgraded. See
[`reports/resume-evaluator-045-instrumentation-2026-08-14.md`](reports/resume-evaluator-045-instrumentation-2026-08-14.md).

The lifecycle excludes `resume-state` from the selected inference archive and
research package. This avoids distributing optimizer-bearing state and prevents
the full training snapshot from being mistaken for the reloadable inference
artifact. Keep the original training output separately if resume is required.

### Experimental OpenAI-compatible speech server

[`tools/openai_speech_server.py`](tools/openai_speech_server.py) exposes one
fixed LoRA adapter or full-SFT checkpoint through `GET /healthz`, `GET
/readyz`, and `POST /v1/audio/speech`. Requests can supply text and optional
instructions, but cannot select paths, checkpoints, seeds, speakers, or output
destinations. The process validates the registered voice before readiness,
serializes mutable model generation, rejects overlap with HTTP 429, and writes
only bounded server-owned temporary WAV files.

The fixed-artifact startup receipt hashes allowlisted imported runtime source,
the inference model, optional adapter, and generation controls without
retaining local paths. It excludes unrelated training outputs and cache state.
The live qualification tools bind a frozen plan row to the HTTP result, compare
it with the matching CLI row, and probe malformed plus overlapping requests.

The first full-SFT CUDA drill produced a valid instruction-bearing row, matched
the neutral CLI WAV byte-for-byte, and reproduced the receipt plus WAV after a
complete restart. A separate 1.7B epoch-10 LoRA drill matched its frozen
long-form CLI WAV byte-for-byte, reproduced it after restart, and passed the
same bounded request probes. These are narrow fixed-artifact compatibility
results, not quality, load, gateway, or production claims. See
[`docs/openai-compatible-serving.md`](docs/openai-compatible-serving.md) and
[`reports/openai-speech-http-runtime-2026-08-14.md`](reports/openai-speech-http-runtime-2026-08-14.md).

### Frozen multi-prompt evaluation

Run every Qwen row from an Instavar Voice generation plan while loading the
base model and adapter once:

```bash
python scripts/run_evaluation_suite.py \
  --qwen-dir /path/to/Qwen3-TTS \
  --inference-mode adapter \
  --base-model /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --adapter /path/to/checkpoint-epoch-10 \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id qwen3-epoch10 \
  --runtime-id pytorch \
  --output-dir evaluation/qwen3-epoch10
```

For a full-SFT checkpoint, replace `--base-model` and `--adapter` with
`--inference-mode full-sft --model /path/to/checkpoint-epoch-2` and use runtime
ID `pytorch_full_sft`.

For an unchanged upstream-base control, use the Base model's ICL voice-clone
path with the exact retained reference shared by the comparison:

```bash
python scripts/run_evaluation_suite.py \
  --qwen-dir /path/to/Qwen3-TTS \
  --inference-mode base-clone \
  --base-model /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --reference-audio /path/to/female01-reference.wav \
  --reference-text "The exact transcript of the reference recording." \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id qwen3-base-clone \
  --runtime-id pytorch \
  --output-dir evaluation/qwen3-base-clone
```

Base-clone mode forbids adapter and full-SFT artifacts, verifies that the model
declares `tts_model_type: base`, and rejects instruction-bearing plan rows
because the Base voice-clone API does not support CustomVoice instructions.
Adapter and full-SFT modes reject reference inputs. Every observation records
an explicit artifact mode and device-aware runtime label, including failed
attempts. Legacy adapter and full-SFT commands without `--inference-mode`
remain accepted only when their artifact flags select exactly one condition.

The runner records one observation for every planned attempt, including
failures, and writes audio under the plan's expected path. It does not run ASR,
speaker similarity, or human listening and therefore does not make a quality
claim.

The first Base ICL versus epoch-10 LoRA long-form pair is documented in
[`reports/matched-long-form-base-adapter-2026-08-13.md`](reports/matched-long-form-base-adapter-2026-08-13.md).
It completed objective and non-directional prosody coverage and prepared a
focused blind pack. It did not produce a quality winner or listening ratings.

The executable lifecycle passes `--allow-invalid-output` so invalid generations
remain evidence instead of aborting before packaging. It then uses evaluator
revision `8feadf7bbda75abe1c305c63e362c41b86451cda` to create
`generation-attempt-receipt.json` and the runtime-bound
`objective-observations.json`. Timing and memory from the raw generation file
must not be used for a version 1.1 comparison before that binding step.

For an exact cross-runtime experiment, also pass `--artifact-set-id` and
`--artifact-set-sha256` together. The runner rejects partial or malformed
bindings. Generate and live-verify the corresponding runtime artifact manifest
with evaluator revision `8feadf7bbda75abe1c305c63e362c41b86451cda` before
using `compare-runtimes`. Converted artifacts remain `derived`, not exact.

| Script | Purpose |
|--------|----------|
| `scripts/run_lora_train.sh` | Training launcher with validated config |
| `scripts/run_lora_infer.sh` | Single-sentence inference |
| `scripts/run_full_sft_train.sh` | Single-process full-SFT launcher with known trainer fixes |
| `scripts/run_full_sft_infer.py` | Fresh-process full-model inference |
| `scripts/train_full_sft.py` | Companion-owned full-weight trainer and checkpoint writer |
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
| **Full SFT** (experimental lifecycle in this repo) | [instavar/qwen3-tts-lora-finetuning](https://github.com/instavar/qwen3-tts-lora-finetuning) | Full-weight adaptation with the known trainer fixes, strict provenance, fresh reload, and frozen evaluation. One bounded two-step GPU resume and ten-row generation smoke is recorded; convergence, quality, full-corpus, and retained-package evidence remain open |
| **Full SFT** (official trainer) | [QwenLM/Qwen3-TTS/finetuning](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) | Upstream reference implementation. Recheck pitfalls #1 and #2 against the exact revision before use |
| **Full SFT + WebUI** | [mozi1924/Qwen3-TTS-EasyFinetuning](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning) | Automated preprocessing + Gradio interface. Good for users who want a GUI workflow. Does not include LoRA support or the upstream bug fixes |
| **ComfyUI integration** | [DarioFT/ComfyUI-Qwen3-TTS](https://github.com/DarioFT/ComfyUI-Qwen3-TTS) | Fine-tuning and inference within ComfyUI node workflows |
| **Audiobook pipeline + LoRA** | [Finrandojin/alexandria-audiobook](https://github.com/Finrandojin/alexandria-audiobook) | LoRA training embedded in a Gradio audiobook workflow with per-line style control |

If you need full SFT with a friendlier interface and do not need strict
lifecycle provenance, `mozi1924/Qwen3-TTS-EasyFinetuning` is worth evaluating.
Use this repo's full-SFT lifecycle when you need fail-closed source binding,
dataset lineage, fresh reload, and the common frozen evaluation contract. Use
the LoRA path when adapter size, scale control, and the existing validated run
matter more than full-weight capacity.


## License

Apache-2.0

## Instavar Voice conformance

[`instavar-voice-capabilities.json`](instavar-voice-capabilities.json) declares the adaptation, runtime, evaluation, and rights boundaries supported by this companion. CI validates it against the pinned public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation). New lifecycle and resume-evidence runs should use evaluator commit `29c38cfd86b889abc8b79df063c817dd8f684903` or a deliberately reviewed successor so POSIX stage timeouts clean the complete process group and schema 1.1 receipts bind live conditioning artifacts. This does not retroactively upgrade earlier run evidence. See [`INSTAVAR_VOICE_CONFORMANCE.md`](INSTAVAR_VOICE_CONFORMANCE.md) for the evidence interpretation rules and local validation command.

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
pronunciation or accent judgments. Version 0.23 preregisters criterion-specific
blind-listening assignments so lexical pronunciation, cadence, fatigue, and
emotion ratings only cover prompts that can support those claims while
preserving candidate-symmetric coverage.
Version 0.24 binds exact requested text, optional instructions, and lexical
target surfaces into each blind stimulus while excluding accepted ASR aliases
and candidate identity. Reviewers no longer need an uncontrolled prompt file.
Version 0.25 binds each listening criterion to a reviewer question, low and
high scale anchors, and an explicit score direction. Harm criteria remain raw
and separate instead of being silently inverted or folded into a composite.
Version 0.26 adds deterministic per-rater presentation schedules that
counterbalance candidate precedence within each prompt and seed. Aggregation
recomputes the private audit, requires the scheduled pseudonymous rater set,
and keeps order, fatigue, carryover, and reviewer-compliance limits explicit.
Version 0.27 exports one privacy-preserving packet per pseudonymous rater and
binds criterion-major presentation logs plus ratings into canonical submission
receipts. Aggregation reconstructs each packet, rejects forged metadata, and
records missing reviewers or cells as attrition. Receipt hashes establish
content integrity, not reviewer identity, delivery, attention, or independence.
This companion bundles neither model
weights nor optional extractor dependencies and runs neither learned metric
automatically. Run them explicitly after generation with trusted, content-addressed
models, frozen decoding, and a preregistered reference plan where applicable.
Runtime-bound observations, same-recording smoke scores, or human-recording ASR
alone are not TTS-quality evidence.
