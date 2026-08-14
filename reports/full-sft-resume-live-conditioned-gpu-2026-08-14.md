# Qwen3-TTS full-SFT evaluator 0.45 GPU resume evidence

Date: 2026-08-14, Asia/Singapore

## Result

A fresh uninterrupted two-epoch full-SFT process and an independently
interrupted and resumed process produced byte-identical model, optimizer,
scheduler, trainer, and RNG role files on one RTX 3090 Ti. Instavar Voice
evaluator 0.45 rehashed four live conditioning inputs, both run receipts, the
interruption receipt, and all ten independently stored final-role files.

The comparison passed at
`byte_exact_live_conditioned_artifact_set` with no mismatched roles. This is
bounded evidence for the exact one-row, two-update, single-process,
same-version epoch-boundary configuration. It does not establish perceptual
quality, adaptation benefit, arbitrary continuation, cross-version behavior,
or distributed equivalence.

## Bound revisions and runtime

- companion: `5469f4105dc3464dcbc70290e6f8bfb42726e637`
- evaluator: `29c38cfd86b889abc8b79df063c817dd8f684903`
- Qwen source: `6cafe5582caea83df269c36b1ce62d953a9cc66b`
- model: Qwen3-TTS-12Hz-0.6B-Base
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- driver: `580.173.02`
- Python: `3.11.9`
- Torch: `2.5.1`, CUDA `12.1`
- Transformers: `4.57.3`
- Accelerate: `1.12.0`
- training seed: `42`
- epochs and total updates: 2 and 2
- training and validation rows consumed: 1 and 1
- batch size and gradient accumulation: 1 and 1
- learning rate: `2e-6`
- scheduler: constant `LambdaLR`, stepped per optimizer update
- precision and attention: bfloat16 and FlashAttention 2

The companion, evaluator, and Qwen source were clean detached checkouts at the
bound revisions. The Base model directory, full train and validation manifests,
training controls, and initial-state receipt were content-bound before the
comparison.

## OOD findings and negative control

The planned run exposed two gaps that synthetic mapper tests did not catch.

First, the full-SFT trainer had no scheduler. Real Accelerate checkpoints
therefore lacked `scheduler.bin`, even though the mapper required that role.
The trainer now creates a constant scheduler, registers it with Accelerate,
advances it only on optimizer updates, and binds its semantics into the training
contract.

The first real run after that fix produced identical model, optimizer,
scheduler, and trainer roles but mismatched CPU Torch RNG:

- uninterrupted RNG SHA-256:
  `fab91243a50e16cb95edab3b6c3e35599cd647d7c500a46b9c620ecf5ae28db6`
- resumed RNG SHA-256:
  `3bfb99fb3e0a03ade7963d6b8e4b998bdfdc098eecd5c9b8f14c93c29540e121`
- negative evaluator report internal SHA-256:
  `cee63a25946edf707a4179426133baff66e6f884488063fb306179f33f75d811`
- negative evaluator report file SHA-256:
  `0ea02124d9e137cd06931717f0d5a8e5a1c2d476220506757434b34a0880a68d`

Semantic inspection found that Python, NumPy, CUDA, and every other compared
role were equal. Only CPU Torch RNG differed. The resumed process restored RNG
and then created a one-row DataLoader iterator while deriving the canonical
speaker embedding. The uninterrupted path performs that initial-model work
only before epoch zero. Moving speaker-embedding derivation before checkpoint
restoration made restoration the final RNG boundary before resumed epoch work.

The negative control remains retained. It demonstrates that matching losses
and model bytes do not make an RNG mismatch disappear and that evaluator 0.45
fails closed on one differing role.

## Interruption evidence

The interrupted process ran in its own process group. The harness observed the
complete epoch-zero checkpoint metadata, confirmed that one scheduler role
existed, sent `SIGTERM` to the process group, and waited for exit status `143`.
Epoch-one output and partial checkpoint paths were absent before a separate
resume process began.

- interrupted checkpoint metadata SHA-256:
  `21993083a51393ea272d79d3db71cc377b055f9f11ec3a84c40ddb5f534e2296`
- interrupted trainer state SHA-256:
  `c3329e0a09553f86f02dc0faee2192e7e48e845de835ebecb762448247a45030`
- interruption receipt SHA-256:
  `f3917f978b8018fa2553ba9973adfd18fe5e34624830c1356e2661f54c612973`
- epoch-zero loss in both initial processes: `12.6527`
- epoch-zero validation loss in both initial processes: `12.8585`
- epoch-one loss in uninterrupted and resumed processes: `12.5317`
- epoch-one validation loss in uninterrupted and resumed processes: `12.8119`

## Five-role comparison

| Role | Bytes | SHA-256 | Exact |
| --- | ---: | --- | --- |
| `model_state` | 1,829,344,304 | `875fab61c93985d3fa3d14c2adb0e7d34fcd06429934445bb62ed0b4d2481b75` | yes |
| `optimizer_state` | 3,623,492,872 | `1babc418ddaa87b432626cfbde41de76e75b871af745f4dfd2b32b10a173c3b5` | yes |
| `rng_state` | 14,408 | `fab91243a50e16cb95edab3b6c3e35599cd647d7c500a46b9c620ecf5ae28db6` | yes |
| `scheduler_state` | 1,000 | `f25b1c329a47674327e5c32955dcec8bf8001680faabef7ca349168d6698359a` | yes |
| `trainer_state` | 145 | `dd17fe1b9b7766008feeca2839c3c98a0f80c07e654f57317d4996bf0ecccd29` | yes |

The independently saved reloadable top-level model files also matched at
SHA-256
`07e26ee2fd194865d05922d6a8c9e1c5c09879ac0a559680274e70c4ced5526f`.
That file is additional evidence, not one of evaluator 0.45's five resume
roles.

## Retained evidence

- complete positive remote root:
  `/mnt/work/chee-wei-jie/voice-models/instavar-qwen-full-sft-resume-live-045-20260814-v2`
- complete negative-control remote root:
  `/mnt/work/chee-wei-jie/voice-models/instavar-qwen-full-sft-resume-live-045-20260814`
- compact remote export:
  `/mnt/work/chee-wei-jie/voice-model-outputs/evaluation/qwen-full-sft-resume-live-045-20260814`
- hash-verified local compact export:
  `/Users/CheeWeiJie/Downloads/desktop-tailscale-tts/qwen-full-sft-resume-live-045-20260814`
- compact manifest SHA-256:
  `e61371d1970281bced991ddf7e26ada895b8c03ec2431e9d10be14d8b3a1a4d7`
- evaluator report internal SHA-256:
  `de5d3ecf129978129775b3bda6fa623705a5ec1f4beab9b22fd972f0745c2645`
- evaluator report file SHA-256:
  `6c85e5484001fc50f2549edccdea6de1c26dce78cd1d9819ca3db92545297cff`
- run summary file SHA-256:
  `5e4e9a6a9e4c7ddedce58814b18ef12687033f2fc6de2c10fb2034a437c383e1`

The 28 manifest entries verified in both compact locations. The compact export
omits the large model and optimizer role files, whose hashes remain in the
report. Rehashing those roles independently requires the complete remote root.

## Evidence boundary and remaining gaps

The evaluator records `proves_training_semantics: false`,
`proves_numerical_resume_equivalence: false`, and `proves_model_quality:
false`. It verifies declared live inputs and final artifact bytes but cannot
prove that every trainer operation honored the receipts, that no hidden state
exists, or that floating-point trajectories are generally equivalent. No audio
was generated in this paired-resume experiment.

Remaining work includes more rows, longer runs, accumulated gradients,
mid-epoch interruption, stochastic workers, another dependency stack,
cross-version resume, and distributed training. Quality claims still require
matched Base-versus-adapted outputs, multi-seed objective evaluation, and blind
listening.
