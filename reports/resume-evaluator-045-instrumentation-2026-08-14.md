# Evaluator 0.45 resume instrumentation

Date: 2026-08-14, Asia/Singapore

## Change

Future Qwen3-TTS full-SFT checkpoints write a metadata-bound
`trainer-state.json` with deterministic epoch-boundary progress. The five
evaluator 0.45 roles map as follows:

| Evaluator role | Qwen full-SFT checkpoint member |
| --- | --- |
| `model_state` | the single `resume-state/model*.safetensors` file |
| `optimizer_state` | the single `resume-state/optimizer*.bin` file |
| `scheduler_state` | the single `resume-state/scheduler*.bin` file |
| `trainer_state` | `trainer-state.json` |
| `rng_state` | the single `resume-state/random_states*.pkl` file |

The mapper rehashes the complete Accelerate state against checkpoint metadata,
rehashes trainer state against its own metadata record, and then requires one
file for every role. Cross-role hardlinks are rejected.

The full-SFT trainer now also creates an explicit constant `LambdaLR`, registers
it with Accelerate, steps it exactly when an optimizer update completes, and
binds that schedule into the training contract. This is load-bearing: without
a registered scheduler, real Accelerate checkpoints do not contain the
`scheduler*.bin` role required by the mapper, even though synthetic mapper
tests can construct one.

`trainer-state.json` records completed epochs, epoch index, microbatch count,
optimizer-step count, and training seed. Runtime diagnostics such as elapsed
time and peak memory remain in the broader metadata and are not treated as
deterministic trainer state.

## OOD and compatibility controls

Dependency-free tests cover:

- one complete five-role mapping;
- source-level creation, Accelerate registration, and optimizer-step-aligned
  advancement of the constant scheduler;
- ambiguous single-process RNG files;
- cross-role optimizer and scheduler hardlinks;
- metadata-bound Accelerate state mutation;
- changed training and runtime contracts;
- completed-target rejection; and
- source-level confirmation that trainer state is written and bound.

Existing metadata schema 1.1 and 1.2 checkpoints remain accepted by the resume
loader. A checkpoint without the new trainer-state record cannot enter the
evaluator 0.45 role mapping.

The public contract workflow pins evaluator revision
`29c38cfd86b889abc8b79df063c817dd8f684903` and verifies the live-conditioning
receipt and comparison APIs.

## Evidence boundary

No model training or GPU test was run for this instrumentation change. The
retained two-step CUDA continuation predates both the new trainer-state file and
schema 1.1 live-conditioning receipts. It is not upgraded.

A stronger comparison must preregister and fingerprint the Base artifact,
dataset-lineage receipt, training controls, and initial state. It then needs an
independent uninterrupted run and an observed interrupted-resumed run that both
reach the same target update. Evaluator 0.45 rehashes the four conditioning
artifacts and the five final-state roles before comparing them.

Even a passing report proves only byte equality for those declared files. It
does not prove trainer semantics, hidden floating-point equivalence, quality,
adaptation benefit, cross-version resume, or distributed resume.
