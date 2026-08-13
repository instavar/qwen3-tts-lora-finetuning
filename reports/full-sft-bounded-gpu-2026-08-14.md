# Qwen3-TTS bounded full-SFT GPU qualification, 2026-08-14

## Outcome

The companion's experimental Qwen3-TTS 0.6B full-SFT path completed a clean,
single-GPU bounded train, an epoch-boundary continuation from saved optimizer
and RNG state, a fresh-process checkpoint reload, and ten frozen generation
rows. The final matrix had ten valid 24 kHz WAV files and complete coverage for
all nine required objective metrics.

This is a compatibility and lifecycle smoke. It is not evidence of convergence,
full-corpus behavior, perceptual quality, accent fidelity, production readiness,
or a successfully retained lifecycle package.

## Frozen scope

- Companion runner revision: `0d37a624a2a3f935d3d04e43708949f6eeba99d3`
- Generation-plan commit: `d20abe0e4e4813cca2066d96d23386c18b30ff4f`
- Clean upstream revision: `6cafe5582caea83df269c36b1ce62d953a9cc66b`
- Evaluator revision: `3030dd1658cd8b74b7cbc87cdf5fbb2b018152fd`
- Model: Qwen3-TTS-12Hz-0.6B-Base
- GPU: NVIDIA GeForce RTX 3090 Ti with 24,564 MiB
- Runtime: Python 3.11.9, PyTorch 2.5.1 with CUDA 12.1,
  Transformers 4.57.3, Accelerate 1.12.0
- Training: bfloat16, FlashAttention 2, batch size one, one training row, one
  validation row, seed 42
- Frozen plan: four prompts and ten samples across neutral Singapore English,
  local context, names and numbers, and structured long form

The local base-model manifest SHA-256 was
`e2696531bdf27be632b1d1ab8459fbdba958fd19b9c78b1917c443f4fce5093d`.
The complete train and validation manifests were hashed even though the runner
consumed only their first row. Their SHA-256 values were
`cb84119c9f9959030c297b61dccd6720a38056d8d2837821f59f3906c7700aee`
and `111a1c30b29f5f8618905773513e98448dbc36a63e3ca575b14ec76eb8ad3db0`.

## Training and interruption resume

The initial process completed one optimization step:

- training loss: 12.652665138244629
- validation loss: 12.85853099822998
- peak CUDA allocated memory: 9,103,285,760 bytes
- peak CUDA reserved memory: 9,302,966,272 bytes
- metadata SHA-256:
  `7733b8850341acd1d11b5556f3318140a36e60e2b834013073ced417319e9b04`

A separate process trusted that local checkpoint only after its training,
runtime, metadata, and resume-state manifests matched. It continued at epoch
one and completed one additional optimization step:

- training loss: 12.531721115112305
- validation loss: 12.811945915222168
- peak CUDA allocated memory: 9,106,292,224 bytes
- peak CUDA reserved memory: 9,305,063,424 bytes
- source resume-state SHA-256:
  `c647ec9a828476dbe17a1fc3b3cf9cdd63feb6631274ce73a24466d9b1d30faf`
- resumed metadata SHA-256:
  `b7a788d19094099d7e668abc323486e16b659e191c16f5366db64db4ab428234`

The small loss changes are observations, not evidence of convergence or
improved quality.

## Checkpoint reload

The resumed checkpoint preserved the original source-compatible nested config,
changed only the custom-voice registration, and copied the speech tokenizer.
The tokenizer tree SHA-256 was
`8bf7372b8cab3ed2f8713a0dd0e3e1b14fc0717fc12f7ffcb6463d46c2f3ff67`.
A fresh CUDA process loaded the checkpoint and wrote a valid WAV with SHA-256
`840b734a72213a1568d4751e570a4feeb2e6c7792f43f921c26406cb7c829db7`.

The loader reported that `speaker_encoder.*` weights were unused. This is
expected for the saved custom-voice architecture: the encoder is used during
training to derive the registered speaker embedding, while custom-voice
inference reads that embedding from the registered codec row. The warning is
retained rather than hidden.

## Frozen generation and objective evaluation

All ten rows were valid. Durations ranged from 9.84 to 32.8 seconds.

- mean ASR word error rate: 0.06318504190844616
- maximum ASR word error rate: 0.23333333333333334
- mean ECAPA speaker similarity: 0.7455952233850496
- mean generation time: 9.508049546950497 seconds
- mean real-time factor: 0.6638293700301922
- mean peak CUDA memory: 2,580,895,590.4 bytes
- mean silence fraction: 0.3850921552626975
- clipping fraction: 0 for every row
- sample rate: 24,000 Hz for every row
- required objective metric coverage: 1.0 for every metric

The content gate failed because two of the three names-and-numbers rows exceeded
the configured high-WER threshold. Eight rows were not flagged. No row showed
repetition excess, reference-transcript overlap, or spoken-instruction overlap.
The reference transcript used for speaker provenance and overlap checks was
ASR-derived and was not human-verified.

Key evidence SHA-256 values:

- generation observations:
  `dea1632b7e58ebea51886cea4799164c11e543a62edc1dedfca628206e66ebc9`
- generation-attempt receipt:
  `5abcee2fa1c440f7a107ac992c5e840ace31d9ee7be0558c285448c7aa88b3b6`
- complete observations:
  `22ae2a4093297fa1dede09bc1fc85fb5a8b4ef43812d6e53b07dcd8126e7ce55`
- content-faithfulness report:
  `b769a5f9d35186a8c8a7855d0dbe58357441fd9e3684484700b8eae8323bca14`
- objective report:
  `5eca3a53667f2ac1b464dee36bb7d09cd1f7228d20185832607767178c3e9d80`
- coverage report:
  `d3054678e2168fca6eefecf6c7ad73ee30cec1815eb1d2bfebd96ada946081b5`

No blind listening was performed. Speaker embeddings and ASR do not measure
Singapore English accent fidelity, lexical pronunciation, cadence, monotony,
naturalness, or listening fatigue.

## Failures retained during qualification

The live probe found four checkpoint gaps that dependency-free tests had not
exposed:

1. Transformers 4.57 failed while diff-serializing an injected nested `dtype`.
2. A first compatibility shim became part of `to_dict` and therefore was not
   JSON serializable.
3. Serializing the complete in-memory nested config preserved runtime-only
   fields that Qwen config constructors reject on reload.
4. A loadable model checkpoint still failed because the required
   `speech_tokenizer` directory had not been copied.

The final implementation uses the original source config as the serialization
base, changes only custom-speaker registration, keeps the compatibility shim on
the config class only for the known `dtype` failure, and copies a verified
speech-tokenizer tree. The evidence pipeline also rejected applying results to
the wrong source-observation fingerprint; each extractor was rebuilt against
the immediately preceding immutable observation document.

## Applicability and remaining gaps

The direct evidence applies to the selected 0.6B Base model, the named source
and runner revisions, one RTX 3090 Ti, and the pinned dependency set. The
serialization and tokenizer-packaging fixes are likely applicable to the same
Qwen configuration family under Transformers 4.57, but other Qwen sizes and
future versions require fresh reload tests.

Open gaps include full-corpus training, multiple seeds, meaningful checkpoint
selection, full test-split evaluation, matched base and LoRA controls, blind
listening, lifecycle package retention and restore, multi-process training,
mid-epoch resume, cross-version resume, alternative runtimes, and production
serving.
