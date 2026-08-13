# Qwen3-TTS full-SFT HTTP runtime evidence, 2026-08-14

## Result

The experimental fixed-artifact OpenAI-compatible speech server loaded the
bounded Qwen3-TTS 0.6B full-SFT checkpoint on an RTX 3090 Ti, generated valid
neutral and instruction-bearing 24 kHz WAV files, matched a separate fresh CLI
process byte-for-byte on the neutral row, and reproduced the same startup
receipt and WAV after a complete server restart.

Malformed fixed-artifact probes failed before model entry. An overlapping
request received HTTP 429 while the primary generation completed successfully.
The server remained ready after the OOD probes.

This is a narrow runtime compatibility qualification. It does not establish
the LoRA HTTP mode, complete frozen-plan coverage, instruction obedience,
quality, sustained load, cancellation, OOM recovery, multi-worker behavior, a
real gateway, or production readiness.

## Bound environment

- executed companion revision:
  `ff0af6d811173ae975bed2fd2e9b04942575c7b5`
- upstream Qwen revision:
  `6cafe5582caea83df269c36b1ce62d953a9cc66b`
- full-SFT model tree SHA-256:
  `4c78d82330508694313d0e9249928aa5ddd25ecdb5fd444bbcec606c6d62e5ef`
- full-SFT model files and bytes: 14 and 2,527,540,564
- runtime Qwen source tree SHA-256:
  `aca794d05bd0fdd0e62c363348b800ae257738697d1f9bbf9dbc8987822344d3`
- runtime source files and bytes: 36 and 8,548,508
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- Python: 3.11.9
- PyTorch: 2.5.1 with CUDA 12.1
- Transformers: 4.57.3
- Accelerate: 1.12.0
- PEFT: 0.18.1
- device and dtype: `cuda:0` and `bf16`
- attention: `flash_attention_2`
- speaker and public voice: `female01`
- language: `auto`
- seed: 42
- maximum new tokens: 1,024

The server checkout and upstream checkout were clean after the drill. The
server processes owned by this task were stopped and ports 18140 and 18141 were
not left listening.

## Startup receipt and restart

The first successful server and the restarted server wrote byte-identical
no-overwrite receipts:

- first successful receipt SHA-256:
  `d034945e37ba108dcf9b65d6ee40486dec8f166e904fbf303785454c93085c03`
- restarted receipt SHA-256:
  `d034945e37ba108dcf9b65d6ee40486dec8f166e904fbf303785454c93085c03`

The receipt binds the visible runtime source and inference-model contents plus
the fixed speaker, language, seed, device, dtype, attention, generation cap,
and public identifiers. It excludes Git, cache, and optimizer resume state.
It does not bind every installed Python dependency or attest the host.

## Generation and CLI parity

The neutral text was:

> On a quiet morning, the streets were nearly empty.

The warm HTTP result was a mono 24 kHz, 16-bit PCM WAV with 86,400 frames and
172,844 bytes. Server generation time was 2.682225 seconds and peak allocated
CUDA memory was 2,276,924,416 bytes.

The HTTP WAV, a separate current-revision fresh CLI WAV, the earlier qualified
fresh-reload WAV, and the post-restart HTTP WAV all had SHA-256:

`840b734a72213a1568d4751e570a4feeb2e6c7792f43f921c26406cb7c829db7`

This proves exact output equality for one deterministic full-SFT row under the
bound settings. It does not prove semantic equivalence for other texts, seeds,
instructions, checkpoints, artifacts, or dependency stacks.

The instruction-bearing request used:

- text: `The rain eased just before sunset, and the city began to glow.`
- instruction: `Speak slowly, with a reflective tone.`

It produced a valid mono 24 kHz, 16-bit PCM WAV with 111,360 frames and 222,764
bytes. Its SHA-256 was
`5fcab92ccf1b7ded4b1fca2d3ba12613d496f019a2e5af5f93b9a2e262883aee`.
Generation time was 3.014352 seconds and peak allocated CUDA memory was
2,308,635,648 bytes. Valid generation proves request forwarding and runtime
compatibility, not instruction obedience or perceptual quality.

## OOD request probes

The live fixed-artifact process returned the expected result for each probe:

| Probe | Result |
| --- | --- |
| Wrong public model | HTTP 400, `unsupported_model` |
| Request-controlled checkpoint path | HTTP 400, `unsupported_field` |
| 1,001-character instructions | HTTP 413, `instructions_too_large` |
| Overlapping synthesis | HTTP 429, `server_busy` |
| Primary overlap request | HTTP 200, valid WAV |
| Duplicate `Content-Length` | HTTP 400, `invalid_content_length` |
| `Transfer-Encoding: chunked` | HTTP 400, `unsupported_transfer_encoding` |
| Readiness after probes | HTTP 200 with the same receipt SHA-256 |

The long primary overlap WAV had SHA-256
`125c3e4112be960d1e37d18670c51f65e300489ef48ec61432f672dbaaee31c0`.
The contract-probe report SHA-256 was
`fec48efd4eb35d6d45fe1db5e7c1114de2a933aae41632e9ae3e75592f6a7eeb`.

These probes do not establish behavior under many simultaneous connections,
client disconnect, request timeout, process termination during generation,
GPU OOM, worker restart, reverse proxy normalization, or adversarial network
traffic.

## Failure that improved the implementation

The first real server started with public voice `speaker`, returned ready, and
then failed generation because the loaded full-SFT checkpoint registered only
`female01`. Dependency-free tests had validated fixed voice selection but did
not prove that the configured voice existed inside the loaded artifact.

The implementation now queries the loaded model's supported speakers and
languages before binding the listener. An absent voice or language fails
startup instead of producing a false ready state. Revision
`ff0af6d811173ae975bed2fd2e9b04942575c7b5` includes that fix and its
dependency-free test.

## Retained evidence

The retained evidence root is:

`/mnt/work/chee-wei-jie/voice-models/instavar-qwen-http-20260814`

Important files include the two successful startup receipts, neutral HTTP and
CLI WAV files, post-restart WAV, instruction WAV, overlap WAV, response-header
records, OOD probe report, and restart comparison. Failed receipts and the
initial false-readiness attempt were preserved instead of overwritten.
