# Experimental OpenAI-compatible speech serving

## Scope

`tools/openai_speech_server.py` serves one operator-selected Qwen3-TTS LoRA
adapter or full-SFT checkpoint through a strict subset of the OpenAI speech
route. It is a reference runtime for local evaluation and controlled
integration testing. It is not a production gateway, a multi-tenant voice
service, or proof of speech quality.

| Route | Contract |
| --- | --- |
| `GET /healthz` | The process accepts HTTP requests. |
| `GET /readyz` | The fixed model and registered voice passed startup validation. |
| `POST /v1/audio/speech` | JSON input and PCM WAV output for the fixed model and voice. |

Accepted speech fields are `model`, `input`, `voice`, `instructions`,
`response_format`, and `speed`. The last two default to `wav` and `1.0`. The
server rejects every other response format and speed.

## Fixed-artifact boundary

Only the operator can choose the following at startup:

- Qwen source checkout
- either one base model plus one LoRA adapter, or one full-SFT model
- speaker name and optional adapter speaker embedding
- device, precision, attention implementation, language, and generation cap
- LoRA scale and merged or unmerged adapter mode
- deterministic seed
- public model and voice identifiers
- optional artifact-set identity and startup receipt destination

HTTP requests cannot name a path, checkpoint, seed, speaker, or output
destination. The process validates the registered speaker and language before
binding the listener. Generated audio is written to a server-owned temporary
directory, validated as a bounded positive-duration PCM WAV, read into the
response, and removed.

When `--startup-receipt` is supplied, the server writes a no-overwrite canonical
receipt after model loading and before serving. The receipt hashes the imported
`qwen_tts` package, the LoRA helper when adapter mode is selected, the inference
model, and the adapter when present. It excludes unrelated training outputs,
model caches, Git state, bytecode caches, and optimizer resume state. Artifact
and source trees reject symbolic links.
It also binds the fixed generation controls. `/readyz` exposes the receipt byte
SHA-256. A receipt does not prove loader honesty, transitive dependency
identity, host trust, quality, rights, or backup durability.

## Example full-SFT server

```bash
python tools/openai_speech_server.py \
  --qwen-dir /path/to/Qwen3-TTS \
  --mode full-sft \
  --model /path/to/checkpoint \
  --speaker-name female01 \
  --model-id qwen3-tts-full-sft \
  --voice-id female01 \
  --device cuda:0 \
  --dtype bf16 \
  --attention flash_attention_2 \
  --seed 42 \
  --startup-receipt /new/path/startup-receipt.json
```

For LoRA, use `--mode adapter --base-model ... --adapter ...`. The default
loads and merges the adapter once. Add `--no-merge-lora` only when that exact
unmerged path is deliberately under test.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "qwen3-tts-full-sft",
    "voice": "female01",
    "input": "The rain eased just before sunset.",
    "instructions": "Speak slowly, with a reflective tone."
  }' \
  --output response.wav
```

## Concurrency, limits, and authentication

One process owns one mutable model instance. Synthesis uses a nonblocking lock.
A second overlapping generation receives HTTP 429 instead of entering the
model or waiting in an unbounded queue. The server also enforces:

- a 16 KiB JSON body limit by default
- 4,000 input characters and 1,000 instruction characters by default
- a 100 MiB generated WAV limit by default
- one decimal `Content-Length` and no `Transfer-Encoding`
- exactly one `Content-Type: application/json`
- strict duplicate-field, unknown-field, Unicode, and value validation
- fixed seed reset before every serialized generation
- bounded client errors and redacted engine errors

These are process guards, not gateway controls. They do not provide TLS,
request-rate policy, bounded connection count, cancellation, worker restart,
GPU OOM recovery, or sustained-load supervision.

The default listener is `127.0.0.1`. A non-loopback listener requires
`--api-key-env`. The server reads the bearer key from the named environment
variable and never accepts the secret as a command-line value. Put any remote
deployment behind TLS and a reviewed gateway.

## Qualification tools

The dependency-light tools bind live output to a frozen generation-plan row,
compare HTTP and CLI audio under the same artifact mode, and exercise malformed
plus overlapping requests:

```bash
python tools/qualify_openai_speech_runtime.py \
  --endpoint http://127.0.0.1:8000 \
  --model-id qwen3-tts-full-sft \
  --voice-id female01 \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id qwen3-full-sft \
  --sample-id qwen3-full-sft--neutral-brief--seed-42 \
  --artifact-mode full_sft \
  --expected-startup-receipt-sha256 SHA256 \
  --output-dir evaluation/http/neutral-brief

python tools/validate_http_cli_parity.py \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id qwen3-full-sft \
  --sample-id qwen3-full-sft--neutral-brief--seed-42 \
  --cli-observations evaluation/cli/generation-observations.json \
  --http-observation evaluation/http/neutral-brief/http-generation-observation.json \
  --startup-receipt startup-receipt.json \
  --output evaluation/http/neutral-brief/parity.json

python tools/probe_openai_speech_runtime.py \
  --endpoint http://127.0.0.1:8000 \
  --model-id qwen3-tts-full-sft \
  --voice-id female01 \
  --input "A long overlap probe keeps generation active." \
  --expected-startup-receipt-sha256 SHA256 \
  --include-concurrency \
  --output evaluation/http/contract-probes.json
```

Run all dependency-free repository tests with the pinned evaluator on
`PYTHONPATH`:

```bash
PYTHONPATH=/path/to/instavar-voice-evaluation \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

## Evidence boundary

The full-SFT and LoRA CUDA qualifications are recorded in
`reports/openai-speech-http-runtime-2026-08-14.md`. It establishes a clean
fixed-artifact load, one exact HTTP versus CLI row, one instruction-bearing
row, exact restart reproduction, and bounded malformed plus overlapping
requests for full-SFT. The LoRA slice independently loaded the 1.7B epoch-10
adapter, reproduced one frozen long-form CLI WAV exactly through HTTP and after
restart, and repeated the bounded request probes. Its source checkout was not
clean, so the receipt binds the allowlisted executable-source content rather
than asserting clean-checkout provenance. These results do not qualify
multiple seeds, the complete frozen prompt pack, the unmerged adapter route,
sustained load, disconnect cancellation, OOM recovery, multi-worker behavior,
a real gateway, or perceptual quality.
