# Matched long-form Base ICL and LoRA evidence, 2026-08-13

## Result

One Qwen3-TTS upstream Base ICL voice-clone versus epoch-10 LoRA pair completed
the focused long-form objective, non-directional prosody, and blind-pack
pipeline. The comparison passed all nine required objective metric checks and
reported `proves_adaptation_benefit: false`. No listening ratings or quality
winner are recorded.

This is a matched practical behavior comparison, not an only-weights-changed
ablation. The upstream Base model uses `generate_voice_clone` with a retained
reference recording and exact transcript. The adapted CustomVoice path uses the
trained speaker configuration and embedding installed by the LoRA checkpoint.
Both use the same base model bytes, prompt text, seed, CUDA device, bfloat16
precision, generation cap, evaluator revision, and target speaker.

## Reproducibility anchors

- runner revision: `3e1f809ded13d05ea6b99bafdae98f71b4541544`
- clean detached runner checkout:
  `/mnt/work/chee-wei-jie/voice-models/instavar-qwen-matched-clean-20260813T1300SGT`
- evaluator revision: `982367abc7837cb6da5ebb94192c9642dea62fce`
- prompt pack: `instavar-singapore-english` version 1.2.0,
  SHA-256 `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`
- generation plan file SHA-256:
  `9064a577af26bd296b229df9c234073133c0b294ea7dc0bd6bfdbf010f7e0c2c`
- canonical generation plan SHA-256:
  `ca340a2afded2188e175d6a053ce78318c38d10cc4312b57fd9c9d7261c00de3`
- base-model tree SHA-256:
  `d177fd60ce760ac4210c8e0060fa3d79a645a4ebc5f3b836cf774765d74e4dbf`
- epoch-10 adapter tree SHA-256:
  `0f93693ec4381b15c19bcc02f69a264e60dbced7658dff362c4c8f35aaa3a926`
- retained reference audio SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- retained reference transcript SHA-256:
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`
- evidence directory:
  `/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260813_qwen3_matched_long_form_v1`

The Base ICL artifact set includes the base model, generation reference audio,
and generation reference transcript. Its canonical digest is
`3b84e5dfcfc08e4240ce4be160637edbfaae972865cac2eff2289999afb7a8ef`.
The adapter set includes the same base model and exact epoch-10 adapter tree.
Its digest is
`a7e47f5cc930108ecaa51bd509eb5bb0f0f43fecada27bab51a847a6ffd93a52`.
These bindings do not independently attest host trust or loader honesty.

Hosted contract workflows passed on both the feature branch and merged main:

- feature run: `31667817918`
- main run: `31668112435`

The clean runner checkout remained unmodified after generation.

## Objective observations

| Measurement | Base ICL | Epoch-10 LoRA | LoRA minus Base ICL |
| --- | ---: | ---: | ---: |
| Audio duration, seconds | 74.00 | 54.72 | -19.28 |
| Generation time, seconds | 50.2513 | 37.5771 | -12.6742 |
| Real-time factor | 0.679071 | 0.686715 | +0.007644 |
| Peak allocated CUDA memory, bytes | 5,053,115,904 | 5,015,686,144 | -37,429,760 |
| ASR word error rate | 0.008658 | 0.004329 | -0.004329 |
| ECAPA cosine similarity | 0.756782 | 0.488020 | -0.268762 |
| Sample rate, Hz | 24,000 | 24,000 | 0 |
| Silence fraction | 0.387428 | 0.307864 | -0.079564 |
| Clipping fraction | 0 | 0.000000761 | +0.000000761 |

Faster-whisper hypotheses cover the full requested passage for both candidates.
The duration difference therefore does not appear to be a gross truncation, but
ASR cannot establish exact prosody, naturalness, or absence of subtle omissions.

The faster-whisper extractor used revision
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` and artifact-set SHA-256
`3433b5ac25f4b005aadfcde370f3615a5d2883fe40d251e823b80204071115d6`.
The SpeechBrain ECAPA extractor used revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` and artifact-set SHA-256
`5a8cd13222e7edf1c932b8695e34c6537c15230e8e47aabe9af454284906dd7c`.
The speaker assignment was frozen after generation but before speaker scoring,
so it is symmetric but not proof of preregistration.

The matched objective comparison SHA-256 is
`bad9d23c3f3da1679e84ceee0da568970b59684ceb006616395e4116a441ee4d`.
The complete objective report SHA-256 is
`198b7d5c3af821866722863341c5a9d94c4dcac714a1d242648f73c22bd9f731`.

The ECAPA difference is large enough to prioritize blind listening and
multi-reference replication. It is not a human speaker-identity verdict. One
reference, one prompt, and different speaker-conditioning paths can all affect
the proxy. One run per candidate also cannot establish runtime or memory
advantages.

## Non-directional prosody proxies

Both outputs were eligible for the long-form proxy comparison. Selected signed
LoRA-minus-Base-ICL deltas were:

- phrase-duration coefficient of variation: `-0.089828`
- pause-duration coefficient of variation: `+0.016616`
- active RMS dB standard deviation: `-0.491848`
- window RMS dB standard deviation: `-0.007855`
- zero-crossing-rate standard deviation: `-252.926678 Hz`
- pause rate: `-3.591750` per minute

The matched prosody comparison SHA-256 is
`6b2a018ba7e488ebcf98f52cf1d632c38cc7989b83b6698426c411241ecfc47b`.
It establishes no direction and no winner. These signal proxies do not prove
cadence naturalness, reduced monotony, accent fidelity, preference, or
causation.

## Blind listening status

Two identity-neutral audio files were staged with a private reveal document.
The focused assignment includes speaker identity, cadence variation, long-form
monotony, naturalness, artifact severity, and listening fatigue. It explicitly
excludes Singapore English accent fidelity, lexical pronunciation, and emotion
obedience because the cadence-only prompt does not route to those criteria.

- listening assignment SHA-256:
  `31bda6f4b13629fbd89c309ccb64f8b677a07f4c943aa03504c70c7380f8b0db`
- identity-neutral review SHA-256:
  `d5952fe50469d353e33c71c0ab812c7c186aa53962c1bbc2223020c2786df9ec`
- staged blind-audio manifest SHA-256:
  `26f429413ac49fb70b2bfe0ffeee908107fa701c2b6c982cc4f31b198e6b3925`

No ratings were invented. The private reveal must remain closed until the
scheduled listening review is complete.

## Scope and remaining work

This run closes the missing executable upstream-base control and establishes
one content-addressed, plan-matched long-form pair. It does not replace:

1. multiple seeds, multiple references, and more long-form structures;
2. prompts that actually route to accent and lexical-pronunciation criteria;
3. completed blinded ratings from multiple reviewers;
4. a pre-generation server-stamped speaker-reference assignment;
5. repeated warm and cold runtime measurements;
6. an ablation that holds the speaker-conditioning API constant;
7. revalidation against the current official Qwen3-TTS fine-tuning path.
