#!/usr/bin/env python3
"""Serve one fixed Qwen3-TTS artifact through a strict OpenAI speech subset."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import ipaddress
import json
import logging
import math
import os
import random
import re
import socket
import struct
import sys
import tempfile
import threading
import time
import uuid
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

LOG = logging.getLogger("qwen3_tts.openai_speech_server")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SPEECH_FIELDS = frozenset(
    {"input", "instructions", "model", "response_format", "speed", "voice"}
)


class ApiError(Exception):
    """A bounded error that is safe to return to an HTTP client."""

    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class DuplicateJsonKey(ValueError):
    """Raised when a JSON object repeats a field name."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result


def decode_json_body(body: bytes) -> Any:
    try:
        return json.loads(body, object_pairs_hook=_strict_json_object)
    except DuplicateJsonKey as error:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "duplicate_json_field",
            "request body contains a duplicate JSON field",
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_json",
            "request body is not valid JSON",
        ) from error


def _validate_text(value: Any, *, label: str, max_chars: int, allow_empty: bool) -> str | None:
    if value is None and allow_empty:
        return None
    if not isinstance(value, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid_{label}", f"{label} must be a string")
    if not allow_empty and not value.strip():
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"invalid_{label}",
            f"{label} must be a nonempty string",
        )
    if len(value) > max_chars:
        raise ApiError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"{label}_too_large",
            f"{label} exceeds the {max_chars}-character limit",
        )
    if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"invalid_{label}",
            f"{label} contains an unsupported control character",
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"invalid_{label}",
            f"{label} is not valid Unicode",
        ) from error
    if allow_empty and not value.strip():
        return None
    return value


class SpeechEngine(Protocol):
    """Minimal engine boundary used by the HTTP service and tests."""

    def generate(
        self,
        text: str,
        instructions: str | None,
        output_path: Path,
    ) -> Mapping[str, int | float] | None: ...


@dataclass(frozen=True)
class SynthesisResult:
    audio: bytes
    generation_seconds: float
    peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class SpeechServerConfig:
    model_id: str
    voice_id: str
    max_input_chars: int = 4_000
    max_instructions_chars: int = 1_000
    max_body_bytes: int = 16_384
    max_audio_bytes: int = 100 * 1024 * 1024
    request_timeout_seconds: float = 30.0
    api_key: str | None = None
    startup_receipt_sha256: str | None = None

    def validate(self) -> None:
        for label, value in (("model id", self.model_id), ("voice id", self.voice_id)):
            if not isinstance(value, str) or not PUBLIC_ID_RE.fullmatch(value):
                raise ValueError(f"{label} must be a bounded public identifier")
        for label, value in (
            ("max input chars", self.max_input_chars),
            ("max instructions chars", self.max_instructions_chars),
            ("max body bytes", self.max_body_bytes),
            ("max audio bytes", self.max_audio_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(float(self.request_timeout_seconds))
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request timeout seconds must be finite and positive")
        if self.api_key is not None and not self.api_key:
            raise ValueError("API key must be nonempty when configured")
        if self.startup_receipt_sha256 is not None and not SHA256_RE.fullmatch(
            self.startup_receipt_sha256
        ):
            raise ValueError("startup receipt sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    instructions: str | None = None


def parse_speech_request(payload: Any, config: SpeechServerConfig) -> SpeechRequest:
    """Validate the supported OpenAI speech subset without coercing values."""

    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_request", "JSON body must be an object")
    unknown = sorted(set(payload) - ALLOWED_SPEECH_FIELDS)
    if unknown:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "unsupported_field",
            f"unsupported request field(s): {', '.join(unknown)}",
        )
    if payload.get("model") != config.model_id:
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported_model", "model is not available")
    if payload.get("voice") != config.voice_id:
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported_voice", "voice is not available")

    text = _validate_text(
        payload.get("input"),
        label="input",
        max_chars=config.max_input_chars,
        allow_empty=False,
    )
    assert text is not None
    instructions = _validate_text(
        payload.get("instructions"),
        label="instructions",
        max_chars=config.max_instructions_chars,
        allow_empty=True,
    )

    response_format = payload.get("response_format", "wav")
    if response_format != "wav":
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "unsupported_response_format",
            "only response_format='wav' is supported",
        )
    speed = payload.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported_speed", "speed must equal 1.0")
    if not math.isfinite(float(speed)) or float(speed) != 1.0:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "unsupported_speed",
            "only speed=1.0 is supported",
        )
    return SpeechRequest(text=text, instructions=instructions)


def validate_wav(path: Path, max_audio_bytes: int) -> bytes:
    """Read a bounded, positive-duration PCM WAV from a server-owned directory."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("synthesis did not produce a regular WAV file")
    size = path.stat().st_size
    if size <= 0 or size > max_audio_bytes:
        raise RuntimeError("synthesis produced an empty or oversized WAV file")
    try:
        with wave.open(str(path), "rb") as wav_file:
            if (
                wav_file.getnchannels() <= 0
                or wav_file.getsampwidth() <= 0
                or wav_file.getframerate() <= 0
                or wav_file.getnframes() <= 0
            ):
                raise RuntimeError("synthesis produced a zero-duration WAV file")
    except (EOFError, wave.Error) as error:
        raise RuntimeError("synthesis produced an invalid PCM WAV file") from error
    return path.read_bytes()


class SpeechService:
    """Serialize one mutable model engine and reject overlapping generation."""

    def __init__(self, engine: SpeechEngine, config: SpeechServerConfig) -> None:
        config.validate()
        self.engine = engine
        self.config = config
        self._generation_lock = threading.Lock()

    def synthesize(self, request: SpeechRequest) -> SynthesisResult:
        if not self._generation_lock.acquire(blocking=False):
            raise ApiError(
                HTTPStatus.TOO_MANY_REQUESTS,
                "server_busy",
                "another synthesis request is in progress",
            )
        try:
            with tempfile.TemporaryDirectory(prefix="qwen3-tts-speech-") as temporary:
                output = Path(temporary) / "response.wav"
                started = time.perf_counter()
                metrics = self.engine.generate(request.text, request.instructions, output) or {}
                elapsed = time.perf_counter() - started
                generation_seconds = metrics.get("generation_seconds", elapsed)
                peak_memory_bytes = metrics.get("peak_memory_bytes")
                if (
                    isinstance(generation_seconds, bool)
                    or not isinstance(generation_seconds, (int, float))
                    or not math.isfinite(float(generation_seconds))
                    or generation_seconds <= 0
                ):
                    raise RuntimeError("engine returned invalid generation timing")
                if peak_memory_bytes is not None and (
                    isinstance(peak_memory_bytes, bool)
                    or not isinstance(peak_memory_bytes, int)
                    or peak_memory_bytes < 0
                ):
                    raise RuntimeError("engine returned invalid peak memory")
                return SynthesisResult(
                    audio=validate_wav(output, self.config.max_audio_bytes),
                    generation_seconds=float(generation_seconds),
                    peak_memory_bytes=peak_memory_bytes,
                )
        finally:
            self._generation_lock.release()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def hash_directory_tree(
    path: Path,
    *,
    excluded_directory_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Hash all visible files by relative name and content without storing local paths."""

    root = _resolve_directory(path, "artifact directory")
    members: list[dict[str, Any]] = []
    total_bytes = 0
    candidates = (
        member
        for member in root.rglob("*")
        if not any(part in excluded_directory_names for part in member.relative_to(root).parts[:-1])
        and member.name not in excluded_directory_names
    )
    for member in sorted(candidates, key=lambda value: value.relative_to(root).as_posix()):
        relative = member.relative_to(root).as_posix()
        if member.is_symlink():
            raise ValueError(f"artifact tree contains a symbolic link: {relative}")
        resolved = member.resolve(strict=True)
        if member.is_dir():
            continue
        if not resolved.is_file():
            raise ValueError(f"artifact tree contains an unsupported entry: {relative}")
        size = resolved.stat().st_size
        members.append({"path": relative, "sha256": sha256_file(resolved), "size_bytes": size})
        total_bytes += size
    if not members:
        raise ValueError("artifact directory contains no files")
    payload = json.dumps(members, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(members),
        "size_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def hash_qwen_runtime_source(qwen_dir: Path, *, mode: str) -> dict[str, Any]:
    """Hash only source files imported by the fixed Qwen runtime."""

    root = _resolve_directory(qwen_dir, "Qwen directory")
    if mode not in {"adapter", "full-sft"}:
        raise ValueError("mode must be adapter or full-sft")
    required = [root / "qwen_tts"]
    if mode == "adapter":
        required.append(root / "finetuning" / "infer_lora_custom_voice.py")
    members: list[dict[str, Any]] = []
    total_bytes = 0
    for entry in required:
        if entry.is_symlink():
            raise ValueError(
                f"Qwen runtime source contains a symbolic link: {entry.relative_to(root).as_posix()}"
            )
        if entry.is_dir():
            candidates = sorted(
                (
                    candidate
                    for candidate in entry.rglob("*")
                    if not any(
                        part in {".DS_Store", "__pycache__"}
                        for part in candidate.relative_to(root).parts
                    )
                    and candidate.suffix != ".pyc"
                ),
                key=lambda value: value.relative_to(root).as_posix(),
            )
        elif entry.is_file():
            candidates = [entry]
        else:
            raise ValueError(
                f"Qwen runtime source is missing: {entry.relative_to(root).as_posix()}"
            )
        for member in candidates:
            relative = member.relative_to(root).as_posix()
            if member.is_symlink():
                raise ValueError(f"Qwen runtime source contains a symbolic link: {relative}")
            if member.is_dir():
                continue
            if not member.is_file():
                raise ValueError(f"Qwen runtime source contains an unsupported entry: {relative}")
            size = member.stat().st_size
            members.append({"path": relative, "sha256": sha256_file(member), "size_bytes": size})
            total_bytes += size
    if not members:
        raise ValueError("Qwen runtime source contains no files")
    payload = json.dumps(members, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(members),
        "size_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_startup_receipt(
    *,
    mode: str,
    qwen_dir: Path,
    primary_model: Path,
    adapter: Path | None,
    model_id: str,
    voice_id: str,
    speaker_name: str,
    device: str,
    dtype: str,
    attention: str,
    language: str,
    lora_scale: float,
    merge_lora: bool,
    seed: int,
    max_new_tokens: int,
    artifact_set_id: str | None,
    artifact_set_sha256: str | None,
) -> dict[str, Any]:
    if mode not in {"adapter", "full-sft"}:
        raise ValueError("mode must be adapter or full-sft")
    if bool(artifact_set_id) != bool(artifact_set_sha256):
        raise ValueError("artifact set id and sha256 must be provided together")
    if artifact_set_id is not None and not PUBLIC_ID_RE.fullmatch(artifact_set_id):
        raise ValueError("artifact set id must be a bounded public identifier")
    if artifact_set_sha256 is not None and not SHA256_RE.fullmatch(artifact_set_sha256):
        raise ValueError("artifact set sha256 must be a lowercase SHA-256 digest")
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "runtime_id": "qwen3_tts_openai_compatible_http",
        "artifact_mode": mode.replace("-", "_"),
        "model_id": model_id,
        "voice_id": voice_id,
        "speaker_name": speaker_name,
        "device": device,
        "dtype": dtype,
        "attention": attention,
        "language": language,
        "lora_scale": lora_scale,
        "merge_lora": merge_lora,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "artifacts": {
            "qwen_source": hash_qwen_runtime_source(qwen_dir, mode=mode),
            "primary_model": hash_directory_tree(
                primary_model,
                excluded_directory_names=frozenset({"resume-state"}),
            ),
            **({"adapter": hash_directory_tree(adapter)} if adapter is not None else {}),
        },
        "boundary": (
            "The receipt binds visible source and artifact file content plus fixed generation controls. "
            "It does not prove loader behavior, transitive Python dependencies, host trust, quality, rights, "
            "or production gateway behavior."
        ),
    }
    if artifact_set_id is not None:
        receipt["artifact_set_id"] = artifact_set_id
        receipt["artifact_set_sha256"] = artifact_set_sha256
    return receipt


def write_startup_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return hashlib.sha256(payload).hexdigest()


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be an integer in the interval [0, 2^63 - 1]")


def _validate_model_type(path: Path, *, expected: str, label: str) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{label} must contain config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} config.json must be valid JSON") from error
    if config.get("tts_model_type") != expected:
        raise ValueError(f"{label} config must declare tts_model_type {expected}")


def validate_adapter_base_compatibility(base_model: Path, adapter: Path) -> None:
    """Reject obvious adapter and base architecture mismatches before model loading."""

    try:
        base_config = json.loads((base_model / "config.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("base model config.json must be valid JSON") from error
    talker = base_config.get("talker_config")
    if not isinstance(talker, dict):
        raise ValueError("base model config must contain talker_config")
    expected_dimensions = {
        ".talker.model.layers.0.mlp.gate_proj.lora_A.weight": talker.get("hidden_size"),
        ".talker.model.layers.0.self_attn.q_proj.lora_A.weight": talker.get("hidden_size"),
        ".talker.model.layers.0.mlp.down_proj.lora_A.weight": talker.get("intermediate_size"),
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in expected_dimensions.values()):
        raise ValueError("base model talker dimensions must be positive integers")

    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file() or weights.is_symlink():
        raise ValueError("adapter must contain a regular adapter_model.safetensors file")
    with weights.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("adapter safetensors header is truncated")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= 64 * 1024 * 1024:
            raise ValueError("adapter safetensors header size is invalid")
        header_bytes = source.read(header_size)
    if len(header_bytes) != header_size:
        raise ValueError("adapter safetensors header is truncated")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("adapter safetensors header must be valid JSON") from error
    if not isinstance(header, dict):
        raise ValueError("adapter safetensors header must be an object")

    checked = 0
    for suffix, expected in expected_dimensions.items():
        matches = [value for key, value in header.items() if key.endswith(suffix)]
        if len(matches) > 1:
            raise ValueError(f"adapter contains duplicate architecture probe tensors for {suffix}")
        if not matches:
            continue
        shape = matches[0].get("shape") if isinstance(matches[0], dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape)
        ):
            raise ValueError(f"adapter architecture probe tensor has an invalid shape for {suffix}")
        if shape[1] != expected:
            raise ValueError(
                f"adapter is incompatible with the selected base model for {suffix}: "
                f"expected input dimension {expected}, observed {shape[1]}"
            )
        checked += 1
    if checked == 0:
        raise ValueError("adapter does not contain a supported architecture probe tensor")


def validate_loaded_voice(tts: Any, *, speaker_name: str, language: str) -> None:
    """Fail startup if the fixed public voice cannot be generated by the loaded model."""

    supported_speakers = tts.get_supported_speakers()
    if supported_speakers is not None and speaker_name.casefold() not in {
        str(value).casefold() for value in supported_speakers
    }:
        raise ValueError(
            f"speaker name is not registered in the loaded model: {speaker_name}"
        )
    supported_languages = tts.get_supported_languages()
    if supported_languages is not None and language.casefold() not in {
        str(value).casefold() for value in supported_languages
    }:
        raise ValueError(f"language is not supported by the loaded model: {language}")


class Qwen3TTSSpeechEngine:
    """Load one fixed LoRA adapter or full-SFT checkpoint exactly once."""

    def __init__(
        self,
        *,
        qwen_dir: Path,
        mode: str,
        base_model: Path | None,
        adapter: Path | None,
        model: Path | None,
        speaker_name: str,
        speaker_id: int,
        speaker_embedding: Path | None,
        language: str,
        device: str,
        dtype: str,
        attention: str,
        lora_scale: float,
        merge_lora: bool,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        if mode == "adapter":
            if base_model is None or adapter is None or model is not None:
                raise ValueError("adapter mode requires base model and adapter only")
        elif mode == "full-sft":
            if model is None or base_model is not None or adapter is not None:
                raise ValueError("full-sft mode requires model only")
        else:
            raise ValueError("mode must be adapter or full-sft")
        if dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 0:
            raise ValueError("speaker id must be a nonnegative integer")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max new tokens must be a positive integer")
        if not math.isfinite(lora_scale) or lora_scale < 0:
            raise ValueError("LoRA scale must be finite and nonnegative")
        _validate_seed(seed)
        self.qwen_dir = _resolve_directory(qwen_dir, "Qwen directory")
        self.mode = mode
        self.base_model = _resolve_directory(base_model, "base model") if base_model else None
        self.adapter = _resolve_directory(adapter, "adapter") if adapter else None
        self.model = _resolve_directory(model, "full-SFT model") if model else None
        if self.base_model is not None:
            _validate_model_type(self.base_model, expected="base", label="base model")
            assert self.adapter is not None
            validate_adapter_base_compatibility(self.base_model, self.adapter)
        if self.model is not None:
            _validate_model_type(self.model, expected="custom_voice", label="full-SFT model")
        self.speaker_name = speaker_name
        self.speaker_id = speaker_id
        self.speaker_embedding = speaker_embedding.resolve(strict=True) if speaker_embedding else None
        if self.speaker_embedding is not None and not self.speaker_embedding.is_file():
            raise ValueError("speaker embedding must be a file")
        self.language = language
        self.device = device
        self.dtype = dtype
        self.attention = attention
        self.lora_scale = lora_scale
        self.merge_lora = merge_lora
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._torch, self._soundfile, self._tts = self._load_model()

    def _load_model(self) -> tuple[Any, Any, Any]:
        finetuning_dir = self.qwen_dir / "finetuning"
        if not finetuning_dir.is_dir():
            raise FileNotFoundError(f"Qwen finetuning directory not found: {finetuning_dir}")
        sys.path.insert(0, str(self.qwen_dir))
        sys.path.insert(0, str(finetuning_dir))
        torch = importlib.import_module("torch")
        soundfile = importlib.import_module("soundfile")
        model_module = importlib.import_module("qwen_tts")
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        tts = model_module.Qwen3TTSModel.from_pretrained(
            str(self.model or self.base_model),
            device_map=self.device,
            torch_dtype=dtype_map[self.dtype],
            attn_implementation=self.attention,
        )
        if self.mode == "adapter":
            peft_model = importlib.import_module("peft").PeftModel.from_pretrained(
                tts.model,
                str(self.adapter),
            )
            helper = importlib.import_module("infer_lora_custom_voice")
            helper._set_lora_scale(peft_model, self.lora_scale)
            tts.model = peft_model.merge_and_unload() if self.merge_lora else peft_model
            core_model = helper._resolve_core_model(tts.model)
            helper._apply_speaker_config(
                core_model,
                str(self.adapter),
                self.speaker_name,
                self.speaker_id,
            )
            helper._apply_speaker_embedding(
                core_model,
                str(self.adapter),
                self.speaker_name,
                str(self.speaker_embedding) if self.speaker_embedding else None,
            )
        tts.model.eval()
        validate_loaded_voice(
            tts,
            speaker_name=self.speaker_name,
            language=self.language,
        )
        return torch, soundfile, tts

    def _set_seed(self) -> None:
        random.seed(self.seed)
        numpy = importlib.import_module("numpy")
        numpy.random.seed(self.seed % (2**32))
        self._torch.manual_seed(self.seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(self.seed)

    def generate(
        self,
        text: str,
        instructions: str | None,
        output_path: Path,
    ) -> Mapping[str, int | float]:
        self._set_seed()
        uses_cuda = self.device.split(":", 1)[0].casefold() == "cuda"
        if uses_cuda:
            self._torch.cuda.reset_peak_memory_stats()
            self._torch.cuda.synchronize()
        started = time.perf_counter()
        wavs, sample_rate = self._tts.generate_custom_voice(
            text=text,
            speaker=self.speaker_name,
            language=self.language,
            instruct=instructions,
            max_new_tokens=self.max_new_tokens,
        )
        self._soundfile.write(output_path, wavs[0], sample_rate, subtype="PCM_16")
        if uses_cuda:
            self._torch.cuda.synchronize()
        return {
            "generation_seconds": time.perf_counter() - started,
            **(
                {"peak_memory_bytes": int(self._torch.cuda.max_memory_allocated())}
                if uses_cuda
                else {}
            ),
        }


def _json_error(error: ApiError, request_id: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "code": error.code,
                "message": error.message,
                "request_id": request_id,
                "type": "invalid_request_error" if error.status < 500 else "server_error",
            }
        },
        separators=(",", ":"),
    ).encode()


def build_handler(service: SpeechService) -> type[BaseHTTPRequestHandler]:
    class SpeechHandler(BaseHTTPRequestHandler):
        server_version = "qwen3-tts-speech/1"
        sys_version = ""

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(service.config.request_timeout_seconds)

        def log_message(self, format_string: str, *args: Any) -> None:
            LOG.info("%s - %s", self.address_string(), format_string % args)

        def _request_id(self) -> str:
            return f"req_{uuid.uuid4().hex}"

        def _send_common_headers(self, request_id: str) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Request-ID", request_id)

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any], request_id: str) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_common_headers(request_id)
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, error: ApiError, request_id: str) -> None:
            body = _json_error(error, request_id)
            self.send_response(error.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_common_headers(request_id)
            self.end_headers()
            self.wfile.write(body)

        def _authenticate(self) -> None:
            expected = service.config.api_key
            if expected is None:
                return
            supplied = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not supplied.startswith(prefix) or not hmac.compare_digest(supplied[len(prefix) :], expected):
                raise ApiError(HTTPStatus.UNAUTHORIZED, "invalid_api_key", "invalid API key")

        def do_GET(self) -> None:
            request_id = self._request_id()
            try:
                self._authenticate()
                if self.path == "/healthz":
                    payload = {"status": "ok"}
                elif self.path == "/readyz":
                    payload = {
                        "status": "ready",
                        "model": service.config.model_id,
                        "voice": service.config.voice_id,
                        **(
                            {"startup_receipt_sha256": service.config.startup_receipt_sha256}
                            if service.config.startup_receipt_sha256
                            else {}
                        ),
                    }
                else:
                    raise ApiError(HTTPStatus.NOT_FOUND, "not_found", "route not found")
                self._send_json(HTTPStatus.OK, payload, request_id)
            except ApiError as error:
                self._send_error(error, request_id)

        def do_POST(self) -> None:
            request_id = self._request_id()
            try:
                self._authenticate()
                if self.path != "/v1/audio/speech":
                    raise ApiError(HTTPStatus.NOT_FOUND, "not_found", "route not found")
                transfer_encoding = self.headers.get_all("Transfer-Encoding", [])
                if transfer_encoding:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "unsupported_transfer_encoding",
                        "Transfer-Encoding is not supported",
                    )
                content_types = self.headers.get_all("Content-Type", [])
                if len(content_types) != 1:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_content_type",
                        "exactly one Content-Type header is required",
                    )
                content_type = content_types[0].split(";", 1)[0].strip().casefold()
                if content_type != "application/json":
                    raise ApiError(
                        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        "unsupported_media_type",
                        "Content-Type must be application/json",
                    )
                content_lengths = self.headers.get_all("Content-Length", [])
                if not content_lengths:
                    raise ApiError(HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required")
                if len(content_lengths) != 1 or not re.fullmatch(r"[0-9]+", content_lengths[0]):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_content_length",
                        "exactly one decimal Content-Length header is required",
                    )
                content_length = int(content_lengths[0])
                if content_length > service.config.max_body_bytes:
                    raise ApiError(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "body_too_large",
                        "request body exceeds the configured limit",
                    )
                body = self.rfile.read(content_length)
                if len(body) != content_length:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "incomplete_body", "request body is incomplete")
                request = parse_speech_request(decode_json_body(body), service.config)
                result = service.synthesize(request)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(result.audio)))
                self.send_header("X-Generation-Seconds", f"{result.generation_seconds:.6f}")
                if result.peak_memory_bytes is not None:
                    self.send_header("X-Peak-Memory-Bytes", str(result.peak_memory_bytes))
                self._send_common_headers(request_id)
                self.end_headers()
                self.wfile.write(result.audio)
            except ApiError as error:
                self._send_error(error, request_id)
            except Exception:
                LOG.exception("synthesis failed for %s", request_id)
                self._send_error(
                    ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "synthesis_failed",
                        "speech synthesis failed",
                    ),
                    request_id,
                )

    return SpeechHandler


class SpeechHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class IPv6SpeechHTTPServer(SpeechHTTPServer):
    address_family = socket.AF_INET6


def read_api_key(host: str, environment_name: str | None) -> str | None:
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.casefold() == "localhost"
    if environment_name is None:
        if is_loopback:
            return None
        raise ValueError("non-loopback listeners require --api-key-env")
    if not ENV_NAME_RE.fullmatch(environment_name):
        raise ValueError("API key environment name is invalid")
    value = os.environ.get(environment_name)
    if not value:
        raise ValueError(f"API key environment variable {environment_name} is unset or empty")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("adapter", "full-sft"), required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--speaker-name", default="female01")
    parser.add_argument("--speaker-id", type=int, default=3000)
    parser.add_argument("--speaker-embedding", type=Path)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--lora-scale", type=float, default=0.3)
    parser.add_argument("--no-merge-lora", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="qwen3-tts")
    parser.add_argument("--voice-id", default="instavar-reference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env")
    parser.add_argument("--max-input-chars", type=int, default=4_000)
    parser.add_argument("--max-instructions-chars", type=int, default=1_000)
    parser.add_argument("--max-body-bytes", type=int, default=16_384)
    parser.add_argument("--max-audio-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--startup-receipt", type=Path)
    parser.add_argument("--artifact-set-id")
    parser.add_argument("--artifact-set-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in the interval [1, 65535]")
    api_key = read_api_key(args.host, args.api_key_env)
    initial_config = SpeechServerConfig(
        model_id=args.model_id,
        voice_id=args.voice_id,
        max_input_chars=args.max_input_chars,
        max_instructions_chars=args.max_instructions_chars,
        max_body_bytes=args.max_body_bytes,
        max_audio_bytes=args.max_audio_bytes,
        request_timeout_seconds=args.request_timeout_seconds,
        api_key=api_key,
    )
    initial_config.validate()
    if bool(args.artifact_set_id) != bool(args.artifact_set_sha256):
        raise ValueError("artifact set id and sha256 must be provided together")
    if (args.artifact_set_id or args.artifact_set_sha256) and args.startup_receipt is None:
        raise ValueError("artifact set identity requires --startup-receipt")
    if args.startup_receipt is not None and (
        args.startup_receipt.exists() or args.startup_receipt.is_symlink()
    ):
        raise FileExistsError(f"startup receipt destination already exists: {args.startup_receipt}")
    engine = Qwen3TTSSpeechEngine(
        qwen_dir=args.qwen_dir,
        mode=args.mode,
        base_model=args.base_model,
        adapter=args.adapter,
        model=args.model,
        speaker_name=args.speaker_name,
        speaker_id=args.speaker_id,
        speaker_embedding=args.speaker_embedding,
        language=args.language,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
        lora_scale=args.lora_scale,
        merge_lora=not args.no_merge_lora,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    receipt_sha256 = None
    if args.startup_receipt is not None:
        primary_model = args.base_model if args.mode == "adapter" else args.model
        assert primary_model is not None
        receipt = build_startup_receipt(
            mode=args.mode,
            qwen_dir=args.qwen_dir,
            primary_model=primary_model,
            adapter=args.adapter,
            model_id=args.model_id,
            voice_id=args.voice_id,
            speaker_name=args.speaker_name,
            device=args.device,
            dtype=args.dtype,
            attention=args.attention,
            language=args.language,
            lora_scale=args.lora_scale,
            merge_lora=not args.no_merge_lora,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            artifact_set_id=args.artifact_set_id,
            artifact_set_sha256=args.artifact_set_sha256,
        )
        receipt_sha256 = write_startup_receipt(args.startup_receipt, receipt)
    config = SpeechServerConfig(
        model_id=args.model_id,
        voice_id=args.voice_id,
        max_input_chars=args.max_input_chars,
        max_instructions_chars=args.max_instructions_chars,
        max_body_bytes=args.max_body_bytes,
        max_audio_bytes=args.max_audio_bytes,
        request_timeout_seconds=args.request_timeout_seconds,
        api_key=api_key,
        startup_receipt_sha256=receipt_sha256,
    )
    service = SpeechService(engine, config)
    server_type = IPv6SpeechHTTPServer if ":" in args.host else SpeechHTTPServer
    httpd = server_type((args.host, args.port), build_handler(service))
    LOG.info("serving %s/%s on http://%s:%d", args.model_id, args.voice_id, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
