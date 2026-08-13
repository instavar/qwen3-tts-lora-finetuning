#!/usr/bin/env python3
"""Generate one frozen plan row through the live Qwen speech HTTP route."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import time
import wave
from pathlib import Path
from urllib.parse import urlsplit

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--artifact-mode", choices=("adapter", "full_sft"), required=True)
    parser.add_argument("--expected-startup-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_plan_row(path: Path, candidate_id: str, sample_id: str) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("unsupported generation plan schema")
    matches = [
        row
        for row in plan.get("samples", [])
        if row.get("candidate_id") == candidate_id and row.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError("candidate and sample selector must resolve exactly one plan row")
    row = matches[0]
    for name in ("sample_id", "candidate_id", "prompt_id", "category", "text"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"selected plan row has an invalid {name}")
    if isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int):
        raise ValueError("selected plan row has an invalid seed")
    if row.get("instruction") is not None and not isinstance(row.get("instruction"), str):
        raise ValueError("selected plan row has an invalid instruction")
    return row


class LiveClient:
    def __init__(self, endpoint: str, timeout: float, api_key: str | None) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an http or https origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("endpoint cannot contain credentials, query, or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("endpoint must not contain a path")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout seconds must be finite and positive")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port
        self.timeout = timeout
        self.api_key = api_key

    def _connection(self) -> http.client.HTTPConnection:
        connection_type = (
            http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        )
        return connection_type(self.host, self.port, timeout=self.timeout)

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        connection = self._connection()
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return (
                response.status,
                {key.casefold(): value for key, value in response.getheaders()},
                response.read(),
            )
        finally:
            connection.close()


def checked_json(status: int, body: bytes, label: str) -> dict:
    if status != 200:
        raise RuntimeError(f"{label} returned HTTP {status}")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} did not return a JSON object")
    return value


def main() -> int:
    args = parse_args()
    if not SHA256_RE.fullmatch(args.expected_startup_receipt_sha256):
        raise ValueError("expected startup receipt sha256 must be a lowercase SHA-256 digest")
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise ValueError("API key environment variable is unset or empty")
    row = load_plan_row(args.generation_plan, args.candidate_id, args.sample_id)
    client = LiveClient(args.endpoint, args.timeout_seconds, api_key)

    health_status, health_headers, health_body = client.request("GET", "/healthz")
    health = checked_json(health_status, health_body, "healthz")
    ready_status, ready_headers, ready_body = client.request("GET", "/readyz")
    ready = checked_json(ready_status, ready_body, "readyz")
    if ready.get("model") != args.model_id or ready.get("voice") != args.voice_id:
        raise RuntimeError("readyz returned a different fixed model or voice")
    if ready.get("startup_receipt_sha256") != args.expected_startup_receipt_sha256:
        raise RuntimeError("readyz did not bind the expected startup receipt")

    request = {
        "model": args.model_id,
        "input": row["text"],
        "voice": args.voice_id,
        "response_format": "wav",
        "speed": 1.0,
    }
    if row.get("instruction"):
        request["instructions"] = row["instruction"]
    request_body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    started = time.perf_counter()
    speech_status, speech_headers, audio = client.request("POST", "/v1/audio/speech", request_body)
    elapsed = time.perf_counter() - started
    if speech_status != 200:
        raise RuntimeError(f"speech route returned HTTP {speech_status}")
    if speech_headers.get("content-type") != "audio/wav":
        raise RuntimeError("speech route did not return audio/wav")
    if speech_headers.get("cache-control") != "no-store":
        raise RuntimeError("speech route did not return Cache-Control: no-store")
    if not speech_headers.get("x-request-id"):
        raise RuntimeError("speech route omitted X-Request-ID")
    try:
        server_generation_seconds = float(speech_headers["x-generation-seconds"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("speech route omitted valid X-Generation-Seconds") from error
    if not math.isfinite(server_generation_seconds) or server_generation_seconds <= 0:
        raise RuntimeError("speech route returned invalid X-Generation-Seconds")
    peak_memory_bytes = None
    if "x-peak-memory-bytes" in speech_headers:
        try:
            peak_memory_bytes = int(speech_headers["x-peak-memory-bytes"])
        except ValueError as error:
            raise RuntimeError("speech route returned invalid X-Peak-Memory-Bytes") from error
        if peak_memory_bytes < 0:
            raise RuntimeError("speech route returned negative X-Peak-Memory-Bytes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = args.output_dir / f"{args.sample_id}.wav"
    observation_path = args.output_dir / "http-generation-observation.json"
    if audio_path.exists() or observation_path.exists():
        raise FileExistsError("qualification output already exists")
    audio_path.write_bytes(audio)
    with wave.open(str(audio_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        duration = frames / sample_rate if sample_rate else 0.0
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
    if frames <= 0 or sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        raise RuntimeError("speech route returned an invalid or zero-duration PCM WAV")

    observation = {
        "observation_schema_version": "1.0.0",
        "sample_id": row["sample_id"],
        "candidate_id": row["candidate_id"],
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "seed": row["seed"],
        "requested_text": row["text"],
        "valid": True,
        "runtime": f"qwen3_tts_openai_compatible_http_{args.artifact_mode}",
        "runtime_id": "openai_compatible_http",
        "artifact_mode": args.artifact_mode,
        "instruction_applied": bool(row.get("instruction")),
        "audio_path": str(audio_path),
        "audio_sha256": sha256_bytes(audio),
        "audio_duration_seconds": duration,
        "generation_seconds": server_generation_seconds,
        "request_seconds": elapsed,
        **({"peak_memory_bytes": peak_memory_bytes} if peak_memory_bytes is not None else {}),
        "startup_receipt_sha256": args.expected_startup_receipt_sha256,
        "health_request_id": health_headers.get("x-request-id"),
        "ready_request_id": ready_headers.get("x-request-id"),
        "speech_request_id": speech_headers["x-request-id"],
        "health_status": health.get("status"),
        "ready_status": ready.get("status"),
    }
    observation_path.write_text(
        json.dumps(observation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
