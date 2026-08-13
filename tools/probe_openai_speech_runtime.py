#!/usr/bin/env python3
"""Exercise malformed and overlapping requests against a live Qwen speech server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path

try:
    from qualify_openai_speech_runtime import LiveClient, checked_json
except ModuleNotFoundError:
    from tools.qualify_openai_speech_runtime import LiveClient, checked_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-startup-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--include-concurrency", action="store_true")
    return parser.parse_args()


def error_code(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    value = payload["error"].get("code")
    return value if isinstance(value, str) else None


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise ValueError("API key environment variable is unset or empty")
    client = LiveClient(args.endpoint, args.timeout_seconds, api_key)
    ready_status, _, ready_body = client.request("GET", "/readyz")
    ready = checked_json(ready_status, ready_body, "readyz")
    if ready.get("startup_receipt_sha256") != args.expected_startup_receipt_sha256:
        raise RuntimeError("readyz did not bind the expected startup receipt")

    cases = [
        (
            "request_path_injection",
            {
                "model": args.model_id,
                "voice": args.voice_id,
                "input": "probe",
                "checkpoint": "/tmp/unreviewed",
            },
            400,
            "unsupported_field",
        ),
        (
            "unsupported_model",
            {"model": "other-model", "voice": args.voice_id, "input": "probe"},
            400,
            "unsupported_model",
        ),
        (
            "unsupported_format",
            {
                "model": args.model_id,
                "voice": args.voice_id,
                "input": "probe",
                "response_format": "mp3",
            },
            400,
            "unsupported_response_format",
        ),
        (
            "oversized_instructions",
            {
                "model": args.model_id,
                "voice": args.voice_id,
                "input": "probe",
                "instructions": "x" * 1_001,
            },
            413,
            "instructions_too_large",
        ),
    ]
    results: list[dict] = []
    for case_id, payload, expected_status, expected_code in cases:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        status, headers, response = client.request("POST", "/v1/audio/speech", body)
        observed_code = error_code(response)
        results.append(
            {
                "case_id": case_id,
                "expected_status": expected_status,
                "observed_status": status,
                "expected_error_code": expected_code,
                "observed_error_code": observed_code,
                "request_id": headers.get("x-request-id"),
                "passed": status == expected_status and observed_code == expected_code,
            }
        )

    if args.include_concurrency:
        payload = json.dumps(
            {"model": args.model_id, "voice": args.voice_id, "input": args.input},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        barrier = threading.Barrier(3)
        outcomes: list[dict] = []
        lock = threading.Lock()

        def request() -> None:
            barrier.wait()
            status, headers, response = client.request("POST", "/v1/audio/speech", payload)
            with lock:
                outcomes.append(
                    {
                        "status": status,
                        "error_code": error_code(response),
                        "request_id": headers.get("x-request-id"),
                        **(
                            {"audio_sha256": hashlib.sha256(response).hexdigest()}
                            if status == 200
                            else {}
                        ),
                    }
                )

        workers = [threading.Thread(target=request) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join()
        statuses = sorted(outcome["status"] for outcome in outcomes)
        results.append(
            {
                "case_id": "overlapping_generation",
                "expected_statuses": [200, 429],
                "observed": sorted(outcomes, key=lambda value: value["status"]),
                "passed": statuses == [200, 429]
                and any(value.get("error_code") == "server_busy" for value in outcomes),
            }
        )

    report = {
        "schema_version": "1.0.0",
        "runtime_id": "qwen3_tts_openai_compatible_http",
        "startup_receipt_sha256": args.expected_startup_receipt_sha256,
        "results": results,
        "passed": all(result["passed"] for result in results),
        "boundary": (
            "These probes cover fixed-artifact request rejection and optional overlapping generation. "
            "They do not establish ingress security, sustained load, cancellation, GPU OOM recovery, "
            "multi-worker behavior, or production readiness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("probe output already exists")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
