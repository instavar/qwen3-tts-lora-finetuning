from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from tools import openai_speech_server as server
from tools import probe_openai_speech_runtime as probe
from tools import qualify_openai_speech_runtime as qualify
from tools import validate_http_cli_parity as parity


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 240)
    return output.getvalue()


class SlowEngine:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str | None]] = []

    def generate(self, text: str, instructions: str | None, output_path: Path) -> dict:
        self.requests.append((text, instructions))
        time.sleep(0.05)
        output_path.write_bytes(wav_bytes())
        return {"generation_seconds": 0.05, "peak_memory_bytes": 123}


class RunningServer:
    def __init__(self, service: server.SpeechService) -> None:
        self.httpd = server.SpeechHTTPServer(("127.0.0.1", 0), server.build_handler(service))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> RunningServer:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"


def invoke(function, arguments: list[str]) -> int:
    with patch.object(sys, "argv", ["tool", *arguments]):
        return function()


class QualificationToolTests(unittest.TestCase):
    def test_live_client_rejects_ambiguous_origins_and_timeouts(self) -> None:
        for endpoint in (
            "ftp://example.com",
            "http://user@example.com",
            "http://example.com/path",
            "http://example.com?query=1",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                qualify.LiveClient(endpoint, 1.0, None)
        with self.assertRaises(ValueError):
            qualify.LiveClient("http://example.com", float("nan"), None)

    def test_qualify_and_parity_support_instruction_bound_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "startup-receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "runtime_id": "qwen3_tts_openai_compatible_http",
                        "artifact_mode": "full_sft",
                        "seed": 42,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
            plan = root / "plan.json"
            row = {
                "sample_id": "candidate--prompt--seed-42",
                "candidate_id": "candidate",
                "prompt_id": "prompt",
                "category": "instruction",
                "seed": 42,
                "text": "Please read this sentence.",
                "instruction": "Speak with measured warmth.",
            }
            plan.write_text(json.dumps({"schema_version": "1.1.0", "samples": [row]}) + "\n")
            output = root / "http"
            engine = SlowEngine()
            config = server.SpeechServerConfig(
                model_id="fixed-model",
                voice_id="fixed-voice",
                startup_receipt_sha256=receipt_sha,
            )
            with RunningServer(server.SpeechService(engine, config)) as running:
                result = invoke(
                    qualify.main,
                    [
                        "--endpoint",
                        running.endpoint,
                        "--model-id",
                        "fixed-model",
                        "--voice-id",
                        "fixed-voice",
                        "--generation-plan",
                        str(plan),
                        "--candidate-id",
                        "candidate",
                        "--sample-id",
                        row["sample_id"],
                        "--artifact-mode",
                        "full_sft",
                        "--expected-startup-receipt-sha256",
                        receipt_sha,
                        "--output-dir",
                        str(output),
                    ],
                )
            self.assertEqual(result, 0)
            observation = json.loads((output / "http-generation-observation.json").read_text())
            self.assertTrue(observation["valid"])
            self.assertTrue(observation["instruction_applied"])
            self.assertEqual(observation["artifact_mode"], "full_sft")
            self.assertEqual(engine.requests, [(row["text"], row["instruction"])])

            cli_observations = root / "cli.json"
            cli_observations.write_text(
                json.dumps(
                    [
                        {
                            **{name: row[name] for name in ("sample_id", "candidate_id", "prompt_id", "category", "seed")},
                            "requested_text": row["text"],
                            "instruction_applied": True,
                            "valid": True,
                            "artifact_mode": "full_sft",
                            "audio_sha256": observation["audio_sha256"],
                            "audio_duration_seconds": observation["audio_duration_seconds"],
                        }
                    ]
                )
                + "\n"
            )
            parity_output = root / "parity.json"
            result = invoke(
                parity.main,
                [
                    "--generation-plan",
                    str(plan),
                    "--candidate-id",
                    "candidate",
                    "--sample-id",
                    row["sample_id"],
                    "--cli-observations",
                    str(cli_observations),
                    "--http-observation",
                    str(output / "http-generation-observation.json"),
                    "--startup-receipt",
                    str(receipt),
                    "--output",
                    str(parity_output),
                ],
            )
            self.assertEqual(result, 0)
            self.assertTrue(json.loads(parity_output.read_text())["exact_wav_equivalent"])

    def test_parity_rejects_artifact_and_instruction_mismatches(self) -> None:
        plan = {
            "sample_id": "sample",
            "candidate_id": "candidate",
            "prompt_id": "prompt",
            "category": "neutral",
            "seed": 42,
            "text": "hello",
        }
        base = {
            **{name: plan[name] for name in ("sample_id", "candidate_id", "prompt_id", "category", "seed")},
            "requested_text": plan["text"],
            "valid": True,
            "audio_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "instruction state"):
            parity.validate_identity({**base, "instruction_applied": True}, plan, "row")

    def test_probe_records_fail_closed_requests_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_sha = "a" * 64
            config = server.SpeechServerConfig(
                model_id="fixed-model",
                voice_id="fixed-voice",
                startup_receipt_sha256=receipt_sha,
            )
            engine = SlowEngine()
            output = root / "probes.json"
            with RunningServer(server.SpeechService(engine, config)) as running:
                result = invoke(
                    probe.main,
                    [
                        "--endpoint",
                        running.endpoint,
                        "--model-id",
                        "fixed-model",
                        "--voice-id",
                        "fixed-voice",
                        "--input",
                        "A long enough overlap probe.",
                        "--expected-startup-receipt-sha256",
                        receipt_sha,
                        "--include-concurrency",
                        "--output",
                        str(output),
                    ],
                )
            self.assertEqual(result, 0)
            report = json.loads(output.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(
                {row["case_id"] for row in report["results"]},
                {
                    "request_path_injection",
                    "unsupported_model",
                    "unsupported_format",
                    "oversized_instructions",
                    "overlapping_generation",
                },
            )


if __name__ == "__main__":
    unittest.main()
