from __future__ import annotations

import http.client
import io
import json
import socket
import struct
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from typing import Self

from tools import openai_speech_server as server


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 240)
    return output.getvalue()


class FakeEngine:
    def __init__(
        self,
        *,
        invalid: bool = False,
        error: Exception | None = None,
        metrics: dict | None = None,
    ) -> None:
        self.invalid = invalid
        self.error = error
        self.metrics = metrics
        self.requests: list[tuple[str, str | None]] = []
        self.output_paths: list[Path] = []

    def generate(self, text: str, instructions: str | None, output_path: Path) -> dict | None:
        self.requests.append((text, instructions))
        self.output_paths.append(output_path)
        if self.error:
            raise self.error
        output_path.write_bytes(b"not-wave" if self.invalid else wav_bytes())
        return self.metrics


class RunningServer:
    def __init__(self, service: server.SpeechService) -> None:
        self.httpd = server.SpeechHTTPServer(("127.0.0.1", 0), server.build_handler(service))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])


def request(
    port: int,
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    authorized: bool = True,
    content_type: str = "application/json",
) -> tuple[int, dict[str, str], bytes]:
    headers: dict[str, str] = {}
    body = None
    if authorized:
        headers["Authorization"] = "Bearer test-only-key"
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body))
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    result = (
        response.status,
        {key.casefold(): value for key, value in response.getheaders()},
        response.read(),
    )
    connection.close()
    return result


class SpeechRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice")

    def test_accepts_fixed_artifact_request_and_optional_instructions(self) -> None:
        parsed = server.parse_speech_request(
            {
                "input": "Hello from Singapore.",
                "instructions": "Speak with measured warmth.",
                "model": "fixed-model",
                "response_format": "wav",
                "speed": 1.0,
                "voice": "fixed-voice",
            },
            self.config,
        )
        self.assertEqual(parsed.text, "Hello from Singapore.")
        self.assertEqual(parsed.instructions, "Speak with measured warmth.")

    def test_rejects_request_controlled_paths_and_unknown_fields(self) -> None:
        for field in ("adapter", "checkpoint", "device", "output", "speaker", "seed"):
            with self.subTest(field=field), self.assertRaisesRegex(
                server.ApiError, "unsupported request field"
            ):
                server.parse_speech_request(
                    {
                        "input": "hello",
                        "model": "fixed-model",
                        "voice": "fixed-voice",
                        field: "/tmp/x",
                    },
                    self.config,
                )

    def test_rejects_wrong_model_voice_format_and_speed(self) -> None:
        cases = (
            ({"input": "x", "model": "other", "voice": "fixed-voice"}, "model"),
            ({"input": "x", "model": "fixed-model", "voice": "other"}, "voice"),
            (
                {
                    "input": "x",
                    "model": "fixed-model",
                    "voice": "fixed-voice",
                    "response_format": "mp3",
                },
                "response_format",
            ),
            ({"input": "x", "model": "fixed-model", "voice": "fixed-voice", "speed": 1.01}, "speed"),
            (
                {"input": "x", "model": "fixed-model", "voice": "fixed-voice", "speed": float("nan")},
                "speed",
            ),
            ({"input": "x", "model": "fixed-model", "voice": "fixed-voice", "speed": True}, "speed"),
        )
        for payload, message in cases:
            with self.subTest(payload=payload), self.assertRaisesRegex(server.ApiError, message):
                server.parse_speech_request(payload, self.config)

    def test_rejects_invalid_and_oversized_text_fields(self) -> None:
        config = server.SpeechServerConfig(
            model_id="fixed-model",
            voice_id="fixed-voice",
            max_input_chars=3,
            max_instructions_chars=3,
        )
        base = {"model": "fixed-model", "voice": "fixed-voice"}
        for value in (None, 7, "", "   ", "four"):
            with self.subTest(input=value), self.assertRaises(server.ApiError):
                server.parse_speech_request({**base, "input": value}, config)
        for value in (7, "four"):
            with self.subTest(instructions=value), self.assertRaises(server.ApiError):
                server.parse_speech_request({**base, "input": "one", "instructions": value}, config)

    def test_rejects_control_characters_unpaired_unicode_and_duplicate_fields(self) -> None:
        base = {"model": "fixed-model", "voice": "fixed-voice"}
        for field, value in (("input", "hello\x00world"), ("input", "\ud800"), ("instructions", "a\x00b")):
            with self.subTest(field=field, value=repr(value)), self.assertRaises(server.ApiError):
                server.parse_speech_request({**base, "input": "hello", field: value}, self.config)
        with self.assertRaisesRegex(server.ApiError, "duplicate JSON field"):
            server.decode_json_body(
                b'{"model":"fixed-model","voice":"fixed-voice","input":"first","input":"second"}'
            )

    def test_blank_instructions_are_normalized_to_absent(self) -> None:
        parsed = server.parse_speech_request(
            {
                "input": "hello",
                "instructions": "  \t",
                "model": "fixed-model",
                "voice": "fixed-voice",
            },
            self.config,
        )
        self.assertIsNone(parsed.instructions)


class StartupValidationTests(unittest.TestCase):
    @staticmethod
    def _write_adapter_header(path: Path, shapes: dict[str, list[int]]) -> None:
        header = {
            name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
            for name, shape in shapes.items()
        }
        payload = json.dumps(header, separators=(",", ":")).encode()
        path.write_bytes(struct.pack("<Q", len(payload)) + payload)

    def test_public_ids_and_numeric_limits_fail_closed(self) -> None:
        for value in ("", " leading", "line\nbreak", "x" * 129):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, "public identifier"):
                server.SpeechServerConfig(model_id=value, voice_id="fixed-voice").validate()
        for kwargs in (
            {"max_input_chars": True},
            {"max_instructions_chars": 0},
            {"max_body_bytes": -1},
            {"request_timeout_seconds": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                server.SpeechServerConfig(
                    model_id="fixed-model",
                    voice_id="fixed-voice",
                    **kwargs,
                ).validate()

    def test_tree_hash_is_path_independent_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            for directory in (first, second):
                (directory / "nested").mkdir(parents=True)
                (directory / "config.json").write_text("{}\n")
                (directory / "nested" / "weights.bin").write_bytes(b"weights")
            self.assertEqual(server.hash_directory_tree(first), server.hash_directory_tree(second))
            (second / "nested" / "weights.bin").write_bytes(b"changed")
            self.assertNotEqual(
                server.hash_directory_tree(first)["tree_sha256"],
                server.hash_directory_tree(second)["tree_sha256"],
            )

    def test_tree_hash_can_exclude_non_runtime_cache_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime").mkdir()
            (root / "runtime" / "source.py").write_text("value = 1\n")
            (root / ".git").mkdir()
            (root / ".git" / "index").write_bytes(b"first")
            first = server.hash_directory_tree(
                root, excluded_directory_names=frozenset({".git"})
            )
            (root / ".git" / "index").write_bytes(b"second")
            second = server.hash_directory_tree(
                root, excluded_directory_names=frozenset({".git"})
            )
            self.assertEqual(first, second)

    def test_tree_hash_rejects_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            artifact.mkdir()
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            (artifact / "linked.bin").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                server.hash_directory_tree(artifact)

    def test_qwen_runtime_hash_excludes_training_outputs_and_requires_adapter_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qwen = Path(temporary) / "qwen"
            (qwen / "qwen_tts").mkdir(parents=True)
            (qwen / "qwen_tts" / "__init__.py").write_text("value = 1\n")
            (qwen / "finetuning").mkdir()
            helper = qwen / "finetuning" / "infer_lora_custom_voice.py"
            helper.write_text("helper = 1\n")
            output = qwen / "finetuning" / "output" / "checkpoint-epoch-10"
            output.mkdir(parents=True)
            weights = output / "adapter_model.safetensors"
            weights.write_bytes(b"first")
            cache = qwen / "qwen_tts" / "__pycache__"
            cache.mkdir()
            compiled = cache / "module.pyc"
            compiled.write_bytes(b"first")

            full_sft = server.hash_qwen_runtime_source(qwen, mode="full-sft")
            adapter = server.hash_qwen_runtime_source(qwen, mode="adapter")
            self.assertEqual(full_sft["file_count"], 1)
            self.assertEqual(adapter["file_count"], 2)
            weights.write_bytes(b"second")
            compiled.write_bytes(b"second")
            self.assertEqual(full_sft, server.hash_qwen_runtime_source(qwen, mode="full-sft"))
            self.assertEqual(adapter, server.hash_qwen_runtime_source(qwen, mode="adapter"))
            helper.unlink()
            with self.assertRaisesRegex(ValueError, "runtime source is missing"):
                server.hash_qwen_runtime_source(qwen, mode="adapter")

    def test_adapter_base_architecture_mismatch_fails_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (base / "config.json").write_text(
                json.dumps(
                    {
                        "tts_model_type": "base",
                        "talker_config": {"hidden_size": 1024, "intermediate_size": 3072},
                    }
                )
            )
            weights = adapter / "adapter_model.safetensors"
            probe = "base_model.model.talker.model.layers.0.mlp.gate_proj.lora_A.weight"
            self._write_adapter_header(weights, {probe: [16, 2048]})
            with self.assertRaisesRegex(ValueError, "incompatible with the selected base model"):
                server.validate_adapter_base_compatibility(base, adapter)
            self._write_adapter_header(weights, {probe: [16, 1024]})
            server.validate_adapter_base_compatibility(base, adapter)

    def test_startup_receipt_binds_mode_trees_and_controls_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qwen = root / "qwen"
            model = root / "model"
            adapter = root / "adapter"
            for directory, name in ((qwen, "source.py"), (model, "config.json"), (adapter, "adapter.bin")):
                directory.mkdir()
                (directory / name).write_bytes(name.encode())
            (qwen / "qwen_tts").mkdir()
            (qwen / "qwen_tts" / "__init__.py").write_text("value = 1\n")
            (qwen / "finetuning").mkdir()
            (qwen / "finetuning" / "infer_lora_custom_voice.py").write_text("helper = 1\n")
            receipt = server.build_startup_receipt(
                mode="adapter",
                qwen_dir=qwen,
                primary_model=model,
                adapter=adapter,
                model_id="fixed-model",
                voice_id="fixed-voice",
                speaker_name="speaker",
                device="cuda:0",
                dtype="bf16",
                attention="flash_attention_2",
                language="auto",
                lora_scale=0.3,
                merge_lora=True,
                seed=42,
                max_new_tokens=4096,
                artifact_set_id="artifact-set",
                artifact_set_sha256="a" * 64,
            )
            output = root / "receipt.json"
            digest = server.write_startup_receipt(output, receipt)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            serialized = output.read_text()
            self.assertNotIn(str(root), serialized)
            self.assertEqual(receipt["artifacts"]["adapter"]["file_count"], 1)
            with self.assertRaises(FileExistsError):
                server.write_startup_receipt(output, receipt)

    def test_startup_receipt_excludes_optimizer_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qwen = root / "qwen"
            model = root / "model"
            qwen.mkdir()
            model.mkdir()
            (qwen / "source.py").write_text("value = 1\n")
            (qwen / "qwen_tts").mkdir()
            (qwen / "qwen_tts" / "__init__.py").write_text("value = 1\n")
            (model / "config.json").write_text('{"tts_model_type":"custom_voice"}\n')
            (model / "resume-state").mkdir()
            (model / "resume-state" / "optimizer.bin").write_bytes(b"first")

            def build() -> dict:
                return server.build_startup_receipt(
                    mode="full-sft",
                    qwen_dir=qwen,
                    primary_model=model,
                    adapter=None,
                    model_id="fixed-model",
                    voice_id="fixed-voice",
                    speaker_name="speaker",
                    device="cuda:0",
                    dtype="bf16",
                    attention="flash_attention_2",
                    language="auto",
                    lora_scale=0.3,
                    merge_lora=True,
                    seed=42,
                    max_new_tokens=4096,
                    artifact_set_id=None,
                    artifact_set_sha256=None,
                )

            first = build()["artifacts"]["primary_model"]
            (model / "resume-state" / "optimizer.bin").write_bytes(b"second")
            self.assertEqual(first, build()["artifacts"]["primary_model"])

    def test_startup_receipt_rejects_partial_artifact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("qwen", "model"):
                directory = root / name
                directory.mkdir()
                (directory / "file").write_bytes(b"x")
            (root / "qwen" / "qwen_tts").mkdir()
            (root / "qwen" / "qwen_tts" / "__init__.py").write_text("value = 1\n")
            with self.assertRaisesRegex(ValueError, "provided together"):
                server.build_startup_receipt(
                    mode="full-sft",
                    qwen_dir=root / "qwen",
                    primary_model=root / "model",
                    adapter=None,
                    model_id="fixed-model",
                    voice_id="fixed-voice",
                    speaker_name="speaker",
                    device="cuda:0",
                    dtype="bf16",
                    attention="flash_attention_2",
                    language="auto",
                    lora_scale=0.3,
                    merge_lora=True,
                    seed=42,
                    max_new_tokens=4096,
                    artifact_set_id="artifact-set",
                    artifact_set_sha256=None,
                )


class SpeechServiceTests(unittest.TestCase):
    def test_generates_valid_wav_in_server_owned_temporary_directory(self) -> None:
        engine = FakeEngine(metrics={"generation_seconds": 0.25, "peak_memory_bytes": 123})
        service = server.SpeechService(
            engine,
            server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice"),
        )
        result = service.synthesize(server.SpeechRequest("hello", "speak warmly"))
        self.assertEqual(result.audio, wav_bytes())
        self.assertEqual(result.generation_seconds, 0.25)
        self.assertEqual(result.peak_memory_bytes, 123)
        self.assertEqual(engine.requests, [("hello", "speak warmly")])
        self.assertFalse(engine.output_paths[0].exists())

    def test_rejects_invalid_wav_and_invalid_engine_metrics(self) -> None:
        cases = (
            (FakeEngine(invalid=True), "invalid PCM WAV"),
            (FakeEngine(metrics={"generation_seconds": float("nan")}), "generation timing"),
            (FakeEngine(metrics={"peak_memory_bytes": -1}), "peak memory"),
        )
        for engine, message in cases:
            with self.subTest(message=message):
                service = server.SpeechService(
                    engine,
                    server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice"),
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    service.synthesize(server.SpeechRequest("hello"))

    def test_rejects_concurrent_generation_without_queueing(self) -> None:
        service = server.SpeechService(
            FakeEngine(),
            server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice"),
        )
        service._generation_lock.acquire()
        try:
            with self.assertRaisesRegex(server.ApiError, "another synthesis request") as raised:
                service.synthesize(server.SpeechRequest("hello"))
            self.assertEqual(raised.exception.status, 429)
        finally:
            service._generation_lock.release()


class SpeechHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine(metrics={"generation_seconds": 0.25, "peak_memory_bytes": 123})
        self.config = server.SpeechServerConfig(
            model_id="fixed-model",
            voice_id="fixed-voice",
            api_key="test-only-key",
            startup_receipt_sha256="a" * 64,
        )

    def test_live_success_returns_wav_metrics_and_security_headers(self) -> None:
        with RunningServer(server.SpeechService(self.engine, self.config)) as running:
            status, headers, body = request(
                running.port,
                "POST",
                "/v1/audio/speech",
                {
                    "input": "hello",
                    "instructions": "speak warmly",
                    "model": "fixed-model",
                    "voice": "fixed-voice",
                },
            )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "audio/wav")
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertTrue(headers["x-request-id"].startswith("req_"))
        self.assertEqual(float(headers["x-generation-seconds"]), 0.25)
        self.assertEqual(int(headers["x-peak-memory-bytes"]), 123)
        self.assertEqual(body, wav_bytes())
        self.assertEqual(self.engine.requests, [("hello", "speak warmly")])

    def test_ready_response_exposes_startup_receipt_binding(self) -> None:
        with RunningServer(server.SpeechService(self.engine, self.config)) as running:
            status, _, body = request(running.port, "GET", "/readyz")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["startup_receipt_sha256"], "a" * 64)
        self.assertEqual(payload["model"], "fixed-model")

    def test_authentication_media_type_body_limit_and_route_fail_closed(self) -> None:
        service = server.SpeechService(self.engine, self.config)
        with RunningServer(service) as running:
            status, _, body = request(running.port, "GET", "/readyz", authorized=False)
            self.assertEqual(status, 401)
            self.assertEqual(json.loads(body)["error"]["code"], "invalid_api_key")

            status, _, body = request(
                running.port,
                "POST",
                "/v1/audio/speech",
                {"input": "hello", "model": "fixed-model", "voice": "fixed-voice"},
                content_type="text/plain",
            )
            self.assertEqual(status, 415)
            self.assertEqual(json.loads(body)["error"]["code"], "unsupported_media_type")

            status, _, body = request(running.port, "POST", "/wrong", {})
            self.assertEqual(status, 404)
            self.assertEqual(json.loads(body)["error"]["code"], "not_found")
        self.assertEqual(self.engine.requests, [])

        tiny = server.SpeechServerConfig(
            model_id="fixed-model",
            voice_id="fixed-voice",
            api_key="test-only-key",
            max_body_bytes=8,
        )
        with RunningServer(server.SpeechService(self.engine, tiny)) as running:
            status, _, body = request(
                running.port,
                "POST",
                "/v1/audio/speech",
                {"input": "hello", "model": "fixed-model", "voice": "fixed-voice"},
            )
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body)["error"]["code"], "body_too_large")
        self.assertEqual(self.engine.requests, [])

    def test_duplicate_fields_and_engine_error_are_bounded(self) -> None:
        service = server.SpeechService(self.engine, self.config)
        body = b'{"model":"fixed-model","voice":"fixed-voice","input":"first","input":"second"}'
        with RunningServer(service) as running:
            connection = http.client.HTTPConnection("127.0.0.1", running.port, timeout=2)
            connection.request(
                "POST",
                "/v1/audio/speech",
                body=body,
                headers={
                    "Authorization": "Bearer test-only-key",
                    "Content-Length": str(len(body)),
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(payload["error"]["code"], "duplicate_json_field")
        self.assertEqual(self.engine.requests, [])

        failed = server.SpeechService(
            FakeEngine(error=RuntimeError("secret internal path /sensitive/model")),
            self.config,
        )
        with RunningServer(failed) as running:
            status, _, body = request(
                running.port,
                "POST",
                "/v1/audio/speech",
                {"input": "hello", "model": "fixed-model", "voice": "fixed-voice"},
            )
        self.assertEqual(status, 500)
        self.assertNotIn(b"sensitive", body)
        self.assertEqual(json.loads(body)["error"]["code"], "synthesis_failed")

    def test_duplicate_length_and_transfer_encoding_are_rejected_before_generation(self) -> None:
        body = b'{"model":"fixed-model","voice":"fixed-voice","input":"hello"}'
        requests = (
            (
                b"POST /v1/audio/speech HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer test-only-key\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body,
                "invalid_content_length",
            ),
            (
                b"POST /v1/audio/speech HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer test-only-key\r\n"
                b"Content-Type: application/json\r\n"
                b"Transfer-Encoding: chunked\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body,
                "unsupported_transfer_encoding",
            ),
        )
        service = server.SpeechService(self.engine, self.config)
        with RunningServer(service) as running:
            for payload, code in requests:
                with self.subTest(code=code):
                    connection = socket.create_connection(("127.0.0.1", running.port), timeout=2)
                    connection.sendall(payload)
                    response = b""
                    while True:
                        chunk = connection.recv(4096)
                        if not chunk:
                            break
                        response += chunk
                    connection.close()
                    self.assertIn(b" 400 ", response.split(b"\r\n", 1)[0])
                    self.assertIn(code.encode(), response)
        self.assertEqual(self.engine.requests, [])


class BindingSecurityTests(unittest.TestCase):
    def test_loopback_can_run_without_authentication(self) -> None:
        self.assertIsNone(server.read_api_key("127.0.0.1", None))
        self.assertIsNone(server.read_api_key("::1", None))
        self.assertIsNone(server.read_api_key("localhost", None))

    def test_non_loopback_requires_environment_delivered_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            server.read_api_key("0.0.0.0", None)
        with self.assertRaisesRegex(ValueError, "unset or empty"):
            server.read_api_key("0.0.0.0", "ABSENT_TEST_KEY")
        with self.assertRaisesRegex(ValueError, "invalid"):
            server.read_api_key("0.0.0.0", "BAD-NAME")

    def test_ipv6_listener_uses_ipv6_socket_family(self) -> None:
        self.assertEqual(server.IPv6SpeechHTTPServer.address_family, socket.AF_INET6)


class EngineContractTests(unittest.TestCase):
    def test_loaded_voice_must_exist_before_ready_state(self) -> None:
        class FakeTTS:
            def get_supported_speakers(self) -> list[str]:
                return ["female01"]

            def get_supported_languages(self) -> list[str]:
                return ["auto", "english"]

        server.validate_loaded_voice(FakeTTS(), speaker_name="FEMALE01", language="auto")
        with self.assertRaisesRegex(ValueError, "not registered"):
            server.validate_loaded_voice(FakeTTS(), speaker_name="speaker", language="auto")
        with self.assertRaisesRegex(ValueError, "not supported"):
            server.validate_loaded_voice(FakeTTS(), speaker_name="female01", language="klingon")

    def test_engine_requires_exactly_one_artifact_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qwen = root / "qwen"
            base = root / "base"
            adapter = root / "adapter"
            model = root / "model"
            for path in (qwen, base, adapter, model):
                path.mkdir()
            cases = (
                {"mode": "adapter", "base_model": None, "adapter": adapter, "model": None},
                {"mode": "adapter", "base_model": base, "adapter": adapter, "model": model},
                {"mode": "full-sft", "base_model": base, "adapter": None, "model": model},
                {"mode": "unknown", "base_model": None, "adapter": None, "model": model},
            )
            for values in cases:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    server.Qwen3TTSSpeechEngine(
                        qwen_dir=qwen,
                        speaker_name="speaker",
                        speaker_id=3000,
                        speaker_embedding=None,
                        language="auto",
                        device="cpu",
                        dtype="fp32",
                        attention="eager",
                        lora_scale=0.3,
                        merge_lora=True,
                        max_new_tokens=16,
                        seed=42,
                        **values,
                    )

    def test_engine_rejects_wrong_model_type_before_importing_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qwen = root / "qwen"
            (qwen / "finetuning").mkdir(parents=True)
            base = root / "base"
            adapter = root / "adapter"
            model = root / "model"
            for path in (base, adapter, model):
                path.mkdir()
            (base / "config.json").write_text('{"tts_model_type":"custom_voice"}\n')
            (model / "config.json").write_text('{"tts_model_type":"base"}\n')
            common = {
                "qwen_dir": qwen,
                "speaker_name": "speaker",
                "speaker_id": 3000,
                "speaker_embedding": None,
                "language": "auto",
                "device": "cpu",
                "dtype": "fp32",
                "attention": "eager",
                "lora_scale": 0.3,
                "merge_lora": True,
                "max_new_tokens": 16,
                "seed": 42,
            }
            with self.assertRaisesRegex(ValueError, "tts_model_type base"):
                server.Qwen3TTSSpeechEngine(
                    mode="adapter", base_model=base, adapter=adapter, model=None, **common
                )
            with self.assertRaisesRegex(ValueError, "tts_model_type custom_voice"):
                server.Qwen3TTSSpeechEngine(
                    mode="full-sft", base_model=None, adapter=None, model=model, **common
                )


if __name__ == "__main__":
    unittest.main()
