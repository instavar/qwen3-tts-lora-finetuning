from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class EvaluationSuiteContractTests(unittest.TestCase):
    def test_runner_loads_model_once_and_records_every_attempt(self) -> None:
        source = (ROOT / "scripts" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertEqual(source.count("Qwen3TTSModel.from_pretrained"), 1)
        self.assertIn("generation-observations.json", source)
        self.assertIn("expected_audio_path", source)
        self.assertIn("artifact set id and sha256 must be provided together", source)
        self.assertIn('"runtime_id": args.runtime_id', source)
        self.assertIn('"artifact_set_sha256": args.artifact_set_sha256', source)
        self.assertIn('"observation_schema_version": "1.0.0"', source)
        self.assertIn("allow-invalid-output", source)
        self.assertIn('not in {"1.0.0", "1.1.0"}', source)
        self.assertNotIn("max_memory_allocated()) if torch.cuda.is_available() else 0", source)
        self.assertTrue(any(isinstance(node, ast.For) for node in ast.walk(tree)))

    def test_lifecycle_binds_runtime_attempt_evidence_before_archiving(self) -> None:
        source = (ROOT / "scripts" / "instavar_voice_lifecycle.py").read_text(encoding="utf-8")
        self.assertIn("build-generation-attempt-receipt", source)
        self.assertIn("apply-generation-attempt-receipt", source)
        self.assertIn("objective-observations.json", source)

    def test_single_inference_forwards_generation_cap(self) -> None:
        patch = (ROOT / "patches" / "0001-qwen3-tts-lora.patch").read_text(encoding="utf-8")
        self.assertIn("+        max_new_tokens=args.max_new_tokens,", patch)


if __name__ == "__main__":
    unittest.main()
