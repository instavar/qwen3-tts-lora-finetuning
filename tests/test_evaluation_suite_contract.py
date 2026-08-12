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
        self.assertIn("provide exactly one of --adapter or --model", source)
        self.assertIn('artifact_kind = "full_sft"', source)
        self.assertIn('artifact_kind = "adapter"', source)

    def test_full_sft_trainer_keeps_known_fixes_and_fails_closed_on_multi_process(self) -> None:
        source = (ROOT / "scripts" / "train_full_sft.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('hasattr(model.talker, "text_projection")', source)
        self.assertIn("labels=None", source)
        self.assertIn("codec_0_labels[:, 1:]", source)
        self.assertIn("accelerator.num_processes != 1", source)
        self.assertIn("speaker_reference_index", source)
        self.assertIn("torch.utils.data.Subset", source)
        self.assertIn("shuffle=False", source)
        self.assertIn("torch.isfinite(total_loss).all()", source)
        self.assertIn("is already assigned", source)
        self.assertIn("set_seed(args.seed, device_specific=True)", source)
        self.assertIn('"no": torch.float32', source)
        self.assertIn("math.isfinite(value)", source)
        self.assertIn("state_dict[codec_weight_key] = weight", source)
        self.assertNotIn("core_model.talker.model.codec_embedding.weight", source)
        self.assertIn("processor.save_pretrained(output_dir)", source)
        self.assertNotIn("from peft", source)

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
