import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PrepareEatCheckpointTest(unittest.TestCase):
    def make_checkpoints(self, root: Path) -> tuple[Path, Path]:
        base = root / "base.pt"
        torch.save(
            {
                "model": {
                    "encoder.weight": torch.ones(4, 4),
                    "encoder.bias": torch.ones(4),
                    "decoder.weight": torch.full((2, 2), 3.0),
                },
                "cfg": {"model": {"num_classes": 0}},
            },
            base,
        )
        source = root / "ft.pt"
        torch.save(
            {
                "model": {
                    "model.encoder.weight": torch.zeros(4, 4),
                    "model.encoder.bias": torch.zeros(4),
                    "head.weight": torch.zeros(8, 768),
                    "head.bias": torch.zeros(8),
                    "fc_norm.weight": torch.zeros(768),
                    "fc_norm.bias": torch.zeros(768),
                },
                "cfg": {"model": {"num_classes": 527}},
            },
            source,
        )
        return source, base

    def test_merges_encoder_into_pretrained_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base = self.make_checkpoints(root)
            output = root / "merged.pt"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools/prepare_eat_checkpoint.py"),
                    "--source", str(source),
                    "--base", str(base),
                    "--output", str(output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(
                set(state["model"]),
                {"encoder.weight", "encoder.bias", "decoder.weight"},
            )
            self.assertTrue(torch.equal(state["model"]["encoder.weight"], torch.zeros(4, 4)))
            self.assertTrue(torch.equal(state["model"]["decoder.weight"], torch.full((2, 2), 3.0)))
            self.assertEqual(state["cfg"]["model"]["num_classes"], 0)
            metadata = json.loads((root / "merged.pt.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["removed_wrapper_keys"],
                ["fc_norm.bias", "fc_norm.weight", "head.bias", "head.weight"],
            )
            self.assertEqual(metadata["encoder_keys"], 2)
            self.assertEqual(metadata["merged_encoder_keys"], 2)
            self.assertEqual(metadata["base_only_keys"], 1)
            self.assertEqual(len(metadata["output_sha256"]), 64)

    def test_refuses_to_overwrite_without_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base = self.make_checkpoints(root)
            output = root / "merged.pt"
            output.write_bytes(b"occupied")
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools/prepare_eat_checkpoint.py"),
                    "--source", str(source),
                    "--base", str(base),
                    "--output", str(output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
