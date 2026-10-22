import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SerNormStatsTest(unittest.TestCase):
    def make_data(self, root: Path) -> Path:
        audio_root = root / "audio"
        audio_root.mkdir()
        sample_rate = 16000
        waveform = 0.1 * np.sin(
            2 * np.pi * 220 * np.arange(sample_rate // 2) / sample_rate
        )
        soundfile.write(audio_root / "tone.wav", waveform, sample_rate, subtype="PCM_16")
        manifest_dir = root / "manifests" / "emodb" / "fold_01"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "inner_train.tsv").write_text(
            f"{audio_root}\ntone.wav\t{waveform.size}\n", encoding="utf-8"
        )
        return root

    def test_computes_valid_frame_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_data(Path(directory))
            output = root / "norm_stats" / "emodb" / "selection" / "fold_01.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools/compute_ser_norm_stats.py"),
                    "--dataset", "emodb",
                    "--fold", "1",
                    "--data-root", str(root),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["stats_schema"], "eat2emo-ser-fbank-v1")
            self.assertEqual(payload["dataset"], "emodb")
            self.assertEqual(payload["fold"], 1)
            self.assertEqual(payload["stage"], "selection")
            self.assertEqual(payload["split"], "inner_train")
            self.assertEqual(payload["target_length"], 640)
            self.assertEqual(payload["mel_bins"], 128)
            self.assertEqual(payload["normalization_scope"], "valid_frames")
            self.assertFalse(payload["padding_included"])
            self.assertEqual(payload["samples"], 1)
            self.assertGreater(payload["valid_frames"], 0)
            self.assertTrue(np.isfinite(payload["mel"]["mean"]))
            self.assertGreater(payload["mel"]["std"], 0)
            self.assertAlmostEqual(
                payload["mel"]["variance"],
                payload["mel"]["std"] ** 2,
                places=12,
            )
            self.assertEqual(len(payload["manifest_sha256"]), 64)

    def test_refuses_to_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_data(Path(directory))
            output = root / "stats.json"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/compute_ser_norm_stats.py"),
                "--dataset", "emodb",
                "--fold", "1",
                "--data-root", str(root),
                "--output", str(output),
            ]
            first = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("exists; pass --overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
