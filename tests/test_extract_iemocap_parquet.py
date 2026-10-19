import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import extract_iemocap_parquet as extractor


ROWS = [
    {
        "file": "audio/Session1/Ses01F_impro01_F000.wav",
        "major_emotion": "angry",
        "audio": {"bytes": b"RIFF-female"},
    },
    {
        "file": "audio/Session1/Ses01F_impro01_M000.wav",
        "major_emotion": "happy",
        "audio": {"bytes": b"RIFF-male"},
    },
    {
        "file": "audio/Session1/Ses01F_impro01_F001.wav",
        "major_emotion": "frustrated",
        "audio": {"bytes": b"RIFF-skip"},
    },
]


def fake_iter_rows(_shards):
    return iter(ROWS)


class ExtractIemocapParquetTest(unittest.TestCase):
    def test_normalize_filename_handles_paths_and_missing_suffix(self):
        self.assertEqual(
            extractor.normalize_filename("audio/Session1/Ses01F_impro01_F000.wav"),
            "Ses01F_impro01_F000.wav",
        )
        self.assertEqual(
            extractor.normalize_filename("Ses01F_impro01_F000"),
            "Ses01F_impro01_F000.wav",
        )
        with self.assertRaises(ValueError):
            extractor.normalize_filename(None)

    def test_identity_uses_session_prefix_and_gendered_suffix(self):
        self.assertEqual(extractor.identity("Ses01F_impro01_F000.wav"), (1, "Ses01F"))
        # The prefix gender is not the speaker; the _M suffix decides the speaker.
        self.assertEqual(extractor.identity("Ses01F_impro01_M000.wav"), (1, "Ses01M"))
        with self.assertRaises(ValueError):
            extractor.identity("not-iemocap.wav")
        with self.assertRaises(ValueError):
            extractor.identity("Ses09F_impro01_F000.wav")

    def test_audit_filters_non_protocol_emotions_and_maps_codes(self):
        with mock.patch.object(extractor, "iter_rows", side_effect=fake_iter_rows):
            records, skipped = extractor.audit_rows([Path("unused.parquet")])
        self.assertEqual([row["emotion"] for row in records], ["ang", "hap"])
        self.assertEqual([row["speaker"] for row in records], ["Ses01F", "Ses01M"])
        self.assertEqual([row["wav"] for row in records], [
            "Session1/Ses01F_impro01_F000.wav",
            "Session1/Ses01F_impro01_M000.wav",
        ])
        self.assertEqual(skipped, {"frustrated": 1})

    def test_audit_rejects_duplicate_filenames(self):
        duplicate = [ROWS[0], ROWS[0]]
        with mock.patch.object(extractor, "iter_rows", return_value=iter(duplicate)):
            with self.assertRaisesRegex(ValueError, "Duplicate IEMOCAP filename"):
                extractor.audit_rows([Path("unused.parquet")])

    def test_write_audio_creates_session_layout(self):
        with mock.patch.object(extractor, "iter_rows", side_effect=fake_iter_rows):
            records, _ = extractor.audit_rows([Path("unused.parquet")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "IEMOCAP-5531"
            with mock.patch.object(extractor, "iter_rows", side_effect=fake_iter_rows):
                extractor.write_audio([Path("unused.parquet")], records, output)
            self.assertEqual(
                (output / "Session1" / "Ses01F_impro01_F000.wav").read_bytes(),
                b"RIFF-female",
            )
            self.assertEqual(
                (output / "Session1" / "Ses01F_impro01_M000.wav").read_bytes(),
                b"RIFF-male",
            )

    def test_main_writes_manifest_compatible_metadata_and_protects_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "parquet"
            source.mkdir()
            (source / "part-0.parquet").touch()
            output = root / "IEMOCAP-5531"
            argv = [
                "extract_iemocap_parquet.py",
                str(source),
                "--output",
                str(output),
                "--expected-samples",
                "2",
            ]
            with mock.patch.object(extractor, "iter_rows", side_effect=fake_iter_rows):
                with mock.patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()):
                        extractor.main()

            with (output / "metadata.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["utterance_id"] for row in rows],
                ["Ses01F_impro01_F000", "Ses01F_impro01_M000"],
            )
            self.assertEqual([row["emotion"] for row in rows], ["ang", "hap"])
            self.assertEqual([row["session"] for row in rows], ["1", "1"])
            self.assertEqual([row["speaker"] for row in rows], ["Ses01F", "Ses01M"])

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as raised:
                    extractor.main()
            self.assertIn("not empty", str(raised.exception))

    def test_main_count_guard_runs_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "parquet"
            source.mkdir()
            (source / "part-0.parquet").touch()
            output = root / "IEMOCAP-5531"
            argv = [
                "extract_iemocap_parquet.py",
                str(source),
                "--output",
                str(output),
                "--expected-samples",
                "3",
            ]
            with mock.patch.object(extractor, "iter_rows", side_effect=fake_iter_rows):
                with mock.patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit) as raised:
                        extractor.main()
            self.assertIn("Kept 2 utterances, expected 3", str(raised.exception))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
