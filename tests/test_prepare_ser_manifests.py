import csv
import json
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path

from tools import prepare_ser_manifests as manifests


def item(name, group, label="happy", speaker=None):
    relative = Path("Corpus") / group / f"{name}.wav"
    return manifests.Item(
        path=Path("/source") / relative,
        relative_path=relative,
        label=label,
        speaker=speaker or group,
        group=group,
        frames=160,
    )


class ProtocolTests(unittest.TestCase):
    def test_emodb_uses_frozen_outer_and_inner_speakers(self):
        folds = manifests.make_folds("emodb")
        self.assertEqual([fold.fold for fold in folds], list(range(1, 11)))
        self.assertEqual(
            [fold.outer_valid_groups[0] for fold in folds],
            list(manifests.EMODB_SPEAKERS),
        )
        self.assertEqual(
            [fold.inner_valid_groups[0] for fold in folds],
            list(manifests.EMODB_INNER_VALID),
        )
        for fold in folds:
            self.assertNotEqual(fold.outer_valid_groups, fold.inner_valid_groups)

    def test_ravdess_uses_published_five_folds_for_both_subsets(self):
        expected = {
            0: ("02", "05", "14", "15", "16"),
            1: ("03", "06", "07", "13", "18"),
            2: ("10", "11", "12", "19", "20"),
            3: ("08", "17", "21", "23", "24"),
            4: ("01", "04", "09", "22"),
        }
        for dataset in ("ravdess-speech", "ravdess-song"):
            folds = manifests.make_folds(dataset)
            self.assertEqual(
                {fold.fold: fold.outer_valid_groups for fold in folds}, expected
            )
        self.assertNotIn(
            "18", manifests.DATASET_SPECS["ravdess-song"].expected_group_counts
        )

    def test_iemocap_uses_session_layout_instead_of_stale_wav_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            datasets_root = Path(temporary) / "SER_DATASET"
            corpus_root = datasets_root / "IEMOCAP-5531"
            audio_dir = corpus_root / "Session1"
            audio_dir.mkdir(parents=True)
            utterance = "Ses01F_impro01_F000"
            audio_path = audio_dir / f"{utterance}.wav"
            with wave.open(str(audio_path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\x00\x00" * 160)
            with (corpus_root / "metadata.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("utterance_id", "emotion", "session", "speaker", "wav"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "utterance_id": utterance,
                        "emotion": "neu",
                        "session": "1",
                        "speaker": "Ses01F",
                        "wav": "/stale/export/path.wav",
                    }
                )

            loaded = manifests.load_iemocap(corpus_root, datasets_root)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].path, audio_path.resolve())
            self.assertEqual(
                loaded[0].relative_path,
                Path("IEMOCAP-5531/Session1/Ses01F_impro01_F000.wav"),
            )
            self.assertEqual(loaded[0].group, "Session1")
            self.assertEqual(loaded[0].label, "neutral")

    def test_song_loader_rejects_actor_folder_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            datasets_root = Path(temporary) / "SER_DATASET"
            actor_dir = datasets_root / "RAVDESS-Song" / "Actor_01"
            actor_dir.mkdir(parents=True)
            path = actor_dir / "03-02-01-01-01-01-02.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\x00\x00" * 16)
            with self.assertRaisesRegex(ValueError, "folder and filename disagree"):
                manifests.load_ravdess_song(actor_dir.parent, datasets_root)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.items = [item("a", "A"), item("b", "B"), item("c", "C")]
        self.spec = manifests.DatasetSpec(
            slug="synthetic",
            directory="Corpus",
            labels=("happy",),
            expected_samples=3,
            expected_label_counts={"happy": 3},
            expected_group_counts={"A": 1, "B": 1, "C": 1},
            protocol="synthetic_v1",
            loader=lambda root, datasets_root: [],
        )
        self.folds = [
            manifests.FoldDefinition(0, ("A",), ("B",)),
            manifests.FoldDefinition(1, ("B",), ("C",)),
            manifests.FoldDefinition(2, ("C",), ("A",)),
        ]

    def test_source_and_fold_coverage_audits_accept_exact_partition(self):
        manifests.audit_source(self.spec, self.items)
        payloads = []
        for fold in self.folds:
            splits = manifests.split_fold(self.items, fold)
            manifests.audit_fold(self.spec, fold, splits)
            payloads.append((fold, splits))
        manifests.audit_fold_coverage(self.spec, self.items, payloads)

    def test_source_audit_rejects_duplicate_paths(self):
        duplicate = manifests.Item(
            path=self.items[0].path,
            relative_path=self.items[0].relative_path,
            label="happy",
            speaker="C",
            group="C",
            frames=160,
        )
        with self.assertRaisesRegex(ValueError, "duplicate audio paths"):
            manifests.audit_source(self.spec, [self.items[0], self.items[1], duplicate])

    def test_source_audit_rejects_non_16khz_audio(self):
        invalid = [replace(self.items[0], sample_rate=48000), *self.items[1:]]
        with self.assertRaisesRegex(ValueError, "mono 16 kHz"):
            manifests.audit_source(self.spec, invalid)

    def test_source_audit_rejects_non_mono_audio(self):
        invalid = [replace(self.items[0], channels=2), *self.items[1:]]
        with self.assertRaisesRegex(ValueError, "mono 16 kHz"):
            manifests.audit_source(self.spec, invalid)

    def test_coverage_audit_rejects_sample_assigned_twice(self):
        payloads = [
            (fold, manifests.split_fold(self.items, fold)) for fold in self.folds
        ]
        broken = dict(payloads[1][1])
        broken["valid"] = list(broken["valid"]) + [self.items[0]]
        payloads[1] = (payloads[1][0], broken)
        with self.assertRaisesRegex(ValueError, "outer-valid coverage is not exactly once"):
            manifests.audit_fold_coverage(self.spec, self.items, payloads)


class OutputTests(unittest.TestCase):
    def test_fold_outputs_are_relative_hashed_and_role_annotated(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "synthetic"
            dataset_dir.mkdir()
            metadata = dataset_dir / "metadata.csv"
            metadata.write_text("audio_path,label\nCorpus/A/a.wav,happy\n", encoding="utf-8")
            sample = item("a", "A")
            spec = manifests.DatasetSpec(
                slug="synthetic",
                directory="Corpus",
                labels=("happy",),
                expected_samples=1,
                expected_label_counts={"happy": 1},
                expected_group_counts={"A": 1},
                protocol="synthetic_v1",
                loader=lambda root, datasets_root: [],
            )
            split_items = {name: [sample] for name in manifests.SPLIT_NAMES}
            server_root = Path("/root/songhao/datasets/SER_DATASET")
            manifests.write_fold(
                dataset_dir,
                server_root,
                spec,
                [sample],
                manifests.FoldDefinition(1, ("A",), ("B",)),
                split_items,
                metadata,
            )

            fold_dir = dataset_dir / "fold_01"
            lines = (fold_dir / "train.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], str(server_root))
            self.assertEqual(lines[1], "Corpus/A/a.wav\t160")
            self.assertFalse(Path(lines[1].split("\t")[0]).is_absolute())

            payload = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], "synthetic")
            self.assertEqual(payload["protocol"], "synthetic_v1")
            self.assertEqual(
                payload["audio_format"], {"sample_rate_hz": 16000, "channels": 1}
            )
            self.assertEqual(payload["splits"]["train"]["role"], "outer_train")
            self.assertEqual(
                payload["splits"]["valid"]["role"], "sealed_outer_valid"
            )
            self.assertEqual(
                payload["splits"]["train"]["files"]["tsv_sha256"],
                manifests.sha256_file(fold_dir / "train.tsv"),
            )
            self.assertEqual(len(payload["audio_inventory_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
