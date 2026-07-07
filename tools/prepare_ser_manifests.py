#!/usr/bin/env python3
"""Create speaker-independent EAT manifests for supported SER corpora."""
import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class Item:
    path: Path
    label: str
    speaker: str
    group: str


RAVDESS_LABELS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
EMODB_LABELS = {
    "W": "angry", "L": "boredom", "E": "disgust", "A": "fear",
    "F": "happy", "T": "sad", "N": "neutral",
}
IEMOCAP_LABELS = {
    "angry": "angry", "neutral": "neutral", "sad": "sad",
    "happy": "happy", "excited": "happy",
}


def load_ravdess(root: Path):
    items = []
    for path in sorted(root.rglob("*.wav")):
        parts = path.stem.split("-")
        if len(parts) != 7 or parts[2] not in RAVDESS_LABELS:
            raise ValueError(f"Unexpected RAVDESS filename: {path.name}")
        speaker = f"Actor_{parts[6]}"
        items.append(Item(path.resolve(), RAVDESS_LABELS[parts[2]], speaker, speaker))
    return items, list(RAVDESS_LABELS.values())


def load_emodb(root: Path):
    items = []
    for path in sorted(root.rglob("*.wav")):
        if len(path.stem) < 6 or path.stem[5] not in EMODB_LABELS:
            raise ValueError(f"Unexpected EMO-DB filename: {path.name}")
        speaker = path.stem[:2]
        items.append(Item(path.resolve(), EMODB_LABELS[path.stem[5]], speaker, speaker))
    return items, list(EMODB_LABELS.values())


def load_iemocap(root: Path):
    metadata = root / "metadata.csv"
    items = []
    with metadata.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label = IEMOCAP_LABELS.get(row["major_emotion"])
            if label is None:
                continue
            path = root / "wav" / row["file"]
            items.append(Item(path.resolve(), label, row["speaker_id"], row["session"]))
    return items, ["angry", "happy", "neutral", "sad"]


def make_folds(dataset: str, items):
    groups = sorted({item.group for item in items})
    if dataset == "ravdess":
        if len(groups) != 24:
            raise ValueError(f"Expected 24 RAVDESS actors, found {len(groups)}")
        test_groups = [groups[i:i + 4] for i in range(0, 24, 4)]
    elif dataset == "emodb":
        if len(groups) != 10:
            raise ValueError(f"Expected 10 EMO-DB speakers, found {len(groups)}")
        test_groups = [[group] for group in groups]
    else:
        if len(groups) != 5:
            raise ValueError(f"Expected 5 IEMOCAP sessions, found {len(groups)}")
        test_groups = [[group] for group in groups]

    folds = []
    for index, test in enumerate(test_groups):
        valid = test_groups[(index + 1) % len(test_groups)]
        train = [group for group in groups if group not in test and group not in valid]
        folds.append((train, valid, test))
    return folds


def write_split(directory: Path, split: str, items):
    items = sorted(items, key=lambda item: str(item.path))
    common_root = Path(os.path.commonpath([item.path for item in items]))
    if common_root.is_file():
        common_root = common_root.parent
    portable_root = common_root.relative_to(Path.cwd().resolve())
    with (directory / f"{split}.tsv").open("w", encoding="utf-8") as manifest, \
         (directory / f"{split}.lbl").open("w", encoding="utf-8") as labels:
        manifest.write(f"{portable_root}\n")
        for item in items:
            frames = sf.info(item.path).frames
            manifest.write(f"{item.path.relative_to(common_root)}\t{frames}\n")
            labels.write(f"{item.path.stem}\t{item.label}\n")


def verify(split_items):
    speakers = {name: {item.speaker for item in items} for name, items in split_items.items()}
    paths = {name: {item.path for item in items} for name, items in split_items.items()}
    for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
        if speakers[left] & speakers[right]:
            raise ValueError(f"Speaker leakage between {left} and {right}")
        if paths[left] & paths[right]:
            raise ValueError(f"File leakage between {left} and {right}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/root/songhao/datasets/ser_datasets"),
        help="Root directory that contains ravdess-speech, emodb, and iemocap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/songhao/datasets/ser_datasets/manifests"),
        help="Where to write speaker-independent manifests",
    )
    args = parser.parse_args()
    loaders = {
        "ravdess": (load_ravdess, args.datasets_root / "ravdess-speech"),
        "emodb": (load_emodb, args.datasets_root / "emodb"),
        "iemocap": (load_iemocap, args.datasets_root / "iemocap"),
    }

    for dataset, (loader, root) in loaders.items():
        items, label_order = loader(root)
        dataset_dir = args.output / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with (dataset_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["audio_path", "label", "speaker_id", "group_id", "dataset"])
            writer.writerows((item.path, item.label, item.speaker, item.group, dataset) for item in items)

        for fold_index, (train_groups, valid_groups, test_groups) in enumerate(make_folds(dataset, items), 1):
            fold_dir = dataset_dir / f"fold_{fold_index:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            split_groups = {"train": train_groups, "eval": valid_groups, "test": test_groups}
            split_items = {
                name: [item for item in items if item.group in groups]
                for name, groups in split_groups.items()
            }
            verify(split_items)
            for split, selected in split_items.items():
                write_split(fold_dir, split, selected)
            with (fold_dir / "label_descriptors.csv").open("w", encoding="utf-8") as stream:
                for index, label in enumerate(label_order):
                    stream.write(f"{index},{label}\n")
            summary = {
                name: {
                    "groups": groups,
                    "speakers": sorted({item.speaker for item in split_items[name]}),
                    "samples": len(split_items[name]),
                    "labels": dict(sorted(Counter(item.label for item in split_items[name]).items())),
                }
                for name, groups in split_groups.items()
            }
            (fold_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        print(f"{dataset}: {len(items)} samples, {len(make_folds(dataset, items))} folds")


if __name__ == "__main__":
    main()
