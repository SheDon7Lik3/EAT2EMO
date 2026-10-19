#!/usr/bin/env python3
"""Build and audit deterministic EAT manifests for the four SER datasets.

The TSV header is the only machine-specific path.  Every audio entry beneath
it is relative to the common SER_DATASET root, so manifests can be generated
locally while pointing at a different server-side root.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import soundfile as sf


SERVER_DATASETS_ROOT = Path("/root/songhao/datasets/SER_DATASET")
SPLIT_NAMES = ("train", "valid", "inner_train", "inner_valid")

EMODB_SPEAKERS = ("03", "08", "09", "10", "11", "12", "13", "14", "15", "16")
EMODB_INNER_VALID = ("13", "15", "11", "14", "09", "16", "03", "10", "08", "12")
RAVDESS_FOLDS = {
    0: ("02", "05", "14", "15", "16"),
    1: ("03", "06", "07", "13", "18"),
    2: ("10", "11", "12", "19", "20"),
    3: ("08", "17", "21", "23", "24"),
    4: ("01", "04", "09", "22"),
}

EMODB_LABELS = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}
RAVDESS_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
IEMOCAP_LABELS = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}

EMODB_FILENAME = re.compile(
    r"^(?P<speaker>03|08|09|10|11|12|13|14|15|16)[ab]\d{2}"
    r"(?P<emotion>[WLEAFTN])[a-z]\.wav$"
)
RAVDESS_FILENAME = re.compile(
    r"^(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.wav$"
)
IEMOCAP_UTTERANCE = re.compile(
    r"^Ses(?P<session>\d{2})[FM]_.+_(?P<speaker_gender>[FM])\d{3}$"
)


@dataclass(frozen=True)
class Item:
    """One canonical sample, with both its local and portable identities."""

    path: Path
    relative_path: Path
    label: str
    speaker: str
    group: str
    frames: int
    sample_rate: int = 16000
    channels: int = 1


@dataclass(frozen=True)
class FoldDefinition:
    fold: int
    outer_valid_groups: Tuple[str, ...]
    inner_valid_groups: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    directory: str
    labels: Tuple[str, ...]
    expected_samples: int
    expected_label_counts: Mapping[str, int]
    expected_group_counts: Mapping[str, int]
    protocol: str
    loader: Callable[[Path, Path], List[Item]]


def _audio_item(
    path: Path,
    datasets_root: Path,
    label: str,
    speaker: str,
    group: str,
) -> Item:
    """Resolve and minimally decode an audio header before accepting a sample."""

    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = datasets_root.resolve(strict=True)
        relative_path = resolved_path.relative_to(resolved_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"Audio is missing or outside datasets root: {path}") from exc
    if any(character.isspace() for character in relative_path.as_posix()):
        raise ValueError(f"Manifest paths cannot contain whitespace: {relative_path}")
    try:
        info = sf.info(str(resolved_path))
    except Exception as exc:
        raise ValueError(f"Cannot read audio header: {resolved_path}: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Invalid or empty audio: {resolved_path}")
    return Item(
        resolved_path,
        relative_path,
        label,
        speaker,
        group,
        int(info.frames),
        int(info.samplerate),
        int(info.channels),
    )


def _wav_paths(root: Path) -> List[Path]:
    """List every WAV below a corpus, including unexpectedly placed files."""

    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )


def load_emodb(root: Path, datasets_root: Path = None) -> List[Item]:
    datasets_root = datasets_root or root.parent
    items = []
    for path in _wav_paths(root):
        match = EMODB_FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected EMO-DB filename: {path.name}")
        speaker = match.group("speaker")
        items.append(
            _audio_item(
                path,
                datasets_root,
                EMODB_LABELS[match.group("emotion")],
                speaker,
                speaker,
            )
        )
    return items


def load_iemocap(root: Path, datasets_root: Path = None) -> List[Item]:
    datasets_root = datasets_root or root.parent
    metadata = root / "metadata.csv"
    required_fields = {"utterance_id", "emotion", "session", "speaker", "wav"}
    items = []
    with metadata.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"IEMOCAP metadata is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            utterance_id = row["utterance_id"].strip()
            emotion = row["emotion"].strip()
            speaker = row["speaker"].strip()
            try:
                session_number = int(row["session"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid IEMOCAP session at metadata line {line_number}: {row['session']}"
                ) from exc
            if emotion not in IEMOCAP_LABELS:
                raise ValueError(
                    f"Unexpected IEMOCAP emotion at metadata line {line_number}: {emotion}"
                )
            if session_number not in range(1, 6):
                raise ValueError(
                    f"Unexpected IEMOCAP session at metadata line {line_number}: {session_number}"
                )
            expected_prefix = f"Ses{session_number:02d}"
            utterance_match = IEMOCAP_UTTERANCE.fullmatch(utterance_id)
            expected_speaker = None
            if utterance_match is not None:
                expected_speaker = (
                    f"Ses{utterance_match.group('session')}"
                    f"{utterance_match.group('speaker_gender')}"
                )
            if (
                utterance_match is None
                or utterance_match.group("session") != f"{session_number:02d}"
                or speaker != expected_speaker
                or not utterance_id.startswith(expected_prefix)
            ):
                raise ValueError(
                    f"IEMOCAP utterance/speaker/session disagree at metadata line {line_number}"
                )
            group = f"Session{session_number}"
            # The metadata's wav column may contain a stale path from the machine
            # that exported it.  The frozen dataset layout is authoritative.
            path = root / group / f"{utterance_id}.wav"
            items.append(
                _audio_item(path, datasets_root, IEMOCAP_LABELS[emotion], speaker, group)
            )
    referenced_paths = {item.path for item in items}
    discovered_paths = {path.resolve() for path in _wav_paths(root)}
    if referenced_paths != discovered_paths:
        missing = sorted(str(path) for path in referenced_paths - discovered_paths)
        unreferenced = sorted(str(path) for path in discovered_paths - referenced_paths)
        raise ValueError(
            "IEMOCAP metadata/audio inventory mismatch: "
            f"missing={missing[:5]}, unreferenced={unreferenced[:5]}"
        )
    return items


def _load_ravdess(root: Path, datasets_root: Path, song: bool) -> List[Item]:
    expected_channel = "02" if song else "01"
    allowed_emotions = set(RAVDESS_LABELS) - ({"07", "08"} if song else set())
    items = []
    for path in _wav_paths(root):
        match = RAVDESS_FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected RAVDESS filename: {path.name}")
        fields = match.groupdict()
        actor = fields["actor"]
        if fields["modality"] != "03" or fields["channel"] != expected_channel:
            subset = "Song" if song else "Speech"
            raise ValueError(f"File does not belong to RAVDESS-{subset}: {path.name}")
        if fields["emotion"] not in allowed_emotions:
            raise ValueError(f"Unexpected emotion for this RAVDESS subset: {path.name}")
        if path.parent.name != f"Actor_{actor}":
            raise ValueError(f"RAVDESS actor folder and filename disagree: {path}")
        if fields["intensity"] not in {"01", "02"}:
            raise ValueError(f"Unexpected RAVDESS intensity: {path.name}")
        if fields["emotion"] == "01" and fields["intensity"] != "01":
            raise ValueError(f"RAVDESS neutral samples must use normal intensity: {path.name}")
        if fields["statement"] not in {"01", "02"} or fields["repetition"] not in {"01", "02"}:
            raise ValueError(f"Unexpected RAVDESS statement/repetition: {path.name}")
        items.append(
            _audio_item(
                path,
                datasets_root,
                RAVDESS_LABELS[fields["emotion"]],
                actor,
                actor,
            )
        )
    return items


def load_ravdess_speech(root: Path, datasets_root: Path = None) -> List[Item]:
    return _load_ravdess(root, datasets_root or root.parent, song=False)


def load_ravdess_song(root: Path, datasets_root: Path = None) -> List[Item]:
    return _load_ravdess(root, datasets_root or root.parent, song=True)


DATASET_SPECS = {
    "emodb": DatasetSpec(
        slug="emodb",
        directory="EMODB",
        labels=("angry", "boredom", "disgust", "fear", "happy", "sad", "neutral"),
        expected_samples=535,
        expected_label_counts={
            "angry": 127,
            "boredom": 81,
            "disgust": 46,
            "fear": 69,
            "happy": 71,
            "sad": 62,
            "neutral": 79,
        },
        expected_group_counts=dict(
            zip(EMODB_SPEAKERS, (49, 58, 43, 38, 55, 35, 61, 69, 56, 71))
        ),
        protocol="emodb_speaker_independent_10fold_loso_nested_v1",
        loader=load_emodb,
    ),
    "iemocap": DatasetSpec(
        slug="iemocap",
        directory="IEMOCAP-5531",
        labels=("angry", "happy", "neutral", "sad"),
        expected_samples=5531,
        expected_label_counts={
            "angry": 1103,
            "happy": 1636,
            "neutral": 1708,
            "sad": 1084,
        },
        expected_group_counts={
            "Session1": 1085,
            "Session2": 1023,
            "Session3": 1151,
            "Session4": 1031,
            "Session5": 1241,
        },
        protocol="iemocap_session_loso_5fold_cyclic_inner_v1",
        loader=load_iemocap,
    ),
    "ravdess-speech": DatasetSpec(
        slug="ravdess-speech",
        directory="RAVDESS-Speech",
        labels=(
            "neutral",
            "calm",
            "happy",
            "sad",
            "angry",
            "fearful",
            "disgust",
            "surprised",
        ),
        expected_samples=1440,
        expected_label_counts={
            "neutral": 96,
            "calm": 192,
            "happy": 192,
            "sad": 192,
            "angry": 192,
            "fearful": 192,
            "disgust": 192,
            "surprised": 192,
        },
        expected_group_counts={f"{actor:02d}": 60 for actor in range(1, 25)},
        protocol="ravdess_speech_subjectwise_5fold_cyclic_inner_v1",
        loader=load_ravdess_speech,
    ),
    "ravdess-song": DatasetSpec(
        slug="ravdess-song",
        directory="RAVDESS-Song",
        labels=("neutral", "calm", "happy", "sad", "angry", "fearful"),
        expected_samples=1012,
        expected_label_counts={
            "neutral": 92,
            "calm": 184,
            "happy": 184,
            "sad": 184,
            "angry": 184,
            "fearful": 184,
        },
        expected_group_counts={
            f"{actor:02d}": 44 for actor in range(1, 25) if actor != 18
        },
        protocol="ravdess_song_subjectwise_5fold_cyclic_inner_v1",
        loader=load_ravdess_song,
    ),
}


def make_folds(dataset: str) -> List[FoldDefinition]:
    """Return the frozen outer and inner-validation group assignments."""

    if dataset == "emodb":
        return [
            FoldDefinition(index, (outer,), (inner,))
            for index, (outer, inner) in enumerate(
                zip(EMODB_SPEAKERS, EMODB_INNER_VALID), start=1
            )
        ]
    if dataset == "iemocap":
        groups = tuple(f"Session{index}" for index in range(1, 6))
        return [
            FoldDefinition(index, (groups[index - 1],), (groups[index % 5],))
            for index in range(1, 6)
        ]
    if dataset in {"ravdess-speech", "ravdess-song"}:
        return [
            FoldDefinition(
                fold,
                tuple(RAVDESS_FOLDS[fold]),
                tuple(RAVDESS_FOLDS[(fold + 1) % len(RAVDESS_FOLDS)]),
            )
            for fold in sorted(RAVDESS_FOLDS)
        ]
    raise ValueError(f"Unknown dataset: {dataset}")


def audit_source(spec: DatasetSpec, items: Sequence[Item]) -> None:
    """Validate the complete canonical dataset before constructing any folds."""

    relative_paths = [item.relative_path.as_posix() for item in items]
    resolved_paths = [str(item.path) for item in items]
    if len(items) != spec.expected_samples:
        raise ValueError(
            f"{spec.slug}: expected {spec.expected_samples} samples, found {len(items)}"
        )
    if len(set(relative_paths)) != len(items) or len(set(resolved_paths)) != len(items):
        raise ValueError(f"{spec.slug}: duplicate audio paths detected")
    invalid_formats = [
        item.relative_path.as_posix()
        for item in items
        if item.sample_rate != 16000 or item.channels != 1
    ]
    if invalid_formats:
        raise ValueError(
            f"{spec.slug}: all WAV files must be mono 16 kHz; "
            f"invalid={invalid_formats[:5]}"
        )
    label_counts = Counter(item.label for item in items)
    if label_counts != Counter(spec.expected_label_counts):
        raise ValueError(f"{spec.slug}: unexpected label counts: {dict(label_counts)}")
    group_counts = Counter(item.group for item in items)
    if group_counts != Counter(spec.expected_group_counts):
        raise ValueError(f"{spec.slug}: unexpected group counts: {dict(group_counts)}")
    speaker_groups: Dict[str, set] = {}
    for item in items:
        speaker_groups.setdefault(item.speaker, set()).add(item.group)
    ambiguous = {speaker: groups for speaker, groups in speaker_groups.items() if len(groups) != 1}
    if ambiguous:
        raise ValueError(f"{spec.slug}: speakers occur in multiple groups: {ambiguous}")


def split_fold(items: Sequence[Item], fold: FoldDefinition) -> Dict[str, List[Item]]:
    outer_groups = set(fold.outer_valid_groups)
    inner_groups = set(fold.inner_valid_groups)
    if outer_groups & inner_groups:
        raise ValueError(f"Fold {fold.fold}: outer-valid and inner-valid groups overlap")
    train = [item for item in items if item.group not in outer_groups]
    valid = [item for item in items if item.group in outer_groups]
    inner_train = [item for item in train if item.group not in inner_groups]
    inner_valid = [item for item in train if item.group in inner_groups]
    return {
        "train": train,
        "valid": valid,
        "inner_train": inner_train,
        "inner_valid": inner_valid,
    }


def _identities(items: Iterable[Item], attribute: str) -> set:
    if attribute == "path":
        return {item.relative_path.as_posix() for item in items}
    return {getattr(item, attribute) for item in items}


def audit_fold(spec: DatasetSpec, fold: FoldDefinition, splits: Mapping[str, Sequence[Item]]) -> None:
    """Check the two legal training/evaluation partitions for leakage."""

    for split in SPLIT_NAMES:
        if not splits.get(split):
            raise ValueError(f"{spec.slug} fold {fold.fold}: empty {split} split")

    all_labels = set(spec.labels)
    if set(item.label for item in splits["train"]) != all_labels:
        raise ValueError(f"{spec.slug} fold {fold.fold}: outer train loses a class")
    if set(item.label for item in splits["inner_train"]) != all_labels:
        raise ValueError(f"{spec.slug} fold {fold.fold}: inner train loses a class")

    comparisons = (
        ("train", "valid"),
        ("inner_train", "inner_valid"),
        ("inner_train", "valid"),
        ("inner_valid", "valid"),
    )
    for left, right in comparisons:
        for identity in ("path", "speaker", "group"):
            overlap = _identities(splits[left], identity) & _identities(splits[right], identity)
            if overlap:
                raise ValueError(
                    f"{spec.slug} fold {fold.fold}: {identity} leakage between "
                    f"{left} and {right}: {sorted(overlap)}"
                )

    train_paths = _identities(splits["train"], "path")
    valid_paths = _identities(splits["valid"], "path")
    inner_train_paths = _identities(splits["inner_train"], "path")
    inner_valid_paths = _identities(splits["inner_valid"], "path")
    source_paths = {item.relative_path.as_posix() for item in splits["train"] + splits["valid"]}
    if len(source_paths) != spec.expected_samples or train_paths | valid_paths != source_paths:
        raise ValueError(f"{spec.slug} fold {fold.fold}: outer split does not cover the dataset")
    if inner_train_paths | inner_valid_paths != train_paths:
        raise ValueError(f"{spec.slug} fold {fold.fold}: inner split does not reconstruct train")


def audit_fold_coverage(
    spec: DatasetSpec,
    items: Sequence[Item],
    fold_payloads: Sequence[Tuple[FoldDefinition, Mapping[str, Sequence[Item]]]],
) -> None:
    """Ensure every sample receives exactly one outer-valid assignment."""

    expected = Counter(item.relative_path.as_posix() for item in items)
    outer_counts = Counter(
        item.relative_path.as_posix()
        for _, splits in fold_payloads
        for item in splits["valid"]
    )
    if outer_counts != expected:
        missing = sorted(set(expected) - set(outer_counts))
        duplicated = sorted(path for path, count in outer_counts.items() if count != 1)
        raise ValueError(
            f"{spec.slug}: outer-valid coverage is not exactly once; "
            f"missing={missing[:5]}, duplicated={duplicated[:5]}"
        )
    inner_counts = Counter(
        item.relative_path.as_posix()
        for _, splits in fold_payloads
        for item in splits["inner_valid"]
    )
    if inner_counts != expected:
        raise ValueError(f"{spec.slug}: inner-valid coverage is not exactly once")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_sha256(items: Sequence[Item]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value.relative_path.as_posix()):
        record = (
            f"{item.relative_path.as_posix()}\t{item.frames}\t{item.sample_rate}\t"
            f"{item.channels}\t{item.label}\t"
            f"{item.speaker}\t{item.group}\n"
        )
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def write_manifest(
    directory: Path,
    split: str,
    items: Sequence[Item],
    manifest_root: Path,
) -> Tuple[Path, Path]:
    ordered = sorted(items, key=lambda item: item.relative_path.as_posix())
    tsv_path = directory / f"{split}.tsv"
    lbl_path = directory / f"{split}.lbl"
    with tsv_path.open("w", encoding="utf-8", newline="\n") as manifest, lbl_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as labels:
        manifest.write(f"{manifest_root}\n")
        for item in ordered:
            identity = item.relative_path.as_posix()
            if Path(identity).is_absolute():
                raise ValueError(f"Manifest entry must be relative: {identity}")
            manifest.write(f"{identity}\t{item.frames}\n")
            labels.write(f"{identity}\t{item.label}\n")
    return tsv_path, lbl_path


def _split_summary(items: Sequence[Item], role: str) -> dict:
    return {
        "role": role,
        "samples": len(items),
        "groups": sorted({item.group for item in items}),
        "speakers": sorted({item.speaker for item in items}),
        "labels": dict(sorted(Counter(item.label for item in items).items())),
    }


def write_dataset_metadata(path: Path, spec: DatasetSpec, items: Sequence[Item]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "audio_path",
                "frames",
                "sample_rate",
                "channels",
                "label",
                "speaker_id",
                "group_id",
                "dataset",
            ]
        )
        for item in sorted(items, key=lambda value: value.relative_path.as_posix()):
            writer.writerow(
                [
                    item.relative_path.as_posix(),
                    item.frames,
                    item.sample_rate,
                    item.channels,
                    item.label,
                    item.speaker,
                    item.group,
                    spec.slug,
                ]
            )


def write_fold(
    dataset_dir: Path,
    manifest_root: Path,
    spec: DatasetSpec,
    items: Sequence[Item],
    fold: FoldDefinition,
    splits: Mapping[str, Sequence[Item]],
    metadata_path: Path,
) -> None:
    fold_dir = dataset_dir / f"fold_{fold.fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    roles = {
        "train": "outer_train",
        "valid": "sealed_outer_valid",
        "inner_train": "selection_train",
        "inner_valid": "selection_valid",
    }
    summaries = {}
    for split in SPLIT_NAMES:
        tsv_path, lbl_path = write_manifest(fold_dir, split, splits[split], manifest_root)
        summaries[split] = {
            **_split_summary(splits[split], roles[split]),
            "files": {
                "tsv": tsv_path.name,
                "tsv_sha256": sha256_file(tsv_path),
                "lbl": lbl_path.name,
                "lbl_sha256": sha256_file(lbl_path),
            },
        }

    descriptors_path = fold_dir / "label_descriptors.csv"
    with descriptors_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, label in enumerate(spec.labels):
            stream.write(f"{index},{label}\n")

    split_manifest = {
        "schema_version": 1,
        "dataset": spec.slug,
        "fold": fold.fold,
        "protocol": spec.protocol,
        "manifest_root": str(manifest_root),
        "dataset_directory": spec.directory,
        "expected_total_samples": spec.expected_samples,
        "actual_total_samples": len(items),
        "audio_format": {"sample_rate_hz": 16000, "channels": 1},
        "audio_inventory_sha256": inventory_sha256(items),
        "outer_valid_groups": list(fold.outer_valid_groups),
        "inner_valid_groups": list(fold.inner_valid_groups),
        "label_order": list(spec.labels),
        "files": {
            "metadata": str(metadata_path.relative_to(dataset_dir)),
            "metadata_sha256": sha256_file(metadata_path),
            "label_descriptors": descriptors_path.name,
            "label_descriptors_sha256": sha256_file(descriptors_path),
        },
        "splits": summaries,
        "audits": {
            "expected_sample_count": True,
            "unique_paths": True,
            "mono_16khz": True,
            "speaker_disjoint": True,
            "path_disjoint": True,
            "inner_reconstructs_outer_train": True,
            "outer_valid_exactly_once_global": True,
        },
    }
    with (fold_dir / "split.json").open("w", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def build_dataset(
    spec: DatasetSpec,
    datasets_root: Path,
    output_root: Path,
    manifest_root: Path,
) -> None:
    source_root = datasets_root / spec.directory
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {source_root}")
    items = spec.loader(source_root, datasets_root)
    audit_source(spec, items)

    fold_payloads = []
    for fold in make_folds(spec.slug):
        splits = split_fold(items, fold)
        audit_fold(spec, fold, splits)
        fold_payloads.append((fold, splits))
    audit_fold_coverage(spec, items, fold_payloads)

    dataset_dir = output_root / spec.slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_dir / "metadata.csv"
    write_dataset_metadata(metadata_path, spec, items)
    for fold, splits in fold_payloads:
        write_fold(
            dataset_dir,
            manifest_root,
            spec,
            items,
            fold,
            splits,
            metadata_path,
        )
    print(f"{spec.slug}: {len(items)} samples, {len(fold_payloads)} folds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strictly audited nested/fixed SER manifests"
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=SERVER_DATASETS_ROOT,
        help="Root used to read the four dataset directories",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=SERVER_DATASETS_ROOT,
        help="Portable root written on the first line of every TSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SERVER_DATASETS_ROOT / "manifests",
        help="Output directory for per-dataset/per-fold manifests",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASET_SPECS),
        default="all",
        help="Build one dataset or all four",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = DATASET_SPECS if args.dataset == "all" else {args.dataset: DATASET_SPECS[args.dataset]}
    for spec in selected.values():
        build_dataset(spec, args.datasets_root, args.output, args.manifest_root)


if __name__ == "__main__":
    main()
