#!/usr/bin/env python3
"""Evaluate a fine-tuned EAT SER checkpoint and save auditable predictions.

The metric and artifact helpers intentionally do not import Fairseq or PyTorch so
they can be tested on machines that do not have the training runtime installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn import metrics as sklearn_metrics


AUDIOSET_NORM_MEAN = -4.268
AUDIOSET_NORM_STD = 4.569
NORM_STATS_SCHEMA = "eat2emo-ser-fbank-v1"
DATASET_ALIASES = {
    "emodb": "emodb",
    "emo-db": "emodb",
    "iemocap": "iemocap",
    "iemocap-5531": "iemocap",
    "ravdess": "ravdess-speech",
    "ravdess-speech": "ravdess-speech",
    "ravdess-song": "ravdess-song",
}
EXPECTED_CLASSES = {
    "emodb": 7,
    "iemocap": 4,
    "ravdess-speech": 8,
    "ravdess-song": 6,
}
EXPECTED_FOLDS = {
    "emodb": tuple(range(1, 11)),
    "iemocap": tuple(range(1, 6)),
    "ravdess-speech": tuple(range(5)),
    "ravdess-song": tuple(range(5)),
}
DEFAULT_TARGET_LENGTHS = {
    "emodb": 640,
    "iemocap": 1024,
    "ravdess-speech": 512,
    "ravdess-song": 640,
}
ARTIFACT_FILENAMES = (
    "metrics.json",
    "predictions.npz",
    "paths.txt",
    "confusion_matrix.csv",
)


@dataclass(frozen=True)
class UserDirModule:
    user_dir: str


@dataclass(frozen=True)
class ManifestData:
    paths: tuple[Path, ...]
    identities: tuple[str, ...]
    targets: np.ndarray
    label_names: tuple[str, ...]


def canonical_dataset(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    try:
        return DATASET_ALIASES[key]
    except KeyError as error:
        choices = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise argparse.ArgumentTypeError(
            f"unsupported dataset {value!r}; choose one of: {choices}"
        ) from error


def parse_fold(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("fold_"):
        normalized = normalized[5:]
    elif normalized.startswith("fold"):
        normalized = normalized[4:]
    try:
        fold = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid fold: {value!r}") from error
    if fold < 0:
        raise argparse.ArgumentTypeError("fold must be non-negative")
    return fold


def _read_label_descriptors(path: Path) -> tuple[dict[str, int], tuple[str, ...]]:
    label_to_index: dict[str, int] = {}
    index_to_label: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row_number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue
            if row_number == 1 and row[0].strip().lower() == "index":
                continue
            if len(row) < 2:
                raise ValueError(f"invalid label descriptor row {row_number}: {path}")
            try:
                index = int(row[0])
            except ValueError as error:
                raise ValueError(
                    f"invalid class index on row {row_number}: {path}"
                ) from error
            label = row[1].strip()
            if not label or label in label_to_index or index in index_to_label:
                raise ValueError(f"duplicate or empty label on row {row_number}: {path}")
            label_to_index[label] = index
            index_to_label[index] = label
    expected = list(range(len(index_to_label)))
    if sorted(index_to_label) != expected:
        raise ValueError(f"class indices must be contiguous from zero: {path}")
    return label_to_index, tuple(index_to_label[index] for index in expected)


def read_manifest(
    data_dir: Path,
    split: str = "valid",
    audio_root: Path | None = None,
    *,
    verify_files: bool = True,
) -> ManifestData:
    """Read a Fairseq TSV/LBL pair without changing its saved sample order."""

    manifest_path = data_dir / f"{split}.tsv"
    label_path = data_dir / f"{split}.lbl"
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(manifest_lines) < 2:
        raise ValueError(f"empty manifest: {manifest_path}")
    if not manifest_lines[0].strip():
        raise ValueError(f"manifest has an empty audio root: {manifest_path}")
    root = (audio_root or Path(manifest_lines[0])).expanduser()
    paths: list[Path] = []
    identities: list[str] = []
    for row_number, line in enumerate(manifest_lines[1:], 2):
        columns = line.split("\t")
        if len(columns) != 2 or not columns[1].isdigit() or int(columns[1]) <= 0:
            raise ValueError(f"invalid manifest row {row_number}: {manifest_path}")
        identity = columns[0].strip()
        if not identity or "\n" in identity or "\r" in identity:
            raise ValueError(f"invalid audio path on row {row_number}: {manifest_path}")
        path = Path(identity)
        if path.is_absolute():
            raise ValueError(
                f"audio path must be relative on row {row_number}: {manifest_path}"
            )
        paths.append(root / path)
        identities.append(path.as_posix())

    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate audio path in manifest: {manifest_path}")
    if verify_files:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} manifest audio file(s) are missing; first: {preview}"
            )

    label_to_index, label_names = _read_label_descriptors(
        data_dir / "label_descriptors.csv"
    )
    targets: list[int] = []
    with label_path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, 1):
            columns = line.rstrip("\r\n").split("\t")
            if len(columns) != 2:
                raise ValueError(f"invalid label row {row_number}: {label_path}")
            identity, label = (column.strip() for column in columns)
            if row_number > len(identities) or identity != identities[row_number - 1]:
                expected = (
                    identities[row_number - 1]
                    if row_number <= len(identities)
                    else "<no corresponding manifest row>"
                )
                raise ValueError(
                    f"TSV/LBL identity mismatch on row {row_number}: "
                    f"expected {expected!r}, found {identity!r}"
                )
            if label not in label_to_index:
                raise ValueError(
                    f"unknown label {label!r} on row {row_number}: {label_path}"
                )
            targets.append(label_to_index[label])
    if len(paths) != len(targets):
        raise ValueError(
            f"manifest has {len(paths)} files but {len(targets)} labels for {split}"
        )
    return ManifestData(
        paths=tuple(paths),
        identities=tuple(identities),
        targets=np.asarray(targets, dtype=np.int64),
        label_names=label_names,
    )


def validate_prediction_arrays(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty [samples, classes] matrix")
    if num_classes is not None and probabilities.shape[1] != num_classes:
        raise ValueError(
            f"expected {num_classes} classes, got {probabilities.shape[1]}"
        )
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must be one-dimensional and align with probabilities")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError("probabilities are outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows do not sum to one")
    if not np.issubdtype(targets.dtype, np.integer):
        if not np.all(np.equal(targets, np.floor(targets))):
            raise ValueError("targets must contain integer class indices")
    targets = targets.astype(np.int64, copy=False)
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("target class index is out of range")
    return probabilities, targets


def compute_ser_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    loss: float | None = None,
    average_over_present_classes: bool = True,
) -> tuple[dict[str, object], np.ndarray]:
    """Compute single-label SER metrics with explicit missing-class behavior."""

    probabilities, targets = validate_prediction_arrays(probabilities, targets)
    predictions = probabilities.argmax(axis=1)
    all_classes = np.arange(probabilities.shape[1])
    present_classes = np.unique(targets)
    averaging_classes = present_classes if average_over_present_classes else all_classes
    accuracy = float(sklearn_metrics.accuracy_score(targets, predictions))
    if loss is None:
        chosen = probabilities[np.arange(len(targets)), targets]
        loss = float(-np.log(np.clip(chosen, np.finfo(np.float64).tiny, 1.0)).mean())
    if not isinstance(loss, (int, float, np.integer, np.floating)) or not math.isfinite(
        float(loss)
    ):
        raise ValueError("loss must be finite")
    result: dict[str, object] = {
        "samples": int(len(targets)),
        "accuracy": accuracy,
        "war": accuracy,
        "uar": float(
            sklearn_metrics.recall_score(
                targets,
                predictions,
                labels=averaging_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            sklearn_metrics.f1_score(
                targets,
                predictions,
                labels=averaging_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "loss": float(loss),
        "present_classes": [int(value) for value in present_classes],
        "metric_average_classes": [int(value) for value in averaging_classes],
    }
    confusion = sklearn_metrics.confusion_matrix(
        targets, predictions, labels=all_classes
    )
    return result, confusion


def save_evaluation_artifacts(
    output: Path,
    probabilities: np.ndarray,
    targets: np.ndarray,
    paths: Sequence[str],
    label_names: Sequence[str],
    metadata: dict[str, object],
    *,
    loss: float | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    probabilities, targets = validate_prediction_arrays(
        probabilities, targets, num_classes=len(label_names)
    )
    normalized_paths = [str(path) for path in paths]
    if len(normalized_paths) != len(targets):
        raise ValueError("paths must align one-to-one with predictions")
    if any(not path or "\n" in path or "\r" in path for path in normalized_paths):
        raise ValueError("paths must be non-empty single-line strings")
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("paths contain duplicate samples")
    metrics_payload, confusion = compute_ser_metrics(
        probabilities, targets, loss=loss, average_over_present_classes=True
    )
    metrics_payload = {
        **metadata,
        **metrics_payload,
        "num_classes": len(label_names),
        "labels": list(label_names),
        "artifacts": {
            "predictions": "predictions.npz",
            "paths": "paths.txt",
            "confusion_matrix": "confusion_matrix.csv",
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in ARTIFACT_FILENAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing evaluation artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    np.savez_compressed(
        output / "predictions.npz",
        probabilities=probabilities.astype(np.float32),
        targets=targets,
        label_names=np.asarray(label_names, dtype=np.str_),
    )
    (output / "paths.txt").write_text(
        "\n".join(normalized_paths) + "\n", encoding="utf-8"
    )
    with (output / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/pred", *label_names])
        for label, row in zip(label_names, confusion):
            writer.writerow([label, *[int(value) for value in row]])
    # Write metrics last: its presence means the other fold artifacts were saved.
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics_payload


class SerDataset:
    """Minimal map-style dataset; heavy audio/torch imports stay runtime-only."""

    def __init__(
        self,
        paths: Sequence[Path],
        targets: Iterable[int],
        target_length: int,
        norm_mean: float,
        norm_std: float,
        norm_mode: str = "audioset",
    ) -> None:
        self.paths = tuple(paths)
        self.targets = tuple(int(target) for target in targets)
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.norm_mode = norm_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import soundfile as sf
        import torch
        import torch.nn.functional as torch_functional
        import torchaudio

        # No fallback or sample substitution: decode failures propagate immediately.
        waveform_array, sample_rate = sf.read(
            self.paths[index], dtype="float32", always_2d=False
        )
        waveform = torch.from_numpy(np.asarray(waveform_array))
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError(f"invalid waveform shape for {self.paths[index]}")
        if not torch.isfinite(waveform).all():
            raise ValueError(f"non-finite waveform values in {self.paths[index]}")
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError(f"invalid sample rate for {self.paths[index]}: {sample_rate}")
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform - waveform.mean()
        features = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        ).unsqueeze(0)
        if self.norm_mode == "fold":
            valid_frames = min(features.shape[1], self.target_length)
            features = features[:, :valid_frames, :]
            features = (features - self.norm_mean) / (self.norm_std * 2)
            if valid_frames < self.target_length:
                features = torch_functional.pad(
                    features, (0, 0, 0, self.target_length - valid_frames)
                )
        else:
            difference = self.target_length - features.shape[1]
            if difference > 0:
                features = torch_functional.pad(features, (0, 0, 0, difference))
            elif difference < 0:
                features = features[:, : self.target_length, :]
            features = (features - self.norm_mean) / (self.norm_std * 2)
        return features, self.targets[index]


def run_inference(args: argparse.Namespace, manifest: ManifestData):
    try:
        import fairseq
        import torch
        import torch.nn.functional as torch_functional
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError(
            "evaluation requires Fairseq, PyTorch, torchaudio, and soundfile"
        ) from error

    fairseq.utils.import_user_module(UserDirModule(str(args.model_dir.resolve())))
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(args.checkpoint)]
    )
    if len(models) != 1:
        raise ValueError(f"expected one checkpoint model, loaded {len(models)}")
    model = models[0].eval().to(args.device)
    loader = DataLoader(
        SerDataset(
            manifest.paths,
            manifest.targets,
            args.target_length,
            args.norm_mean,
            args.norm_std,
            norm_mode=args.norm_mode,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    probability_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    total_loss = 0.0
    with torch.inference_mode():
        for features, targets in loader:
            device_targets = targets.to(args.device)
            logits = model(features.to(args.device))
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise ValueError("model must return a [batch, classes] logits tensor")
            if logits.shape[1] != len(manifest.label_names):
                raise ValueError(
                    f"checkpoint has {logits.shape[1]} outputs but manifest has "
                    f"{len(manifest.label_names)} classes"
                )
            total_loss += float(
                torch_functional.cross_entropy(
                    logits.float(), device_targets, reduction="sum"
                ).item()
            )
            probability_batches.append(
                torch.softmax(logits.float(), dim=-1).cpu().numpy()
            )
            target_batches.append(targets.numpy())
    probabilities = np.concatenate(probability_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return probabilities, targets, total_loss / len(targets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned EAT SER checkpoint"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=canonical_dataset, required=True)
    parser.add_argument("--protocol", choices=("earlystop", "fixed"), required=True)
    parser.add_argument("--fold", type=parse_fold, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selected-epoch", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--fixed-epochs", type=int)
    parser.add_argument("--config", default="eat_base_sr16k_audiosetnorm_bs32")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--target-length", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--norm-mode", choices=("audioset", "fold"), default="audioset")
    parser.add_argument("--norm-stats-path", type=Path)
    parser.add_argument("--norm-mean", type=float, default=AUDIOSET_NORM_MEAN)
    parser.add_argument("--norm-std", type=float, default=AUDIOSET_NORM_STD)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.fold not in EXPECTED_FOLDS[args.dataset]:
        allowed = EXPECTED_FOLDS[args.dataset]
        parser.error(
            f"--fold for {args.dataset} must be {allowed[0]}..{allowed[-1]}"
        )
    if args.protocol == "earlystop":
        if args.selected_epoch is None or args.max_epochs is None:
            parser.error("earlystop evaluation requires --selected-epoch and --max-epochs")
        if args.fixed_epochs is not None:
            parser.error("earlystop evaluation must not set --fixed-epochs")
        if args.selected_epoch <= 0 or args.max_epochs <= 0:
            parser.error("--selected-epoch and --max-epochs must be positive")
        if args.selected_epoch > args.max_epochs:
            parser.error("--selected-epoch must not exceed --max-epochs")
    else:
        if args.fixed_epochs is None:
            parser.error("fixed evaluation requires --fixed-epochs")
        if args.selected_epoch is not None or args.max_epochs is not None:
            parser.error("fixed evaluation must not set --selected-epoch/--max-epochs")
        if args.fixed_epochs <= 0:
            parser.error("--fixed-epochs must be positive")
    if args.target_length is None:
        args.target_length = DEFAULT_TARGET_LENGTHS[args.dataset]
    if args.target_length <= 0:
        parser.error("--target-length must be positive")
    args.norm_stats_sha256 = ""
    if args.norm_mode == "fold":
        if args.norm_stats_path is None:
            parser.error("--norm-stats-path is required with --norm-mode fold")
        stats_path = args.norm_stats_path.expanduser().resolve()
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"could not read --norm-stats-path: {exc}")
        expected = {
            "stats_schema": NORM_STATS_SCHEMA,
            "target_length": args.target_length,
            "mel_bins": 128,
            "sample_rate": 16000,
            "normalization_scope": "valid_frames",
            "padding_included": False,
        }
        mismatches = {
            key: (value, stats.get(key))
            for key, value in expected.items()
            if stats.get(key) != value
        }
        if mismatches:
            parser.error(f"invalid norm stats metadata: {mismatches}")
        stats_split = stats.get("split")
        if stats_split not in ("train", "inner_train"):
            parser.error(f"unexpected norm stats split: {stats_split!r}")
        stats_manifest = args.data / f"{stats_split}.tsv"
        manifest_sha256 = hashlib.sha256(stats_manifest.read_bytes()).hexdigest()
        if stats.get("manifest_sha256") != manifest_sha256:
            parser.error("norm stats do not match their training manifest")
        mel = stats.get("mel")
        if not isinstance(mel, dict):
            parser.error("norm stats missing mel object")
        args.norm_mean = float(mel["mean"])
        args.norm_std = float(mel["std"])
        args.norm_stats_path = stats_path
        args.norm_stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    else:
        if not math.isclose(args.norm_mean, AUDIOSET_NORM_MEAN) or not math.isclose(
            args.norm_std, AUDIOSET_NORM_STD
        ):
            parser.error("SER evaluation is frozen to the EAT AudioSet normalization")
    if args.device is None:
        try:
            import torch
        except ImportError as error:
            raise SystemExit("PyTorch is required for checkpoint evaluation") from error
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = read_manifest(args.data, args.split, args.audio_root)
    expected_classes = EXPECTED_CLASSES[args.dataset]
    if len(manifest.label_names) != expected_classes:
        raise ValueError(
            f"{args.dataset} requires {expected_classes} classes, manifest has "
            f"{len(manifest.label_names)}"
        )
    probabilities, targets, loss = run_inference(args, manifest)
    metadata = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "config": args.config,
        "fold": args.fold,
        "seed": args.seed,
        "split": args.split,
        "selected_epoch": args.selected_epoch,
        "max_epochs": args.max_epochs,
        "fixed_epochs": args.fixed_epochs,
        "trained_epochs": (
            args.selected_epoch if args.protocol == "earlystop" else args.fixed_epochs
        ),
        "target_length": args.target_length,
        "norm_mode": args.norm_mode,
        "norm_mean": args.norm_mean,
        "norm_std": args.norm_std,
        "norm_stats_path": str(args.norm_stats_path) if args.norm_stats_path else None,
        "norm_stats_sha256": args.norm_stats_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_role": (
            "earlystop_best_inner_valid"
            if args.protocol == "earlystop"
            else "fixed_last_checkpoint"
        ),
        "used_for_model_selection": args.protocol == "earlystop",
    }
    result = save_evaluation_artifacts(
        args.output,
        probabilities,
        targets,
        manifest.identities,
        manifest.label_names,
        metadata,
        loss=loss,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
