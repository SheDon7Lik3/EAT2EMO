#!/usr/bin/env python3
"""Compute per-fold/per-stage SER fbank normalization statistics.

Statistics are accumulated over valid frames only (after target-length
truncation, before padding).  Audio preprocessing matches the training loader:
mono, 16 kHz, waveform mean subtraction, 128-bin Kaldi fbank, 10 ms shift.
All forms of augmentation are disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Kaldi fbank on short utterances is dramatically faster single-threaded;
# the default 88-thread pool adds more overhead than work per file.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ser_protocol import DATASETS, fold_dir_name  # noqa: E402


STATS_SCHEMA = "eat2emo-ser-fbank-v1"
STAGE_SPLITS = {
    "fixed": "train",
    "selection": "inner_train",
    "final": "train",
}
DEFAULT_DATA_ROOT = os.environ.get(
    "SER_DATA_ROOT", "/root/songhao/datasets/SER_DATASET"
)


def resolve_manifest_root(manifest_path: Path, root_value: str) -> Path:
    root = Path(root_value).expanduser()
    if root.is_absolute():
        return root
    manifest_dir = manifest_path.resolve().parent
    search = manifest_dir
    while True:
        candidate = search / root
        if candidate.is_dir():
            return candidate
        parent = search.parent
        if parent == search:
            raise FileNotFoundError(
                f"could not resolve manifest root {root_value!r} from {manifest_path}"
            )
        search = parent


def read_manifest(manifest_path: Path) -> tuple[Path, list[str]]:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty manifest: {manifest_path}")
    root = resolve_manifest_root(manifest_path, lines[0].strip())
    entries: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"invalid manifest row in {manifest_path}: {line!r}")
        entries.append(fields[0])
    if not entries:
        raise ValueError(f"manifest has no samples: {manifest_path}")
    return root, entries


def compute_stats(
    entries: list[str],
    root: Path,
    target_length: int,
) -> dict[str, float | int]:
    total = 0.0
    total_sq = 0.0
    count = 0
    for relative in entries:
        path = root / relative
        try:
            waveform_array, sample_rate = sf.read(
                path, dtype="float32", always_2d=False
            )
        except Exception as error:
            raise RuntimeError(f"failed to read audio {path}: {error}") from error
        waveform = torch.from_numpy(np.asarray(waveform_array)).float()
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=-1)
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError(f"invalid waveform shape for {path}")
        if not torch.isfinite(waveform).all():
            raise ValueError(f"non-finite waveform values in {path}")
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
        )
        # Only frames that survive the target-length crop are valid training
        # frames.  Padding is deliberately excluded from the statistics.
        features = features[:target_length].to(dtype=torch.float64)
        if features.numel() == 0:
            raise ValueError(f"audio too short for one fbank frame: {path}")
        total += features.sum().item()
        total_sq += features.square().sum().item()
        count += features.numel()

    if count == 0:
        raise ValueError("cannot compute statistics from zero frames")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if not all(math.isfinite(value) for value in (mean, variance, std)):
        raise ValueError(f"invalid statistics: mean={mean}, var={variance}, std={std}")
    if std <= 0:
        raise ValueError("fbank standard deviation must be positive")
    return {
        "count": count,
        "mean": mean,
        "variance": variance,
        "std": std,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-fold SER fbank normalization statistics"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_SPLITS),
        default="fixed",
        help="fixed/final use train; selection uses inner_train",
    )
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def output_path(data_root: Path, dataset: str, stage: str, fold: int) -> Path:
    return data_root / "norm_stats" / dataset / stage / f"{fold_dir_name(fold)}.json"


def main() -> None:
    args = build_parser().parse_args()
    spec = DATASETS[args.dataset]
    allowed_folds = tuple(range(spec.first_fold, spec.last_fold + 1))
    if args.all_folds == (args.fold is not None):
        raise SystemExit("specify exactly one of --fold or --all-folds")
    folds = allowed_folds if args.all_folds else (args.fold,)
    invalid = [fold for fold in folds if fold not in allowed_folds]
    if invalid:
        raise SystemExit(
            f"invalid fold(s) {invalid} for {args.dataset}; expected {allowed_folds[0]}..{allowed_folds[-1]}"
        )
    if args.all_folds and args.output is not None:
        raise SystemExit("--output can only be used with a single --fold")

    data_root = args.data_root.expanduser().resolve()
    split = STAGE_SPLITS[args.stage]
    for fold in folds:
        manifest_dir = (
            data_root / "manifests" / args.dataset / fold_dir_name(fold)
        )
        manifest_path = manifest_dir / f"{split}.tsv"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        root, entries = read_manifest(manifest_path)
        stats = compute_stats(entries, root, spec.target_length)
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        payload = {
            "stats_schema": STATS_SCHEMA,
            "dataset": args.dataset,
            "fold": fold,
            "stage": args.stage,
            "split": split,
            "target_length": spec.target_length,
            "mel_bins": 128,
            "sample_rate": 16000,
            "frame_shift_ms": 10,
            "normalization_scope": "valid_frames",
            "padding_included": False,
            "augmentation": False,
            "samples": len(entries),
            "valid_frames": stats["count"] // 128,
            "mel": {
                "mean": stats["mean"],
                "std": stats["std"],
                "variance": stats["variance"],
                "count": stats["count"],
            },
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
        }
        destination = args.output or output_path(
            data_root, args.dataset, args.stage, fold
        )
        destination = destination.expanduser().resolve()
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"{destination} exists; pass --overwrite to replace it"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"{args.dataset} fold {fold} {args.stage}: "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"var={stats['variance']:.6f} samples={len(entries)} "
            f"valid_frames={stats['count'] // 128}"
        )
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
