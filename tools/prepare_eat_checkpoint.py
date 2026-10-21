#!/usr/bin/env python3
"""Merge an EAT fine-tuned encoder into a pretrained EAT checkpoint.

Official EAT fine-tuned checkpoints store a classifier wrapper:

    model.<encoder keys>
    fc_norm.weight/bias
    head.weight/bias

SER fine-tuning expects the pretraining checkpoint format.  This tool strips the
wrapper and classifier keys, then merges the encoder tensors into a copy of the
pretrained checkpoint.  Pretraining-only keys (e.g. decoder) are kept from the
base checkpoint.  The source files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


MODEL_PREFIX = "model."
WRAPPER_PREFIXES = (
    "head.",
    "model.head.",
    "classification_head.",
    "model.classification_head.",
    "fc_norm.",
    "model.fc_norm.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_wrapper_key(key: str) -> bool:
    return key.startswith(WRAPPER_PREFIXES)


def extract_encoder(model_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("checkpoint['model'] is missing or empty")

    removed_wrapper = sorted(key for key in model_state if is_wrapper_key(key))
    encoder = {key: value for key, value in model_state.items() if not is_wrapper_key(key)}

    stripped_prefix = False
    prefixed = [key for key in encoder if key.startswith(MODEL_PREFIX)]
    if prefixed:
        if len(prefixed) != len(encoder):
            raise ValueError(
                "checkpoint has a mix of 'model.' and non-'model.' keys; cannot convert safely"
            )
        encoder = {key[len(MODEL_PREFIX):]: value for key, value in encoder.items()}
        stripped_prefix = True

    remaining = sorted(key for key in encoder if is_wrapper_key(key))
    if remaining:
        raise ValueError(f"wrapper keys remain after extraction: {remaining[:5]}")
    if not encoder:
        raise ValueError("conversion removed every encoder tensor")

    return encoder, {
        "source_model_keys": len(model_state),
        "encoder_keys": len(encoder),
        "removed_wrapper_keys": removed_wrapper,
        "stripped_model_prefix": stripped_prefix,
    }


def merge_encoder(base_state: dict, encoder: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    if not isinstance(base_state, dict) or "model" not in base_state:
        raise ValueError("base checkpoint has no 'model' state")
    base_model = base_state["model"]
    if not isinstance(base_model, dict) or not base_model:
        raise ValueError("base checkpoint['model'] is missing or empty")

    missing = sorted(set(encoder) - set(base_model))
    if missing:
        raise ValueError(
            f"encoder contains {len(missing)} keys not present in base checkpoint: {missing[:5]}"
        )

    merged_model = dict(base_model)
    merged_model.update(encoder)
    merged_state = dict(base_state)
    merged_state["model"] = merged_model
    base_only = sorted(set(base_model) - set(encoder))
    return merged_state, {
        "base_model_keys": len(base_model),
        "merged_encoder_keys": len(encoder),
        "base_only_keys": len(base_only),
        "base_only_sample": base_only[:10],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge an EAT fine-tuned encoder into a pretrained checkpoint"
    )
    parser.add_argument("--source", type=Path, required=True, help="fine-tuned classifier checkpoint")
    parser.add_argument("--base", type=Path, required=True, help="pretrained EAT checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = args.source.expanduser().resolve()
    base = args.base.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output in (source, base):
        raise SystemExit("--output must differ from --source and --base")
    if not source.is_file():
        raise SystemExit(f"source checkpoint not found: {source}")
    if not base.is_file():
        raise SystemExit(f"base checkpoint not found: {base}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")

    source_state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(source_state, dict) or "model" not in source_state:
        raise SystemExit("source checkpoint has no 'model' state")
    encoder, metadata = extract_encoder(source_state["model"])

    base_state = torch.load(base, map_location="cpu", weights_only=False)
    merged, merge_metadata = merge_encoder(base_state, encoder)
    metadata.update(merge_metadata)
    metadata.update(
        {
            "source": str(source),
            "base": str(base),
            "source_sha256": sha256_file(source),
            "base_sha256": sha256_file(base),
        }
    )

    print(json.dumps(metadata, indent=2))
    if args.dry_run:
        print("dry-run: no file written")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(merged, temporary)
    os.replace(temporary, output)
    metadata["output"] = str(output)
    metadata["output_sha256"] = sha256_file(output)
    metadata_path = output.with_name(output.name + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved merged checkpoint: {output}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
