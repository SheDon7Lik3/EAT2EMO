#!/usr/bin/env python3
"""Convert IEMOCAP parquet shards into the EAT2EMO IEMOCAP-5531 layout.

Output:

    <output>/metadata.csv
    <output>/SessionN/<utterance_id>.wav

The manifest generator (tools/prepare_ser_manifests.py) requires metadata fields
``utterance_id,emotion,session,speaker,wav`` and audio below ``SessionN/``.
Only the four SER protocol emotions are kept; other emotions are skipped.

By default the tool requires exactly 5531 kept utterances (IEMOCAP-5531).
Pass ``--expected-samples 0`` to disable that check for a schema-only
conversion; the manifest generator will still reject non-protocol counts.
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


EMOTION_MAP = {
    "angry": "ang",
    "happy": "hap",
    "neutral": "neu",
    "sad": "sad",
}
FILENAME = re.compile(
    r"^Ses(?P<session>\d{2})(?P<gender>[FM])_.+_(?P<utterance_gender>[FM])\d{3}$"
)
META_FIELDS = ["utterance_id", "emotion", "session", "speaker", "wav"]


def normalize_filename(value: object) -> str:
    """Return one basename ending in .wav, regardless of the parquet path style."""

    if value is None:
        raise ValueError("IEMOCAP row has no file/name value")
    filename = Path(str(value)).name
    if not filename:
        raise ValueError("IEMOCAP row has an empty filename")
    if not filename.lower().endswith(".wav"):
        filename += ".wav"
    return filename


def identity(filename: str) -> tuple[int, str]:
    """Resolve session and speaker from an IEMOCAP utterance filename.

    Session comes from the ``SesNN`` prefix; speaker comes from the gendered
    ``_F``/``_M`` utterance suffix, which is also what the manifest generator
    validates.
    """

    utterance_id = filename[:-4]
    match = FILENAME.fullmatch(utterance_id)
    if match is None:
        raise ValueError(f"Unexpected IEMOCAP filename: {filename}")
    session = int(match.group("session"))
    if session not in range(1, 6):
        raise ValueError(f"IEMOCAP session must be 1..5: {filename}")
    speaker = f"Ses{session:02d}{match.group('utterance_gender')}"
    return session, speaker


def iter_rows(shards: Iterable[Path]):
    # Keep pyarrow optional at import time so the pure metadata/layout helpers
    # can be tested without the parquet runtime installed.
    import pyarrow.parquet as pq

    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=128):
            yield from batch.to_pylist()


def audit_rows(shards: list[Path]) -> tuple[list[dict], Counter]:
    """Validate and filter metadata before writing any output."""

    records: list[dict] = []
    skipped: Counter = Counter()
    seen_files: set[str] = set()
    seen_ids: set[str] = set()

    for row in iter_rows(shards):
        filename = normalize_filename(row.get("file"))
        if filename in seen_files:
            raise ValueError(f"Duplicate IEMOCAP filename: {filename}")
        seen_files.add(filename)

        session, speaker = identity(filename)
        utterance_id = filename[:-4]
        if utterance_id in seen_ids:
            raise ValueError(f"Duplicate IEMOCAP utterance: {utterance_id}")
        seen_ids.add(utterance_id)

        raw_emotion = str(row.get("major_emotion") or "").strip().lower()
        emotion = EMOTION_MAP.get(raw_emotion)
        if emotion is None:
            skipped[raw_emotion or "<missing>"] += 1
            continue

        records.append(
            {
                "utterance_id": utterance_id,
                "emotion": emotion,
                "session": session,
                "speaker": speaker,
                "wav": f"Session{session}/{filename}",
            }
        )

    return records, skipped


def write_audio(shards: list[Path], records: list[dict], output: Path) -> None:
    wanted = {record["utterance_id"]: record for record in records}
    written: set[str] = set()

    for row in iter_rows(shards):
        filename = normalize_filename(row.get("file"))
        utterance_id = filename[:-4]
        record = wanted.get(utterance_id)
        if record is None:
            continue

        audio = row.get("audio")
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            raise ValueError(f"Missing audio bytes: {filename}")

        destination = output / record["wav"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        written.add(utterance_id)

    missing = sorted(set(wanted) - written)
    if missing:
        raise ValueError(f"Audio missing for {len(missing)} utterances: {missing[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory containing *.parquet shards")
    parser.add_argument("--output", type=Path, required=True, help="IEMOCAP-5531 output directory")
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=5531,
        help="require this many kept utterances; 0 disables the check (default: 5531)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into an existing non-empty output directory",
    )
    args = parser.parse_args()

    if args.expected_samples < 0:
        parser.error("--expected-samples must be >= 0")

    shards = sorted(args.source.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet files found in {args.source}")

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {args.output}; pass --overwrite to replace files"
        )

    records, skipped = audit_rows(shards)
    if args.expected_samples and len(records) != args.expected_samples:
        skipped_text = ", ".join(f"{name}={count}" for name, count in sorted(skipped.items()))
        raise SystemExit(
            f"Kept {len(records)} utterances, expected {args.expected_samples}. "
            f"Skipped emotions: {skipped_text or 'none'}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_audio(shards, records, args.output)

    metadata_path = args.output / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=META_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda value: value["wav"]):
            writer.writerow({field: record[field] for field in META_FIELDS})

    skipped_total = sum(skipped.values())
    print(f"Kept {len(records)} utterances; skipped {skipped_total} non-protocol rows")
    if skipped:
        print("Skipped emotions:", ", ".join(f"{name}={count}" for name, count in sorted(skipped.items())))
    print(f"Metadata: {metadata_path}")
    print(f"Audio root: {args.output}")


if __name__ == "__main__":
    main()
