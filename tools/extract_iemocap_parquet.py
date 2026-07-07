#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import pyarrow.parquet as pq


META_FIELDS = [
    "file", "audio_path", "session", "speaker_id", "gender", "major_emotion",
    "transcription", "EmoAct", "EmoVal", "EmoDom", "frustrated", "angry",
    "sad", "disgust", "excited", "fear", "neutral", "surprise", "happy",
]


def identity(filename: str):
    match = re.match(r"Ses(\d{2})([FM])_", filename)
    if not match:
        raise ValueError(f"Unexpected IEMOCAP filename: {filename}")
    session, gender = match.groups()
    return f"Session{session}", f"Ses{session}{gender}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shards = sorted(args.source.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet files found in {args.source}")

    wav_dir = args.output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.csv"
    seen = set()
    count = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=META_FIELDS)
        writer.writeheader()
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    filename = Path(row["file"]).name
                    if filename in seen:
                        raise ValueError(f"Duplicate filename: {filename}")
                    seen.add(filename)
                    session, speaker_id = identity(filename)
                    audio = row["audio"]["bytes"]
                    if not audio:
                        raise ValueError(f"Missing audio bytes: {filename}")
                    destination = wav_dir / filename
                    destination.write_bytes(audio)
                    record = {key: row.get(key) for key in META_FIELDS}
                    record.update(
                        file=filename,
                        audio_path=str(destination.resolve()),
                        session=session,
                        speaker_id=speaker_id,
                    )
                    writer.writerow(record)
                    count += 1

    print(f"Extracted {count} WAV files")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
