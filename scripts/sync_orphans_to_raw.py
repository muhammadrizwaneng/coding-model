"""Export main-dataset records that are missing from datasets/raw into a raw batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAIN_FILE = Path("datasets/coding_dataset.jsonl")
RAW_DIR = Path("datasets/raw")
DEFAULT_OUTPUT = RAW_DIR / "curated_extra.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def key_of(record: dict) -> tuple[str, str]:
    return (
        str(record.get("instruction", "")).strip().lower(),
        str(record.get("input", "")).strip().lower(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync orphaned main-dataset rows into raw/.")
    parser.add_argument("--main-file", default=MAIN_FILE, type=Path)
    parser.add_argument("--raw-dir", default=RAW_DIR, type=Path)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()

    if not args.main_file.exists():
        raise FileNotFoundError(args.main_file)

    raw_keys: set[tuple[str, str]] = set()
    for batch in sorted(args.raw_dir.glob("*.jsonl")):
        if batch.resolve() == args.output_file.resolve():
            continue
        for record in load_jsonl(batch):
            raw_keys.add(key_of(record))

    orphans = [record for record in load_jsonl(args.main_file) if key_of(record) not in raw_keys]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as file:
        for record in orphans:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(orphans)} orphaned records to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
