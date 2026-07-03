import argparse
import json
from pathlib import Path

RAW_DIR = Path("datasets/raw")
DEFAULT_OUTPUT = Path("datasets/coding_dataset.jsonl")


def load_jsonl(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            records.append(record)
    return records


def dedupe_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []

    for record in records:
        key = (
            str(record.get("instruction", "")).strip().lower(),
            str(record.get("input", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    return unique


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge raw JSONL batch files into the main dataset.")
    parser.add_argument("--raw-dir", default=RAW_DIR, type=Path)
    parser.add_argument("--base-file", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Include existing records from the base dataset before adding raw batches.",
    )
    args = parser.parse_args()

    records: list[dict[str, str]] = []
    if args.include_base and args.base_file.exists():
        records.extend(load_jsonl(args.base_file))

    batch_files = sorted(args.raw_dir.glob("*.jsonl"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {args.raw_dir}")

    for batch_file in batch_files:
        records.extend(load_jsonl(batch_file))

    merged = dedupe_records(records)
    write_jsonl(args.output_file, merged)

    print(f"Merged {len(batch_files)} batch files into {args.output_file}")
    print(f"Total unique records: {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
