#!/usr/bin/env python3
"""Build the final training JSONL from curated raw/ + imported HF datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW_DIR = Path("datasets/raw")
HF_DIR = Path("datasets/hf")
SEED_FILE = Path("datasets/seed.jsonl")
DEFAULT_OUTPUT = Path("datasets/coding_dataset.jsonl")
DEFAULT_MANIFEST = Path("datasets/coding_dataset.manifest.json")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
    return records


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for record in records:
        key = (
            str(record.get("instruction", "")).strip().lower(),
            str(record.get("input", "")).strip().lower(),
        )
        if not key[0]:
            continue
        if key in seen:
            continue
        seen.add(key)
        # Keep training schema clean for the trainer.
        clean = {
            "instruction": str(record.get("instruction", "")).strip(),
            "input": str(record.get("input", "") or "").strip(),
            "output": str(record.get("output", "")).strip(),
        }
        if record.get("source"):
            clean["source"] = str(record["source"])
        if clean["instruction"] and clean["output"]:
            unique.append(clean)
    return unique


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            # Trainer only needs instruction/input/output
            payload = {
                "instruction": record["instruction"],
                "input": record.get("input", ""),
                "output": record["output"],
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def collect_files(raw_dir: Path, hf_dir: Path, include_seed: bool) -> list[Path]:
    files: list[Path] = []
    if include_seed and SEED_FILE.exists():
        files.append(SEED_FILE)
    if raw_dir.exists():
        files.extend(sorted(raw_dir.glob("*.jsonl")))
    if hf_dir.exists():
        files.extend(sorted(hf_dir.glob("*.jsonl")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge curated + Hugging Face coding datasets into one training JSONL."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--hf-dir", type=Path, default=HF_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--include-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-hf",
        action="store_true",
        help="Fail if datasets/hf has no imported JSONL files.",
    )
    parser.add_argument(
        "--curated-upsample",
        type=int,
        default=3,
        help="Repeat curated/seed rows this many times so specialty data is not drowned out (default: 3).",
    )
    args = parser.parse_args()

    files = collect_files(args.raw_dir, args.hf_dir, include_seed=args.include_seed)
    hf_files = [path for path in files if is_under(path, args.hf_dir)]
    curated_files = [path for path in files if path not in hf_files]

    if args.require_hf and not hf_files:
        raise SystemExit(
            f"No HF imports found in {args.hf_dir}. "
            "Run: python scripts/import_hf_datasets.py --preset colab"
        )

    curated_records: list[dict] = []
    for path in curated_files:
        curated_records.extend(load_jsonl(path))

    hf_records: list[dict] = []
    for path in hf_files:
        hf_records.extend(load_jsonl(path))

    # Upsample curated specialty examples so LoRA still learns your stack.
    weighted_curated: list[dict] = []
    for _ in range(max(1, args.curated_upsample)):
        weighted_curated.extend(curated_records)

    merged = dedupe(weighted_curated + hf_records)
    write_jsonl(args.output_file, merged)

    by_source: dict[str, int] = {}
    for record in merged:
        # source was stripped for trainer file; recount from pre-clean if present in inputs
        pass

    # Recount sources from pre-dedupe inputs for the manifest
    source_counts: dict[str, int] = {"curated": 0}
    for record in curated_records:
        source_counts["curated"] += 1
    for record in hf_records:
        source = str(record.get("source") or "hf_unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    manifest = {
        "output_file": str(args.output_file),
        "total_unique_training_rows": len(merged),
        "curated_rows": len(curated_records),
        "curated_upsample": args.curated_upsample,
        "hf_rows": len(hf_records),
        "source_counts_before_dedupe": source_counts,
        "files": [str(path) for path in files],
    }
    args.manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Curated rows: {len(curated_records)} (x{args.curated_upsample} before dedupe)")
    print(f"HF rows:      {len(hf_records)}")
    print(f"Training rows (unique): {len(merged)}")
    print(f"Wrote {args.output_file}")
    print(f"Wrote {args.manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
