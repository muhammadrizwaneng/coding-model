#!/usr/bin/env python3
"""Download public coding SFT datasets from Hugging Face and convert to Alpaca JSONL.

Default mode uses practical sample caps for Colab. Use --preset full carefully:
OpenCodeInstruct alone is ~5M rows and can take many hours/GPUs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

HF_DIR = Path("datasets/hf")
DEFAULT_SEED = 42
MIN_OUTPUT_CHARS = 40

PRESETS: dict[str, dict[str, int | None]] = {
    # Practical Colab mix (~100k-150k before dedupe)
    "colab": {
        "magicoder_oss": 25000,
        "magicoder_evol": 25000,
        "codealpaca": 20000,
        "opencoder_s1_realuser": 15000,
        "opencoder_s2": 10000,
        "opencoder_s2_evol": 10000,
        "opencodeinstruct": 20000,
    },
    # Larger GPU host mix
    "large": {
        "magicoder_oss": None,  # full ~75k
        "magicoder_evol": None,  # full ~110k
        "codealpaca": None,  # full ~20k
        "opencoder_s1_realuser": 50000,
        "opencoder_s2": 50000,
        "opencoder_s2_evol": 50000,
        "opencoder_s2_package": 30000,
        "opencodeinstruct": 100000,
    },
    # As complete as practical; still samples OpenCodeInstruct unless overridden
    "full": {
        "magicoder_oss": None,
        "magicoder_evol": None,
        "codealpaca": None,
        "opencoder_s1_realuser": None,
        "opencoder_s1_diverse": 100000,
        "opencoder_s2": None,
        "opencoder_s2_evol": None,
        "opencoder_s2_package": None,
        "opencodeinstruct": 500000,
    },
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _pick(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and _text(row[key]):
            return _text(row[key])
    return ""


def normalize_record(
    instruction: str,
    output: str,
    input_text: str = "",
    source: str = "",
) -> dict[str, str] | None:
    instruction = instruction.strip()
    output = output.strip()
    input_text = input_text.strip()
    if not instruction or not output:
        return None
    if len(output) < MIN_OUTPUT_CHARS:
        return None
    record = {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }
    if source:
        record["source"] = source
    return record


def map_magicoder_oss(row: dict[str, Any], source: str) -> dict[str, str] | None:
    return normalize_record(
        instruction=_pick(row, "problem", "instruction", "query", "prompt"),
        output=_pick(row, "solution", "response", "output", "answer"),
        input_text=_pick(row, "input", "context"),
        source=source,
    )


def map_magicoder_evol(row: dict[str, Any], source: str) -> dict[str, str] | None:
    return normalize_record(
        instruction=_pick(row, "instruction", "problem", "query", "prompt"),
        output=_pick(row, "response", "output", "solution", "answer"),
        input_text=_pick(row, "input", "context"),
        source=source,
    )


def map_codealpaca(row: dict[str, Any], source: str) -> dict[str, str] | None:
    return normalize_record(
        instruction=_pick(row, "instruction", "problem"),
        output=_pick(row, "output", "response", "solution"),
        input_text=_pick(row, "input", "context"),
        source=source,
    )


def map_opencoder(row: dict[str, Any], source: str) -> dict[str, str] | None:
    return normalize_record(
        instruction=_pick(row, "instruction", "problem", "query", "prompt", "input"),
        output=_pick(row, "output", "response", "solution", "code"),
        input_text=_pick(row, "input", "context") if "instruction" in row else "",
        source=source,
    )


def map_opencodeinstruct(row: dict[str, Any], source: str) -> dict[str, str] | None:
    # Prefer rows that pass at least some unit tests when available.
    score = row.get("average_test_score")
    try:
        if score is not None and float(score) < 0.5:
            return None
    except (TypeError, ValueError):
        pass
    return normalize_record(
        instruction=_pick(row, "input", "instruction", "problem", "prompt"),
        output=_pick(row, "output", "response", "solution"),
        input_text="",
        source=source,
    )


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    config: str | None
    split: str
    mapper: Callable[[dict[str, Any], str], dict[str, str] | None]
    streaming_recommended: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "magicoder_oss": DatasetSpec(
        key="magicoder_oss",
        hf_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        config=None,
        split="train",
        mapper=map_magicoder_oss,
    ),
    "magicoder_evol": DatasetSpec(
        key="magicoder_evol",
        hf_id="ise-uiuc/Magicoder-Evol-Instruct-110K",
        config=None,
        split="train",
        mapper=map_magicoder_evol,
    ),
    "codealpaca": DatasetSpec(
        key="codealpaca",
        hf_id="sahil2801/CodeAlpaca-20k",
        config=None,
        split="train",
        mapper=map_codealpaca,
    ),
    "opencoder_s1_realuser": DatasetSpec(
        key="opencoder_s1_realuser",
        hf_id="OpenCoder-LLM/opc-sft-stage1",
        config="realuser_instruct",
        split="train",
        mapper=map_opencoder,
    ),
    "opencoder_s1_diverse": DatasetSpec(
        key="opencoder_s1_diverse",
        hf_id="OpenCoder-LLM/opc-sft-stage1",
        config="largescale_diverse_instruct",
        split="train",
        mapper=map_opencoder,
        streaming_recommended=True,
    ),
    "opencoder_s2": DatasetSpec(
        key="opencoder_s2",
        hf_id="OpenCoder-LLM/opc-sft-stage2",
        config="educational_instruct",
        split="train",
        mapper=map_opencoder,
        streaming_recommended=True,
    ),
    "opencoder_s2_evol": DatasetSpec(
        key="opencoder_s2_evol",
        hf_id="OpenCoder-LLM/opc-sft-stage2",
        config="evol_instruct",
        split="train",
        mapper=map_opencoder,
        streaming_recommended=True,
    ),
    "opencoder_s2_package": DatasetSpec(
        key="opencoder_s2_package",
        hf_id="OpenCoder-LLM/opc-sft-stage2",
        config="package_instruct",
        split="train",
        mapper=map_opencoder,
        streaming_recommended=True,
    ),
    "opencodeinstruct": DatasetSpec(
        key="opencodeinstruct",
        hf_id="nvidia/OpenCodeInstruct",
        config=None,
        split="train",
        mapper=map_opencodeinstruct,
        streaming_recommended=True,
    ),
}


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_mapped_rows(
    spec: DatasetSpec,
    max_samples: int | None,
    seed: int,
) -> Iterator[dict[str, str]]:
    from datasets import load_dataset

    use_stream = bool(spec.streaming_recommended or (max_samples is not None and max_samples <= 200_000))
    print(f"\nLoading {spec.key} from {spec.hf_id}" + (f" ({spec.config})" if spec.config else ""))
    print(f"  mode={'streaming' if use_stream else 'full'} max_samples={max_samples}")

    kwargs: dict[str, Any] = {"split": spec.split, "streaming": use_stream}
    if spec.config:
        dataset = load_dataset(spec.hf_id, spec.config, **kwargs)
    else:
        dataset = load_dataset(spec.hf_id, **kwargs)

    if use_stream:
        kept = 0
        seen = 0
        target = max_samples if max_samples is not None else None
        for row in dataset:
            seen += 1
            if target is not None and kept >= target:
                break
            # Bound scan depth when filters reject many rows (OpenCodeInstruct tests).
            if target is not None and seen > max(target * 30, 50_000):
                break
            mapped = spec.mapper(dict(row), spec.key)
            if mapped is None:
                continue
            kept += 1
            if kept % 2000 == 0:
                print(f"  kept {kept} / scanned {seen}")
            yield mapped
        print(f"  done: kept {kept} from {seen} scanned")
        return

    # Non-streaming: optional shuffle + select
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    kept = 0
    for row in dataset:
        mapped = spec.mapper(dict(row), spec.key)
        if mapped is None:
            continue
        kept += 1
        yield mapped
    print(f"  done: kept {kept}")


def import_one(
    spec: DatasetSpec,
    output_dir: Path,
    max_samples: int | None,
    seed: int,
) -> Path:
    output_path = output_dir / f"{spec.key}.jsonl"
    records = list(iter_mapped_rows(spec, max_samples=max_samples, seed=seed))
    write_jsonl(output_path, records)
    print(f"  wrote {len(records)} -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import public coding instruction datasets into datasets/hf/*.jsonl"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="colab",
        help="Sampling preset (default: colab).",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys, or 'all' for every key in the chosen preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HF_DIR,
        help="Where to write converted JSONL files.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Override preset cap for every selected dataset.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset keys and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for key, spec in DATASETS.items():
            print(f"{key:24} {spec.hf_id}" + (f" [{spec.config}]" if spec.config else ""))
        return 0

    preset = PRESETS[args.preset]
    if args.datasets.strip().lower() == "all":
        selected = [key for key in preset if key in DATASETS]
    else:
        selected = [part.strip() for part in args.datasets.split(",") if part.strip()]
        unknown = [key for key in selected if key not in DATASETS]
        if unknown:
            raise SystemExit(f"Unknown dataset keys: {unknown}. Use --list.")

    if args.preset == "full":
        print(
            "WARNING: --preset full can download hundreds of thousands to millions of rows.\n"
            "OpenCodeInstruct alone may take a long time and lots of disk. Prefer --preset colab "
            "or --preset large on Colab / free GPUs."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failed: list[str] = []
    for key in selected:
        spec = DATASETS[key]
        cap = args.max_per_dataset if args.max_per_dataset is not None else preset.get(key)
        if not (isinstance(cap, int) or cap is None):
            raise SystemExit(f"Invalid cap for {key}: {cap}")
        try:
            path = import_one(spec, args.output_dir, max_samples=cap, seed=args.seed)
            written.append(path)
        except Exception as exc:
            failed.append(key)
            print(f"ERROR importing {key}: {exc}")
            print("  Continuing with remaining datasets...")

    manifest = {
        "preset": args.preset,
        "seed": args.seed,
        "files": [str(path) for path in written],
        "failed": failed,
    }
    manifest_path = args.output_dir / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nImport complete. Manifest: {manifest_path}")
    if failed:
        print(f"Failed datasets: {', '.join(failed)}")
    print("Next: python scripts/build_training_dataset.py")
    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
