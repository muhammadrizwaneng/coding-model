import argparse
import json
from pathlib import Path

REQUIRED_TEXT_FIELDS = ("instruction", "output")
OPTIONAL_TEXT_FIELDS = ("input",)
MIN_OUTPUT_CHARS = 40


def validate_record(record: object, line_number: int, seen_keys: set[tuple[str, str]]) -> list[str]:
    errors: list[str] = []

    if not isinstance(record, dict):
        return [f"line {line_number}: record must be a JSON object"]

    for field in REQUIRED_TEXT_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"line {line_number}: missing or empty '{field}'")

    for field in OPTIONAL_TEXT_FIELDS:
        value = record.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"line {line_number}: '{field}' must be a string when provided")

    key = (
        str(record.get("instruction", "")).strip().lower(),
        str(record.get("input", "")).strip().lower(),
    )
    if key in seen_keys:
        errors.append(f"line {line_number}: duplicate instruction/input pair")
    seen_keys.add(key)

    output = record.get("output")
    if isinstance(output, str) and len(output.strip()) < MIN_OUTPUT_CHARS:
        errors.append(f"line {line_number}: output is too short for a useful training example")

    return errors


def validate_jsonl(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    record_count = 0

    if not path.exists():
        return 0, [f"{path} does not exist"]

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                errors.append(f"line {line_number}: blank lines are not allowed")
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue

            record_count += 1
            errors.extend(validate_record(record, line_number, seen_keys))

    return record_count, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate instruction-tuning JSONL datasets.")
    parser.add_argument(
        "path",
        nargs="?",
        default="datasets/coding_dataset.jsonl",
        help="Path to the JSONL dataset to validate.",
    )
    args = parser.parse_args()

    count, errors = validate_jsonl(Path(args.path))

    if errors:
        print(f"Dataset validation failed for {args.path}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Dataset is valid: {count} records checked in {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
