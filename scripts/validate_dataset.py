"""Validate instruction-tuning JSONL datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUIRED_TEXT_FIELDS = ("instruction", "output")
OPTIONAL_TEXT_FIELDS = ("input",)
MIN_OUTPUT_CHARS = 40
WARN_OUTPUT_CHARS = 80
MIN_CRUD_OUTPUT_CHARS = 250

SECRET_PATTERNS = [
    (re.compile(r"\bSECRET\s*=\s*['\"][^'\"]{3,}['\"]", re.I), "hardcoded secret literal"),
    (re.compile(r"\bpassword\s*===?\s*['\"][^'\"]{3,}['\"]", re.I), "hardcoded password comparison"),
    (re.compile(r"\bpassword\s*:\s*['\"][^'\"]{3,}['\"]", re.I), "hardcoded password literal"),
    (re.compile(r"allow_origins\s*=\s*\[\s*['\"]\*['\"]\s*\]", re.I), "CORS allow_origins=['*']"),
    (re.compile(r"postgresql://[^:\s'\"]+:[^@\s'\"]+@", re.I), "database URL with embedded credentials"),
    (re.compile(r"mock-jwt-token|token_secret_", re.I), "mock/hardcoded JWT token"),
]


CRUD_HINT = re.compile(r"\bcrud\b", re.I)
CRUD_VERBS = (
    re.compile(r"\b(create|post)\b", re.I),
    re.compile(r"\b(list|get|read|find)\b", re.I),
    re.compile(r"\b(update|put|patch)\b", re.I),
    re.compile(r"\b(delete|remove)\b", re.I),
)


def find_secret_issues(text: str) -> list[str]:
    issues: list[str] = []
    for pattern, label in SECRET_PATTERNS:
        if pattern.search(text):
            issues.append(label)
    return issues


def crud_incomplete(instruction: str, output: str) -> bool:
    if not CRUD_HINT.search(instruction):
        return False
    # Partial examples that only ask for one slice of CRUD.
    if re.search(r"\b(create endpoint|only create|create-only)\b", instruction, re.I):
        return False
    if len(output.strip()) < MIN_CRUD_OUTPUT_CHARS:
        return True
    return sum(1 for verb in CRUD_VERBS if verb.search(output)) < 3


def validate_record(
    record: object,
    line_number: int,
    seen_keys: set[tuple[str, str]],
    *,
    strict: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(record, dict):
        return [f"line {line_number}: record must be a JSON object"], warnings

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

    instruction = record.get("instruction") if isinstance(record.get("instruction"), str) else ""
    output = record.get("output") if isinstance(record.get("output"), str) else ""
    input_text = record.get("input") if isinstance(record.get("input"), str) else ""

    if output and len(output.strip()) < MIN_OUTPUT_CHARS:
        errors.append(
            f"line {line_number}: output is too short for a useful training example "
            f"(<{MIN_OUTPUT_CHARS} chars)"
        )
    elif output and len(output.strip()) < WARN_OUTPUT_CHARS:
        warnings.append(
            f"line {line_number}: short output ({len(output.strip())} chars); prefer fuller answers"
        )

    if crud_incomplete(instruction, output):
        message = (
            f"line {line_number}: instruction mentions CRUD but output looks incomplete "
            "(missing create/list/update/delete coverage)"
        )
        if strict:
            errors.append(message)
        else:
            warnings.append(message)

    blob = output
    for issue in find_secret_issues(blob):
        message = f"line {line_number}: insecure training pattern ({issue})"
        if strict:
            errors.append(message)
        else:
            warnings.append(message)

    if not input_text.strip():
        warnings.append(f"line {line_number}: empty input (prefer buggy code, schemas, or constraints)")

    return errors, warnings


def validate_jsonl(path: Path, *, strict: bool) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    record_count = 0

    if not path.exists():
        return 0, [f"{path} does not exist"], warnings

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
            record_errors, record_warnings = validate_record(
                record,
                line_number,
                seen_keys,
                strict=strict,
            )
            errors.extend(record_errors)
            warnings.extend(record_warnings)

    return record_count, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate instruction-tuning JSONL datasets.")
    parser.add_argument(
        "path",
        nargs="?",
        default="datasets/coding_dataset.jsonl",
        help="Path to the JSONL dataset to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat insecure patterns and incomplete CRUD examples as errors.",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=20,
        help="How many warnings to print (default: 20).",
    )
    args = parser.parse_args()

    count, errors, warnings = validate_jsonl(Path(args.path), strict=args.strict)

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings[: args.max_warnings]:
            print(f"- {warning}")
        if len(warnings) > args.max_warnings:
            print(f"- ... and {len(warnings) - args.max_warnings} more")

    if errors:
        print(f"Dataset validation failed for {args.path}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Dataset is valid: {count} records checked in {args.path}")
    if warnings:
        print(f"Note: {len(warnings)} quality warnings (re-run with --strict to enforce).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
