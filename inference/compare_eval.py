"""Compare baseline vs fine-tuned evaluation result files with a simple rubric."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CODE_FENCE = re.compile(r"```")
HTTP_HINT = re.compile(r"\b(get|post|put|patch|delete)\b|@app\.|router\.|app\.(get|post)", re.I)
EXPLAIN_HINT = re.compile(r"\b(because|issue|bug|fix|cause|explain)\b", re.I)


def load_results(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            rows[record["task"]] = record
    return rows


def score_response(task: str, prompt: str, response: str, criteria: list[str] | None = None) -> dict:
    text = response or ""
    scores = {
        "non_empty": 1.0 if len(text.strip()) > 40 else 0.0,
        "has_code": 1.0 if CODE_FENCE.search(text) or "def " in text or "function " in text or "const " in text else 0.0,
        "substantial": 1.0 if len(text.strip()) >= 200 else 0.5 if len(text.strip()) >= 80 else 0.0,
    }

    criteria = criteria or []
    criteria_hits = 0
    for item in criteria:
        if item.lower() in text.lower():
            criteria_hits += 1
    if criteria:
        scores["criteria_coverage"] = criteria_hits / len(criteria)

    if "debug" in task or "fix" in prompt.lower():
        scores["explains_issue"] = 1.0 if EXPLAIN_HINT.search(text) else 0.0
    if "api" in task or "fastapi" in task or "crud" in task:
        scores["mentions_http"] = 1.0 if HTTP_HINT.search(text) else 0.0

    values = list(scores.values())
    scores["total"] = round(sum(values) / len(values), 3)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two eval result JSONL files.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=Path("evaluation/comparison_report.json"),
        type=Path,
    )
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise SystemExit("No shared tasks between baseline and candidate files.")

    rows = []
    baseline_total = 0.0
    candidate_total = 0.0
    for task in shared:
        base_row = baseline[task]
        cand_row = candidate[task]
        base_score = score_response(
            task,
            base_row.get("prompt", ""),
            base_row.get("response", ""),
            base_row.get("criteria"),
        )
        cand_score = score_response(
            task,
            cand_row.get("prompt", ""),
            cand_row.get("response", ""),
            cand_row.get("criteria"),
        )
        baseline_total += base_score["total"]
        candidate_total += cand_score["total"]
        rows.append(
            {
                "task": task,
                "baseline_score": base_score,
                "candidate_score": cand_score,
                "delta": round(cand_score["total"] - base_score["total"], 3),
            }
        )

    report = {
        "tasks_compared": len(shared),
        "baseline_avg": round(baseline_total / len(shared), 3),
        "candidate_avg": round(candidate_total / len(shared), 3),
        "avg_delta": round((candidate_total - baseline_total) / len(shared), 3),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Compared {len(shared)} tasks")
    print(f"Baseline avg:  {report['baseline_avg']}")
    print(f"Candidate avg: {report['candidate_avg']}")
    print(f"Avg delta:     {report['avg_delta']}")
    print(f"Saved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
