import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_router import chat, get_backend

DEFAULT_EVAL_FILE = PROJECT_ROOT / "evaluation" / "eval_prompts.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "eval_results.jsonl"


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "task" not in record or "prompt" not in record:
                raise ValueError(f"line {line_number}: expected 'task' and 'prompt' fields")
            prompts.append(record)
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evaluation prompts through the active model backend (auto/ollama/finetuned)."
    )
    parser.add_argument("--model", default=None, help="Ollama model name when backend is ollama.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE, type=Path)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, type=Path)
    args = parser.parse_args()

    prompts = load_prompts(args.eval_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    backend = get_backend()

    with args.output_file.open("w", encoding="utf-8") as output:
        for index, item in enumerate(prompts, start=1):
            print(f"\n[{index}/{len(prompts)}] {item['task']} ({backend})")
            started_at = time.time()
            result = chat(message=item["prompt"], model=args.model)
            output.write(
                json.dumps(
                    {
                        "task": item["task"],
                        "prompt": item["prompt"],
                        "backend": backend,
                        "model": result.get("model"),
                        "provider": result.get("provider"),
                        "response": result["response"],
                        "duration_seconds": round(time.time() - started_at, 2),
                        "criteria": item.get("criteria", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"\nSaved evaluation results to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
