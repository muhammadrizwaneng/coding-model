import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_router import chat, get_backend
from inference.ollama_client import DEFAULT_MODEL


def ask_my_model(prompt_text: str, model: str | None = None) -> str:
    result = chat(message=prompt_text, model=model)
    print("\n--- AI RESPONSE ---")
    print(result["response"])
    print("-------------------------")
    print(f"Backend: {get_backend()}")
    print(f"Speed: generated response in {result['duration_seconds']} seconds.")
    return str(result["response"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the active coding model backend.")
    parser.add_argument(
        "--prompt",
        default="Write a FastAPI endpoint snippet for an item upload.",
        help="Prompt to send to the model.",
    )
    parser.add_argument("--model", default=None, help="Ollama model name when backend is ollama.")
    args = parser.parse_args()

    ask_my_model(args.prompt, model=args.model or DEFAULT_MODEL)


if __name__ == "__main__":
    main()
