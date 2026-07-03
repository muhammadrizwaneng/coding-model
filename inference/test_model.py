import argparse

from inference.ollama_client import DEFAULT_MODEL, chat


def ask_my_model(prompt_text: str, model: str = DEFAULT_MODEL) -> str:
    result = chat(message=prompt_text, model=model)
    print("\n--- AI RESPONSE ---")
    print(result["response"])
    print("-------------------------")
    print(f"Speed: generated response in {result['duration_seconds']} seconds.")
    return result["response"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the local Ollama coding model.")
    parser.add_argument(
        "--prompt",
        default="Write a FastAPI endpoint snippet for an item upload.",
        help="Prompt to send to the model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    args = parser.parse_args()

    ask_my_model(args.prompt, model=args.model)


if __name__ == "__main__":
    main()