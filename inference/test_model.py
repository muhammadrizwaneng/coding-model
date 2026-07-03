import argparse
import time

import ollama

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional full-stack coding assistant specializing in "
    "FastAPI, React Native, PostgreSQL, debugging, and code explanation."
)


def ask_my_model(prompt_text: str, model: str = DEFAULT_MODEL, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    print(f"Sending request to {model}...")
    start_time = time.time()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            "Ollama request failed. Make sure Ollama is installed, running, "
            f"and the model is pulled with: ollama pull {model}"
        ) from exc

    end_time = time.time()
    duration = end_time - start_time

    answer = response["message"]["content"]
    print("\n--- AI RESPONSE ---")
    print(answer)
    print("-------------------------")
    print(f"Speed: generated response in {duration:.2f} seconds.")
    return answer


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