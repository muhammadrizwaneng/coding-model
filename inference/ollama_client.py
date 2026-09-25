import time
from typing import Dict, List, Optional

from inference.prompts import DEFAULT_SYSTEM_PROMPT, truncate_history

DEFAULT_MODEL = "qwen2.5-coder:7b"


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, object]:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The 'ollama' package is not installed. "
            "Install it with: pip install ollama"
        ) from exc

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(truncate_history(history))
    messages.append({"role": "user", "content": message})

    started_at = time.time()
    try:
        response = ollama.chat(model=model, messages=messages)
    except Exception as exc:
        raise RuntimeError(
            "Ollama request failed. Make sure Ollama is installed, running, "
            f"and the model is pulled with: ollama pull {model}"
        ) from exc

    answer = response["message"]["content"]
    duration = round(time.time() - started_at, 2)

    return {
        "model": model,
        "provider": "ollama",
        "response": answer,
        "duration_seconds": duration,
    }
