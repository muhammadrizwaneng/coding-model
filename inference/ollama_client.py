import time
from typing import Dict, List, Optional

import ollama

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional full-stack coding assistant specializing in "
    "FastAPI, React, Next.js, Node.js, PostgreSQL, debugging, refactoring, "
    "testing, and code explanation. Provide clear, practical answers with "
    "complete code when appropriate."
)


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, object]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
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
        "response": answer,
        "duration_seconds": duration,
    }
