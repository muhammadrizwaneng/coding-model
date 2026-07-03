import os
from pathlib import Path
from typing import Dict, List, Optional

from inference import ollama_client

DEFAULT_BACKEND = os.getenv("MODEL_BACKEND", "finetuned").strip().lower()
DEFAULT_ADAPTER_PATH = Path(os.getenv("ADAPTER_PATH", "models/qwen2.5-coder-7b-qlora"))


def adapter_exists() -> bool:
    return DEFAULT_ADAPTER_PATH.is_dir() and (DEFAULT_ADAPTER_PATH / "adapter_config.json").exists()


def get_backend() -> str:
    if DEFAULT_BACKEND == "ollama":
        return "ollama"
    if DEFAULT_BACKEND == "finetuned":
        return "finetuned"
    if adapter_exists():
        return "finetuned"
    return "ollama"


def model_info() -> Dict[str, str]:
    backend = get_backend()
    if backend == "finetuned":
        from inference import hf_client

        return hf_client.model_info()
    return {
        "model": ollama_client.DEFAULT_MODEL,
        "provider": "ollama",
        "backend": backend,
    }


def chat(
    message: str,
    model: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, object]:
    backend = get_backend()
    if backend == "finetuned":
        from inference import hf_client

        return hf_client.chat(message=message, history=history)
    return ollama_client.chat(message=message, model=model or ollama_client.DEFAULT_MODEL, history=history)


def preload_model() -> None:
    if get_backend() == "finetuned":
        from inference import hf_client

        hf_client.load_model()
