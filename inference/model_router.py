import os
from pathlib import Path
from typing import Dict, List, Optional

from inference import ollama_client

DEFAULT_BACKEND = os.getenv("MODEL_BACKEND", "auto").strip().lower()
DEFAULT_ADAPTER_PATH = Path(
    os.getenv("ADAPTER_PATH", "models/qwen2.5-coder-1.5b-qlora")
)
LEGACY_ADAPTER_PATH = Path("models/qwen2.5-coder-7b-qlora")


def resolve_adapter_path() -> Path:
    configured = DEFAULT_ADAPTER_PATH
    if adapter_exists(configured):
        return configured
    if adapter_exists(LEGACY_ADAPTER_PATH):
        return LEGACY_ADAPTER_PATH
    return configured


def adapter_exists(adapter_path: Optional[Path] = None) -> bool:
    path = adapter_path or resolve_adapter_path()
    return path.is_dir() and (path / "adapter_config.json").exists()


def get_backend() -> str:
    """Resolve inference backend.

    - ollama: force Ollama
    - finetuned: force PEFT adapter (errors if missing)
    - auto (default): use adapter when present, otherwise Ollama
    """
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
        "adapter_ready": str(adapter_exists()).lower(),
    }


def chat(
    message: str,
    model: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, object]:
    backend = get_backend()
    if backend == "finetuned":
        if not adapter_exists():
            raise RuntimeError(
                "Fine-tuned backend selected but no adapter was found. "
                f"Expected files under {resolve_adapter_path()} "
                "(or set MODEL_BACKEND=ollama / MODEL_BACKEND=auto)."
            )
        from inference import hf_client

        return hf_client.chat(message=message, history=history)
    return ollama_client.chat(
        message=message,
        model=model or ollama_client.DEFAULT_MODEL,
        history=history,
    )


def preload_model() -> None:
    if get_backend() == "finetuned":
        from inference import hf_client

        hf_client.load_model()
