import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftConfig, PeftModel, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE_MODEL = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
DEFAULT_ADAPTER_PATH = Path(os.getenv("ADAPTER_PATH", "models/qwen2.5-coder-7b-qlora"))
DEFAULT_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "my-coding-model-1.5b")
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional full-stack coding assistant specializing in "
    "FastAPI, React, Next.js, Node.js, PostgreSQL, debugging, refactoring, "
    "testing, and code explanation. Provide clear, practical answers with "
    "complete code when appropriate."
)

_model = None
_tokenizer = None
_device = None


def adapter_exists(adapter_path: Path = DEFAULT_ADAPTER_PATH) -> bool:
    return adapter_path.is_dir() and (adapter_path / "adapter_config.json").exists()


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Colab LoRA adapters are saved in bfloat16; Mac MPS cannot load them.
    return "cpu"


def load_model(
    base_model_id: str = DEFAULT_BASE_MODEL,
    adapter_path: Path = DEFAULT_ADAPTER_PATH,
) -> None:
    global _model, _tokenizer, _device

    if _model is not None and _tokenizer is not None:
        return

    if not adapter_exists(adapter_path):
        raise RuntimeError(
            f"Fine-tuned adapter not found at {adapter_path}. "
            "Download your Colab zip and extract it there, for example:\n"
            "models/qwen2.5-coder-7b-qlora/adapter_config.json"
        )

    _device = _resolve_device()
    try:
        _tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    except Exception:
        _tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            use_fast=False,
        )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = _load_finetuned_model(base_model_id, adapter_path, _device)
    _model.eval()


def _load_finetuned_model(base_model_id: str, adapter_path: Path, device: str):
    """Load base + LoRA adapter with dtypes compatible with Mac CPU inference."""
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        return PeftModel.from_pretrained(base_model, str(adapter_path))

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": "cpu"},
    )

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    model = get_peft_model(base_model, peft_config)

    adapter_weights = load_file(str(adapter_path / "adapter_model.safetensors"))
    adapter_weights = {
        key: tensor.to(dtype=torch.float32, device="cpu")
        for key, tensor in adapter_weights.items()
    }
    set_peft_model_state_dict(model, adapter_weights)
    return model.to("cpu")


def _build_messages(
    message: str,
    system_prompt: str,
    history: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        for item in history:
            if item["role"] in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})
    return messages


def chat(
    message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 800,
) -> Dict[str, object]:
    load_model()

    messages = _build_messages(message, system_prompt, history)
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    started_at = time.time()
    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = _tokenizer.decode(generated, skip_special_tokens=True).strip()
    duration = round(time.time() - started_at, 2)

    return {
        "model": DEFAULT_DISPLAY_NAME,
        "provider": "huggingface-peft",
        "base_model": DEFAULT_BASE_MODEL,
        "adapter_path": str(DEFAULT_ADAPTER_PATH),
        "response": answer,
        "duration_seconds": duration,
    }


def model_info() -> Dict[str, str]:
    return {
        "model": DEFAULT_DISPLAY_NAME,
        "provider": "huggingface-peft",
        "base_model": DEFAULT_BASE_MODEL,
        "adapter_path": str(DEFAULT_ADAPTER_PATH),
        "adapter_ready": str(adapter_exists()).lower(),
        "inference_device": _resolve_device(),
    }
