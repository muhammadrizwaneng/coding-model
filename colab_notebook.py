# -*- coding: utf-8 -*-
"""Colab workflow for fine-tuning and smoke-testing the coding model.

Prefer running these cells in Google Colab with a GPU runtime.
Paths assume the repo is cloned to /content/coding-model.
"""

# Cell 1: clone + validate
# %cd /content
# !rm -rf coding-model
# !git clone https://github.com/muhammadrizwaneng/coding-model.git
# %cd coding-model
# !python scripts/validate_dataset.py

# Cell 2: install training deps
# !pip uninstall -y transformers peft trl accelerate bitsandbytes torchvision torchaudio
# !pip install --no-cache-dir -r requirements-colab.txt
# Runtime > Restart runtime, then %cd /content/coding-model

# Cell 3: train 1.5B QLoRA (saves under a matching path + adapter_meta.json)
# !python training/train_qlora.py \
#   --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
#   --output-dir models/qwen2.5-coder-1.5b-qlora \
#   --epochs 2 \
#   --batch-size 1 \
#   --learning-rate 1e-4 \
#   --gradient-accumulation-steps 8 \
#   --early-stopping-patience 3

# Cell 4: zip adapters for download
# !zip -r qwen2.5-coder-1.5b-qlora.zip models/qwen2.5-coder-1.5b-qlora
# !ls -lh qwen2.5-coder-1.5b-qlora.zip

# Cell 5: smoke test with the same chat template used in inference
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from inference.prompts import DEFAULT_SYSTEM_PROMPT

base_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = os.environ.get(
    "ADAPTER_PATH",
    "/content/coding-model/models/qwen2.5-coder-1.5b-qlora",
)

print("Adapter exists:", os.path.isdir(adapter_path))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


def generate_chat(prompt: str, max_new_tokens: int = 800) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


print(generate_chat("Create a FastAPI CRUD API for products with SQLAlchemy."))
print(generate_chat("Create a reusable React Native login screen with email, password, loading, and error state."))
print(
    generate_chat(
        """Fix this Python bug and explain the issue:

def get_average(numbers):
    total = 0
    for i in range(len(numbers) + 1):
        total += numbers[i]
    return total / len(numbers)
"""
    )
)
