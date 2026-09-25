import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.prompts import DEFAULT_SYSTEM_PROMPT

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATASET_PATH = Path("datasets/coding_dataset.jsonl")


def default_output_dir(model_id: str) -> Path:
    slug = model_id.split("/")[-1].lower().replace("-instruct", "")
    return Path(f"models/{slug}-qlora")


def check_training_environment() -> None:
    if sys.version_info < (3, 10):
        print(
            "Warning: Python 3.9 support is ending for the training stack. "
            "Use Python 3.10+ on your GPU machine."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training requires a CUDA GPU, but CUDA is not available. "
            "Run dataset validation and Ollama inference locally, then run this "
            "training script on a CUDA GPU machine such as RunPod, Vast.ai, "
            "Colab Pro, or a local NVIDIA GPU setup."
        )


def format_example(example: dict[str, str], tokenizer) -> dict[str, str]:
    user_content = example["instruction"].strip()
    if example.get("input", "").strip():
        user_content = f"{user_content}\n\nContext:\n{example['input'].strip()}"

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"].strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def prepare_dataset(
    dataset_path: Path,
    tokenizer,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda example: format_example(example, tokenizer))

    if len(dataset) < 10:
        raise ValueError("Dataset is too small for a train/eval split. Add more examples first.")

    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def build_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def write_adapter_meta(output_dir: Path, model_id: str, dataset_path: Path) -> None:
    meta = {
        "base_model_id": model_id,
        "dataset_path": str(dataset_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    (output_dir / "adapter_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen Coder with QLoRA.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Defaults to models/<model-slug>-qlora based on --model-id.",
    )
    parser.add_argument("--eval-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=1, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--max-seq-length", default=2048, type=int)
    parser.add_argument("--early-stopping-patience", default=3, type=int)
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir(args.model_id)

    check_training_environment()
    model, tokenizer = build_model_and_tokenizer(args.model_id)
    train_dataset, eval_dataset = prepare_dataset(
        args.dataset_path,
        tokenizer,
        args.eval_ratio,
        args.seed,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_adapter_meta(output_dir, args.model_id, args.dataset_path)

    print(f"Saved QLoRA adapter, tokenizer, and adapter_meta.json to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
