import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATASET_PATH = Path("datasets/coding_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("models/qwen2.5-coder-7b-qlora")


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


def format_example(example: dict[str, str]) -> str:
    user_content = example["instruction"].strip()
    if example.get("input", "").strip():
        user_content = f"{user_content}\n\nContext:\n{example['input'].strip()}"

    return (
        "<|im_start|>system\n"
        "You are a professional full-stack coding assistant specializing in FastAPI, "
        "React Native, PostgreSQL, debugging, refactoring, testing, and API integration."
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output'].strip()}<|im_end|>"
    )


def prepare_dataset(dataset_path: Path, eval_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda example: {"text": format_example(example)})

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen Coder with QLoRA.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--eval-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=1, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--max-seq-length", default=2048, type=int)
    args = parser.parse_args()

    check_training_environment()
    train_dataset, eval_dataset = prepare_dataset(args.dataset_path, args.eval_ratio, args.seed)
    model, tokenizer = build_model_and_tokenizer(args.model_id)

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
        output_dir=str(args.output_dir),
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
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Saved QLoRA adapter and tokenizer files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
