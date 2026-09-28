import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.prompts import DEFAULT_SYSTEM_PROMPT

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATASET_PATH = Path("datasets/coding_dataset.jsonl")
MODEL_NAME = "rizwan-code-model"
DEFAULT_OUTPUT_DIR = Path(f"models/{MODEL_NAME}")
# Older Colab/Kaggle cells still pass this path. Save under the new name instead.
RENAMED_OUTPUT_DIRS = {Path("models/qwen2.5-coder-1.5b-qlora")}


def resolve_output_dir(requested: Path | None) -> Path:
    if requested is None or requested in RENAMED_OUTPUT_DIRS:
        if requested is not None:
            print(f"Saving {MODEL_NAME} to {DEFAULT_OUTPUT_DIR} instead of {requested}.")
        return DEFAULT_OUTPUT_DIR
    return requested


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


def load_jsonl_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            continue
        records.append(
            {
                "instruction": instruction,
                "input": str(row.get("input") or "").strip(),
                "output": output,
            }
        )
    return records


def load_curated_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seed_file = PROJECT_ROOT / "datasets" / "seed.jsonl"
    raw_dir = PROJECT_ROOT / "datasets" / "raw"
    records.extend(load_jsonl_records(seed_file))
    if raw_dir.exists():
        for path in sorted(raw_dir.glob("*.jsonl")):
            records.extend(load_jsonl_records(path))
    return records


def expand_curated_for_budget(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Repeat FastAPI + SQLAlchemy rows so a short run actually sees the fix."""
    expanded: list[dict[str, str]] = []
    for record in records:
        text = f"{record['instruction']} {record['input']}".lower()
        if "fastapi" in text and "sqlalchemy" in text:
            expanded.extend([record] * 30)
        elif "fastapi" in text:
            expanded.extend([record] * 4)
        else:
            expanded.append(record)
    return expanded


def prepare_dataset(
    dataset_path: Path,
    tokenizer,
    eval_ratio: float,
    seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    if max_train_samples is not None or max_eval_samples is not None:
        eval_n = max_eval_samples if max_eval_samples is not None else max(1, int(len(dataset) * eval_ratio))
        train_n = max_train_samples if max_train_samples is not None else max(1, len(dataset) - eval_n)
        need = train_n + eval_n
        curated = expand_curated_for_budget(load_curated_records())
        hf_need = max(0, need - len(curated))
        parts: list[Dataset] = []
        if curated:
            parts.append(Dataset.from_list(curated))
        if hf_need:
            hf = dataset.shuffle(seed=seed)
            hf = hf.select(range(min(len(hf), hf_need)))
            keep_columns = [name for name in ("instruction", "input", "output") if name in hf.column_names]
            hf = hf.select_columns(keep_columns)
            parts.append(hf)
        dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        print(
            f"Time budget: {len(curated)} curated rows "
            f"(FastAPI + SQLAlchemy repeated) and {hf_need} other rows."
        )

    dataset = dataset.map(lambda example: format_example(example, tokenizer))

    if len(dataset) < 10:
        raise ValueError("Dataset is too small for a train/eval split. Add more examples first.")

    if max_eval_samples is not None:
        eval_n = min(max_eval_samples, max(1, len(dataset) // 5))
        split = dataset.train_test_split(test_size=eval_n, seed=seed)
    else:
        split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    print(f"Split: {len(split['train'])} train, {len(split['test'])} eval")
    return split["train"], split["test"]


class AdapterCheckpointCallback(TrainerCallback):
    """Save the LoRA adapter only, so a killed session still leaves a downloadable checkpoint."""

    def __init__(self, output_dir: Path, tokenizer, save_steps: int, model_id: str, dataset_path: Path):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.save_steps = save_steps
        self.model_id = model_id
        self.dataset_path = dataset_path

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0 or self.save_steps <= 0 or step % self.save_steps != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        step_dir = self.output_dir / f"checkpoint-{step}"
        model.save_pretrained(step_dir)
        self.tokenizer.save_pretrained(step_dir)
        write_adapter_meta(step_dir, self.model_id, self.dataset_path)
        print(f"Saved adapter checkpoint to {step_dir}", flush=True)


def configure_for_time_budget(args) -> None:
    """Shrink steps and eval so the whole run finishes inside the wall-clock budget.

    On a Kaggle T4 this dataset costs about 33s per optimizer step, and a full
    eval is about 74 minutes. 800 steps with eval every 25 steps is more than
    a day, and the session dies when the laptop shuts down.
    """
    args.use_time_budget = args.time_budget_minutes > 0
    if not args.use_time_budget:
        if args.eval_steps is None:
            args.eval_steps = 25
        return

    reserve_seconds = 6 * 60
    seconds_per_step = 40.0
    fitted_steps = max(20, int((args.time_budget_minutes * 60 - reserve_seconds) / seconds_per_step))
    if args.max_steps < 0 or args.max_steps > fitted_steps:
        print(
            f"Time budget {args.time_budget_minutes:g} min: "
            f"capping max_steps {args.max_steps} -> {fitted_steps}."
        )
        args.max_steps = fitted_steps

    eval_cap = 32
    if args.max_eval_samples < 0 or args.max_eval_samples > eval_cap:
        args.max_eval_samples = eval_cap
    args.eval_steps = args.max_steps
    expected_hours = (args.max_steps * 33) / 3600
    print(
        f"Planned run: {args.max_steps} optimizer steps, "
        f"one eval of {args.max_eval_samples} samples at the end, "
        f"adapter checkpoint every 20 steps. "
        f"Expected about {expected_hours:.1f} hours on a T4, "
        f"and it stops by {args.time_budget_minutes / 60:.0f} hours. "
        f"Pass --time-budget-minutes 0 for a full run."
    )


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
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def write_adapter_meta(output_dir: Path, model_id: str, dataset_path: Path) -> None:
    meta = {
        "model_name": MODEL_NAME,
        "base_model_id": model_id,
        "dataset_path": str(dataset_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    (output_dir / "adapter_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_sft_config(**kwargs) -> SFTConfig:
    """Create SFTConfig with Kaggle/Colab-safe defaults.

    Newer TRL defaults to loss_type='chunked_nll', which crashes when model.forward
    is a functools.partial (common with PEFT-wrapped Qwen on Kaggle).
    """
    params = set(inspect.signature(SFTConfig.__init__).parameters)
    if "loss_type" in params:
        kwargs.setdefault("loss_type", "nll")
    # Older TRL used max_seq_length; newer uses max_length.
    if "max_length" not in params and "max_seq_length" in params and "max_length" in kwargs:
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    return SFTConfig(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen Coder with QLoRA.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Defaults to models/rizwan-code-model. The old qwen2.5-coder-1.5b-qlora path is renamed.",
    )
    parser.add_argument("--eval-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=1, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--max-seq-length", default=2048, type=int)
    parser.add_argument("--early-stopping-patience", default=3, type=int)
    parser.add_argument(
        "--max-steps",
        default=-1,
        type=int,
        help="Stop after N optimizer steps (-1 = use epochs). Capped by --time-budget-minutes.",
    )
    parser.add_argument(
        "--time-budget-minutes",
        default=240,
        type=float,
        help="Wall-clock cap for the whole training run. Default 240 minutes (4 hours). 0 disables the cap.",
    )
    parser.add_argument(
        "--max-eval-samples",
        default=-1,
        type=int,
        help="Eval rows to keep. Default 32 when a time budget is set, otherwise the full split.",
    )
    parser.add_argument(
        "--eval-steps",
        default=None,
        type=int,
        help="Evaluate every N optimizer steps. Ignored when a time budget is set (one eval at the end).",
    )
    args = parser.parse_args()
    configure_for_time_budget(args)

    output_dir = resolve_output_dir(args.output_dir)

    check_training_environment()
    model, tokenizer = build_model_and_tokenizer(args.model_id)
    examples_per_step = args.batch_size * args.gradient_accumulation_steps
    max_train_samples = None
    max_eval_samples = None
    if args.use_time_budget:
        # Extra rows cover examples dropped while building labels, so training still reaches max_steps.
        max_train_samples = int(args.max_steps * examples_per_step * 1.1) + 8
        max_eval_samples = args.max_eval_samples
    elif args.max_eval_samples > 0:
        max_eval_samples = args.max_eval_samples
    train_dataset, eval_dataset = prepare_dataset(
        args.dataset_path,
        tokenizer,
        args.eval_ratio,
        args.seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
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
    # Apply LoRA before SFTTrainer so TRL does not re-wrap in a way that
    # leaves model.forward as functools.partial during chunked-CE patching.
    # If a previous rizwan-code-model adapter is already in the output folder,
    # keep training it instead of starting over.
    existing_adapter = output_dir / "adapter_config.json"
    if existing_adapter.exists():
        print(f"Continuing training from existing adapter at {output_dir}")
        model = PeftModel.from_pretrained(model, str(output_dir), is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)

    # Frequent full evals dominate runtime (~74 min each). Under a time budget,
    # evaluate once at the end and save the adapter without the optimizer state.
    if args.use_time_budget:
        sft_kwargs = dict(
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="no",
            load_best_model_at_end=False,
        )
        callbacks = [
            AdapterCheckpointCallback(
                output_dir,
                tokenizer,
                save_steps=20,
                model_id=args.model_id,
                dataset_path=args.dataset_path,
            ),
        ]
    else:
        sft_kwargs = dict(
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]

    training_args = build_sft_config(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        **sft_kwargs,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "callbacks": callbacks,
    }
    # Prefer processing_class (new TRL); fall back to tokenizer= for older TRL.
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_adapter_meta(output_dir, args.model_id, args.dataset_path)

    print(f"Saved QLoRA adapter, tokenizer, and adapter_meta.json to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
