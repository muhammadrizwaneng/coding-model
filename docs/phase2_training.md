# Phase 2 QLoRA Training

This phase fine-tunes a Hugging Face Qwen Coder base model with QLoRA.

Run this on a CUDA GPU machine. Local Mac training for a 7B model is usually not practical.
Your Mac can validate the dataset and run Ollama inference, but it should not run the 7B QLoRA
training command unless you have a CUDA-capable NVIDIA GPU.

If your shell shows `pyenv: python: command not found` or `pyenv: pip: command not found`,
activate the project virtual environment first:

```bash
source venv/bin/activate
```

## 1. Validate Dataset

```bash
venv/bin/python scripts/validate_dataset.py
```

Do not train until this passes.

## 2. Install Training Dependencies

```bash
venv/bin/python -m pip install -r requirements-training.txt
```

## Google Colab Setup

Use a GPU runtime:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Install deps, restart runtime, then train the **1.5B** model into a matching output directory:

```bash
!pip uninstall -y transformers peft trl accelerate bitsandbytes torchvision torchaudio
!pip install --no-cache-dir -r requirements-colab.txt
```

```bash
%cd /content/coding-model
!python scripts/validate_dataset.py
!python training/train_qlora.py \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --output-dir models/qwen2.5-coder-1.5b-qlora \
  --epochs 2 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 8
```

The trainer writes `adapter_meta.json` next to the adapter so inference can load the correct base model.

Zip for download:

```bash
!zip -r qwen2.5-coder-1.5b-qlora.zip models/qwen2.5-coder-1.5b-qlora
```

## 3. Start a Small Test Run (7B on a real GPU host)

```bash
venv/bin/python training/train_qlora.py \
  --dataset-path datasets/coding_dataset.jsonl \
  --model-id Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir models/qwen2.5-coder-7b-qlora \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

The script:

- Formats JSONL records with the same chat template + system prompt used at inference
- Splits train/eval, trains LoRA with 4-bit quantization, early-stops on eval loss
- Saves adapters + `adapter_meta.json` under the output directory

## 4. Compare Against Baseline

```bash
MODEL_BACKEND=ollama python inference/run_baseline_eval.py \
  --output-file evaluation/baseline_results.jsonl

MODEL_BACKEND=finetuned ADAPTER_PATH=models/qwen2.5-coder-1.5b-qlora \
  python inference/run_baseline_eval.py \
  --output-file evaluation/finetuned_results.jsonl

python inference/compare_eval.py \
  --baseline evaluation/baseline_results.jsonl \
  --candidate evaluation/finetuned_results.jsonl
```

## 5. Serve Locally

```bash
# Auto: adapter if present, otherwise Ollama
MODEL_BACKEND=auto uvicorn server.app:app --reload
```
