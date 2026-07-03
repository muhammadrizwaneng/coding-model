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

Or run commands explicitly through the venv, for example:

```bash
venv/bin/python -m pip install -r requirements.txt
```

If PyTorch prints a NumPy 2 warning, reinstall the pinned requirements:

```bash
venv/bin/python -m pip install --upgrade --force-reinstall "numpy<2"
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

Colab preinstalls many ML packages, so clean the conflicting packages before installing:

```bash
!pip uninstall -y transformers peft trl accelerate bitsandbytes torchvision torchaudio
!pip install --no-cache-dir -r requirements-colab.txt
```

After installing, restart the runtime:

```text
Runtime > Restart runtime
```

Then run:

```bash
%cd /content/coding-model/coding-model
!python scripts/validate_dataset.py
!python training/train_qlora.py \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --epochs 1 \
  --batch-size 1
```

Start with the 1.5B model on free Colab. Use the 7B model only if Colab gives you enough GPU memory.

## 3. Start a Small Test Run

Run this section on a CUDA GPU machine, not on a normal Mac:

```bash
venv/bin/python training/train_qlora.py \
  --dataset-path datasets/coding_dataset.jsonl \
  --output-dir models/qwen2.5-coder-7b-qlora \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

The script:

- Loads `Qwen/Qwen2.5-Coder-7B-Instruct`
- Formats your JSONL records into chat-style training text
- Splits the dataset into train/eval sets
- Trains LoRA adapters with 4-bit quantization
- Saves adapters under `models/qwen2.5-coder-7b-qlora`

## 4. Compare Against Baseline

After training, generate responses from the fine-tuned model and compare them with:

```text
evaluation/baseline_results.jsonl
```

The next improvement should be a comparison script that scores baseline vs fine-tuned responses with a small rubric.
