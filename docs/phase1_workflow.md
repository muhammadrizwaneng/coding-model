# Phase 1 Workflow

This phase builds a reliable baseline coding assistant before fine-tuning.

## 1. Create Environment

```bash
/usr/local/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Pull Base Model

```bash
ollama pull qwen2.5-coder:7b
```

## 3. Generate Seed Dataset

```bash
python generate_data.py
```

## 4. Validate Dataset

```bash
python scripts/validate_dataset.py
```

Do not start training until validation passes and the dataset has at least 100 strong examples.

## 5. Test One Prompt

```bash
python inference/test_model.py --prompt "Create a FastAPI CRUD API for products."
```

## 6. Run Baseline Evaluation

```bash
python inference/run_baseline_eval.py
```

This writes model outputs to:

```text
evaluation/baseline_results.jsonl
```

Use this file later to compare the base model with your fine-tuned model.
