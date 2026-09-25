# Phase 1 Workflow

This phase builds a reliable baseline coding assistant before fine-tuning.

## 1. Create Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Pull Base Model

```bash
ollama pull qwen2.5-coder:7b
```

## 3. Generate Seed Dataset (safe)

Seeds write to `datasets/seed.jsonl` and do **not** overwrite the main dataset:

```bash
python generate_data.py
```

## 4. Sync + Merge Dataset

Keep `datasets/raw/` as the source of truth, including curated orphans:

```bash
python scripts/sync_orphans_to_raw.py
python scripts/merge_datasets.py
```

## 5. Validate Dataset

```bash
python scripts/validate_dataset.py
```

Optional stricter checks (secrets / incomplete CRUD):

```bash
python scripts/validate_dataset.py --strict
```

Do not start training until validation passes and the dataset has at least 100 strong examples.

## 6. Test One Prompt

Uses the active backend (`MODEL_BACKEND=auto` by default → fine-tuned adapter if present, else Ollama):

```bash
python inference/test_model.py --prompt "Create a FastAPI CRUD API for products."
```

## 7. Run Baseline Evaluation

```bash
MODEL_BACKEND=ollama python inference/run_baseline_eval.py \
  --output-file evaluation/baseline_results.jsonl
```

Later compare against a fine-tuned run with `inference/compare_eval.py`.
