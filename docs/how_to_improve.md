# How to improve rizwan-code-model

The current adapter was trained for about one hour, on about 650 examples, on a 1.5B base. Quality improves when the run is longer, the base is larger, and the training answers are actually correct.

## 1. Train longer

The script stops by 4 hours unless you turn the cap off:

```bash
python training/train_qlora.py \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset-path datasets/coding_dataset.jsonl \
  --output-dir models/rizwan-code-model \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 4 \
  --time-budget-minutes 0
```

`--time-budget-minutes 0` uses the full epoch. On a Kaggle T4 that is many hours, and the session must stay alive until the adapter is zipped and downloaded. The 4 hour default still saves an adapter checkpoint every 20 steps. Download the final `models/rizwan-code-model` folder before the session ends.

## 2. Use a larger base

`Qwen/Qwen2.5-Coder-7B-Instruct` writes more reliable code than the 1.5B model. It needs more GPU memory and a longer run. Keep the output directory as `models/rizwan-code-model` if this app should load it, and keep `base_model_id` in `adapter_meta.json` pointed at the 7B model. The test machine must download that 7B base, not the 1.5B one.

## 3. Fix the training data

The model copies the style of `datasets/coding_dataset.jsonl`. The public rows include old tutorials (`from_orm`, mixed async and sync SQLAlchemy, mismatched file names). Add curated rows whose `output` is code you would accept, especially:

- FastAPI with sync SQLAlchemy, or async end to end, not a mix
- Pydantic v2 (`model_validate`, `from_attributes`)
- helper functions and route functions with different names
- a short bug-fix pair: broken code in `input`, fixed code in `output`

Retrain after those rows are in the merged file. More clean examples matter more than training again on the same noisy set.

## 4. Check answers by running them

Loss and token accuracy do not prove the code works. After each new zip:

1. Replace `models/rizwan-code-model`.
2. Restart the local server.
3. Ask for a small function, then a FastAPI CRUD API.
4. Run the code, or compare it with `inference/compare_eval.py` against an Ollama baseline.

Keep a new adapter only when those checks are better than the previous zip.

## 5. What not to expect

Repeating the same one-hour 1.5B run will produce another adapter of similar quality. Improvement comes from a longer run, a 7B base, or cleaner examples, then a new download of that adapter into `models/rizwan-code-model`.
