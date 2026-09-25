# Large Public Coding Datasets

This project can train on a mix of:

1. Your curated examples in `datasets/raw/` + `datasets/seed.jsonl`
2. Public Hugging Face coding SFT datasets imported into `datasets/hf/`

## Important

Training on literally **every row of every public dataset** (especially `nvidia/OpenCodeInstruct` ~5M) is not practical on free Colab. Use presets:

| Preset | Approx size | When to use |
|--------|-------------|-------------|
| `colab` (default) | ~100k–130k | Free Colab / first broad train |
| `large` | ~300k–500k | Bigger GPU (A100 / Colab Pro) |
| `full` | 500k+ | Serious multi-GPU / long runs |

Your curated rows are **upsampled** during the build step so FastAPI/RN specialty is not drowned out.

## Colab cells

```python
%cd /content/coding-model
!pip install -q datasets huggingface_hub
```

```python
%cd /content/coding-model
# Download + convert all configured public coding datasets (sampled)
!python scripts/import_hf_datasets.py --preset colab

# Merge curated + HF into datasets/coding_dataset.jsonl
!python scripts/build_training_dataset.py --require-hf --curated-upsample 5

!python scripts/validate_dataset.py
!wc -l datasets/coding_dataset.jsonl
!cat datasets/coding_dataset.manifest.json
```

Then train as usual:

```python
!python training/train_qlora.py \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset-path datasets/coding_dataset.jsonl \
  --output-dir models/qwen2.5-coder-1.5b-qlora \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 8 \
  --early-stopping-patience 3
```

Use **1 epoch** when the merged set is large (100k+).

## Local commands

```bash
# List sources
python scripts/import_hf_datasets.py --list

# Import (needs network + `pip install datasets`)
python scripts/import_hf_datasets.py --preset colab

# Optional: only some sources
python scripts/import_hf_datasets.py --preset colab \
  --datasets magicoder_oss,magicoder_evol,codealpaca,opencodeinstruct

# Build final training file
python scripts/build_training_dataset.py --require-hf

# Validate
python scripts/validate_dataset.py
```

## Included Hugging Face sources

- `ise-uiuc/Magicoder-OSS-Instruct-75K`
- `ise-uiuc/Magicoder-Evol-Instruct-110K`
- `sahil2801/CodeAlpaca-20k`
- `OpenCoder-LLM/opc-sft-stage1` (`realuser_instruct`, and diverse in `full`)
- `OpenCoder-LLM/opc-sft-stage2`
- `nvidia/OpenCodeInstruct` (sampled; filters weak unit-test scores)

Check each dataset license before commercial use.
