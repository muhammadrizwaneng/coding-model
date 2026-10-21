# rizwan-code-model — Kaggle training

Train the **QLoRA adapter** on the **small curated dataset only** (~400 correct examples in `datasets/raw/` + `datasets/seed.jsonl`), then download **`rizwan-code-model.zip`**.

The default **4-hour** run keeps repeating those examples. **Do not** import the large public Hugging Face presets for this workflow — they include outdated tutorials the previous adapter copied.

## Before you run

1. **Settings → Accelerator → GPU** (T4 is enough).
2. **Settings → Internet → On** (needed for `git clone` and the Qwen download).
3. Leave this tab open for about **3 to 4 hours**. The trainer stops at **4 hours** by default (`--time-budget-minutes 240`).
4. If you already have **`rizwan-code-model.zip`**, use **Add Input** and upload it. Unzip into `models/rizwan-code-model` before training — the train cell **continues** from that adapter. If you skip this, training starts a **new** adapter.
5. Run cells **from top to bottom**. Do **not** use **Run all** again after training finishes.

## Notebook cells (copy into a Kaggle notebook)

### 1. Clone repo

```python
import os

os.chdir("/kaggle/working")
!rm -rf /kaggle/working/coding-model
!git clone https://github.com/muhammadrizwaneng/coding-model.git
os.chdir("/kaggle/working/coding-model")
print("cwd:", os.getcwd())
!git log -1 --oneline
```

### 2. Optional — resume from a previous adapter

```python
from pathlib import Path
import shutil
import zipfile

dest = Path("/kaggle/working/coding-model/models/rizwan-code-model")
dest.parent.mkdir(parents=True, exist_ok=True)

# Upload rizwan-code-model.zip as a Kaggle dataset; name the input folder e.g. rizwan-code-model
input_root = Path("/kaggle/input")
for zip_path in input_root.glob("**/rizwan-code-model.zip"):
    print("Unzipping", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/kaggle/working/coding-model/models/")
    break
else:
    for adapter_cfg in input_root.glob("**/rizwan-code-model/adapter_config.json"):
        src = adapter_cfg.parent
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print("Copied adapter from", src)
        break
    else:
        print("No prior adapter in /kaggle/input — starting fresh.")
```

### 3. Install training dependencies

```python
!pip install --no-cache-dir -r requirements-colab.txt
```

### 4. Build curated-only dataset (no public HF import)

```python
!python scripts/build_training_dataset.py --curated-only --curated-upsample 8
!python scripts/validate_dataset.py
!wc -l datasets/coding_dataset.jsonl
```

### 5. Train QLoRA (4-hour budget on T4)

```python
!python training/train_qlora.py \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset-path datasets/coding_dataset.jsonl \
  --output-dir models/rizwan-code-model \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 8
```

Checkpoints are written under `models/rizwan-code-model` every 20 steps while the time budget is active.

### 6. Zip and download

```python
!cd /kaggle/working/coding-model && zip -r rizwan-code-model.zip models/rizwan-code-model
!ls -lh /kaggle/working/coding-model/rizwan-code-model.zip
```

Download **`rizwan-code-model.zip`** from the Kaggle **Output** panel before the session ends.

## Use the adapter locally

Unzip into `coding-model/models/rizwan-code-model`, then see [phase2_training.md](phase2_training.md) and [how_prompt_works.md](how_prompt_works.md).
