# -*- coding: utf-8 -*-
"""Kaggle workflow for rizwan-code-model (curated-only QLoRA, 4-hour T4 run).

See docs/kaggle_training.md for settings (GPU, Internet on) and Add Input resume steps.
Paths assume the repo is cloned to /kaggle/working/coding-model.
"""

# Cell 1: clone
# import os
# os.chdir("/kaggle/working")
# !rm -rf /kaggle/working/coding-model
# !git clone https://github.com/muhammadrizwaneng/coding-model.git
# os.chdir("/kaggle/working/coding-model")
# print("cwd:", os.getcwd())
# !git log -1 --oneline

# Cell 2: optional resume from rizwan-code-model.zip (Kaggle Add Input)
# (see docs/kaggle_training.md)

# Cell 3: deps
# !pip install --no-cache-dir -r requirements-colab.txt

# Cell 4: curated-only dataset (~400 rows, no public HF)
# !python scripts/build_training_dataset.py --curated-only --curated-upsample 8
# !python scripts/validate_dataset.py

# Cell 5: train (default 4h time budget)
# !python training/train_qlora.py \
#   --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
#   --dataset-path datasets/coding_dataset.jsonl \
#   --output-dir models/rizwan-code-model \
#   --epochs 1 \
#   --batch-size 1 \
#   --learning-rate 1e-4 \
#   --gradient-accumulation-steps 8

# Cell 6: zip for download
# !cd /kaggle/working/coding-model && zip -r rizwan-code-model.zip models/rizwan-code-model
# !ls -lh rizwan-code-model.zip
