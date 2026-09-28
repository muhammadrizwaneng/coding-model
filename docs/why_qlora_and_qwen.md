# Why this project uses QLoRA and Qwen

## Why Qwen

`Qwen/Qwen2.5-Coder-1.5B-Instruct` is already a coding model. It was trained by its authors to read and write code, and it understands the chat format this app uses.

Starting from that model is faster than training a coding assistant from empty weights. The 1.5B size is small enough for a free Kaggle or Colab GPU. The 7B Qwen Coder model is stronger and is the better base when a longer GPU session is available.

Your name, `rizwan-code-model`, is the adapter. Qwen remains the base underneath it.

## Why QLoRA

A normal fine-tune updates every weight in the base model. For the 1.5B model that needs a lot of GPU memory and produces another multi-gigabyte checkpoint.

QLoRA does two things:

- **4-bit load (the Q).** The frozen base model is stored in 4-bit, so it fits on a T4-class GPU.
- **LoRA.** Training adds small matrices next to selected layers and updates only those. The saved result is about 37 MB, not 3 GB.

That is why Kaggle can train it, why the zip is small, and why testing still downloads Qwen: the adapter has to be applied to the original base.

## What this combination does not do

QLoRA on a 1.5B model for about an hour does not replace Qwen's existing habits. It only nudges them. Wrong or outdated code in the base model, or in the public datasets, can still show up in answers. A wrong FastAPI sample after this run is expected, not a sign that the adapter failed to load.
