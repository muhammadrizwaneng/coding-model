# How this model is trained

Training does not create a new 3 GB model. It teaches a small adapter, then saves that adapter as `models/rizwan-code-model`.

## 1. Build the dataset

Public coding sets (Magicoder, CodeAlpaca, OpenCoder, and others) are imported, merged with the curated rows, and written to `datasets/coding_dataset.jsonl`.

Each row has:

- `instruction`: what the user asked
- `input`: optional extra context (buggy code, schema, constraints)
- `output`: the answer the model should learn

`training/train_qlora.py` turns each row into a chat: system prompt, user text, assistant answer. That is the same chat shape used later when you prompt the app.

The last Kaggle run had **34,354** unique rows. The one-hour cap did not train on all of them.

## 2. Load Qwen in 4-bit and add LoRA

On a CUDA GPU (Kaggle or Colab), the script:

1. Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct` in 4-bit (QLoRA), so the full model fits in GPU memory.
2. Freezes those base weights. They are not updated.
3. Adds a LoRA adapter (`r=16`) on the attention and MLP layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
4. Trains only those adapter matrices.

## 3. The one-hour run

`train_qlora.py` defaults to a **4 hour** budget (about 3 to 3.5 hours on a T4) so a closed laptop does not lose a longer job. The first saved adapter used the old 60 minute cap.

For the run that produced this zip, that meant:

- about **81 optimizer steps** (batch size 2, gradient accumulation 4, so about 650 examples)
- one short eval of 32 samples at the end
- an adapter checkpoint every 20 steps (`checkpoint-20`, `checkpoint-40`, `checkpoint-60`, `checkpoint-80`)
- the final adapter in `models/rizwan-code-model`

Loss near `0.66` means the adapter got better at predicting the training text. It does not mean the generated code runs.

## 4. What is saved

The folder contains the adapter, tokenizer files, and `adapter_meta.json` (`model_name`, base model id, dataset path, system prompt). It does not contain the 3 GB Qwen weights. Those stay on Hugging Face and are downloaded again when you test.
