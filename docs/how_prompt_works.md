# How the model works when you give a prompt

`rizwan-code-model` is not a separate full model. A prompt runs **Qwen2.5-Coder-1.5B-Instruct** with your trained adapter attached on top.

## What is loaded at startup

When the server starts with `MODEL_BACKEND=finetuned`, `server/app.py` calls `preload_model()`.

1. The app looks for `models/rizwan-code-model/adapter_config.json`.
2. `adapter_meta.json` says the base model is `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
3. That base model is downloaded once (about 3 GB) if it is not already on the machine.
4. The adapter weights (`adapter_model.safetensors`, about 37 MB) are loaded on top of Qwen.

On a Mac there is no CUDA GPU, so this load uses the CPU. The first answer is slow. Later answers reuse the same loaded model until you stop the server.

## What happens to one prompt

1. The browser sends your text to `POST /api/chat`.
2. `inference/model_router.py` sees the adapter and chooses the fine-tuned backend.
3. `inference/hf_client.py` builds a chat with three parts:
   - a fixed system prompt (coding assistant instructions)
   - earlier messages in this chat, if any
   - your new prompt as the user message
4. The Qwen tokenizer turns that chat into numbers.
5. The model predicts the next token, then the next, up to 800 new tokens. Sampling is off (`do_sample=False`), so the same prompt tends to produce the same answer.
6. Those new tokens are turned back into text and shown on the page. The response name is `rizwan-code-model`.

The base model supplies the general coding ability. The adapter slightly shifts the answers toward the examples it was trained on.

## If the adapter is missing

`MODEL_BACKEND=auto` falls back to Ollama. `MODEL_BACKEND=finetuned` returns an error until `models/rizwan-code-model` is present.
