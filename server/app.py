from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference.model_router import adapter_exists, chat, get_backend, model_info, preload_model, resolve_adapter_path
from inference.prompts import truncate_history

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    backend = get_backend()
    if backend == "finetuned":
        if not adapter_exists():
            print(
                f"Warning: MODEL_BACKEND expects a fine-tuned adapter at {resolve_adapter_path()}, "
                "but it was not found. Chat requests will return 503 until the adapter is present "
                "or MODEL_BACKEND=auto/ollama is set."
            )
        else:
            try:
                preload_model()
                print("Fine-tuned model loaded successfully.")
            except Exception as exc:
                print(f"Warning: model preload failed: {exc}")
    else:
        print(f"Using backend '{backend}'. Adapter ready: {adapter_exists()}")
    yield


app = FastAPI(title="Coding Model Chat", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model: str = ""
    history: List[ChatMessage] = Field(default_factory=list)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "backend": get_backend(),
        "adapter_ready": str(adapter_exists()).lower(),
    }


@app.get("/api/model")
def model_info_endpoint() -> Dict[str, str]:
    info = model_info()
    info["backend"] = get_backend()
    info["adapter_ready"] = str(adapter_exists()).lower()
    return info


@app.post("/api/chat")
def chat_endpoint(payload: ChatRequest) -> Dict[str, object]:
    try:
        history = truncate_history([message.model_dump() for message in payload.history])
        ollama_model = payload.model.strip() or None
        # Ignore HF display names that are not valid Ollama tags.
        if ollama_model and ("/" in ollama_model or ollama_model.startswith("my-coding-model")):
            ollama_model = None
        return chat(
            message=payload.message,
            model=ollama_model,
            history=history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
