from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference.model_router import chat, get_backend, model_info, preload_model

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Coding Model Chat", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model: str = ""
    history: List[ChatMessage] = Field(default_factory=list)


@app.on_event("startup")
def startup_load_model() -> None:
    if get_backend() != "finetuned":
        return
    try:
        preload_model()
        print("Fine-tuned model loaded successfully.")
    except Exception as exc:
        print(f"Warning: model preload skipped: {exc}")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "backend": get_backend()}


@app.get("/api/model")
def model_info_endpoint() -> Dict[str, str]:
    info = model_info()
    info["backend"] = get_backend()
    return info


@app.post("/api/chat")
def chat_endpoint(payload: ChatRequest) -> Dict[str, object]:
    try:
        history = [message.model_dump() for message in payload.history]
        return chat(
            message=payload.message,
            model=payload.model or None,
            history=history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
