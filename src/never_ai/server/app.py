"""FastAPI Live Observatory & Control Server for Never-AI (bound to 0.0.0.0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from never_ai.engine import NeverAIEngine


app = FastAPI(
    title="Never-AI Observatory (ChronoMorph-LM)",
    version="0.1.0",
)

# Allow all preview origins & iframe embedding cleanly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
_ENGINE: NeverAIEngine | None = None


def get_engine() -> NeverAIEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NeverAIEngine(seq_len=128)
    return _ENGINE


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=120, ge=8, le=320)
    temperature: float = Field(default=0.25, ge=0.0, le=1.5)


class TrainBurstRequest(BaseModel):
    num_steps: int = Field(default=8, ge=1, le=100)
    batch_size: int = Field(default=6, ge=2, le=16)


class MorphogenesisRequest(BaseModel):
    mode: str = Field(default="moe_double")


class SEALRequest(BaseModel):
    title: str = Field(..., min_length=2)
    text: str = Field(..., min_length=20)
    steps: int = Field(default=6, ge=1, le=30)


class PluginInvokeRequest(BaseModel):
    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = _STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return get_engine().status()


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> dict[str, Any]:
    return get_engine().chat(
        message=req.message,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )


@app.post("/api/train/burst")
def api_train_burst(req: TrainBurstRequest) -> dict[str, Any]:
    return get_engine().train_steps(
        num_steps=req.num_steps,
        batch_size=req.batch_size,
        snapshot_every=max(2, req.num_steps // 2),
        log_every=max(1, req.num_steps // 2),
    )


@app.post("/api/train/background/start")
def api_bg_start() -> dict[str, Any]:
    started = get_engine().start_background_growth_loop(batch_size=6)
    return {"ok": True, "started": started, "running": True}


@app.post("/api/train/background/stop")
def api_bg_stop() -> dict[str, Any]:
    get_engine().stop_background_growth_loop()
    return {"ok": True, "running": False}


@app.post("/api/morphogenesis")
def api_morphogenesis(req: MorphogenesisRequest) -> dict[str, Any]:
    try:
        return get_engine().trigger_morphogenesis(mode=req.mode)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/seal/consolidate")
def api_seal_consolidate(req: SEALRequest) -> dict[str, Any]:
    return get_engine().consolidate_knowledge_seal(
        title=req.title,
        text=req.text,
        steps=req.steps,
    )


@app.post("/api/plugins/invoke")
def api_plugin_invoke(req: PluginInvokeRequest) -> dict[str, Any]:
    return get_engine().registry.invoke(req.plugin, **req.params)
