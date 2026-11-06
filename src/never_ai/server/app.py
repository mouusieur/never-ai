"""FastAPI Sovereign Desktop & Streaming Chatbot Backend for Never-AI v1.0 Omega."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from never_ai.desktop.installer_builder import build_windows_installer_zip
from never_ai.engine import NeverAIEngine


_STATIC_DIR = Path(__file__).parent / "static"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRAND_DIR = _REPO_ROOT / "assets" / "brand"
_ENGINE: NeverAIEngine | None = None


def get_engine() -> NeverAIEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NeverAIEngine(seq_len=160)
    return _ENGINE


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_engine()
    yield


app = FastAPI(
    title="Never-AI v1.0 Omega — Sovereign Chatbot & Desktop Studio (ChronoMorph-LM)",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthRegisterRequest(BaseModel):
    handle: str = Field(..., min_length=2, max_length=32)
    display_name: str = Field(default="", max_length=50)
    passphrase: str = Field(default="", max_length=128)
    account_type: str = Field(default="user")


class AuthLoginRequest(BaseModel):
    handle: str = Field(..., min_length=2, max_length=32)
    passphrase: str = Field(default="", max_length=128)


class AuthGuestRequest(BaseModel):
    account_type: str = Field(default="user")


class AccountTypeSwitchRequest(BaseModel):
    token: str = Field(..., min_length=4)
    account_type: str = Field(..., pattern="^(user|lab)$")


class ThreadCreateRequest(BaseModel):
    owner: str = Field(default="local")
    title: str = Field(default="Nouvelle discussion")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=160, ge=8, le=360)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    thread_id: str | None = None
    owner: str = Field(default="local")
    deep_think: bool = Field(default=False)
    persona: str = Field(default="polymathe")
    style_mode: str = Field(default="normal")
    web_search: bool = Field(default=False)


class WebSearchDirectRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300)
    max_results: int = Field(default=6, ge=1, le=10)


class TeachExchangeRequest(BaseModel):
    question: str = Field(..., min_length=2)
    thought: str = Field(default="Consolidation d'un échange validé par l'utilisateur.")
    answer: str = Field(..., min_length=2)


class FeedbackCorrectionRequest(BaseModel):
    prompt: str = Field(..., min_length=2)
    corrected_answer: str = Field(..., min_length=2)
    thought: str = Field(default="")
    owner: str = Field(default="local")
    train_steps: int = Field(default=3, ge=1, le=12)


class RAGUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=120)
    content: str = Field(..., min_length=1)
    owner: str = Field(default="local")


class BackgroundStartRequest(BaseModel):
    auto_morphogenesis: bool = Field(default=True)
    batch_size: int = Field(default=2, ge=1, le=8)


class TrainBurstRequest(BaseModel):
    num_steps: int = Field(default=6, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=16)


class MorphogenesisRequest(BaseModel):
    mode: str = Field(default="moe_double")


class SEALRequest(BaseModel):
    title: str = Field(..., min_length=2)
    text: str = Field(..., min_length=20)
    steps: int = Field(default=5, ge=1, le=30)


class PluginInvokeRequest(BaseModel):
    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = _STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    ico = _BRAND_DIR / "never_ai.ico"
    if ico.exists():
        return FileResponse(ico, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="Icon not found")


@app.api_route("/api/download/installer", methods=["GET", "HEAD"])
def api_download_installer() -> Response:
    """Build and stream the official Windows 10/11 Graphical Installer ZIP."""
    zip_bytes = build_windows_installer_zip(_REPO_ROOT)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="NeverAI-Installer-Windows-x64.zip"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/auth/register")
def api_auth_register(req: AuthRegisterRequest) -> dict[str, Any]:
    try:
        profile = get_engine().identities.register(
            handle=req.handle,
            display_name=req.display_name,
            passphrase=req.passphrase,
            account_type=req.account_type,
        )
        get_engine().memory.store(
            key=f"user_profile_{profile['handle']}",
            content=f"Utilisateur souverain @{profile['handle']} ({profile['display_name']}, mode={profile['account_type']}) sur Never-AI Desktop.",
            category="episodic",
            tags=["user", profile["handle"], profile["account_type"]],
        )
        return {"ok": True, "profile": profile}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/auth/login")
def api_auth_login(req: AuthLoginRequest) -> dict[str, Any]:
    try:
        profile = get_engine().identities.login(handle=req.handle, passphrase=req.passphrase)
        return {"ok": True, "profile": profile}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/api/auth/guest")
def api_auth_guest(req: AuthGuestRequest | None = None) -> dict[str, Any]:
    acc = req.account_type if req is not None else "user"
    profile = get_engine().identities.guest_session(account_type=acc)
    return {"ok": True, "profile": profile}


@app.post("/api/auth/account-type")
def api_auth_set_account_type(req: AccountTypeSwitchRequest) -> dict[str, Any]:
    try:
        profile = get_engine().identities.set_account_type(req.token, req.account_type)
        return {"ok": True, "profile": profile}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/api/auth/me")
def api_auth_me(token: str = Query(default="")) -> dict[str, Any]:
    profile = get_engine().identities.resolve_token(token)
    if not profile:
        return {"authenticated": False, "profile": None}
    return {"authenticated": True, "profile": profile}


@app.get("/api/threads")
def api_list_threads(owner: str = Query(default="local")) -> dict[str, Any]:
    return {"threads": get_engine().conversations.list_threads(owner=owner)}


@app.post("/api/threads")
def api_create_thread(req: ThreadCreateRequest) -> dict[str, Any]:
    t = get_engine().conversations.create_thread(owner=req.owner, title=req.title)
    return {"ok": True, "thread": t}


@app.get("/api/threads/{thread_id}")
def api_get_thread(thread_id: str) -> dict[str, Any]:
    t = get_engine().conversations.get_thread(thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    return {"thread": t}


@app.delete("/api/threads/{thread_id}")
def api_delete_thread(thread_id: str) -> dict[str, Any]:
    deleted = get_engine().conversations.delete_thread(thread_id)
    return {"ok": deleted}


@app.get("/api/threads/{thread_id}/export")
def api_export_thread(thread_id: str) -> Response:
    t = get_engine().conversations.get_thread(thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    lines = [f"# {t['title']}\n", f"_Export Never-AI v1.0 Omega · @{t.get('owner', 'local')}_\n"]
    for m in t.get("messages", []):
        role_lbl = "Utilisateur" if m.get("role") == "user" else "Never-AI"
        lines.append(f"## {role_lbl}\n")
        if m.get("thoughts"):
            th_txt = "\n".join(f"> {th}" for th in m["thoughts"])
            lines.append(f"**Pensée interne `<|think|>` :**\n{th_txt}\n")
        lines.append(f"{m.get('content', '')}\n")
    md_bytes = "\n".join(lines).encode("utf-8")
    return Response(
        content=md_bytes,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="never-ai-{thread_id}.md"'},
    )


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return get_engine().status()


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> dict[str, Any]:
    return get_engine().chat(
        message=req.message,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        thread_id=req.thread_id,
        owner=req.owner,
        deep_think=req.deep_think,
        persona=req.persona,
        style_mode=req.style_mode,
        web_search=req.web_search,
    )


@app.post("/api/chat/stream")
def api_chat_stream(req: ChatRequest) -> StreamingResponse:
    def _sse_gen() -> Iterator[str]:
        for ev in get_engine().chat_stream(
            message=req.message,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            thread_id=req.thread_id,
            owner=req.owner,
            deep_think=req.deep_think,
            persona=req.persona,
            style_mode=req.style_mode,
            web_search=req.web_search,
        ):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/web/search")
def api_web_search(req: WebSearchDirectRequest) -> dict[str, Any]:
    return get_engine().registry.invoke("web_search_engine", query=req.query, max_results=req.max_results)


@app.post("/api/chat/feedback")
def api_chat_feedback(req: FeedbackCorrectionRequest) -> dict[str, Any]:
    return get_engine().teach_correction(
        prompt=req.prompt,
        corrected_answer=req.corrected_answer,
        thought=req.thought,
        owner=req.owner,
        train_steps=req.train_steps,
    )


@app.get("/api/rag/documents")
def api_rag_list() -> dict[str, Any]:
    return {"documents": get_engine().list_rag_documents()}


@app.post("/api/rag/upload")
def api_rag_upload(req: RAGUploadRequest) -> dict[str, Any]:
    return get_engine().ingest_document_rag(
        filename=req.filename,
        content=req.content,
        owner=req.owner,
    )


@app.delete("/api/rag/documents/{filename}")
def api_rag_delete(filename: str) -> dict[str, Any]:
    return get_engine().clear_rag_document(filename=filename)


@app.post("/api/benchmark")
def api_benchmark() -> dict[str, Any]:
    return get_engine().run_full_benchmark()


@app.get("/api/commercial/overview")
def api_commercial_overview() -> dict[str, Any]:
    return get_engine().commercial_overview()


@app.post("/api/commercial/quick-benchmark")
def api_commercial_quick_benchmark() -> dict[str, Any]:
    return get_engine().run_commercial_quick_benchmark()


@app.post("/api/chat/teach")
def api_chat_teach(req: TeachExchangeRequest) -> dict[str, Any]:
    return get_engine().learn_dialogue_exchange(
        question=req.question,
        thought=req.thought,
        answer=req.answer,
        priority=3.0,
    )


@app.post("/api/train/burst")
def api_train_burst(req: TrainBurstRequest) -> dict[str, Any]:
    return get_engine().train_steps(
        num_steps=req.num_steps,
        batch_size=req.batch_size,
        snapshot_every=max(2, req.num_steps // 2),
        log_every=1,
    )


@app.post("/api/train/background/start")
def api_bg_start(req: BackgroundStartRequest | None = None) -> dict[str, Any]:
    auto_m = req.auto_morphogenesis if req is not None else True
    b_sz = req.batch_size if req is not None else 2
    started = get_engine().start_background_growth_loop(
        batch_size=b_sz,
        auto_morphogenesis=auto_m,
    )
    return {"ok": True, "started": started, "running": True, "auto_morphogenesis": auto_m}


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
