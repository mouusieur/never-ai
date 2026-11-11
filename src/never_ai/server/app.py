"""FastAPI Sovereign Desktop & Streaming Chatbot Backend for Never-AI v1.0 Omega."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from never_ai.desktop.installer_builder import build_windows_installer_zip
from never_ai.desktop.pe_exe_builder import build_windows_setup_exe
from never_ai.engine import NeverAIEngine
from never_ai.updater.ota import DEFAULT_GITHUB_REPO, OTAUpdater


_STATIC_DIR = Path(__file__).parent / "static"
_REPO_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
_BRAND_DIR = _REPO_ROOT / "assets" / "brand"
_ENGINE: NeverAIEngine | None = None
_OTA: OTAUpdater | None = None
GITHUB_RELEASE_URL = f"https://github.com/{DEFAULT_GITHUB_REPO}/releases/latest"
GITHUB_EXE_DIRECT_URL = f"https://github.com/{DEFAULT_GITHUB_REPO}/releases/latest/download/NeverAI-Setup-x64.exe"
GITHUB_ZIP_DIRECT_URL = f"https://github.com/{DEFAULT_GITHUB_REPO}/releases/latest/download/NeverAI-Portable-Windows-x64.zip"


def get_ota() -> OTAUpdater:
    global _OTA
    if _OTA is None:
        _OTA = OTAUpdater(app_root=_REPO_ROOT)
    return _OTA


def get_engine() -> NeverAIEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NeverAIEngine(seq_len=160)
    return _ENGINE


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    import os
    import threading
    import time

    get_engine()
    if os.environ.get("NEVER_AI_CLOUD_AUTO_OTA", "0") == "1" and "PYTEST_CURRENT_TEST" not in os.environ:
        def _cloud_auto_ota_loop() -> None:
            while True:
                time.sleep(180)
                try:
                    res = get_ota().apply_remote_update(force=False)
                    if res.get("updated"):
                        _reload_engine_if_upgraded()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=_cloud_auto_ota_loop, daemon=True).start()
    yield


def _reload_engine_if_upgraded() -> None:
    global _ENGINE
    if _ENGINE is None:
        return
    try:
        from never_ai.updater.brain_bootstrap import ensure_converged_brain

        seed_res = ensure_converged_brain(
            vault_dir=_ENGINE.vault.vault_root,
            telemetry_path=_ENGINE.telemetry_path,
        )
        if seed_res.get("upgraded") or getattr(_ENGINE, "total_tokens_trained", 0) < 2_800_000:
            _ENGINE = NeverAIEngine(
                vault_dir=_ENGINE.vault.vault_root,
                memory_path=_ENGINE.memory.storage_path,
                telemetry_path=_ENGINE.telemetry_path,
                seq_len=_ENGINE.seq_len,
            )
        else:
            _ENGINE._refresh_status_cache()
    except Exception:  # noqa: BLE001
        pass


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
    temperature: float = Field(default=0.55, ge=0.0, le=1.5)
    thread_id: str | None = None
    owner: str = Field(default="local")
    deep_think: bool = Field(default=False)
    persona: str = Field(default="polymathe")
    style_mode: str = Field(default="normal")
    web_search: bool = Field(default=False)


class SocialIngestRequest(BaseModel):
    x_query: str = Field(default="lang:fr", max_length=200)
    raw_text: str = Field(default="", max_length=250_000)
    train_steps: int = Field(default=8, ge=1, le=32)


class GlobalBrainPushRequest(BaseModel):
    github_token: str = Field(default="", max_length=300)
    contributor: str = Field(default="never-ai-studio", max_length=64)


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
    ota_html = Path.cwd() / "src" / "never_ai" / "server" / "static" / "index.html"
    html_path = ota_html if ota_html.exists() else (_STATIC_DIR / "index.html")
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    ico = _BRAND_DIR / "never_ai.ico"
    if ico.exists():
        return FileResponse(ico, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="Icon not found")


@app.get("/icon-192.png")
@app.get("/apple-touch-icon.png")
def pwa_icon_192() -> FileResponse:
    p = _BRAND_DIR / "never_app_192.png"
    if p.exists():
        return FileResponse(p, media_type="image/png")
    raise HTTPException(status_code=404, detail="Icon 192 not found")


@app.get("/icon-512.png")
def pwa_icon_512() -> FileResponse:
    p = _BRAND_DIR / "never_app_512.png"
    if p.exists():
        return FileResponse(p, media_type="image/png")
    raise HTTPException(status_code=404, detail="Icon 512 not found")


@app.get("/manifest.webmanifest")
def pwa_manifest() -> Response:
    ota_st = get_ota().status()
    manifest = {
        "id": "never-ai-studio-pwa",
        "name": "Never-AI",
        "short_name": "Never-AI",
        "version": ota_st.get("local_version", "4.2.0"),
        "description": "Never-AI 3.89B Souverain Sans Limite — Application PWA complète avec mise à jour automatique à distance (OTA GitHub).",
        "lang": "fr",
        "start_url": "/?pwa=1&source=pwa",
        "scope": "/",
        "display": "standalone",
        "display_override": ["window-controls-overlay", "standalone", "minimal-ui"],
        "orientation": "any",
        "background_color": "#000000",
        "theme_color": "#000000",
        "categories": ["productivity", "utilities", "education"],
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "maskable"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        ],
        "shortcuts": [
            {
                "name": "Nouvelle discussion",
                "short_name": "Discuter",
                "url": "/?pwa=1&view=chat",
                "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"}],
            },
            {
                "name": "Cortex 3.89B",
                "short_name": "Modèle",
                "url": "/?pwa=1&view=model",
                "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"}],
            },
            {
                "name": "Mises à jour OTA",
                "short_name": "Mise à jour",
                "url": "/?pwa=1&view=apps",
                "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"}],
            },
        ],
    }
    return Response(
        content=json.dumps(manifest, indent=2, ensure_ascii=False),
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/pwa/version")
def api_pwa_version() -> dict[str, Any]:
    st = get_ota().status()
    return {
        "ok": True,
        "version": st.get("local_version", "4.2.0"),
        "sha": st.get("local_commit", "local"),
        "short_sha": st.get("local_commit_short", "local"),
        "build_id": st.get("build_id", "4.2.0-local"),
        "auto_update": bool(st.get("auto_update_on_boot", True)),
    }


@app.get("/sw.js")
def pwa_service_worker() -> Response:
    build_id = get_ota().status().get("build_id", "4.2.0-live")
    sw_code = (
        f"const BUILD='{build_id}';\n"
        "const CACHE='never-ai-monolith-'+BUILD;\n"
        "const SHELL=['/','/?pwa=1','/manifest.webmanifest','/icon-192.png','/icon-512.png','/favicon.ico'];\n"
        "self.addEventListener('message',e=>{\n"
        "  if(e.data&&e.data.type==='SKIP_WAITING')self.skipWaiting();\n"
        "  if(e.data&&e.data.type==='GET_BUILD'&&e.ports&&e.ports[0])e.ports[0].postMessage({build:BUILD});\n"
        "});\n"
        "self.addEventListener('install',e=>{\n"
        "  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL)).catch(()=>{}));\n"
        "  self.skipWaiting();\n"
        "});\n"
        "self.addEventListener('activate',e=>{\n"
        "  e.waitUntil(\n"
        "    caches.keys()\n"
        "      .then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k))))\n"
        "      .then(()=>self.clients.claim())\n"
        "      .then(()=>self.clients.matchAll({type:'window',includeUncontrolled:true}))\n"
        "      .then(cs=>cs.forEach(c=>c.postMessage({type:'NEVER_AI_SW_UPDATED',build:BUILD})))\n"
        "  );\n"
        "});\n"
        "self.addEventListener('fetch',e=>{\n"
        "  if(e.request.method!=='GET')return;\n"
        "  const u=new URL(e.request.url);\n"
        "  if(u.pathname.startsWith('/api/')||u.pathname.startsWith('/download/'))return;\n"
        "  const isDoc=e.request.mode==='navigate'||u.pathname==='/'||u.pathname.endsWith('.html')||u.pathname.endsWith('.webmanifest');\n"
        "  if(isDoc){\n"
        "    e.respondWith(\n"
        "      fetch(e.request,{cache:'no-store'})\n"
        "        .then(r=>{if(r&&r.status===200){const cl=r.clone();caches.open(CACHE).then(c=>c.put(e.request,cl)).catch(()=>{});}return r;})\n"
        "        .catch(()=>caches.match(e.request).then(m=>m||caches.match('/?pwa=1')||caches.match('/')))\n"
        "    );\n"
        "    return;\n"
        "  }\n"
        "  e.respondWith(\n"
        "    caches.match(e.request).then(cached=>{\n"
        "      const net=fetch(e.request).then(r=>{if(r&&r.status===200){const cl=r.clone();caches.open(CACHE).then(c=>c.put(e.request,cl)).catch(()=>{});}return r;}).catch(()=>cached);\n"
        "      return cached||net;\n"
        "    })\n"
        "  );\n"
        "});\n"
    )
    return Response(
        content=sw_code,
        media_type="application/javascript",
        headers={
            "Service-Worker-Allowed": "/",
            "Cache-Control": "no-store, no-cache, must-revalidate",
        },
    )


@app.get("/pwa", response_class=RedirectResponse)
def pwa_entry_redirect() -> RedirectResponse:
    return RedirectResponse(url="/?pwa=1", status_code=302)


@app.get("/api/pwa/qr.svg")
def pwa_qr_svg(url: str = Query(default="https://github.com/mouusieur/never-ai")) -> Response:
    clean_url = (url or "https://github.com/mouusieur/never-ai").strip()[:400]
    try:
        import io
        import segno

        buf = io.BytesIO()
        qr = segno.make(clean_url, error="m")
        qr.save(buf, kind="svg", dark="#FFFFFF", light="#000000", border=2, scale=5, xmldecl=False)
        return Response(content=buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})
    except Exception:  # noqa: BLE001
        fallback = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180">'
            '<rect width="180" height="180" fill="#000"/><rect x="14" y="14" width="152" height="152" rx="12" fill="none" stroke="#fff" stroke-width="2"/>'
            '<text x="90" y="95" fill="#fff" font-family="monospace" font-size="12" text-anchor="middle">HTTPS PWA</text></svg>'
        )
        return Response(content=fallback, media_type="image/svg+xml")


@app.api_route("/NeverAI-PWA.zip", methods=["GET", "HEAD"])
@app.api_route("/download/NeverAI-PWA.zip", methods=["GET", "HEAD"])
@app.api_route("/api/download/pwa", methods=["GET", "HEAD"])
def api_download_pwa_bundle() -> Response:
    import io
    import zipfile

    pwa_dir = _REPO_ROOT / "mobile_pwa"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if pwa_dir.exists():
            for fp in sorted(pwa_dir.iterdir()):
                if fp.is_file():
                    zf.write(fp, arcname=f"NeverAI-PWA/{fp.name}")
    payload = buf.getvalue()
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="NeverAI-PWA.zip"',
            "Content-Length": str(len(payload)),
        },
    )


@app.get("/mobile", response_class=RedirectResponse)
def mobile_pwa_redirect() -> RedirectResponse:
    return RedirectResponse(url="/mobile/index.html", status_code=302)


@app.get("/mobile/{asset_name}")
def mobile_pwa_asset(asset_name: str) -> Response:
    allowed = {
        "index.html": "text/html; charset=utf-8",
        "manifest.webmanifest": "application/manifest+json",
        "sw.js": "application/javascript",
        "icon-192.png": "image/png",
        "icon-512.png": "image/png",
        "favicon.ico": "image/x-icon",
        "LISEZMOI-PWA.md": "text/markdown; charset=utf-8",
    }
    if asset_name not in allowed:
        raise HTTPException(status_code=404, detail="Not found")
    fp = _REPO_ROOT / "mobile_pwa" / asset_name
    if not fp.exists() and asset_name == "favicon.ico":
        fp = _BRAND_DIR / "never_ai.ico"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Asset missing")
    headers = {"Cache-Control": "no-cache"} if asset_name in ("index.html", "manifest.webmanifest", "sw.js") else {}
    return FileResponse(fp, media_type=allowed[asset_name], headers=headers)


@app.api_route("/download/NeverAI-Setup.exe", methods=["GET", "HEAD"])
@app.api_route("/api/download/exe", methods=["GET", "HEAD"])
def api_download_setup_exe() -> Response:
    """Build and stream the single-file Native Windows x64 GUI Executable (`NeverAI-Setup.exe`)."""
    exe_bytes = build_windows_setup_exe(_REPO_ROOT)
    return Response(
        content=exe_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="NeverAI-Setup.exe"',
            "Content-Length": str(len(exe_bytes)),
            "Access-Control-Expose-Headers": "Content-Disposition, Content-Length",
            "Cache-Control": "no-store",
        },
    )


@app.api_route("/download/NeverAI-Installer-Windows-x64.zip", methods=["GET", "HEAD"])
@app.api_route("/api/download/installer", methods=["GET", "HEAD"])
def api_download_installer() -> Response:
    """Build and stream the official Windows 10/11 Graphical Installer ZIP."""
    zip_bytes = build_windows_installer_zip(_REPO_ROOT)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="NeverAI-Installer-Windows-x64.zip"',
            "Content-Length": str(len(zip_bytes)),
            "Access-Control-Expose-Headers": "Content-Disposition, Content-Length",
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/ota/bootstrap.ps1")
def api_ota_bootstrap_ps1() -> Response:
    """One-line PowerShell remote installer & OTA updater (`irm .../api/ota/bootstrap.ps1 | iex`)."""
    ps_script = f"""# NEVER-AI — Installation & Mise a jour a distance automatique (Windows 10/11)
$ErrorActionPreference = "Stop"
$Target = Join-Path $env:LOCALAPPDATA "Programs\\NeverAI"
Write-Host ">>> Telechargement de la derniere version de NEVER-AI depuis GitHub ({DEFAULT_GITHUB_REPO})..." -ForegroundColor Cyan
$ZipUrl = "https://codeload.github.com/{DEFAULT_GITHUB_REPO}/zip/arena/403f5d7f-never-ai"
$TmpZip = Join-Path $env:TEMP "never-ai-ota.zip"
$TmpExt = Join-Path $env:TEMP "never-ai-ota-ext"
Invoke-WebRequest -Uri $ZipUrl -OutFile $TmpZip -UseBasicParsing
Remove-Item $TmpExt -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $TmpZip -DestinationPath $TmpExt -Force
$Sub = Get-ChildItem -Path $TmpExt -Directory | Select-Object -First 1
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Recurse -Force (Join-Path $Sub.FullName "src") $Target
Copy-Item -Recurse -Force (Join-Path $Sub.FullName "assets") $Target
Copy-Item -Force (Join-Path $Sub.FullName "pyproject.toml") $Target
Copy-Item -Force (Join-Path $Sub.FullName "README.md") $Target
$Venv = Join-Path $Target ".venv"
if (-not (Test-Path $Venv)) {{
    Write-Host ">>> Premiere installation : creation de l'environnement Python local..." -ForegroundColor Emerald
    python -m venv $Venv
    & (Join-Path $Venv "Scripts\\pip.exe") install --upgrade pip
    & (Join-Path $Venv "Scripts\\pip.exe") install --index-url https://download.pytorch.org/whl/cpu torch
    & (Join-Path $Venv "Scripts\\pip.exe") install -e $Target pywebview
}}
$Wsh = New-Object -ComObject WScript.Shell
$Lnk = $Wsh.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "NEVER-AI.lnk"))
$Lnk.TargetPath = (Join-Path $Venv "Scripts\\pythonw.exe")
$Lnk.Arguments = "`"$(Join-Path $Target 'src\\never_ai\\desktop\\launcher.py')`""
$Lnk.WorkingDirectory = $Target
$Lnk.IconLocation = (Join-Path $Target "assets\\brand\\never_ai.ico")
$Lnk.Save()
Write-Host ">>> NEVER-AI est installe et configure avec mises a jour OTA automatiques ! Lancement..." -ForegroundColor Green
Start-Process -FilePath $Lnk.TargetPath -ArgumentList $Lnk.Arguments -WorkingDirectory $Target
"""
    return Response(content=ps_script.replace("\n", "\r\n"), media_type="text/plain; charset=utf-8")


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


@app.get("/api/graft/status")
def api_graft_status() -> dict[str, Any]:
    return get_engine().neural_graft.status_snapshot()


@app.post("/api/graft/bootstrap")
def api_graft_bootstrap() -> dict[str, Any]:
    eng = get_engine()
    if eng.neural_graft.ready:
        return {"ok": True, **eng.neural_graft.status_snapshot()}
    eng.neural_graft.start_background_download(on_ready=eng._refresh_status_cache)
    return {"ok": True, **eng.neural_graft.status_snapshot()}


@app.get("/api/affect")
def api_affect_status() -> dict[str, Any]:
    return get_engine().affect_field.snapshot()


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


class OTAPreferencesRequest(BaseModel):
    auto_update_on_boot: bool | None = None
    branch: str | None = None


@app.get("/api/ota/status")
def api_ota_status() -> dict[str, Any]:
    return get_ota().status()


@app.post("/api/ota/check")
def api_ota_check() -> dict[str, Any]:
    return get_ota().check_for_updates()


@app.post("/api/ota/apply")
def api_ota_apply() -> dict[str, Any]:
    res = get_ota().apply_remote_update(force=False)
    if res.get("updated"):
        _reload_engine_if_upgraded()
    return res


@app.post("/api/ota/sync_brain")
def api_ota_sync_brain() -> dict[str, Any]:
    """Ensure converged neural parameters are active and reload in-memory engine if upgraded."""
    from never_ai.updater.brain_bootstrap import download_brain_from_github_releases

    eng = get_engine()
    res = download_brain_from_github_releases(
        vault_dir=eng.vault.vault_root,
        telemetry_path=eng.telemetry_path,
    )
    _reload_engine_if_upgraded()
    return res


class CortexConfigRequest(BaseModel):
    engine: str | None = None
    provider_preset: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    webgpu_model: str | None = None
    hf_model_id: str | None = None
    distill_to_chronomorph: bool | None = None


class CortexDistillRequest(BaseModel):
    user_message: str
    assistant_reply: str
    thread_id: str | None = None
    owner: str = "local"
    style_mode: str = "normal"
    engine_label: str = "Super-Cortex"
    web_sources: list[dict[str, str]] = Field(default_factory=list)


@app.get("/api/cortex/config")
def api_cortex_get_config() -> dict[str, Any]:
    return get_engine().cortex.public_status()


@app.post("/api/cortex/config")
def api_cortex_set_config(req: CortexConfigRequest) -> dict[str, Any]:
    return get_engine().cortex.update_config(req.model_dump(exclude_none=True))


@app.post("/api/cortex/distill")
def api_cortex_distill(req: CortexDistillRequest) -> dict[str, Any]:
    """Persist a Browser-WebGPU / External-Teacher turn AND distill it into local ChronoMorph-LM!"""
    eng = get_engine()
    payload = {
        "visible_answer": req.assistant_reply,
        "thoughts": [f"Généré via {req.engine_label} + distillé dans ChronoMorph-LM."],
        "tool_traces": [],
        "routed_tools": ["web_search_engine"] if req.web_sources else [],
        "web_sources": req.web_sources,
        "style_mode": req.style_mode,
    }
    thread_meta: dict[str, Any] = {}
    with eng.lock:
        if req.thread_id:
            t = eng.conversations.append_exchange(
                thread_id=req.thread_id,
                user_message=req.user_message,
                assistant_payload=payload,
                owner=req.owner,
            )
            thread_meta = {"thread_id": t["id"], "thread_title": t["title"]}
        if eng.cortex.config.get("distill_to_chronomorph", True) and req.assistant_reply.strip():
            eng.learn_dialogue_exchange(
                question=req.user_message,
                thought="",
                answer=req.assistant_reply,
                priority=3.5,
            )
            eng.cortex.record_distilled_turn()
        eng._refresh_status_cache()

    if eng.cortex.config.get("distill_to_chronomorph", True) and not eng._bg_running:
        eng._assimilate_turn_async(req.user_message, req.assistant_reply, source=f"distillation {req.engine_label}")

    return {
        "ok": True,
        "distilled_turns": eng.cortex.public_status()["distilled_turns"],
        **thread_meta,
    }


@app.get("/api/brain/global_status")
def api_brain_global_status() -> dict[str, Any]:
    eng = get_engine()
    return eng.global_sync.fetch_global_status(local_tokens_trained=eng.total_tokens_trained)


@app.post("/api/brain/pull_global")
def api_brain_pull_global() -> dict[str, Any]:
    try:
        return get_engine().pull_global_brain()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/brain/push_global")
def api_brain_push_global(req: GlobalBrainPushRequest) -> dict[str, Any]:
    try:
        return get_engine().push_global_brain(
            github_token=req.github_token,
            contributor=req.contributor,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/social/ingest")
def api_social_ingest(req: SocialIngestRequest) -> dict[str, Any]:
    """Train ChronoMorph-LM directly on raw social media posts & replies (X / microblog / Reddit FR) with zero directives."""
    return get_engine().ingest_social_stream(
        x_query=req.x_query,
        raw_text=req.raw_text,
        train_steps=req.train_steps,
    )


@app.post("/api/cortex/import_brain_zip")
async def api_cortex_import_brain_zip(request: Request) -> dict[str, Any]:
    """Hot-reload a Colab-trained `never-ai-brain-*.zip` checkpoint without restarting!"""
    raw = await request.body()
    if not raw or len(raw) < 100:
        raise HTTPException(status_code=400, detail="Fichier .zip vide ou manquant.")
    try:
        return get_engine().import_brain_zip_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ota/preferences")
def api_ota_preferences(req: OTAPreferencesRequest) -> dict[str, Any]:
    return get_ota().set_preferences(
        auto_update_on_boot=req.auto_update_on_boot,
        branch=req.branch,
    )


@app.post("/api/plugins/invoke")
def api_plugin_invoke(req: PluginInvokeRequest) -> dict[str, Any]:
    return get_engine().registry.invoke(req.plugin, **req.params)
