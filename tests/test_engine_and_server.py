"""End-to-end tests for NeverAIEngine, Continual Growth, SSE Chatbot & Windows Installer."""

from __future__ import annotations

import io
from pathlib import Path
import time
import zipfile
from fastapi.testclient import TestClient

from never_ai.engine import NeverAIEngine
import never_ai.server.app as server_app


def test_engine_continual_growth_streaming_memory_and_installer(tmp_path: Path, monkeypatch) -> None:
    engine = NeverAIEngine(
        vault_dir=tmp_path / "ckpts",
        memory_path=tmp_path / "mem" / "evolvemem.json",
        telemetry_path=tmp_path / "logs" / "telemetry.jsonl",
        seq_len=64,
    )
    monkeypatch.setattr(server_app, "_ENGINE", engine)
    client = TestClient(server_app.app)

    # 1. Landing HTML, Windows .ico & Downloadable Windows Installer ZIP
    assert client.get("/").status_code == 200
    assert client.get("/favicon.ico").status_code == 200

    r_zip = client.get("/api/download/installer")
    assert r_zip.status_code == 200
    assert r_zip.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r_zip.content), "r") as zf:
        names = set(zf.namelist())
        assert "Installer-NeverAI.bat" in names
        assert "setup_wizard.ps1" in names
        assert "LISEZ-MOI.txt" in names
        assert "never-ai-src/assets/brand/never_ai.ico" in names

    # 2. Local Sovereign Auth
    r_reg = client.post(
        "/api/auth/register",
        json={"handle": "@Mouusieur", "display_name": "Mouusieur", "passphrase": "secret-local-key"},
    )
    assert r_reg.status_code == 200

    # 3. Episodic Personal Memory ("Je m'appelle..." -> "Comment je m'appelle ?")
    r_intro = client.post(
        "/api/chat",
        json={"message": "Je m'appelle Lucas", "owner": "mouusieur"},
    )
    assert r_intro.status_code == 200
    assert "Lucas" in r_intro.json()["visible_answer"]

    r_recall = client.post(
        "/api/chat",
        json={"message": "Comment je m'appelle ?", "owner": "mouusieur"},
    )
    assert r_recall.status_code == 200
    assert "Lucas" in r_recall.json()["visible_answer"]

    # 4. Real-Time SSE Streaming Chatbot Endpoint
    r_sse = client.post(
        "/api/chat/stream",
        json={"message": "Calcule 28 * 34 + 19", "max_new_tokens": 32, "owner": "mouusieur"},
    )
    assert r_sse.status_code == 200
    assert "971" in r_sse.text
    assert '"type": "done"' in r_sse.text

    # 5. Non-blocking Continuous Background Growth Verification!
    r_bg_start = client.post(
        "/api/train/background/start",
        json={"auto_morphogenesis": True, "batch_size": 2},
    )
    assert r_bg_start.status_code == 200
    assert r_bg_start.json()["running"] is True

    # Wait briefly and verify lock-free /api/status responds immediately with live steps!
    deadline = time.time() + 6.0
    st = {}
    while time.time() < deadline:
        t0 = time.perf_counter()
        r_st = client.get("/api/status")
        assert (time.perf_counter() - t0) < 0.25  # Must NEVER block!
        st = r_st.json()
        if st["bg_steps_session"] >= 2:
            break
        time.sleep(0.2)

    assert st["background_training"] is True
    assert st["bg_steps_session"] >= 2
    assert st["total_tokens_trained"] > 0

    r_bg_stop = client.post("/api/train/background/stop")
    assert r_bg_stop.status_code == 200
    assert client.get("/api/status").json()["background_training"] is False
