"""End-to-end tests for NeverAIEngine, Sovereign Auth, Threads & Observatory Server."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from never_ai.engine import NeverAIEngine
import never_ai.server.app as server_app


def test_engine_auth_threads_and_observatory_api(tmp_path: Path, monkeypatch) -> None:
    engine = NeverAIEngine(
        vault_dir=tmp_path / "ckpts",
        memory_path=tmp_path / "mem" / "evolvemem.json",
        telemetry_path=tmp_path / "logs" / "telemetry.jsonl",
        seq_len=64,
    )
    monkeypatch.setattr(server_app, "_ENGINE", engine)
    client = TestClient(server_app.app)

    # 1. X.com-inspired Monolith Landing HTML & Windows .ico load
    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "La pensée libre." in r_html.text

    r_ico = client.get("/favicon.ico")
    assert r_ico.status_code == 200

    # 2. Local Sovereign Auth (PBKDF2-SHA256 register + login + wrong passphrase check)
    r_reg = client.post(
        "/api/auth/register",
        json={"handle": "@Mouusieur", "display_name": "Mouusieur", "passphrase": "secret-local-key"},
    )
    assert r_reg.status_code == 200
    prof = r_reg.json()["profile"]
    assert prof["handle"] == "mouusieur"

    r_bad = client.post(
        "/api/auth/login",
        json={"handle": "mouusieur", "passphrase": "wrong"},
    )
    assert r_bad.status_code == 401

    r_ok = client.post(
        "/api/auth/login",
        json={"handle": "mouusieur", "passphrase": "secret-local-key"},
    )
    assert r_ok.status_code == 200

    # 3. Multi-Thread Conversation Creation + Neuro-Symbolic Math Verification
    r_th = client.post("/api/threads", json={"owner": "mouusieur", "title": "Nouvelle discussion"})
    assert r_th.status_code == 200
    tid = r_th.json()["thread"]["id"]

    r_chat = client.post(
        "/api/chat",
        json={
            "message": "Calcule 28 * 34 + 19",
            "max_new_tokens": 32,
            "temperature": 0.1,
            "thread_id": tid,
            "owner": "mouusieur",
        },
    )
    assert r_chat.status_code == 200
    chat_payload = r_chat.json()
    assert "971" in chat_payload["visible_answer"]
    assert chat_payload["thread_id"] == tid

    r_th_get = client.get(f"/api/threads/{tid}")
    assert r_th_get.status_code == 200
    assert len(r_th_get.json()["thread"]["messages"]) == 2

    # 4. Status, Train Burst, Zero-Shock Morphogenesis & SEAL Consolidation
    assert client.get("/api/status").status_code == 200
    assert client.post("/api/train/burst", json={"num_steps": 2, "batch_size": 4}).status_code == 200

    r_molt = client.post("/api/morphogenesis", json={"mode": "depth"})
    assert r_molt.status_code == 200
    assert r_molt.json()["max_abs_logit_diff"] < 1e-5

    r_seal = client.post(
        "/api/seal/consolidate",
        json={
            "title": "Identité Monolith Noir",
            "text": "Never-AI adopte une esthétique noire monolithique et tourne en local sur Windows.",
            "steps": 2,
        },
    )
    assert r_seal.status_code == 200
