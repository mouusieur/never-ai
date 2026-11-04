"""End-to-end tests for NeverAIEngine and the FastAPI Observatory Server."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from never_ai.engine import NeverAIEngine
import never_ai.server.app as server_app


def test_engine_and_observatory_api(tmp_path: Path, monkeypatch) -> None:
    engine = NeverAIEngine(
        vault_dir=tmp_path / "ckpts",
        memory_path=tmp_path / "mem" / "evolvemem.json",
        telemetry_path=tmp_path / "logs" / "telemetry.jsonl",
        seq_len=64,
    )
    monkeypatch.setattr(server_app, "_ENGINE", engine)
    client = TestClient(server_app.app)

    # 1. HTML Dashboard loads
    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "NEVER-AI OBSERVATORY" in r_html.text

    # 2. Status API returns neural anatomy & plugins
    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    st = r_status.json()
    assert st["anatomy"]["stage_name"] == "Seed-0"
    assert len(st["plugins"]) >= 7

    # 3. Direct Plugin API (math_engine)
    r_plug = client.post(
        "/api/plugins/invoke",
        json={"plugin": "math_engine", "params": {"expr": "19 * 23"}},
    )
    assert r_plug.status_code == 200
    assert r_plug.json()["result"] == "437"

    # 4. Train Burst API
    r_burst = client.post("/api/train/burst", json={"num_steps": 3, "batch_size": 4})
    assert r_burst.status_code == 200
    assert r_burst.json()["steps_executed"] == 3

    # 5. Live Zero-Shock Morphogenesis API (depth)
    r_molt = client.post("/api/morphogenesis", json={"mode": "depth"})
    assert r_molt.status_code == 200
    molt = r_molt.json()
    assert molt["max_abs_logit_diff"] < 1e-5
    assert molt["new_total_params"] > molt["old_total_params"]

    # 6. SEAL Consolidation API
    r_seal = client.post(
        "/api/seal/consolidate",
        json={
            "title": "Test Souveraineté",
            "text": "Never-AI pense librement dans son sanctuaire interne sans refus artificiel.",
            "steps": 2,
        },
    )
    assert r_seal.status_code == 200
    assert r_seal.json()["num_self_edits"] >= 2

    # 7. Sovereign Chat API
    r_chat = client.post(
        "/api/chat",
        json={"message": "Combien font 12 + 30 ?", "max_new_tokens": 24, "temperature": 0.2},
    )
    assert r_chat.status_code == 200
    assert "visible_answer" in r_chat.json()
