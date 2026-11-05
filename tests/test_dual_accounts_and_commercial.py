"""Tests for Never-AI v1.2 Dual-Horizon: Commercial User vs R&D Lab Accounts, Live Learning Curve & Comparator."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from never_ai.engine import NeverAIEngine
import never_ai.server.app as server_app


def _make_isolated_client(tmp_path: Path, monkeypatch) -> TestClient:
    engine = NeverAIEngine(
        vault_dir=tmp_path / "ckpts",
        memory_path=tmp_path / "mem" / "evolvemem.json",
        telemetry_path=tmp_path / "logs" / "telemetry.jsonl",
        seq_len=64,
    )
    monkeypatch.setattr(server_app, "_ENGINE", engine)
    return TestClient(server_app.app)


def test_dual_account_registration_guest_and_live_switch(tmp_path: Path, monkeypatch) -> None:
    client = _make_isolated_client(tmp_path, monkeypatch)

    # 1. Guest User Account (Commercial)
    r_guest_user = client.post("/api/auth/guest", json={"account_type": "user"})
    assert r_guest_user.status_code == 200
    prof_u = r_guest_user.json()["profile"]
    assert prof_u["account_type"] == "user"

    # 2. Guest Lab Account (R&D)
    r_guest_lab = client.post("/api/auth/guest", json={"account_type": "lab"})
    assert r_guest_lab.status_code == 200
    assert r_guest_lab.json()["profile"]["account_type"] == "lab"

    # 3. Register Commercial User Account
    r_reg_u = client.post(
        "/api/auth/register",
        json={
            "handle": "alice_pro",
            "display_name": "Alice",
            "passphrase": "secret",
            "account_type": "user",
        },
    )
    assert r_reg_u.status_code == 200
    alice = r_reg_u.json()["profile"]
    assert alice["account_type"] == "user"

    # 4. Register R&D Lab Account
    r_reg_lab = client.post(
        "/api/auth/register",
        json={
            "handle": "dr_chrono",
            "display_name": "Dr Chrono",
            "passphrase": "secret",
            "account_type": "lab",
        },
    )
    assert r_reg_lab.status_code == 200
    assert r_reg_lab.json()["profile"]["account_type"] == "lab"

    # 5. Live switch Alice from User -> Lab -> User and verify /api/auth/me persistence
    r_sw = client.post(
        "/api/auth/account-type",
        json={"token": alice["token"], "account_type": "lab"},
    )
    assert r_sw.status_code == 200
    assert r_sw.json()["profile"]["account_type"] == "lab"

    r_me = client.get(f"/api/auth/me?token={alice['token']}")
    assert r_me.status_code == 200
    assert r_me.json()["profile"]["account_type"] == "lab"


def test_commercial_overview_live_curve_and_quick_benchmark(tmp_path: Path, monkeypatch) -> None:
    client = _make_isolated_client(tmp_path, monkeypatch)

    # Initial overview
    r_ov1 = client.get("/api/commercial/overview")
    assert r_ov1.status_code == 200
    ov1 = r_ov1.json()
    assert 0.0 <= ov1["mastery_pct"] <= 99.5
    assert len(ov1["learning_curve"]) >= 2
    assert ov1["learning_curve"][0]["mastery"] == 0.0
    assert len(ov1["benchmark_comparator"]) == 6
    assert len(ov1["evolution_milestones"]) >= 2

    # Run a live training burst and verify the live learning curve grows dynamically
    r_burst = client.post("/api/train/burst", json={"num_steps": 3, "batch_size": 2})
    assert r_burst.status_code == 200

    r_ov2 = client.get("/api/commercial/overview")
    ov2 = r_ov2.json()
    assert len(ov2["learning_curve"]) >= len(ov1["learning_curve"])
    assert ov2["words_assimilated"] > ov1["words_assimilated"]

    # Consumer Quick-Benchmark (9 plain-French live checks)
    r_qb = client.post("/api/commercial/quick-benchmark")
    assert r_qb.status_code == 200
    qb = r_qb.json()
    assert qb["total"] == 9
    assert qb["passed"] == 9
    assert qb["score_pct"] == 100.0
    assert len(qb["cards"]) == 9
