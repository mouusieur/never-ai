"""End-to-end tests for Never-AI v1.0 Omega: 15 Plugins, 10 Domains, Deep Think, Local RAG & Live Teaching."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from never_ai.data.curriculum import synthesize_bilingual_curriculum
from never_ai.data.omega_domains import OMEGA_DOMAIN_KNOWLEDGE
from never_ai.engine import NeverAIEngine
from never_ai.memory.evolvemem import EvolveMemStore
from never_ai.plugins.builtin import build_default_plugin_registry
import never_ai.server.app as server_app


def test_omega_curriculum_cleanliness_and_domains() -> None:
    assert len(OMEGA_DOMAIN_KNOWLEDGE) >= 35
    domains = {item["domain"] for item in OMEGA_DOMAIN_KNOWLEDGE}
    assert len(domains) == 10

    train_docs, val_docs = synthesize_bilingual_curriculum(seed=42, scale_factor=1)
    assert len(train_docs) > 250
    assert len(val_docs) > 15
    # Verify no visible answer prose contains raw <|think|> after </think>
    for item in OMEGA_DOMAIN_KNOWLEDGE:
        assert "<|think|>" not in item["a"]
        assert "<|think|>" not in item["thought"]


def test_all_15_sovereign_plugins_and_benchmark(tmp_path: Path) -> None:
    mem = EvolveMemStore(tmp_path / "mem.json")
    reg = build_default_plugin_registry(
        mem_store=mem,
        anatomy_provider=lambda: {"stage_name": "Test", "total_params": 1000},
        workspace_root=tmp_path,
    )
    plugins = reg.list_plugins()
    assert len(plugins) == 15

    # 1. SVG Plotter
    plot_res = reg.invoke("plot_svg_engine", expressions=["sin(x)", "0.5*x"], x_min=-5, x_max=5)
    assert plot_res["ok"] is True
    assert "<svg" in plot_res["svg"]
    assert len(plot_res["series"]) == 2

    # 2. Stats & OLS Regression
    st_res = reg.invoke("stats_and_data_engine", action="describe", data=[10, 12, 14, 15, 100])
    assert st_res["ok"] is True
    assert 100.0 in st_res["outliers_iqr"]
    reg_res = reg.invoke("stats_and_data_engine", action="regression", x=[1, 2, 3, 4], y=[3, 5, 7, 9])
    assert reg_res["ok"] is True
    assert abs(reg_res["slope"] - 2.0) < 1e-6
    assert abs(reg_res["intercept"] - 1.0) < 1e-6

    # 3. Physics & Units
    u_res = reg.invoke("unit_and_physics_engine", action="convert", value=36, from_unit="km/h", to_unit="m/s")
    assert u_res["ok"] is True
    assert abs(u_res["result"] - 10.0) < 1e-5
    bh_res = reg.invoke("unit_and_physics_engine", action="schwarzschild_radius", mass_kg=1.98847e30)
    assert bh_res["ok"] is True
    assert 2900 < bh_res["radius_m"] < 3000

    # 4. AST Security & Complexity Auditor
    ast_clean = reg.invoke(
        "code_ast_auditor",
        code='def add(a: int, b: int) -> int:\n    """Add two ints."""\n    return a + b\n',
    )
    assert ast_clean["ok"] is True
    assert ast_clean["security_clean"] is True
    assert ast_clean["quality_score"] >= 90

    ast_risky = reg.invoke("code_ast_auditor", code="x = eval('1+1')\n")
    assert ast_risky["ok"] is True
    assert ast_risky["security_clean"] is False

    # 5. Crypto & Entropy
    cr_res = reg.invoke("crypto_and_encoding_engine", action="entropy", text="AAAA")
    assert cr_res["ok"] is True
    assert cr_res["bits_per_char"] == 0.0

    # 6. Formal Logic & Truth Table
    lg_res = reg.invoke("logic_and_truth_table", expr="((P >> Q) & (Q >> R)) >> (P >> R)")
    assert lg_res["ok"] is True
    assert lg_res["is_tautology"] is True

    # 7. Live Benchmark Suite (9/9 domains)
    bench = reg.invoke("live_benchmark_suite")
    assert bench["ok"] is True
    assert bench["passed"] == bench["total"] == 9
    assert bench["score_pct"] == 100.0


def test_omega_server_rag_teaching_deep_think_and_routes(tmp_path: Path, monkeypatch) -> None:
    eng = NeverAIEngine(
        vault_dir=tmp_path / "vault",
        memory_path=tmp_path / "mem" / "evolvemem.json",
        telemetry_path=tmp_path / "logs" / "telemetry.jsonl",
        seq_len=96,
    )
    monkeypatch.setattr(server_app, "_ENGINE", eng)
    client = TestClient(server_app.app)

    # Status exposes 10 domains, 15 plugins, personas, version 1.0.0-omega
    st = client.get("/api/status").json()
    assert st["version"].startswith("1.")
    assert len(st["plugins"]) == 15
    assert len(st["domains"]) == 10
    assert len(st["personas"]) == 5

    # HEAD & GET installer work cleanly
    head_inst = client.head("/api/download/installer")
    assert head_inst.status_code == 200

    # Natural-language / slash SVG plot routing
    plot_chat = client.post(
        "/api/chat",
        json={"message": "/plot sin(x) * cos(x) sur [-4, 4]", "owner": "tester"},
    ).json()
    assert "<svg" in plot_chat["visible_answer"]
    assert plot_chat["artifacts"] and plot_chat["artifacts"][0]["type"] == "svg"

    # Document RAG upload + query + delete
    up = client.post(
        "/api/rag/upload",
        json={
            "filename": "architecture_secrete.md",
            "content": "Le protocole Hyperion-7 utilise un consensus quantique tolérant aux pannes byzantines sur 12 noeuds.",
            "owner": "tester",
        },
    ).json()
    assert up["ok"] is True
    assert up["chunks_indexed"] >= 1

    rag_ans = client.post(
        "/api/chat",
        json={"message": "/doc Que dit le fichier sur Hyperion-7 ?", "owner": "tester"},
    ).json()
    assert "Hyperion-7" in rag_ans["visible_answer"]
    assert "architecture_secrete.md" in rag_ans.get("citations", [])

    del_rag = client.delete("/api/rag/documents/architecture_secrete.md").json()
    assert del_rag["ok"] is True
    assert del_rag["rag_documents"] == []

    # Human Correction & Live Muon Teaching
    fb = client.post(
        "/api/chat/feedback",
        json={
            "prompt": "Quel est le nom de code du réacteur local ?",
            "corrected_answer": "Le réacteur local porte le nom de code **Aether-IX**.",
            "owner": "tester",
            "train_steps": 2,
        },
    ).json()
    assert fb["ok"] is True
    assert len(fb["losses"]) == 2

    recalled = client.post(
        "/api/chat",
        json={"message": "Quel est le nom de code du réacteur local ?", "owner": "tester"},
    ).json()
    assert "Aether-IX" in recalled["visible_answer"]

    # Deep Think + Persona + Full Benchmark
    dt_res = client.post(
        "/api/chat",
        json={
            "message": "Calcule 25 * 16 + 9",
            "deep_think": True,
            "persona": "mathematicien",
            "owner": "tester",
        },
    ).json()
    assert "409" in dt_res["visible_answer"]
    assert any("Étape 1" in t for t in dt_res["thoughts"])

    full_bench = client.post("/api/benchmark").json()
    assert full_bench["ok"] is True
    assert full_bench["symbolic_benchmark"]["score_pct"] == 100.0
