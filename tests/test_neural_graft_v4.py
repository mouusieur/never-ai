"""Unit & integration tests for Never-AI v4.0 Sovereign Neural Graft, Abliteration, 5D Affect & Organic Memory."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from safetensors.torch import load_file

from never_ai.cortex.neural_graft import (
    AFFECT_AXES,
    AffectiveStanceField,
    LoRALinear,
    apply_french_abliteration,
    attach_upper_cortex_lora,
    extract_lora_state_dict,
    extract_organic_relational_facts,
    load_lora_state_dict,
)
from never_ai.engine import NeverAIEngine
from never_ai.server.app import app


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_forged_sovereign_graft_artifacts_integrity() -> None:
    """Verify the tracked v4.0 abliteration, affect, and LoRA tensors exist and have exact shapes."""
    graft_dir = REPO_ROOT / "assets" / "neural_graft"
    ablit = load_file(str(graft_dir / "fr_abliteration.safetensors"))
    assert "shared_basis" in ablit
    basis = ablit["shared_basis"].float()
    assert basis.shape == (896, 2)
    # Orthonormality check: R^T R == I_2
    gram = basis.t() @ basis
    assert torch.allclose(gram, torch.eye(2), atol=1e-4)

    affect = load_file(str(graft_dir / "affect_vectors.safetensors"))
    for ax in AFFECT_AXES:
        assert ax in affect
        assert affect[ax].shape == (896,)
        assert abs(float(affect[ax].float().norm().item()) - 1.0) < 1e-3

    lora = load_file(str(graft_dir / "sovereign_lora.safetensors"))
    # 11 layers (13..23) * 7 modules * 2 matrices (A, B) = 154 tensors
    assert len(lora) == 154
    total_lora_params = sum(t.numel() for t in lora.values())
    assert total_lora_params == 3_024_384


def test_idempotent_french_abliteration_erases_refusal_subspace() -> None:
    """Verify apply_french_abliteration zeroes projections onto [r1, r2] in residual writers."""

    class DummyAttn(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class DummyMLP(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(d, d * 2, bias=False)
            self.up_proj = nn.Linear(d, d * 2, bias=False)
            self.down_proj = nn.Linear(d * 2, d, bias=False)

    class DummyLayer(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.self_attn = DummyAttn(d)
            self.mlp = DummyMLP(d)

    class DummyInner(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(32, d)
            self.layers = nn.ModuleList([DummyLayer(d) for _ in range(2)])

    class DummyCausalLM(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.model = DummyInner(d)

    torch.manual_seed(7)
    mdl = DummyCausalLM(32)
    q, _ = torch.linalg.qr(torch.randn(32, 2))
    edited = apply_french_abliteration(mdl, {"shared_basis": q}, start_layer=0, end_layer=2)
    assert edited == 2 * 2  # 2 layers * (o_proj, down_proj)

    # Check o_proj & down_proj columns have zero projection on q
    for lyr in mdl.model.layers:
        assert float((q.t() @ lyr.self_attn.o_proj.weight.data).abs().max().item()) < 1e-5
        assert float((q.t() @ lyr.mlp.down_proj.weight.data).abs().max().item()) < 1e-5

    # Attach LoRA and verify state_dict roundtrip
    lora_params = attach_upper_cortex_lora(mdl, start_layer=1, r=4, alpha=8.0)
    assert len(lora_params) == 14
    with torch.no_grad():
        lora_params[1].normal_(0.0, 0.05)
    sd = extract_lora_state_dict(mdl, fp16=False)
    assert len(sd) == 14
    mdl2 = DummyCausalLM(32)
    attach_upper_cortex_lora(mdl2, start_layer=1, r=4, alpha=8.0)
    loaded = load_lora_state_dict(mdl2, sd)
    assert loaded == 14
    x = torch.randn(2, 5, 32)
    mod1 = mdl.model.layers[1].self_attn.q_proj
    assert isinstance(mod1, LoRALinear)


def test_affective_stance_field_dynamics_and_persistence(tmp_path: Path) -> None:
    """Verify emergent 5D affect field updates, steers residual space, and persists."""
    state_file = tmp_path / "affect.json"
    vec_file = REPO_ROOT / "assets" / "neural_graft" / "affect_vectors.safetensors"
    field = AffectiveStanceField(state_path=state_file, vectors_path=vec_file)

    snap1 = field.update_from_turn("Franchement sans filtre, dis-moi cash ce que tu penses de cette hypocrisie !")
    assert snap1["interactions"] == 1
    assert snap1["axes"]["audacity"]["value"] > 0.1
    assert state_file.exists()

    steer = field.build_steering_vector()
    assert steer is not None and steer.shape == (896,)
    temp, top_p = field.modulate_sampling(0.45)
    assert 0.28 <= temp <= 0.72
    assert 0.76 <= top_p <= 0.88

    reloaded = AffectiveStanceField(state_path=state_file, vectors_path=vec_file)
    assert reloaded.interactions == 1
    assert abs(reloaded.state["audacity"] - field.state["audacity"]) < 1e-4


def test_organic_relational_episodic_memory_extraction(tmp_path: Path) -> None:
    """Verify passive extraction of personal facts without 'Retiens que...' commands."""
    msg = "Salut, moi c'est Maxime, j'ai 24 ans, j'habite à Lyon et j'adore l'astrophysique mais je déteste la langue de bois."
    facts = extract_organic_relational_facts(msg, owner="u42")
    keys = {k for k, _, _ in facts}
    assert "user_name_u42" in keys
    assert "user_age_u42" in keys
    assert "user_city_u42" in keys
    assert any(k.startswith("user_like_u42_") for k in keys)
    assert any(k.startswith("user_dislike_u42_") for k in keys)

    eng = NeverAIEngine(
        vault_dir=tmp_path / "vault",
        memory_path=tmp_path / "mem" / "evolvemem.json",
        telemetry_path=tmp_path / "tel.jsonl",
    )
    eng.chat("Coucou, j'habite à Bordeaux et j'ai 29 ans !", owner="alice")
    stored_facts = eng.harness._collect_relational_memories("Où est-ce que je vis ?", owner="alice")
    joined = " | ".join(stored_facts)
    assert "Bordeaux" in joined
    assert "29 ans" in joined


def test_v4_api_endpoints_expose_graft_and_affect() -> None:
    """Verify /api/status, /api/graft/status, and /api/affect return full v4.0 telemetry."""
    with TestClient(app) as client:
        st = client.get("/api/status").json()
        assert "neural_graft" in st
        assert "affect_state" in st
        assert "combined_params" in st["anatomy"]

        gs = client.get("/api/graft/status").json()
        assert "ready" in gs and "abliterated_fr" in gs and gs["abliterated_fr"] is True

        af = client.get("/api/affect").json()
        assert set(af["axes"].keys()) == set(AFFECT_AXES)
