"""Tests for EvolveMem, ToolRAG, 7 Core Plugins, SkillForge, SEAL & GRPO."""

from __future__ import annotations

from pathlib import Path
import torch

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import BOS_TOKEN, NEVER_TOKEN, USER_TOKEN
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data import ContinualDatasetAndReplay, synthesize_bilingual_curriculum
from never_ai.evolution import GRPOReasoningTrainer, ImmutableEvalHarness, SEALConsolidator
from never_ai.memory import EvolveMemStore
from never_ai.optim import CompositeMuonOptimizer
from never_ai.plugins import SkillForgePlugin, SovereignAgentHarness, build_default_plugin_registry


def test_evolvemem_and_all_builtin_plugins_plus_skill_forge(tmp_path: Path) -> None:
    mem = EvolveMemStore(tmp_path / "evolvemem.json")
    mem.store("euler", "L'identité d'Euler relie e^(i*pi) + 1 = 0.", tags=["math", "euler"])
    mem.store("shannon", "Claude Shannon a fondé la théorie mathématique de l'information.", tags=["info"])
    mem.store("muon", "L'optimiseur Muon orthogonalise le momentum via Newton-Schulz.", tags=["optim", "ml"])

    hits = mem.query("Newton-Schulz momentum optimiseur", top_k=2)
    assert hits and hits[0]["key"] == "muon"

    evo = mem.evolve_retrieval_config()
    assert evo["final_mrr"] >= evo["base_mrr"]

    reg = build_default_plugin_registry(
        mem_store=mem,
        anatomy_provider=lambda: {"stage_name": "Seed-0", "total_params": 4800000},
        workspace_root=".",
    )
    # Verify ToolRAG dynamic routing
    math_tools = reg.retrieve_relevant_tools("Calcule l'équation et le nombre premier")
    assert any(t.name == "math_engine" for t in math_tools)

    # Verify MathEnginePlugin
    res_math = reg.invoke("math_engine", expr="sum(primerange(1, 20))")
    assert res_math["ok"] is True and res_math["result"] == "77"

    # Verify PythonSandboxPlugin
    res_py = reg.invoke("python_sandbox", code="print(sum(i*i for i in range(6)))")
    assert res_py["ok"] is True and res_py["stdout"].strip() == "55"

    # Verify IntrospectionProbePlugin
    res_intro = reg.invoke("introspection_probe")
    assert res_intro["ok"] is True and res_intro["anatomy"]["stage_name"] == "Seed-0"

    # Verify SkillForgePlugin rejects unsafe imports (Anti-Goodhart / Security gate)
    forge = SkillForgePlugin(reg, forged_dir=tmp_path / "forged")
    bad = forge.execute(
        plugin_name="evil_tool",
        plugin_code="import subprocess\n",
        test_code="def test_a(): assert True\ndef test_b(): assert True\n",
    )
    assert bad["ok"] is False and "Forbidden import" in bad["error"]

    # Verify SkillForgePlugin synthesizes, tests, and hot-registers a valid new plugin!
    valid_plugin_code = '''
from never_ai.plugins.base import NeverPlugin

class BaseConverterPlugin(NeverPlugin):
    name = "base_converter"
    description = "Convert integers to binary and hexadecimal representations."
    keywords = ["binary", "hex", "binaire", "base"]
    parameters_schema = {"n": {"type": "integer"}}

    def execute(self, n: int = 0, **_):
        val = int(n)
        return {"ok": True, "bin": bin(val), "hex": hex(val)}
'''
    valid_test_code = '''
from base_converter import BaseConverterPlugin

def test_zero():
    p = BaseConverterPlugin()
    assert p.execute(n=0) == {"ok": True, "bin": "0b0", "hex": "0x0"}

def test_forty_two():
    p = BaseConverterPlugin()
    assert p.execute(n=42) == {"ok": True, "bin": "0b101010", "hex": "0x2a"}
'''
    good = forge.execute(
        plugin_name="base_converter",
        plugin_code=valid_plugin_code,
        test_code=valid_test_code,
    )
    assert good["ok"] is True, f"SkillForge failed: {good}"
    assert reg.invoke("base_converter", n=255)["hex"] == "0xff"


def test_curriculum_seal_consolidation_grpo_and_harness(tmp_path: Path) -> None:
    torch.manual_seed(11)
    train_docs, val_docs = synthesize_bilingual_curriculum(seed=7, scale_factor=1)
    tok = MorphTokenizer.train_from_iterator(train_docs, vocab_size=512)

    ds = ContinualDatasetAndReplay(tok, seq_len=64)
    stats = ds.load_corpora(train_docs, val_docs)
    assert stats["train_chunks"] > 10 and stats["val_chunks"] >= 1

    cfg = ChronoMorphConfig(
        stage_name="Test-SEAL",
        vocab_size=tok.vocab_size,
        max_seq_len=128,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_head=32,
        d_kv_latent=32,
        d_rope=16,
        d_ff=128,
    )
    model = ChronoMorphLM(cfg)
    opt = CompositeMuonOptimizer(model, muon_lr=1e-2, adamw_lr=2e-3, warmup_steps=2)

    harness_eval = ImmutableEvalHarness(ds, eval_batches=2, batch_size=2)
    seal = SEALConsolidator(ds, harness_eval)
    seal_rep = seal.run_sleep_consolidation(
        model,
        opt,
        knowledge_items=[
            (
                "Théorème de Noether",
                "Chaque symétrie continue d'un système physique correspond à une grandeur conservée. L'invariance par translation dans le temps implique la conservation de l'énergie.",
            )
        ],
        steps=3,
        batch_size=4,
    )
    assert seal_rep["num_self_edits"] >= 3

    mem = EvolveMemStore(tmp_path / "mem.json")
    reg = build_default_plugin_registry(mem, model.inspect_anatomy, workspace_root=".")

    grpo = GRPOReasoningTrainer(tok, reg, group_size=3, lr=1e-4)
    grpo_rep = grpo.step_grpo(
        model,
        prompt_text=f"{BOS_TOKEN}{USER_TOKEN}Calcule 6 * 7.{NEVER_TOKEN}",
        ground_truth_int=42,
        max_new_tokens=16,
    )
    assert "mean_reward" in grpo_rep

    agent = SovereignAgentHarness(model, tok, reg, mem)
    turn = agent.run_turn("Qui es-tu ?", max_new_tokens=20)
    assert "raw_generation" in turn
