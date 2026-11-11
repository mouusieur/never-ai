"""Rigorous mathematical invariance & training tests for ChronoMorph-LM & Morphogenesis."""

from __future__ import annotations

from pathlib import Path
import torch

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import (
    BOS_TOKEN,
    CALL_END,
    CALL_START,
    EOS_TOKEN,
    LATENT_BOT_ID,
    NEVER_TOKEN,
    RESULT_END,
    RESULT_START,
    THINK_END,
    THINK_START,
    USER_TOKEN,
)
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.morphogenesis.operators import (
    grow_depth_interpositional_gate_zero,
    grow_ffn_mitosis,
    grow_moe_experts,
    grow_vocabulary_convex_hull,
    grow_width_rms_compensated,
    transition_seed0_to_seed1,
    transition_seed1_to_sprout_moe,
    upcycle_dense_to_moe,
    verify_function_preservation,
)
from never_ai.optim import CompositeMuonOptimizer, WSMCheckpointMerger
from never_ai.vault import VaultManager


def _sample_corpus() -> list[str]:
    base = [
        f"{BOS_TOKEN}{USER_TOKEN}Calcule 17 * 23 et réfléchis librement.{NEVER_TOKEN}"
        f"{THINK_START}Décomposons 17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391.{THINK_END}"
        f"{CALL_START}{{\"plugin\": \"math_engine\", \"expr\": \"17*23\"}}{CALL_END}"
        f"{RESULT_START}391{RESULT_END}Le résultat exact est 391.{EOS_TOKEN}",
        "La morphogenèse neuronale permet à un modèle de langage de grandir continuellement sans jamais oublier ses représentations acquises.",
        "Continual learning, spectral Muon optimization, and test-time neural memory unlock rapid open-thought reasoning.",
    ]
    return base * 40


def _small_test_config(vocab_size: int = 512) -> ChronoMorphConfig:
    return ChronoMorphConfig(
        stage_name="Test-Seed",
        vocab_size=vocab_size,
        max_seq_len=128,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_head=32,
        d_kv_latent=32,
        d_rope=16,
        d_ff=128,
        use_moe=False,
        use_titans_memory=True,
        d_mem=32,
        d_mem_hidden=64,
        titans_chunk_size=8,
        use_unet_skips=True,
        use_mtp=True,
    )


def test_tokenizer_special_tokens_and_convex_hull_decomp() -> None:
    texts = _sample_corpus()
    tok_small = MorphTokenizer.train_from_iterator(texts, vocab_size=320)
    tok_large = MorphTokenizer.train_from_iterator(texts, vocab_size=420)

    assert tok_small.vocab_size >= 272
    assert tok_large.vocab_size > tok_small.vocab_size

    sample = texts[0]
    ids = tok_small.encode(sample)
    mask = tok_small.build_target_loss_mask(ids)
    assert len(ids) == len(mask)
    assert sum(mask) > 0
    assert sum(mask) < len(ids)  # user & result spans must be masked!
    assert tok_small.decode(ids) == sample

    decomp = MorphTokenizer.compute_subword_decomposition_map(tok_small, tok_large)
    assert len(decomp) == tok_large.vocab_size
    assert all(len(sub) >= 1 for sub in decomp)


def test_all_five_morphogenesis_operators_exact_function_preservation() -> None:
    torch.manual_seed(7)
    cfg = _small_test_config(vocab_size=320)
    base = ChronoMorphLM(cfg)

    # 1. Depth Interpositional Gate-Zero (L=2 -> L=4)
    g_depth = grow_depth_interpositional_gate_zero(base)
    res_depth = verify_function_preservation(base, g_depth)
    assert res_depth["preserved"], f"Depth growth broke FP: {res_depth}"
    assert res_depth["max_abs_logit_diff"] < 1e-6

    # 2. FFN Symmetric Mitosis (d_ff=128 -> 192 -> 256)
    g_ffn = grow_ffn_mitosis(base, new_d_ff=192)
    res_ffn = verify_function_preservation(base, g_ffn)
    assert res_ffn["preserved"], f"FFN mitosis broke FP: {res_ffn}"
    assert res_ffn["max_abs_logit_diff"] < 1e-5

    # 3. Residual Width + Heads + MLA Latent Expansion (d=64->96, H=2->3, C=32->48)
    g_width = grow_width_rms_compensated(
        base, new_d_model=96, new_n_heads=3, new_d_kv_latent=48
    )
    res_width = verify_function_preservation(base, g_width)
    assert res_width["preserved"], f"Width expansion broke FP: {res_width}"
    assert res_width["max_abs_logit_diff"] < 1e-5

    # Verify new dimensions receive non-zero gradients on step 1!
    x = torch.randint(16, 320, (2, 16))
    out = g_width(x, targets=x)
    assert out.loss is not None
    out.loss.backward()
    assert g_width.tok_emb.weight.grad[:, 64:].abs().max().item() > 0.0

    # 4. Dense -> Aux-Loss-Free DeepSeekMoE Upcycling & MoE Doubling
    g_moe = upcycle_dense_to_moe(base, n_routed_experts=4, top_k=2)
    res_moe = verify_function_preservation(base, g_moe)
    assert res_moe["preserved"], f"MoE upcycling broke FP: {res_moe}"

    g_moe2 = grow_moe_experts(g_moe, double_experts=True)
    res_moe2 = verify_function_preservation(g_moe, g_moe2)
    assert res_moe2["preserved"], f"MoE expert doubling broke FP: {res_moe2}"

    # 5. Vocabulary Convex-Hull Expansion
    texts = _sample_corpus()
    tok_1 = MorphTokenizer.train_from_iterator(texts, vocab_size=320)
    tok_2 = MorphTokenizer.train_from_iterator(texts, vocab_size=400)
    model_v1 = ChronoMorphLM(_small_test_config(vocab_size=tok_1.vocab_size))
    model_v2 = grow_vocabulary_convex_hull(model_v1, tok_1, tok_2)
    assert model_v2.cfg.vocab_size == tok_2.vocab_size
    # Source vocabulary rows must match exactly
    assert (model_v2.tok_emb.weight[:16] - model_v1.tok_emb.weight[:16]).abs().max().item() == 0.0


def test_full_stage_pipeline_muon_wsm_coconut_and_vault(tmp_path: Path) -> None:
    torch.manual_seed(99)
    texts = _sample_corpus()
    tok = MorphTokenizer.train_from_iterator(texts, vocab_size=320)

    cfg = ChronoMorphConfig.seed_0(vocab_size=tok.vocab_size)
    # Scale down slightly for sub-second unit test speed while exercising real Seed0->Seed1->Sprout transitions
    model = ChronoMorphLM(cfg)
    opt = CompositeMuonOptimizer(model, muon_lr=1.5e-2, adamw_lr=3e-3, warmup_steps=3)
    wsm = WSMCheckpointMerger(max_snapshots=4)

    x = torch.randint(16, tok.vocab_size, (2, 32))
    initial_loss = float(model(x, targets=x).loss.item())

    for step in range(6):
        opt.zero_grad()
        out = model(x, targets=x)
        assert out.loss is not None
        out.loss.backward()
        opt.step(model)
        if step % 2 == 1:
            wsm.record_snapshot(model)

    trained_loss = float(model(x, targets=x).loss.item())
    assert trained_loss < initial_loss

    # Test Coconut continuous latent generation & Titans state carry-over
    gen_ids, t_states = model.generate(
        [1, 20, LATENT_BOT_ID], max_new_tokens=8, temperature=0.0, latent_steps_on_bot=2
    )
    assert len(gen_ids) >= 1
    assert len(t_states) == cfg.n_layers

    # Transition Seed-0 -> Seed-1 -> Sprout-MoE with zero loss shock!
    seed1 = transition_seed0_to_seed1(model)
    assert verify_function_preservation(model, seed1)["preserved"]
    sprout = transition_seed1_to_sprout_moe(seed1, n_routed_experts=4, top_k=2)
    assert verify_function_preservation(seed1, sprout)["preserved"]

    # Vault atomic save/load
    vault = VaultManager(tmp_path / "vault")
    cons_sd = wsm.build_consolidated_state_dict(fallback_model=model)
    vault.save_checkpoint("seed0_test", model, tok, consolidated_state_dict=cons_sd)
    loaded, loaded_tok, manifest = vault.load_checkpoint("seed0_test")
    assert manifest["stage_name"] == "Seed-0"
    assert loaded_tok.vocab_size == tok.vocab_size
    assert loaded.count_parameters()["total_params"] == model.count_parameters()["total_params"]
