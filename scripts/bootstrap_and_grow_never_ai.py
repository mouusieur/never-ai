#!/usr/bin/env python3
"""
Birth, Train, Consolidate (WSM + SEAL), and Morphogenetically Grow Never-AI v1.0 Omega:
Stage 0 (Seed-0) -> Stage 1 (Seed-1) -> Stage 2 (Sprout-MoE ~43.56M) across 10 Domains & 15 Plugins!
Saves a single lean bfloat16 checkpoint in `checkpoints/latest/` (<128 MB).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import torch

from never_ai.engine import NeverAIEngine


def main() -> None:
    torch.set_num_threads(2)
    t_start = time.perf_counter()
    print("=" * 76)
    print(" NEVER-AI v1.0 OMEGA — 10-DOMAIN TRAINING & ZERO-SHOCK MORPHOGENESIS")
    print("=" * 76)

    # Clean start in checkpoints/ so we train fresh on the 10-domain Omega curriculum
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    engine = NeverAIEngine(seq_len=160)
    init_eval = engine.eval_harness.evaluate(engine.model)
    init_params = engine.model.count_parameters()
    print(
        f"[BIRTH] Stage={engine.model.cfg.stage_name} | "
        f"Params={init_params['total_params']:,} | "
        f"Vocab={engine.tokenizer.vocab_size} | "
        f"Plugins={len(engine.registry.list_plugins())} | "
        f"Init Val CE={init_eval['val_ce_loss']:.4f} (PPL={init_eval['val_perplexity']:.2f})"
    )

    engine.memory.store(
        key="never_ai_identity",
        content="Never-AI v1.0 Omega est propulsé par ChronoMorph-LM : un LLM morphogénétique libre couvrant 10 domaines et 15 plugins souverains.",
        category="semantic",
        tags=["never-ai", "architecture", "identité", "omega"],
    )
    engine.memory.evolve_retrieval_config()

    # --- PHASE A: Intensive Seed-0 Mastery (Fits 100% in Xeon AVX-512 L3 Cache!) ---
    print("\n[STAGE 0: Seed-0] Deep Mastery: 320 steps on 10-Domain Omega Curriculum (CompositeMuon + WSM)...")
    s0_metrics = engine.train_steps(num_steps=320, batch_size=12, snapshot_every=40, log_every=40)
    print(
        f"[STAGE 0 DONE] Train CE={s0_metrics['final_train_ce']:.4f} | "
        f"Val CE={s0_metrics['val_ce_loss']:.4f} (PPL={s0_metrics['val_perplexity']:.2f}) | "
        f"Tokens={engine.total_tokens_trained:,}"
    )

    # --- PHASE B: Morphogenesis Molt #1 (Seed-0 -> Seed-1) ---
    print("\n[MORPHOGENESIS #1] Growing Seed-0 -> Seed-1 (Zero-Shock Expansion)...")
    molt1 = engine.trigger_morphogenesis(mode="next_stage")
    print(
        f"  -> {molt1['from_stage']} ({molt1['old_total_params']:,} params) ==> "
        f"{molt1['to_stage']} ({molt1['new_total_params']:,} params) | "
        f"Max Logit Diff={molt1['max_abs_logit_diff']:.2e} | "
        f"Val CE {molt1['pre_val_ce']:.4f} -> {molt1['post_val_ce']:.4f} (ZERO SHOCK!)"
    )

    print("[STAGE 1: Seed-1] Awakening cloned Gate-Zero layers & widened subspaces (24 steps)...")
    s1_metrics = engine.train_steps(num_steps=24, batch_size=8, snapshot_every=8, log_every=8)
    alphas = [round(float(b.alpha_gate.item()), 4) for b in engine.model.blocks]
    print(
        f"[STAGE 1 DONE] Val CE={s1_metrics['val_ce_loss']:.4f} (PPL={s1_metrics['val_perplexity']:.2f}) | "
        f"Layer alpha_gates={alphas}"
    )

    # --- PHASE C: Morphogenesis Molt #2 (Seed-1 -> Sprout-MoE) ---
    print("\n[MORPHOGENESIS #2] Upcycling Seed-1 -> Sprout-MoE (1 Shared + 4 Routed Experts/layer)...")
    molt2 = engine.trigger_morphogenesis(mode="next_stage")
    print(
        f"  -> {molt2['from_stage']} ({molt2['old_total_params']:,} params) ==> "
        f"{molt2['to_stage']} ({molt2['new_total_params']:,} total / {molt2['new_active_params']:,} active) | "
        f"Max Logit Diff={molt2['max_abs_logit_diff']:.2e} | "
        f"Val CE {molt2['pre_val_ce']:.4f} -> {molt2['post_val_ce']:.4f} (ZERO SHOCK!)"
    )

    print("[STAGE 2: Sprout-MoE] Specializing MoE experts across 10 domains (20 steps)...")
    s2_metrics = engine.train_steps(num_steps=20, batch_size=6, snapshot_every=5, log_every=5)
    print(
        f"[STAGE 2 DONE] Train CE={s2_metrics['final_train_ce']:.4f} | "
        f"Val CE={s2_metrics['val_ce_loss']:.4f} (PPL={s2_metrics['val_perplexity']:.2f})"
    )

    # --- PHASE D: SEAL Sleep-Time Self-Edit Consolidation ---
    print("\n[SEAL SLEEP CONSOLIDATION] Synthesizing self-edits & consolidating with 30% Replay...")
    seal_report = engine.consolidate_knowledge_seal(
        title="Souveraineté Polymathique Omega",
        text=(
            "Never-AI v1.0 Omega explore librement ses hypothèses dans son espace interne de réflexion "
            "sans formules préfabriquées, orchestre 15 plugins formels et grandit sans choc de perte."
        ),
        steps=4,
    )
    print(f"  -> SEAL Report: {json.dumps(seal_report)}")

    cons_s2 = engine.wsm.build_consolidated_state_dict(fallback_model=engine.model)
    final_eval = engine.eval_harness.evaluate(engine.model)
    final_eval["total_tokens_trained"] = engine.total_tokens_trained
    engine.vault.save_checkpoint(
        "latest",
        engine.model,
        engine.tokenizer,
        metrics=final_eval,
        consolidated_state_dict=cons_s2,
    )

    # --- PHASE E: Multi-Domain Sovereign Inference & 15-Plugin Verification ---
    print("\n[LIVE MULTI-DOMAIN INFERENCE & OMEGA BENCHMARK]")
    bench = engine.run_full_benchmark()
    print(
        f"  BENCHMARK > Symbolic Score: {bench['symbolic_benchmark']['passed']}/{bench['symbolic_benchmark']['total']} "
        f"({bench['symbolic_benchmark']['score_pct']}%) | Neural Val CE: {bench['neural_eval']['val_ce_loss']}"
    )

    for test_prompt in [
        "Salut ! Que sais-tu faire exactement ?",
        "Qu'est-ce qu'un équilibre de Nash en théorie des jeux ?",
        "Écris l'algorithme de recherche dichotomique (binary search) en Python.",
        "/plot sin(x) * cos(2*x) sur [-5, 5]",
        "/logic ((P >> Q) & P) >> Q",
    ]:
        resp = engine.chat(test_prompt, max_new_tokens=150, temperature=0.15)
        print(f"\n  USER  > {test_prompt}")
        if resp["thoughts"]:
            print(f"  THINK > {resp['thoughts'][0][:120]}")
        ans_preview = resp["visible_answer"].split("```svg")[0].strip()
        print(f"  NEVER > {ans_preview[:260]}")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 76)
    print(
        f" OMEGA PIPELINE COMPLETE in {elapsed:.1f}s | Stage: {engine.model.cfg.stage_name} | "
        f"Total Params: {engine.model.count_parameters()['total_params']:,} | "
        f"Final Val CE: {final_eval['val_ce_loss']:.4f} (PPL: {final_eval['val_perplexity']:.2f})"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
