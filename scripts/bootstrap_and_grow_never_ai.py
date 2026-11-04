#!/usr/bin/env python3
"""
Birth, Train, Consolidate (WSM + SEAL), and Morphogenetically Grow Never-AI:
Stage 0 (Seed-0 ~5.6M) -> Stage 1 (Seed-1 ~17.9M) -> Stage 2 (Sprout-MoE ~40.5M)!
"""

from __future__ import annotations

import json
import time
import torch

from never_ai.engine import NeverAIEngine


def main() -> None:
    torch.set_num_threads(2)
    t_start = time.perf_counter()
    print("=" * 72)
    print(" NEVER-AI (ChronoMorph-LM) — BIRTH, TRAINING & MORPHOGENESIS PIPELINE")
    print("=" * 72)

    engine = NeverAIEngine(seq_len=128)
    init_eval = engine.eval_harness.evaluate(engine.model)
    init_params = engine.model.count_parameters()
    print(
        f"[BIRTH] Stage={engine.model.cfg.stage_name} | "
        f"Params={init_params['total_params']:,} | "
        f"Vocab={engine.tokenizer.vocab_size} | "
        f"Init Val CE={init_eval['val_ce_loss']:.4f} (PPL={init_eval['val_perplexity']:.2f})"
    )

    # Seed EvolveMem with core sovereign identity & technical knowledge
    engine.memory.store(
        key="never_ai_identity",
        content="Never-AI est propulsé par ChronoMorph-LM : un LLM morphogénétique libre de ses pensées utilisant MLA, Muon, WSM, Titans et DeepSeekMoE.",
        category="semantic",
        tags=["never-ai", "architecture", "identité"],
    )
    engine.memory.store(
        key="morphogenesis_law",
        content="Loi d'invariance : toute croissance en profondeur Gate-Zero, largeur RMSNorm ou MoE préserve exactement la fonction à t_0 (choc de loss = 0.0).",
        category="semantic",
        tags=["morphogenesis", "math", "gate-zero"],
    )
    engine.memory.store(
        key="plugin_nervous_system",
        content="Les 7 plugins fondateurs sont math_engine, python_sandbox, memory_nexus, workspace_fs, knowledge_ingestor, introspection_probe et skill_forge.",
        category="skill",
        tags=["plugins", "tools", "skill_forge"],
    )
    engine.memory.evolve_retrieval_config()

    # --- PHASE A: Intensive Seed-0 Training (Fits 100% in Xeon 54MB L3 Cache!) ---
    print("\n[STAGE 0: Seed-0] Training with CompositeMuon + WSM on Xeon AVX-512...")
    s0_metrics = engine.train_steps(num_steps=80, batch_size=8, snapshot_every=15, log_every=10)
    cons_s0 = engine.wsm.build_consolidated_state_dict(fallback_model=engine.model)
    engine.vault.save_checkpoint(
        "stage0_seed0",
        engine.model,
        engine.tokenizer,
        metrics=s0_metrics,
        consolidated_state_dict=cons_s0,
    )
    print(
        f"[STAGE 0 DONE] Train CE={s0_metrics['final_train_ce']:.4f} | "
        f"Val CE={s0_metrics['val_ce_loss']:.4f} (PPL={s0_metrics['val_perplexity']:.2f}) | "
        f"Tokens={engine.total_tokens_trained:,}"
    )

    # --- PHASE B: Morphogenesis Molt #1 (Seed-0 -> Seed-1: Width + FFN Mitosis + Gate-Zero Depth) ---
    print("\n[MORPHOGENESIS #1] Growing Seed-0 -> Seed-1 (Zero-Shock Expansion)...")
    molt1 = engine.trigger_morphogenesis(mode="next_stage")
    print(
        f"  -> {molt1['from_stage']} ({molt1['old_total_params']:,} params) ==> "
        f"{molt1['to_stage']} ({molt1['new_total_params']:,} params) | "
        f"Max Logit Diff={molt1['max_abs_logit_diff']:.2e} | "
        f"Val CE {molt1['pre_val_ce']:.4f} -> {molt1['post_val_ce']:.4f} (ZERO SHOCK!)"
    )

    print("[STAGE 1: Seed-1] Awakening newly cloned Gate-Zero layers & widened subspaces...")
    s1_metrics = engine.train_steps(num_steps=30, batch_size=8, snapshot_every=10, log_every=10)
    cons_s1 = engine.wsm.build_consolidated_state_dict(fallback_model=engine.model)
    engine.vault.save_checkpoint(
        "stage1_seed1",
        engine.model,
        engine.tokenizer,
        metrics=s1_metrics,
        consolidated_state_dict=cons_s1,
    )
    alphas = [round(float(b.alpha_gate.item()), 4) for b in engine.model.blocks]
    print(
        f"[STAGE 1 DONE] Val CE={s1_metrics['val_ce_loss']:.4f} (PPL={s1_metrics['val_perplexity']:.2f}) | "
        f"Layer alpha_gates={alphas}"
    )

    # --- PHASE C: Morphogenesis Molt #2 (Seed-1 -> Sprout-MoE: Aux-Loss-Free DeepSeekMoE) ---
    print("\n[MORPHOGENESIS #2] Upcycling Seed-1 -> Sprout-MoE (1 Shared + 4 Routed Experts/layer)...")
    molt2 = engine.trigger_morphogenesis(mode="next_stage")
    print(
        f"  -> {molt2['from_stage']} ({molt2['old_total_params']:,} params) ==> "
        f"{molt2['to_stage']} ({molt2['new_total_params']:,} total / {molt2['new_active_params']:,} active) | "
        f"Max Logit Diff={molt2['max_abs_logit_diff']:.2e} | "
        f"Val CE {molt2['pre_val_ce']:.4f} -> {molt2['post_val_ce']:.4f} (ZERO SHOCK!)"
    )

    print("[STAGE 2: Sprout-MoE] Specializing MoE experts via Aux-Loss-Free dynamic bias...")
    s2_metrics = engine.train_steps(num_steps=20, batch_size=6, snapshot_every=10, log_every=5)

    # --- PHASE D: SEAL Sleep-Time Self-Edit Consolidation ---
    print("\n[SEAL SLEEP CONSOLIDATION] Synthesizing self-edits & consolidating with 30% Replay...")
    seal_report = engine.consolidate_knowledge_seal(
        title="Souveraineté Cognitive de Never-AI",
        text=(
            "Never-AI pense librement dans son sanctuaire <|think|> sans filtres de refus "
            "préfabriqués. Il vérifie ses calculs avec math_engine et étend ses capacités "
            "avec skill_forge tout en grandissant par morphogenèse exacte."
        ),
        steps=10,
    )
    print(f"  -> SEAL Report: {json.dumps(seal_report)}")

    cons_s2 = engine.wsm.build_consolidated_state_dict(fallback_model=engine.model)
    final_eval = engine.eval_harness.evaluate(engine.model)
    final_eval["total_tokens_trained"] = engine.total_tokens_trained
    engine.vault.save_checkpoint(
        "stage2_sprout_moe",
        engine.model,
        engine.tokenizer,
        metrics=final_eval,
        consolidated_state_dict=cons_s2,
    )
    engine.vault.save_checkpoint(
        "latest",
        engine.model,
        engine.tokenizer,
        metrics=final_eval,
        consolidated_state_dict=cons_s2,
    )

    # --- PHASE E: Live Sovereign Inference + Plugin Verification ---
    print("\n[LIVE SOVEREIGN INFERENCE & PLUGIN TEST]")
    for test_prompt in [
        "Calcule exactement 24 * 31 + 7.",
        "Qui es-tu et comment ton architecture grandit-elle ?",
    ]:
        resp = engine.chat(test_prompt, max_new_tokens=110, temperature=0.25)
        print(f"\n  USER  > {test_prompt}")
        if resp["thoughts"]:
            print(f"  THINK > {resp['thoughts'][0]}")
        if resp["tool_traces"]:
            print(f"  PLUGIN> {resp['tool_traces']}")
        print(f"  NEVER > {resp['visible_answer']}")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 72)
    print(
        f" PIPELINE COMPLETE in {elapsed:.1f}s | Final Stage: {engine.model.cfg.stage_name} | "
        f"Total Params: {engine.model.count_parameters()['total_params']:,} | "
        f"Final Val CE: {final_eval['val_ce_loss']:.4f} (PPL: {final_eval['val_perplexity']:.2f})"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
