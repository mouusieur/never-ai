"""Unified Never-AI Engine orchestrating Training, WSM, Morphogenesis, SEAL, Plugins & Vault."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import torch

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.model import ChronoMorphLM
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data.curriculum import ContinualDatasetAndReplay, synthesize_bilingual_curriculum
from never_ai.evolution.seal_and_grpo import (
    GRPOReasoningTrainer,
    ImmutableEvalHarness,
    SEALConsolidator,
)
from never_ai.memory.evolvemem import EvolveMemStore
from never_ai.morphogenesis.operators import (
    grow_depth_interpositional_gate_zero,
    grow_ffn_mitosis,
    grow_moe_experts,
    transition_seed0_to_seed1,
    transition_seed1_to_sprout_moe,
    verify_function_preservation,
)
from never_ai.optim.composite_muon import CompositeMuonOptimizer, WSMCheckpointMerger
from never_ai.plugins.agent_harness import SovereignAgentHarness
from never_ai.plugins.builtin import build_default_plugin_registry
from never_ai.vault.manager import VaultManager


class NeverAIEngine:
    """
    Thread-safe master runtime of Never-AI:
    - Live ChronoMorphLM brain + MorphTokenizer
    - Perpetual CompositeMuon + WSM 1-sqrt consolidation
    - Zero-Shock Morphogenesis Engine
    - EvolveMem + 7 Plugins + SkillForge + SovereignAgentHarness
    - Background Continual Learning Daemon
    """

    def __init__(
        self,
        vault_dir: str | Path = "out/checkpoints",
        memory_path: str | Path = "out/memory_store/evolvemem.json",
        telemetry_path: str | Path = "out/logs/training_telemetry.jsonl",
        seq_len: int = 128,
    ) -> None:
        self.lock = threading.RLock()
        self.vault = VaultManager(vault_dir)
        self.memory = EvolveMemStore(memory_path)
        self.telemetry_path = Path(telemetry_path)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load latest checkpoint if available, else birth Seed-0
        ckpts = self.vault.list_checkpoints()
        train_docs, val_docs = synthesize_bilingual_curriculum(seed=42, scale_factor=2)

        if ckpts:
            latest_tag = self._select_latest_tag([c.get("folder", c["tag"]) for c in ckpts])
            self.model, self.tokenizer, manifest = self.vault.load_checkpoint(
                latest_tag, prefer_wsm=False, device=self.device
            )
            self.total_tokens_trained: int = int(
                manifest.get("metrics", {}).get("total_tokens_trained", 0)
            )
        else:
            self.tokenizer = MorphTokenizer.train_from_iterator(train_docs, vocab_size=4096)
            cfg = ChronoMorphConfig.seed_0(vocab_size=self.tokenizer.vocab_size)
            self.model = ChronoMorphLM(cfg).to(self.device)
            self.total_tokens_trained = 0

        self.dataset = ContinualDatasetAndReplay(self.tokenizer, seq_len=self.seq_len)
        self.dataset.load_corpora(train_docs, val_docs)

        self.optimizer = CompositeMuonOptimizer(
            self.model,
            muon_lr=1.2e-2,
            adamw_lr=2.5e-3,
            weight_decay=0.01,
            warmup_steps=25,
        )
        self.wsm = WSMCheckpointMerger(max_snapshots=6)
        self.wsm.record_snapshot(self.model)

        self.eval_harness = ImmutableEvalHarness(self.dataset, eval_batches=4, batch_size=4)
        self.seal = SEALConsolidator(self.dataset, self.eval_harness)

        self.registry = build_default_plugin_registry(
            mem_store=self.memory,
            anatomy_provider=self.model.inspect_anatomy,
            workspace_root=".",
        )
        self.grpo = GRPOReasoningTrainer(self.tokenizer, self.registry, group_size=4, lr=1e-4)
        self.harness = SovereignAgentHarness(
            self.model, self.tokenizer, self.registry, self.memory
        )

        self.telemetry_history: list[dict[str, Any]] = self._load_recent_telemetry()
        self._bg_running = False
        self._bg_thread: threading.Thread | None = None

    @staticmethod
    def _select_latest_tag(tags: list[str]) -> str:
        if "latest" in tags:
            return "latest"
        return tags[-1]

    def _load_recent_telemetry(self, limit: int = 250) -> list[dict[str, Any]]:
        if not self.telemetry_path.exists():
            return []
        lines = self.telemetry_path.read_text(encoding="utf-8").splitlines()
        items: list[dict[str, Any]] = []
        for ln in lines[-limit:]:
            if ln.strip():
                try:
                    items.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        return items

    def _append_telemetry(self, event: dict[str, Any]) -> None:
        self.telemetry_history.append(event)
        if len(self.telemetry_history) > 500:
            self.telemetry_history = self.telemetry_history[-500:]
        with self.telemetry_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def train_steps(
        self,
        num_steps: int = 20,
        batch_size: int = 8,
        snapshot_every: int = 10,
        log_every: int = 5,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        losses: list[float] = []

        for step_idx in range(1, num_steps + 1):
            with self.lock:
                self.model.train()
                batch = self.dataset.sample_batch(
                    batch_size=batch_size,
                    split="train",
                    replay_ratio=0.25,
                    device=self.device,
                )
                self.optimizer.zero_grad()
                out = self.model(
                    batch.input_ids,
                    targets=batch.targets,
                    loss_mask=batch.loss_mask,
                    update_titans=True,
                )
                assert out.loss is not None
                out.loss.backward()
                opt_stats = self.optimizer.step(self.model)
                self.model.update_moe_biases()

                tokens_step = int(batch.input_ids.numel())
                self.total_tokens_trained += tokens_step
                losses.append(out.ce_loss)

                if step_idx % snapshot_every == 0:
                    self.wsm.record_snapshot(self.model)

                if step_idx % log_every == 0 or step_idx == num_steps:
                    elapsed = max(1e-6, time.perf_counter() - t0)
                    tok_per_s = (step_idx * tokens_step) / elapsed
                    rec = {
                        "type": "train_step",
                        "timestamp": round(time.time(), 2),
                        "stage": self.model.cfg.stage_name,
                        "generation": self.model.cfg.generation,
                        "global_step": int(opt_stats["step"]),
                        "total_tokens": self.total_tokens_trained,
                        "ce_loss": round(out.ce_loss, 4),
                        "mtp_loss": round(out.mtp_loss, 4),
                        "grad_norm": round(opt_stats["grad_norm"], 4),
                        "tok_per_sec": round(tok_per_s, 1),
                    }
                    self._append_telemetry(rec)

        with self.lock:
            eval_metrics = self.eval_harness.evaluate(self.model)
            eval_metrics["total_tokens_trained"] = self.total_tokens_trained
            cons_sd = self.wsm.build_consolidated_state_dict(fallback_model=self.model)
            self.vault.save_checkpoint(
                "latest",
                self.model,
                self.tokenizer,
                metrics=eval_metrics,
                consolidated_state_dict=cons_sd,
            )

        return {
            "steps_executed": num_steps,
            "mean_train_ce": round(sum(losses) / max(1, len(losses)), 4),
            "final_train_ce": round(losses[-1], 4) if losses else 0.0,
            **eval_metrics,
        }

    def trigger_morphogenesis(self, mode: str = "next_stage") -> dict[str, Any]:
        """
        Execute verified Zero-Shock Morphogenesis on the live brain!
        Supported modes: 'next_stage' | 'depth' | 'ffn' | 'moe_upcycle' | 'moe_double'
        """
        with self.lock:
            pre_eval = self.eval_harness.evaluate(self.model)
            old_counts = self.model.count_parameters()
            old_stage = self.model.cfg.stage_name

            if mode == "next_stage":
                if self.model.cfg.stage_name == "Seed-0":
                    grown = transition_seed0_to_seed1(self.model)
                elif not self.model.cfg.use_moe:
                    grown = transition_seed1_to_sprout_moe(self.model, n_routed_experts=4, top_k=2)
                else:
                    grown = grow_moe_experts(self.model, double_experts=True)
            elif mode == "depth":
                grown = grow_depth_interpositional_gate_zero(self.model)
            elif mode == "ffn":
                new_ff = int(self.model.cfg.d_ff * 1.5)
                grown = grow_ffn_mitosis(self.model, new_d_ff=new_ff)
            elif mode == "moe_upcycle":
                grown = transition_seed1_to_sprout_moe(self.model, n_routed_experts=4, top_k=2)
            elif mode == "moe_double":
                grown = grow_moe_experts(self.model, double_experts=True)
            else:
                raise ValueError(f"Unknown morphogenesis mode '{mode}'")

            grown = grown.to(self.device)
            invariance = verify_function_preservation(self.model, grown)
            if not invariance["preserved"]:
                raise RuntimeError(f"Morphogenesis rejected by Directive 1 gate: {invariance}")

            self.model = grown
            self.harness.model = self.model
            self.harness.reset_session_fast_weights()
            self.registry = build_default_plugin_registry(
                mem_store=self.memory,
                anatomy_provider=self.model.inspect_anatomy,
                workspace_root=".",
            )
            self.harness.registry = self.registry

            self.optimizer.rebuild_for_model(self.model)
            self.wsm.clear()
            self.wsm.record_snapshot(self.model)

            post_eval = self.eval_harness.evaluate(self.model)
            new_counts = self.model.count_parameters()

            event = {
                "type": "morphogenesis",
                "timestamp": round(time.time(), 2),
                "from_stage": old_stage,
                "to_stage": self.model.cfg.stage_name,
                "generation": self.model.cfg.generation,
                "old_total_params": old_counts["total_params"],
                "new_total_params": new_counts["total_params"],
                "new_active_params": new_counts["active_params"],
                "max_abs_logit_diff": invariance["max_abs_logit_diff"],
                "pre_val_ce": pre_eval["val_ce_loss"],
                "post_val_ce": post_eval["val_ce_loss"],
            }
            self._append_telemetry(event)

            self.vault.save_checkpoint(
                "latest",
                self.model,
                self.tokenizer,
                metrics={**post_eval, "total_tokens_trained": self.total_tokens_trained},
            )
            return event

    def consolidate_knowledge_seal(
        self, title: str, text: str, steps: int = 10
    ) -> dict[str, Any]:
        with self.lock:
            self.memory.store(key=title, content=text, category="semantic", tags=["seal"])
            rep = self.seal.run_sleep_consolidation(
                self.model,
                self.optimizer,
                knowledge_items=[(title, text)],
                steps=steps,
                batch_size=6,
            )
            self.wsm.record_snapshot(self.model)
            self._append_telemetry(
                {
                    "type": "seal_consolidation",
                    "timestamp": round(time.time(), 2),
                    "title": title,
                    **rep,
                }
            )
            return rep

    def chat(
        self,
        message: str,
        max_new_tokens: int = 140,
        temperature: float = 0.45,
    ) -> dict[str, Any]:
        with self.lock:
            return self.harness.run_turn(
                message,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def start_background_growth_loop(self, batch_size: int = 6) -> bool:
        if self._bg_running:
            return False
        self._bg_running = True

        def _worker() -> None:
            while self._bg_running:
                try:
                    self.train_steps(num_steps=4, batch_size=batch_size, snapshot_every=4, log_every=2)
                except Exception:  # noqa: BLE001
                    self._bg_running = False
                    break
                time.sleep(0.15)

        self._bg_thread = threading.Thread(target=_worker, daemon=True)
        self._bg_thread.start()
        return True

    def stop_background_growth_loop(self) -> bool:
        self._bg_running = False
        return True

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "background_training": self._bg_running,
                "device": str(self.device),
                "total_tokens_trained": self.total_tokens_trained,
                "anatomy": self.model.inspect_anatomy(),
                "wsm": self.wsm.describe(),
                "memories_count": len(self.memory.units),
                "plugins": self.registry.list_plugins(),
                "checkpoints": self.vault.list_checkpoints(),
                "recent_telemetry": self.telemetry_history[-60:],
            }
