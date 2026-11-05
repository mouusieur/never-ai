"""Unified Never-AI Continual Growth, Morphogenesis & Streaming Chatbot Engine."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Iterator

import torch

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    NEVER_TOKEN,
    THINK_END,
    THINK_START,
    USER_TOKEN,
)
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data.curriculum import ContinualDatasetAndReplay, synthesize_bilingual_curriculum
from never_ai.evolution.seal_and_grpo import (
    GRPOReasoningTrainer,
    ImmutableEvalHarness,
    SEALConsolidator,
)
from never_ai.memory.conversations import ConversationStore, LocalIdentityVault
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
    Thread-safe, non-blocking Master Runtime orchestrating:
    - Live ChronoMorphLM brain + MorphTokenizer
    - Lock-free real-time status telemetry (zero UI freeze during continual training!)
    - Chat-priority cooperative Continual Growth daemon with Auto-Morphogenesis milestones
    - Live learning from user dialogues (Replay Reservoir + EvolveMem + SEAL)
    - Streaming token-by-token chatbot inference
    """

    AUTO_MOLT_TOKEN_INTERVAL = 12_288  # Automatic Zero-Shock growth milestone in Continuous Growth

    def __init__(
        self,
        vault_dir: str | Path = "checkpoints",
        memory_path: str | Path = "memory_store/evolvemem.json",
        telemetry_path: str | Path = "logs/training_telemetry.jsonl",
        seq_len: int = 160,
    ) -> None:
        self.lock = threading.RLock()
        self.vault = VaultManager(vault_dir)
        mem_dir = Path(memory_path).parent
        self.memory = EvolveMemStore(memory_path)
        self.identities = LocalIdentityVault(mem_dir / "identities.json")
        self.conversations = ConversationStore(mem_dir / "conversations.json")
        self.telemetry_path = Path(telemetry_path)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpts = self.vault.list_checkpoints()
        train_docs, val_docs = synthesize_bilingual_curriculum(seed=42, scale_factor=2)

        if ckpts:
            latest_tag = self._select_latest_tag([c.get("folder", c["tag"]) for c in ckpts])
            self.model, self.tokenizer, manifest = self.vault.load_checkpoint(
                latest_tag, prefer_wsm=True, device=self.device
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
        self._auto_morphogenesis = True
        self._bg_steps_session = 0
        self._bg_tokens_since_molt = 0
        self._last_bg_ce: float | None = None
        self._last_bg_tok_s: float = 0.0
        self._last_bg_error: str | None = None
        self._active_chats = 0
        self._bg_thread: threading.Thread | None = None

        self._status_cache: dict[str, Any] = {}
        self._refresh_status_cache()

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

    def _refresh_status_cache(self) -> None:
        """Build an atomic lock-free status dictionary so /api/status responds in <1ms."""
        pct_next_molt = min(
            100.0,
            round(100.0 * self._bg_tokens_since_molt / max(1, self.AUTO_MOLT_TOKEN_INTERVAL), 1),
        )
        self._status_cache = {
            "background_training": self._bg_running,
            "auto_morphogenesis": self._auto_morphogenesis,
            "bg_steps_session": self._bg_steps_session,
            "bg_last_ce_loss": self._last_bg_ce,
            "bg_tok_per_sec": self._last_bg_tok_s,
            "bg_next_molt_pct": pct_next_molt,
            "bg_tokens_since_molt": self._bg_tokens_since_molt,
            "bg_molt_interval_tokens": self.AUTO_MOLT_TOKEN_INTERVAL,
            "bg_last_error": self._last_bg_error,
            "device": str(self.device),
            "total_tokens_trained": self.total_tokens_trained,
            "replay_buffer_size": len(self.dataset.replay_reservoir),
            "anatomy": self.model.inspect_anatomy(),
            "wsm": self.wsm.describe(),
            "memories_count": len(self.memory.units),
            "plugins": self.registry.list_plugins(),
            "checkpoints": self.vault.list_checkpoints(),
            "recent_telemetry": list(self.telemetry_history[-80:]),
        }

    def status(self) -> dict[str, Any]:
        """Lock-free instant read — never blocked by Muon backward passes!"""
        snap = dict(self._status_cache)
        snap["background_training"] = self._bg_running
        snap["auto_morphogenesis"] = self._auto_morphogenesis
        snap["bg_steps_session"] = self._bg_steps_session
        return snap

    def learn_dialogue_exchange(
        self,
        question: str,
        thought: str,
        answer: str,
        priority: float = 2.5,
    ) -> dict[str, Any]:
        """Inject a dialogue exchange directly into the Continual Replay Reservoir & EvolveMem."""
        doc = (
            f"{BOS_TOKEN}{USER_TOKEN}{question.strip()}{NEVER_TOKEN}"
            f"{THINK_START}{thought.strip()}{THINK_END}{answer.strip()}{EOS_TOKEN}"
        )
        ids = self.tokenizer.encode(doc)
        mask = self.tokenizer.build_target_loss_mask(ids)
        L = self.seq_len + 1
        if len(ids) < L:
            pad_n = L - len(ids)
            ids = ids + [0] * pad_n
            mask = mask + [0.0] * pad_n
        else:
            ids = ids[:L]
            mask = mask[:L]

        with self.lock:
            self.dataset.push_to_replay(ids, mask, surprise=priority)
            self.dataset.train_sequences.append((ids, mask))
            self._refresh_status_cache()

        return {"ok": True, "replay_size": len(self.dataset.replay_reservoir)}

    def _run_single_micro_step(self, batch_size: int = 2) -> dict[str, float]:
        """Single fast (~0.4s) training micro-step for smooth responsive Continuous Growth."""
        t0 = time.perf_counter()
        with self.lock:
            self.model.train()
            batch = self.dataset.sample_batch(
                batch_size=batch_size,
                split="train",
                replay_ratio=0.30,
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
            self._bg_steps_session += 1
            self._bg_tokens_since_molt += tokens_step

            elapsed = max(1e-4, time.perf_counter() - t0)
            tok_s = round(tokens_step / elapsed, 1)
            self._last_bg_ce = round(out.ce_loss, 4)
            self._last_bg_tok_s = tok_s

            if self._bg_steps_session % 8 == 0:
                self.wsm.record_snapshot(self.model)

            rec = {
                "type": "train_step",
                "timestamp": round(time.time(), 2),
                "stage": self.model.cfg.stage_name,
                "generation": self.model.cfg.generation,
                "global_step": int(opt_stats["step"]),
                "total_tokens": self.total_tokens_trained,
                "ce_loss": self._last_bg_ce,
                "mtp_loss": round(out.mtp_loss, 4),
                "grad_norm": round(opt_stats["grad_norm"], 4),
                "tok_per_sec": tok_s,
            }
            self._append_telemetry(rec)
            self._refresh_status_cache()
            return rec

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
                    replay_ratio=0.30,
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
                    self._refresh_status_cache()

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
            self._refresh_status_cache()

        return {
            "steps_executed": num_steps,
            "mean_train_ce": round(sum(losses) / max(1, len(losses)), 4),
            "final_train_ce": round(losses[-1], 4) if losses else 0.0,
            **eval_metrics,
        }

    def trigger_morphogenesis(self, mode: str = "next_stage") -> dict[str, Any]:
        """
        Execute verified Zero-Shock Morphogenesis on the live brain!
        Supported modes: 'next_stage' | 'depth' | 'ffn' | 'moe_upcycle' | 'moe_double' | 'auto'
        """
        with self.lock:
            pre_eval = self.eval_harness.evaluate(self.model)
            old_counts = self.model.count_parameters()
            old_stage = self.model.cfg.stage_name

            if mode == "auto":
                if self.model.cfg.stage_name == "Seed-0":
                    mode = "next_stage"
                elif not self.model.cfg.use_moe:
                    mode = "next_stage"
                elif self.model.cfg.n_routed_experts < 8:
                    mode = "moe_double"
                else:
                    mode = "ffn"

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
                new_ff = ((int(self.model.cfg.d_ff * 1.25) + 63) // 64) * 64
                grown = grow_ffn_mitosis(self.model, new_d_ff=new_ff)
            elif mode == "moe_upcycle":
                grown = transition_seed1_to_sprout_moe(self.model, n_routed_experts=4, top_k=2)
            elif mode == "moe_double":
                if self.model.cfg.use_moe:
                    grown = grow_moe_experts(self.model, double_experts=True)
                else:
                    grown = transition_seed1_to_sprout_moe(self.model, n_routed_experts=4, top_k=2)
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
            self._bg_tokens_since_molt = 0

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
                metrics=post_eval,
                consolidated_state_dict=None,
            )
            self._refresh_status_cache()
            return event

    def consolidate_knowledge_seal(
        self, title: str, text: str, steps: int = 8
    ) -> dict[str, Any]:
        with self.lock:
            self.memory.store(
                key=title.lower().replace(" ", "_")[:48],
                content=text,
                category="semantic",
                tags=["seal", title.lower()],
            )
            rep = self.seal.run_sleep_consolidation(
                self.model,
                self.optimizer,
                knowledge_items=[(title, text)],
                steps=steps,
                batch_size=4,
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
            self._refresh_status_cache()
            return rep

    def _get_recent_thread_messages(self, thread_id: str | None) -> list[dict[str, Any]]:
        if not thread_id:
            return []
        t = self.conversations.get_thread(thread_id)
        if not t:
            return []
        return list(t.get("messages", []))[-8:]

    def chat(
        self,
        message: str,
        max_new_tokens: int = 140,
        temperature: float = 0.2,
        thread_id: str | None = None,
        owner: str = "local",
    ) -> dict[str, Any]:
        self._active_chats += 1
        try:
            recent_hist = self._get_recent_thread_messages(thread_id)
            with self.lock:
                result = self.harness.run_turn(
                    message,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    owner=owner,
                    recent_history=recent_hist,
                )
                learned = result.pop("learned_pair", None)
                if learned:
                    self.learn_dialogue_exchange(*learned, priority=3.0)
                if thread_id:
                    thread = self.conversations.append_exchange(
                        thread_id=thread_id,
                        user_message=message,
                        assistant_payload=result,
                        owner=owner,
                    )
                    result["thread_id"] = thread["id"]
                    result["thread_title"] = thread["title"]
                self._refresh_status_cache()
                return result
        finally:
            self._active_chats = max(0, self._active_chats - 1)

    def chat_stream(
        self,
        message: str,
        max_new_tokens: int = 140,
        temperature: float = 0.2,
        thread_id: str | None = None,
        owner: str = "local",
    ) -> Iterator[dict[str, Any]]:
        """Stream live SSE events (`think_delta`, `tool_call`, `answer_delta`, `done`)."""
        q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        aborted_flag = threading.Event()

        def _worker() -> None:
            self._active_chats += 1
            try:
                recent_hist = self._get_recent_thread_messages(thread_id)
                with self.lock:
                    res = self.harness.run_turn(
                        message,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        owner=owner,
                        recent_history=recent_hist,
                        on_event=lambda ev: (False if aborted_flag.is_set() else (q.put(ev) or True)),
                    )
                    learned = res.pop("learned_pair", None)
                    if learned:
                        self.learn_dialogue_exchange(*learned, priority=3.0)
                    if thread_id:
                        thread = self.conversations.append_exchange(
                            thread_id=thread_id,
                            user_message=message,
                            assistant_payload=res,
                            owner=owner,
                        )
                        res["thread_id"] = thread["id"]
                        res["thread_title"] = thread["title"]
                    self._refresh_status_cache()
                    q.put({"type": "done", "result": res})
            except Exception as exc:  # noqa: BLE001
                q.put({"type": "error", "error": str(exc)})
            finally:
                self._active_chats = max(0, self._active_chats - 1)
                q.put(None)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            while True:
                ev = q.get()
                if ev is None:
                    break
                yield ev
        finally:
            aborted_flag.set()

    def start_background_growth_loop(
        self,
        batch_size: int = 2,
        auto_morphogenesis: bool = True,
    ) -> bool:
        self._auto_morphogenesis = auto_morphogenesis
        if self._bg_running and self._bg_thread is not None and self._bg_thread.is_alive():
            self._refresh_status_cache()
            return False

        self._bg_running = True
        self._last_bg_error = None
        self._refresh_status_cache()

        def _worker() -> None:
            steps_since_save = 0
            while self._bg_running:
                # Yield immediately whenever a user chat is generating!
                if self._active_chats > 0:
                    time.sleep(0.12)
                    continue
                try:
                    self._run_single_micro_step(batch_size=batch_size)
                    steps_since_save += 1

                    # Auto-Morphogenesis milestone check!
                    if (
                        self._auto_morphogenesis
                        and self._bg_tokens_since_molt >= self.AUTO_MOLT_TOKEN_INTERVAL
                        and self.model.count_parameters()["total_params"] < 95_000_000
                    ):
                        self.trigger_morphogenesis(mode="auto")
                        steps_since_save = 0
                    elif steps_since_save >= 30:
                        with self.lock:
                            cons_sd = self.wsm.build_consolidated_state_dict(fallback_model=self.model)
                            self.vault.save_checkpoint(
                                "latest",
                                self.model,
                                self.tokenizer,
                                metrics={"total_tokens_trained": self.total_tokens_trained},
                                consolidated_state_dict=cons_sd,
                            )
                        steps_since_save = 0
                except Exception as exc:  # noqa: BLE001
                    self._last_bg_error = str(exc)
                    self._bg_running = False
                    self._refresh_status_cache()
                    break

                # Cooperative yield so HTTP & UI stay silky-smooth at 60fps
                time.sleep(0.35)

        self._bg_thread = threading.Thread(target=_worker, daemon=True, name="never-ai-continual-growth")
        self._bg_thread.start()
        return True

    def stop_background_growth_loop(self) -> bool:
        self._bg_running = False
        self._refresh_status_cache()
        return True
