"""
MIT SEAL Sleep-Time Self-Edit Consolidation + DeepSeek-R1-Zero Style Verifiable GRPO
+ Immutable Anti-Goodhart Evaluation Harness (Directive 8).
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

import torch
import torch.nn.functional as F

from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import (
    BOS_TOKEN,
    CALL_END,
    CALL_START,
    EOS_TOKEN,
    NEVER_TOKEN,
    SELF_EDIT_END,
    SELF_EDIT_START,
    THINK_END,
    THINK_START,
    USER_TOKEN,
)
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data.curriculum import ContinualDatasetAndReplay
from never_ai.optim.composite_muon import CompositeMuonOptimizer
from never_ai.plugins.base import PluginRegistry


class ImmutableEvalHarness:
    """
    Read-only objective benchmark suite enforcing Revert-on-Regression (Directive 8).
    """

    def __init__(
        self,
        dataset: ContinualDatasetAndReplay,
        eval_batches: int = 6,
        batch_size: int = 4,
    ) -> None:
        self.fixed_batches = [
            dataset.sample_batch(batch_size=batch_size, split="val", replay_ratio=0.0)
            for _ in range(eval_batches)
        ]

    @torch.no_grad()
    def evaluate(self, model: ChronoMorphLM) -> dict[str, float]:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        ce_losses: list[float] = []
        mtp_losses: list[float] = []
        for b in self.fixed_batches:
            out = model(
                b.input_ids.to(device),
                targets=b.targets.to(device),
                loss_mask=b.loss_mask.to(device),
                update_titans=True,
            )
            ce_losses.append(out.ce_loss)
            mtp_losses.append(out.mtp_loss)

        if was_training:
            model.train()

        mean_ce = sum(ce_losses) / max(1, len(ce_losses))
        mean_mtp = sum(mtp_losses) / max(1, len(mtp_losses))
        ppl = math.exp(min(20.0, mean_ce))
        return {
            "val_ce_loss": round(mean_ce, 4),
            "val_mtp_loss": round(mean_mtp, 4),
            "val_perplexity": round(ppl, 3),
        }


class SEALConsolidator:
    """
    MIT SEAL (arXiv:2506.10943) Sleep-Time Self-Adapting Engine:
    1. Synthesizes multi-perspective atomic implications & reasoning traces (<|self_edit|>)
       from raw new knowledge passages.
    2. Mixes 70% synthetic self-edits with 30% Surprise-Weighted Historical Replay.
    3. Updates synaptic weights via CompositeMuon and guards against catastrophic drift
       with automatic Revert-on-Regression!
    """

    def __init__(
        self,
        dataset: ContinualDatasetAndReplay,
        eval_harness: ImmutableEvalHarness,
        max_regression_ratio: float = 0.03,
    ) -> None:
        self.dataset = dataset
        self.eval_harness = eval_harness
        self.max_regression_ratio = max_regression_ratio

    @staticmethod
    def synthesize_self_edits(title: str, raw_passage: str) -> list[str]:
        """Decompose raw passage into structured study sheets optimized for weight retention."""
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", raw_passage.strip())
            if len(s.strip()) > 15
        ]
        if not sentences:
            sentences = [raw_passage.strip()]

        edits: list[str] = []
        # 1. Atomic implication sheet
        bullets = " ; ".join(sentences[:4])
        edits.append(
            f"{BOS_TOKEN}{SELF_EDIT_START}[{title}] Implications atomiques : {bullets}{SELF_EDIT_END}{EOS_TOKEN}"
        )

        # 2. Active Reasoning QA synthesis
        core_fact = sentences[0]
        edits.append(
            f"{BOS_TOKEN}{USER_TOKEN}Explique et synthétise le concept : {title}.{NEVER_TOKEN}"
            f"{THINK_START}Analysons les faits fondamentaux associés à '{title}' : {core_fact}{THINK_END}"
            f"{raw_passage.strip()}{EOS_TOKEN}"
        )

        # 3. English/Bilingual analytical transfer
        edits.append(
            f"{BOS_TOKEN}{USER_TOKEN}What is the key principle regarding {title}?{NEVER_TOKEN}"
            f"{THINK_START}Retrieving consolidated parametric knowledge about {title}.{THINK_END}"
            f"Regarding **{title}**: {bullets}{EOS_TOKEN}"
        )
        return edits

    def run_sleep_consolidation(
        self,
        model: ChronoMorphLM,
        optimizer: CompositeMuonOptimizer,
        knowledge_items: list[tuple[str, str]],
        steps: int = 12,
        batch_size: int = 6,
    ) -> dict[str, Any]:
        pre_eval = self.eval_harness.evaluate(model)
        backup_state = copy.deepcopy(model.state_dict())

        all_edits: list[str] = []
        for title, passage in knowledge_items:
            all_edits.extend(self.synthesize_self_edits(title, passage))

        device = next(model.parameters()).device
        model.train()
        step_losses: list[float] = []

        for _ in range(steps):
            batch = self.dataset.build_self_edit_batch(
                all_edits, batch_size=batch_size, replay_ratio=0.30, device=device
            )
            optimizer.zero_grad()
            out = model(batch.input_ids, targets=batch.targets, loss_mask=batch.loss_mask)
            assert out.loss is not None
            out.loss.backward()
            optimizer.step(model, lr_override_scale=0.25)
            model.update_moe_biases()
            step_losses.append(out.ce_loss)

        post_eval = self.eval_harness.evaluate(model)
        allowed_ceiling = pre_eval["val_ce_loss"] * (1.0 + self.max_regression_ratio) + 0.05

        reverted = False
        if post_eval["val_ce_loss"] > allowed_ceiling:
            model.load_state_dict(backup_state)
            reverted = True
        else:
            # Promote consolidated self-edits into long-term replay reservoir!
            for ed in all_edits:
                ids = self.dataset.tokenizer.encode(ed)
                msk = self.dataset.tokenizer.build_target_loss_mask(ids)
                self.dataset.push_to_replay(ids, msk, surprise=1.5)

        return {
            "num_self_edits": len(all_edits),
            "steps": steps,
            "mean_edit_loss": round(sum(step_losses) / max(1, len(step_losses)), 4),
            "pre_val_ce": pre_eval["val_ce_loss"],
            "post_val_ce": post_eval["val_ce_loss"],
            "reverted_on_regression": reverted,
        }


_NUM_RE = re.compile(r"-?\d+")


class GRPOReasoningTrainer:
    """
    DeepSeek-R1-Zero inspired Group Relative Policy Optimization (GRPO):
    Reinforces open, unconstrained `<|think|>...<|/think|>` reasoning purely via
    verifiable mathematical / plugin outcomes — zero human stylistic censorship!
    """

    def __init__(
        self,
        tokenizer: MorphTokenizer,
        registry: PluginRegistry,
        group_size: int = 4,
        lr: float = 1e-4,
    ) -> None:
        self.tokenizer = tokenizer
        self.registry = registry
        self.group_size = group_size
        self.lr = lr

    def score_trajectory(self, generated_text: str, ground_truth_int: int) -> float:
        reward = 0.0
        # 1. Structural cognitive scratchpad bonus (encourages free deliberation)
        if THINK_START in generated_text and THINK_END in generated_text:
            reward += 0.25
        # 2. Valid plugin invocation bonus if attempted
        if CALL_START in generated_text and CALL_END in generated_text:
            m = re.search(re.escape(CALL_START) + r"(.*?)" + re.escape(CALL_END), generated_text, re.DOTALL)
            if m:
                try:
                    payload = json.loads(m.group(1).strip())
                    if isinstance(payload, dict) and "plugin" in payload:
                        reward += 0.25
                except json.JSONDecodeError:
                    reward -= 0.1
        # 3. Objective verifiable answer accuracy
        nums = [int(x) for x in _NUM_RE.findall(generated_text)]
        if ground_truth_int in nums:
            reward += 1.0
            if nums and nums[-1] == ground_truth_int:
                reward += 0.5
        return reward

    def step_grpo(
        self,
        model: ChronoMorphLM,
        prompt_text: str,
        ground_truth_int: int,
        max_new_tokens: int = 48,
    ) -> dict[str, Any]:
        device = next(model.parameters()).device
        prompt_ids = self.tokenizer.encode(prompt_text)

        trajectories: list[list[int]] = []
        rewards: list[float] = []

        for _ in range(self.group_size):
            gen_ids, _ = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.85,
                top_k=32,
            )
            if not gen_ids:
                gen_ids = [self.tokenizer.encode("0")[0]]
            gen_txt = self.tokenizer.decode(gen_ids)
            r = self.score_trajectory(gen_txt, ground_truth_int)
            trajectories.append(gen_ids)
            rewards.append(r)

        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = r_t.std()
        if float(std.item()) < 1e-6:
            return {
                "mean_reward": round(float(r_t.mean().item()), 4),
                "max_reward": round(float(r_t.max().item()), 4),
                "policy_loss": 0.0,
                "updated": False,
            }

        advantages = (r_t - r_t.mean()) / (std + 1e-5)

        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        opt.zero_grad()

        total_pg_loss = torch.tensor(0.0, device=device)
        P = len(prompt_ids)

        for gen_ids, adv in zip(trajectories, advantages):
            full_seq = prompt_ids + gen_ids
            inp = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)
            tgt = torch.tensor([full_seq[1:]], dtype=torch.long, device=device)

            out = model(inp, update_titans=False)
            # Slice response token logits starting at index P-1
            resp_logits = out.logits[0, P - 1 :, :]
            resp_targets = tgt[0, P - 1 :]
            log_probs = -F.cross_entropy(resp_logits, resp_targets, reduction="mean")
            total_pg_loss = total_pg_loss - (adv * log_probs) / self.group_size

        total_pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        return {
            "mean_reward": round(float(r_t.mean().item()), 4),
            "max_reward": round(float(r_t.max().item()), 4),
            "policy_loss": round(float(total_pg_loss.detach().item()), 5),
            "updated": True,
        }
