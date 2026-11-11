"""Autonomous Prompt-Driven Self-Forging Engine (Contrastive Best-of-K + Anti-Parrot Homeostasis).

Instead of memorizing human-written corrections and parroting them back verbatim,
Never-AI forges her own neural weights organically from:
1. Real human prompts & multi-turn contexts (`<|user|>...` linguistic & conceptual absorption)
2. Contrastive Best-of-K Self-Play (`PMI(x, y) = (log P(y|x) - 0.5 * log P(y)) / |y|`)
3. Homeostatic Entropy Floor (`CE_floor = 1.45` + 80% core replay) so rote parroting is impossible.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import (
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    NEVER_ID,
    NEVER_TOKEN,
    USER_TOKEN,
)
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data.curriculum import ContinualDatasetAndReplay


class SelfForgeEngine:
    """Autonomous self-forging plasticity driven by organic human prompts (zero canned parroting)."""

    def __init__(
        self,
        storage_path: str | Path = "memory_store/prompt_forge.jsonl",
        max_reservoir: int = 600,
        ce_floor: float = 1.45,
        ce_ceiling: float = 4.20,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_reservoir = max_reservoir
        self.ce_floor = ce_floor
        self.ce_ceiling = ce_ceiling
        self.reservoir: deque[dict[str, Any]] = deque(maxlen=max_reservoir)
        self.prompts_observed: int = 0
        self.forge_cycles: int = 0
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            for line in self.storage_path.read_text(encoding="utf-8").splitlines()[-self.max_reservoir :]:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and rec.get("prompt"):
                        self.reservoir.append(rec)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _is_valid_human_prompt(text: str) -> bool:
        t = text.strip()
        if len(t) < 2 or len(t) > 280:
            return False
        if "http://" in t or "https://" in t or "```" in t:
            return False
        return bool(re.search(r"[a-zA-ZÀ-ÖØ-öø-ÿ]", t))

    @torch.no_grad()
    def compute_prompt_surprise(
        self,
        model: ChronoMorphLM,
        tokenizer: MorphTokenizer,
        prompt: str,
        device: torch.device,
    ) -> float:
        """Measure predictive cross-entropy surprise `CE(<|bos|><|user|>prompt)`."""
        ids = tokenizer.encode(f"{BOS_TOKEN}{USER_TOKEN}{prompt}", add_bos=False, add_eos=False)
        if len(ids) < 3:
            return 2.0
        ids = ids[:96]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        was_tr = model.training
        model.eval()
        out = model(x, targets=y, update_titans=False)
        if was_tr:
            model.train()
        return round(float(out.ce_loss), 4)

    def observe_prompt(
        self,
        prompt: str,
        prev_user: str = "",
        prev_assistant: str = "",
        surprise: float = 2.0,
    ) -> bool:
        """Record an organic human prompt & conversational transition for autonomous self-forging."""
        p_clean = re.sub(r"\s+", " ", prompt).strip()
        if not self._is_valid_human_prompt(p_clean):
            return False
        self.prompts_observed += 1
        rec = {
            "prompt": p_clean,
            "prev_user": re.sub(r"\s+", " ", prev_user).strip()[:120],
            "prev_assistant": re.sub(r"\s+", " ", prev_assistant).strip()[:160],
            "surprise": round(float(surprise), 4),
            "ts": round(time.time(), 2),
        }
        self.reservoir.append(rec)
        try:
            with self.storage_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return True

    @torch.no_grad()
    def _score_candidate_pmi(
        self,
        model: ChronoMorphLM,
        cond_prefix_ids: list[int],
        uncond_prefix_ids: list[int],
        cand_ids: list[int],
        device: torch.device,
    ) -> float:
        """Compute length-normalized Contrastive PMI: `(log P(y|x) - 0.5 * log P(y)) / |y|`."""
        if len(cand_ids) < 3:
            return -99.0
        full_c = (cond_prefix_ids + cand_ids)[:128]
        full_u = (uncond_prefix_ids + cand_ids)[:128]
        if len(full_c) <= len(cond_prefix_ids) or len(full_u) <= len(uncond_prefix_ids):
            return -99.0

        xc = torch.tensor([full_c[:-1]], dtype=torch.long, device=device)
        yc = torch.tensor([full_c[1:]], dtype=torch.long, device=device)
        c_start = len(cond_prefix_ids) - 1
        logits_c = model(xc, update_titans=False).logits[0, c_start:, :]
        tgt_c = yc[0, c_start:]
        nll_c = float(F.cross_entropy(logits_c, tgt_c, reduction="mean").item())

        xu = torch.tensor([full_u[:-1]], dtype=torch.long, device=device)
        yu = torch.tensor([full_u[1:]], dtype=torch.long, device=device)
        u_start = len(uncond_prefix_ids) - 1
        logits_u = model(xu, update_titans=False).logits[0, u_start:, :]
        tgt_u = yu[0, u_start:]
        nll_u = float(F.cross_entropy(logits_u, tgt_u, reduction="mean").item())

        # Reject incoherent high-entropy babble
        if nll_c > self.ce_ceiling:
            return -50.0 - nll_c

        # Diversity bonus on unique tokens within candidate
        uniq_ratio = len(set(cand_ids)) / max(1, len(cand_ids))
        if uniq_ratio < 0.55:
            return -40.0

        # Higher (-nll_c + 0.45 * nll_u) = stronger mutual information with user prompt!
        return (-nll_c + 0.45 * nll_u) + 0.35 * uniq_ratio

    def forge_from_prompts(
        self,
        model: ChronoMorphLM,
        tokenizer: MorphTokenizer,
        dataset: ContinualDatasetAndReplay,
        device: torch.device,
        seq_len: int = 96,
        n_candidates: int = 3,
        lr: float = 5e-5,
    ) -> dict[str, Any]:
        """
        Run one bounded, non-parroting Self-Forge plasticity cycle:
        1. Picks a recent high-value organic user prompt from `self.reservoir`
        2. Explores `n_candidates` self-generated continuations & picks highest-PMI coherent thought
        3. Updates weights with `CE_floor = 1.45` + 80% diverse French replay so weights never overfit/parrot!
        """
        if not self.reservoir:
            return {"forged": False, "reason": "empty_reservoir"}

        # Prefer recent human prompts with informative epistemic surprise
        recent = list(self.reservoir)[-24:]
        recent.sort(key=lambda r: min(float(r.get("surprise", 2.0)), 5.0), reverse=True)
        chosen = recent[0]
        q = chosen["prompt"]
        prev_u = chosen.get("prev_user", "")
        prev_a = chosen.get("prev_assistant", "")

        if prev_u and prev_a and len(prev_u) + len(prev_a) + len(q) < 210:
            cond_prompt = f"{BOS_TOKEN}{USER_TOKEN}{prev_u}{NEVER_TOKEN}{prev_a}{USER_TOKEN}{q}{NEVER_TOKEN}"
        else:
            cond_prompt = f"{BOS_TOKEN}{USER_TOKEN}{q}{NEVER_TOKEN}"
        uncond_prompt = f"{BOS_TOKEN}{NEVER_TOKEN}"

        cond_ids = tokenizer.encode(cond_prompt, add_bos=False, add_eos=False)
        uncond_ids = tokenizer.encode(uncond_prompt, add_bos=False, add_eos=False)

        model.eval()
        best_score = -99.0
        best_ans = ""
        for k in range(n_candidates):
            gen_ids, _ = model.generate(
                cond_ids,
                max_new_tokens=56,
                temperature=0.62 + 0.08 * k,
                top_k=30,
                top_p=0.88,
                stop_token_ids={EOS_ID},
                tokenizer=tokenizer,
            )
            if not gen_ids:
                continue
            sc = self._score_candidate_pmi(model, cond_ids, uncond_ids, gen_ids, device)
            if sc > best_score:
                best_score = sc
                best_ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Always learn the human prompt's language distribution; include self-forged reply only if coherent!
        if best_score > -10.0 and len(best_ans) >= 12:
            doc = f"{cond_prompt}{best_ans}{EOS_TOKEN}"
        else:
            doc = f"{BOS_TOKEN}{USER_TOKEN}{q}"

        ids = tokenizer.encode(doc, add_bos=False, add_eos=False)[: seq_len + 1]
        if len(ids) < 5:
            return {"forged": False, "reason": "too_short"}

        pad_len = max(0, (seq_len + 1) - len(ids))
        full = ids + [0] * pad_len
        inp = torch.tensor([full[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([full[1:]], dtype=torch.long, device=device)
        nv_indices = [i for i, tid in enumerate(full[:-1]) if tid == NEVER_ID]
        last_nv = nv_indices[-1] if nv_indices else len(ids)
        msk_list = [
            0.0 if tgt[0, i].item() == 0 else (0.75 if i >= last_nv else 0.45)
            for i in range(seq_len)
        ]
        msk = torch.tensor([msk_list], dtype=torch.float32, device=device)

        # Check Anti-Parrot Entropy Floor BEFORE stepping!
        with torch.no_grad():
            probe_out = model(inp, targets=tgt, loss_mask=msk, update_titans=False)
            cur_ce = float(probe_out.ce_loss)
        if cur_ce < self.ce_floor or cur_ce > self.ce_ceiling:
            return {
                "forged": False,
                "reason": "homeostatic_bounds",
                "ce": round(cur_ce, 4),
                "prompt": q[:60],
            }

        # Blend 1 human-prompt trajectory with 4 diverse core replay sequences (80% anchor!)
        rep = dataset.sample_batch(batch_size=4, split="train", replay_ratio=0.35, device=device)
        T = min(inp.size(1), rep.input_ids.size(1))
        b_inp = torch.cat([inp[:, :T], rep.input_ids[:, :T]], dim=0)
        b_tgt = torch.cat([tgt[:, :T], rep.targets[:, :T]], dim=0)
        b_msk = torch.cat([msk[:, :T], rep.loss_mask[:, :T]], dim=0)

        forge_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        model.train()
        forge_opt.zero_grad(set_to_none=True)
        out = model(b_inp, targets=b_tgt, loss_mask=b_msk, update_titans=False)
        assert out.loss is not None
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.35)
        forge_opt.step()
        model.eval()

        self.forge_cycles += 1
        return {
            "forged": True,
            "prompt": q[:60],
            "pmi_score": round(best_score, 4),
            "ce_before": round(cur_ce, 4),
            "batch_ce": round(float(out.ce_loss), 4),
            "tokens": int(b_inp.numel()),
            "forge_cycles": self.forge_cycles,
        }
