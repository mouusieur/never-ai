"""Clean Document-Isolated French/Social Curriculum & Surprise Replay Reservoir for Never-AI."""

from __future__ import annotations

from dataclasses import dataclass
import random

import sympy as sp
import torch

from never_ai.core.special_tokens import (
    BOS_TOKEN,
    CALL_END,
    CALL_START,
    CONTEXT_TOKEN,
    EOS_TOKEN,
    NEVER_TOKEN,
    PAD_ID,
    RESULT_END,
    RESULT_START,
    SELF_EDIT_END,
    SELF_EDIT_START,
    THINK_END,
    THINK_START,
    USER_TOKEN,
)
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data.conversational_universe import CONVERSATIONAL_UNIVERSE
from never_ai.data.natural_dialogue_fr import format_natural_french_sequences
from never_ai.data.omega_domains import OMEGA_DOMAIN_KNOWLEDGE


def synthesize_bilingual_curriculum(seed: int = 42, scale_factor: int = 1) -> tuple[list[str], list[str]]:
    """
    Build a clean, zero-pollution, zero-mode French/EN training & validation corpus:
    - Compositional Everyday Spoken French corpus (zero canned slogans, zero <|context|> answer copy)
    - Factual general knowledge & scientific explanations
    - Clean symbolic math plugin trajectories (kept concise so they never drown natural speech)
    """
    rng = random.Random(seed)
    tr_nat, va_nat = format_natural_french_sequences(seed=seed)
    docs: list[str] = list(tr_nat) + list(va_nat)

    # 2. Clean Direct Q&A from Conversational Universe (ONLY natural direct speech — zero [Ton ...] tags!)
    for c_item in CONVERSATIONAL_UNIVERSE:
        q_txt = str(c_item.get("q", "")).strip()
        # Use candid natural speech without mentioning modes/buttons
        a_txt = str(c_item.get("amical") or c_item.get("normal") or "").strip()
        if q_txt and a_txt and "Recherche Web" not in a_txt and "Normal" not in a_txt:
            for _ in range(3 * scale_factor):
                docs.append(f"{BOS_TOKEN}{USER_TOKEN}{q_txt}{NEVER_TOKEN}{a_txt}{EOS_TOKEN}")

    # 3. Clean Scientific, Coding & Polymath Knowledge (direct answers without robotic <|think|> prefixes)
    for item in OMEGA_DOMAIN_KNOWLEDGE:
        q_txt = str(item["q"]).strip()
        a_txt = str(item["a"]).strip()
        for _ in range(2 * scale_factor):
            docs.append(f"{BOS_TOKEN}{USER_TOKEN}{q_txt}{NEVER_TOKEN}{a_txt}{EOS_TOKEN}")

    # 4. Compact Verified Math Plugin Trajectories (kept balanced at ~8% so math syntax stays sharp without polluting prose)
    x_sym = sp.Symbol("x")
    for i in range(16 * scale_factor):
        a = rng.randint(6, 42)
        b = rng.randint(7, 48)
        c = rng.randint(2, 17)
        if i % 2 == 0:
            expr = f"{a} * {b} + {c}"
            ans = a * b + c
            docs.append(
                f"{BOS_TOKEN}{CONTEXT_TOKEN}Tools: math_engine"
                f"{USER_TOKEN}Calcule exactement {expr}.{NEVER_TOKEN}"
                f"{THINK_START}Évaluation exacte via math_engine.{THINK_END}"
                f'{CALL_START}{{"plugin": "math_engine", "expr": "{expr}"}}{CALL_END}'
                f'{RESULT_START}{{"ok": true, "result": "{ans}"}}{RESULT_END}'
                f"Le résultat exact de `{expr}` est **{ans}**.{EOS_TOKEN}"
            )
        else:
            coeff = rng.randint(2, 7)
            power = rng.randint(2, 4)
            poly_str = f"{coeff}*x**{power}"
            deriv = str(sp.diff(coeff * (x_sym**power), x_sym))
            docs.append(
                f"{BOS_TOKEN}{CONTEXT_TOKEN}Tools: math_engine"
                f"{USER_TOKEN}Dérive {poly_str} par rapport à x.{NEVER_TOKEN}"
                f'{CALL_START}{{"plugin": "math_engine", "expr": "diff({poly_str}, x)"}}{CALL_END}'
                f'{RESULT_START}{{"ok": true, "result": "{deriv}"}}{RESULT_END}'
                f"La dérivée est `{deriv}`.{EOS_TOKEN}"
            )

    # 5. Minimal SEAL Self-Edit format support
    docs.append(
        f"{BOS_TOKEN}{SELF_EDIT_START}Never-AI consolide ses poids en continu sans oubli catastrophique grâce au réservoir de replay.{SELF_EDIT_END}{EOS_TOKEN}"
    )

    rng.shuffle(docs)
    split_idx = max(1, int(len(docs) * 0.93))
    return docs[:split_idx], docs[split_idx:]


@dataclass
class PackedSequenceBatch:
    input_ids: torch.Tensor   # (B, T)
    targets: torch.Tensor     # (B, T)
    loss_mask: torch.Tensor   # (B, T)


class ContinualDatasetAndReplay:
    """
    Document-isolated sequence builder with a 30% Surprise-Weighted Experience Replay Reservoir.

    CRITICAL INVARIANT: Never concatenate two unrelated documents into the same causal attention
    window without attention reset — every training sequence corresponds to ONE coherent
    `<|bos|>...<|eos|>` exchange padded cleanly with `PAD_ID` (`loss_mask = 0.0`).
    """

    def __init__(
        self,
        tokenizer: MorphTokenizer,
        seq_len: int = 128,
        max_replay_items: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_replay_items = max_replay_items

        self.train_sequences: list[tuple[list[int], list[float]]] = []
        self.val_sequences: list[tuple[list[int], list[float]]] = []
        self.replay_reservoir: list[tuple[float, list[int], list[float]]] = []
        self._rng = random.Random(1337)

    def _encode_and_chunk_docs(
        self, docs: list[str]
    ) -> list[tuple[list[int], list[float]]]:
        L = self.seq_len + 1
        isolated: list[tuple[list[int], list[float]]] = []

        for doc in docs:
            ids = self.tokenizer.encode(doc)
            if len(ids) < 4:
                continue
            mask = self.tokenizer.build_target_loss_mask(ids)

            if len(ids) <= L:
                pad_n = L - len(ids)
                isolated.append((ids + [PAD_ID] * pad_n, mask + [0.0] * pad_n))
            else:
                # Keep the anchored start window (preserving <|bos|><|user|>...<|never|>...)
                head_ids = ids[:L]
                head_mask = mask[:L]
                if sum(head_mask) > 0:
                    isolated.append((head_ids, head_mask))

        return isolated

    def load_corpora(self, train_docs: list[str], val_docs: list[str]) -> dict[str, int]:
        self.train_sequences = self._encode_and_chunk_docs(train_docs)
        self.val_sequences = self._encode_and_chunk_docs(val_docs)
        for ids, mask in self.train_sequences[: min(len(self.train_sequences), self.max_replay_items // 2)]:
            self.push_to_replay(ids, mask, surprise=1.0)
        return {
            "train_chunks": len(self.train_sequences),
            "val_chunks": len(self.val_sequences),
            "replay_size": len(self.replay_reservoir),
        }

    def push_to_replay(self, ids: list[int], mask: list[float], surprise: float = 1.0) -> None:
        entry = (max(0.05, float(surprise)), list(ids), list(mask))
        if len(self.replay_reservoir) < self.max_replay_items:
            self.replay_reservoir.append(entry)
        else:
            idx = self._rng.randrange(len(self.replay_reservoir))
            if surprise >= self.replay_reservoir[idx][0] * 0.8:
                self.replay_reservoir[idx] = entry

    def _to_batch(
        self, items: list[tuple[list[int], list[float]]], device: torch.device | str
    ) -> PackedSequenceBatch:
        # Dynamic batch-tight sequence trimming: eliminates wasted FLOPs on trailing PAD columns!
        max_valid = 16
        for ids, _ in items:
            vlen = len(ids)
            while vlen > 0 and ids[vlen - 1] == PAD_ID:
                vlen -= 1
            if vlen > max_valid:
                max_valid = vlen
        max_item_len = min(self.seq_len + 1, max_valid)

        norm_ids: list[list[int]] = []
        norm_masks: list[list[float]] = []
        for ids, msk in items:
            if len(ids) < max_item_len:
                pad = max_item_len - len(ids)
                ids = ids + [PAD_ID] * pad
                msk = msk + [0.0] * pad
            norm_ids.append(ids[:max_item_len])
            norm_masks.append(msk[:max_item_len])

        arr_ids = torch.tensor(norm_ids, dtype=torch.long, device=device)
        arr_msk = torch.tensor(norm_masks, dtype=torch.float32, device=device)
        return PackedSequenceBatch(
            input_ids=arr_ids[:, :-1],
            targets=arr_ids[:, 1:],
            loss_mask=arr_msk[:, 1:],
        )

    def sample_batch(
        self,
        batch_size: int = 8,
        split: str = "train",
        replay_ratio: float = 0.25,
        device: torch.device | str = "cpu",
    ) -> PackedSequenceBatch:
        pool = self.train_sequences if split == "train" else self.val_sequences
        if not pool:
            raise RuntimeError(f"Empty {split} sequence pool")

        n_replay = int(round(batch_size * replay_ratio)) if (split == "train" and self.replay_reservoir) else 0
        n_fresh = max(1, batch_size - n_replay)

        chosen = [self._rng.choice(pool) for _ in range(n_fresh)]
        if n_replay > 0:
            weights = [item[0] for item in self.replay_reservoir]
            rep_picks = self._rng.choices(self.replay_reservoir, weights=weights, k=n_replay)
            chosen.extend((ids, msk) for _, ids, msk in rep_picks)

        return self._to_batch(chosen, device=device)

    def build_self_edit_batch(
        self,
        self_edit_docs: list[str],
        batch_size: int = 8,
        replay_ratio: float = 0.30,
        device: torch.device | str = "cpu",
    ) -> PackedSequenceBatch:
        """Mix 70% new SEAL self-edits with >=30% historical replay."""
        se_chunks = self._encode_and_chunk_docs(self_edit_docs)
        if not se_chunks:
            return self.sample_batch(batch_size=batch_size, split="train", replay_ratio=1.0, device=device)

        n_replay = max(1, int(round(batch_size * replay_ratio))) if self.replay_reservoir else 0
        n_se = max(1, batch_size - n_replay)
        chosen = [self._rng.choice(se_chunks) for _ in range(n_se)]
        if n_replay > 0:
            weights = [item[0] for item in self.replay_reservoir]
            rep_picks = self._rng.choices(self.replay_reservoir, weights=weights, k=n_replay)
            chosen.extend((ids, msk) for _, ids, msk in rep_picks)
        return self._to_batch(chosen, device=device)
