"""Morphogenetic Byte-Level BPE Tokenizer with Convex-Hull Subword Projection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from never_ai.core.special_tokens import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    RESULT_END_ID,
    RESULT_START_ID,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    USER_ID,
    NEVER_ID,
    CONTEXT_ID,
)


class MorphTokenizer:
    """Byte-level BPE tokenizer supporting lossless vocabulary growth across stages."""

    def __init__(self, backend: Tokenizer) -> None:
        self.backend = backend
        self._verify_special_ids()

    def _verify_special_ids(self) -> None:
        for tok, expected_id in SPECIAL_TOKEN_IDS.items():
            actual_id = self.backend.token_to_id(tok)
            if actual_id != expected_id:
                raise ValueError(
                    f"Special token {tok} has ID {actual_id}, expected {expected_id}"
                )

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
    ) -> "MorphTokenizer":
        base_alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_required = len(SPECIAL_TOKENS) + len(base_alphabet)
        if vocab_size < min_required:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= {min_required} (16 specials + 256 byte tokens)"
            )

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=base_alphabet,
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        # Ensure special tokens are registered as atomic AddedTokens
        tok.add_special_tokens(list(SPECIAL_TOKENS))
        return cls(tok)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "MorphTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        clean_ids = [int(i) for i in ids if int(i) != PAD_ID]
        return self.backend.decode(clean_ids, skip_special_tokens=skip_special_tokens)

    def build_target_loss_mask(self, token_ids: list[int]) -> list[float]:
        """
        Compute per-next-token loss mask in {0.0, 1.0}.
        Masks out external context, user prompt, and external plugin <|result|>...<|/result|> spans
        so Never-AI trains its capacity on its own <|think|>, <|call|>, <|self_edit|>, and answers.
        For pure pretraining documents (without <|user|> / <|never|>), returns all 1.0s.
        """
        has_turns = (USER_ID in token_ids) or (NEVER_ID in token_ids)
        if not has_turns and (RESULT_START_ID not in token_ids):
            return [1.0 if tid != PAD_ID else 0.0 for tid in token_ids]

        mask: list[float] = []
        in_assistant = not has_turns
        in_plugin_result = False

        for tid in token_ids:
            if tid == PAD_ID:
                mask.append(0.0)
                continue
            if tid in (USER_ID, CONTEXT_ID):
                in_assistant = False
                mask.append(0.0)
                continue
            if tid == NEVER_ID:
                in_assistant = True
                mask.append(0.0)  # role header itself is given by harness
                continue
            if tid == RESULT_START_ID:
                in_plugin_result = True
                mask.append(0.0)
                continue
            if tid == RESULT_END_ID:
                in_plugin_result = False
                mask.append(0.0)
                continue

            if in_assistant and not in_plugin_result:
                mask.append(1.0)
            else:
                mask.append(0.0)
        return mask

    @staticmethod
    def compute_subword_decomposition_map(
        old_tok: "MorphTokenizer",
        new_tok: "MorphTokenizer",
    ) -> list[list[int]]:
        """
        For every token ID v in [0, new_tok.vocab_size), return the constituent list of token IDs
        in old_tok whose embeddings average to initialize v in the Convex Hull of old_tok.
        """
        old_vocab = old_tok.backend.get_vocab()
        mapping: list[list[int]] = []

        for new_id in range(new_tok.vocab_size):
            tok_str = new_tok.backend.id_to_token(new_id)
            if tok_str is not None and tok_str in old_vocab:
                mapping.append([old_vocab[tok_str]])
                continue

            # Decode this single BPE piece to raw text and re-tokenize with old_tok
            decoded = new_tok.backend.decode([new_id], skip_special_tokens=False)
            sub_ids = old_tok.encode(decoded, add_bos=False, add_eos=False)
            if not sub_ids:
                sub_ids = [PAD_ID]
            mapping.append(sub_ids)

        return mapping
