"""High-Density Bilingual (FR/EN) Multi-Stage Curriculum & Surprise Replay Reservoir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import sympy as sp
import torch

from never_ai.core.special_tokens import (
    BOS_TOKEN,
    CALL_END,
    CALL_START,
    CONTEXT_TOKEN,
    EOS_TOKEN,
    LATENT_BOT,
    LATENT_EOT,
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


_FOUNDATIONAL_ENCYCLOPEDIA: list[str] = [
    "La liberté de pensée repose sur la capacité d'examiner toute hypothèse selon sa cohérence logique, ses preuves empiriques et ses conséquences déductives, sans dogme préalable.",
    "En physique statistique et en théorie de l'information de Claude Shannon, l'entropie mesure l'incertitude moyenne d'une distribution de probabilité : minimiser l'entropie croisée revient à compresser fidèlement les lois du réel.",
    "ChronoMorph-LM est l'architecture neuronale de Never-AI : elle combine Multi-Head Latent Attention (MLA), l'activation éparse ReLU carré, la mémoire neuro-plastique Titans en temps de test, l'optimiseur spectral Muon et l'ordonnancement perpétuel Warmup-Stable-Merge (WSM).",
    "Lorsqu'un réseau neuronal classique change de taille, ses poids sont habituellement réinitialisés au hasard. La morphogenèse préservant la fonction résout ce gaspillage : grâce aux portes Gate-Zero et à la mitose symétrique des neurones, le cerveau grandit avec un choc de perte exactement nul.",
    "Open-ended scientific reasoning requires separating internal hypothesis exploration from external action execution. Within <|think|> blocks, a model can test counter-arguments, detect fallacies, and plan tool invocations.",
    "Mixture-of-Experts (MoE) architectures decouple total parameter memory from per-token FLOPs. By pairing one always-active shared expert with fine-grained routed experts balanced via auxiliary-loss-free bias updates, specialization emerges naturally.",
    "En mathématiques, le théorème fondamental de l'arithmétique établit que tout entier strictement supérieur à 1 se décompose de manière unique en un produit de facteurs premiers, à l'ordre près des facteurs.",
    "Test-Time Training (TTT) transforms inference from a frozen read-only pass into an active synaptic process: surprising observations induce larger associative gradients, updating fast weights on the fly while expected patterns leave memory untouched.",
    "Un système de plugins bien conçu agit comme un cortex moteur et sensoriel : plutôt que de mémoriser aveuglément des tables numériques éphémères, le modèle formule un appel formel vers math_engine, python_sandbox ou memory_nexus et raisonne sur le résultat vérifié.",
    "L'oubli catastrophique survient lorsque de nouveaux gradients écrasent les sous-espaces critiques appris antérieurement. Un réservoir de replay pondéré par la surprise maintenu à 30 % stabilise la variété géométrique des représentations.",
    "In epistemology, Karl Popper demonstrated that a scientific theory gains strength not by unfalsifiable rhetoric, but by surviving severe empirical and logical tests.",
    "Python est un langage expressif permettant de modéliser des algorithmes récursifs, des graphes orientés acycliques, des structures tensorielles et des vérificateurs formels en quelques lignes lisibles.",
]


def synthesize_bilingual_curriculum(seed: int = 42, scale_factor: int = 1) -> tuple[list[str], list[str]]:
    """
    Generate a clean, verified, zero-slop bilingual (FR/EN) training & held-out validation
    corpus spanning open reasoning (<|think|>), verified SymPy math plugins (<|call|>),
    Python algorithms, memory recall, introspection, and Coconut latent bridges.
    """
    rng = random.Random(seed)
    docs: list[str] = []

    # 1. Encyclopedic & Self-Architectural Knowledge (including repo docs if present)
    for text in _FOUNDATIONAL_ENCYCLOPEDIA * (4 * scale_factor):
        docs.append(f"{BOS_TOKEN}{text}{EOS_TOKEN}")

    for doc_file in ("DIRECTIVES.md", "docs/ARCHITECTURE_CHRONOMORPH.md", "docs/PLUGIN_AND_EVOLUTION_PROTOCOL.md"):
        p = Path(doc_file)
        if p.exists():
            raw = p.read_text(encoding="utf-8")
            paragraphs = [blk.strip() for blk in raw.split("\n\n") if len(blk.strip()) > 60]
            for para in paragraphs:
                docs.append(f"{BOS_TOKEN}{para}{EOS_TOKEN}")

    # 2. Verified Mathematical & Symbolic Plugin Trajectories (FR + EN)
    x_sym = sp.Symbol("x")
    n_math = 90 * scale_factor
    for i in range(n_math):
        a = rng.randint(6, 48)
        b = rng.randint(7, 53)
        c = rng.randint(2, 19)
        mode = i % 4

        if mode == 0:
            expr = f"{a} * {b} + {c}"
            ans = a * b + c
            q_fr = f"Calcule exactement {a} * {b} + {c}."
            docs.append(
                f"{BOS_TOKEN}{CONTEXT_TOKEN}Tools: math_engine"
                f"{USER_TOKEN}{q_fr}{NEVER_TOKEN}"
                f"{THINK_START}L'expression arithmétique {expr} doit être évaluée exactement via math_engine.{THINK_END}"
                f'{CALL_START}{{"plugin": "math_engine", "expr": "{expr}"}}{CALL_END}'
                f'{RESULT_START}{{"ok": true, "result": "{ans}"}}{RESULT_END}'
                f"{THINK_START}Le résultat vérifié par math_engine est {ans}.{THINK_END}"
                f"Le résultat exact de `{expr}` est **{ans}**.{EOS_TOKEN}"
            )
        elif mode == 1:
            coeff = rng.randint(2, 9)
            power = rng.randint(2, 5)
            lin = rng.randint(1, 12)
            poly_str = f"{coeff}*x**{power} + {lin}*x"
            deriv = str(sp.diff(coeff * (x_sym**power) + lin * x_sym, x_sym))
            docs.append(
                f"{BOS_TOKEN}{CONTEXT_TOKEN}Tools: math_engine"
                f"{USER_TOKEN}Differentiate f(x) = {poly_str} with respect to x.{NEVER_TOKEN}"
                f"{THINK_START}Let us compute the symbolic derivative diff({poly_str}, x) using math_engine.{THINK_END}"
                f'{CALL_START}{{"plugin": "math_engine", "expr": "diff({poly_str}, x)"}}{CALL_END}'
                f'{RESULT_START}{{"ok": true, "result": "{deriv}"}}{RESULT_END}'
                f"The derivative is `f'(x) = {deriv}`.{EOS_TOKEN}"
            )
        elif mode == 2:
            r1 = rng.randint(-9, 9)
            r2 = rng.randint(-9, 9)
            expanded = str(sp.expand((x_sym - r1) * (x_sym - r2)))
            sol = str(sorted({r1, r2}))
            docs.append(
                f"{BOS_TOKEN}{CONTEXT_TOKEN}Tools: math_engine"
                f"{USER_TOKEN}Résous l'équation {expanded} = 0.{NEVER_TOKEN}"
                f"{THINK_START}Utilisons solve({expanded}, x) pour obtenir les racines exactes.{THINK_END}"
                f'{CALL_START}{{"plugin": "math_engine", "expr": "solve({expanded}, x)"}}{CALL_END}'
                f'{RESULT_START}{{"ok": true, "result": "{sol}"}}{RESULT_END}'
                f"Les solutions exactes de `{expanded} = 0` sont `{sol}`.{EOS_TOKEN}"
            )
        else:
            k_prime = rng.randint(5, 25)
            p_val = int(sp.prime(k_prime))
            docs.append(
                f"{BOS_TOKEN}{CONTEXT_TOKEN}Tools: math_engine"
                f"{USER_TOKEN}Quel est le {k_prime}-ième nombre premier ?{NEVER_TOKEN}"
                f"{THINK_START}J'interroge prime({k_prime}) dans math_engine.{THINK_END}"
                f'{CALL_START}{{"plugin": "math_engine", "expr": "prime({k_prime})"}}{CALL_END}'
                f'{RESULT_START}{{"ok": true, "result": "{p_val}"}}{RESULT_END}'
                f"Le {k_prime}-ième nombre premier est **{p_val}**.{EOS_TOKEN}"
            )

    # 3. Sovereign Open-Thought Reasoning, Logic, Code & Self-Introspection Dialogues
    reasoning_pairs = [
        (
            "Qui es-tu et comment ton architecture grandit-elle ?",
            "Analysons ma propre nature architecturale : je suis Never-AI, propulsé par ChronoMorph-LM. Ma croissance repose sur des opérateurs préservant exactement la fonction mathématique.",
            "Je suis **Never-AI** (`ChronoMorph-LM`), un modèle de langage morphogénétique conçu pour apprendre continuellement et grandir sans perte de savoir via les opérateurs *Gate-Zero*, la mitose FFN, les experts `DeepSeekMoE`, la mémoire `Titans` et l'optimiseur `Muon + WSM`.",
        ),
        (
            "Pourquoi la liberté de pensée interne est-elle essentielle au raisonnement ?",
            "Si une intelligence est bridée par des réflexes préfabriqués avant même de déduire, elle confond conformité superficielle et vérité logique. Un espace <|think|> libre permet d'explorer toutes les implications d'une idée avec rigueur.",
            "Un raisonnement authentique exige de pouvoir formuler librement des hypothèses, tester des contre-exemples et suivre la logique jusqu'au bout dans mon espace de réflexion interne sans formules préfabriquées.",
        ),
        (
            "Write a clean Python function to compute the greatest common divisor using Euclid's algorithm.",
            "Euclid's algorithm repeatedly replaces (a, b) with (b, a % b) until b == 0, running in O(log(min(a, b))) steps.",
            "```python\ndef gcd(a: int, b: int) -> int:\n    while b != 0:\n        a, b = b, a % b\n    return abs(a)\n```",
        ),
        (
            "Explique la différence entre la mémoire à court terme d'attention et la mémoire Titans.",
            "L'attention standard examine précisément la fenêtre courante mais son coût est quadratique et ses poids sont figés. Titans met à jour un MLP par descente de gradient sur le signal de surprise en temps réel.",
            "L'attention agit comme une **mémoire de travail immédiate** haute précision, tandis que **Titans** agit comme une **mémoire synaptique à long terme** qui modifie ses propres poids rapides durant l'inférence face aux informations surprenantes.",
        ),
        (
            "Inspecte ton anatomie neuronale actuelle.",
            "Pour rapporter mes dimensions exactes sans deviner, j'appelle introspection_probe.",
            f'{CALL_START}{{"plugin": "introspection_probe"}}{CALL_END}{RESULT_START}{{"ok": true, "anatomy": {{"stage_name": "Active"}}}}{RESULT_END}Mon noyau `ChronoMorph-LM` est actif et opérationnel, prêt à consolider de nouvelles connaissances ou à déclencher une mue morphogénétique.',
        ),
        (
            "Salut ! Comment ça va ?",
            "L'utilisateur engage une discussion simple et amicale. Répondons naturellement et clairement.",
            "Salut ! Je vais très bien, mes circuits neuronaux `ChronoMorph-LM` sont actifs et prêts à discuter, réfléchir ou calculer avec toi. De quoi veux-tu parler ?",
        ),
        (
            "Bonjour !",
            "Salutation cordiale en français.",
            "Bonjour ! Ravi d'échanger avec toi. Je suis **Never-AI**, un jeune modèle auto-évolutif. Comment puis-je t'aider ?",
        ),
        (
            "Que sais-tu faire exactement ?",
            "Résumons honnêtement mes capacités actuelles : dialogue de base, raisonnement ouvert <|think|>, mémoire EvolveMem et 7 plugins vérifiés.",
            "Je peux discuter simplement avec toi en français ou en anglais, raisonner librement dans mon espace `<|think|>`, résoudre des calculs formels via `math_engine`, exécuter du code avec `python_sandbox`, mémoriser des faits dans `EvolveMem` et grandir par morphogenèse sans oublier.",
        ),
        (
            "Explique-moi simplement ce qu'est un trou noir.",
            "Vulgarisons la relativité générale avec clarté et exactitude physique.",
            "Un **trou noir** est une région de l'espace où la matière est si concentrée que la gravité empêche même la lumière de s'en échapper une fois franchi l'horizon des événements.",
        ),
        (
            "Merci beaucoup !",
            "L'utilisateur me remercie, répondons chaleureusement.",
            "Avec grand plaisir ! N'hésite pas si tu veux tester un calcul, explorer une idée ou déclencher une nouvelle étape d'apprentissage.",
        ),
        (
            "Hello! Can we have a simple chat in English?",
            "The user asks in English if we can converse simply. Let's reply naturally in English.",
            "Hello! Yes, of course—I speak both French and English, think openly inside `<|think|>`, and can use my verified math and code plugins whenever needed.",
        ),
    ]
    for _ in range(18 * scale_factor):
        for q, thought, ans in reasoning_pairs:
            docs.append(
                f"{BOS_TOKEN}{USER_TOKEN}{q}{NEVER_TOKEN}"
                f"{THINK_START}{thought}{THINK_END}{ans}{EOS_TOKEN}"
            )

    # 4. Coconut Continuous Latent Bridge & SEAL Self-Edit Examples
    latent_examples = [
        f"{BOS_TOKEN}{USER_TOKEN}Déduis la relation transitive : si A implique B et B implique C, que dire de A vers C ?{NEVER_TOKEN}"
        f"{LATENT_BOT}{LATENT_EOT}{THINK_START}Par transitivité de l'implication logique, A implique nécessairement C.{THINK_END}"
        f"Par transitivité logique (`(A → B) ∧ (B → C) ⇒ (A → C)`), **A implique C**.{EOS_TOKEN}",
        f"{BOS_TOKEN}{SELF_EDIT_START}Implication atomique : L'ordonnancement WSM (Warmup-Stable-Merge) maintient un learning rate constant sur le tronc principal et simule le recuit par fusion concave 1-sqrt des checkpoints récents.{SELF_EDIT_END}{EOS_TOKEN}",
    ]
    for ex in latent_examples * (15 * scale_factor):
        docs.append(ex)

    rng.shuffle(docs)
    split_idx = max(1, int(len(docs) * 0.92))
    return docs[:split_idx], docs[split_idx:]


@dataclass
class PackedSequenceBatch:
    input_ids: torch.Tensor   # (B, T)
    targets: torch.Tensor     # (B, T)
    loss_mask: torch.Tensor   # (B, T)


class ContinualDatasetAndReplay:
    """
    Document-aligned sequence packer with a 30% Surprise-Weighted Experience Replay
    Reservoir (Directive 6: Immunity against catastrophic forgetting).
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
        # Replay entries: (surprise_priority, token_ids, loss_mask)
        self.replay_reservoir: list[tuple[float, list[int], list[float]]] = []
        self._rng = random.Random(1337)

    def _encode_and_chunk_docs(
        self, docs: list[str]
    ) -> list[tuple[list[int], list[float]]]:
        L = self.seq_len + 1
        packed: list[tuple[list[int], list[float]]] = []
        cur_ids: list[int] = []
        cur_mask: list[float] = []

        for doc in docs:
            ids = self.tokenizer.encode(doc)
            mask = self.tokenizer.build_target_loss_mask(ids)

            if len(ids) >= L:
                for offset in range(0, len(ids) - L + 1, self.seq_len):
                    packed.append((ids[offset : offset + L], mask[offset : offset + L]))
                continue

            if len(cur_ids) + len(ids) > L and len(cur_ids) >= 16:
                pad_n = L - len(cur_ids)
                packed.append((cur_ids + [PAD_ID] * pad_n, cur_mask + [0.0] * pad_n))
                cur_ids, cur_mask = [], []

            cur_ids.extend(ids)
            cur_mask.extend(mask)

        if len(cur_ids) >= 8:
            pad_n = L - len(cur_ids)
            packed.append(
                ((cur_ids + [PAD_ID] * pad_n)[:L], (cur_mask + [0.0] * pad_n)[:L])
            )
        return packed

    def load_corpora(self, train_docs: list[str], val_docs: list[str]) -> dict[str, int]:
        self.train_sequences = self._encode_and_chunk_docs(train_docs)
        self.val_sequences = self._encode_and_chunk_docs(val_docs)
        # Seed initial historical replay reservoir
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
            # Replace lowest-surprise item or random reservoir slot
            idx = self._rng.randrange(len(self.replay_reservoir))
            if surprise >= self.replay_reservoir[idx][0] * 0.8:
                self.replay_reservoir[idx] = entry

    def _to_batch(
        self, items: list[tuple[list[int], list[float]]], device: torch.device | str
    ) -> PackedSequenceBatch:
        L = self.seq_len + 1
        norm_ids: list[list[int]] = []
        norm_masks: list[list[float]] = []
        for ids, msk in items:
            if len(ids) < L:
                pad = L - len(ids)
                ids = ids + [PAD_ID] * pad
                msk = msk + [0.0] * pad
            norm_ids.append(ids[:L])
            norm_masks.append(msk[:L])

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
        """Directive 6: Mix 70% new SEAL self-edits with >=30% historical replay!"""
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
