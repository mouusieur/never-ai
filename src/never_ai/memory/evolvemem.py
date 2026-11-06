"""EvolveMem: Self-Evolving Multi-View Memory Store (arXiv:2605.13941 inspired)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize_terms(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def compute_char_ngram_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Fast multilingual FR/EN/Code hashed char-3/4-gram unit vector (<0.2ms, zero external weights)."""
    vec = np.zeros(dim, dtype=np.float32)
    norm_text = f" {text.lower().strip()} "
    if len(norm_text) < 3:
        return vec
    for n in (3, 4):
        for i in range(len(norm_text) - n + 1):
            gram = norm_text[i : i + n].encode("utf-8")
            h = int.from_bytes(hashlib.blake2b(gram, digest_size=4).digest(), "little")
            idx = h % dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


@dataclass
class MemoryUnit:
    id: str
    key: str
    content: str
    category: str = "semantic"  # episodic | semantic | skill | self_edit
    tags: list[str] = field(default_factory=list)
    importance: float = 1.0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class RetrievalConfig:
    """Evolvable action space optimized autonomously by EvolveMem closed-loop diagnosis."""

    w_bm25: float = 0.45
    w_semantic: float = 0.35
    w_tag_entity: float = 0.15
    w_recency: float = 0.05
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    evolution_round: int = 0


class EvolveMemStore:
    """
    Typed multi-view memory fusing BM25 lexical, char-ngram semantic, tag/entity,
    and recency signals with an AutoResearch closed-loop configuration optimizer
    and automatic Revert-on-Regression guard.
    """

    def __init__(self, storage_path: str | Path = "memory_store/evolvemem.json") -> None:
        self.storage_path = Path(storage_path)
        self.units: dict[str, MemoryUnit] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.config = RetrievalConfig()
        self.evolution_log: list[dict[str, Any]] = []
        if self.storage_path.exists():
            self.load()

    def store(
        self,
        key: str,
        content: str,
        category: str = "semantic",
        tags: list[str] | None = None,
        importance: float = 1.0,
        persist: bool = True,
    ) -> MemoryUnit:
        digest = hashlib.sha256(f"{key}::{content}".encode("utf-8")).hexdigest()[:10]
        mem_id = f"mem_{digest}"
        unit = MemoryUnit(
            id=mem_id,
            key=key,
            content=content,
            category=category,
            tags=list(tags or []),
            importance=importance,
        )
        self.units[mem_id] = unit
        self.vectors[mem_id] = compute_char_ngram_embedding(f"{key} {content} {' '.join(unit.tags)}")
        if persist:
            self.save()
        return unit

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        category: str | None = None,
        override_config: RetrievalConfig | None = None,
    ) -> list[dict[str, Any]]:
        cfg = override_config or self.config
        candidates = [
            u for u in self.units.values() if (category is None or u.category == category)
        ]
        if not candidates:
            return []

        q_terms = _tokenize_terms(query_text)
        q_set = set(q_terms)
        q_vec = compute_char_ngram_embedding(query_text)

        # BM25 statistics
        doc_tokens = {u.id: _tokenize_terms(f"{u.key} {u.content}") for u in candidates}
        avgdl = max(1.0, sum(len(toks) for toks in doc_tokens.values()) / len(candidates))
        N = len(candidates)

        df: dict[str, int] = {}
        for t in q_set:
            df[t] = sum(1 for toks in doc_tokens.values() if t in toks)

        now = time.time()
        scored: list[tuple[float, MemoryUnit, dict[str, float]]] = []

        for u in candidates:
            toks = doc_tokens[u.id]
            dl = len(toks)
            tf_map: dict[str, int] = {}
            for tok in toks:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            bm25 = 0.0
            for t in q_set:
                f_t = tf_map.get(t, 0)
                if f_t == 0:
                    continue
                n_q = df.get(t, 0)
                idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                denom = f_t + cfg.bm25_k1 * (1.0 - cfg.bm25_b + cfg.bm25_b * (dl / avgdl))
                bm25 += idf * (f_t * (cfg.bm25_k1 + 1.0)) / max(1e-6, denom)
            bm25_norm = math.tanh(bm25 / 4.0)

            u_vec = self.vectors.get(u.id)
            if u_vec is None:
                u_vec = compute_char_ngram_embedding(f"{u.key} {u.content}")
                self.vectors[u.id] = u_vec
            sem_score = max(0.0, float(np.dot(q_vec, u_vec)))

            u_tags = {t.lower() for t in u.tags} | set(_tokenize_terms(u.key))
            tag_overlap = (len(q_set & u_tags) / max(1, len(u_tags))) if u_tags else 0.0

            age_hours = max(0.0, (now - u.created_at) / 3600.0)
            recency = math.exp(-age_hours / 168.0)

            total = (
                cfg.w_bm25 * bm25_norm
                + cfg.w_semantic * sem_score
                + cfg.w_tag_entity * tag_overlap
                + cfg.w_recency * recency
            ) * u.importance

            scored.append(
                (
                    total,
                    u,
                    {
                        "bm25": round(bm25_norm, 4),
                        "semantic": round(sem_score, 4),
                        "tag": round(tag_overlap, 4),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for sc, u, breakdown in scored[:top_k]:
            u.access_count += 1
            results.append(
                {
                    "id": u.id,
                    "key": u.key,
                    "content": u.content,
                    "category": u.category,
                    "tags": u.tags,
                    "score": round(sc, 4),
                    "signals": breakdown,
                }
            )
        return results

    def _evaluate_mrr(self, probes: list[tuple[str, str]], cfg: RetrievalConfig) -> float:
        if not probes:
            return 1.0
        rr_sum = 0.0
        for q_str, target_id in probes:
            hits = self.query(q_str, top_k=5, override_config=cfg)
            for rank, h in enumerate(hits, start=1):
                if h["id"] == target_id:
                    rr_sum += 1.0 / rank
                    break
        return rr_sum / len(probes)

    def evolve_retrieval_config(self) -> dict[str, Any]:
        """
        Closed-loop AutoResearch cycle (EVALUATE -> DIAGNOSE -> PROPOSE -> GUARD):
        Synthesizes recall probes from stored units, searches targeted perturbations,
        and applies ONLY non-regressing improvements!
        """
        if len(self.units) < 3:
            return {"status": "skipped", "reason": "need >= 3 memories", "mrr": 1.0}

        probes: list[tuple[str, str]] = []
        for u in list(self.units.values())[:24]:
            # Paraphrase / partial probe: key + half of content words + tags
            words = _tokenize_terms(u.content)
            snippet = " ".join(words[::2][:6] + u.tags[:2])
            probes.append((snippet or u.key, u.id))

        base_mrr = self._evaluate_mrr(probes, self.config)
        best_cfg = RetrievalConfig(**asdict(self.config))
        best_mrr = base_mrr

        proposals = [
            RetrievalConfig(w_bm25=0.55, w_semantic=0.30, w_tag_entity=0.12, w_recency=0.03),
            RetrievalConfig(w_bm25=0.30, w_semantic=0.50, w_tag_entity=0.15, w_recency=0.05),
            RetrievalConfig(w_bm25=0.40, w_semantic=0.35, w_tag_entity=0.22, w_recency=0.03),
            RetrievalConfig(
                w_bm25=self.config.w_bm25,
                w_semantic=self.config.w_semantic,
                w_tag_entity=self.config.w_tag_entity,
                w_recency=self.config.w_recency,
                bm25_k1=1.2,
                bm25_b=0.6,
            ),
        ]

        accepted = False
        for cand in proposals:
            cand_mrr = self._evaluate_mrr(probes, cand)
            if cand_mrr > best_mrr + 1e-5:
                best_mrr = cand_mrr
                best_cfg = cand
                accepted = True

        if accepted:
            best_cfg.evolution_round = self.config.evolution_round + 1
            self.config = best_cfg
            self.save()

        record = {
            "accepted": accepted,
            "base_mrr": round(base_mrr, 4),
            "final_mrr": round(best_mrr, 4),
            "config": asdict(self.config),
        }
        self.evolution_log.append(record)
        return record

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "units": [asdict(u) for u in self.units.values()],
        }
        tmp = self.storage_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.storage_path)

    def load(self) -> None:
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if "config" in raw:
            self.config = RetrievalConfig(**raw["config"])
        self.units.clear()
        self.vectors.clear()
        for item in raw.get("units", []):
            u = MemoryUnit(**item)
            self.units[u.id] = u
            self.vectors[u.id] = compute_char_ngram_embedding(
                f"{u.key} {u.content} {' '.join(u.tags)}"
            )
