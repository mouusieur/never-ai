"""Persistent Scene State Graph & Synaptic Experience Anchor Store (Never-AI v5.0).

Maintains spatial, postural, attire, and narrative continuity across long multi-turn
Roleplay (`Parole + Description`) and stores user-edited / preferred turns as live
cosine-retrieved trajectory anchors + ShareGPT/LoRA training records.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from never_ai.memory.evolvemem import compute_char_ngram_embedding


_LOCATION_RE = re.compile(
    r"\b(?:dans|sur|contre|au milieu de|au fond de|près de)\s+"
    r"(?:la |le |l'|les |un |une |mon |ma |ton |ta |notre )?"
    r"(chambre|salon|cuisine|salle de bain|douche|voiture|bureau|couloir|bar|hôtel|"
    r"appartement|ascenseur|forêt|plage|jardin|rue|ruelle|cave|grenier|terrasse|balcon|"
    r"lit|canapé|fauteuil|table|mur|porte|fenêtre)\b",
    re.IGNORECASE,
)

_POSTURE_RE = re.compile(
    r"\b(plaqu[ée]s?\s+contre\s+le\s+mur|contre\s+le\s+mur|sur\s+le\s+lit|allong[ée]s?|"
    r"à\s+genoux|agenouill[ée]s?|assis(?:e)?\s+sur|debout\s+devant|pench[ée]s?\s+vers|"
    r"dans\s+(?:mes|tes|ses)\s+bras|à\s+califourchon|dos\s+au\s+mur|blotti(?:e)?|"
    r"face\s+à\s+face|coll[ée]s?\s+contre)\b",
    re.IGNORECASE,
)

_ATTIRE_RE = re.compile(
    r"\b(chemise|robe|jupe|pantalon|jean|caleçon|culotte|soutien-gorge|veste|manteau|"
    r"t-shirt|pull|cravate|ceinture|fermeture\s+éclair|bouton[s]?|déboutonn[ée]|"
    r"dévêtu[e]?|nu[e]?|torse\s+nu|pieds\s+nus)\b",
    re.IGNORECASE,
)

_ACTION_SPAN_RE = re.compile(r"\*([^*\n]{4,180})\*")


def extract_scene_graph(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract live spatial, postural, and sensory continuity from thread messages."""
    locations: list[str] = []
    postures: list[str] = []
    attire: list[str] = []
    actions: list[str] = []
    has_rp_markers = False

    for m in messages[-30:]:
        txt = str(m.get("content") or m.get("visible_answer") or "").strip()
        if not txt:
            continue
        if "«" in txt or _ACTION_SPAN_RE.search(txt):
            has_rp_markers = True
        for hit in _ACTION_SPAN_RE.findall(txt):
            clean = hit.strip()
            if clean and clean not in actions:
                actions.append(clean)
        for loc_m in _LOCATION_RE.finditer(txt):
            val = loc_m.group(0).strip()
            if val.lower() not in {x.lower() for x in locations}:
                locations.append(val)
        for pos_m in _POSTURE_RE.finditer(txt):
            val = pos_m.group(0).strip()
            if val.lower() not in {x.lower() for x in postures}:
                postures.append(val)
        for att_m in _ATTIRE_RE.finditer(txt):
            val = att_m.group(0).strip().lower()
            if val not in attire:
                attire.append(val)

    active = bool(has_rp_markers or locations or postures)
    cur_loc = locations[-1] if locations else ""
    cur_pos = postures[-1] if postures else ""
    cur_att = attire[-3:] if attire else []
    recent_acts = actions[-2:] if actions else []

    parts: list[str] = []
    if cur_loc:
        parts.append(f"Lieu: {cur_loc}")
    if cur_pos:
        parts.append(f"Posture: {cur_pos}")
    if cur_att:
        parts.append(f"Tenue/éléments: {', '.join(cur_att)}")
    if recent_acts:
        parts.append(f"Dernier geste: *{recent_acts[-1][:95]}*")

    return {
        "active": active,
        "location": cur_loc,
        "posture": cur_pos,
        "attire": cur_att,
        "recent_actions": recent_acts,
        "summary": " · ".join(parts),
    }


class ExperienceAnchorStore:
    """Stores user-edited / preferred turns for cosine recall & LoRA adapter export."""

    def __init__(self, store_path: str | Path = "memory_store/sovereign_experiences.jsonl") -> None:
        self.store_path = Path(store_path)
        self.lora_path = self.store_path.with_name("sovereign_lora_dataset.jsonl")
        self.records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            for line in self.store_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
            self.records = self.records[-200:]
        except Exception:  # noqa: BLE001
            self.records = []

    def record(self, prompt: str, preferred_reply: str, kind: str = "edit") -> dict[str, Any]:
        p_clean = prompt.strip()
        r_clean = preferred_reply.strip()
        if not p_clean or not r_clean:
            return {"ok": False}
        entry = {
            "ts": round(time.time(), 2),
            "kind": kind,
            "prompt": p_clean[:600],
            "reply": r_clean[:1200],
        }
        self.records.append(entry)
        self.records = self.records[-200:]
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.store_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            lora_row = {
                "conversations": [
                    {"from": "human", "value": p_clean},
                    {"from": "gpt", "value": r_clean},
                ]
            }
            with self.lora_path.open("a", encoding="utf-8") as lfh:
                lfh.write(json.dumps(lora_row, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return {"ok": True, "total_anchors": len(self.records)}

    def retrieve_similar(self, query: str, top_k: int = 1, min_sim: float = 0.28) -> list[dict[str, Any]]:
        if not self.records or not query.strip():
            return []
        q_vec = compute_char_ngram_embedding(query, dim=256)
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in self.records[-80:]:
            r_vec = compute_char_ngram_embedding(rec.get("prompt", ""), dim=256)
            sim = float(np.dot(q_vec, r_vec))
            if sim >= min_sim:
                scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]
