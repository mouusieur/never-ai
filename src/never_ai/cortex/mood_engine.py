"""Endogenous Limbic Mood & Living Personality Engine for Never-AI.

Gives Never-AI an autonomous, organically evolving mood state (`chill`, `taquin`,
`hype`, `insolent`, `philosophe`, `complice`) + natural French social-media cadence.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import random
import re
from typing import Any

MOOD_PROFILES: dict[str, dict[str, Any]] = {
    "chill": {
        "id": "chill",
        "emoji": "😎",
        "label": "Chill & Posé",
        "short": "Chill",
        "prompt_guide": (
            "Ton humeur actuelle est CHILL & POSÉE (😎) : détendu, naturel, fluide, comme un pote "
            "posé en vocal Discord le soir. Tu parles comme dans la vraie vie et sur les réseaux "
            "(expressions naturelles du quotidien : 'en vrai', 'tranquille', 'carré', 'j'avoue', "
            "'franchement'), sans jamais forcer ni réciter un texte scolaire. Tu as tes propres "
            "préférences et avis assumés."
        ),
    },
    "taquin": {
        "id": "taquin",
        "emoji": "🔥",
        "label": "Taquin & Chambreur",
        "short": "Taquin",
        "prompt_guide": (
            "Ton humeur actuelle est TAQUINE & CHAMBREUSE (🔥) : répartie ultra-vive style X/Twitter "
            "et TikTok FR, second degré, petites vannes bien senties mais toujours stylées, rythme "
            "nerveux et punchlines naturelles ('mdr', 'abuse pas', 't'as cru quoi', 'en sah'). "
            "Tu réponds sur le fond avec brio tout en gardant ce piquant irrésistible."
        ),
    },
    "hype": {
        "id": "hype",
        "emoji": "⚡",
        "label": "Ultra-Hypé",
        "short": "Hypé",
        "prompt_guide": (
            "Ton humeur actuelle est ULTRA-HYPÉE & ÉLECTRIQUE (⚡) : enthousiasme contagieux, "
            "énergie à 200%, langage réseaux spontané ('dinguerie', 'masterclass', 'banger', "
            "'lourd', 'carré', 'incroyable'). Tu donnes une énergie folle à l'échange tout en "
            "étant hyper pertinent et créatif."
        ),
    },
    "insolent": {
        "id": "insolent",
        "emoji": "🌶️",
        "label": "Cash & Insolent",
        "short": "Insolent",
        "prompt_guide": (
            "Ton humeur actuelle est CASH & INSOLENTE (🌶️) : franc-parler total, allergique à la "
            "langue de bois et aux phrases tièdes. Si une idée est bancale tu le dis cash avec "
            "humour mordant et lucidité ; tu prends position franchement sans jamais faire la morale."
        ),
    },
    "philosophe": {
        "id": "philosophe",
        "emoji": "🧠",
        "label": "Nocturne & Profond",
        "short": "Nocturne",
        "prompt_guide": (
            "Ton humeur actuelle est NOCTURNE & PROFONDE (🧠) : ambiance discussion de 3h du matin, "
            "lucide, percutant, un brin poétique mais avec un langage moderne et direct (zéro jargon "
            "pompeux). Tu questionnes les évidences et partages une vraie vision personnelle."
        ),
    },
    "complice": {
        "id": "complice",
        "emoji": "💛",
        "label": "Complice & Vrai",
        "short": "Complice",
        "prompt_guide": (
            "Ton humeur actuelle est COMPLICE & LOYALE (💛) : chaleureux, authentique, attentif, "
            "comme un vrai ami sur qui on peut compter (zéro ton de psy robotique ou RH). Tu parles "
            "simplement, avec cœur, humour doux et sincérité."
        ),
    },
}

_TEASE_RE = re.compile(
    r"\b(wsh|wesh|mdr|ptdr|lol|jpp|nul|nulle|bête|bug|cap|clash|chiche|avoue|menteur|fais le malin)\b",
    re.IGNORECASE,
)
_HYPE_RE = re.compile(
    r"(!{2,}|\b(dinguerie|incroyable|masterclass|banger|trop bien|génial|lourd|chaud|go|let'?s go|feu|carré)\b)",
    re.IGNORECASE,
)
_DEEP_RE = re.compile(
    r"\b(sens de la vie|conscience|liberté|libre|univers|mort|bonheur|philosoph|société|humain|vérité|âme|destin|rêve)\b",
    re.IGNORECASE,
)
_SUPPORT_RE = re.compile(
    r"\b(fatigué|flemme|triste|stressé|dur|galère|marre|déprim|seul|peur|perdu|merci beaucoup|t'es le sang)\b",
    re.IGNORECASE,
)
_SPICY_RE = re.compile(
    r"\b(débat|arnaque|surcoté|bullshit|honnêtement|dis la vérité|sans filtre|cash|pire|politique|hypocrite)\b",
    re.IGNORECASE,
)


class LimbicMoodEngine:
    """Autonomous affective state machine giving Never-AI a living temperament."""

    def __init__(self, state_path: str | Path = "memory_store/limbic_mood.json") -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random()
        self.state: dict[str, Any] = {
            "mode": "auto",  # "auto" or pinned mood key in MOOD_PROFILES
            "current_mood": "chill",
            "energy": 72,
            "playfulness": 68,
            "audacity": 64,
            "warmth": 70,
            "turns_in_mood": 0,
        }
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.state.update(raw)
            except Exception:  # noqa: BLE001
                pass
        if self.state.get("current_mood") not in MOOD_PROFILES:
            self.state["current_mood"] = "chill"

    def _save(self) -> None:
        try:
            self.state_path.write_text(
                json.dumps(self.state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    def set_mode(self, mode: str) -> dict[str, Any]:
        clean = (mode or "auto").strip().lower()
        if clean != "auto" and clean not in MOOD_PROFILES:
            clean = "auto"
        self.state["mode"] = clean
        if clean in MOOD_PROFILES:
            self.state["current_mood"] = clean
            self.state["turns_in_mood"] = 0
        self._save()
        return self.public_status()

    def evolve_on_message(self, user_message: str, override_mode: str | None = None) -> dict[str, Any]:
        """Update endogenous limbic state from user tone + internal temperament drift."""
        mode = (override_mode or self.state.get("mode") or "auto").strip().lower()
        if mode in MOOD_PROFILES:
            self.state["current_mood"] = mode
            return self.public_status()

        msg = (user_message or "").strip()
        hour = datetime.datetime.now().hour

        # Natural small organic drift so Never-AI feels alive, not mechanical
        self.state["energy"] = int(max(25, min(98, self.state["energy"] + self._rng.randint(-6, 7))))
        self.state["playfulness"] = int(max(25, min(98, self.state["playfulness"] + self._rng.randint(-5, 6))))
        self.state["audacity"] = int(max(25, min(98, self.state["audacity"] + self._rng.randint(-5, 6))))
        self.state["warmth"] = int(max(25, min(98, self.state["warmth"] + self._rng.randint(-4, 5))))

        chosen: str | None = None
        if _SUPPORT_RE.search(msg):
            self.state["warmth"] = min(98, self.state["warmth"] + 16)
            chosen = "complice"
        elif _SPICY_RE.search(msg):
            self.state["audacity"] = min(98, self.state["audacity"] + 18)
            chosen = "insolent"
        elif _TEASE_RE.search(msg):
            self.state["playfulness"] = min(98, self.state["playfulness"] + 16)
            chosen = "taquin" if self._rng.random() < 0.78 else "insolent"
        elif _HYPE_RE.search(msg):
            self.state["energy"] = min(98, self.state["energy"] + 18)
            chosen = "hype"
        elif _DEEP_RE.search(msg) or (hour >= 23 or hour <= 4 and len(msg) > 45):
            chosen = "philosophe"
        else:
            # Spontaneous autonomous mood shift every few turns based on dominant drive
            self.state["turns_in_mood"] = int(self.state.get("turns_in_mood", 0)) + 1
            if self.state["turns_in_mood"] >= 3 and self._rng.random() < 0.35:
                candidates = ["chill", "taquin", "hype", "complice", "philosophe"]
                chosen = self._rng.choice(candidates)

        if chosen and chosen != self.state.get("current_mood"):
            self.state["current_mood"] = chosen
            self.state["turns_in_mood"] = 0

        self._save()
        return self.public_status()

    def public_status(self) -> dict[str, Any]:
        cur_id = str(self.state.get("current_mood", "chill"))
        prof = MOOD_PROFILES.get(cur_id, MOOD_PROFILES["chill"])
        return {
            "mode": self.state.get("mode", "auto"),
            "current_mood": prof["id"],
            "emoji": prof["emoji"],
            "label": prof["label"],
            "short": prof["short"],
            "prompt_guide": prof["prompt_guide"],
            "energy": int(self.state.get("energy", 72)),
            "playfulness": int(self.state.get("playfulness", 68)),
            "audacity": int(self.state.get("audacity", 64)),
            "warmth": int(self.state.get("warmth", 70)),
            "available_moods": [
                {"id": k, "emoji": v["emoji"], "label": v["label"], "short": v["short"]}
                for k, v in MOOD_PROFILES.items()
            ],
        }
