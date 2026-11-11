"""Plugin SDK, MCP-Compatible Schema Export, and ToolRAG Dynamic Router."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any

import numpy as np

from never_ai.memory.evolvemem import compute_char_ngram_embedding


_WORD_RE = re.compile(r"\w+", re.UNICODE)


class NeverPlugin(ABC):
    """Base class for all builtin and SkillForge-synthesized Never-AI plugins."""

    name: str = ""
    description: str = ""
    keywords: list[str] = []
    parameters_schema: dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def to_mcp_tool_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters_schema,
            },
        }

    def compact_prompt_signature(self) -> str:
        params = ", ".join(self.parameters_schema.keys())
        return f"{self.name}({params}): {self.description}"


class PluginRegistry:
    """Dynamic hot-reloadable plugin registry paired with TinyAgent-style ToolRAG."""

    def __init__(self) -> None:
        self._plugins: dict[str, NeverPlugin] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def register(self, plugin: NeverPlugin) -> None:
        if not plugin.name:
            raise ValueError("Plugin must define a non-empty `name`.")
        self._plugins[plugin.name] = plugin
        doc = f"{plugin.name} {plugin.description} {' '.join(plugin.keywords)}"
        self._vectors[plugin.name] = compute_char_ngram_embedding(doc)

    def get(self, name: str) -> NeverPlugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        return [p.to_mcp_tool_spec() for p in self._plugins.values()]

    def retrieve_relevant_tools(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.08,
    ) -> list[NeverPlugin]:
        """
        ToolRAG (UC Berkeley TinyAgent inspired):
        Scores plugins in <1ms so small growing models receive ONLY relevant tool schemas!
        """
        if not self._plugins:
            return []
        q_words = {w.lower() for w in _WORD_RE.findall(query)}
        q_vec = compute_char_ngram_embedding(query)

        scored: list[tuple[float, NeverPlugin]] = []
        for name, plugin in self._plugins.items():
            p_words = {
                w.lower()
                for w in _WORD_RE.findall(
                    f"{plugin.name} {plugin.description} {' '.join(plugin.keywords)}"
                )
            }
            lex_overlap = len(q_words & p_words) / max(1.0, math.sqrt(len(p_words))) if p_words else 0.0
            sem = max(0.0, float(np.dot(q_vec, self._vectors[name])))
            score = 0.6 * lex_overlap + 0.4 * sem
            if score >= threshold:
                scored.append((score, plugin))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def invoke(self, name: str, **kwargs: Any) -> dict[str, Any]:
        plugin = self._plugins.get(name)
        if plugin is None:
            return {"ok": False, "error": f"Unknown plugin '{name}'"}
        try:
            return plugin.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


import math  # placed after annotations cleanly
