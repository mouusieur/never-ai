"""Sovereign Open-Thought & Multi-Hop Plugin Inference Harness."""

from __future__ import annotations

import json
import re
from typing import Any

from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import (
    BOS_TOKEN,
    CALL_END,
    CALL_END_ID,
    CALL_START,
    CONTEXT_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    NEVER_TOKEN,
    RESULT_END,
    RESULT_START,
    THINK_END,
    THINK_START,
    USER_TOKEN,
)
from never_ai.core.titans_memory import TitansMemoryState
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.memory.evolvemem import EvolveMemStore
from never_ai.plugins.base import PluginRegistry


_CALL_PATTERN = re.compile(
    re.escape(CALL_START) + r"(.*?)" + re.escape(CALL_END),
    re.DOTALL,
)
_THINK_PATTERN = re.compile(
    re.escape(THINK_START) + r"(.*?)" + re.escape(THINK_END),
    re.DOTALL,
)


class SovereignAgentHarness:
    """
    Zero-refusal-filter cognitive runtime connecting ChronoMorphLM with:
    - ToolRAG dynamic plugin selection
    - EvolveMem episodic/semantic recall
    - Titans online test-time neural memory across turns
    - Multi-hop <|call|> -> execute -> <|result|> loop
    """

    def __init__(
        self,
        model: ChronoMorphLM,
        tokenizer: MorphTokenizer,
        registry: PluginRegistry,
        memory: EvolveMemStore,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.memory = memory
        self.titans_session_states: list[TitansMemoryState | None] | None = None

    def reset_session_fast_weights(self) -> None:
        self.titans_session_states = None

    def build_prompt(self, user_message: str, use_tool_rag: bool = True) -> tuple[str, list[str]]:
        active_tools: list[str] = []
        ctx_parts: list[str] = []

        if use_tool_rag:
            tools = self.registry.retrieve_relevant_tools(user_message, top_k=2)
            if tools:
                active_tools = [t.name for t in tools]
                ctx_parts.append(f"Tools: {tools[0].name}")

        mem_hits = self.memory.query(user_message, top_k=2)
        strong_hits = [m for m in mem_hits if m["score"] >= 0.28]
        if strong_hits:
            ctx_parts.append(
                "Memory: " + " ; ".join(f"{m['key']}={m['content'][:160]}" for m in strong_hits)
            )

        ctx_block = f"{CONTEXT_TOKEN}{' // '.join(ctx_parts)}" if ctx_parts else ""
        prompt = f"{BOS_TOKEN}{ctx_block}{USER_TOKEN}{user_message}{NEVER_TOKEN}"
        return prompt, active_tools

    def run_turn(
        self,
        user_message: str,
        max_new_tokens: int = 160,
        temperature: float = 0.5,
        top_k: int = 30,
        max_tool_hops: int = 3,
    ) -> dict[str, Any]:
        prompt_text, routed_tools = self.build_prompt(user_message)
        running_text = prompt_text
        tool_traces: list[dict[str, Any]] = []
        generated_segments: list[str] = []

        for _ in range(max_tool_hops + 1):
            prompt_ids = self.tokenizer.encode(running_text)
            # Keep within context window gracefully
            if len(prompt_ids) > self.model.cfg.max_seq_len - 16:
                prompt_ids = prompt_ids[-(self.model.cfg.max_seq_len - 32) :]

            out_ids, self.titans_session_states = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token_ids={EOS_ID, CALL_END_ID},
                titans_states=self.titans_session_states,
            )
            seg_text = self.tokenizer.decode(out_ids, skip_special_tokens=False)
            generated_segments.append(seg_text)
            running_text += seg_text

            if out_ids and out_ids[-1] == CALL_END_ID:
                matches = _CALL_PATTERN.findall(seg_text)
                if matches:
                    raw_call = matches[-1].strip()
                    plugin_result = self._execute_raw_call(raw_call)
                    tool_traces.append({"call": raw_call, "result": plugin_result})
                    res_json = json.dumps(plugin_result, ensure_ascii=False)
                    res_block = f"{RESULT_START}{res_json}{RESULT_END}"
                    running_text += res_block
                    generated_segments.append(res_block)
                    continue
            break

        full_gen = "".join(generated_segments)
        thoughts = [t.strip() for t in _THINK_PATTERN.findall(full_gen)]

        # Extract clean visible answer while keeping full raw trace accessible
        visible = _THINK_PATTERN.sub("", full_gen)
        visible = _CALL_PATTERN.sub("", visible)
        visible = re.sub(
            re.escape(RESULT_START) + r".*?" + re.escape(RESULT_END),
            "",
            visible,
            flags=re.DOTALL,
        )
        for sp_tok in (EOS_TOKEN, BOS_TOKEN, NEVER_TOKEN, USER_TOKEN):
            visible = visible.replace(sp_tok, "")
        visible = visible.strip()

        surprises = [
            round(s.last_surprise, 6)
            for s in (self.titans_session_states or [])
            if s is not None
        ]

        return {
            "visible_answer": visible,
            "thoughts": thoughts,
            "tool_traces": tool_traces,
            "routed_tools": routed_tools,
            "raw_generation": full_gen,
            "titans_mean_surprise": round(sum(surprises) / max(1, len(surprises)), 6),
        }

    def _execute_raw_call(self, raw_call: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_call)
            if not isinstance(payload, dict):
                return {"ok": False, "error": "Call payload must be a JSON object"}
            plugin_name = str(payload.pop("plugin", ""))
            return self.registry.invoke(plugin_name, **payload)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Invalid plugin JSON: {exc}"}
