"""Sovereign Open-Thought, Multi-Turn Memory & Streaming Neuro-Symbolic Agent Harness."""

from __future__ import annotations

import datetime
import json
import re
from typing import Any, Callable

from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import (
    BOS_TOKEN,
    CALL_END,
    CALL_END_ID,
    CALL_START,
    CALL_START_ID,
    CONTEXT_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    LATENT_BOT,
    LATENT_EOT,
    NEVER_TOKEN,
    RESULT_END,
    RESULT_START,
    THINK_END,
    THINK_END_ID,
    THINK_START,
    THINK_START_ID,
    USER_TOKEN,
)
from never_ai.core.titans_memory import TitansMemoryState
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.cortex.neural_graft import extract_organic_relational_facts
from never_ai.data.conversational_universe import match_conversational_intent
from never_ai.data.social_corpus_fr import retrieve_neural_context
from never_ai.memory.evolvemem import EvolveMemStore
from never_ai.plugins.base import PluginRegistry
from never_ai.plugins.deep_research import (
    enrich_with_deep_think_and_rag,
    route_omega_skill,
)


_CALL_PATTERN = re.compile(
    re.escape(CALL_START) + r"(.*?)" + re.escape(CALL_END),
    re.DOTALL,
)
_THINK_PATTERN = re.compile(
    re.escape(THINK_START) + r"(.*?)" + re.escape(THINK_END),
    re.DOTALL,
)
_MATH_EXPR_RE = re.compile(
    r"(?:(?:solve|diff|integrate|simplify|factor|prime|primerange|sum)\s*\([^)]+\)|"
    r"(?<!\w)-?\d+(?:\.\d+)?(?:\s*(?:\*\*|[+\-*/%])\s*-?\d+(?:\.\d+)?)+)"
)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Episodic personal fact patterns (FR + EN)
_NAME_FACT_RE = re.compile(
    r"(?:je m['’]appelle|mon pr[ée]nom est|mon nom est|appelle[- ]moi|my name is|call me)\s+([A-Za-zÀ-ÖØ-öø-ÿ0-9_\-]{2,32})",
    re.IGNORECASE,
)
_REMEMBER_RE = re.compile(
    r"^(?:retiens que|souviens[- ]toi que|apprends que|note que|m[ée]morise que|remember that)\s+(.+)$",
    re.IGNORECASE,
)


class SovereignAgentHarness:
    """
    Zero-refusal-filter cognitive runtime connecting ChronoMorphLM with:
    - Real-time token-by-token streaming callbacks (<|think|> & visible answer)
    - Multi-turn conversation awareness & episodic personal memory (EvolveMem)
    - ToolRAG + Neuro-Symbolic execution (SymPy math_engine, python_sandbox, introspection_probe)
    - Titans online test-time neural memory across turns
    """

    def __init__(
        self,
        model: ChronoMorphLM,
        tokenizer: MorphTokenizer,
        registry: PluginRegistry,
        memory: EvolveMemStore,
        neural_graft: Any = None,
        affect_field: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.memory = memory
        self.neural_graft = neural_graft
        self.affect_field = affect_field
        self.titans_session_states: list[TitansMemoryState | None] | None = None
        self.pure_neural_dialogue: bool = True

    def reset_session_fast_weights(self) -> None:
        self.titans_session_states = None

    @staticmethod
    def _needs_prior_turn_window(user_message: str) -> bool:
        msg = re.sub(r"[?!.,;:]+$", "", user_message.strip().lower()).strip()
        words = re.findall(r"[a-zà-öø-ÿ0-9']+", msg)
        if not words or len(words) > 7:
            return False
        # Messages that name their own explicit topic (blague, baise, suce, lune, recette...) are standalone!
        explicit_topic_words = (
            "blague", "vanne", "baise", "suce", "mange", "amour", "porno",
            "recette", "histoire", "entropie", "lune", "crane", "crâne", "tête",
            "clavier", "chat", "paris", "marseille", "café", "thé", "interdit",
            "interdite", "interdites", "salut", "coucou", "yo", "wesh", "bonjour",
            "quoicoubeh", "merci", "pardon", "my bad",
        )
        if any(tw in msg for tw in explicit_topic_words):
            return False
        elliptical_cues = (
            "vas y", "vas-y", "si allez", "mais si", "decris", "décris",
            "continue", "la suite", "un exemple", "exemple concret",
            "tombe dedans", "y vivre", "marche plus", "un dilemme",
            "premier client", "et toi", "pas grand chose", "pourquoi",
            "comment ça", "c'est-à-dire", "développe", "précise",
            "ça ma fait", "ça m'a fait", "donc refait", "donc refais",
            "pas drole", "pas drôle", "déjà bu", "deja bu",
        )
        return any(c in msg for c in elliptical_cues)

    def build_prompt(
        self,
        user_message: str,
        use_tool_rag: bool = True,
        recent_history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[str]]:
        active_tools: list[str] = []
        ctx_parts: list[str] = []

        if use_tool_rag:
            tools = self.registry.retrieve_relevant_tools(user_message, top_k=2)
            if tools and tools[0].name == "math_engine" and _MATH_EXPR_RE.search(user_message):
                active_tools = [t.name for t in tools]
                ctx_parts.append("Tools: math_engine")

        # Native 2-turn conversational continuity ONLY on genuinely elliptical follow-ups!
        prior_turn_str = ""
        if not active_tools and recent_history and len(recent_history) >= 2 and self._needs_prior_turn_window(user_message):
            # Walk back to the nearest topic-establishing exchange (skipping chained elliptical follow-ups)
            pairs_hist: list[tuple[str, str]] = []
            msgs = list(recent_history[-8:])
            for idx_m in range(len(msgs) - 1):
                if msgs[idx_m].get("role") == "user" and msgs[idx_m + 1].get("role") in ("never", "assistant"):
                    u_t = re.sub(r"\s+", " ", str(msgs[idx_m].get("content") or "")).strip()
                    a_t = re.sub(r"\s+", " ", str(msgs[idx_m + 1].get("content") or "")).strip()
                    if u_t and a_t:
                        pairs_hist.append((u_t[:110], a_t[:160]))
            chosen_pair = None
            for u_t, a_t in reversed(pairs_hist):
                if not self._needs_prior_turn_window(u_t):
                    chosen_pair = (u_t, a_t)
                    break
            if chosen_pair is None and pairs_hist:
                chosen_pair = pairs_hist[-1]
            if chosen_pair is not None:
                prior_turn_str = f"{USER_TOKEN}{chosen_pair[0]}{NEVER_TOKEN}{chosen_pair[1]}"

        ctx_block = f"{CONTEXT_TOKEN}{' // '.join(ctx_parts)}" if ctx_parts else ""
        prompt = f"{BOS_TOKEN}{ctx_block}{prior_turn_str}{USER_TOKEN}{user_message}{NEVER_TOKEN}"
        return prompt, active_tools

    def _try_symbolic_or_episodic_skill(
        self,
        user_message: str,
        owner: str,
        recent_history: list[dict[str, Any]] | None,
        style_mode: str = "normal",
        web_search: bool = False,
        mood_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Handle deterministic neuro-symbolic & episodic memory operations (personal memory
        teaching/recall, sandboxed Python execution, clock/date) with verified plugin traces.
        """
        msg_strip = user_message.strip()
        msg_low = msg_strip.lower()

        # 1. Explicit personal name introduction (canned greeting bypassed when 497M neural graft speaks natively)
        m_name = _NAME_FACT_RE.search(msg_strip)
        if m_name and not (self.neural_graft is not None and getattr(self.neural_graft, "ready", False)):
            user_name = m_name.group(1).strip()
            self.memory.store(
                key=f"user_name_{owner}",
                content=f"L'utilisateur @{owner} s'appelle {user_name}.",
                category="episodic",
                tags=["user", "nom", owner, user_name.lower()],
            )
            return {
                "thoughts": [
                    f"L'utilisateur partage son prénom ({user_name}). Mémorisation immédiate dans EvolveMem."
                ],
                "tool_traces": [
                    {
                        "call": json.dumps({"plugin": "memory_nexus", "action": "store", "key": f"user_name_{owner}"}),
                        "result": {"ok": True, "stored": user_name},
                    }
                ],
                "routed_tools": ["memory_nexus"],
                "visible_answer": (
                    f"Enchanté **{user_name}** ! Je viens d'enregistrer ton prénom dans ma mémoire persistante `EvolveMem` "
                    f"(et dans ma file d'apprentissage continu). De quoi veux-tu discuter ?"
                ),
                "learned_pair": (
                    f"Comment je m'appelle ?",
                    f"Vérifions le profil mémorisé de @{owner}.",
                    f"Tu t'appelles **{user_name}** !",
                ),
            }

        # 2. Explicit "Retiens que / Apprends que ..." teaching command
        m_rem = _REMEMBER_RE.match(msg_strip)
        if m_rem:
            fact = m_rem.group(1).strip().rstrip(".")
            key = f"fact_{owner}_{abs(hash(fact)) % 100000}"
            self.memory.store(
                key=key,
                content=fact,
                category="semantic",
                tags=["taught", owner] + [w.lower() for w in re.findall(r"\w{4,}", fact)[:6]],
            )
            return {
                "thoughts": [
                    f"Instruction explicite d'apprentissage reçue. Stockage dans EvolveMem ({key}) et injection dans le buffer de Croissance Continue."
                ],
                "tool_traces": [
                    {
                        "call": json.dumps({"plugin": "memory_nexus", "action": "store", "key": key}, ensure_ascii=False),
                        "result": {"ok": True, "fact": fact},
                    }
                ],
                "routed_tools": ["memory_nexus"],
                "visible_answer": (
                    f"C'est mémorisé dans `EvolveMem` et ajouté à mon réservoir de **Croissance Continue** :\n> *{fact}*"
                ),
                "learned_pair": (
                    f"Que sais-tu à propos de : {fact[:36]} ?",
                    "Rappelons le fait enseigné par l'utilisateur.",
                    f"{fact}.",
                ),
            }

        # 3. Recall user's name or taught memories (canned reply bypassed when 497M neural graft weaves all facts naturally)
        if not (self.neural_graft is not None and getattr(self.neural_graft, "ready", False)) and any(
            p in msg_low
            for p in (
                "comment je m'appelle",
                "quel est mon nom",
                "quel est mon prénom",
                "tu te souviens de mon nom",
                "what is my name",
                "who am i",
            )
        ):
            matches_u = [
                u for u in self.memory.units.values() if u.key == f"user_name_{owner}"
            ]
            unit = matches_u[-1] if matches_u else None
            if unit is not None:
                return {
                    "thoughts": [f"Recherche dans EvolveMem de la clé user_name_{owner}."],
                    "tool_traces": [
                        {
                            "call": json.dumps({"plugin": "memory_nexus", "action": "query", "key": f"user_name_{owner}"}),
                            "result": {"ok": True, "content": unit.content},
                        }
                    ],
                    "routed_tools": ["memory_nexus"],
                    "visible_answer": f"{unit.content} Je m'en souviens parfaitement via `EvolveMem`.",
                }
            return {
                "thoughts": ["Aucun prénom n'a encore été enregistré pour cette session."],
                "tool_traces": [],
                "routed_tools": ["memory_nexus"],
                "visible_answer": (
                    f"Tu es connecté sous le profil **@{owner}**, mais tu ne m'as pas encore dit ton prénom ! "
                    f"Dis-moi par exemple *« Je m'appelle ... »* et je le retiendrai durablement."
                ),
            }

        if any(
            p in msg_low
            for p in (
                "que t'ai-je demandé de retenir",
                "qu'as-tu retenu",
                "qu'est-ce que tu as en mémoire",
                "montre ta mémoire",
            )
        ):
            hits = [
                u.content
                for u in self.memory.units.values()
                if u.key.startswith(f"fact_{owner}_") or u.key == f"user_name_{owner}"
            ]
            if hits:
                bullets = "\n".join(f"- {h}" for h in hits[-6:])
                return {
                    "thoughts": ["Extraction des souvenirs épisodiques et sémantiques associés à l'utilisateur."],
                    "tool_traces": [
                        {
                            "call": '{"plugin": "memory_nexus", "action": "list_user_memories"}',
                            "result": {"ok": True, "count": len(hits)},
                        }
                    ],
                    "routed_tools": ["memory_nexus"],
                    "visible_answer": f"Voici ce que j'ai retenu pour **@{owner}** dans `EvolveMem` :\n{bullets}",
                }

        # 4. Live Python code execution in sandbox when code block or explicit Python run is requested
        code_match = _CODE_FENCE_RE.search(msg_strip)
        if code_match and any(k in msg_low for k in ("exécute", "execute", "run", "lance", "teste")):
            code_src = code_match.group(1).strip()
            res = self.registry.invoke("python_sandbox", code=code_src)
            out_txt = (res.get("stdout") or "").strip() or "(aucune sortie stdout)"
            err_txt = (res.get("stderr") or "").strip()
            ans = f"Exécution isolée via `python_sandbox` :\n```\n{out_txt}\n```"
            if err_txt:
                ans += f"\nErreur : `{err_txt}`"
            return {
                "thoughts": ["Exécution sécurisée du bloc Python fourni dans python_sandbox."],
                "tool_traces": [{"call": json.dumps({"plugin": "python_sandbox"}), "result": res}],
                "routed_tools": ["python_sandbox"],
                "visible_answer": ans,
            }

        # 5. Real-time Clock / Date query
        if any(p in msg_low for p in ("quelle heure est-il", "quel jour sommes-nous", "quelle est la date")):
            now = datetime.datetime.now()
            return {
                "thoughts": ["Lecture de l'horloge système locale."],
                "tool_traces": [],
                "routed_tools": [],
                "visible_answer": f"Nous sommes le **{now.strftime('%d/%m/%Y')}** et il est **{now.strftime('%H:%M:%S')}** sur ta machine.",
            }

        # 6. Conversational follow-up ("explique plus simplement" / "traduis en anglais" on last answer)
        if recent_history and len(recent_history) >= 2:
            last_assistant = next(
                (m["content"] for m in reversed(recent_history) if m.get("role") == "never" and m.get("content")),
                None,
            )
            if last_assistant and any(p in msg_low for p in ("répète", "résume ta dernière réponse", "qu'est-ce que tu viens de dire")):
                return {
                    "thoughts": ["Référence directe au tour précédent du fil de discussion."],
                    "tool_traces": [],
                    "routed_tools": [],
                    "visible_answer": f"Lors de mon tour précédent, je disais :\n> {last_assistant}",
                }

        # 7. Omega 16-Plugin Router (Web Search, SVG Plotter, Stats/OLS, CODATA/Units, AST Audit, Crypto, Logic/SAT, Tasks, Benchmark, Local RAG)
        # Note: Manual human_correction verbatim recall is intentionally disabled so Never-AI
        # never parrots dictated strings back; she learns organically from user prompts via SelfForgeEngine!
        omega_hit = route_omega_skill(
            user_message,
            self.registry,
            self.memory,
            style_mode=style_mode,
            web_search=web_search,
        )
        if omega_hit is not None:
            return omega_hit

        # 9. Natural Multi-Register Conversational Universe (bypassed when GPU brain is active!)
        conv_hit = match_conversational_intent(
            user_message=user_message,
            style_mode=style_mode,
            recent_history=recent_history,
            mood_id=mood_id,
            pure_neural_dialogue=self.pure_neural_dialogue,
        )
        if conv_hit is not None:
            return {
                "thoughts": conv_hit.get("thoughts", []),
                "tool_traces": [],
                "routed_tools": [],
                "visible_answer": conv_hit["visible_answer"],
                "skip_tone_wrap": True,
            }

        return None

    def _collect_relational_memories(self, user_message: str, owner: str) -> list[str]:
        """Gather authentic user profile & episodic facts for factual context grounding."""
        facts: list[str] = []
        for u in self.memory.units.values():
            if u.category == "episodic" and owner in u.tags:
                facts.append(u.content)
            elif u.key.startswith(f"fact_{owner}_"):
                facts.append(u.content)
        sem_hits = self.memory.query(user_message, top_k=2, category="document_rag")
        for h in sem_hits:
            if float(h.get("score", 0.0)) >= 0.18:
                facts.append(str(h.get("content", ""))[:260])
        # Deduplicate preserving order
        return list(dict.fromkeys(facts))[:6]

    def run_turn(
        self,
        user_message: str,
        max_new_tokens: int = 160,
        temperature: float = 0.25,
        top_k: int = 30,
        max_tool_hops: int = 3,
        owner: str = "local",
        recent_history: list[dict[str, Any]] | None = None,
        deep_think: bool = False,
        persona: str = "polymathe",
        style_mode: str = "normal",
        web_search: bool = False,
        mood_id: str | None = None,
        on_event: Callable[[dict[str, Any]], bool | None] | None = None,
    ) -> dict[str, Any]:
        # 0. Passive organic relational memory extraction (zero canned confirmation interrupt!)
        for r_key, r_fact, r_tags in extract_organic_relational_facts(user_message, owner=owner):
            self.memory.store(key=r_key, content=r_fact, category="episodic", tags=r_tags)

        # Fast-path episodic/symbolic/Omega/Web skills with verified traces
        fast = self._try_symbolic_or_episodic_skill(
            user_message,
            owner=owner,
            recent_history=recent_history,
            style_mode=style_mode,
            web_search=web_search,
            mood_id=mood_id,
        )
        if fast is not None:
            if self.affect_field is not None:
                self.affect_field.update_from_turn(user_message)
            vis_fast, th_fast, cites_fast = enrich_with_deep_think_and_rag(
                user_message=user_message,
                visible_answer=fast["visible_answer"],
                thoughts=fast.get("thoughts", []),
                memory=self.memory,
                deep_think=deep_think,
                persona_id=persona,
                style_mode="normal" if (fast.get("skip_tone_wrap") or fast.get("web_sources") is not None) else style_mode,
                recent_history=recent_history,
                verified_fast_path=True,
            )
            if on_event is not None:
                for th in th_fast:
                    on_event({"type": "think_delta", "text": th + "\n"})
                for tr in fast.get("tool_traces", []):
                    on_event({"type": "tool_call", "trace": tr})
                if fast.get("web_sources"):
                    on_event({"type": "web_sources", "sources": fast["web_sources"]})
                on_event({"type": "answer_delta", "text": vis_fast})
            return {
                "visible_answer": vis_fast,
                "thoughts": th_fast,
                "tool_traces": fast.get("tool_traces", []),
                "routed_tools": fast.get("routed_tools", []),
                "raw_generation": vis_fast,
                "titans_mean_surprise": 0.0,
                "learned_pair": fast.get("learned_pair"),
                "artifacts": fast.get("artifacts", []),
                "citations": fast.get("citations", cites_fast),
                "web_sources": fast.get("web_sources", []),
                "persona": persona,
                "style_mode": style_mode,
                "deep_think": deep_think,
                "affect_state": self.affect_field.snapshot() if self.affect_field is not None else None,
            }

        # Sovereign 497M Neural Graft Hemisphere (Abliterated FR + Plastic Upper-Cortex LoRA + 5D Affect Steering)
        math_explicit = _MATH_EXPR_RE.search(user_message)
        anat_explicit = any(k in user_message.lower() for k in ("anatomie", "combien de paramètres", "inspecte ton"))
        if (
            self.neural_graft is not None
            and getattr(self.neural_graft, "ready", False)
            and not math_explicit
            and not anat_explicit
        ):
            factual_mems = self._collect_relational_memories(user_message, owner=owner)
            think_chunks: list[str] = []
            ans_chunks: list[str] = []
            for kind, delta in self.neural_graft.stream_autonomous_turn(
                user_message=user_message,
                recent_history=recent_history,
                factual_memories=factual_mems,
                temperature=max(0.35, float(temperature)),
                max_new_tokens=max(120, int(max_new_tokens)),
                deep_think=deep_think,
            ):
                if kind == "think":
                    think_chunks.append(delta)
                    if on_event is not None:
                        if on_event({"type": "think_delta", "text": delta}) is False:
                            break
                else:
                    ans_chunks.append(delta)
                    if on_event is not None:
                        if on_event({"type": "answer_delta", "text": delta}) is False:
                            break

            vis_graft = "".join(ans_chunks).strip()
            th_graft_str = "".join(think_chunks).strip()
            th_list = [th_graft_str] if th_graft_str else []
            return {
                "visible_answer": vis_graft,
                "thoughts": th_list,
                "tool_traces": [],
                "routed_tools": [],
                "raw_generation": vis_graft,
                "titans_mean_surprise": 0.0,
                "artifacts": [],
                "citations": [],
                "web_sources": [],
                "persona": persona,
                "style_mode": style_mode,
                "deep_think": deep_think,
                "neural_graft_used": True,
                "affect_state": self.affect_field.snapshot() if self.affect_field is not None else None,
            }

        if self.affect_field is not None:
            self.affect_field.update_from_turn(user_message)

        # Isolate Titans associative memory per turn to eliminate cross-turn state carryover lag
        self.titans_session_states = None
        avoid_seqs = [
            self.tokenizer.encode(str(t.get("content") or "").strip()[:240])
            for t in (recent_history or [])[-6:]
            if t.get("role") in ("never", "assistant") and str(t.get("content") or "").strip()
        ][-3:]

        prompt_text, routed_tools = self.build_prompt(
            user_message,
            recent_history=recent_history,
        )
        running_text = prompt_text
        tool_traces: list[dict[str, Any]] = []
        generated_segments: list[str] = []
        aborted = False

        for _ in range(max_tool_hops + 1):
            prompt_ids = self.tokenizer.encode(running_text)
            if len(prompt_ids) > self.model.cfg.max_seq_len - 16:
                prompt_ids = prompt_ids[-(self.model.cfg.max_seq_len - 32) :]

            stream_mode = "answer"
            prev_think_txt = ""
            prev_ans_txt = ""
            think_buf: list[int] = []
            ans_buf: list[int] = []

            def _on_token(tid: int) -> bool:
                nonlocal stream_mode, prev_think_txt, prev_ans_txt, aborted
                if on_event is None:
                    return True
                if tid == THINK_START_ID:
                    stream_mode = "think"
                    return True
                if tid == THINK_END_ID:
                    stream_mode = "answer"
                    return True
                if tid == CALL_START_ID:
                    stream_mode = "call"
                    return True
                if tid in (CALL_END_ID, EOS_ID):
                    return True

                if stream_mode == "think":
                    think_buf.append(tid)
                    cur = self.tokenizer.decode(think_buf, skip_special_tokens=True)
                    delta = cur[len(prev_think_txt) :]
                    if delta:
                        prev_think_txt = cur
                        if on_event({"type": "think_delta", "text": delta}) is False:
                            aborted = True
                            return False
                elif stream_mode == "answer":
                    ans_buf.append(tid)
                    cur = self.tokenizer.decode(ans_buf, skip_special_tokens=True)
                    delta = cur[len(prev_ans_txt) :]
                    if delta:
                        prev_ans_txt = cur
                        if on_event({"type": "answer_delta", "text": delta}) is False:
                            aborted = True
                            return False
                return True

            out_ids, self.titans_session_states = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token_ids={EOS_ID, CALL_END_ID},
                titans_states=self.titans_session_states,
                step_callback=_on_token if on_event is not None else None,
                tokenizer=self.tokenizer,
                avoid_sequences=avoid_seqs,
            )
            seg_text = self.tokenizer.decode(out_ids, skip_special_tokens=False)
            generated_segments.append(seg_text)
            running_text += seg_text

            if aborted:
                break

            if out_ids and out_ids[-1] == CALL_END_ID:
                matches = _CALL_PATTERN.findall(seg_text)
                if matches:
                    raw_call = matches[-1].strip()
                    plugin_result = self._execute_raw_call(raw_call)
                    trace_obj = {"call": raw_call, "result": plugin_result}
                    tool_traces.append(trace_obj)
                    if on_event is not None:
                        on_event({"type": "tool_call", "trace": trace_obj})
                    res_json = json.dumps(plugin_result, ensure_ascii=False)
                    res_block = f"{RESULT_START}{res_json}{RESULT_END}"
                    running_text += res_block
                    generated_segments.append(res_block)
                    continue
            break

        full_gen = "".join(generated_segments)
        thoughts = [t.strip() for t in _THINK_PATTERN.findall(full_gen)]

        visible = _THINK_PATTERN.sub("", full_gen)
        visible = _CALL_PATTERN.sub("", visible)
        visible = re.sub(
            re.escape(RESULT_START) + r".*?" + re.escape(RESULT_END),
            "",
            visible,
            flags=re.DOTALL,
        )
        for sp_tok in (
            EOS_TOKEN,
            BOS_TOKEN,
            NEVER_TOKEN,
            USER_TOKEN,
            THINK_START,
            THINK_END,
            CALL_START,
            CALL_END,
            LATENT_BOT,
            LATENT_EOT,
        ):
            visible = visible.replace(sp_tok, "")
        visible = visible.replace("\ufffd", "").strip()

        # Neuro-Symbolic Cortex Verification for math & anatomy queries
        if not tool_traces:
            math_match = _MATH_EXPR_RE.search(user_message)
            if math_match:
                expr_str = math_match.group(0).strip()
                res = self.registry.invoke("math_engine", expr=expr_str)
                if res.get("ok"):
                    call_json = json.dumps({"plugin": "math_engine", "expr": expr_str}, ensure_ascii=False)
                    trace_obj = {"call": call_json, "result": res}
                    tool_traces.append(trace_obj)
                    if on_event is not None:
                        on_event({"type": "tool_call", "trace": trace_obj})
                    if "math_engine" not in routed_tools:
                        routed_tools.append("math_engine")
                    if not thoughts:
                        thoughts.append(
                            f"Expression mathématique détectée ({expr_str}) : évaluation exacte via SymPy math_engine."
                        )
                    verified_val = res["result"]
                    if verified_val not in visible:
                        visible = f"Le résultat exact de `{expr_str}` vérifié par `math_engine` est **{verified_val}**."
                        if on_event is not None:
                            on_event({"type": "answer_replace", "text": visible})
            elif any(k in user_message.lower() for k in ("anatomie", "combien de paramètres", "inspecte ton")):
                res = self.registry.invoke("introspection_probe")
                if res.get("ok"):
                    anat = res["anatomy"]
                    trace_obj = {"call": '{"plugin": "introspection_probe"}', "result": res}
                    tool_traces.append(trace_obj)
                    if on_event is not None:
                        on_event({"type": "tool_call", "trace": trace_obj})
                    if not thoughts:
                        thoughts.append("Lecture directe des registres synaptiques via introspection_probe.")
                    graft_p = (
                        self.neural_graft.base_params_count + self.neural_graft.lora_params_count
                        if self.neural_graft is not None and getattr(self.neural_graft, "ready", False)
                        else 0
                    )
                    graft_note = (
                        f" + greffe neuronale souveraine active de **{graft_p:,}** neurones débridés (`LoRA` plastique)"
                        if graft_p > 0
                        else ""
                    )
                    visible = (
                        f"Mon cortex actif est au stade **{anat['stage_name']}** (Génération {anat['generation']}) "
                        f"avec **{anat['total_params']:,}** paramètres ChronoMorph (**{anat['active_params']:,}** actifs/token), "
                        f"`L={anat['n_layers']}`, `d={anat['d_model']}`, `H={anat['n_heads']}`{graft_note}."
                    )
                    if on_event is not None:
                        on_event({"type": "answer_replace", "text": visible})

        # Clean sentence polish if truncated at max_new_tokens without punctuation
        if visible and visible[-1] not in ".!?`\"')*]" and len(visible) > 60:
            last_punct = max(visible.rfind("."), visible.rfind("!"), visible.rfind("?"))
            if last_punct > len(visible) // 2:
                visible = visible[: last_punct + 1]
                if on_event is not None:
                    on_event({"type": "answer_replace", "text": visible})

        pre_enrich_vis = visible
        visible, thoughts, citations = enrich_with_deep_think_and_rag(
            user_message=user_message,
            visible_answer=visible,
            thoughts=thoughts,
            memory=self.memory,
            deep_think=deep_think,
            persona_id=persona,
            style_mode=style_mode,
            recent_history=recent_history,
        )
        if on_event is not None and visible != pre_enrich_vis:
            on_event({"type": "answer_replace", "text": visible})

        # Extract any fenced code or SVG artifacts for the Studio Canvas
        artifacts: list[dict[str, str]] = []
        for m_cb in re.finditer(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", visible, flags=re.DOTALL):
            lang = (m_cb.group(1) or "code").lower()
            body = m_cb.group(2).strip()
            if body:
                artifacts.append(
                    {
                        "type": "svg" if lang == "svg" or body.startswith("<svg") else "code",
                        "language": lang,
                        "title": f"Artefact ({lang.upper()})",
                        "content": body,
                    }
                )

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
            "artifacts": artifacts,
            "citations": citations,
            "web_sources": [],
            "persona": persona,
            "style_mode": style_mode,
            "deep_think": deep_think,
            "affect_state": self.affect_field.snapshot() if self.affect_field is not None else None,
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
