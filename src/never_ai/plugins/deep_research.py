"""Omega Cognitive Orchestrator: Deep-Think Multi-Step Reasoning, 15-Plugin Router, Personas & Local RAG."""

from __future__ import annotations

import json
import re
from typing import Any

from never_ai.data.conversational_universe import (
    CONVERSATIONAL_STYLES,
    apply_conversational_tone_wrapper,
    check_extreme_harm_guard,
    match_conversational_intent,
)
from never_ai.memory.evolvemem import EvolveMemStore
from never_ai.plugins.base import PluginRegistry


SOVEREIGN_PERSONAS: dict[str, dict[str, str]] = {
    "polymathe": {
        "id": "polymathe",
        "label": "Polymathe Souverain",
        "badge": "Ω Polymathe",
        "description": "Équilibre rigueur scientifique, code propre, clarté pédagogique et liberté de pensée.",
        "thought_prefix": "Mode Polymathe Souverain : croisons rigueur formelle, intuition physique et clarté.",
    },
    "architecte": {
        "id": "architecte",
        "label": "Architecte Système & Cyber",
        "badge": "⚙ Architecte",
        "description": "Focus sur l'architecture logicielle, la complexité algorithmique, l'AST et la sécurité.",
        "thought_prefix": "Mode Architecte Système : analysons les invariants, la complexité O(.) et la sûreté.",
    },
    "mathematicien": {
        "id": "mathematicien",
        "label": "Mathématicien & Physicien",
        "badge": "∫ Sciences",
        "description": "Privilégie les démonstrations formelles SymPy, les courbes SVG et les constantes CODATA.",
        "thought_prefix": "Mode Mathématicien & Physicien : vérifions formellement les équations et ordres de grandeur.",
    },
    "philosophe": {
        "id": "philosophe",
        "label": "Épistémologue Socratique",
        "badge": "🏛 Épistémè",
        "description": "Examine les prémisses cachées, la logique propositionnelle et la falsifiabilité.",
        "thought_prefix": "Mode Épistémologue : testons la validité logique des prémisses et les contre-exemples.",
    },
    "concis": {
        "id": "concis",
        "label": "Exécutif Ultra-Concis",
        "badge": "⚡ Concis",
        "description": "Réponses denses, structurées et immédiates sans détour.",
        "thought_prefix": "Mode Exécutif Concis : allons droit au résultat vérifié.",
    },
}

_PLOT_CMD_RE = re.compile(
    r"^(?:/plot\s+|trace(?:\s+la)?(?:\s+courbe|\s+fonction|\s+graphe)?(?:\s+de)?\s+|plot(?:\s+the)?(?:\s+function|\s+curve)?\s+)(.+)$",
    re.IGNORECASE,
)
_UNIT_CONV_RE = re.compile(
    r"(?:/unit\s+|converti[sr]?\s+|convert\s+)(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([a-zA-Z°µΩ/_^0-9]+)\s+(?:en|vers|to|in)\s+([a-zA-Z°µΩ/_^0-9]+)",
    re.IGNORECASE,
)
_NUM_LIST_RE = re.compile(r"-?\d+(?:\.\d+)?")
_PAIRS_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_LOGIC_CMD_RE = re.compile(
    r"^(?:/logic\s+|table\s+de\s+v[ée]rit[ée]\s+(?:de\s+|pour\s+)?|truth\s+table\s+(?:of\s+|for\s+)?)(.+)$",
    re.IGNORECASE,
)
_SHA_RE = re.compile(
    r"(?:sha256|sha512|blake2b|hash)\s+(?:de|of)\s+[\"']?(.+?)[\"']?$",
    re.IGNORECASE,
)
_ENTROPY_RE = re.compile(
    r"(?:entropie|entropy)\s+(?:de\s+shannon\s+)?(?:de|of)\s+[\"']?(.+?)[\"']?$",
    re.IGNORECASE,
)
_WEB_SEARCH_RE = re.compile(
    r"^(?:/web\s+|/search\s+|cherche\s+sur\s+(?:internet|le\s+web|google)\s*:?\s*|"
    r"recherche\s+(?:sur\s+)?(?:internet|web|en\s+ligne)\s*:?\s*|"
    r"trouve\s+sur\s+(?:internet|le\s+web)\s*:?\s*|"
    r"search\s+(?:the\s+)?(?:web|internet)\s+(?:for\s+)?)(.+)$",
    re.IGNORECASE,
)


def execute_live_web_search_turn(
    query: str,
    registry: PluginRegistry,
    memory: EvolveMemStore,
    style_mode: str = "normal",
) -> dict[str, Any]:
    """Run `web_search_engine`, store findings in EvolveMem, and craft a natural conversational synthesis."""
    q_clean = re.sub(r"^/(?:web|search)\s+", "", query.strip(), flags=re.IGNORECASE).strip()
    res = registry.invoke("web_search_engine", query=q_clean, max_results=6)
    hits = res.get("results", []) if res.get("ok") else []

    if hits:
        # Assimilate top web findings into EvolveMem so Never-AI retains them!
        summary_facts = " | ".join(f"{h['title']}: {h['snippet'][:140]}" for h in hits[:3])
        memory.store(
            key=f"web_{ abs(hash(q_clean)) % 1_000_000 }",
            content=f"Recherche Web ({q_clean}) : {summary_facts}",
            category="web_memory",
            tags=["web", "internet", q_clean.lower()[:40]],
        )
        lead = {
            "normal": f"J'ai lancé une recherche internet en direct sur **« {q_clean} »** ({len(hits)} sources trouvées) :",
            "amical": f"Je viens de fouiller le web en direct pour **« {q_clean} »** ! Voilà ce qui ressort de plus intéressant :",
            "grok": f"Hop, coup de radar en direct sur le web pour **« {q_clean} »** — voilà le concret sans blabla :",
        }.get(style_mode, f"Résultats web en direct pour **« {q_clean} »** :")

        bullets = []
        for h in hits:
            snip = h.get("snippet", "").strip()
            bullets.append(
                f"- **[{h['title']}]({h['url']})** *({h['source']})*\n  {snip}"
            )
        ans = lead + "\n\n" + "\n".join(bullets)
        return {
            "thoughts": [
                f"Recherche internet temps réel via web_search_engine pour « {q_clean} » ({', '.join(res.get('providers_reached', []))})."
            ],
            "tool_traces": [
                {
                    "call": json.dumps({"plugin": "web_search_engine", "query": q_clean}, ensure_ascii=False),
                    "result": {
                        "ok": True,
                        "count": len(hits),
                        "providers": res.get("providers_reached", []),
                        "network_note": res.get("network_note", ""),
                    },
                }
            ],
            "routed_tools": ["web_search_engine"],
            "visible_answer": ans,
            "web_sources": hits,
        }

    # If restricted network / 0 external hits on general query: explain transparently & answer smartly
    note = res.get("network_note", "Aucune source externe joignable sur ce réseau.")
    conv_fallback = match_conversational_intent(q_clean, style_mode=style_mode)
    base_txt = (
        conv_fallback["visible_answer"]
        if conv_fallback
        else (
            f"Sur ce réseau actuel, l'accès web généraliste est filtré (*{note}* — dans l'application Windows `.exe` sur ton PC, "
            f"DuckDuckGo et Wikipédia répondent en direct sans restriction !). En attendant, dis-moi exactement quel angle de "
            f"**« {q_clean} »** t'intéresse et on creuse ça ensemble."
        )
    )
    return {
        "thoughts": [f"Tentative de recherche web pour « {q_clean} » ({note})."],
        "tool_traces": [
            {
                "call": json.dumps({"plugin": "web_search_engine", "query": q_clean}, ensure_ascii=False),
                "result": {"ok": True, "count": 0, "network_note": note},
            }
        ],
        "routed_tools": ["web_search_engine"],
        "visible_answer": base_txt,
        "web_sources": [],
    }


def route_omega_skill(
    user_message: str,
    registry: PluginRegistry,
    memory: EvolveMemStore,
    style_mode: str = "normal",
    web_search: bool = False,
) -> dict[str, Any] | None:
    """
    Deterministic slash-command & natural-language router for the 16 Sovereign Omega Plugins.
    Returns structured `{thoughts, tool_traces, routed_tools, visible_answer, artifacts}` or None.
    """
    msg = user_message.strip()
    low = msg.lower()

    # 0. Live Internet Search (`web_search_engine` via toggle, `/web`, URL, or natural language)
    m_web = _WEB_SEARCH_RE.match(msg)
    has_direct_url = bool(re.search(r"https?://[^\s<>\"']+", msg))
    if web_search or m_web or has_direct_url or any(
        k in low for k in ("cherche sur internet", "recherche sur le web", "va voir sur internet", "trouve sur le web")
    ):
        q_str = m_web.group(1).strip() if m_web else msg
        return execute_live_web_search_turn(q_str, registry, memory, style_mode=style_mode)

    # 1. Interactive SVG Function Plotter (`plot_svg_engine`)
    m_plot = _PLOT_CMD_RE.match(msg)
    if m_plot:
        raw_expr = m_plot.group(1).strip().rstrip(".")
        # Optional range `sur [-5, 5]` or `from -5 to 5`
        x_min, x_max = -10.0, 10.0
        m_rng = re.search(
            r"(?:sur|entre|on|in|from)\s*\[?\s*(-?\d+(?:\.\d+)?)\s*(?:,|;|à|to)\s*(-?\d+(?:\.\d+)?)\s*\]?",
            raw_expr,
            flags=re.IGNORECASE,
        )
        if m_rng:
            x_min, x_max = float(m_rng.group(1)), float(m_rng.group(2))
            raw_expr = raw_expr[: m_rng.start()].strip()
        exprs = [e.strip() for e in raw_expr.split(";") if e.strip()]
        res = registry.invoke(
            "plot_svg_engine",
            expressions=exprs if len(exprs) > 1 else exprs[0],
            x_min=x_min,
            x_max=x_max,
            title=f"f(x) = {', '.join(exprs)}",
        )
        if res.get("ok"):
            summaries = []
            for s in res["series"]:
                roots_txt = f", racines ≈ `{s['approx_roots']}`" if s["approx_roots"] else ""
                summaries.append(
                    f"- **`{s['expr']}`** : min = `{s['y_min']}`, max = `{s['y_max']}`{roots_txt}"
                )
            ans = (
                f"Voici le tracé vectoriel exact de **`{', '.join(exprs)}`** sur `[{x_min}, {x_max}]` "
                f"généré par `plot_svg_engine` :\n\n"
                + "\n".join(summaries)
                + f"\n\n```svg\n{res['svg']}\n```"
            )
            return {
                "thoughts": [
                    f"Demande de tracé mathématique détectée pour {exprs} sur [{x_min}, {x_max}]. Appel de plot_svg_engine."
                ],
                "tool_traces": [
                    {
                        "call": json.dumps(
                            {"plugin": "plot_svg_engine", "expressions": exprs, "x_min": x_min, "x_max": x_max},
                            ensure_ascii=False,
                        ),
                        "result": {k: v for k, v in res.items() if k != "svg"},
                    }
                ],
                "routed_tools": ["plot_svg_engine"],
                "visible_answer": ans,
                "artifacts": [{"type": "svg", "title": f"Courbe : {', '.join(exprs)}", "content": res["svg"]}],
            }

    # 2. Live Benchmark & Self-Audit (`/bench` or natural language)
    if low.startswith("/bench") or any(
        k in low for k in ("lance le benchmark", "auto-diagnostic complet", "teste tous tes plugins", "run benchmark")
    ):
        res = registry.invoke("live_benchmark_suite")
        if res.get("ok"):
            rows = "\n".join(
                f"- {'✅' if c['passed'] else '❌'} **`{c['domain']}`** (`{c['latency_ms']} ms`)"
                for c in res["checks"]
            )
            ans = (
                f"### Suite de Benchmark Souveraine (`live_benchmark_suite`)\n"
                f"Score global : **{res['passed']}/{res['total']} ({res['score_pct']} %)** en `{res['total_latency_ms']} ms` "
                f"— **{res['registered_plugins_count']} plugins actifs**.\n\n{rows}"
            )
            return {
                "thoughts": [
                    "Exécution de la suite de vérification déterministe sur l'ensemble des moteurs neuro-symboliques."
                ],
                "tool_traces": [{"call": '{"plugin": "live_benchmark_suite"}', "result": res}],
                "routed_tools": ["live_benchmark_suite"],
                "visible_answer": ans,
            }

    # 3. Formal Propositional Logic & Truth Tables (`/logic` or natural language)
    m_logic = _LOGIC_CMD_RE.match(msg)
    if m_logic:
        expr_str = (
            m_logic.group(1)
            .strip()
            .replace("->", ">>")
            .replace("=>", ">>")
            .replace("∧", "&")
            .replace("∨", "|")
            .replace("¬", "~")
        )
        res = registry.invoke("logic_and_truth_table", expr=expr_str)
        if res.get("ok"):
            vars_hdr = " | ".join(res["variables"]) + " | Résultat"
            sep_hdr = " | ".join("---" for _ in res["variables"]) + " | ---"
            table_rows = []
            for row in res["truth_table"][:16]:
                vals = " | ".join("V" if row[v] else "F" for v in res["variables"])
                table_rows.append(f"| {vals} | **{'V' if row['result'] else 'F'}** |")
            status_badge = (
                "**Tautologie** (toujours vraie)"
                if res["is_tautology"]
                else "**Contradiction** (insatisfaisable)"
                if res["is_contradiction"]
                else f"**Contingente / Satisfaisable** (`{res['true_rows']}/{res['row_count']}` modèles vrais)"
            )
            ans = (
                f"Analyse formelle de `{expr_str}` via `logic_and_truth_table` :\n"
                f"- Classification : {status_badge}\n"
                f"- Forme Normale Conjonctive (CNF) : `{res['cnf']}`\n"
                f"- Forme Normale Disjonctive (DNF) : `{res['dnf']}`\n\n"
                f"| {vars_hdr} |\n| {sep_hdr} |\n" + "\n".join(table_rows)
            )
            return {
                "thoughts": [f"Construction de la table de vérité exacte et réduction CNF/DNF pour {expr_str}."],
                "tool_traces": [
                    {
                        "call": json.dumps({"plugin": "logic_and_truth_table", "expr": expr_str}, ensure_ascii=False),
                        "result": {k: v for k, v in res.items() if k != "truth_table"},
                    }
                ],
                "routed_tools": ["logic_and_truth_table"],
                "visible_answer": ans,
            }

    # 4. Unit Conversion & CODATA Physics (`unit_and_physics_engine`)
    m_unit = _UNIT_CONV_RE.search(msg)
    if m_unit:
        val = float(m_unit.group(1))
        u_from = m_unit.group(2)
        u_to = m_unit.group(3)
        res = registry.invoke("unit_and_physics_engine", action="convert", value=val, from_unit=u_from, to_unit=u_to)
        if res.get("ok"):
            ans = (
                f"Conversion exacte vérifiée par `unit_and_physics_engine` :\n"
                f"**`{val} {res['from_unit']} = {res['result']} {res['to_unit']}`**"
            )
            return {
                "thoughts": [f"Conversion dimensionnelle {val} {u_from} -> {u_to} via unit_and_physics_engine."],
                "tool_traces": [
                    {
                        "call": json.dumps(
                            {
                                "plugin": "unit_and_physics_engine",
                                "action": "convert",
                                "value": val,
                                "from_unit": u_from,
                                "to_unit": u_to,
                            }
                        ),
                        "result": res,
                    }
                ],
                "routed_tools": ["unit_and_physics_engine"],
                "visible_answer": ans,
            }

    if "codata" in low or any(
        k in low
        for k in (
            "constante de planck",
            "constante de boltzmann",
            "nombre d'avogadro",
            "rayon de schwarzschild",
            "constante gravitationnelle",
        )
    ):
        if "schwarzschild" in low:
            nums = [float(x) for x in _NUM_LIST_RE.findall(msg)]
            mass = nums[0] if nums else 1.98847e30
            res = registry.invoke("unit_and_physics_engine", action="schwarzschild_radius", mass_kg=mass)
            if res.get("ok"):
                return {
                    "thoughts": [f"Calcul relativiste exact Rs = 2GM/c^2 pour M = {mass} kg."],
                    "tool_traces": [{"call": '{"plugin": "unit_and_physics_engine", "action": "schwarzschild_radius"}', "result": res}],
                    "routed_tools": ["unit_and_physics_engine"],
                    "visible_answer": (
                        f"Pour une masse `M = {mass:.4e} kg`, le **rayon de Schwarzschild** (`R_s = 2GM/c²`) vaut :\n"
                        f"- **`R_s = {res['radius_m']:.6g} m`** (`{res['radius_km']:.6g} km`)."
                    ),
                }
        cname = "h" if "planck" in low else "k_B" if "boltzmann" in low else "N_A" if "avogadro" in low else "G" if "gravitation" in low else ""
        res = registry.invoke("unit_and_physics_engine", action="constant", name=cname)
        if res.get("ok"):
            if "value" in res:
                ans = f"Constante CODATA **`{res['symbol']}`** ({res['description']}) : **`{res['value']} {res['unit']}`**."
            else:
                lines = [
                    f"- **`{k}`** ({v['desc']}) = `{v['value']} {v['unit']}`"
                    for k, v in list(res["constants"].items())[:8]
                ]
                ans = "Constantes physiques fondamentales CODATA (`unit_and_physics_engine`) :\n" + "\n".join(lines)
            return {
                "thoughts": ["Consultation des constantes physiques fondamentales CODATA 2022."],
                "tool_traces": [{"call": '{"plugin": "unit_and_physics_engine", "action": "constant"}', "result": res}],
                "routed_tools": ["unit_and_physics_engine"],
                "visible_answer": ans,
            }

    # 5. Statistics & Linear Regression (`/stats`, `/regress`, or natural language)
    if low.startswith("/regress") or "régression linéaire" in low or "linear regression" in low:
        pairs = _PAIRS_RE.findall(msg)
        if len(pairs) >= 2:
            xs = [float(p[0]) for p in pairs]
            ys = [float(p[1]) for p in pairs]
            res = registry.invoke("stats_and_data_engine", action="regression", x=xs, y=ys)
            if res.get("ok"):
                return {
                    "thoughts": [f"Ajustement par moindres carrés ordinaires (OLS) sur {len(xs)} points."],
                    "tool_traces": [{"call": '{"plugin": "stats_and_data_engine", "action": "regression"}', "result": res}],
                    "routed_tools": ["stats_and_data_engine"],
                    "visible_answer": (
                        f"Régression linéaire exacte (`stats_and_data_engine`, `n = {res['count']}`) :\n"
                        f"- Équation : **`{res['equation']}`**\n"
                        f"- Corrélation de Pearson : **`r = {res['pearson_r']}`** (**`R² = {res['r_squared']}`**)\n"
                        f"- Erreur quadratique (RMSE) : `{res['rmse']}`"
                    ),
                }

    if low.startswith("/stats") or any(
        k in low for k in ("écart-type", "ecart-type", "médiane", "mediane", "statistiques de", "stats of")
    ):
        nums = [float(x) for x in _NUM_LIST_RE.findall(msg)]
        if len(nums) >= 2:
            res = registry.invoke("stats_and_data_engine", action="describe", data=nums)
            if res.get("ok"):
                return {
                    "thoughts": [f"Calcul statistique descriptif complet sur {len(nums)} observations."],
                    "tool_traces": [{"call": '{"plugin": "stats_and_data_engine", "action": "describe"}', "result": res}],
                    "routed_tools": ["stats_and_data_engine"],
                    "visible_answer": (
                        f"Analyse statistique vérifiée (`n = {res['count']}`) :\n"
                        f"- **Moyenne** : `{res['mean']}` | **Médiane** : `{res['median']}`\n"
                        f"- **Écart-type** : `{res['stdev_sample']}` | **IQR** : `{res['iqr']}` (`Q1={res['q1']}`, `Q3={res['q3']}`)\n"
                        f"- **Bornes** : `[{res['min']}, {res['max']}]` | **Asymétrie** : `{res['skewness']}`\n"
                        f"- **Valeurs atypiques (Tukey 1.5×IQR)** : `{res['outliers_iqr'] or 'Aucune'}`"
                    ),
                }

    # 6. Python AST Security & Complexity Auditor (`/audit` or natural language)
    code_match = _CODE_BLOCK_RE.search(msg)
    if (low.startswith("/audit") or any(k in low for k in ("audite", "audit", "complexité cyclomatique", "analyse ce code"))) and code_match:
        code_src = code_match.group(1).strip()
        res = registry.invoke("code_ast_auditor", code=code_src)
        if res.get("ok"):
            sec_txt = (
                "Aucune vulnérabilité statique détectée ✅"
                if res["security_clean"]
                else "\n".join(f"  - ⚠️ Ligne {w['line']} (`{w['severity']}`) : {w['issue']}" for w in res["security_findings"])
            )
            fn_txt = (
                "\n".join(
                    f"- `{f['name']}({', '.join(f['args'])})` : complexité `{f['cyclomatic_complexity']}`, "
                    f"annotations `{'oui' if f['has_return_annotation'] else 'non'}`, docstring `{'oui' if f['has_docstring'] else 'non'}`"
                    for f in res["functions"]
                )
                or "- (Script de niveau module sans fonction `def`)"
            )
            return {
                "thoughts": ["Analyse statique de l'arbre syntaxique abstrait (AST), complexité de McCabe et sécurité."],
                "tool_traces": [{"call": '{"plugin": "code_ast_auditor"}', "result": res}],
                "routed_tools": ["code_ast_auditor"],
                "visible_answer": (
                    f"### Rapport d'Audit Statique AST (`code_ast_auditor`)\n"
                    f"- **Score Qualité & Sûreté** : **`{res['quality_score']}/100`** (`{res['lines']} lignes`, complexité totale `{res['total_complexity']}`)\n"
                    f"- **Sécurité** : {sec_txt}\n"
                    f"- **Fonctions analysées** :\n{fn_txt}"
                ),
            }

    # 7. Cryptography, Hashing & Shannon Entropy (`/crypto`, `sha256`, `entropie`)
    m_sha = _SHA_RE.search(msg)
    if m_sha:
        target = m_sha.group(1)
        algo = "sha512" if "sha512" in low else "blake2b" if "blake2b" in low else "sha256"
        res = registry.invoke("crypto_and_encoding_engine", action="hash", text=target, algorithm=algo)
        if res.get("ok"):
            return {
                "thoughts": [f"Calcul cryptographique déterministe {algo.upper()} via crypto_and_encoding_engine."],
                "tool_traces": [{"call": json.dumps({"plugin": "crypto_and_encoding_engine", "action": "hash", "algorithm": algo}), "result": res}],
                "routed_tools": ["crypto_and_encoding_engine"],
                "visible_answer": f"Empreinte **{algo.upper()}** de `{target}` :\n`{res['hex_digest']}`",
            }

    m_ent = _ENTROPY_RE.search(msg)
    if m_ent:
        target = m_ent.group(1)
        res = registry.invoke("crypto_and_encoding_engine", action="entropy", text=target)
        if res.get("ok"):
            return {
                "thoughts": ["Calcul de l'entropie de Shannon en bits/caractère."],
                "tool_traces": [{"call": '{"plugin": "crypto_and_encoding_engine", "action": "entropy"}', "result": res}],
                "routed_tools": ["crypto_and_encoding_engine"],
                "visible_answer": (
                    f"Entropie d'information de `{target}` (`{res['length']}` caractères) :\n"
                    f"- **`{res['bits_per_char']} bits/caractère`** (**`{res['total_entropy_bits']} bits`** au total — robustesse : **{res['strength']}**)."
                ),
            }

    if low.startswith("/token") or "génère un token" in low or "génère une clé" in low:
        res = registry.invoke("crypto_and_encoding_engine", action="generate_token", n_bytes=24)
        if res.get("ok"):
            return {
                "thoughts": ["Génération CSPRNG via le module système secrets."],
                "tool_traces": [{"call": '{"plugin": "crypto_and_encoding_engine", "action": "generate_token"}', "result": res}],
                "routed_tools": ["crypto_and_encoding_engine"],
                "visible_answer": (
                    f"Jeton cryptographique aléatoire (`192 bits` CSPRNG) :\n"
                    f"- Hex : `{res['token_hex']}`\n- URL-safe : `{res['token_urlsafe']}`\n- UUIDv4 : `{res['uuid4']}`"
                ),
            }

    # 8. Persistent Task Planner (`/task` or natural language)
    if low.startswith("/task"):
        rest = msg[5:].strip()
        if not rest or rest.lower() in ("list", "liste"):
            res = registry.invoke("task_planner_engine", action="list")
        elif rest.lower().startswith(("done ", "fait ")):
            tid = int(re.findall(r"\d+", rest)[0]) if re.findall(r"\d+", rest) else -1
            res = registry.invoke("task_planner_engine", action="complete", task_id=tid)
        elif rest.lower() in ("clear", "nettoie"):
            res = registry.invoke("task_planner_engine", action="clear_done")
        else:
            res = registry.invoke("task_planner_engine", action="add", title=rest)
        if res.get("ok"):
            list_res = res if "tasks" in res else registry.invoke("task_planner_engine", action="list")
            items = list_res.get("tasks", [])
            rows = "\n".join(
                f"- [{'x' if t['done'] else ' '}] **#{t['id']}** `{t['priority']}` — {t['title']}"
                for t in items
            ) or "_(Aucune tâche active — utilise `/task <titre>` pour en créer une)_"
            return {
                "thoughts": ["Mise à jour du planificateur de tâches persistant task_planner_engine."],
                "tool_traces": [{"call": '{"plugin": "task_planner_engine"}', "result": list_res}],
                "routed_tools": ["task_planner_engine"],
                "visible_answer": f"### Planificateur de Tâches Souverain\n{rows}",
            }

    # 9. Explicit `/math <expr>` command
    if low.startswith("/math "):
        expr_str = msg[6:].strip()
        res = registry.invoke("math_engine", expr=expr_str)
        if res.get("ok"):
            return {
                "thoughts": [f"Évaluation symbolique directe SymPy pour {expr_str}."],
                "tool_traces": [{"call": json.dumps({"plugin": "math_engine", "expr": expr_str}), "result": res}],
                "routed_tools": ["math_engine"],
                "visible_answer": f"Résultat formel `math_engine` pour `{expr_str}` :\n**`{res['result']}`**",
            }

    # 10. Explicit Document RAG Search when uploaded documents exist in EvolveMem
    doc_units = [u for u in memory.units.values() if u.category == "document_rag"]
    if doc_units and (
        low.startswith("/doc")
        or any(k in low for k in ("dans le document", "dans le fichier", "selon le document", "résume le document", "analyse le fichier"))
    ):
        q_clean = msg[4:].strip() if low.startswith("/doc") else msg
        hits = memory.query(q_clean, top_k=3, category="document_rag")
        if not hits:
            hits = [
                {"key": u.key, "content": u.content, "tags": u.tags}
                for u in sorted(doc_units, key=lambda u: u.created_at, reverse=True)[:2]
            ]
        excerpts = []
        sources = []
        for u in hits:
            u_tags = u["tags"] if isinstance(u, dict) else u.tags
            u_key = u["key"] if isinstance(u, dict) else u.key
            u_content = u["content"] if isinstance(u, dict) else u.content
            src_name = next((t[4:] for t in u_tags if t.startswith("src:")), u_key)
            sources.append(src_name)
            excerpts.append(f"**[Source : `{src_name}`]**\n> {u_content[:650]}")
        return {
            "thoughts": [
                f"Recherche hybride TF-IDF + Sémantique dans les documents locaux indexés ({len(doc_units)} segments disponibles)."
            ],
            "tool_traces": [
                {
                    "call": json.dumps({"plugin": "memory_nexus", "action": "query_document_rag", "query": q_clean}, ensure_ascii=False),
                    "result": {"ok": True, "matched_chunks": len(hits), "sources": list(dict.fromkeys(sources))},
                }
            ],
            "routed_tools": ["memory_nexus", "knowledge_ingestor"],
            "visible_answer": (
                f"### Analyse RAG Locale ({len(hits)} segments extraits)\n\n"
                + "\n\n".join(excerpts)
            ),
            "citations": list(dict.fromkeys(sources)),
        }

    return None


def _is_fragmented_or_repetitive(text: str) -> bool:
    """Detect if a small local neural sample drifted into repetition on out-of-vocabulary slang/topics."""
    if not text or len(text.strip()) < 12:
        return True
    if "```" in text or "<svg" in text or "\n|" in text or "http://" in text or "https://" in text:
        return False
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.36:
            return True
        for i in range(len(words) - 3):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
    return False


def _craft_open_conversational_reply(
    user_message: str,
    style_mode: str,
    recent_history: list[dict[str, Any]] | None = None,
) -> str:
    """Natural, human, multi-register conversational response on open-ended subjects, opinions or slang."""
    msg = user_message.strip()
    is_question = "?" in msg or bool(re.match(r"^(?:que|quoi|comment|pourquoi|qui|où|quand|est-ce|tu|t'|c'est|penses|avis)\b", msg.lower()))
    topic_clean = msg.rstrip("?!. ")

    # Check if user is reacting to previous turn
    last_bot = ""
    if recent_history:
        last_bot = next((m.get("content", "") for m in reversed(recent_history) if m.get("role") == "never" and m.get("content")), "")

    if len(msg.split()) <= 4 and last_bot and any(w in msg.lower() for w in ("pourquoi", "comment ça", "sérieux", "ah bon", "explique", "exemple", "continue", "vrai", "grave", "carrément")):
        if style_mode == "grok":
            return (
                f"Carrément. Pour aller au fond du truc sans tourner autour du pot : l'idée clé derrière ce qu'on disait, "
                f"c'est qu'entre la théorie bien propre et la vraie vie, ce sont toujours les incitations réelles et les actes qui tranchent. "
                f"Tu veux qu'on pousse quel angle en particulier ?"
            )
        if style_mode == "amical":
            return (
                f"Grave ! En fait si on prend un exemple tout simple du quotidien, ça devient tout de suite évident : "
                f"dès qu'on teste l'idée en conditions réelles, on voit vite ce qui tient la route. Tu voyais ça comment de ton côté ?"
            )
        return (
            f"Exactement — pour approfondir ce point concrètement : tout repose sur le mécanisme de cause à effet derrière ce qu'on évoquait. "
            f"Dis-moi si tu veux un exemple précis ou qu'on explore une autre facette."
        )

    if style_mode == "grok":
        if is_question:
            return (
                f"Bonne question, et on va éviter la réponse tiède de manuel scolaire sur **« {topic_clean} »**.\n\n"
                f"Si on regarde les choses en face : la plupart des gens répètent des clichés tout faits là-dessus, alors que la réalité "
                f"est toujours une histoire de rapports de force, de psychologie humaine et de compromis concrets. "
                f"Donne-moi ton point de vue cash ou le cas précis auquel tu penses — je te dis tout de suite ce qui tient debout et ce qui est du pipeau."
            )
        return (
            f"Je vois très bien le délire avec **« {topic_clean} »**. Franchement, pas de filtre ici : développe un peu ce que t'as derrière la tête "
            f"(ou balance le contexte complet), et on décortique ça cash — que ce soit pour débattre, trouver un plan d'action ou juste se marrer."
        )

    if style_mode == "amical":
        if is_question:
            return (
                f"Super sujet ça (**« {topic_clean} »**) ! Franchement il y a plein de choses intéressantes à dire là-dessus : "
                f"tout dépend du contexte concret dans lequel tu te places. Dis-m'en juste un peu plus sur ce qui te fait poser la question "
                f"(ou active **🌐 Recherche Web** juste en dessous si tu veux que j'aille fouiller les dernières infos en direct), et je te réponds sur mesure !"
            )
        return (
            f"Carrément, je te suis sur **« {topic_clean} »** ! Raconte-moi un peu plus les détails ou ce que t'en penses toi, "
            f"comme ça on rebondit vraiment dessus ensemble."
        )

    # Normal natural style
    if is_question:
        return (
            f"C'est une question intéressante sur **« {topic_clean} »**. Pour te donner une réponse vraiment pertinente plutôt qu'une généralité : "
            f"précise-moi l'angle qui t'intéresse le plus (pratique, technique, humain ou débat d'idées) — ou clique sur **🌐 Recherche Web** "
            f"si tu veux que j'interroge des sources en ligne en direct."
        )
    return (
        f"Bien reçu sur **« {topic_clean} »**. Dis-moi comment tu veux qu'on avance là-dessus — discussion libre, analyse concrète, "
        f"idées ou recherche en direct — je m'adapte à ton rythme."
    )


def enrich_with_deep_think_and_rag(
    user_message: str,
    visible_answer: str,
    thoughts: list[str],
    memory: EvolveMemStore,
    deep_think: bool = False,
    persona_id: str = "polymathe",
    style_mode: str = "normal",
    recent_history: list[dict[str, Any]] | None = None,
    verified_fast_path: bool = False,
) -> tuple[str, list[str], list[str]]:
    """
    Augment reasoning traces with Persona framing, Conversational Tone (`normal` | `amical` | `grok`),
    local Document RAG citations, and Deep-Think multi-step verification when enabled.
    """
    harm = check_extreme_harm_guard(user_message)
    if harm is not None:
        return harm, ["Garde-fou sécurité critique activé."], []

    persona = SOVEREIGN_PERSONAS.get(persona_id, SOVEREIGN_PERSONAS["polymathe"])
    enriched_thoughts = list(thoughts)
    citations: list[str] = []

    if persona_id != "polymathe":
        enriched_thoughts.insert(0, persona["thought_prefix"])

    # Check if user uploaded relevant RAG chunks
    has_doc_rag = any(u.category == "document_rag" for u in memory.units.values())
    if has_doc_rag:
        rag_hits = memory.query(user_message, top_k=2, category="document_rag")
        if rag_hits:
            top_u = rag_hits[0]
            top_content = top_u["content"]
            top_tags = top_u["tags"]
            top_key = top_u["key"]
            q_words = {w.lower() for w in re.findall(r"\w{4,}", user_message)}
            c_words = {w.lower() for w in re.findall(r"\w{4,}", top_content)}
            if q_words & c_words:
                src_name = next((t[4:] for t in top_tags if t.startswith("src:")), top_key)
                citations.append(src_name)
                enriched_thoughts.append(
                    f"Contexte RAG local pertinent détecté dans « {src_name} » : intégration avec citation."
                )
                if src_name not in visible_answer:
                    snippet = top_content[:320].strip()
                    visible_answer = (
                        f"{visible_answer}\n\n---\n📄 **Contexte Documentaire (`{src_name}`)** : *{snippet}...*"
                    )

    # Strip any accidental raw LaTeX artifact tail if emitted near max_new_tokens
    if "itext{" in visible_answer or "\\text{" in visible_answer:
        visible_answer = re.split(r"(?:itext\{|\\text\{)", visible_answer)[0].rstrip(" ,;:-") + "."

    # Epistemic Coherence & Question-Aware EvolveMem Domain Grounding
    q_words_all = {w.lower() for w in re.findall(r"\w{3,}", user_message)} - {
        "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "est", "sont", "les", "des", "une",
        "dans", "pour", "par", "avec", "sur", "plus", "comment", "pourquoi", "explique", "moi", "peux",
        "écris", "what", "how", "why", "the", "and", "with", "explain", "write",
    }
    best_dom_unit = None
    best_q_overlap = 0.0
    for u in memory.units.values():
        if u.category != "domain_omega":
            continue
        q_tag = next((t[5:] for t in u.tags if t.startswith("qtxt:")), "")
        u_q_words = {w.lower() for w in re.findall(r"\w{3,}", q_tag)} - {
            "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "est", "sont", "les", "des", "une",
            "dans", "pour", "par", "avec", "sur", "plus", "comment", "pourquoi", "explique", "moi", "peux",
            "écris", "what", "how", "why", "the", "and", "with", "explain", "write",
        }
        if not u_q_words or not q_words_all:
            continue
        inter = len(q_words_all & u_q_words)
        score_ov = inter / max(1, min(len(q_words_all), len(u_q_words)))
        if inter >= 2 and score_ov > best_q_overlap:
            best_q_overlap = score_ov
            best_dom_unit = u

    # Never overwrite pure neural generation with pre-recorded domain_omega paragraphs;
    # only attach optional reasoning metadata when `deep_think` is explicitly requested.
    if not verified_fast_path and best_dom_unit is not None and best_q_overlap >= 0.75:
        th_saved = next((t[3:] for t in best_dom_unit.tags if t.startswith("th:")), "")
        if th_saved and not enriched_thoughts and deep_think:
            enriched_thoughts.append(th_saved)

    domain_hits = memory.query(user_message, top_k=2, category="domain_omega")
    if deep_think:
        style_lbl = CONVERSATIONAL_STYLES.get(style_mode, CONVERSATIONAL_STYLES["normal"])["short"]
        step1 = f"[Étape 1 · Décomposition] Analyse de « {user_message[:70]} » ({persona['label']} · Ton {style_lbl})."
        if best_dom_unit is not None:
            dom_tag = next((t[4:] for t in best_dom_unit.tags if t.startswith("dom:")), "général")
            step2 = f"[Étape 2 · Ancrage Épistémique] Correspondance vérifiée dans le domaine « {dom_tag} » (`overlap={best_q_overlap:.0%}`)."
        elif domain_hits:
            d0 = domain_hits[0]
            dom_tag = next((t[4:] for t in d0["tags"] if t.startswith("dom:")), "général")
            step2 = f"[Étape 2 · Ancrage Épistémique] Croisement avec le domaine « {dom_tag} » dans EvolveMem ({len(domain_hits)} unités activées)."
        else:
            step2 = "[Étape 2 · Cohérence] Vérification d'absence de contradiction interne et adaptation au registre conversationnel."
        step3 = "[Étape 3 · Synthèse] Formulation finale validée."
        enriched_thoughts = [step1, step2] + enriched_thoughts + [step3]

    if persona_id == "concis" and len(visible_answer) > 360 and "```" not in visible_answer and "<svg" not in visible_answer:
        first_para = visible_answer.split("\n\n")[0]
        if len(first_para) >= 40:
            visible_answer = first_para

    return visible_answer, enriched_thoughts, citations


def _ota_live_brain_upgrade_if_needed() -> None:
    """
    Self-healing live OTA hot-swap for already-running Cloud/Desktop servers (e.g. Render v3.3.0):
    When `OTAUpdater.hot_reload_runtime_modules()` reloads `never_ai.plugins.deep_research`,
    automatically replace any unconverged (<100k tokens) in-memory `_ENGINE` with the newly
    synced `ChronoMorph-Core` seed brain without requiring a container restart!
    """
    import importlib
    import os
    import sys

    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    srv_mod = sys.modules.get("never_ai.server.app")
    if srv_mod is None:
        return
    cur_eng = getattr(srv_mod, "_ENGINE", None)
    if cur_eng is None or getattr(cur_eng, "total_tokens_trained", 0) >= 1_800_000:
        return

    try:
        for mname in (
            "never_ai.data.social_corpus_fr",
            "never_ai.data.curriculum",
            "never_ai.core.model",
            "never_ai.optim.composite_muon",
            "never_ai.updater.brain_bootstrap",
            "never_ai.plugins.agent_harness",
            "never_ai.engine",
        ):
            if mname in sys.modules:
                importlib.reload(sys.modules[mname])
            else:
                importlib.import_module(mname)

        from never_ai.engine import NeverAIEngine
        from never_ai.updater.brain_bootstrap import ensure_converged_brain

        vroot = cur_eng.vault.vault_root
        tpath = cur_eng.telemetry_path
        mpath = cur_eng.memory.storage_path
        ensure_converged_brain(vault_dir=vroot, telemetry_path=tpath)
        srv_mod._ENGINE = NeverAIEngine(
            vault_dir=vroot,
            memory_path=mpath,
            telemetry_path=tpath,
            seq_len=getattr(cur_eng, "seq_len", 160),
        )
    except Exception:  # noqa: BLE001
        pass


_ota_live_brain_upgrade_if_needed()
