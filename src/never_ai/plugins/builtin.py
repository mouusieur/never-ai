"""The 7 Core Plugins of Never-AI + SkillForge Autonomous Tool Synthesis."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import sympy as sp

from never_ai.memory.evolvemem import EvolveMemStore
from never_ai.plugins.base import NeverPlugin, PluginRegistry


_SAFE_SYMPY_NS: dict[str, Any] = {
    "pi": sp.pi,
    "E": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.ln,
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "isprime": sp.isprime,
    "prime": sp.prime,
    "primerange": lambda a, b: list(sp.primerange(a, b)),
    "factorint": sp.factorint,
    "simplify": sp.simplify,
    "expand": sp.expand,
    "factor": sp.factor,
    "diff": sp.diff,
    "integrate": sp.integrate,
    "solve": sp.solve,
    "Symbol": sp.Symbol,
    "Matrix": sp.Matrix,
    "sum": sum,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "range": range,
}


class MathEnginePlugin(NeverPlugin):
    name = "math_engine"
    description = "Exact symbolic & numerical math (SymPy: algebra, primes, calculus, equations)."
    keywords = [
        "math", "calcul", "somme", "équation", "dérivée", "intégrale", "premier",
        "prime", "solve", "simplify", "factor", "arithmétique", "nombre",
    ]
    parameters_schema = {
        "expr": {"type": "string", "description": "Mathematical or SymPy expression to evaluate."},
    }

    def execute(self, expr: str = "", **_: Any) -> dict[str, Any]:
        if not expr.strip():
            return {"ok": False, "error": "Empty expression"}
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                return {"ok": False, "error": "Dunder attribute access forbidden"}
        local_ns = dict(_SAFE_SYMPY_NS)
        local_ns.update({ch: sp.Symbol(ch) for ch in ("x", "y", "z", "n", "k", "t")})
        val = eval(compile(tree, "<math_engine>", "eval"), {"__builtins__": {}}, local_ns)  # noqa: S307
        if isinstance(val, sp.Basic):
            val_simplified = sp.simplify(val)
            return {"ok": True, "result": str(val_simplified)}
        return {"ok": True, "result": str(val)}


class PythonSandboxPlugin(NeverPlugin):
    name = "python_sandbox"
    description = "Execute isolated Python code in a time-limited sandbox and return stdout/stderr."
    keywords = ["python", "code", "script", "run", "exécuter", "algorithme", "test", "programme"]
    parameters_schema = {
        "code": {"type": "string", "description": "Python snippet to execute."},
        "timeout_sec": {"type": "number", "description": "Max execution seconds (<= 5)."},
    }

    def execute(self, code: str = "", timeout_sec: float = 4.0, **_: Any) -> dict[str, Any]:
        timeout = min(max(0.5, float(timeout_sec)), 5.0)
        with tempfile.TemporaryDirectory(prefix="never_py_") as tmpdir:
            script_path = Path(tmpdir) / "snippet.py"
            script_path.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[:4000],
                "stderr": proc.stderr[:2000],
            }


class MemoryNexusPlugin(NeverPlugin):
    name = "memory_nexus"
    description = "Store, query, or self-evolve long-term knowledge inside EvolveMem."
    keywords = ["memory", "mémoire", "retenir", "souvenir", "store", "recall", "rappel", "fait"]
    parameters_schema = {
        "action": {"type": "string", "description": "'store' | 'query' | 'evolve' | 'stats'"},
        "key": {"type": "string", "description": "Short topic key (for store)"},
        "content": {"type": "string", "description": "Fact/episode text (for store) or query string"},
        "tags": {"type": "array", "description": "Optional tags list"},
    }

    def __init__(self, store: EvolveMemStore) -> None:
        self.store = store

    def execute(
        self,
        action: str = "query",
        key: str = "",
        content: str = "",
        tags: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if action == "store":
            u = self.store.store(key=key or "note", content=content, tags=tags)
            return {"ok": True, "stored_id": u.id, "total_memories": len(self.store.units)}
        if action == "query":
            hits = self.store.query(content or key, top_k=4)
            return {"ok": True, "hits": hits}
        if action == "evolve":
            rep = self.store.evolve_retrieval_config()
            return {"ok": True, "evolution": rep}
        return {
            "ok": True,
            "total_memories": len(self.store.units),
            "config": self.store.config.__dict__,
        }


class WorkspaceFSPlugin(NeverPlugin):
    name = "workspace_fs"
    description = "Inspect files and directories safely inside the project workspace."
    keywords = ["file", "fichier", "dossier", "read", "lire", "codebase", "repo", "search"]
    parameters_schema = {
        "action": {"type": "string", "description": "'list' | 'read'"},
        "path": {"type": "string", "description": "Relative path inside workspace"},
    }

    def __init__(self, root_dir: str | Path = ".") -> None:
        self.root_dir = Path(root_dir).resolve()

    def _resolve_safe(self, rel: str) -> Path:
        candidate = (self.root_dir / rel).resolve()
        if not str(candidate).startswith(str(self.root_dir)):
            raise PermissionError("Path escapes workspace root")
        for part in candidate.parts:
            if part in (".git", ".venv"):
                raise PermissionError("Access to .git or .venv restricted")
        return candidate

    def execute(self, action: str = "list", path: str = ".", **_: Any) -> dict[str, Any]:
        target = self._resolve_safe(path)
        if action == "list":
            if not target.is_dir():
                return {"ok": False, "error": f"Not a directory: {path}"}
            entries = sorted(
                p.name + ("/" if p.is_dir() else "")
                for p in target.iterdir()
                if p.name not in (".git", ".venv", "__pycache__")
            )
            return {"ok": True, "entries": entries[:100]}
        if action == "read":
            if not target.is_file():
                return {"ok": False, "error": f"Not a file: {path}"}
            return {"ok": True, "content": target.read_text(encoding="utf-8", errors="replace")[:4000]}
        return {"ok": False, "error": f"Unsupported action '{action}'"}


class KnowledgeIngestorPlugin(NeverPlugin):
    name = "knowledge_ingestor"
    description = "Epistemically filter, deduplicate, and stage high-density knowledge for SEAL learning."
    keywords = ["ingest", "apprendre", "corpus", "document", "connaissance", "dataset", "seal"]
    parameters_schema = {
        "title": {"type": "string", "description": "Document title"},
        "text": {"type": "string", "description": "Raw knowledge text to filter & stage"},
        "domain": {"type": "string", "description": "Domain tag (science, math, code, fr, en, philosophy)"},
    }

    def __init__(
        self,
        mem_store: EvolveMemStore,
        staging_path: str | Path = "memory_store/seal_staging.jsonl",
    ) -> None:
        self.mem_store = mem_store
        self.staging_path = Path(staging_path)
        self._seen_hashes: set[str] = set()

    @staticmethod
    def epistemic_quality_score(text: str) -> float:
        cleaned = text.strip()
        if len(cleaned) < 30:
            return 0.0
        alpha_num = sum(c.isalnum() for c in cleaned)
        density = alpha_num / len(cleaned)
        unique_chars = len(set(cleaned))
        if density < 0.45 or unique_chars < 12:
            return 0.1
        return min(1.0, round(0.5 * density + 0.5 * min(1.0, unique_chars / 40.0), 3))

    def execute(
        self,
        title: str = "knowledge",
        text: str = "",
        domain: str = "general",
        **_: Any,
    ) -> dict[str, Any]:
        q_score = self.epistemic_quality_score(text)
        if q_score < 0.45:
            return {"ok": False, "rejected": True, "quality_score": q_score, "reason": "Low epistemic signal"}

        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
        if digest in self._seen_hashes:
            return {"ok": True, "deduplicated": True, "hash": digest}
        self._seen_hashes.add(digest)

        import json

        self.staging_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"hash": digest, "title": title, "domain": domain, "quality": q_score, "text": text.strip()}
        with self.staging_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.mem_store.store(key=title, content=text.strip()[:600], category="semantic", tags=[domain, "staged"])
        return {"ok": True, "staged_hash": digest, "quality_score": q_score}


class IntrospectionProbePlugin(NeverPlugin):
    name = "introspection_probe"
    description = "Inspect Never-AI's own live neural anatomy, parameters, Gate-Zero alphas, MoE load & lineage."
    keywords = ["introspection", "cerveau", "paramètres", "couches", "moe", "experts", "stade", "anatomy"]
    parameters_schema = {}

    def __init__(self, anatomy_provider: Callable[[], dict[str, Any]]) -> None:
        self.anatomy_provider = anatomy_provider

    def execute(self, **_: Any) -> dict[str, Any]:
        return {"ok": True, "anatomy": self.anatomy_provider()}


_FORBIDDEN_FORGE_IMPORTS = {
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "pty",
    "resource",
}
_FORBIDDEN_FORGE_CALLS = {"system", "popen", "remove", "unlink", "rmdir", "rmtree"}


class SkillForgePlugin(NeverPlugin):
    """
    Voyager & Darwin-Gödel Machine inspired self-tooling plugin with strict
    AST security + immutable pytest verification gate (Directive 8 Anti-Goodhart).
    """

    name = "skill_forge"
    description = "Synthesize, unit-test in isolation, and hot-register a brand new NeverPlugin."
    keywords = ["forge", "plugin", "outil", "créer", "extension", "skill", "capability"]
    parameters_schema = {
        "plugin_name": {"type": "string", "description": "Snake_case name of the new plugin"},
        "plugin_code": {"type": "string", "description": "Python module defining a NeverPlugin subclass"},
        "test_code": {"type": "string", "description": "Pytest suite verifying the new plugin (>= 2 tests)"},
    }

    def __init__(
        self,
        registry: PluginRegistry,
        forged_dir: str | Path = "src/never_ai/plugins/forged",
    ) -> None:
        self.registry = registry
        self.forged_dir = Path(forged_dir)
        self.forged_dir.mkdir(parents=True, exist_ok=True)
        init_f = self.forged_dir / "__init__.py"
        if not init_f.exists():
            init_f.write_text('"""Autonomously forged plugins."""\n', encoding="utf-8")

    @staticmethod
    def audit_ast_safety(code: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in _FORBIDDEN_FORGE_IMPORTS:
                        return False, f"Forbidden import '{top}'"
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in _FORBIDDEN_FORGE_IMPORTS:
                    return False, f"Forbidden import from '{top}'"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in _FORBIDDEN_FORGE_CALLS:
                    return False, f"Forbidden call '{node.func.attr}'"
        return True, "ok"

    def execute(
        self,
        plugin_name: str = "",
        plugin_code: str = "",
        test_code: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        if not plugin_name.isidentifier():
            return {"ok": False, "error": f"Invalid plugin_name '{plugin_name}'"}

        safe_p, msg_p = self.audit_ast_safety(plugin_code)
        if not safe_p:
            return {"ok": False, "error": f"AST safety gate rejected plugin_code: {msg_p}"}
        safe_t, msg_t = self.audit_ast_safety(test_code)
        if not safe_t:
            return {"ok": False, "error": f"AST safety gate rejected test_code: {msg_t}"}

        if test_code.count("def test_") < 2:
            return {"ok": False, "error": "SkillForge requires at least 2 `def test_...` unit tests."}

        # Run isolated pytest verification before promoting
        with tempfile.TemporaryDirectory(prefix="skill_forge_") as tmpdir:
            tmp_path = Path(tmpdir)
            cand_mod = tmp_path / f"{plugin_name}.py"
            cand_test = tmp_path / f"test_{plugin_name}.py"
            cand_mod.write_text(plugin_code, encoding="utf-8")
            cand_test.write_text(test_code, encoding="utf-8")

            env_pythonpath = f"{tmpdir}:{Path('src').resolve()}"
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(cand_test), "-q"],
                cwd=tmpdir,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": env_pythonpath, "HOME": tmpdir},
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "error": "Unit tests failed in SkillForge sandbox",
                    "pytest_output": (proc.stdout + "\n" + proc.stderr)[:2000],
                }

        # Promote verified plugin & hot-register!
        dest = self.forged_dir / f"{plugin_name}.py"
        dest.write_text(plugin_code, encoding="utf-8")

        spec = importlib.util.spec_from_file_location(f"never_ai.plugins.forged.{plugin_name}", dest)
        if spec is None or spec.loader is None:
            return {"ok": False, "error": "Failed to create module spec"}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        registered_cls = None
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, NeverPlugin) and obj is not NeverPlugin:
                instance = obj()
                self.registry.register(instance)
                registered_cls = instance.name
                break

        if registered_cls is None:
            dest.unlink(missing_ok=True)
            return {"ok": False, "error": "No NeverPlugin subclass found in forged module"}

        return {
            "ok": True,
            "forged_plugin": registered_cls,
            "path": str(dest),
            "pytest_summary": proc.stdout.strip(),
        }


def build_default_plugin_registry(
    mem_store: EvolveMemStore,
    anatomy_provider: Callable[[], dict[str, Any]],
    workspace_root: str | Path = ".",
) -> PluginRegistry:
    reg = PluginRegistry()
    reg.register(MathEnginePlugin())
    reg.register(PythonSandboxPlugin())
    reg.register(MemoryNexusPlugin(mem_store))
    reg.register(WorkspaceFSPlugin(workspace_root))
    reg.register(KnowledgeIngestorPlugin(mem_store))
    reg.register(IntrospectionProbePlugin(anatomy_provider))
    reg.register(SkillForgePlugin(reg))
    return reg
