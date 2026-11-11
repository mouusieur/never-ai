"""
Omega Suite: 8 Advanced Deterministic Sovereign Plugins (Plugins #8 -> #15)
expanding Never-AI into a full polymathic scientific, coding, data & productivity workstation.
"""

from __future__ import annotations

import ast
import base64
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import secrets
import time
from typing import Any
import uuid

import numpy as np
import sympy as sp

from never_ai.plugins.base import NeverPlugin


# ============================================================================
# 8. VECTOR SVG MATHEMATICAL PLOTTER (`plot_svg_engine`)
# ============================================================================
class PlotSVGEnginePlugin(NeverPlugin):
    name = "plot_svg_engine"
    description = "Plot 1 or more mathematical functions f(x) as a crisp Monolith Noir vector SVG chart."
    keywords = [
        "trace", "courbe", "graphique", "plot", "fonction", "svg", "dessine", "chart", "sin", "cos",
    ]
    parameters_schema = {
        "expressions": {"type": "array", "description": "Function(s) of x (list or ';'-separated string)"},
        "x_min": {"type": "number", "description": "Minimum x (default -6.28)"},
        "x_max": {"type": "number", "description": "Maximum x (default 6.28)"},
        "title": {"type": "string", "description": "Optional chart title"},
    }

    def execute(
        self,
        expr: str = "",
        expressions: list[str] | str | None = None,
        x_min: float = -6.28318,
        x_max: float = 6.28318,
        title: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        x_min = float(x_min)
        x_max = float(x_max)
        if x_max <= x_min:
            return {"ok": False, "error": "x_max must be greater than x_min"}

        if expressions is not None:
            if isinstance(expressions, str):
                raw_funcs = [s.strip() for s in expressions.split(";") if s.strip()][:4]
            else:
                raw_funcs = [str(s).strip() for s in expressions if str(s).strip()][:4]
        else:
            raw_funcs = [s.strip() for s in (expr or "sin(x)").split(";") if s.strip()][:4]

        if not raw_funcs:
            return {"ok": False, "error": "Empty function expression"}

        x_sym = sp.Symbol("x")
        xs = np.linspace(x_min, x_max, 260)
        curves: list[tuple[str, np.ndarray]] = []
        series_summary: list[dict[str, Any]] = []

        safe_locals = {
            "x": x_sym,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "pi": sp.pi,
            "E": sp.E,
        }

        for f_str in raw_funcs:
            try:
                f_clean = f_str.replace("^", "**")
                if "=" in f_clean:
                    f_clean = f_clean.split("=", 1)[-1].strip()
                sym_expr = sp.sympify(f_clean, locals=safe_locals)
                fn = sp.lambdify(x_sym, sym_expr, modules=["numpy"])
                ys_raw = np.asarray(fn(xs), dtype=np.float64)
                if ys_raw.ndim == 0:
                    ys_raw = np.full_like(xs, float(ys_raw))
                ys = np.where(np.isfinite(ys_raw), ys_raw, np.nan)
                valid = ys[~np.isnan(ys)]
                if valid.size == 0:
                    return {"ok": False, "error": f"No finite values on [{x_min}, {x_max}] for {f_str}"}
                # Detect approximate zero crossings
                roots: list[float] = []
                for i in range(len(xs) - 1):
                    y0, y1 = ys[i], ys[i + 1]
                    if np.isfinite(y0) and np.isfinite(y1):
                        if abs(y0) < 1e-9:
                            roots.append(round(float(xs[i]), 3))
                        elif y0 * y1 < 0 and abs(y1 - y0) < 10.0 * (np.max(valid) - np.min(valid) + 1.0):
                            xr = xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0)
                            roots.append(round(float(xr), 3))
                curves.append((f_clean, ys))
                series_summary.append(
                    {
                        "expr": f_clean,
                        "y_min": round(float(np.min(valid)), 4),
                        "y_max": round(float(np.max(valid)), 4),
                        "approx_roots": roots[:6],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"Cannot plot '{f_str}': {exc}"}

        all_valid = np.concatenate([ys[~np.isnan(ys)] for _, ys in curves])
        y_lo = float(np.percentile(all_valid, 1))
        y_hi = float(np.percentile(all_valid, 99))
        if abs(y_hi - y_lo) < 1e-6:
            y_lo -= 1.0
            y_hi += 1.0
        pad_y = 0.12 * (y_hi - y_lo)
        y_lo -= pad_y
        y_hi += pad_y

        W, H = 580, 280
        pad_l, pad_r, pad_t, pad_b = 48, 22, 32, 32
        pw = W - pad_l - pad_r
        ph = H - pad_t - pad_b

        def map_x(xv: float) -> float:
            return pad_l + ((xv - x_min) / (x_max - x_min)) * pw

        def map_y(yv: float) -> float:
            return pad_t + (1.0 - (yv - y_lo) / (y_hi - y_lo)) * ph

        colors = ["#EFF3F4", "#1D9BF0", "#00BA7C", "#F59E0B"]
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;background:#050608;border:1px solid #2F3336;border-radius:12px;display:block;">',
        ]
        if title:
            safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            svg_lines.append(
                f'<text x="{W - pad_r}" y="18" fill="#71767B" font-size="10.5" text-anchor="end" font-family="sans-serif">{safe_title[:42]}</text>'
            )

        for gx in np.linspace(x_min, x_max, 5):
            px = map_x(float(gx))
            svg_lines.append(f'<line x1="{px:.1f}" y1="{pad_t}" x2="{px:.1f}" y2="{H-pad_b}" stroke="#191C20" stroke-width="1"/>')
            svg_lines.append(f'<text x="{px:.1f}" y="{H-10}" fill="#71767B" font-size="10" text-anchor="middle" font-family="monospace">{gx:.2g}</text>')

        for gy in np.linspace(y_lo, y_hi, 5):
            py = map_y(float(gy))
            svg_lines.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{W-pad_r}" y2="{py:.1f}" stroke="#191C20" stroke-width="1"/>')
            svg_lines.append(f'<text x="{pad_l-6}" y="{py+3:.1f}" fill="#71767B" font-size="10" text-anchor="end" font-family="monospace">{gy:.2g}</text>')

        if x_min <= 0 <= x_max:
            zx = map_x(0.0)
            svg_lines.append(f'<line x1="{zx:.1f}" y1="{pad_t}" x2="{zx:.1f}" y2="{H-pad_b}" stroke="#3E4347" stroke-width="1.2"/>')
        if y_lo <= 0 <= y_hi:
            zy = map_y(0.0)
            svg_lines.append(f'<line x1="{pad_l}" y1="{zy:.1f}" x2="{W-pad_r}" y2="{zy:.1f}" stroke="#3E4347" stroke-width="1.2"/>')

        for idx, (label, ys) in enumerate(curves):
            col = colors[idx % len(colors)]
            segments: list[str] = []
            pen_down = False
            for xv, yv in zip(xs, ys):
                if np.isnan(yv) or yv < y_lo - (y_hi - y_lo) or yv > y_hi + (y_hi - y_lo):
                    pen_down = False
                    continue
                px = map_x(float(xv))
                py = float(np.clip(map_y(float(yv)), pad_t, H - pad_b))
                cmd = "L" if pen_down else "M"
                segments.append(f"{cmd}{px:.2f},{py:.2f}")
                pen_down = True
            if segments:
                svg_lines.append(
                    f'<path d="{" ".join(segments)}" fill="none" stroke="{col}" stroke-width="2.2" stroke-linecap="round"/>'
                )
            lx = pad_l + idx * 145
            safe_lbl = label[:18].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            svg_lines.append(
                f'<text x="{lx}" y="18" fill="{col}" font-size="11" font-weight="bold" font-family="monospace">f{idx+1}={safe_lbl}</text>'
            )

        svg_lines.append("</svg>")
        return {
            "ok": True,
            "svg": "".join(svg_lines),
            "domain": [round(x_min, 3), round(x_max, 3)],
            "series": series_summary,
            "curves": series_summary,
        }


# ============================================================================
# 9. STATISTICS, CORRELATION & OLS REGRESSION ENGINE (`stats_and_data_engine`)
# ============================================================================
class StatsAndDataEnginePlugin(NeverPlugin):
    name = "stats_and_data_engine"
    description = "Compute descriptive statistics, IQR outliers, Pearson correlation & OLS linear regression."
    keywords = ["statistiques", "moyenne", "écart-type", "médiane", "régression", "corrélation", "stats", "data"]
    parameters_schema = {
        "action": {"type": "string", "description": "describe | regression"},
        "data": {"type": "array", "description": "Numeric list X"},
        "x": {"type": "array", "description": "Predictor list X for OLS regression"},
        "y": {"type": "array", "description": "Target list Y for OLS regression"},
    }

    def execute(
        self,
        action: str = "describe",
        data: list[float] | str | None = None,
        values: list[float] | str | None = None,
        x: list[float] | None = None,
        y: list[float] | None = None,
        values_y: list[float] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        xs_in = x if x is not None else (data if data is not None else values)
        ys_in = y if y is not None else values_y

        if isinstance(xs_in, str):
            nums = [float(tok) for tok in re.findall(r"-?\d+(?:\.\d+)?", xs_in)]
        elif xs_in is not None:
            nums = [float(v) for v in xs_in]
        else:
            nums = []

        if not nums:
            return {"ok": False, "error": "Provide at least 1 numeric value."}

        arr = np.asarray(nums, dtype=np.float64)

        if action == "regression" or ys_in is not None:
            if not ys_in:
                return {"ok": False, "error": "Provide y values for regression."}
            arr_y = np.asarray([float(v) for v in ys_in], dtype=np.float64)
            if arr_y.size != arr.size or arr.size < 2:
                return {"ok": False, "error": "x and y must have equal length >= 2."}
            slope, intercept = np.polyfit(arr, arr_y, 1)
            preds = slope * arr + intercept
            rmse = float(np.sqrt(np.mean((arr_y - preds) ** 2)))
            corr = float(np.corrcoef(arr, arr_y)[0, 1]) if np.std(arr) > 0 and np.std(arr_y) > 0 else 0.0
            reg_dict = {
                "ok": True,
                "count": int(arr.size),
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 6),
                "pearson_r": round(corr, 6),
                "r_squared": round(corr * corr, 6),
                "rmse": round(rmse, 6),
                "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            }
            reg_dict["regression"] = dict(reg_dict)
            return reg_dict

        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        outliers = [round(float(v), 6) for v in arr if v < lo_fence or v > hi_fence]
        std_pop = float(np.std(arr, ddof=0))
        std_samp = float(np.std(arr, ddof=1 if arr.size > 1 else 0))
        skew = float(np.mean(((arr - np.mean(arr)) / std_pop) ** 3)) if std_pop > 1e-12 else 0.0

        return {
            "ok": True,
            "count": int(arr.size),
            "sum": round(float(np.sum(arr)), 6),
            "mean": round(float(np.mean(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "std": round(std_samp, 6),
            "stdev_sample": round(std_samp, 6),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "q1": round(q1, 6),
            "q3": round(q3, 6),
            "iqr": round(iqr, 6),
            "skewness": round(skew, 4),
            "outliers_iqr": outliers,
        }


# ============================================================================
# 10. CODATA PHYSICS CONSTANTS & DIMENSIONAL CONVERTER (`unit_and_physics_engine`)
# ============================================================================
_CODATA_CONSTANTS: dict[str, tuple[float, str, str]] = {
    "c": (299_792_458.0, "m/s", "Vitesse de la lumière dans le vide"),
    "h": (6.62607015e-34, "J·s", "Constante de Planck"),
    "hbar": (1.054571817e-34, "J·s", "Constante de Planck réduite"),
    "G": (6.67430e-11, "m³/(kg·s²)", "Constante gravitationnelle de Newton"),
    "k_B": (1.380649e-23, "J/K", "Constante de Boltzmann"),
    "N_A": (6.02214076e23, "mol⁻¹", "Constante d'Avogadro"),
    "e": (1.602176634e-19, "C", "Charge élémentaire"),
    "m_e": (9.1093837015e-31, "kg", "Masse de l'électron"),
    "m_p": (1.67262192369e-27, "kg", "Masse du proton"),
    "g0": (9.80665, "m/s²", "Pesanteur terrestre standard"),
}

_UNIT_TO_SI: dict[str, tuple[str, float]] = {
    "m": ("length", 1.0),
    "km": ("length", 1e3),
    "cm": ("length", 1e-2),
    "mm": ("length", 1e-3),
    "um": ("length", 1e-6),
    "nm": ("length", 1e-9),
    "mi": ("length", 1609.344),
    "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    "au": ("length", 149_597_870_700.0),
    "ly": ("length", 9.4607304725808e15),
    "kg": ("mass", 1.0),
    "g": ("mass", 1e-3),
    "mg": ("mass", 1e-6),
    "t": ("mass", 1e3),
    "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "j": ("energy", 1.0),
    "kj": ("energy", 1e3),
    "mj": ("energy", 1e6),
    "wh": ("energy", 3600.0),
    "kwh": ("energy", 3.6e6),
    "cal": ("energy", 4.184),
    "kcal": ("energy", 4184.0),
    "ev": ("energy", 1.602176634e-19),
    "m/s": ("speed", 1.0),
    "km/h": ("speed", 1.0 / 3.6),
    "mph": ("speed", 0.44704),
    "pa": ("pressure", 1.0),
    "kpa": ("pressure", 1e3),
    "bar": ("pressure", 1e5),
    "atm": ("pressure", 101_325.0),
    "psi": ("pressure", 6894.75729),
    "b": ("data", 1.0),
    "kb": ("data", 1024.0),
    "mb": ("data", 1024.0**2),
    "gb": ("data", 1024.0**3),
    "tb": ("data", 1024.0**4),
}


class UnitAndPhysicsEnginePlugin(NeverPlugin):
    name = "unit_and_physics_engine"
    description = "Convert scientific units, compute Schwarzschild radius & query CODATA constants."
    keywords = ["convertis", "unité", "constante", "physique", "km", "miles", "celsius", "kelvin", "joules", "kwh"]
    parameters_schema = {
        "action": {"type": "string", "description": "convert | constant | schwarzschild_radius"},
        "value": {"type": "number", "description": "Numeric value to convert"},
        "from_unit": {"type": "string", "description": "Source unit (e.g. km/h, C, kWh, eV)"},
        "to_unit": {"type": "string", "description": "Target unit (e.g. m/s, K, J)"},
        "name": {"type": "string", "description": "CODATA constant symbol (c, h, hbar, G, k_B, N_A, e)"},
    }

    def execute(
        self,
        action: str = "convert",
        value: float = 1.0,
        from_unit: str = "",
        to_unit: str = "",
        constant: str = "",
        name: str = "",
        mass_kg: float = 1.98847e30,
        **_: Any,
    ) -> dict[str, Any]:
        if action == "schwarzschild_radius":
            G_val = _CODATA_CONSTANTS["G"][0]
            c_val = _CODATA_CONSTANTS["c"][0]
            r_m = 2.0 * G_val * float(mass_kg) / (c_val * c_val)
            return {"ok": True, "mass_kg": float(mass_kg), "radius_m": r_m, "radius_km": r_m / 1000.0}

        cname = (constant or name).strip()
        if action == "constant" or cname:
            if not cname:
                return {
                    "ok": True,
                    "constants": {
                        k: {"value": v[0], "unit": v[1], "desc": v[2]} for k, v in _CODATA_CONSTANTS.items()
                    },
                }
            item = _CODATA_CONSTANTS.get(cname)
            if not item:
                return {"ok": False, "available_constants": list(_CODATA_CONSTANTS.keys())}
            return {"ok": True, "symbol": cname, "value": item[0], "unit": item[1], "description": item[2]}

        u_src = from_unit.strip()
        u_dst = to_unit.strip()
        if u_src.upper() in ("C", "F", "K", "°C", "°F") and u_dst.upper() in ("C", "F", "K", "°C", "°F"):
            s, d = u_src.upper().lstrip("°"), u_dst.upper().lstrip("°")
            kelvin = (
                value + 273.15
                if s == "C"
                else ((value - 32.0) * 5.0 / 9.0 + 273.15 if s == "F" else value)
            )
            out = (
                kelvin - 273.15
                if d == "C"
                else ((kelvin - 273.15) * 9.0 / 5.0 + 32.0 if d == "F" else kelvin)
            )
            return {"ok": True, "from_unit": s, "to_unit": d, "result": round(out, 6), "unit": d}

        s_low, d_low = u_src.lower(), u_dst.lower()
        if s_low not in _UNIT_TO_SI or d_low not in _UNIT_TO_SI:
            return {"ok": False, "error": f"Unsupported units '{from_unit}' -> '{to_unit}'"}
        dim_s, fac_s = _UNIT_TO_SI[s_low]
        dim_d, fac_d = _UNIT_TO_SI[d_low]
        if dim_s != dim_d:
            return {"ok": False, "error": f"Incompatible dimensions: {dim_s} vs {dim_d}"}
        converted = (float(value) * fac_s) / fac_d
        return {
            "ok": True,
            "dimension": dim_s,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": converted if abs(converted) < 1e-3 or abs(converted) > 1e6 else round(converted, 6),
            "unit": to_unit,
        }


# ============================================================================
# 11. STATIC PYTHON AST QUALITY & SECURITY AUDITOR (`code_ast_auditor`)
# ============================================================================
class CodeASTAuditorPlugin(NeverPlugin):
    name = "code_ast_auditor"
    description = "Analyze Python code via AST: cyclomatic complexity, type annotations, docstrings & security risks."
    keywords = ["audit", "ast", "qualité", "complexité", "sécurité code", "analyse code", "refactor"]
    parameters_schema = {
        "code": {"type": "string", "description": "Python source code to audit"},
    }

    def execute(self, code: str = "", **_: Any) -> dict[str, Any]:
        if not code.strip():
            return {"ok": False, "error": "Empty code"}
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {"ok": False, "syntax_valid": False, "error": f"Line {exc.lineno}: {exc.msg}"}

        branch_nodes = (
            ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With, ast.BoolOp, ast.IfExp
        )
        fn_details: list[dict[str, Any]] = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_cc = 1 + sum(1 for sub in ast.walk(n) if isinstance(sub, branch_nodes))
                fn_details.append(
                    {
                        "name": n.name,
                        "args": [a.arg for a in n.args.args],
                        "cyclomatic_complexity": fn_cc,
                        "has_return_annotation": n.returns is not None,
                        "has_docstring": ast.get_docstring(n) is not None,
                    }
                )

        total_cc = 1 + sum(1 for n in ast.walk(tree) if isinstance(n, branch_nodes))
        findings: list[dict[str, Any]] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.ExceptHandler) and n.type is None:
                findings.append({"line": getattr(n, "lineno", 1), "severity": "medium", "issue": "Bare `except:` clause"})
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec"):
                findings.append(
                    {"line": getattr(n, "lineno", 1), "severity": "high", "issue": f"Unsafe `{n.func.id}()` execution"}
                )
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "system":
                findings.append(
                    {"line": getattr(n, "lineno", 1), "severity": "high", "issue": "Shell command `os.system()` call"}
                )

        typed_all = bool(fn_details) and all(f["has_return_annotation"] for f in fn_details)
        doc_all = bool(fn_details) and all(f["has_docstring"] for f in fn_details)
        score = max(
            0,
            min(
                100,
                86 + (7 if typed_all else 0) + (7 if doc_all else 0) - 20 * len(findings) - max(0, total_cc - 15) * 2,
            ),
        )
        return {
            "ok": True,
            "syntax_valid": True,
            "lines": len(code.splitlines()),
            "functions": fn_details,
            "total_complexity": total_cc,
            "security_clean": len(findings) == 0,
            "security_findings": findings,
            "quality_score": score,
        }


# ============================================================================
# 12. CRYPTOGRAPHY, HASHING & SHANNON ENTROPY (`crypto_and_encoding_engine`)
# ============================================================================
class CryptoAndEncodingPlugin(NeverPlugin):
    name = "crypto_and_encoding_engine"
    description = "Compute SHA-256/512, BLAKE2b, Base64 encode/decode, Shannon entropy & CSPRNG tokens."
    keywords = ["sha256", "hash", "base64", "entropie", "crypto", "token", "empreinte"]
    parameters_schema = {
        "action": {"type": "string", "description": "hash | base64_encode | base64_decode | entropy | generate_token"},
        "text": {"type": "string", "description": "Input string"},
        "algorithm": {"type": "string", "description": "sha256 | sha512 | blake2b"},
    }

    def execute(
        self,
        action: str = "hash",
        text: str = "",
        algorithm: str = "sha256",
        n_bytes: int = 24,
        **_: Any,
    ) -> dict[str, Any]:
        raw = text.encode("utf-8")
        if action == "hash":
            algo = algorithm.lower()
            if algo == "sha512":
                dig = hashlib.sha512(raw).hexdigest()
            elif algo == "blake2b":
                dig = hashlib.blake2b(raw, digest_size=32).hexdigest()
            else:
                dig = hashlib.sha256(raw).hexdigest()
            return {"ok": True, "algorithm": algo, "hex_digest": dig, "sha256": hashlib.sha256(raw).hexdigest()}
        if action == "base64_encode":
            return {"ok": True, "base64": base64.b64encode(raw).decode("ascii")}
        if action == "base64_decode":
            try:
                return {"ok": True, "decoded": base64.b64decode(raw).decode("utf-8", errors="replace")}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        if action == "entropy":
            if not text:
                return {"ok": True, "length": 0, "bits_per_char": 0.0, "total_entropy_bits": 0.0, "strength": "vide"}
            counts = Counter(text)
            n = len(text)
            h_val = -sum((c / n) * math.log2(c / n) for c in counts.values())
            tot = h_val * n
            strength = "très élevée" if tot >= 80 else "élevée" if tot >= 50 else "modérée" if tot >= 28 else "faible"
            return {
                "ok": True,
                "length": n,
                "bits_per_char": round(h_val, 4),
                "total_entropy_bits": round(tot, 2),
                "strength": strength,
            }
        if action in ("token", "generate_token"):
            nb = max(8, min(64, int(n_bytes)))
            return {
                "ok": True,
                "token_hex": secrets.token_hex(nb),
                "token_urlsafe": secrets.token_urlsafe(nb),
                "uuid4": str(uuid.uuid4()),
            }
        return {"ok": False, "error": f"Unknown action '{action}'"}


# ============================================================================
# 13. FORMAL PROPOSITIONAL LOGIC & SAT SOLVER (`logic_and_truth_table`)
# ============================================================================
class LogicAndTruthTablePlugin(NeverPlugin):
    name = "logic_and_truth_table"
    description = "Evaluate propositional logic formulas via SymPy, generate truth tables, CNF/DNF & SAT."
    keywords = ["logique", "table de vérité", "tautologie", "sat", "booléen", "implique"]
    parameters_schema = {
        "expr": {"type": "string", "description": "Boolean formula e.g. '((P >> Q) & P) >> Q'"},
    }

    def execute(self, expr: str = "", formula: str = "", **_: Any) -> dict[str, Any]:
        raw = (expr or formula or "(P & Q) >> P").strip()
        norm = (
            raw.replace("<->", "Equivalent")
            .replace("->", ">>")
            .replace("=>", ">>")
            .replace("&&", "&")
            .replace("||", "|")
            .replace("∧", "&")
            .replace("∨", "|")
            .replace("¬", "~")
        )
        norm = re.sub(r"\band\b", "&", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\bor\b", "|", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\bnot\b", "~", norm, flags=re.IGNORECASE)

        names = sorted(set(re.findall(r"\b[A-Za-z]\w*\b", norm)) - {"True", "False", "Equivalent", "Implies", "Xor"})
        if not names or len(names) > 6:
            return {"ok": False, "error": "Provide 1 to 6 propositional variables."}

        syms = {n: sp.Symbol(n) for n in names}
        try:
            parsed = sp.sympify(norm, locals=syms)
            cnf_str = str(sp.to_cnf(parsed, simplify=True))
            dnf_str = str(sp.to_dnf(parsed, simplify=True))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Invalid logic expression '{raw}': {exc}"}

        rows: list[dict[str, Any]] = []
        true_rows = 0
        for combo in itertools.product([False, True], repeat=len(names)):
            subs_map = {syms[n]: val for n, val in zip(names, combo)}
            res_bool = bool(parsed.subs(subs_map))
            if res_bool:
                true_rows += 1
            row_dict: dict[str, Any] = {n: val for n, val in zip(names, combo)}
            row_dict["result"] = res_bool
            rows.append(row_dict)

        is_taut = true_rows == len(rows)
        is_contra = true_rows == 0
        return {
            "ok": True,
            "variables": names,
            "row_count": len(rows),
            "true_rows": true_rows,
            "is_tautology": is_taut,
            "is_contradiction": is_contra,
            "satisfiable": true_rows > 0,
            "cnf": cnf_str,
            "dnf": dnf_str,
            "truth_table": rows,
        }


# ============================================================================
# 14. PERSISTENT TASK & PROJECT PLANNER (`task_planner_engine`)
# ============================================================================
class TaskPlannerPlugin(NeverPlugin):
    name = "task_planner_engine"
    description = "Manage persistent user tasks & project milestones (add, list, complete, clear_done)."
    keywords = ["tâche", "todo", "projet", "planifie", "task", "objectif", "liste"]
    parameters_schema = {
        "action": {"type": "string", "description": "add | list | complete | clear_done"},
        "title": {"type": "string", "description": "Task description"},
        "task_id": {"type": "integer", "description": "Task ID to complete"},
    }

    def __init__(self, store_path: str | Path = "memory_store/tasks.json") -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        if self.store_path.exists():
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        return []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.store_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def execute(
        self,
        action: str = "list",
        title: str = "",
        priority: str = "normale",
        task_id: int = -1,
        **_: Any,
    ) -> dict[str, Any]:
        tasks = self._load()
        if action == "add":
            if not title.strip():
                return {"ok": False, "error": "Task title required"}
            next_id = max((int(t["id"]) for t in tasks), default=0) + 1
            item = {
                "id": next_id,
                "title": title.strip(),
                "priority": priority,
                "done": False,
                "created_at": round(time.time(), 2),
            }
            tasks.append(item)
            self._save(tasks)
            return {"ok": True, "added": item, "tasks": tasks}
        if action == "complete":
            for t in tasks:
                if int(t["id"]) == int(task_id) or (title and title.lower() in t["title"].lower()):
                    t["done"] = True
            self._save(tasks)
            return {"ok": True, "tasks": tasks}
        if action in ("clear", "clear_done"):
            tasks = [t for t in tasks if not t.get("done")] if action == "clear_done" else []
            self._save(tasks)
            return {"ok": True, "tasks": tasks}
        return {"ok": True, "tasks": tasks}


# ============================================================================
# 15. LIVE 9-DOMAIN SCIENTIFIC BENCHMARK (`live_benchmark_suite`)
# ============================================================================
class LiveBenchmarkSuitePlugin(NeverPlugin):
    name = "live_benchmark_suite"
    description = "Run an objective 9-domain evaluation across Math, SVG, Stats, Physics, AST, Crypto, SAT, Sandbox & Introspection."
    keywords = ["benchmark", "évaluation", "score", "performance", "test complet"]
    parameters_schema = {}

    def __init__(self, registry_getter: Any) -> None:
        self._get_registry = registry_getter

    def execute(self, **_: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        reg = self._get_registry()
        checks: list[dict[str, Any]] = []

        def _run(label: str, fn: Any) -> None:
            s0 = time.perf_counter()
            ok = bool(fn())
            checks.append({"domain": label, "passed": ok, "latency_ms": round((time.perf_counter() - s0) * 1000.0, 2)})

        _run(
            "1. Mathématiques Symboliques (SymPy)",
            lambda: sp.expand(sp.sympify(reg.invoke("math_engine", expr="integrate(3*x**2 + 2*x, x)").get("result", "0")))
            == sp.Symbol("x") ** 3 + sp.Symbol("x") ** 2,
        )
        _run(
            "2. Traceur Vectoriel SVG",
            lambda: "<svg" in reg.invoke("plot_svg_engine", expressions=["sin(x)", "cos(x)"]).get("svg", ""),
        )
        _run(
            "3. Statistiques & Régression OLS",
            lambda: abs(reg.invoke("stats_and_data_engine", action="regression", x=[1, 2, 3], y=[3, 5, 7]).get("slope", 0) - 2.0) < 1e-6,
        )
        _run(
            "4. Physique CODATA & Unités",
            lambda: abs(reg.invoke("unit_and_physics_engine", action="convert", value=36, from_unit="km/h", to_unit="m/s").get("result", 0) - 10.0) < 1e-5,
        )
        _run(
            "5. Audit Statique AST & Sécurité",
            lambda: reg.invoke("code_ast_auditor", code="def f(x: int) -> int:\n    return x + 1\n").get("security_clean") is True,
        )
        _run(
            "6. Cryptographie & Entropie",
            lambda: len(reg.invoke("crypto_and_encoding_engine", action="hash", text="NeverAI").get("hex_digest", "")) == 64,
        )
        _run(
            "7. Logique Formelle & SAT",
            lambda: reg.invoke("logic_and_truth_table", expr="((P >> Q) & P) >> Q").get("is_tautology") is True,
        )
        _run(
            "8. Exécution Python Sandbox",
            lambda: (reg.invoke("python_sandbox", code="print(sum(range(1, 101)))").get("stdout") or "").strip() == "5050",
        )
        _run(
            "9. Introspection Cortex ChronoMorph",
            lambda: reg.invoke("introspection_probe").get("ok") is True,
        )

        passed_n = sum(1 for c in checks if c["passed"])
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": True,
            "passed": passed_n,
            "total": len(checks),
            "score_pct": round(100.0 * passed_n / len(checks), 1),
            "total_latency_ms": total_ms,
            "registered_plugins_count": len(reg.list_plugins()),
            "checks": checks,
        }
