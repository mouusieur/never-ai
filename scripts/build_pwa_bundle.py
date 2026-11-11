#!/usr/bin/env python3
"""Modular PWA v5.0 Compiler & Syntax Verifier for Never-AI.

Assembles `src/never_ai/pwa_modules/` into:
  - `src/never_ai/server/static/index.html`
  - `mobile_pwa/index.html`
  - `docs/index.html`
  - `app/index.html`
  - `NeverAI-PWA.zip` (mirrored to `docs/` and `app/`)

Validates compiled JavaScript syntax via `node --check` before writing outputs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MOD_DIR = REPO_ROOT / "src" / "never_ai" / "pwa_modules"
STATIC_INDEX = REPO_ROOT / "src" / "never_ai" / "server" / "static" / "index.html"


def build_all() -> None:
    tpl_path = MOD_DIR / "base_template.html"
    if not tpl_path.exists():
        shutil.copy2(STATIC_INDEX, tpl_path)

    tpl = tpl_path.read_text(encoding="utf-8")
    css_v5 = (MOD_DIR / "styles_v5.css").read_text(encoding="utf-8")
    js_parts = [
        (MOD_DIR / "vault_idb_v5.js").read_text(encoding="utf-8"),
        (MOD_DIR / "rp_and_router_v5.js").read_text(encoding="utf-8"),
        (MOD_DIR / "multimodal_v5.js").read_text(encoding="utf-8"),
    ]
    js_v5 = "\n\n".join(js_parts)

    assert "/* __STYLES_V5__ */" in tpl, "Missing /* __STYLES_V5__ */ marker in base_template.html"
    assert "/* __MODULES_V5__ */" in tpl, "Missing /* __MODULES_V5__ */ marker in base_template.html"

    compiled = tpl.replace("/* __STYLES_V5__ */", css_v5).replace("/* __MODULES_V5__ */", js_v5)

    # Extract <script> block and run `node --check`
    s_start = compiled.index("<script>") + len("<script>")
    s_end = compiled.rindex("</script>")
    script_body = compiled[s_start:s_end]
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as tf:
        tf.write(script_body)
        tmp_js = Path(tf.name)
    try:
        subprocess.run(["node", "--check", str(tmp_js)], check=True)
    finally:
        tmp_js.unlink(missing_ok=True)

    # Write compiled index.html targets
    STATIC_INDEX.write_text(compiled, encoding="utf-8")
    pwa_dir = REPO_ROOT / "mobile_pwa"
    (pwa_dir / "index.html").write_text(compiled, encoding="utf-8")

    # Build NeverAI-PWA.zip
    out_zip = REPO_ROOT / "NeverAI-PWA.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(pwa_dir.iterdir()):
            if fp.is_file():
                zf.write(fp, arcname=f"NeverAI-PWA/{fp.name}")

    # Sync to docs/ and app/
    for dst_name in ("docs", "app"):
        dst = REPO_ROOT / dst_name
        dst.mkdir(parents=True, exist_ok=True)
        for asset in ("index.html", "manifest.webmanifest", "sw.js", "icon-192.png", "icon-512.png", "favicon.ico", "LISEZMOI-PWA.md"):
            src_f = pwa_dir / asset
            if src_f.exists():
                shutil.copy2(src_f, dst / asset)
        shutil.copy2(out_zip, dst / "NeverAI-PWA.zip")

    print(f"OK: Compiled Never-AI v5.0 ({len(compiled):,} chars, JS syntax verified via node --check)")


if __name__ == "__main__":
    build_all()
