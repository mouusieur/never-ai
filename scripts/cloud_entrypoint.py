#!/usr/bin/env python3
"""Universal External Cloud Entrypoint for NEVER-AI (Hugging Face Spaces, Render, Railway, Koyeb, Fly.io).

1. Uses writable `NEVER_AI_DATA_DIR` (defaults to `/data` if present, else current working directory).
2. Optionally checks GitHub for newer OTA source updates (`NEVER_AI_CLOUD_AUTO_OTA=1`).
3. Automatically downloads the trained `Sprout-MoE` neural parameters (`never-ai-brain-sprout-moe.zip`)
   from GitHub Releases (`NEVER_AI_BOOTSTRAP_GITHUB_BRAIN=1`) when `checkpoints/latest/` is empty.
4. Starts `uvicorn never_ai.server.app:app` on `0.0.0.0:$PORT` (default `7860` for Hugging Face Spaces).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import uvicorn


def main() -> None:
    data_dir = Path(os.environ.get("NEVER_AI_DATA_DIR", ".")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(data_dir)

    local_src = data_dir / "src"
    if local_src.exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))

    if os.environ.get("NEVER_AI_CLOUD_AUTO_OTA", "1") == "1" and not (data_dir / ".git").exists():
        try:
            from never_ai.updater.ota import OTAUpdater

            ota = OTAUpdater(app_root=data_dir)
            ota.apply_remote_update(force=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[NEVER-AI Cloud] OTA check skipped: {exc}", flush=True)

    os.environ.setdefault("NEVER_AI_BOOTSTRAP_GITHUB_BRAIN", "1")
    port = int(os.environ.get("PORT", "7860"))
    print(f"[NEVER-AI Cloud] Starting Sovereign Server on 0.0.0.0:{port} (data_dir={data_dir})", flush=True)

    uvicorn.run(
        "never_ai.server.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
