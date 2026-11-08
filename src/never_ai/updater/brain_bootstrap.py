"""GitHub Releases Neural Parameter Bootstrap (`never-ai-brain-sprout-moe.zip`).

Allows any external cloud service (Hugging Face Spaces, Render, Koyeb, Railway, Fly.io)
or fresh installation to automatically fetch the trained ChronoMorph-LM (`Sprout-MoE`)
weights directly from GitHub Releases on startup.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import urllib.request
import zipfile

from never_ai.updater.ota import DEFAULT_GITHUB_REPO, USER_AGENT


BRAIN_ASSET_NAME = "never-ai-brain-sprout-moe.zip"


def detect_available_ram_mb() -> int:
    """Detect container cgroup v2 / v1 memory limit or host `/proc/meminfo` in MiB."""
    # 1. cgroup v2
    cg2 = Path("/sys/fs/cgroup/memory.max")
    if cg2.exists():
        try:
            raw = cg2.read_text(encoding="utf-8").strip()
            if raw != "max" and raw.isdigit():
                val_mb = int(raw) // (1024 * 1024)
                if 64 <= val_mb <= 1_048_576:
                    return val_mb
        except OSError:
            pass

    # 2. cgroup v1
    cg1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cg1.exists():
        try:
            raw = cg1.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                val_mb = int(raw) // (1024 * 1024)
                if 64 <= val_mb <= 1_048_576:
                    return val_mb
        except OSError:
            pass

    # 3. /proc/meminfo
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
        except (OSError, ValueError, IndexError):
            pass

    return 4096


def download_brain_from_github_releases(
    vault_dir: str | Path = "checkpoints",
    telemetry_path: str | Path | None = "logs/training_telemetry.jsonl",
    github_repo: str = DEFAULT_GITHUB_REPO,
    timeout_s: float = 45.0,
) -> dict[str, object]:
    """Download `never-ai-brain-sprout-moe.zip` from GitHub Releases into `vault_dir/latest/`."""
    vdir = Path(vault_dir)
    latest_dir = vdir / "latest"
    if (latest_dir / "config.json").exists() and (
        (latest_dir / "model_wsm.safetensors").exists() or (latest_dir / "model_live.safetensors").exists()
    ):
        return {"ok": True, "downloaded": False, "reason": "already_present"}

    url = os.environ.get(
        "NEVER_AI_BRAIN_URL",
        f"https://github.com/{github_repo}/releases/latest/download/{BRAIN_ASSET_NAME}",
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            zip_bytes = resp.read()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "downloaded": False, "url": url, "error": str(exc)}

    latest_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if name == "training_telemetry.jsonl" and telemetry_path is not None:
                tp = Path(telemetry_path)
                if not tp.exists():
                    tp.parent.mkdir(parents=True, exist_ok=True)
                    tp.write_bytes(zf.read(member))
                    extracted_files.append(str(tp))
            elif name in {"config.json", "manifest.json", "tokenizer.json", "model_wsm.safetensors", "model_live.safetensors"}:
                out_p = latest_dir / name
                out_p.write_bytes(zf.read(member))
                extracted_files.append(str(out_p))

    return {
        "ok": True,
        "downloaded": True,
        "url": url,
        "bytes": len(zip_bytes),
        "files": extracted_files,
    }
