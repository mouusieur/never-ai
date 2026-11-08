#!/usr/bin/env python3
"""Train & Morphogenetically Grow ChronoMorph-LM (`Sprout-MoE`) and package `never-ai-brain-sprout-moe.zip`.

Used in GitHub Actions so the complete trained neural parameters live on GitHub Releases
and can be pulled automatically by any external cloud host (Hugging Face, Render, Koyeb, Railway)
and connected Android PWAs!
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import zipfile

from never_ai.engine import NeverAIEngine


def main() -> None:
    out_zip = Path(sys.argv[1] if len(sys.argv) > 1 else "never-ai-brain-sprout-moe.zip")
    os.environ["NEVER_AI_BOOTSTRAP_GITHUB_BRAIN"] = "0"

    print("[1/4] Initializing Seed-0 ChronoMorph-LM...", flush=True)
    eng = NeverAIEngine(
        vault_dir="checkpoints_build",
        memory_path="mem_build/evolvemem.json",
        telemetry_path="logs_build/training_telemetry.jsonl",
        seq_len=128,
    )

    print("[2/4] Training Seed-0 & growing Zero-Shock -> Seed-1 -> Sprout-MoE...", flush=True)
    eng.train_steps(num_steps=28, batch_size=8)
    eng.trigger_morphogenesis("next_stage")
    eng.train_steps(num_steps=20, batch_size=6)
    eng.trigger_morphogenesis("next_stage")
    eng.train_steps(num_steps=16, batch_size=4)

    ckpt_dir = Path("checkpoints_build/latest")
    telem_p = Path("logs_build/training_telemetry.jsonl")
    assert ckpt_dir.exists(), "Checkpoint latest missing!"

    print(f"[3/4] Packaging {out_zip}...", flush=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in ckpt_dir.iterdir():
            if fp.is_file():
                zf.write(fp, arcname=f"latest/{fp.name}")
        if telem_p.exists():
            zf.write(telem_p, arcname="training_telemetry.jsonl")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[4/4] Done! Built {out_zip} ({size_mb:.2f} MiB)", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
