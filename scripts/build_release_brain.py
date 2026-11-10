#!/usr/bin/env python3
"""Cumulatively Evolve the Shared Global Never-AI (`Sprout-MoE`) Brain on GitHub Releases.

1. Pulls the existing global `never-ai-brain-sprout-moe.zip` from GitHub Releases so prior
   collective training is 100% preserved (never starts over from scratch!).
2. Harvests live social network posts/replies (X mirrors, French microblogging, Reddit FR)
   and runs a cumulative Muon + Titans training pass.
3. Packages `never-ai-brain-sprout-moe.zip` + `global_lineage.json` for GitHub Releases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import zipfile

from never_ai.engine import NeverAIEngine


def main() -> None:
    out_zip = Path(sys.argv[1] if len(sys.argv) > 1 else "never-ai-brain-sprout-moe.zip")
    os.environ["NEVER_AI_BOOTSTRAP_GITHUB_BRAIN"] = "1"

    print("[1/4] Bootstrapping existing global Never-AI brain from GitHub Releases...", flush=True)
    eng = NeverAIEngine(
        vault_dir="checkpoints_build",
        memory_path="mem_build/evolvemem.json",
        telemetry_path="logs_build/training_telemetry.jsonl",
        seq_len=128,
    )

    if eng.model.cfg.stage_name == "Seed-0":
        print("[2/4] Growing Seed-0 -> Seed-1 -> Sprout-MoE...", flush=True)
        eng.train_steps(num_steps=20, batch_size=6)
        eng.trigger_morphogenesis("next_stage")
        eng.train_steps(num_steps=16, batch_size=6)
        eng.trigger_morphogenesis("next_stage")

    print("[2.5/4] Harvesting live social network stream (X / Microblog FR / Reddit FR) & training cumulatively...", flush=True)
    soc_res = eng.ingest_social_stream(x_query="lang:fr", train_steps=14)
    print("   Social harvest:", soc_res.get("counts"), flush=True)
    eng.train_steps(num_steps=16, batch_size=4)

    status = eng.global_sync.fetch_global_status(local_tokens_trained=eng.total_tokens_trained)
    prev_lin = status.get("lineage") or {}
    lineage_doc = {
        "model_name": "Never-AI (ChronoMorph-LM)",
        "stage_name": eng.model.cfg.stage_name,
        "vocab_size": eng.tokenizer.vocab_size,
        "contributions_count": int(prev_lin.get("contributions_count", 0)) + 1,
        "global_tokens_trained": int(prev_lin.get("global_tokens_trained", 0)) + max(8192, eng.total_tokens_trained),
        "last_contributor": "github-actions-global-evolver",
        "updated_at_unix": round(time.time(), 2),
    }
    Path("global_lineage.json").write_text(json.dumps(lineage_doc, indent=2), encoding="utf-8")

    ckpt_dir = Path("checkpoints_build/latest")
    telem_p = Path("logs_build/training_telemetry.jsonl")
    assert ckpt_dir.exists(), "Checkpoint latest missing!"

    print(f"[3/4] Packaging {out_zip} + global_lineage.json...", flush=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in ckpt_dir.iterdir():
            if fp.is_file():
                zf.write(fp, arcname=f"latest/{fp.name}")
        zf.write("global_lineage.json", arcname="global_lineage.json")
        if telem_p.exists():
            zf.write(telem_p, arcname="training_telemetry.jsonl")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[4/4] Done! Evolved global Never-AI brain: {out_zip} ({size_mb:.2f} MiB)", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
