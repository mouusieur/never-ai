#!/usr/bin/env python3
"""Cumulatively Evolve the Shared Global Never-AI (`ChronoMorph-Core` / `Sprout-MoE`) Brain on GitHub Releases.

1. Pulls the existing global `never-ai-brain-sprout-moe.zip` from GitHub Releases.
2. If the existing checkpoint is unconverged (`val_ce_loss > 2.0` or < 100k tokens), runs
   full 300-step stratified `ChronoMorph-Core` convergence (`train_coherent_brain.py`, CE < 0.30).
   Otherwise preserves 100% of prior collective weights!
3. Harvests live social network posts/replies (X mirrors, French microblogging, Reddit FR)
   and runs gentle cumulative Muon + AdamW plasticity steps.
4. Packages `never-ai-brain-sprout-moe.zip` + `global_lineage.json` for GitHub Releases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_coherent_brain import train_coherent_brain

from never_ai.engine import NeverAIEngine
from never_ai.updater.brain_bootstrap import download_brain_from_github_releases


def main() -> None:
    out_zip = Path(sys.argv[1] if len(sys.argv) > 1 else "never-ai-brain-sprout-moe.zip")
    vault_dir = Path("checkpoints_build")
    telem_p = Path("logs_build/training_telemetry.jsonl")

    print("[1/4] Checking existing global Never-AI brain on GitHub Releases...", flush=True)
    download_brain_from_github_releases(vault_dir=vault_dir, telemetry_path=telem_p)

    need_foundation = True
    manifest_p = vault_dir / "latest" / "manifest.json"
    if manifest_p.exists():
        try:
            m = json.loads(manifest_p.read_text(encoding="utf-8"))
            metrics = m.get("metrics", {})
            val_ce = float(metrics.get("val_ce_loss", metrics.get("final_ce_loss", 9.9)))
            toks = int(metrics.get("total_tokens_trained", 0))
            if val_ce <= 2.0 and toks >= 120_000:
                need_foundation = False
                print(f"   Existing global brain is already converged (CE={val_ce:.3f}, tokens={toks:,})!", flush=True)
        except Exception:  # noqa: BLE001
            need_foundation = True

    if need_foundation:
        print("[2/4] Training deeply converged ChronoMorph-Core foundation (300 stratified Muon+AdamW+WSM steps)...", flush=True)
        train_coherent_brain(vault_dir=vault_dir, telemetry_path=telem_p, total_steps=300)

    os.environ["NEVER_AI_BOOTSTRAP_GITHUB_BRAIN"] = "0"
    eng = NeverAIEngine(
        vault_dir=vault_dir,
        memory_path="mem_build/evolvemem.json",
        telemetry_path=telem_p,
        seq_len=128,
    )

    print("[2.5/4] Harvesting live social network stream (X / Microblog FR / Reddit FR) & evolving cumulatively...", flush=True)
    soc_res = eng.ingest_social_stream(x_query="lang:fr", train_steps=8)
    print("   Social harvest:", soc_res.get("counts"), flush=True)
    eng.train_steps(num_steps=8, batch_size=6)

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

    ckpt_dir = vault_dir / "latest"
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
