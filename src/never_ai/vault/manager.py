"""Atomic Safetensors Vault Manager & Cloud GPU Burst Bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.model import ChronoMorphLM
from never_ai.core.tokenizer import MorphTokenizer


class VaultManager:
    """
    Directive 4 & 5: Atomic safetensors checkpointing outside .git, tracking full
    morphogenetic lineage, WSM consolidated weights, and Cloud Vault manifests.
    """

    def __init__(self, vault_root: str | Path = "checkpoints") -> None:
        self.vault_root = Path(vault_root)
        self.vault_root.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        tag: str,
        model: ChronoMorphLM,
        tokenizer: MorphTokenizer,
        metrics: dict[str, Any] | None = None,
        consolidated_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> Path:
        ckpt_dir = self.vault_root / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save live trunk weights atomically
        live_tensors = {
            k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
        }
        tmp_live = ckpt_dir / "model_live.safetensors.tmp"
        final_live = ckpt_dir / "model_live.safetensors"
        save_file(live_tensors, str(tmp_live))
        os.replace(tmp_live, final_live)

        # 2. Save WSM 1-sqrt consolidated weights if provided
        if consolidated_state_dict is not None:
            cons_tensors = {
                k: v.detach().cpu().contiguous() for k, v in consolidated_state_dict.items()
            }
            tmp_cons = ckpt_dir / "model_wsm.safetensors.tmp"
            final_cons = ckpt_dir / "model_wsm.safetensors"
            save_file(cons_tensors, str(tmp_cons))
            os.replace(tmp_cons, final_cons)

        # 3. Save config, tokenizer & metadata
        model.cfg.save(ckpt_dir / "config.json")
        tokenizer.save(ckpt_dir / "tokenizer.json")

        manifest = {
            "tag": tag,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage_name": model.cfg.stage_name,
            "generation": model.cfg.generation,
            "parameter_counts": model.count_parameters(),
            "has_wsm_consolidated": consolidated_state_dict is not None,
            "metrics": metrics or {},
        }
        (ckpt_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return ckpt_dir

    def load_checkpoint(
        self,
        tag: str,
        prefer_wsm: bool = True,
        device: str | torch.device = "cpu",
    ) -> tuple[ChronoMorphLM, MorphTokenizer, dict[str, Any]]:
        ckpt_dir = self.vault_root / tag
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint '{tag}' not found in {self.vault_root}")

        cfg = ChronoMorphConfig.load(ckpt_dir / "config.json")
        tokenizer = MorphTokenizer.load(ckpt_dir / "tokenizer.json")
        model = ChronoMorphLM(cfg)

        wsm_path = ckpt_dir / "model_wsm.safetensors"
        live_path = ckpt_dir / "model_live.safetensors"
        weights_path = wsm_path if (prefer_wsm and wsm_path.exists()) else live_path

        state = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state)
        model.to(device)

        manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
        return model, tokenizer, manifest

    def list_checkpoints(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not self.vault_root.exists():
            return results
        for p in sorted(self.vault_root.iterdir()):
            mf = p / "manifest.json"
            if p.is_dir() and mf.exists():
                entry = json.loads(mf.read_text(encoding="utf-8"))
                entry["folder"] = p.name
                results.append(entry)
        return results
