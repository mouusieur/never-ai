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

        # Store tensors in compact bfloat16 on disk (4x smaller than dual fp32!)
        source_sd = consolidated_state_dict if consolidated_state_dict is not None else model.state_dict()
        compact_tensors = {
            k: (
                v.detach().cpu().to(torch.bfloat16).contiguous()
                if v.is_floating_point()
                else v.detach().cpu().contiguous()
            )
            for k, v in source_sd.items()
        }
        target_name = "model_wsm.safetensors" if consolidated_state_dict is not None else "model_live.safetensors"
        tmp_file = ckpt_dir / f"{target_name}.tmp"
        final_file = ckpt_dir / target_name
        save_file(compact_tensors, str(tmp_file))
        os.replace(tmp_file, final_file)
        # Remove redundant twin file if present to keep workspace < 100 MB
        other_file = ckpt_dir / ("model_live.safetensors" if consolidated_state_dict is not None else "model_wsm.safetensors")
        if other_file.exists():
            other_file.unlink()

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
        weights_path = (
            wsm_path
            if (prefer_wsm and wsm_path.exists())
            else (live_path if live_path.exists() else wsm_path)
        )

        raw_state = load_file(str(weights_path), device="cpu")
        target_dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
        state = {
            k: (v.to(target_dtype) if v.is_floating_point() else v)
            for k, v in raw_state.items()
        }
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
