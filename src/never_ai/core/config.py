"""Configuration and evolutionary stage presets for ChronoMorph-LM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChronoMorphConfig:
    """Complete architectural specification for any morphogenesis stage of Never-AI."""

    stage_name: str = "Seed-0"
    generation: int = 0

    # Vocabulary & Sequence
    vocab_size: int = 4096
    max_seq_len: int = 512

    # Backbone Dimensions
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    d_head: int = 64

    # Multi-Head Latent Attention (MLA) & Decoupled RoPE
    d_kv_latent: int = 64
    d_rope: int = 32
    rope_theta: float = 10000.0
    yarn_scale: float = 1.0

    # Feed-Forward (ReLU^2) & Aux-Loss-Free DeepSeekMoE
    d_ff: int = 768
    use_moe: bool = False
    n_routed_experts: int = 0
    top_k_experts: int = 2
    moe_bias_lr: float = 1e-3

    # Google Titans Test-Time Neural Long-Term Memory
    use_titans_memory: bool = True
    d_mem: int = 64
    d_mem_hidden: int = 128
    titans_chunk_size: int = 32

    # Modded-NanoGPT Speedrun Accelerators
    use_unet_skips: bool = True
    logit_softcap: float = 15.0

    # Multi-Token Prediction (MTP) & Coconut Continuous Latent Reasoning
    use_mtp: bool = True
    mtp_loss_weight: float = 0.25

    # Numerical & Lineage Metadata
    rms_eps: float = 1e-6
    lineage_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChronoMorphConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def seed_0(cls, vocab_size: int = 4096) -> "ChronoMorphConfig":
        """Stage 0 (Embryon ~4.8M params): Fits 100% inside Xeon L3 cache for rapid birth."""
        return cls(
            stage_name="Seed-0",
            generation=0,
            vocab_size=vocab_size,
            max_seq_len=512,
            n_layers=4,
            d_model=256,
            n_heads=4,
            d_head=64,
            d_kv_latent=64,
            d_rope=32,
            d_ff=768,
            use_moe=False,
            use_titans_memory=True,
            d_mem=64,
            d_mem_hidden=128,
            titans_chunk_size=32,
            use_unet_skips=True,
            use_mtp=True,
        )

    @classmethod
    def seed_1(cls, vocab_size: int = 4096) -> "ChronoMorphConfig":
        """Stage 1 (Graine ~18.5M params): Grown depth & width, fits in Xeon L3 cache."""
        return cls(
            stage_name="Seed-1",
            generation=1,
            vocab_size=vocab_size,
            max_seq_len=1024,
            n_layers=8,
            d_model=384,
            n_heads=6,
            d_head=64,
            d_kv_latent=96,
            d_rope=32,
            d_ff=1152,
            use_moe=False,
            use_titans_memory=True,
            d_mem=64,
            d_mem_hidden=128,
            titans_chunk_size=32,
            use_unet_skips=True,
            use_mtp=True,
        )

    @classmethod
    def sprout_moe(cls, vocab_size: int = 8192) -> "ChronoMorphConfig":
        """Stage 2 (Jeune Pousse ~68M total / ~26M active): Sparse Aux-Loss-Free MoE."""
        return cls(
            stage_name="Sprout-MoE",
            generation=2,
            vocab_size=vocab_size,
            max_seq_len=2048,
            n_layers=12,
            d_model=512,
            n_heads=8,
            d_head=64,
            d_kv_latent=128,
            d_rope=32,
            d_ff=512,
            use_moe=True,
            n_routed_experts=8,
            top_k_experts=2,
            use_titans_memory=True,
            d_mem=96,
            d_mem_hidden=192,
            titans_chunk_size=64,
            use_unet_skips=True,
            use_mtp=True,
        )
