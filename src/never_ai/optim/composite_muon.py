"""CompositeMuon Optimizer + WSM (Warmup-Stable-Merge, ICLR 2026) Perpetual Scheduler."""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import torch
import torch.nn as nn


def partition_muon_adamw_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Partition parameters following Keller Jordan & Moonshot AI (arXiv:2502.16982):
    - 2D internal hidden matrices -> Muon (spectral steepest descent via Newton-Schulz)
    - Embeddings, LM Head, 1D norms, gates, biases, routers -> AdamW
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []

    excluded_substrings = ("tok_emb", "lm_head", "router", "gates_proj")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and not any(ex in name for ex in excluded_substrings):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


class CompositeMuonOptimizer:
    """
    Unified wrapper combining native PyTorch `torch.optim.Muon` on 2D hidden matrices
    with `torch.optim.AdamW` on embeddings, output head, norms, and scalar morphogenesis gates,
    governed by an eternal WSM (Warmup-Stable-Merge) constant-peak schedule!
    """

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 1.5e-2,
        adamw_lr: float = 3e-3,
        weight_decay: float = 0.02,
        warmup_steps: int = 50,
        grad_clip: float = 1.0,
    ) -> None:
        self.muon_base_lr = muon_lr
        self.adamw_base_lr = adamw_lr
        self.weight_decay = weight_decay
        self.warmup_steps = max(1, warmup_steps)
        self.grad_clip = grad_clip
        self.step_count = 0

        self.rebuild_for_model(model)

    def rebuild_for_model(
        self,
        model: nn.Module,
        preserved_muon_buffers: dict[str, torch.Tensor] | None = None,
    ) -> None:
        muon_params, adamw_params = partition_muon_adamw_parameters(model)
        d_model = float(getattr(getattr(model, "cfg", None), "d_model", 256))
        self.mup_scale = min(1.0, 256.0 / max(256.0, d_model))
        self._steps_since_rebuild = 0

        self.muon_opt = (
            torch.optim.Muon(
                muon_params,
                lr=self.muon_base_lr * self.mup_scale,
                weight_decay=self.weight_decay,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            )
            if muon_params
            else None
        )
        self.adamw_opt = (
            torch.optim.AdamW(
                adamw_params,
                lr=self.adamw_base_lr * self.mup_scale,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            )
            if adamw_params
            else None
        )

        if preserved_muon_buffers and self.muon_opt is not None:
            for name, param in model.named_parameters():
                if name in preserved_muon_buffers and param in self.muon_opt.param_groups[0]["params"]:
                    buf = preserved_muon_buffers[name]
                    if buf.shape == param.shape:
                        self.muon_opt.state[param]["momentum_buffer"] = buf.to(
                            device=param.device, dtype=param.dtype
                        )

    def current_lr_multiplier(self, lr_override_scale: float = 1.0) -> float:
        """Warmup / Post-Morphogenesis Micro-Warmup -> Eternal Stable Trunk (1.0 forever)."""
        if self.step_count < self.warmup_steps:
            base = float(self.step_count + 1) / float(self.warmup_steps)
        elif getattr(self, "_steps_since_rebuild", 999) < 15:
            base = float(self._steps_since_rebuild + 1) / 15.0
        else:
            base = 1.0
        return base * getattr(self, "mup_scale", 1.0) * lr_override_scale

    def zero_grad(self) -> None:
        if self.muon_opt is not None:
            self.muon_opt.zero_grad(set_to_none=True)
        if self.adamw_opt is not None:
            self.adamw_opt.zero_grad(set_to_none=True)

    def step(self, model: nn.Module, lr_override_scale: float = 1.0) -> dict[str, float]:
        mult = self.current_lr_multiplier(lr_override_scale=lr_override_scale)
        if self.muon_opt is not None:
            for g in self.muon_opt.param_groups:
                g["lr"] = self.muon_base_lr * mult
        if self.adamw_opt is not None:
            for g in self.adamw_opt.param_groups:
                g["lr"] = self.adamw_base_lr * mult

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip).item())

        if self.muon_opt is not None:
            self.muon_opt.step()
        if self.adamw_opt is not None:
            self.adamw_opt.step()

        self.step_count += 1
        self._steps_since_rebuild = getattr(self, "_steps_since_rebuild", 0) + 1
        return {
            "step": float(self.step_count),
            "lr_mult": mult,
            "muon_lr": self.muon_base_lr * mult,
            "adamw_lr": self.adamw_base_lr * mult,
            "grad_norm": grad_norm,
        }

    def export_muon_momentum_by_name(self, model: nn.Module) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.muon_opt is None:
            return out
        for name, param in model.named_parameters():
            st = self.muon_opt.state.get(param)
            if st and "momentum_buffer" in st:
                out[name] = st["momentum_buffer"].detach().clone()
        return out


class WSMCheckpointMerger:
    """
    ICLR 2026 Warmup-Stable-Merge (WSM) asynchronous checkpoint merger.
    Simulates optimal `1-sqrt` concave LR annealing via weighted averaging of recent
    constant-LR trunk snapshots without ever decaying or halting the live trunk!
    """

    def __init__(self, max_snapshots: int = 6) -> None:
        self.max_snapshots = max(2, max_snapshots)
        self.snapshots: deque[dict[str, torch.Tensor]] = deque(maxlen=self.max_snapshots)

    def clear(self) -> None:
        self.snapshots.clear()

    @torch.no_grad()
    def record_snapshot(self, model: nn.Module) -> None:
        snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Reset snapshot buffer automatically if model underwent morphogenesis shape change
        if self.snapshots:
            prev = self.snapshots[-1]
            if set(prev.keys()) != set(snap.keys()) or any(
                prev[k].shape != snap[k].shape for k in snap
            ):
                self.snapshots.clear()
        self.snapshots.append(snap)

    @staticmethod
    def compute_1_sqrt_weights(k_count: int) -> list[float]:
        """Concave `1 - sqrt((K - k) / K)` weights for k = 1..K (newest gets largest weight)."""
        if k_count <= 1:
            return [1.0]
        raw = [1.0 - math.sqrt(float(k_count - k) / float(k_count)) for k in range(1, k_count + 1)]
        s = sum(raw)
        return [w / s for w in raw]

    @torch.no_grad()
    def build_consolidated_state_dict(
        self, fallback_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.snapshots:
            if fallback_model is None:
                raise RuntimeError("No WSM snapshots recorded and no fallback model provided.")
            return {k: v.detach().cpu().clone() for k, v in fallback_model.state_dict().items()}

        n = len(self.snapshots)
        weights = self.compute_1_sqrt_weights(n)
        merged: dict[str, torch.Tensor] = {}

        keys = list(self.snapshots[-1].keys())
        for key in keys:
            t0 = self.snapshots[-1][key]
            if not t0.is_floating_point():
                merged[key] = t0.clone()
                continue
            acc = torch.zeros_like(t0, dtype=torch.float32)
            for w, snap in zip(weights, self.snapshots):
                acc.add_(snap[key].float(), alpha=w)
            merged[key] = acc.to(t0.dtype)
        return merged

    def describe(self) -> dict[str, Any]:
        n = len(self.snapshots)
        return {
            "num_snapshots": n,
            "max_snapshots": self.max_snapshots,
            "merge_curve": "1-sqrt (ICLR 2026 WSM)",
            "weights": [round(w, 4) for w in self.compute_1_sqrt_weights(n)] if n else [],
        }
