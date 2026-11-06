#!/usr/bin/env python3
"""Empirical hardware capability & precision dispatch probe for Never-AI."""

from __future__ import annotations

import json
import os
import platform
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_optimal_compute_profile() -> dict:
    has_cuda = torch.cuda.is_available()
    cpu_cap = torch.backends.cpu.get_cpu_capability()
    mkldnn = torch.backends.mkldnn.is_available()
    has_native_muon = hasattr(torch.optim, "Muon")

    if has_cuda:
        device = "cuda"
        recommended_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        device = "cpu"
        # Ice Lake AVX-512 executes native FP32 FMA faster than software-emulated BF16
        recommended_dtype = "float32"

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "device": device,
        "cpu_capability": cpu_cap,
        "cpu_count": os.cpu_count(),
        "mkldnn_available": mkldnn,
        "native_muon_available": has_native_muon,
        "recommended_dtype": recommended_dtype,
    }


def verify_gate_zero_invariance() -> dict:
    """Verify exact function preservation (max_diff == 0.0) of Gate-Zero & Symmetric FFN Mitosis."""
    torch.manual_seed(42)
    B, T, D, P = 2, 16, 64, 128
    x = torch.randn(B, T, D)

    # 1. Symmetric FFN ReLU^2 Mitosis: P -> 2P
    w_up = torch.randn(D, P) * 0.02
    w_down = torch.randn(P, D) * 0.02
    y_base = F.relu(x @ w_up).square() @ w_down

    delta = torch.randn_like(w_down) * 1e-3
    w_up_grown = torch.cat([w_up, w_up], dim=1)
    w_down_grown = 0.5 * torch.cat([w_down + delta, w_down - delta], dim=0)
    y_grown = F.relu(x @ w_up_grown).square() @ w_down_grown
    ffn_max_diff = (y_grown - y_base).abs().max().item()

    # 2. RMSNorm-Compensated Residual Width Expansion: D -> 2D
    D_new = D * 2
    eps = 1e-6
    gamma = torch.ones(D)
    rms_base = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma

    x_padded = F.pad(x, (0, D_new - D), value=0.0)
    gamma_grown = torch.cat([gamma * ((D / D_new) ** 0.5), torch.zeros(D_new - D)])
    # Adjusted eps scales cleanly by D/D_new for exact numerical equivalence
    rms_grown = (
        x_padded
        * torch.rsqrt(x_padded.pow(2).mean(dim=-1, keepdim=True) + eps * (D / D_new))
        * gamma_grown
    )
    rms_max_diff = (rms_grown[..., :D] - rms_base).abs().max().item()

    # 3. Interpositional Gate-Zero Depth Cloning
    alpha_new = nn.Parameter(torch.zeros(1))
    cloned_block_out = torch.randn_like(x)
    y_depth_grown = x + alpha_new * cloned_block_out
    depth_max_diff = (y_depth_grown - x).abs().max().item()

    return {
        "ffn_mitosis_max_diff": ffn_max_diff,
        "rms_width_expansion_max_diff": rms_max_diff,
        "gate_zero_depth_max_diff": depth_max_diff,
        "all_invariants_exact": max(ffn_max_diff, rms_max_diff, depth_max_diff) < 1e-6,
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    report = {
        "compute_profile": detect_optimal_compute_profile(),
        "morphogenesis_invariants": verify_gate_zero_invariance(),
        "probe_elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    print(json.dumps(report, indent=2))
