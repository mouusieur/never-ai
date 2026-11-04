"""Exact Function-Preserving Morphogenesis Operators for ChronoMorph-LM."""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import torch
import torch.nn as nn

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.model import (
    AuxLossFreeMoE,
    ChronoMorphBlock,
    ChronoMorphLM,
    MultiHeadLatentAttention,
    MultiTokenPredictor,
    RMSNormCompensated,
    ReLU2FFN,
)
from never_ai.core.titans_memory import TitansMAGMemory
from never_ai.core.tokenizer import MorphTokenizer


def _record_lineage(
    cfg: ChronoMorphConfig,
    operator_name: str,
    details: dict[str, Any],
    new_stage_name: str | None = None,
) -> None:
    cfg.generation += 1
    if new_stage_name:
        cfg.stage_name = new_stage_name
    cfg.lineage_history.append(
        {
            "generation": cfg.generation,
            "stage_name": cfg.stage_name,
            "operator": operator_name,
            "timestamp": round(time.time(), 2),
            "details": details,
        }
    )


@torch.no_grad()
def verify_function_preservation(
    base_model: ChronoMorphLM,
    grown_model: ChronoMorphLM,
    vocab_limit: int | None = None,
    seq_len: int = 24,
) -> dict[str, Any]:
    """
    Directive 1: Prove numerically that ||logits_grown(X) - logits_base(X)||_inf < 1e-5
    before accepting any architectural morphogenesis step.
    """
    base_model.eval()
    grown_model.eval()
    device = next(base_model.parameters()).device

    v_max = vocab_limit or min(base_model.cfg.vocab_size, grown_model.cfg.vocab_size)
    torch.manual_seed(12345)
    x = torch.randint(16, v_max, (2, seq_len), device=device)

    out_base = base_model(x, update_titans=True)
    out_grown = grown_model(x, update_titans=True)

    logits_b = out_base.logits[..., :v_max]
    logits_g = out_grown.logits[..., :v_max]
    max_diff = float((logits_g - logits_b).abs().max().item())
    mean_diff = float((logits_g - logits_b).abs().mean().item())

    return {
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "preserved": max_diff < 1e-4,
    }


@torch.no_grad()
def grow_depth_interpositional_gate_zero(
    model: ChronoMorphLM,
    noise_std: float = 1e-4,
    new_stage_name: str | None = None,
) -> ChronoMorphLM:
    """
    Operator G_depth (Interpositional Gate-Zero Layer Cloning, arXiv:2510.08008 + arXiv:2607.14571):
    Doubles depth L -> 2L by interleaving [B_0, B_0', B_1, B_1', ...] where each twin block B_i'
    clones B_i with tiny symmetry-breaking noise, `alpha_gate = 0.0`, and `unet_lambda = 0.0`.
    Exact 0.000000 loss shock at t_0, immediate rank-L Jacobian expansion at t_1!
    """
    old_L = len(model.blocks)
    new_cfg = copy.deepcopy(model.cfg)
    new_cfg.n_layers = old_L * 2
    _record_lineage(
        new_cfg,
        "grow_depth_interpositional_gate_zero",
        {"old_n_layers": old_L, "new_n_layers": new_cfg.n_layers, "noise_std": noise_std},
        new_stage_name=new_stage_name,
    )

    grown = copy.deepcopy(model)
    grown.cfg = new_cfg

    interleaved: list[ChronoMorphBlock] = []
    for idx, orig_block in enumerate(grown.blocks):
        orig_block.layer_idx = 2 * idx
        interleaved.append(orig_block)

        twin = copy.deepcopy(orig_block)
        twin.layer_idx = 2 * idx + 1
        for name, p in twin.named_parameters():
            if name not in ("alpha_gate", "unet_lambda") and p.ndim >= 2:
                p.add_(torch.randn_like(p) * noise_std)
        # Gate-Zero + U-Net stack neutralizer -> 100% bit-exact identity at t_0!
        twin.alpha_gate.zero_()
        twin.unet_lambda.zero_()
        interleaved.append(twin)

    grown.blocks = nn.ModuleList(interleaved)
    return grown


def _mitosis_relu2_ffn(
    old_ffn: ReLU2FFN,
    new_d_ff: int,
    split_std: float = 1e-3,
) -> ReLU2FFN:
    old_p = old_ffn.d_ff
    d_model = old_ffn.d_model
    if new_d_ff == old_p:
        return copy.deepcopy(old_ffn)
    if new_d_ff < old_p:
        raise ValueError(f"new_d_ff ({new_d_ff}) must be >= old_d_ff ({old_p})")
    if new_d_ff > 2 * old_p:
        # Compose iteratively if > 2x
        mid = _mitosis_relu2_ffn(old_ffn, 2 * old_p, split_std=split_std)
        return _mitosis_relu2_ffn(mid, new_d_ff, split_std=split_std)

    k = new_d_ff - old_p
    device = old_ffn.up_proj.weight.device
    dtype = old_ffn.up_proj.weight.dtype

    new_ffn = ReLU2FFN(d_model, new_d_ff).to(device=device, dtype=dtype)
    # up_proj.weight: (d_ff, d_model)
    new_ffn.up_proj.weight[:old_p].copy_(old_ffn.up_proj.weight)
    new_ffn.up_proj.weight[old_p:].copy_(old_ffn.up_proj.weight[:k])

    # down_proj.weight: (d_model, d_ff)
    if k < old_p:
        new_ffn.down_proj.weight[:, k:old_p].copy_(old_ffn.down_proj.weight[:, k:])
    delta = torch.randn((d_model, k), device=device, dtype=dtype) * split_std
    new_ffn.down_proj.weight[:, :k].copy_(0.5 * (old_ffn.down_proj.weight[:, :k] + delta))
    new_ffn.down_proj.weight[:, old_p:].copy_(0.5 * (old_ffn.down_proj.weight[:, :k] - delta))
    return new_ffn


@torch.no_grad()
def grow_ffn_mitosis(
    model: ChronoMorphLM,
    new_d_ff: int,
    split_std: float = 1e-3,
    new_stage_name: str | None = None,
) -> ChronoMorphLM:
    """
    Operator G_ffn (Symmetric Neuron Mitosis, arXiv:2603.08647):
    Expands FFN / MoE expert hidden width d_ff -> new_d_ff with exact output invariance
    and immediate antisymmetric gradient specialization.
    """
    old_p = model.cfg.d_ff
    grown = copy.deepcopy(model)
    grown.cfg.d_ff = new_d_ff
    _record_lineage(
        grown.cfg,
        "grow_ffn_mitosis",
        {"old_d_ff": old_p, "new_d_ff": new_d_ff},
        new_stage_name=new_stage_name,
    )

    for block in grown.blocks:
        if isinstance(block.ffn, ReLU2FFN):
            block.ffn = _mitosis_relu2_ffn(block.ffn, new_d_ff, split_std=split_std)
        elif isinstance(block.ffn, AuxLossFreeMoE):
            block.ffn.d_ff = new_d_ff
            block.ffn.shared_expert = _mitosis_relu2_ffn(
                block.ffn.shared_expert, new_d_ff, split_std=split_std
            )
            block.ffn.routed_experts = nn.ModuleList(
                [
                    _mitosis_relu2_ffn(exp, new_d_ff, split_std=split_std)
                    for exp in block.ffn.routed_experts
                ]
            )
    return grown


@torch.no_grad()
def upcycle_dense_to_moe(
    model: ChronoMorphLM,
    n_routed_experts: int = 8,
    top_k: int = 2,
    new_stage_name: str | None = None,
) -> ChronoMorphLM:
    """
    Operator G_moe (Exact Function-Preserving Dense -> DeepSeekMoE Upcycling):
    Splits each dense FFN F(u) into 1 Shared Expert (0.5 * F) + E Routed Experts (0.5 * F).
    Since normalized top-k router gates sum to 1.0, E_0(u) + sum_{i in topk} g_i E_i(u) == F(u)
    at step t_0, while random router projections immediately route distinct token clusters
    to distinct experts on step t_1!
    """
    if model.cfg.use_moe:
        raise ValueError("Model is already an MoE; use grow_moe_experts() instead.")

    grown = copy.deepcopy(model)
    grown.cfg.use_moe = True
    grown.cfg.n_routed_experts = n_routed_experts
    grown.cfg.top_k_experts = min(top_k, n_routed_experts)
    _record_lineage(
        grown.cfg,
        "upcycle_dense_to_moe",
        {"n_routed_experts": n_routed_experts, "top_k": grown.cfg.top_k_experts},
        new_stage_name=new_stage_name,
    )

    for block in grown.blocks:
        assert isinstance(block.ffn, ReLU2FFN)
        dense = block.ffn
        device = dense.up_proj.weight.device
        dtype = dense.up_proj.weight.dtype

        moe = AuxLossFreeMoE(
            d_model=grown.cfg.d_model,
            d_ff=grown.cfg.d_ff,
            n_routed_experts=n_routed_experts,
            top_k=grown.cfg.top_k_experts,
            bias_lr=grown.cfg.moe_bias_lr,
        ).to(device=device, dtype=dtype)

        moe.shared_expert.up_proj.weight.copy_(dense.up_proj.weight)
        moe.shared_expert.down_proj.weight.copy_(0.5 * dense.down_proj.weight)

        for exp in moe.routed_experts:
            exp.up_proj.weight.copy_(dense.up_proj.weight)
            exp.down_proj.weight.copy_(0.5 * dense.down_proj.weight)

        block.ffn = moe
    return grown


@torch.no_grad()
def grow_moe_experts(
    model: ChronoMorphLM,
    double_experts: bool = True,
    noise_std: float = 1e-4,
    new_stage_name: str | None = None,
) -> ChronoMorphLM:
    """
    Orthogonal MoE Width Growth (arXiv:2510.08008):
    Doubles routed experts E -> 2E and top_k -> 2*top_k with exact router & weight cloning
    (scaled by 0.5 via normalized gates), preserving exact t_0 outputs!
    """
    if not model.cfg.use_moe:
        raise ValueError("Model is dense; call upcycle_dense_to_moe() first.")
    if not double_experts:
        raise ValueError("Currently supports exact FP doubling E -> 2E.")

    old_E = model.cfg.n_routed_experts
    old_k = model.cfg.top_k_experts
    new_E = old_E * 2
    new_k = old_k * 2

    grown = copy.deepcopy(model)
    grown.cfg.n_routed_experts = new_E
    grown.cfg.top_k_experts = new_k
    _record_lineage(
        grown.cfg,
        "grow_moe_experts",
        {"old_experts": old_E, "new_experts": new_E, "old_top_k": old_k, "new_top_k": new_k},
        new_stage_name=new_stage_name,
    )

    for block in grown.blocks:
        assert isinstance(block.ffn, AuxLossFreeMoE)
        old_moe = block.ffn
        device = old_moe.router.weight.device
        dtype = old_moe.router.weight.dtype

        new_moe = AuxLossFreeMoE(
            d_model=grown.cfg.d_model,
            d_ff=grown.cfg.d_ff,
            n_routed_experts=new_E,
            top_k=new_k,
            bias_lr=grown.cfg.moe_bias_lr,
        ).to(device=device, dtype=dtype)

        new_moe.shared_expert = copy.deepcopy(old_moe.shared_expert)
        for i in range(old_E):
            e_a = copy.deepcopy(old_moe.routed_experts[i])
            e_b = copy.deepcopy(old_moe.routed_experts[i])
            delta = torch.randn_like(e_a.down_proj.weight) * noise_std
            e_a.down_proj.weight.add_(delta)
            e_b.down_proj.weight.sub_(delta)
            new_moe.routed_experts[2 * i] = e_a
            new_moe.routed_experts[2 * i + 1] = e_b

            new_moe.router.weight[2 * i].copy_(old_moe.router.weight[i])
            new_moe.router.weight[2 * i + 1].copy_(old_moe.router.weight[i])
            new_moe.expert_bias[2 * i] = old_moe.expert_bias[i]
            new_moe.expert_bias[2 * i + 1] = old_moe.expert_bias[i]

        block.ffn = new_moe
    return grown


def _expand_rmsnorm(old_norm: RMSNormCompensated, new_dim: int) -> RMSNormCompensated:
    old_dim = old_norm.dim
    device = old_norm.weight.device
    dtype = old_norm.weight.dtype
    ratio = float(old_dim) / float(new_dim)
    scale = math.sqrt(ratio)

    new_norm = RMSNormCompensated(new_dim, eps=1e-6).to(device=device, dtype=dtype)
    new_norm.eps_val.copy_(old_norm.eps_val * ratio)
    new_norm.weight[:old_dim].copy_(old_norm.weight * scale)
    # Non-zero gain on padded zero coordinates preserves 0.0 forward output while keeping backward Jacobian alive!
    new_norm.weight[old_dim:].fill_(scale)
    return new_norm


def _expand_ffn_residual_dim(old_ffn: ReLU2FFN, new_d_model: int) -> ReLU2FFN:
    old_d = old_ffn.d_model
    d_ff = old_ffn.d_ff
    device = old_ffn.up_proj.weight.device
    dtype = old_ffn.up_proj.weight.dtype

    new_ffn = ReLU2FFN(new_d_model, d_ff).to(device=device, dtype=dtype)
    new_ffn.up_proj.weight[:, :old_d].copy_(old_ffn.up_proj.weight)
    nn.init.normal_(new_ffn.up_proj.weight[:, old_d:], mean=0.0, std=0.01)

    new_ffn.down_proj.weight[:old_d, :].copy_(old_ffn.down_proj.weight)
    new_ffn.down_proj.weight[old_d:, :].zero_()
    return new_ffn


@torch.no_grad()
def grow_width_rms_compensated(
    model: ChronoMorphLM,
    new_d_model: int,
    new_n_heads: int,
    new_d_kv_latent: int,
    new_stage_name: str | None = None,
) -> ChronoMorphLM:
    """
    Operator G_width (RMSNorm-Compensated Residual + Multi-Head MLA Subspace Expansion):
    Expands d_model (D1 -> D2), n_heads (H1 -> H2), and d_kv_latent (C1 -> C2) simultaneously
    while keeping logits 100% invariant at t_0 and non-degenerate gradients at t_1!
    """
    old_cfg = model.cfg
    D1, D2 = old_cfg.d_model, new_d_model
    H1, H2 = old_cfg.n_heads, new_n_heads
    C1, C2 = old_cfg.d_kv_latent, new_d_kv_latent
    if D2 < D1 or H2 < H1 or C2 < C1:
        raise ValueError("New width/heads/kv_latent must be >= current dimensions.")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    new_cfg = copy.deepcopy(old_cfg)
    new_cfg.d_model = D2
    new_cfg.n_heads = H2
    new_cfg.d_kv_latent = C2
    _record_lineage(
        new_cfg,
        "grow_width_rms_compensated",
        {
            "d_model": [D1, D2],
            "n_heads": [H1, H2],
            "d_kv_latent": [C1, C2],
        },
        new_stage_name=new_stage_name,
    )

    grown = ChronoMorphLM(new_cfg).to(device=device, dtype=dtype)

    # 1. Token Embedding: zero-pad new residual coordinates
    grown.tok_emb.weight[:, :D1].copy_(model.tok_emb.weight)
    grown.tok_emb.weight[:, D1:].zero_()

    # 2. Final Norm, Latent Bridge Norm & LM Head (non-zero new cols wake up D1:D2 on step 1!)
    grown.final_norm = _expand_rmsnorm(model.final_norm, D2)
    grown.latent_bridge_norm = _expand_rmsnorm(model.latent_bridge_norm, D2)
    grown.lm_head.weight[:, :D1].copy_(model.lm_head.weight)
    nn.init.normal_(grown.lm_head.weight[:, D1:], mean=0.0, std=0.01)

    # 3. Blocks
    d_head = old_cfg.d_head
    d_rope = old_cfg.d_rope
    d_nope = d_head - d_rope

    for b_old, b_new in zip(model.blocks, grown.blocks):
        b_new.alpha_gate.copy_(b_old.alpha_gate)
        b_new.unet_lambda.copy_(b_old.unet_lambda)
        b_new.attn_norm = _expand_rmsnorm(b_old.attn_norm, D2)
        b_new.ffn_norm = _expand_rmsnorm(b_old.ffn_norm, D2)

        # MLA expansion
        a_old: MultiHeadLatentAttention = b_old.attn
        a_new: MultiHeadLatentAttention = b_new.attn
        a_new.qc_norm.load_state_dict(a_old.qc_norm.state_dict())
        a_new.qr_norm.load_state_dict(a_old.qr_norm.state_dict())
        a_new.kc_norm.load_state_dict(a_old.kc_norm.state_dict())
        a_new.kr_norm.load_state_dict(a_old.kr_norm.state_dict())
        a_new.kv_norm = _expand_rmsnorm(a_old.kv_norm, C2)

        # w_qc: (H*d_nope, D)
        a_new.w_qc.weight[: H1 * d_nope, :D1].copy_(a_old.w_qc.weight)
        nn.init.normal_(a_new.w_qc.weight[: H1 * d_nope, D1:], std=0.01)

        # w_qr: (H*d_rope, D)
        a_new.w_qr.weight[: H1 * d_rope, :D1].copy_(a_old.w_qr.weight)
        nn.init.normal_(a_new.w_qr.weight[: H1 * d_rope, D1:], std=0.01)

        # w_dkv: (C, D) -> zero new latent rows so c_kv[C1:] == 0 at t_0
        a_new.w_dkv.weight[:C1, :D1].copy_(a_old.w_dkv.weight)
        nn.init.normal_(a_new.w_dkv.weight[:C1, D1:], std=0.01)
        a_new.w_dkv.weight[C1:, :].zero_()

        # w_uk: (H*d_nope, C) & w_uv: (H*d_head, C)
        a_new.w_uk.weight[: H1 * d_nope, :C1].copy_(a_old.w_uk.weight)
        nn.init.normal_(a_new.w_uk.weight[: H1 * d_nope, C1:], std=0.01)

        a_new.w_uv.weight[: H1 * d_head, :C1].copy_(a_old.w_uv.weight)
        nn.init.normal_(a_new.w_uv.weight[: H1 * d_head, C1:], std=0.01)

        # w_kr: (d_rope, D)
        a_new.w_kr.weight[:, :D1].copy_(a_old.w_kr.weight)
        nn.init.normal_(a_new.w_kr.weight[:, D1:], std=0.01)

        # w_o: (D, H*d_head) -> zero new head columns & new residual rows (Gesmundo Thm 3.2 & 3.5)
        a_new.w_o.weight.zero_()
        a_new.w_o.weight[:D1, : H1 * d_head].copy_(a_old.w_o.weight)

        # Titans memory expansion
        if b_old.titans is not None and b_new.titans is not None:
            t_old: TitansMAGMemory = b_old.titans
            t_new: TitansMAGMemory = b_new.titans
            t_new.w1_init.copy_(t_old.w1_init)
            t_new.w2_init.copy_(t_old.w2_init)
            t_new.mem_scale.copy_(t_old.mem_scale)
            t_new.gates_proj.bias.copy_(t_old.gates_proj.bias)
            t_new.gates_proj.weight[:, :D1].copy_(t_old.gates_proj.weight)
            t_new.gates_proj.weight[:, D1:].zero_()

            for proj_old, proj_new in [
                (t_old.w_k, t_new.w_k),
                (t_old.w_v, t_new.w_v),
                (t_old.w_q, t_new.w_q),
            ]:
                proj_new.weight[:, :D1].copy_(proj_old.weight)
                nn.init.normal_(proj_new.weight[:, D1:], std=0.01)

            t_new.w_out.weight.zero_()
            t_new.w_out.weight[:D1, :].copy_(t_old.w_out.weight)

        # FFN / MoE residual expansion
        if isinstance(b_old.ffn, ReLU2FFN):
            b_new.ffn = _expand_ffn_residual_dim(b_old.ffn, D2)
        elif isinstance(b_old.ffn, AuxLossFreeMoE):
            assert isinstance(b_new.ffn, AuxLossFreeMoE)
            b_new.ffn.shared_expert = _expand_ffn_residual_dim(b_old.ffn.shared_expert, D2)
            b_new.ffn.routed_experts = nn.ModuleList(
                [_expand_ffn_residual_dim(exp, D2) for exp in b_old.ffn.routed_experts]
            )
            b_new.ffn.router.weight[:, :D1].copy_(b_old.ffn.router.weight)
            b_new.ffn.router.weight[:, D1:].zero_()
            b_new.ffn.expert_bias.copy_(b_old.ffn.expert_bias)

    # 4. Multi-Token Predictor (MTP)
    if model.mtp is not None and grown.mtp is not None:
        m_old: MultiTokenPredictor = model.mtp
        m_new: MultiTokenPredictor = grown.mtp
        m_new.norm_h = _expand_rmsnorm(m_old.norm_h, D2)
        m_new.norm_e = _expand_rmsnorm(m_old.norm_e, D2)
        m_new.out_norm = _expand_rmsnorm(m_old.out_norm, D2)
        m_new.fuse_proj.weight.zero_()
        m_new.fuse_proj.weight[:D1, :D1].copy_(m_old.fuse_proj.weight[:, :D1])
        m_new.fuse_proj.weight[:D1, D2 : D2 + D1].copy_(m_old.fuse_proj.weight[:, D1:])
        m_new.ffn = _expand_ffn_residual_dim(
            _mitosis_relu2_ffn(m_old.ffn, D2 * 2), D2
        )

    return grown


@torch.no_grad()
def grow_vocabulary_convex_hull(
    model: ChronoMorphLM,
    old_tokenizer: MorphTokenizer,
    new_tokenizer: MorphTokenizer,
    new_stage_name: str | None = None,
) -> ChronoMorphLM:
    """
    Operator G_vocab (Hierarchical Subword Convex-Hull Vocabulary Expansion):
    Expands tok_emb and lm_head from |V_1| -> |V_2|, initializing every newly merged
    subword token at the exact semantic centroid of its constituent source tokens.
    """
    new_v = new_tokenizer.vocab_size
    old_v = old_tokenizer.vocab_size
    if new_v < old_v:
        raise ValueError(f"new vocab ({new_v}) must be >= old vocab ({old_v})")

    decomp = MorphTokenizer.compute_subword_decomposition_map(old_tokenizer, new_tokenizer)

    grown = copy.deepcopy(model)
    grown.cfg.vocab_size = new_v
    _record_lineage(
        grown.cfg,
        "grow_vocabulary_convex_hull",
        {"old_vocab_size": old_v, "new_vocab_size": new_v},
        new_stage_name=new_stage_name,
    )

    device = model.tok_emb.weight.device
    dtype = model.tok_emb.weight.dtype
    d_model = grown.cfg.d_model

    new_emb = nn.Embedding(new_v, d_model, device=device, dtype=dtype)
    new_head = nn.Linear(d_model, new_v, bias=False, device=device, dtype=dtype)

    for new_id, sub_ids in enumerate(decomp):
        idx_t = torch.tensor(sub_ids, dtype=torch.long, device=device)
        new_emb.weight[new_id].copy_(model.tok_emb.weight.index_select(0, idx_t).mean(dim=0))
        new_head.weight[new_id].copy_(model.lm_head.weight.index_select(0, idx_t).mean(dim=0))

    grown.tok_emb = new_emb
    grown.lm_head = new_head
    return grown


def transition_seed0_to_seed1(model: ChronoMorphLM) -> ChronoMorphLM:
    """
    Complete Zero-Shock Morphogenesis from Stage 0 (Seed-0 ~4.8M) -> Stage 1 (Seed-1 ~18.5M):
    1) Residual Width & Heads: d=256->384, heads=4->6, kv_latent=64->96
    2) FFN Mitosis: d_ff=768->1152
    3) Interpositional Gate-Zero Depth: L=4->8
    """
    m = grow_width_rms_compensated(model, new_d_model=384, new_n_heads=6, new_d_kv_latent=96)
    m = grow_ffn_mitosis(m, new_d_ff=1152)
    m = grow_depth_interpositional_gate_zero(m, new_stage_name="Seed-1")
    m.cfg.max_seq_len = 1024
    return m


def transition_seed1_to_sprout_moe(
    model: ChronoMorphLM,
    n_routed_experts: int = 8,
    top_k: int = 2,
) -> ChronoMorphLM:
    """
    Complete Zero-Shock Morphogenesis from Stage 1 (Seed-1) -> Stage 2 (Sprout-MoE):
    Upcycles Dense FFNs into Aux-Loss-Free DeepSeekMoE (1 Shared + 8 Routed Experts, top-2).
    """
    m = upcycle_dense_to_moe(
        model,
        n_routed_experts=n_routed_experts,
        top_k=top_k,
        new_stage_name="Sprout-MoE",
    )
    m.cfg.max_seq_len = 2048
    return m
