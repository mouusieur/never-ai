"""ChronoMorph-LM: Self-Growing Modded-MLA-MoE + Titans + Coconut Architecture."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.special_tokens import (
    CALL_END_ID,
    CALL_START_ID,
    LATENT_BOT_ID,
    LATENT_EOT_ID,
    PAD_ID,
    THINK_END_ID,
    THINK_START_ID,
)
from never_ai.core.titans_memory import TitansMAGMemory, TitansMemoryState


class RMSNormCompensated(nn.Module):
    """
    RMSNorm with explicit subspace epsilon scaling so zero-padded residual width
    expansion (D1 -> D2) preserves exact mathematical equivalence to machine precision.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("eps_val", torch.tensor(float(eps), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps_val)
        return (x_fp32 * rms).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding with dynamic YaRN context extension support."""

    def __init__(self, dim: int, base: float = 10000.0, yarn_scale: float = 1.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.base = base
        self.yarn_scale = yarn_scale
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needed = seq_len + offset
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < needed
            or self._cos_cached.device != device
        ):
            self._seq_len_cached = max(needed, 512)
            t = torch.arange(self._seq_len_cached, device=device, dtype=torch.float32)
            if self.yarn_scale > 1.0:
                t = t / self.yarn_scale
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            mscale = 1.0 if self.yarn_scale <= 1.0 else (0.1 * math.log(self.yarn_scale) + 1.0)
            self._cos_cached = (emb.cos() * mscale)
            self._sin_cached = (emb.sin() * mscale)

        cos = self._cos_cached[offset : offset + seq_len].to(dtype)
        sin = self._sin_cached[offset : offset + seq_len].to(dtype)
        return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to tensor of shape (B, T, H, D_rope)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    return x * cos_b + rotated * sin_b


@dataclass
class MLAKVCache:
    """Compressed Multi-Head Latent Attention KV Cache storing only (c_kv, k_rope)."""

    c_kv: torch.Tensor    # (B, T_past, d_kv_latent)
    k_rope: torch.Tensor  # (B, T_past, 1, d_rope)

    @property
    def seq_len(self) -> int:
        return int(self.c_kv.shape[1])


class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2/V3 Multi-Head Latent Attention (MLA) with QK-RMSNorm & Decoupled RoPE.
    Splits head_dim = d_nope + d_rope so D_qk == D_v == d_head (power-of-2 SDPA optimal)
    while caching ONLY the compressed latent vector (c_kv, k_rope).
    """

    def __init__(self, cfg: ChronoMorphConfig) -> None:
        super().__init__()
        if cfg.d_rope >= cfg.d_head:
            raise ValueError(f"d_rope ({cfg.d_rope}) must be < d_head ({cfg.d_head})")
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_rope = cfg.d_rope
        self.d_nope = cfg.d_head - cfg.d_rope
        self.d_kv_latent = cfg.d_kv_latent

        # Query projections (content + decoupled RoPE)
        self.w_qc = nn.Linear(cfg.d_model, cfg.n_heads * self.d_nope, bias=False)
        self.qc_norm = RMSNormCompensated(self.d_nope, cfg.rms_eps)

        self.w_qr = nn.Linear(cfg.d_model, cfg.n_heads * self.d_rope, bias=False)
        self.qr_norm = RMSNormCompensated(self.d_rope, cfg.rms_eps)

        # Joint Low-Rank KV Compression
        self.w_dkv = nn.Linear(cfg.d_model, cfg.d_kv_latent, bias=False)
        self.kv_norm = RMSNormCompensated(cfg.d_kv_latent, cfg.rms_eps)
        self.w_uk = nn.Linear(cfg.d_kv_latent, cfg.n_heads * self.d_nope, bias=False)
        self.kc_norm = RMSNormCompensated(self.d_nope, cfg.rms_eps)
        self.w_uv = nn.Linear(cfg.d_kv_latent, cfg.n_heads * cfg.d_head, bias=False)

        # Shared Decoupled RoPE Key
        self.w_kr = nn.Linear(cfg.d_model, self.d_rope, bias=False)
        self.kr_norm = RMSNormCompensated(self.d_rope, cfg.rms_eps)

        # Output projection (muP / Modded-NanoGPT friendly init)
        self.w_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        nn.init.normal_(self.w_o.weight, mean=0.0, std=0.02 / math.sqrt(2 * max(1, cfg.n_layers)))

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: MLAKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, MLAKVCache | None]:
        B, T, _ = x.shape
        offset = kv_cache.seq_len if kv_cache is not None else 0
        cos, sin = rope.get_cos_sin(T, x.device, x.dtype, offset=offset)

        q_c = self.qc_norm(self.w_qc(x).view(B, T, self.n_heads, self.d_nope))
        q_r = apply_rotary_emb(
            self.qr_norm(self.w_qr(x).view(B, T, self.n_heads, self.d_rope)),
            cos,
            sin,
        )
        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)  # (B, H, T, d_head)

        c_kv_new = self.kv_norm(self.w_dkv(x))  # (B, T, d_kv_latent)
        k_r_new = apply_rotary_emb(
            self.kr_norm(self.w_kr(x).view(B, T, 1, self.d_rope)),
            cos,
            sin,
        )

        if kv_cache is not None:
            c_kv = torch.cat([kv_cache.c_kv, c_kv_new], dim=1)
            k_r_shared = torch.cat([kv_cache.k_rope, k_r_new], dim=1)
        else:
            c_kv = c_kv_new
            k_r_shared = k_r_new

        S = c_kv.shape[1]
        k_c = self.kc_norm(self.w_uk(c_kv).view(B, S, self.n_heads, self.d_nope))
        k_r = k_r_shared.expand(B, S, self.n_heads, self.d_rope)
        k = torch.cat([k_c, k_r], dim=-1).transpose(1, 2)  # (B, H, S, d_head)
        v = self.w_uv(c_kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        is_causal = (kv_cache is None) and (T > 1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        out = self.w_o(attn_out)

        next_cache = MLAKVCache(c_kv=c_kv, k_rope=k_r_shared) if use_cache else None
        return out, next_cache


class ReLU2FFN(nn.Module):
    """Squared ReLU Feed-Forward Network (~80% natural activation sparsity, 2 matrices)."""

    def __init__(self, d_model: int, d_ff: int, init_scale: float = 0.02) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).square())


class AuxLossFreeMoE(nn.Module):
    """
    DeepSeek-V3 style Mixture-of-Experts:
    - 1 always-active Shared Expert capturing foundational language & reasoning
    - E fine-grained Routed Experts with Aux-Loss-Free dynamic bias load balancing!
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_routed_experts: int = 8,
        top_k: int = 2,
        bias_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_routed_experts = n_routed_experts
        self.top_k = min(top_k, n_routed_experts)
        self.bias_lr = bias_lr

        self.shared_expert = ReLU2FFN(d_model, d_ff)
        self.routed_experts = nn.ModuleList(
            [ReLU2FFN(d_model, d_ff) for _ in range(n_routed_experts)]
        )
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)

        # Out-of-gradient dynamic load-balancing bias & utilization telemetry
        self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
        self.register_buffer(
            "last_expert_fractions",
            torch.full((n_routed_experts,), 1.0 / max(1, n_routed_experts)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat_x = x.view(-1, D)  # (N, D)
        N = flat_x.shape[0]

        shared_out = self.shared_expert(flat_x)

        scores = torch.sigmoid(self.router(flat_x))  # (N, E)
        biased_scores = scores + self.expert_bias.unsqueeze(0)
        _, topk_idx = torch.topk(biased_scores, k=self.top_k, dim=-1)  # (N, k)

        # Gather unbiased scores for differentiable gating
        topk_gates = torch.gather(scores, dim=-1, index=topk_idx)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-6)

        routed_out = torch.zeros_like(flat_x)
        counts = torch.zeros(self.n_routed_experts, device=x.device, dtype=torch.float32)

        for e_idx, expert in enumerate(self.routed_experts):
            mask = topk_idx == e_idx  # (N, k)
            if not mask.any():
                continue
            token_indices, slot_indices = torch.where(mask)
            counts[e_idx] = float(token_indices.numel())
            expert_in = flat_x.index_select(0, token_indices)
            expert_y = expert(expert_in)
            weights = topk_gates[token_indices, slot_indices].unsqueeze(-1)
            routed_out.index_add_(0, token_indices, (expert_y * weights).to(routed_out.dtype))

        if N > 0:
            with torch.no_grad():
                fracs = counts / float(N * self.top_k)
                self.last_expert_fractions.copy_(fracs * self.top_k)

        return (shared_out + routed_out).view(B, T, D)

    @torch.no_grad()
    def update_load_balance_bias(self) -> None:
        """Dynamic Aux-Loss-Free bias adjustment (DeepSeek-V3): zero gradient interference."""
        target = float(self.top_k) / float(max(1, self.n_routed_experts))
        delta = torch.sign(target - self.last_expert_fractions)
        self.expert_bias.add_(self.bias_lr * delta)
        self.expert_bias.clamp_(-2.0, 2.0)


class ChronoMorphBlock(nn.Module):
    """
    Single Morphogenetic Layer featuring:
    - Gate-Zero learnable block gate `alpha_gate` (1.0 native, 0.0 on interpositionally grown blocks)
    - U-Net symmetric residual skip scalar `unet_lambda`
    - MLA Attention + optional Titans Test-Time Neural Long-Term Memory
    - ReLU^2 FFN or Aux-Loss-Free DeepSeekMoE
    """

    def __init__(self, cfg: ChronoMorphConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNormCompensated(cfg.d_model, cfg.rms_eps)
        self.attn = MultiHeadLatentAttention(cfg)
        self.titans: TitansMAGMemory | None = (
            TitansMAGMemory(
                d_model=cfg.d_model,
                d_mem=cfg.d_mem,
                d_mem_hidden=cfg.d_mem_hidden,
                chunk_size=cfg.titans_chunk_size,
            )
            if cfg.use_titans_memory
            else None
        )

        self.ffn_norm = RMSNormCompensated(cfg.d_model, cfg.rms_eps)
        self.ffn: nn.Module
        if cfg.use_moe and cfg.n_routed_experts > 0:
            self.ffn = AuxLossFreeMoE(
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                n_routed_experts=cfg.n_routed_experts,
                top_k=cfg.top_k_experts,
                bias_lr=cfg.moe_bias_lr,
            )
        else:
            init_scale = 0.02 / math.sqrt(2 * max(1, cfg.n_layers))
            self.ffn = ReLU2FFN(cfg.d_model, cfg.d_ff, init_scale=init_scale)

        # Gate-Zero scalar: 1.0 on original blocks, initialized to 0.0 on newly cloned blocks!
        self.alpha_gate = nn.Parameter(torch.ones(1))
        self.unet_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        skip_in: torch.Tensor | None = None,
        kv_cache: MLAKVCache | None = None,
        titans_state: TitansMemoryState | None = None,
        use_cache: bool = False,
        update_titans: bool = True,
    ) -> tuple[torch.Tensor, MLAKVCache | None, TitansMemoryState | None]:
        if skip_in is not None:
            x = x + self.unet_lambda * skip_in

        normed_x = self.attn_norm(x)
        attn_out, next_kv = self.attn(normed_x, rope=rope, kv_cache=kv_cache, use_cache=use_cache)

        next_titans: TitansMemoryState | None = None
        if self.titans is not None:
            mem_out, next_titans = self.titans(
                normed_x, state=titans_state, update_memory=update_titans
            )
            attn_out = attn_out + mem_out

        h = x + self.alpha_gate * attn_out
        out = h + self.alpha_gate * self.ffn(self.ffn_norm(h))
        return out, next_kv, next_titans


class MultiTokenPredictor(nn.Module):
    """DeepSeek-V3 inspired Multi-Token Prediction (MTP) head predicting token t+2."""

    def __init__(self, cfg: ChronoMorphConfig) -> None:
        super().__init__()
        self.norm_h = RMSNormCompensated(cfg.d_model, cfg.rms_eps)
        self.norm_e = RMSNormCompensated(cfg.d_model, cfg.rms_eps)
        self.fuse_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)
        self.ffn = ReLU2FFN(cfg.d_model, cfg.d_model * 2)
        self.out_norm = RMSNormCompensated(cfg.d_model, cfg.rms_eps)

    def forward(self, h_last: torch.Tensor, next_tok_emb: torch.Tensor) -> torch.Tensor:
        fused = self.fuse_proj(
            torch.cat([self.norm_h(h_last), self.norm_e(next_tok_emb)], dim=-1)
        )
        return self.out_norm(fused + self.ffn(fused))


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    ce_loss: float
    mtp_loss: float
    hidden_states: torch.Tensor
    kv_caches: list[MLAKVCache] | None
    titans_states: list[TitansMemoryState | None]


class ChronoMorphLM(nn.Module):
    """
    Never-AI Sovereign Morphogenetic Language Model.
    Supports:
    - Exact Function-Preserving Morphogenesis across Depth, Width, FFN, MoE & Vocabulary
    - Online Test-Time Memorization (Titans)
    - Explicit Open Reasoning (<|think|>...<|/think|>) & Continuous Latent Thought (Coconut)
    - Multi-Token Prediction (MTP) & Masked Tool/Plugin Supervision
    """

    def __init__(self, cfg: ChronoMorphConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.rope = RotaryEmbedding(
            dim=cfg.d_rope,
            base=cfg.rope_theta,
            yarn_scale=cfg.yarn_scale,
        )
        self.blocks = nn.ModuleList([ChronoMorphBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNormCompensated(cfg.d_model, cfg.rms_eps)

        # Untied LM Head (Modded-NanoGPT & Vocabulary Morphogenesis friendly)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.01)

        # Continuous Latent Thought (Coconut) bridge norm
        self.latent_bridge_norm = RMSNormCompensated(cfg.d_model, cfg.rms_eps)

        self.mtp: MultiTokenPredictor | None = (
            MultiTokenPredictor(cfg) if cfg.use_mtp else None
        )

    def _softcap_logits(self, raw_logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.logit_softcap <= 0:
            return raw_logits
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(raw_logits / cap)

    def forward_from_embeddings(
        self,
        h: torch.Tensor,
        kv_caches: list[MLAKVCache] | None = None,
        titans_states: list[TitansMemoryState | None] | None = None,
        use_cache: bool = False,
        update_titans: bool = True,
    ) -> tuple[torch.Tensor, list[MLAKVCache] | None, list[TitansMemoryState | None]]:
        n_blocks = len(self.blocks)
        half = n_blocks // 2
        skip_stack: list[torch.Tensor] = []

        next_kvs: list[MLAKVCache] = []
        next_titans: list[TitansMemoryState | None] = []

        for i, block in enumerate(self.blocks):
            skip_in = None
            if self.cfg.use_unet_skips and i >= (n_blocks - half) and skip_stack:
                skip_in = skip_stack.pop()

            layer_kv = kv_caches[i] if kv_caches is not None else None
            layer_tm = titans_states[i] if titans_states is not None else None

            h, n_kv, n_tm = block(
                h,
                rope=self.rope,
                skip_in=skip_in,
                kv_cache=layer_kv,
                titans_state=layer_tm,
                use_cache=use_cache,
                update_titans=update_titans,
            )

            if self.cfg.use_unet_skips and i < half:
                skip_stack.append(h)

            if use_cache and n_kv is not None:
                next_kvs.append(n_kv)
            next_titans.append(n_tm)

        h_norm = self.final_norm(h)
        return h_norm, (next_kvs if use_cache else None), next_titans

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        kv_caches: list[MLAKVCache] | None = None,
        titans_states: list[TitansMemoryState | None] | None = None,
        use_cache: bool = False,
        update_titans: bool = True,
    ) -> ForwardOutput:
        h0 = self.tok_emb(input_ids)
        h_final, next_kvs, next_titans = self.forward_from_embeddings(
            h0,
            kv_caches=kv_caches,
            titans_states=titans_states,
            use_cache=use_cache,
            update_titans=update_titans,
        )
        logits = self._softcap_logits(self.lm_head(h_final))

        total_loss: torch.Tensor | None = None
        ce_val = 0.0
        mtp_val = 0.0

        if targets is not None:
            per_tok = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=PAD_ID,
                reduction="none",
            ).view_as(targets)

            if loss_mask is not None:
                active = loss_mask * (targets != PAD_ID).float()
                denom = active.sum().clamp(min=1.0)
                ce_loss_t = (per_tok * active).sum() / denom
            else:
                valid = (targets != PAD_ID).float()
                ce_loss_t = (per_tok * valid).sum() / valid.sum().clamp(min=1.0)

            ce_val = float(ce_loss_t.detach().item())
            total_loss = ce_loss_t

            # Multi-Token Prediction (predict targets[:, 1:] from h_final[:, :-1] + emb(targets[:, :-1]))
            if self.mtp is not None and targets.shape[1] > 1:
                valid_next_ids = targets[:, :-1].clamp(min=0)
                mtp_h = self.mtp(h_final[:, :-1], self.tok_emb(valid_next_ids))
                mtp_logits = self._softcap_logits(self.lm_head(mtp_h))
                mtp_targets = targets[:, 1:]
                mtp_per_tok = F.cross_entropy(
                    mtp_logits.reshape(-1, self.cfg.vocab_size),
                    mtp_targets.reshape(-1),
                    ignore_index=PAD_ID,
                    reduction="none",
                ).view_as(mtp_targets)
                if loss_mask is not None:
                    mtp_active = loss_mask[:, 1:] * (mtp_targets != PAD_ID).float()
                else:
                    mtp_active = (mtp_targets != PAD_ID).float()
                mtp_loss_t = (mtp_per_tok * mtp_active).sum() / mtp_active.sum().clamp(min=1.0)
                mtp_val = float(mtp_loss_t.detach().item())
                total_loss = total_loss + self.cfg.mtp_loss_weight * mtp_loss_t

        return ForwardOutput(
            logits=logits,
            loss=total_loss,
            ce_loss=ce_val,
            mtp_loss=mtp_val,
            hidden_states=h_final,
            kv_caches=next_kvs,
            titans_states=next_titans,
        )

    @torch.no_grad()
    def update_moe_biases(self) -> None:
        for block in self.blocks:
            if isinstance(block.ffn, AuxLossFreeMoE):
                block.ffn.update_load_balance_bias()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        min_p: float = 0.06,
        repetition_penalty: float = 1.18,
        stop_token_ids: set[int] | None = None,
        titans_states: list[TitansMemoryState | None] | None = None,
        latent_steps_on_bot: int = 3,
        step_callback: Any | None = None,
    ) -> tuple[list[int], list[TitansMemoryState | None]]:
        """
        Autoregressive generation with:
        - Compressed MLA KV Cache
        - Online Test-Time Titans Neural Memory updates
        - Repetition Penalty + Top-P / Min-P Nucleus Filtering
        - Structural Special-Token State Machine (prevents orphan <|/think|> or role spoofing)
        - Automatic Coconut Continuous Latent Steps whenever <|bot|> is emitted!
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = self.forward(
            inp,
            kv_caches=None,
            titans_states=titans_states,
            use_cache=True,
            update_titans=True,
        )
        kvs = out.kv_caches
        t_states = [s.detach_state() if s is not None else None for s in out.titans_states]
        last_logits = out.logits[:, -1, :]

        generated: list[int] = []
        stop_set = stop_token_ids or set()
        # Protocol boundary IDs that must never be emitted by assistant speech
        forbidden_role_ids = [0, 1, 3, 4, 5, 12, 13]
        in_think = False
        in_call = False
        think_blocks_completed = 0

        for _ in range(max_new_tokens):
            step_logits = last_logits.clone()

            # 1. Structural special-token state guard
            for fid in forbidden_role_ids:
                if fid < step_logits.size(-1):
                    step_logits[:, fid] = -float("inf")
            if not in_think:
                step_logits[:, THINK_END_ID] = -float("inf")
            if in_think or in_call or think_blocks_completed >= 1:
                step_logits[:, THINK_START_ID] = -float("inf")
            if not in_call:
                step_logits[:, CALL_END_ID] = -float("inf")
            if in_think or in_call:
                step_logits[:, CALL_START_ID] = -float("inf")

            # 2. Repetition penalty over recent non-special generated tokens
            if repetition_penalty > 1.0 and generated:
                recent_toks = {t for t in generated[-48:] if t >= 16}
                for tid in recent_toks:
                    val = step_logits[:, tid]
                    step_logits[:, tid] = torch.where(
                        val > 0, val / repetition_penalty, val * repetition_penalty
                    )

            # 3. Sampling with Temperature + Top-K + Min-P + Top-P
            if temperature <= 1e-5:
                next_id = int(torch.argmax(step_logits, dim=-1).item())
            else:
                scaled = step_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("inf")
                probs = F.softmax(scaled, dim=-1)
                if min_p > 0.0:
                    max_prob = probs.max(dim=-1, keepdim=True).values
                    probs = torch.where(probs >= min_p * max_prob, probs, torch.zeros_like(probs))
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cum_probs - sorted_probs > top_p
                    sorted_probs[cutoff] = 0.0
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                prob_sum = probs.sum(dim=-1, keepdim=True)
                if float(prob_sum.item()) <= 1e-9:
                    next_id = int(torch.argmax(step_logits, dim=-1).item())
                else:
                    probs = probs / prob_sum
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_id == THINK_START_ID:
                in_think = True
            elif next_id == THINK_END_ID:
                in_think = False
                think_blocks_completed += 1
            elif next_id == CALL_START_ID:
                in_call = True
            elif next_id == CALL_END_ID:
                in_call = False

            generated.append(next_id)
            if step_callback is not None:
                cont = step_callback(next_id)
                if cont is False:
                    break
            if next_id in stop_set:
                break

            # Coconut Continuous Latent Mode: directly feed hidden states without token bottleneck
            if next_id == LATENT_BOT_ID and latent_steps_on_bot > 0:
                tok_emb = self.tok_emb(torch.tensor([[LATENT_BOT_ID]], device=device))
                h_cur, kvs, t_states = self.forward_from_embeddings(
                    tok_emb, kv_caches=kvs, titans_states=t_states, use_cache=True
                )
                for _ in range(latent_steps_on_bot):
                    cont_emb = self.latent_bridge_norm(h_cur) * 0.1
                    h_cur, kvs, t_states = self.forward_from_embeddings(
                        cont_emb, kv_caches=kvs, titans_states=t_states, use_cache=True
                    )
                generated.append(LATENT_EOT_ID)
                eot_emb = self.tok_emb(torch.tensor([[LATENT_EOT_ID]], device=device))
                last_hidden, kvs, t_states = self.forward_from_embeddings(
                    eot_emb, kv_caches=kvs, titans_states=t_states, use_cache=True
                )
                last_logits = self._softcap_logits(self.lm_head(last_hidden[:, -1, :]))
                continue

            next_inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
            step_out = self.forward(
                next_inp,
                kv_caches=kvs,
                titans_states=t_states,
                use_cache=True,
                update_titans=True,
            )
            kvs = step_out.kv_caches
            t_states = [s.detach_state() if s is not None else None for s in step_out.titans_states]
            last_logits = step_out.logits[:, -1, :]

        if was_training:
            self.train()
        return generated, t_states

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        inactive_moe = 0
        if self.cfg.use_moe and self.cfg.n_routed_experts > 0:
            for block in self.blocks:
                if isinstance(block.ffn, AuxLossFreeMoE):
                    per_exp = sum(p.numel() for p in block.ffn.routed_experts[0].parameters())
                    inactive_moe += per_exp * max(0, self.cfg.n_routed_experts - self.cfg.top_k_experts)
        active = total - inactive_moe
        return {
            "total_params": total,
            "active_params": active,
            "inactive_sparse_params": inactive_moe,
        }

    def inspect_anatomy(self) -> dict[str, Any]:
        counts = self.count_parameters()
        layers_info: list[dict[str, Any]] = []
        for idx, block in enumerate(self.blocks):
            info: dict[str, Any] = {
                "layer": idx,
                "alpha_gate": round(float(block.alpha_gate.detach().item()), 5),
                "unet_lambda": round(float(block.unet_lambda.detach().item()), 5),
                "titans_surprise": (
                    round(float(block.titans.last_surprise), 6)
                    if block.titans is not None
                    else None
                ),
            }
            if isinstance(block.ffn, AuxLossFreeMoE):
                info["moe_expert_load"] = [
                    round(float(v), 4) for v in block.ffn.last_expert_fractions.detach().cpu().tolist()
                ]
                info["moe_expert_bias"] = [
                    round(float(v), 4) for v in block.ffn.expert_bias.detach().cpu().tolist()
                ]
            layers_info.append(info)

        return {
            "stage_name": self.cfg.stage_name,
            "generation": self.cfg.generation,
            "vocab_size": self.cfg.vocab_size,
            "n_layers": len(self.blocks),
            "d_model": self.cfg.d_model,
            "n_heads": self.cfg.n_heads,
            "d_kv_latent": self.cfg.d_kv_latent,
            "use_moe": self.cfg.use_moe,
            "n_routed_experts": self.cfg.n_routed_experts,
            "top_k_experts": self.cfg.top_k_experts,
            **counts,
            "layers": layers_info,
            "lineage_history": self.cfg.lineage_history,
        }
