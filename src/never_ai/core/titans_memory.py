"""Google Titans Neural Long-Term Memory (Test-Time Training via Surprise Gradient)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TitansMemoryState:
    """Online fast-weight MLP parameters and surprise momentum carried across chunks/turns."""

    w1: torch.Tensor  # (B, d_mem, d_mem_hidden)
    w2: torch.Tensor  # (B, d_mem_hidden, d_mem)
    s1: torch.Tensor  # (B, d_mem, d_mem_hidden)
    s2: torch.Tensor  # (B, d_mem_hidden, d_mem)
    last_surprise: float = 0.0

    def detach_state(self) -> "TitansMemoryState":
        return TitansMemoryState(
            w1=self.w1.detach().clone(),
            w2=self.w2.detach().clone(),
            s1=self.s1.detach().clone(),
            s2=self.s2.detach().clone(),
            last_surprise=self.last_surprise,
        )


def _silu_and_grad(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sig = torch.sigmoid(u)
    act = u * sig
    grad = sig * (1.0 + u * (1.0 - sig))
    return act, grad


class TitansMAGMemory(nn.Module):
    """
    Chunkwise-parallel Neural Long-Term Memory inspired by Google Titans (arXiv:2501.00663).
    Updates a 2-layer MLP memory M(k) = SiLU(k @ W1) @ W2 online using closed-form analytical
    surprise gradients, both during meta-learning (training) and torch.no_grad() inference!
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int = 64,
        d_mem_hidden: int = 128,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.d_mem_hidden = d_mem_hidden
        self.chunk_size = chunk_size

        self.w_k = nn.Linear(d_model, d_mem, bias=False)
        self.w_v = nn.Linear(d_model, d_mem, bias=False)
        self.w_q = nn.Linear(d_model, d_mem, bias=False)
        self.w_out = nn.Linear(d_mem, d_model, bias=False)

        # Meta-learned initial fast weights for the associative MLP
        self.w1_init = nn.Parameter(torch.randn(d_mem, d_mem_hidden) * (d_mem**-0.5) * 0.25)
        self.w2_init = nn.Parameter(torch.zeros(d_mem_hidden, d_mem))

        # Data-dependent plasticity gates: [forgetting alpha, momentum eta, lr theta]
        self.gates_proj = nn.Linear(d_model, 3, bias=True)
        nn.init.zeros_(self.gates_proj.weight)
        with torch.no_grad():
            self.gates_proj.bias.copy_(torch.tensor([-3.0, 1.5, -1.5]))

        # Residual contribution gate (allows exact zero-shock insertion/expansion)
        self.mem_scale = nn.Parameter(torch.tensor(0.2))
        self.last_surprise: float = 0.0

        # Zero-init output projection at birth for smooth, stable warm-start
        nn.init.normal_(self.w_out.weight, mean=0.0, std=0.01)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> TitansMemoryState:
        w1 = self.w1_init.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        w2 = self.w2_init.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        s1 = torch.zeros_like(w1)
        s2 = torch.zeros_like(w2)
        return TitansMemoryState(w1=w1, w2=w2, s1=s1, s2=s2, last_surprise=0.0)

    @staticmethod
    def _soft_bound_matrix(m: torch.Tensor, max_fro: float = 8.0) -> torch.Tensor:
        fro = m.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(max_fro / fro, max=1.0)
        return m * scale

    def forward(
        self,
        x: torch.Tensor,
        state: TitansMemoryState | None = None,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, TitansMemoryState]:
        """
        Args:
            x: Normalized block input (B, T, d_model)
            state: Optional carry-over TitansMemoryState from prior context/turn
            update_memory: Whether to execute test-time fast-weight gradient updates
        Returns:
            mem_delta: (B, T, d_model) long-term memory contribution
            new_state: Updated TitansMemoryState
        """
        B, T, _ = x.shape
        if state is None or state.w1.shape[0] != B:
            cur = self.init_state(B, x.device, x.dtype)
        else:
            cur = TitansMemoryState(
                w1=state.w1.to(device=x.device, dtype=x.dtype),
                w2=state.w2.to(device=x.device, dtype=x.dtype),
                s1=state.s1.to(device=x.device, dtype=x.dtype),
                s2=state.s2.to(device=x.device, dtype=x.dtype),
                last_surprise=state.last_surprise,
            )

        k_all = F.normalize(self.w_k(x), dim=-1, eps=1e-6)
        v_all = self.w_v(x)
        q_all = F.normalize(self.w_q(x), dim=-1, eps=1e-6)

        outputs: list[torch.Tensor] = []
        surprise_accum = 0.0
        n_chunks = 0

        w1, w2, s1, s2 = cur.w1, cur.w2, cur.s1, cur.s2

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            c_len = end - start
            x_c = x[:, start:end]
            k_c = k_all[:, start:end]
            v_c = v_all[:, start:end]
            q_c = q_all[:, start:end]

            # 1. Query current long-term neural memory
            hq = F.silu(torch.bmm(q_c, w1))
            mem_read = torch.bmm(hq, w2)
            outputs.append(mem_read)

            # 2. Online Surprise-Driven Test-Time Gradient Update
            if update_memory:
                u_k = torch.bmm(k_c, w1)  # (B, C, d_mem_hidden)
                act_k, dact_k = _silu_and_grad(u_k)
                v_pred = torch.bmm(act_k, w2)
                err = v_pred - v_c  # (B, C, d_mem)

                with torch.no_grad():
                    surprise_accum += float(err.detach().pow(2).mean().item())
                    n_chunks += 1

                scale = 2.0 / float(c_len * self.d_mem)
                grad_w2 = scale * torch.bmm(act_k.transpose(1, 2), err)
                grad_act = scale * torch.bmm(err, w2.transpose(1, 2)) * dact_k
                grad_w1 = torch.bmm(k_c.transpose(1, 2), grad_act)

                gate_raw = self.gates_proj(x_c.mean(dim=1))  # (B, 3)
                alpha = (0.05 * torch.sigmoid(gate_raw[:, 0])).view(B, 1, 1)
                eta = (0.80 + 0.19 * torch.sigmoid(gate_raw[:, 1])).view(B, 1, 1)
                theta = (0.25 * torch.sigmoid(gate_raw[:, 2])).view(B, 1, 1)

                s1 = eta * s1 - theta * grad_w1
                s2 = eta * s2 - theta * grad_w2
                w1 = self._soft_bound_matrix((1.0 - alpha) * w1 + s1)
                w2 = self._soft_bound_matrix((1.0 - alpha) * w2 + s2)

        mem_seq = torch.cat(outputs, dim=1)
        out = self.mem_scale * self.w_out(mem_seq)

        avg_surprise = surprise_accum / max(1, n_chunks)
        self.last_surprise = avg_surprise
        new_state = TitansMemoryState(w1=w1, w2=w2, s1=s1, s2=s2, last_surprise=avg_surprise)
        return out, new_state
