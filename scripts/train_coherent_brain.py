"""Train `ChronoMorph-Forge v3.7` Open-World & Multi-Turn French Brain (>3.1M tokens, Length-Bucketed)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
import zipfile

import torch
from safetensors.torch import save

from never_ai.core.config import ChronoMorphConfig
from never_ai.core.model import ChronoMorphLM
from never_ai.core.special_tokens import (
    BOS_TOKEN,
    EOS_ID,
    NEVER_ID,
    NEVER_TOKEN,
    USER_ID,
    USER_TOKEN,
)
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data.natural_dialogue_fr import build_open_french_sequences
from never_ai.optim.composite_muon import partition_muon_adamw_parameters
from never_ai.vault.manager import VaultManager


class FastMuon3(torch.optim.Optimizer):
    """3-step Newton-Schulz Spectral Muon for high-throughput CPU training."""

    def __init__(self, params, lr: float = 0.024, wd: float = 0.012, mom: float = 0.95) -> None:
        super().__init__(params, dict(lr=lr, wd=wd, mom=mom))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        for grp in self.param_groups:
            lr, wd, mom = grp["lr"], grp["wd"], grp["mom"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                buf = st.setdefault("mb", torch.zeros_like(g))
                buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom)
                tr = u.size(0) > u.size(1)
                if tr:
                    u = u.T
                u = u / (u.norm() + 1e-7)
                for _ in range(3):
                    a = u @ u.T
                    u = 3.4445 * u - 4.7750 * (a @ u) + 2.0315 * (a @ a @ u)
                if tr:
                    u = u.T
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr * 0.2 * math.sqrt(max(p.size(0), p.size(1))))


def _build_multiturn_mask(inp: list[int], tgt: list[int]) -> list[float]:
    """Assign 1.0 on final assistant turn, 0.50 on first assistant turn, 0.22 on user prompt tokens."""
    T = len(inp)
    nv_positions = [i for i, tid in enumerate(inp) if tid == NEVER_ID]
    last_nv = nv_positions[-1] if nv_positions else T
    in_assistant = False
    mask: list[float] = []
    for i in range(T):
        tok_in = inp[i]
        tok_tg = tgt[i]
        if tok_in == NEVER_ID:
            in_assistant = True
        elif tok_in == USER_ID:
            in_assistant = False
        if tok_tg == 0:
            mask.append(0.0)
        elif i >= last_nv and in_assistant:
            mask.append(1.0)
        elif in_assistant:
            mask.append(0.50)
        else:
            mask.append(0.22)
    return mask


def main() -> None:
    torch.set_num_threads(2)
    t_start = time.perf_counter()

    tr_seqs, va_seqs = build_open_french_sequences(seed=42)
    print(f"[Corpus] Train sequences: {len(tr_seqs):,} | Val sequences: {len(va_seqs):,}")

    tok = MorphTokenizer.train_from_iterator(tr_seqs + va_seqs, vocab_size=3072, min_frequency=2)
    print(f"[Tokenizer] Trained BPE vocab_size={tok.vocab_size}")

    # Length buckets eliminate >50% zero-padding FLOP waste!
    bucket_specs = [(48, 44), (76, 32), (112, 24)]
    buckets_x: list[torch.Tensor] = []
    buckets_y: list[torch.Tensor] = []
    buckets_m: list[torch.Tensor] = []

    raw_by_b: dict[int, tuple[list[list[int]], list[list[int]], list[list[float]]]] = {
        b_len: ([], [], []) for b_len, _ in bucket_specs
    }
    uniq_toks = 0
    for doc in tr_seqs:
        ids = tok.encode(doc, add_bos=False, add_eos=False)
        if NEVER_ID not in ids or len(ids) < 6:
            continue
        target_b = None
        for b_len, _ in bucket_specs:
            if len(ids) <= b_len + 1:
                target_b = b_len
                break
        if target_b is None:
            continue
        full = ids + [0] * ((target_b + 1) - len(ids))
        inp = full[:-1]
        tgt = full[1:]
        msk = _build_multiturn_mask(inp, tgt)
        if sum(msk) >= 4.0:
            raw_by_b[target_b][0].append(inp)
            raw_by_b[target_b][1].append(tgt)
            raw_by_b[target_b][2].append(msk)
            uniq_toks += len(ids) - 1

    bucket_bs: list[int] = []
    bucket_weights: list[float] = []
    for b_len, b_sz in bucket_specs:
        xs, ys, ms = raw_by_b[b_len]
        if len(xs) >= b_sz:
            buckets_x.append(torch.tensor(xs, dtype=torch.long))
            buckets_y.append(torch.tensor(ys, dtype=torch.long))
            buckets_m.append(torch.tensor(ms, dtype=torch.float32))
            bucket_bs.append(b_sz)
            bucket_weights.append(float(len(xs)))
            print(f"  Bucket T={b_len:3d} (B={b_sz}): {len(xs):,} sequences")
    print(f"[Corpus] Total active unique tokens encoded: {uniq_toks:,}")

    # Encode validation split at T=96
    va_T = 96
    vx, vy, vm = [], [], []
    for doc in va_seqs:
        ids = tok.encode(doc, add_bos=False, add_eos=False)
        if NEVER_ID not in ids or len(ids) > va_T + 1:
            continue
        full = ids + [0] * ((va_T + 1) - len(ids))
        inp, tgt = full[:-1], full[1:]
        vx.append(inp)
        vy.append(tgt)
        vm.append(_build_multiturn_mask(inp, tgt))
    X_va = torch.tensor(vx, dtype=torch.long)
    Y_va = torch.tensor(vy, dtype=torch.long)
    M_va = torch.tensor(vm, dtype=torch.float32)

    cfg = ChronoMorphConfig(
        stage_name="ChronoMorph-Forge",
        generation=7,
        vocab_size=tok.vocab_size,
        max_seq_len=256,
        n_layers=6,
        d_model=288,
        n_heads=6,
        d_head=48,
        d_kv_latent=96,
        d_rope=16,
        d_ff=864,
        use_moe=False,
        use_titans_memory=False,
        use_unet_skips=True,
        use_mtp=False,
    )
    model = ChronoMorphLM(cfg)
    p_counts = model.count_parameters()
    print(f"[Model] ChronoMorph-Forge v3.7 params: {p_counts['total_params']:,}")

    muon_p, adam_p = partition_muon_adamw_parameters(model)
    base_muon_lr = 0.024
    base_adam_lr = 0.0065
    opt_muon = FastMuon3(muon_p, lr=base_muon_lr, wd=0.012)
    opt_adam = torch.optim.AdamW(adam_p, lr=base_adam_lr, weight_decay=0.012)

    total_steps = 1420
    warmup = 45
    wsm_start = total_steps - 200
    wsm_sd: dict[str, torch.Tensor] | None = None
    wsm_count = 0

    rng = random.Random(42)
    total_tokens_seen = 0
    t_loop = time.perf_counter()
    telem_path = Path("logs/training_telemetry.jsonl")
    telem_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    last_val_ce = 9.9
    last_tr_ce = 9.9

    for step in range(1, total_steps + 1):
        if step <= warmup:
            mult = step / warmup
        else:
            prog = (step - warmup) / max(1, total_steps - warmup)
            mult = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt_muon.param_groups:
            g["lr"] = base_muon_lr * mult
        for g in opt_adam.param_groups:
            g["lr"] = base_adam_lr * mult

        b_idx = rng.choices(range(len(buckets_x)), weights=bucket_weights, k=1)[0]
        Xb, Yb, Mb = buckets_x[b_idx], buckets_y[b_idx], buckets_m[b_idx]
        bsz = bucket_bs[b_idx]
        ridx = torch.randint(0, Xb.size(0), (bsz,))
        bx, by, bm = Xb[ridx], Yb[ridx], Mb[ridx]

        out = model(bx, targets=by, loss_mask=bm, update_titans=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adam.step()
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        total_tokens_seen += int(bx.numel())
        last_tr_ce = float(out.ce_loss)

        if step >= wsm_start and step % 4 == 0:
            with torch.no_grad():
                if wsm_sd is None:
                    wsm_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    wsm_count = 1
                else:
                    wsm_count += 1
                    alpha = 1.0 / wsm_count
                    for k, v in model.state_dict().items():
                        if v.is_floating_point():
                            wsm_sd[k].lerp_(v.detach(), alpha)
                        else:
                            wsm_sd[k] = v.detach().clone()

        if step % 160 == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                vo = model(X_va, targets=Y_va, loss_mask=M_va, update_titans=False)
                last_val_ce = float(vo.ce_loss)
            model.train()
            dt = time.perf_counter() - t_loop
            tps = total_tokens_seen / max(1e-3, dt)
            print(
                f"Step {step:4d}/{total_steps} | train_ce={last_tr_ce:.4f} | val_ce={last_val_ce:.4f} "
                f"| tokens={total_tokens_seen:,} | {tps:,.0f} tok/s ({dt:.1f}s)"
            )
            with telem_path.open("a", encoding="utf-8") as tf:
                tf.write(
                    json.dumps(
                        {
                            "type": "train_step",
                            "step": step,
                            "ce_loss": round(last_tr_ce, 4),
                            "val_ce_loss": round(last_val_ce, 4),
                            "tokens_trained": total_tokens_seen,
                            "tok_per_sec": round(tps, 1),
                            "source": "forge v3.7",
                            "timestamp": round(time.time(), 2),
                        }
                    )
                    + "\n"
                )

    if wsm_sd is not None:
        model.load_state_dict(wsm_sd)
        model.eval()
        with torch.no_grad():
            vo = model(X_va, targets=Y_va, loss_mask=M_va, update_titans=False)
            last_val_ce = float(vo.ce_loss)
        print(f"[WSM Merged] Final val_ce={last_val_ce:.4f} | total_tokens={total_tokens_seen:,}")

    vault = VaultManager("checkpoints")
    vault.save_checkpoint(
        "latest",
        model,
        tok,
        metrics={
            "total_tokens_trained": total_tokens_seen,
            "final_ce_loss": round(last_tr_ce, 4),
            "val_ce_loss": round(last_val_ce, 4),
            "version": "3.7.0",
        },
        consolidated_state_dict=model.state_dict(),
    )

    sd_fp16 = {
        k: (v.to(torch.float16) if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    zip_path = Path("assets/seed_brain/chronomorph_core_v34.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/latest")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("latest/model_wsm.safetensors", save(sd_fp16))
        for fn in ("config.json", "manifest.json", "tokenizer.json"):
            zf.write(ckpt_dir / fn, arcname=f"latest/{fn}")
        if telem_path.exists():
            zf.write(telem_path, arcname="training_telemetry.jsonl")

    print(
        f"[Done] Saved checkpoints/latest & {zip_path} ({zip_path.stat().st_size / 1048576:.2f} MiB) "
        f"in {time.perf_counter() - t_start:.1f}s!"
    )


if __name__ == "__main__":
    main()
