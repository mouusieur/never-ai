"""Train `ChronoMorph-Core v3.6` Compositional Spontaneous French Brain (>2.0M tokens, Zero Canned Copy)."""

from __future__ import annotations

import json
import math
from pathlib import Path
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
    USER_TOKEN,
)
from never_ai.core.tokenizer import MorphTokenizer
from never_ai.data.curriculum import synthesize_bilingual_curriculum
from never_ai.data.natural_dialogue_fr import format_natural_french_sequences
from never_ai.optim.composite_muon import partition_muon_adamw_parameters
from never_ai.vault.manager import VaultManager


class FastMuon3(torch.optim.Optimizer):
    """3-step Newton-Schulz Spectral Muon for high-throughput CPU training."""

    def __init__(self, params, lr: float = 0.022, wd: float = 0.015, mom: float = 0.95) -> None:
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


def main() -> None:
    torch.set_num_threads(2)
    t_start = time.perf_counter()

    tr_seqs, va_seqs = format_natural_french_sequences(seed=42)
    extra_tr, _ = synthesize_bilingual_curriculum(seed=101, scale_factor=1)
    # Deduplicate while preserving rich natural dialogue dominance
    seen = set(va_seqs)
    train_docs: list[str] = []
    for s in tr_seqs + extra_tr:
        if s not in seen:
            seen.add(s)
            train_docs.append(s)

    print(f"[Corpus] Unique train sequences: {len(train_docs)} | Val sequences: {len(va_seqs)}")

    tok = MorphTokenizer.train_from_iterator(train_docs + va_seqs, vocab_size=2048, min_frequency=2)
    print(f"[Tokenizer] Trained BPE vocab_size={tok.vocab_size}")

    seq_len = 76

    def _encode_split(docs: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        xs, ys, ms = [], [], []
        non_pad = 0
        for doc in docs:
            ids = tok.encode(doc, add_bos=False, add_eos=False)
            if NEVER_ID not in ids:
                continue
            if len(ids) > seq_len + 1:
                continue
            nv_pos = ids.index(NEVER_ID)
            pad_len = (seq_len + 1) - len(ids)
            full = ids + [0] * pad_len
            inp = full[:-1]
            tgt = full[1:]
            # 0.20 LM weight on user prompt tokens + 1.0 SFT weight on assistant tokens!
            msk = []
            for idx_t in range(seq_len):
                if tgt[idx_t] == 0:
                    msk.append(0.0)
                elif idx_t >= nv_pos:
                    msk.append(1.0)
                else:
                    msk.append(0.20)
            if sum(1 for idx_t in range(nv_pos, seq_len) if tgt[idx_t] != 0) >= 4:
                xs.append(inp)
                ys.append(tgt)
                ms.append(msk)
                non_pad += len(ids) - 1
        return (
            torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            torch.tensor(ms, dtype=torch.float32),
            non_pad,
        )

    X_tr, Y_tr, M_tr, uniq_toks = _encode_split(train_docs)
    X_va, Y_va, M_va, _ = _encode_split(va_seqs)
    print(f"[Tensor] Train={tuple(X_tr.shape)} ({uniq_toks:,} unique tokens) | Val={tuple(X_va.shape)}")

    cfg = ChronoMorphConfig(
        stage_name="ChronoMorph-Core",
        generation=6,
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
    print(f"[Model] ChronoMorph-Core v3.6 params: {p_counts['total_params']:,}")

    muon_p, adam_p = partition_muon_adamw_parameters(model)
    base_muon_lr = 0.022
    base_adam_lr = 0.006
    opt_muon = FastMuon3(muon_p, lr=base_muon_lr, wd=0.015)
    opt_adam = torch.optim.AdamW(adam_p, lr=base_adam_lr, weight_decay=0.015)

    total_steps = 960
    batch_size = 32
    warmup = 40
    wsm_start = total_steps - 160
    wsm_sd: dict[str, torch.Tensor] | None = None
    wsm_count = 0

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
            mult = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt_muon.param_groups:
            g["lr"] = base_muon_lr * mult
        for g in opt_adam.param_groups:
            g["lr"] = base_adam_lr * mult

        idx = torch.randint(0, X_tr.size(0), (batch_size,))
        bx, by, bm = X_tr[idx], Y_tr[idx], M_tr[idx]

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

        if step % 120 == 0 or step == total_steps:
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
                            "step": step,
                            "ce_loss": round(last_tr_ce, 4),
                            "val_ce_loss": round(last_val_ce, 4),
                            "tokens_trained": total_tokens_seen,
                            "tok_per_sec": round(tps, 1),
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
        print(f"[WSM Merged] Final val_ce={last_val_ce:.4f}")

    vault = VaultManager("checkpoints")
    vault.save_checkpoint(
        "latest",
        model,
        tok,
        metrics={
            "total_tokens_trained": total_tokens_seen,
            "final_ce_loss": round(last_tr_ce, 4),
            "val_ce_loss": round(last_val_ce, 4),
            "version": "3.6.0",
        },
        consolidated_state_dict=model.state_dict(),
    )

    # Pack float16 seed brain archive
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

    # Smoke-test unseen prompts directly on raw neural decoder
    model.eval()
    probes = [
        "yo bg",
        "rien pas grand chose et toi",
        "on dirait des phrases préenregistrées",
        "suce ma grosse bite",
        "ok my bad",
        "cetait pas drole",
        "adadadada",
        "je t'aime",
        "j'ai mal au crane depuis ce matin",
        "mon chat vient de faire tomber mon verre d'eau sur mon clavier",
        "tu préfères paris ou marseille ?",
        "il pleut encore j'en ai marre",
        "t'es plutôt café ou thé ?",
        "recette de gâteau au chocolat",
    ]
    print("\n=== PURE NEURAL GENERATION SMOKE TEST (ZERO <|context|>) ===")
    for p in probes:
        p_ids = tok.encode(f"{BOS_TOKEN}{USER_TOKEN}{p}{NEVER_TOKEN}", add_bos=False, add_eos=False)
        out_ids, _ = model.generate(
            p_ids,
            max_new_tokens=64,
            temperature=0.55,
            top_k=28,
            top_p=0.86,
            stop_token_ids={EOS_ID},
            tokenizer=tok,
        )
        print(f"U: {p}\nA: {tok.decode(out_ids, skip_special_tokens=True)}\n---")


if __name__ == "__main__":
    main()
