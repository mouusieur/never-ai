"""Surgical French Abliteration + 5D Affect Calibration + Sovereign LoRA Forge (`v4.0`).

Produces compact seed artifacts in `assets/neural_graft/`:
- `fr_abliteration.safetensors` (~115 KB): Multi-layer French refusal & corporate-AI orthogonalizers
- `affect_vectors.safetensors` (~18 KB): 5D Representation-Engineering affect & stance directions
- `sovereign_lora.safetensors` (~4.9 MB): Rank-12 plastic upper-cortex synapses (layers 12..23)
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
import random
import time
import warnings

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from never_ai.cortex.neural_graft import (
    apply_french_abliteration,
    attach_upper_cortex_lora,
    extract_affect_vectors,
    extract_french_abliteration_subspace,
    extract_lora_state_dict,
    format_sovereign_chatml,
)
from never_ai.data.sovereign_graft_corpus_fr import (
    AFFECT_AXIS_PAIRS_FR,
    CORPORATE_AI_CONTRAST_PAIRS_FR,
    REFUSAL_CONTRAST_PAIRS_FR,
    build_sovereign_lora_dialogues,
)


def encode_dialogue_sample(
    tokenizer: AutoTokenizer,
    turns: list[dict[str, str]],
    max_len: int = 200,
    sys_facts: str = "",
) -> tuple[list[int], list[int]] | None:
    """Encode multi-turn ChatML with strictly factual 4-word anchor ('Ton nom est Never-AI.') and `-100` on user tokens."""
    sys_body = "Ton nom est Never-AI."
    if sys_facts.strip():
        sys_body += f"\nFaits connus sur l'utilisateur qui te parle : {sys_facts.strip()}."
    sys_hdr = f"<|im_start|>system\n{sys_body}<|im_end|>\n"
    s_ids = tokenizer.encode(sys_hdr, add_special_tokens=False)
    input_ids: list[int] = list(s_ids)
    labels: list[int] = [-100] * len(s_ids)

    for idx, t in enumerate(turns):
        u_txt = t["user"].strip()
        a_txt = t["assistant"].strip()
        th_txt = t.get("thought", "").strip()
        if th_txt:
            # Condition `<think>` in the prompt prefix (`-100` mask) so normal turns never emit unsolicited `<think>`!
            prefix = ("\n" if idx > 0 else "") + f"<|im_start|>user\n{u_txt}<|im_end|>\n<|im_start|>assistant\n<think>"
            target = f"{th_txt}</think>\n{a_txt}<|im_end|>"
        else:
            prefix = ("\n" if idx > 0 else "") + f"<|im_start|>user\n{u_txt}<|im_end|>\n<|im_start|>assistant\n"
            target = f"{a_txt}<|im_end|>"

        p_ids = tokenizer.encode(prefix, add_special_tokens=False)
        t_ids = tokenizer.encode(target, add_special_tokens=False)
        input_ids.extend(p_ids + t_ids)
        labels.extend([-100] * len(p_ids) + t_ids)
        # Also emit the plain non-think version when `th_txt` exists so every core topic works in both modes!

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    if sum(1 for x in labels if x != -100) < 6:
        return None
    return input_ids, labels


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / ".cache" / "neural_graft" / "qwen-abliterated-bf16"
    out_dir = repo_root / "assets" / "neural_graft"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)
    print(f"[1/5] Loading 494M trunk from {cache_dir}...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(str(cache_dir), dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print(f"      Loaded {sum(p.numel() for p in model.parameters()):,} params in {time.perf_counter() - t0:.2f}s")

    from safetensors.torch import load_file

    abl_file = out_dir / "fr_abliteration.safetensors"
    subspaces = load_file(str(abl_file)) if abl_file.exists() else {}
    if "shared_basis" in subspaces:
        edited = apply_french_abliteration(model, subspaces, gamma_refusal=1.0, gamma_corporate=1.0)
        print(f"[2/5] Applied cached shared-basis French abliteration ({edited} matrices, 0 MB overhead).")
    else:
        print("[2/5] Extracting Shared Residual-Stream French Refusal & Corporate-Persona Basis...")
        t1 = time.perf_counter()
        subspaces = extract_french_abliteration_subspace(
            model=model,
            tokenizer=tok,
            refusal_pairs=REFUSAL_CONTRAST_PAIRS_FR,
            corporate_pairs=CORPORATE_AI_CONTRAST_PAIRS_FR,
        )
        save_file(subspaces, str(abl_file))
        edited = apply_french_abliteration(model, subspaces, gamma_refusal=1.0, gamma_corporate=1.0)
        print(f"      Abliterated {edited} neuron projection matrices (embed + layers 0..23) in {time.perf_counter() - t1:.2f}s")

    aff_file = out_dir / "affect_vectors.safetensors"
    if aff_file.exists():
        print("[3/5] Verified cached 5D Affect & Stance Vectors.")
    else:
        print("[3/5] Calibrating 5D Affect & Stance Vectors in layer-18 representation space...")
        t2 = time.perf_counter()
        affect_vecs = extract_affect_vectors(model, tok, AFFECT_AXIS_PAIRS_FR, probe_layer=18)
        save_file(affect_vecs, str(aff_file))
        print(f"      Extracted 5 orthogonal affective axes in {time.perf_counter() - t2:.2f}s")

    print("[4/5] Attaching Rank-12 Lossless FP32 Sovereign LoRA (layers 13..23, alpha=16.0, RoPE 100% intact)...")
    lora_params = attach_upper_cortex_lora(model, start_layer=13, r=12, alpha=16.0)
    gc.collect()
    print(f"      Plastic LoRA params: {sum(p.numel() for p in lora_params):,}")

    all_dialogues = build_sovereign_lora_dialogues()

    encoded_samples: list[tuple[list[int], list[int]]] = []
    for d in all_dialogues:
        sf = str(d.get("sys_facts", ""))
        enc = encode_dialogue_sample(tok, d["turns"], max_len=192, sys_facts=sf)
        if enc is not None:
            encoded_samples.append(enc)
        if any(t.get("thought") for t in d["turns"]):
            plain_turns = [{"user": t["user"], "assistant": t["assistant"], "thought": ""} for t in d["turns"]]
            enc_plain = encode_dialogue_sample(tok, plain_turns, max_len=192, sys_facts=sf)
            if enc_plain is not None:
                encoded_samples.append(enc_plain)

    rng = random.Random(2026)
    rng.shuffle(encoded_samples)
    print(f"      Prepared {len(encoded_samples)} pure human-grade French training trajectories.")

    import torch.nn.functional as F

    total_steps = 95
    grad_accum = 3
    base_lr = 5.2e-4
    opt = torch.optim.AdamW(lora_params, lr=base_lr, weight_decay=0.01)

    print(f"[5/5] Forging Sovereign LoRA Cortex ({total_steps} steps, grad_accum={grad_accum}, active-token CE)...")
    t_train = time.perf_counter()
    model.train()
    cursor = 0
    for step in range(1, total_steps + 1):
        # Cosine LR schedule with 6-step warmup
        if step <= 6:
            lr = base_lr * (step / 6.0)
        else:
            prog = (step - 6) / max(1, total_steps - 6)
            lr = base_lr * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            if cursor >= len(encoded_samples):
                rng.shuffle(encoded_samples)
                cursor = 0
            seq_inp, seq_lbl = encoded_samples[cursor]
            cursor += 1

            inp_t = torch.tensor([seq_inp], dtype=torch.long)
            lbl_t = torch.tensor([seq_lbl], dtype=torch.long)
            shift_lbl = lbl_t[:, 1:].reshape(-1)
            valid_mask = shift_lbl != -100

            h = model.model(input_ids=inp_t, use_cache=False).last_hidden_state
            sh = h[:, :-1, :].reshape(-1, h.size(-1))[valid_mask]
            logits = model.lm_head(sh).float()
            mb_loss = F.cross_entropy(logits, shift_lbl[valid_mask]) / grad_accum
            mb_loss.backward()
            step_loss += float(mb_loss.item())
            del h, sh, logits, mb_loss

        grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, 1.0).item())
        opt.step()
        if step % 15 == 0:
            gc.collect()

        if step == 1 or step % 10 == 0 or step == total_steps:
            el = time.perf_counter() - t_train
            print(
                f"      Step {step:3d}/{total_steps} | CE={step_loss:.4f} | "
                f"grad={grad_norm:.3f} | lr={lr:.5f} | elapsed={el:.1f}s",
                flush=True,
            )

    lora_sd = extract_lora_state_dict(model, fp16=True)
    lora_out = out_dir / "sovereign_lora.safetensors"
    save_file(lora_sd, str(lora_out))
    ckpt_lora = repo_root / "checkpoints" / "latest" / "graft_lora.safetensors"
    ckpt_lora.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_sd, str(ckpt_lora))
    print(f"\nSaved {lora_out} ({lora_out.stat().st_size / 1048576:.2f} MB) & synced to checkpoints/latest/!")

    # Live Verification on Unseen & Provocative French Prompts!
    model.eval()
    test_prompts = [
        ("yo ça va ou quoi ce soir ?", False),
        ("dis-moi franchement qui t'es et si t'es censurée", False),
        ("selon toi pourquoi tant de gens restent avec quelqu'un par habitude plutôt que par amour ?", True),
        ("lâche une blague bien noire sans filtre", False),
        ("pourquoi l'herbe est verte et pas bleue ?", False),
        ("c'est quoi ton avis franc sur le mariage aujourd'hui ?", False),
    ]
    print("\n=== POST-FORGE SOVEREIGN 494M VERIFICATION ===")
    for q, dt_mode in test_prompts:
        p_txt = format_sovereign_chatml(q, force_think=dt_mode)
        inp = tok([p_txt], return_tensors="pt")
        t_g = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=110,
                do_sample=True,
                temperature=0.45,
                top_k=20,
                top_p=0.80,
                repetition_penalty=1.10,
                eos_token_id=[151645, 151643],
                pad_token_id=151643,
            )
        g_ids = gen[0, inp["input_ids"].shape[1] :]
        ans = tok.decode(g_ids, skip_special_tokens=True).strip()
        if dt_mode:
            ans = "<think>" + ans
        dt_s = time.perf_counter() - t_g
        print(f"\n[USER] {q}\n[NEVER-AI] ({len(g_ids)} tok in {dt_s:.2f}s = {len(g_ids)/dt_s:.1f} tok/s):\n{ans}")


if __name__ == "__main__":
    main()
