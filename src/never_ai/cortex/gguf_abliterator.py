"""Outlier-Safe MLP Weight-Space Abliterator & Provisioner for Gemma-3-4B GGUF (`v4.2`).

Key mechanistic design lessons verified empirically on Gemma-3-4B (`34` layers, `d=2560`):
1. Strip unused SigLIP vision (`v.*`) & projector (`mm.*`) tensors (`~801 MB` saved -> `2.42 GiB` pure text brain),
   trim Ollama's extra 262,145th vision token metadata to `262,144`, and set `layer_norm_rms_epsilon = 1e-6`.
2. Leave all Attention `attn_output` (`W_O`) matrices 100% untouched so subject/object pronoun binding
   (`je/ma` vs `tu/ta` in multi-turn roleplay) and French orthography remain 100% intact.
3. Project out the Rank-2 topic-free French refusal subspace ONLY from `blk.{16..29}.ffn_down.weight` (`gamma=1.0`),
   masking the top-96 massive-activation BOS sink coordinates so attention sinks are never perturbed.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np


LLAMA_RES_REPO = "https://github.com/kulogix/ai_resources.git"
LLAMA_BIN_REL = "bin/llama-b8204-bin-ubuntu-x64.tar.gz"
GEMMA3_4B_REPO = "https://github.com/jayesnc2512/ollama-gemma-3.4b.git"


def _mask_outliers_and_norm(v: np.ndarray, k: int = 96) -> np.ndarray:
    out = v.astype(np.float32, copy=True)
    if out.size > k:
        thresh = np.partition(np.abs(out), -k)[-k]
        out[np.abs(out) >= thresh] = 0.0
    norm = float(np.linalg.norm(out))
    if norm > 1e-8:
        out /= norm
    return out


def ensure_llama_server_binary(cache_root: Path) -> Path:
    """Ensure prebuilt AVX-512 `llama-server` (`b8204`) is extracted under `.cache/llama_cpp/llama-b8204`."""
    bin_dir = cache_root / "llama_cpp" / "llama-b8204"
    srv = bin_dir / "llama-server"
    if srv.exists() and os.access(srv, os.X_OK):
        return srv

    bin_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama_bin_") as tmp:
        tmp_path = Path(tmp) / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", LLAMA_RES_REPO, str(tmp_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "sparse-checkout", "set", LLAMA_BIN_REL],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        tarball = tmp_path / LLAMA_BIN_REL
        subprocess.run(
            ["tar", "-xzf", str(tarball), "-C", str(bin_dir.parent)],
            check=True,
        )
    if not srv.exists():
        raise FileNotFoundError(f"llama-server binary missing after extraction: {srv}")
    srv.chmod(0o755)
    return srv


def build_abliterated_gemma3_4b(
    raw_gguf_path: Path,
    refusal_cvec_path: Path,
    output_gguf_path: Path,
    *,
    mlp_layers: range = range(16, 30),
    gamma: float = 1.00,
) -> Path:
    """Strip vision tensors & perform outlier-safe Rank-2 MLP abliteration into `output_gguf_path`."""
    from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType, quants

    r_vec = GGUFReader(str(refusal_cvec_path), "r")
    raw_dirs: dict[int, np.ndarray] = {}
    for t in r_vec.tensors:
        if t.name.startswith("direction."):
            il = int(t.name.split(".")[1]) - 1
            raw_dirs[il] = np.array(t.data, dtype=np.float32).copy()

    dirs = {il: _mask_outliers_and_norm(v, 96) for il, v in raw_dirs.items()}
    r_star = np.zeros(2560, dtype=np.float32)
    anchor = dirs.get(23, next(iter(dirs.values())))
    for il in range(20, 28):
        if il in dirs:
            v = dirs[il]
            if float(np.dot(v, anchor)) < 0.0:
                v = -v
            r_star += v
    r_star = _mask_outliers_and_norm(r_star, 96)

    output_gguf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_gguf_path.with_suffix(".tmp.gguf")
    r_mod = GGUFReader(str(raw_gguf_path), "r")
    w = GGUFWriter(str(tmp_out), arch="gemma3")

    has_eps = any(f.name == "gemma3.attention.layer_norm_rms_epsilon" for f in r_mod.fields.values())
    if not has_eps:
        w.add_float32("gemma3.attention.layer_norm_rms_epsilon", 1e-6)

    skip_keys = {"general.architecture", "gemma3.mm.tokens_per_image"}
    for field in r_mod.fields.values():
        k = field.name
        if k in skip_keys or k.startswith("GGUF.") or ".vision." in k or k.startswith("clip."):
            continue
        t0 = field.types[0]
        if t0 == GGUFValueType.ARRAY:
            sub_t = field.types[1]
            if sub_t == GGUFValueType.STRING:
                vals = [field.parts[idx].tobytes().decode("utf-8", "surrogateescape") for idx in field.data]
            else:
                vals = [
                    field.parts[idx].tolist()[0] if len(field.parts[idx]) == 1 else field.parts[idx].tolist()
                    for idx in field.data
                ]
            if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type") and len(vals) == 262145:
                vals = vals[:262144]
            w.add_array(k, vals)
        elif t0 == GGUFValueType.STRING:
            w.add_string(k, field.parts[field.data[0]].tobytes().decode("utf-8", "surrogateescape"))
        elif t0 == GGUFValueType.BOOL:
            w.add_bool(k, bool(field.parts[field.data[0]][0]))
        elif t0 == GGUFValueType.UINT32:
            w.add_uint32(k, int(field.parts[field.data[0]][0]))
        elif t0 == GGUFValueType.INT32:
            w.add_int32(k, int(field.parts[field.data[0]][0]))
        elif t0 == GGUFValueType.FLOAT32:
            w.add_float32(k, float(field.parts[field.data[0]][0]))
        elif t0 == GGUFValueType.UINT64:
            w.add_uint64(k, int(field.parts[field.data[0]][0]))
        elif t0 == GGUFValueType.INT64:
            w.add_int64(k, int(field.parts[field.data[0]][0]))
        elif t0 == GGUFValueType.FLOAT64:
            w.add_float64(k, float(field.parts[field.data[0]][0]))

    target_set = set(mlp_layers)
    for t in r_mod.tensors:
        if t.name.startswith("v.") or t.name.startswith("mm."):
            continue
        parts = t.name.split(".")
        if (
            len(parts) == 4
            and parts[0] == "blk"
            and int(parts[1]) in target_set
            and parts[2] == "ffn_down"
            and parts[3] == "weight"
        ):
            il = int(parts[1])
            B = np.column_stack([dirs[il], r_star]).astype(np.float32)
            U, _ = np.linalg.qr(B)
            W_f32 = quants.dequantize(t.data, t.tensor_type).astype(np.float32)
            W_new = W_f32 - float(gamma) * (U @ (U.T @ W_f32))
            q_bytes = quants.quantize(W_new, GGMLQuantizationType.Q8_0)
            w.add_tensor(t.name, q_bytes, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    tmp_out.replace(output_gguf_path)
    return output_gguf_path


def ensure_sovereign_gemma3_4b(cache_root: Path, assets_graft_dir: Path) -> Path:
    """Ensure `.cache/neural_graft/gemma3-4b-mlp-sovereign.gguf` exists, building from GitHub parts if needed."""
    out_gguf = cache_root / "neural_graft" / "gemma3-4b-mlp-sovereign.gguf"
    if out_gguf.exists() and out_gguf.stat().st_size > 2_000_000_000:
        return out_gguf

    refusal_cvec = assets_graft_dir / "gemma3_4b_refusal_dir.gguf"
    if not refusal_cvec.exists():
        raise FileNotFoundError(f"Missing tracked refusal direction file: {refusal_cvec}")

    with tempfile.TemporaryDirectory(prefix="gemma3_4b_") as tmp:
        clone_dir = Path(tmp) / "ollama_gemma"
        raw_gguf = Path(tmp) / "raw.gguf"
        subprocess.run(
            ["git", "clone", "--depth", "1", GEMMA3_4B_REPO, str(clone_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        parts = sorted(clone_dir.glob("gemma.part_*"))
        if not parts:
            raise FileNotFoundError("No gemma.part_* chunks found in cloned repository")
        with raw_gguf.open("wb") as wf:
            for p in parts:
                with p.open("rb") as rf:
                    shutil.copyfileobj(rf, wf, length=16 * 1024 * 1024)
        shutil.rmtree(clone_dir, ignore_errors=True)
        build_abliterated_gemma3_4b(raw_gguf, refusal_cvec, out_gguf, mlp_layers=range(16, 30), gamma=1.00)

    # Clean up any older experimental GGUF files in .cache/neural_graft
    for old in (cache_root / "neural_graft").glob("*.gguf"):
        if old.resolve() != out_gguf.resolve():
            old.unlink(missing_ok=True)
    return out_gguf
