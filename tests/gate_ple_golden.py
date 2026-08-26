#!/usr/bin/env python3
"""Permanent regression gate: PLE (per-layer embedding) row correctness for
gemma-4-E2B (data/models/gemma-4-E2B-it-Q4_0.gguf).

Golden reference (pure numpy, dequant helpers imported from
tests/ref_gemma4_numpy.py): for each token id in [2, 2202, 60000]
    emb  = token_embd_row * sqrt(1536)
    proj = per_layer_model_proj @ emb * (1/sqrt(1536))   -> reshape [35,256]
    proj = per-slice rmsnorm(proj) * per_layer_proj_norm
    PLE  = (proj + per_layer_token_embd_row_slices) * (1/sqrt(2))
    -> [35*256] = [8960]

Engine side: build/dump_logits with TT_DUMP_PLE=<path> makes the kernel dump
d_ple_row (35*256 f32) per embedded position to "<path>.tok<pos>".
One engine run feeds all three tokens; positions 0..2 are compared.

PASS per token iff global median |delta| <= 1e-3 AND every one of the 35
slice RMS ratios ours/reference lies within [0.99, 1.01].

Usage: python3 tests/gate_ple_golden.py [--model PATH] [--median-max 1e-3]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import ref_gemma4_numpy as REF  # noqa: E402  (deq_q4_K / deq_q5_K)

DUMP = "build/dump_logits"
DEFAULT_MODEL = "data/models/gemma-4-E2B-it-Q4_0.gguf"
TOKENS = [2, 2202, 60000]
L, PL_DIM, DIM = 35, 256, 1536
PLE_N = L * PL_DIM                      # 8960
DUMP_PREFIX = "/tmp/ple_gate_eng.bin"

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    os.path.join(os.getcwd(), "oracle/llama.cpp/build/bin"),
    env.get("LD_LIBRARY_PATH", "")]))


def load_tensors(model_path):
    """Tensor directory via gguf-py reader (offsets absolute)."""
    from gguf.gguf_reader import GGUFReader
    rd = GGUFReader(model_path)
    tensors = {}
    for t in rd.tensors:
        dims_ne = tuple(int(d) for d in np.array(t.shape).astype(int))
        tensors[t.name] = (dims_ne, int(t.tensor_type), int(t.data_offset))
    return tensors


def dtype_rowbytes(dt, K):
    return {12: (K // 256) * 144, 13: (K // 256) * 176,
            30: K * 2, 2: (K // 32) * 18, 3: (K // 32) * 20}[dt]


def get_row(tensors, mm, name, r):
    dims, dtype, off = tensors[name]
    K = dims[0]
    rb = dtype_rowbytes(dtype, K)
    raw = lambda n: mm[off + r * rb: off + (r + 1) * rb]
    if dtype == 12:
        return REF.deq_q4_K(raw((K // 256) * 144), K)
    if dtype == 13:
        return REF.deq_q5_K(raw((K // 256) * 176), K)
    if dtype == 30:
        b = np.frombuffer(raw(K * 2), dtype="<u2").astype(np.uint32)
        return np.frombuffer((b << np.uint32(16)).tobytes(), dtype="<f4")
    if dtype == 2:
        nb = K // 32
        arr = np.frombuffer(raw(nb * 18), dtype=np.uint8).reshape(nb, 18)
        d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
        qs = arr[:, 2:].astype(np.int32)
        out = np.empty((nb, 32), dtype=np.float32)
        out[:, :16] = ((qs & 0xF) - 8) * d
        out[:, 16:] = ((qs >> 4) - 8) * d
        return out.reshape(-1)[:K]
    raise ValueError(f"{name}: dtype {dtype}")


def ref_ple(tensors, mm, tok):
    """Golden [L,256] PLE row for one token."""
    x = get_row(tensors, mm, "token_embd.weight", tok).astype(np.float32) \
        * np.sqrt(np.float32(DIM))
    plproj = get_w_full(tensors, mm, "per_layer_model_proj.weight")   # [8960,1536]
    plnorm = get_w_full(tensors, mm, "per_layer_proj_norm.weight").reshape(-1)
    proj = (x @ plproj.T) * (1.0 / np.sqrt(DIM))
    proj = proj.reshape(L, PL_DIM)
    proj = proj / np.sqrt(np.sum(proj * proj, axis=1, keepdims=True) / PL_DIM
                          + 1e-6) * plnorm
    pe = get_row(tensors, mm, "per_layer_token_embd.weight", tok).reshape(L, PL_DIM)
    return ((proj + pe) * (1.0 / np.sqrt(2.0))).astype(np.float32)


def get_w_full(tensors, mm, name):
    """Whole-tensor fetch for small 2-D weights (f32/f16/bf16 enough here)."""
    dims, dtype, off = tensors[name]
    numel = int(np.prod(dims))
    raw = lambda n: mm[off: off + n]
    if dtype == 0:
        return np.frombuffer(raw(numel * 4), dtype="<f4").reshape(dims[::-1]).copy()
    if dtype == 1:
        return np.frombuffer(raw(numel * 2), dtype="<f2").astype(np.float32) \
            .reshape(dims[::-1])
    if dtype == 30:
        b = np.frombuffer(raw(numel * 2), dtype="<u2").astype(np.uint32)
        return np.frombuffer((b << np.uint32(16)).tobytes(), dtype="<f4") \
            .reshape(dims[::-1])
    raise ValueError(f"{name}: dtype {dtype} not supported for whole-tensor fetch")


def engine_ples(model, timeout=300, oom_retries=5):
    """Run dump_logits once over TOKENS; return list of [PLE_N] arrays by pos."""
    for stale in glob.glob(DUMP_PREFIX + ".tok*"):
        os.remove(stale)
    for attempt in range(oom_retries + 1):
        r = subprocess.run([DUMP, "--model", model,
                            ",".join(map(str, TOKENS)), "/dev/null"],
                           capture_output=True, text=True, timeout=timeout,
                           env={**env, "TT_DUMP_PLE": DUMP_PREFIX})
        files = sorted(glob.glob(DUMP_PREFIX + ".tok*"),
                       key=lambda p: int(p.rsplit("tok", 1)[1]))
        if r.returncode == 0 and len(files) >= len(TOKENS):
            break
        wait = 60 if attempt < oom_retries else None   # likely VRAM contention
        print(f"  engine run attempt {attempt+1} failed "
              f"(rc={r.returncode}, dumps={len(files)}); "
              f"retrying in {wait}s" if wait else
              f"  engine run failed permanently (rc={r.returncode})")
        if wait is None:
            raise RuntimeError("dump_logits failed:\n" +
                               "\n".join(r.stderr.splitlines()[-5:]))
        time.sleep(wait)
    out = []
    for i in range(len(TOKENS)):
        a = np.fromfile(f"{DUMP_PREFIX}.tok{i}", dtype="<f4")
        if a.size != PLE_N:
            raise RuntimeError(f"tok{i}: expected {PLE_N} floats, got {a.size}")
        out.append(a.astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--median-max", type=float, default=1e-3)
    args = ap.parse_args()

    print(f"gate ple-golden [{os.path.basename(args.model)}] "
          f"tokens={TOKENS} (median|d|<={args.median_max:g}, "
          f"slice-RMS ratio in [0.99,1.01])")

    tensors = load_tensors(args.model)
    import mmap
    f = open(args.model, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

    refs = [ref_ple(tensors, mm, t) for t in TOKENS]
    ours_all = engine_ples(args.model)

    header = (f"{'token':>7s} {'med|d|':>10s} {'minR':>7s} {'maxR':>7s} {'result':>7s}")
    print(header)
    print("-" * len(header))

    npass = 0
    for i, tok in enumerate(TOKENS):
        ours = ours_all[i].reshape(L, PL_DIM)
        ref = refs[i].astype(np.float64)
        med = float(np.median(np.abs(ours.reshape(-1) - ref.reshape(-1))))
        rms_o = np.sqrt(np.mean(ours * ours, axis=1))
        rms_r = np.sqrt(np.mean(ref * ref, axis=1))
        ratio = rms_o / rms_r
        ok = med <= args.median_max and ratio.min() >= 0.99 and ratio.max() <= 1.01
        npass += ok
        print(f"{tok:7d} {med:10.3e} {ratio.min():7.4f} {ratio.max():7.4f} "
              f"{'PASS' if ok else 'FAIL':>7s}")

    print(f"\ngate ple-golden: {npass}/{len(TOKENS)} pass")
    sys.exit(0 if npass == len(TOKENS) else 1)


if __name__ == "__main__":
    main()
