#!/usr/bin/env python3
"""M11 gate: CPU quant-aware GEMV backend (Q4_0 / Q8_0).

Correctness: real tensors from data/models/qwen2.5-0.5b-instruct-q4_0.gguf
(parsed with the same minimal GGUF reader as ref_qwen2_numpy.py), random x,
compare tt_cpu_gemv output vs numpy dot of dequantized weights.
Tolerance: fp32 accumulation -> max|Δ| / ||y_ref||∞ < 1e-2 (naive pointwise
rel is dominated by cancellation rows near zero and is reported as med_rel
over non-degenerate rows instead).

Benchmark: M=11008 K=1536 synthetic, threads=1 vs OMP thread counts,
effective GB/s = W bytes read per call.

Compile line (see include/cpu_backend.h):
  gcc -O3 -mavx2 -mfma -fopenmp -std=c11 -Iinclude \
      -DCPU_BACKEND_MAIN -o build/cpu_backend src/cpu_backend.c -lm
"""
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_qwen2_numpy import load_gguf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build/cpu_backend")
MODEL = os.path.join(ROOT, "data/models/qwen2.5-0.5b-instruct-q4_0.gguf")

TTQ_Q4_0, TTQ_Q8_0 = 2, 8
BLOCK_BYTES = {TTQ_Q4_0: 18, TTQ_Q8_0: 34}
DTYPE_NAMES = {TTQ_Q4_0: "q4_0", TTQ_Q8_0: "q8_0"}

failures = []


def run_gemv(w_raw, dtype, M, K, threads, x):
    wb = os.path.join("/tmp", f"cb_w_{os.getpid()}.bin")
    xb = os.path.join("/tmp", f"cb_x_{os.getpid()}.bin")
    yb = os.path.join("/tmp", f"cb_y_{os.getpid()}.bin")
    open(wb, "wb").write(w_raw)
    x.astype("<f4").tofile(xb)
    r = subprocess.run(
        [BIN, wb, str(dtype), str(M), str(K), str(threads), xb, yb],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpu_backend failed: {r.stderr}")
    y = np.fromfile(yb, dtype="<f4")
    os.unlink(wb); os.unlink(xb); os.unlink(yb)
    return y, r.stdout.strip()


def check(name, y_c, y_ref):
    """Gate on max|Δ| scaled by ||y_ref||inf (cancellation rows make naive
    pointwise rel meaningless for vocab-size GEMV); also report median rel."""
    scale = max(float(np.abs(y_ref).max()), 1e-6)
    abs_err = np.abs(y_c - y_ref)
    scaled = float(abs_err.max()) / scale
    nz = np.abs(y_ref) > 1e-3 * scale
    med_rel = float(np.median(abs_err[nz] / np.abs(y_ref)[nz]))
    ok = scaled < 1e-2
    print(f"  {name:34s} max|Δ|/||y||∞={scaled:.2e} med_rel={med_rel:.2e}"
          f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def raw_tensor(mm, base, off, dtype, numel):
    bs = BLOCK_BYTES[dtype]
    return mm[base + off : base + off + (numel // 32) * bs]


def correctness():
    print("== correctness vs numpy reference (real qwen tensors) ==")
    kv, tensors, mm, base = load_gguf(MODEL)
    rng = np.random.default_rng(42)

    cases = [
        ("ffn_down.weight", "blk.0.ffn_down.weight"),   # K=1536 -> M hidden? see dims
        ("attn_q.weight",   "blk.0.attn_q.weight"),
        ("output.weight",   "output.weight"),
        ("ffn_gate.weight", "blk.1.ffn_gate.weight"),
    ]
    for label, tname in cases:
        if tname not in tensors:
            print(f"  {label}: {tname} not in file, skipped")
            continue
        dims, dtype, off = tensors[tname]
        # GGUF stores [K, M] nelements; rows are the last-listed dim
        K, M = int(dims[0]), int(dims[1])
        numel = K * M
        w_raw = raw_tensor(mm, base, off, dtype, numel)
        W = np.frombuffer(w_raw, dtype=np.uint8)
        # dequantize via ref module for the numpy golden
        if dtype == TTQ_Q4_0:
            from ref_qwen2_numpy import dequant_q4_0
            Wd = dequant_q4_0(bytes(w_raw), numel).reshape(M, K)
        elif dtype == TTQ_Q8_0:
            from ref_qwen2_numpy import dequant_q8_0
            Wd = dequant_q8_0(bytes(w_raw), numel).reshape(M, K)
        else:
            print(f"  {label}: dtype {dtype} out of scope, skipped")
            continue
        x = rng.standard_normal(K).astype(np.float32)
        y_ref = Wd @ x
        y_c, _ = run_gemv(np.frombuffer(w_raw, dtype=np.uint8),
                          dtype, M, K, 1, x)
        assert y_c.shape == (M,), y_c.shape
        check(f"{label} ({DTYPE_NAMES[dtype]} {M}x{K})", y_c, y_ref)


def benchmark():
    print("== benchmark: M=11008 K=1536 synthetic, best-of-20 ==")
    M, K = 11008, 1536
    rng = np.random.default_rng(7)
    x = rng.standard_normal(K).astype(np.float32)
    for dtype in (TTQ_Q4_0, TTQ_Q8_0):
        bs = BLOCK_BYTES[dtype]
        nbytes = M * (K // 32) * bs
        w_raw = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
        print(f" {DTYPE_NAMES[dtype]} ({nbytes/1e6:.1f} MB of W):")
        for th in (1, 2, 4, 8):
            _, line = run_gemv(w_raw, dtype, M, K, th, x)
            print(f"   {line}")


if __name__ == "__main__":
    if not os.path.exists(BIN):
        sys.exit("build/cpu_backend missing — compile line in include/cpu_backend.h")
    correctness()
    benchmark()
    if failures:
        sys.exit(f"FAILED: {failures}")
    print("all CPU GEMV gates passed")
