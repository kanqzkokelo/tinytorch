#!/usr/bin/env python3
"""Phase-2 gate: CPU quant-aware GEMV backend
(Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K, scalar + AVX2 dispatch).

Correctness:
  - legacy: real tensors from data/models/qwen2.5-0.5b-instruct-q4_0.gguf
    (parsed with the same minimal GGUF reader as ref_qwen2_numpy.py),
    numpy dot of dequantized weights as golden.
  - K-quants: real tensors from smollm2-135m-instruct-Q{4,5,6}_K.gguf,
    golden = build/dequant_ref CLI (itself validated against gguf-py by
    tests/test_dequant_golden.py).
Tolerance: fp32 accumulation -> max|delta| / ||y_ref||inf < 1e-2 (naive
pointwise rel is dominated by cancellation rows; reported as med_rel).

Path agreement: AVX2 vs CPU_BACKEND_SCALAR=1 on identical inputs must agree
to max|delta|/||y||inf < 1e-5 and med_rel < 1e-6 (pure accumulation-order
differences).

Benchmark: M=11008 K=1536 synthetic (valid fp16 scales, random payloads),
threads=1/4/8, effective GB/s = W bytes read per call.

Compile line (see include/cpu_backend.h):
  gcc -O3 -mavx2 -mfma -fopenmp -std=c11 -Iinclude \
      -DCPU_BACKEND_MAIN -o build/cpu_backend src/cpu_backend.c -lm
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_qwen2_numpy import load_gguf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build/cpu_backend")
DQBINDIR = os.path.join(ROOT, "build/dequant_ref")
MODEL = os.path.join(ROOT, "data/models/qwen2.5-0.5b-instruct-q4_0.gguf")

TTQ_Q4_0, TTQ_Q8_0 = 2, 8
TTQ_Q2_K, TTQ_Q3_K = 10, 11
TTQ_Q4_K, TTQ_Q5_K, TTQ_Q6_K = 12, 13, 14
BLOCK_BYTES = {TTQ_Q4_0: 18, TTQ_Q8_0: 34, TTQ_Q2_K: 84, TTQ_Q3_K: 110,
               TTQ_Q4_K: 144, TTQ_Q5_K: 176, TTQ_Q6_K: 210}
DTYPE_NAMES = {TTQ_Q4_0: "q4_0", TTQ_Q8_0: "q8_0", TTQ_Q2_K: "q2_k", TTQ_Q3_K: "q3_k",
               TTQ_Q4_K: "q4_k", TTQ_Q5_K: "q5_k", TTQ_Q6_K: "q6_k"}
ALL_DTYPES = (TTQ_Q4_0, TTQ_Q8_0, TTQ_Q2_K, TTQ_Q3_K, TTQ_Q4_K, TTQ_Q5_K, TTQ_Q6_K)

# K-quant golden sources: (model file, tensor) — tensor dtype verified below.
K_MODELS = {
    TTQ_Q4_K: ("smollm2-135m-instruct-Q4_K.gguf", "blk.3.ffn_down.weight"),
    TTQ_Q5_K: ("smollm2-135m-instruct-Q5_K.gguf", "blk.3.ffn_down.weight"),
    TTQ_Q6_K: ("smollm2-135m-instruct-Q6_K.gguf", "blk.0.ffn_down.weight"),
}

failures = []


def run_gemv(w_raw, dtype, M, K, threads, x, force_scalar=False):
    wb = os.path.join("/tmp", f"cb_w_{os.getpid()}.bin")
    xb = os.path.join("/tmp", f"cb_x_{os.getpid()}.bin")
    yb = os.path.join("/tmp", f"cb_y_{os.getpid()}.bin")
    open(wb, "wb").write(w_raw)
    x.astype("<f4").tofile(xb)
    env = dict(os.environ)
    if force_scalar:
        env["CPU_BACKEND_SCALAR"] = "1"
    else:
        env.pop("CPU_BACKEND_SCALAR", None)
    r = subprocess.run(
        [BIN, wb, str(dtype), str(M), str(K), str(threads), xb, yb],
        capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"cpu_backend failed: {r.stderr}")
    y = np.fromfile(yb, dtype="<f4")
    os.unlink(wb); os.unlink(xb); os.unlink(yb)
    return y, r.stdout.strip()


def check(name, y_c, y_ref, tol=1e-2, med_tol=None):
    """Gate on max|Δ| scaled by ||y_ref||inf (cancellation rows make naive
    pointwise rel meaningless for vocab-size GEMV); also report median rel."""
    scale = max(float(np.abs(y_ref).max()), 1e-6)
    abs_err = np.abs(y_c - y_ref)
    scaled = float(abs_err.max()) / scale
    nz = np.abs(y_ref) > 1e-3 * scale
    med_rel = float(np.median(abs_err[nz] / np.abs(y_ref)[nz]))
    ok = scaled < tol and (med_tol is None or med_rel < med_tol)
    print(f"  {name:40s} max|Δ|/||y||∞={scaled:.2e} med_rel={med_rel:.2e}"
          f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def raw_tensor(mm, base, off, dtype, numel):
    bs = BLOCK_BYTES[dtype]
    return mm[base + off : base + off + (numel // 32) * bs]


def dequant_golden(gguf_path, tensor_name, numel):
    """fp32 golden via build/dequant_ref CLI (validated vs gguf-py elsewhere)."""
    out_bin = f"/tmp/cb_gold_{os.getpid()}.bin"
    r = subprocess.run([DQBINDIR, gguf_path, tensor_name, out_bin],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dequant_ref failed: {r.stderr}")
    w = np.fromfile(out_bin, dtype="<f4")
    os.unlink(out_bin)
    assert w.size == numel, (w.size, numel)
    return w


def correctness_legacy():
    print("== correctness vs numpy reference (legacy quants, qwen tensors) ==")
    kv, tensors, mm, base = load_gguf(MODEL)
    rng = np.random.default_rng(42)

    cases = [
        ("ffn_down.weight", "blk.0.ffn_down.weight"),
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


def correctness_kquants():
    print("== correctness vs dequant_ref golden (K-quants, smollm2 tensors) ==")
    rng = np.random.default_rng(43)
    for dtype, (fname, tname) in K_MODELS.items():
        path = os.path.join(ROOT, "data/testmodels", fname)
        if not os.path.exists(path):
            print(f"  {DTYPE_NAMES[dtype]}: {fname} missing, skipped")
            continue
        _, tensors, mm, base = load_gguf(path)
        if tname not in tensors:
            print(f"  {DTYPE_NAMES[dtype]}: {tname} not in file, skipped")
            continue
        dims, tdtype, off = tensors[tname]
        if tdtype != dtype:
            print(f"  {DTYPE_NAMES[dtype]}: {tname} is "
                  f"{DTYPE_NAMES.get(tdtype, tdtype)}, skipped")
            continue
        K, M = int(dims[0]), int(dims[1])
        numel = K * M
        assert K % 256 == 0, f"K={K} not a multiple of 256"
        w_raw = raw_tensor(mm, base, off, dtype, numel)
        Wd = dequant_golden(path, tname, numel).reshape(M, K)
        x = rng.standard_normal(K).astype(np.float32)
        y_ref = Wd @ x
        y_c, _ = run_gemv(np.frombuffer(w_raw, dtype=np.uint8),
                          dtype, M, K, 1, x)
        check(f"{fname.replace('smollm2-135m-instruct-', '')}: {tname} "
              f"({M}x{K})", y_c, y_ref)


def synth_weights(dtype, M, K, seed):
    """Random payload but valid fp16 scale fields (avoid subnormal slowdowns)."""
    rng = np.random.default_rng(seed)
    bs = BLOCK_BYTES[dtype]
    row_vals = 256 if bs >= 84 else 32
    nbytes = M * (K // row_vals) * bs
    w = rng.integers(0, 256, size=nbytes, dtype=np.uint8).reshape(-1, bs).copy()
    one16 = np.uint16(np.float16(1.0).view(np.uint16))
    le = np.array([one16], dtype="<u2").view(np.uint8)  # LE bytes of 1.0h
    if dtype in (TTQ_Q4_0, TTQ_Q8_0, TTQ_Q4_K, TTQ_Q5_K):
        nsc = 2 if bs >= 144 else 1
        w[:, : 2 * nsc] = np.tile(le, nsc)
    elif dtype == TTQ_Q2_K:
        w[:, 80:84] = np.tile(le, 2)
    elif dtype == TTQ_Q3_K:
        w[:, 108:110] = le
    else:  # q6_K: d at offset 208
        w[:, 208:210] = le
    return w.reshape(-1)

def path_agreement():
    print("== AVX2 vs scalar agreement (same bytes, 1 thread) ==")
    rng = np.random.default_rng(44)
    for dtype in ALL_DTYPES:
        M, K = 512, 1536
        w_raw = synth_weights(dtype, M, K, 100 + dtype)
        x = rng.standard_normal(K).astype(np.float32)
        y_avx, line_avx = run_gemv(w_raw, dtype, M, K, 1, x,
                                   force_scalar=False)
        y_sc, _ = run_gemv(w_raw, dtype, M, K, 1, x, force_scalar=True)
        tag = "avx2" if "avx2" in line_avx else "scalar?"
        check(f"path agreement {DTYPE_NAMES[dtype]:5s} [{tag}]",
              y_avx, y_sc, tol=1e-5, med_tol=1e-6)


def benchmark():
    print("== benchmark: M=11008 K=1536 synthetic, best-of-20, GB/s ==")
    M, K = 11008, 1536
    rng = np.random.default_rng(7)
    x = rng.standard_normal(K).astype(np.float32)
    header = f"  {'dtype':6s} {'bytes':>9s}"
    results = {}
    for dtype in ALL_DTYPES:
        w_raw = synth_weights(dtype, M, K, 200 + dtype)
        nbytes = w_raw.nbytes
        row = {}
        for th in (1, 4, 8):
            _, line = run_gemv(w_raw, dtype, M, K, th, x)
            gbs = float(line.split()[-2])
            row[th] = (line, gbs)
        results[dtype] = row
        print(f"  {DTYPE_NAMES[dtype]:6s} {nbytes/1e6:7.1f}MB")
        for th in (1, 4, 8):
            print(f"    T{th}: {row[th][0]}")
    # scaling summary table
    print("  scaling table (GB/s):")
    print(f"  {'dtype':6s} {'1T':>8s} {'4T':>8s} {'8T':>8s} {'x(1->8T)':>9s}")
    for dtype in ALL_DTYPES:
        r = results[dtype]
        print(f"  {DTYPE_NAMES[dtype]:6s} {r[1][1]:8.2f} {r[4][1]:8.2f} "
              f"{r[8][1]:8.2f} {r[8][1]/max(r[1][1],1e-9):8.2f}x")
    target = results[TTQ_Q4_0][8][1]
    print(f"  target q4_0 >= 8 GB/s @8T: {target:.2f} "
          f"{'PASS' if target >= 8.0 else 'FAIL'}")
    if target < 8.0:
        failures.append("perf-q4_0-8GBps@8T")


if __name__ == "__main__":
    if not os.path.exists(BIN):
        sys.exit("build/cpu_backend missing — compile line in include/cpu_backend.h")
    correctness_legacy()
    correctness_kquants()
    path_agreement()
    benchmark()
    if failures:
        sys.exit(f"FAILED: {failures}")
    print("all CPU GEMV gates passed")
