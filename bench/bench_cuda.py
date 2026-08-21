#!/usr/bin/env python3
"""M3 Gate B: tiled >= 10x naive @1024^3; report % of cuBLAS sgemm.

Protocol (PLAN rule 5): warmup 5, median of 20 runs.
"""
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.run(["make", "-s", "cuda"], cwd=ROOT, check=True)

lib = ctypes.CDLL(os.path.join(ROOT, "build", "libtinytorch_cuda.so"))
lib.tt_cuda_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.tt_cuda_alloc.restype = ctypes.c_int
lib.tt_cuda_free.argtypes = [ctypes.c_void_p]
lib.tt_cuda_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.tt_cuda_h2d.restype = ctypes.c_int
lib.tt_cuda_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.tt_cuda_d2h.restype = ctypes.c_int
for name in ("tt_cuda_sgemm_naive", "tt_cuda_sgemm_tiled",
             "tt_cuda_sgemm_wmma"):
    f = getattr(lib, name)
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3

cl = ctypes.CDLL(os.path.join(ROOT, 'build', 'libtt_cublas.so'))
cl.tt_cublas_ref.restype = ctypes.c_int
cl.tt_cublas_ref.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3

WARMUP, RUNS = 5, 20
SIZES = tuple(int(x) for x in
    os.environ.get("TT_CUDA_SIZES", "512,1024,2048").split(","))


def timed(fn):
    def once():
        rc = fn()
        if rc != 0:
            return rc
        return lib.tt_cuda_sync()
    for _ in range(WARMUP):
        rc = once()
        if rc != 0:
            return None
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        rc = once()
        dt = time.perf_counter() - t0
        if rc != 0:
            return None
        ts.append(dt)
    return statistics.median(ts)


def bench_size(M, N, K, use_cublas=True):
    rng = np.random.default_rng(1)
    a = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
    b = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
    dA, dB, dC = (ctypes.c_void_p() for _ in range(3))
    assert lib.tt_cuda_alloc(ctypes.byref(dA), a.nbytes) == 0
    assert lib.tt_cuda_alloc(ctypes.byref(dB), b.nbytes) == 0
    assert lib.tt_cuda_alloc(ctypes.byref(dC), M * N * 4) == 0
    assert lib.tt_cuda_h2d(dA, a.ctypes.data, a.nbytes) == 0
    assert lib.tt_cuda_h2d(dB, b.ctypes.data, b.nbytes) == 0
    try:
        res = {}
        for name in ("tt_cuda_sgemm_naive", "tt_cuda_sgemm_tiled",
                     "tt_cuda_sgemm_wmma"):
            fn = lambda n=name: getattr(lib, n)(dA, dB, dC, M, N, K)
            t = timed(fn)
            res[name] = 2.0 * M * N * K / t / 1e9 if t else float("nan")
        if use_cublas:
            def cb():
                return cl.tt_cublas_ref(dA, dB, dC, M, N, K)
            t = timed(cb)
            ok = False
            if t:
                out = np.empty((M, N), dtype=np.float32)
                assert lib.tt_cuda_d2h(out.ctypes.data, dC, M * N * 4) == 0
                ok = np.allclose(out, a @ b, atol=1e-2, rtol=1e-2)
            res["cublas"] = 2.0 * M * N * K / t / 1e9 if (t and ok) \
                else float("nan")
        else:
            res["cublas"] = float("nan")
        return res
    finally:
        lib.tt_cuda_free(dA)
        lib.tt_cuda_free(dB)
        lib.tt_cuda_free(dC)


def main():
    gate = "--gate" in sys.argv
    worker = "--worker" in sys.argv
    if worker:
        n = SIZES[0]
        print(json.dumps(bench_size(n, n, n)))
        return 0
    rows = []
    gate_ok = None
    # per-size subprocess isolation: one wedged context cannot kill the rest
    for n in SIZES:
        env = dict(os.environ, TT_CUDA_SIZES=str(n))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"],
                 env=env, capture_output=True, text=True)
        try:
            j = json.loads(p.stdout[p.stdout.index("{"):])
        except Exception:
            print(f"size {n}: worker failed:\n{p.stderr[-400:]}")
            j = {k: float("nan") for k in
                 ("tt_cuda_sgemm_naive", "tt_cuda_sgemm_tiled",
                  "tt_cuda_sgemm_wmma", "cublas")}
        rows.append((n, j))
        if n == 1024 and all(np.isfinite(j.get(k, float("nan")))
                             for k in ("tt_cuda_sgemm_tiled",
                                       "tt_cuda_sgemm_naive")):
            gate_ok = j["tt_cuda_sgemm_tiled"] >= 10.0 * \
                j["tt_cuda_sgemm_naive"]

    lines = ["", "## CUDA matmul ladder (RTX 3050 laptop, sm_86)",
             "",
             "| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | "
             "tiled %cuBLAS | wmma %cuBLAS |",
             "|---|-------|-------|-----------|--------|-------------|"
             "--------------|---------------|"]
    for n, r in rows:
        tb = r["tt_cuda_sgemm_tiled"]
        cb = r["cublas"]
        lines.append(
            f"| {n} | {r['tt_cuda_sgemm_naive']:.1f} | {tb:.1f} "
            f"| {r['tt_cuda_sgemm_wmma']:.1f} | {cb:.1f} "
            f"| {tb/r['tt_cuda_sgemm_naive']:.1f}x "
            f"| {100*tb/cb:.1f}% | {100*r['tt_cuda_sgemm_wmma']/cb:.1f}% |")
    lines.append("")
    with open(os.path.join(ROOT, "bench", "results.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if gate:
        g1024 = [r for n_, r in rows if n_ == 1024]
        if not g1024 or not np.isfinite(
                g1024[0].get("tt_cuda_sgemm_naive", float("nan"))):
            print("GATE B INCONCLUSIVE: no valid 1024^3 row")
            return 2
        nb = g1024[0]["tt_cuda_sgemm_naive"]
        tb = g1024[0]["tt_cuda_sgemm_tiled"]
        if gate_ok:
            print(f"GATE B GREEN: tiled {tb:.0f} >= 10x naive {nb:.0f} "
                  f"GFLOPS @1024^3")
            return 0
        print(f"GATE B RED: tiled {tb:.0f} < 10x naive {nb:.0f} GFLOPS "
              f"@1024^3")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
