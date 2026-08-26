#!/usr/bin/env python3
"""GEMM optimization ladder bench (M2 kernel vs OpenBLAS reference).

Protocol: warmup 2, median of 5 runs. Correctness gate at every shape:
max relative error vs float64 numpy <= 1e-4.
Reference: numpy's OpenBLAS (reference only, never linked into our code).
Reports load average with each measurement (other agents may use the CPU).
"""
import os

for var in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[var] = "1"

import ctypes
import math
import os
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests"))
import torch_py  # noqa: E402  (builds the lib)
from torch_py import LIB, Tensor  # noqa: E402

# (label, M, N, K): squares 512..2048 + skinny prefill-realistic shapes
SHAPES = [
    ("sq512", 512, 512, 512),
    ("sq1024", 1024, 1024, 1024),
    ("sq1536", 1536, 1536, 1536),
    ("sq2048", 2048, 2048, 2048),
    ("skinny-a", 1536, 512, 512),   # many tokens x small layer
    ("skinny-b", 512, 1536, 1536),  # few tokens x big layer
]
WARMUP = 2
RUNS = 5
OMP_THREADS = 6
REL_TOL = 1e-4


def np_threads_set(n):
    try:
        lib = ctypes.CDLL(None)
        for sym in ("openblas_set_num_threads", "openblas_set_num_threads64_"):
            if hasattr(lib, sym):
                getattr(lib, sym)(n)
                return True
    except Exception:
        pass
    return False


def load_avg():
    with open("/proc/loadavg") as f:
        parts = f.read().split()
    return f"{parts[0]}/{os.cpu_count()}"


def time_call(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def rel_err(got, ref64):
    denom = np.abs(ref64).max()
    return np.abs(got - ref64).max() / max(denom, 1e-30)


def main():
    rng = np.random.default_rng(0)
    print(f"load={load_avg()}  threads={os.cpu_count()}")
    hdr = (f"| {'shape':10s} | {'ours 1T':>8s} | {'ours MP':>8s} | "
           f"{'OBAS 1T':>8s} | {'OBAS MT':>8s} | {'vs OB 1T':>8s} | "
           f"{'relerr':>8s} |")
    print(hdr)
    print("|" + "---|" * 7)

    np_threads_set(1)
    results = []
    for label, M, N, K in SHAPES:
        a = rng.standard_normal((M, K)).astype(np.float32)
        b = rng.standard_normal((K, N)).astype(np.float32)
        ref64 = a.astype(np.float64) @ b.astype(np.float64)
        flops = 2.0 * M * N * K

        ta, tb = Tensor.from_np(a), Tensor.from_np(b)

        # correctness gate (1T path)
        out = LIB.tt_matmul_fast(ta.ptr, tb.ptr)
        got = Tensor(out).to_np()
        re = rel_err(got, ref64)
        if re > REL_TOL:
            print(f"CORRECTNESS FAIL {label}: rel err {re:.3e}")
            sys.exit(1)

        t_1t = time_call(lambda: LIB.tt_matmul_fast(ta.ptr, tb.ptr))
        t_mp = time_call(lambda: LIB.tt_matmul_omp(ta.ptr, tb.ptr,
                                                   OMP_THREADS))
        ta.release(); tb.release()

        t_np1 = time_call(lambda: a @ b)
        np_threads_set(os.cpu_count())
        t_npmt = time_call(lambda: a @ b)
        np_threads_set(1)

        g1t, gmp = flops / t_1t / 1e9, flops / t_mp / 1e9
        gn1 = flops / t_np1 / 1e9
        gnm = flops / t_npmt / 1e9
        results.append((label, M, N, K, g1t, gmp, gn1, gnm))
        print(f"| {label:10s} | {g1t:8.1f} | {gmp:8.1f} | {gn1:8.1f} | "
              f"{gnm:8.1f} | {g1t/gn1:7.2f}x | {re:7.1e} |")

    gm = math.prod(r[4] for r in results) ** (1 / len(results))
    gm_sq = math.prod(r[4] for r in results[:4]) ** (1 / 4)
    gm_ref = math.prod(r[6] for r in results) ** (1 / len(results))
    print(f"\ngeomean ours-1T {gm:.1f} GFLOPS (squares {gm_sq:.1f}), "
          f"ref-1T geomean {gm_ref:.1f}, ratio {gm/gm_ref:.2f}x; "
          f"load end {load_avg()}")


if __name__ == "__main__":
    main()
