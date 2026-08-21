#!/usr/bin/env python3
"""M2 gate: single-threaded AVX2 matmul >= NumPy float32 on 1024^3.

Protocol (PLAN rule 5): warmup 5, median of 20 runs.
NumPy is pinned to 1 OpenBLAS thread for the fair single-thread comparison.
Writes the speedup table to bench/results.md.
"""
import os

# must precede numpy import; do not add OMP_NUM_THREADS (our OMP kernel
# sets thread counts explicitly per call)
for var in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[var] = "1"

import ctypes
import datetime
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests"))
from torch_py import LIB, Tensor  # noqa: E402

SIZES = [256, 512, 1024]
WARMUP = 5
RUNS = 20
GATE_SIZE = 1024
OMP_THREADS = 6  # physical cores; SMT typically hurts sgemm


def np_threads_set(n):
    """Best-effort OpenBLAS thread control via ctypes."""
    try:
        lib = ctypes.CDLL(None)
        for sym in ("openblas_set_num_threads", "openblas_set_num_threads64_"):
            if hasattr(lib, sym):
                getattr(lib, sym)(n)
                return True
    except Exception:
        pass
    try:
        import glob
        import site
        cands = []
        try:
            sps = list(site.getsitepackages())
        except Exception:
            sps = []
        sps.append(site.getusersitepackages())
        sps += [os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")]
        for sp in sps:
            cands += glob.glob(f"{sp}/numpy.libs/libscipy_openblas*.so")
            cands += glob.glob(f"{sp}/scipy_openblas*/lib/*openblas*.so")
        for path in cands:
            try:
                lib = ctypes.CDLL(path)
                for sym in ("openblas_set_num_threads",
                            "openblas_set_num_threads64_"):
                    if hasattr(lib, sym):
                        getattr(lib, sym)(n)
                        return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def time_call(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_impl(name, c_fn, a, b, ref, omp_threads=None):
    ta, tb = Tensor.from_np(a), Tensor.from_np(b)
    if omp_threads:
        out = LIB.tt_matmul_omp(ta.ptr, tb.ptr, omp_threads)
        got = Tensor(out).to_np()
        call = lambda: LIB.tt_matmul_omp(ta.ptr, tb.ptr, omp_threads)
    else:
        out = c_fn(ta.ptr, tb.ptr)
        got = Tensor(out).to_np()
        call = lambda: c_fn(ta.ptr, tb.ptr)
    if not np.allclose(got, ref, atol=1e-3, rtol=1e-3):
        print(f"CORRECTNESS FAIL: {name} max_err="
              f"{np.abs(got - ref).max():.3e}")
        sys.exit(1)
    dt = time_call(call)
    ta.release()
    tb.release()
    return dt


def main():
    gate = "--gate" in sys.argv
    rng = np.random.default_rng(0)
    rows = []
    gate_ok = None

    for n in SIZES:
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)
        ref = a @ b
        flops = 2.0 * n ** 3

        t_naive = bench_impl("naive", LIB.tt_matmul, a, b, ref)
        t_avx2 = bench_impl("avx2", LIB.tt_matmul_fast, a, b, ref)
        t_omp = bench_impl("omp", None, a, b, ref, omp_threads=OMP_THREADS)

        np_threads_set(1)
        t_np1 = time_call(lambda: a @ b)
        mt_ok = np_threads_set(os.cpu_count())
        t_npmt = time_call(lambda: a @ b) if mt_ok else float("nan")
        np_threads_set(1)

        row = {
            "n": n,
            "naive": flops / t_naive / 1e9,
            "avx2": flops / t_avx2 / 1e9,
            "omp": flops / t_omp / 1e9,
            "np1": flops / t_np1 / 1e9,
            "npmt": flops / t_npmt / 1e9,
        }
        rows.append(row)
        if n == GATE_SIZE:
            gate_ok = row["avx2"] >= row["np1"]

    cpu = subprocess.run(
        ["sh", "-c", "lscpu | grep 'Model name' | cut -d: -f2"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    try:
        clocks = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip()
    except Exception:
        clocks = "n/a"

    lines = [
        "# CPU matmul benchmark results",
        "",
        f"- Date: {datetime.date.today()}",
        f"- CPU: {cpu.strip()} ({os.cpu_count()} logical cores)",
        f"- GPU clocks (context): {clocks}",
        f"- Protocol: warmup {WARMUP}, median of {RUNS} runs, fp32",
        "- NumPy backend: OpenBLAS (pinned to 1 thread for np-1t row)",
        f"- OMP kernel threads: {OMP_THREADS}",
        "",
        "| N | naive | AVX2 1T | AVX2 OMP | NumPy 1T | NumPy MT |"
        " AVX2/naive | AVX2/NumPy1T |",
        "|---|-------|---------|----------|----------|----------|"
        "------------|--------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['naive']:.1f} | {r['avx2']:.1f} "
            f"| {r['omp']:.1f} | {r['np1']:.1f} | {r['npmt']:.1f} "
            f"| {r['avx2']/r['naive']:.1f}x | {r['avx2']/r['np1']:.2f}x |")
    lines += ["", f"GFLOPS. Gate (AVX2 1T >= NumPy 1T @ {GATE_SIZE}^3): "
              f"{'PASS' if gate_ok else 'FAIL'}", ""]
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    if gate:
        if gate_ok:
            print(f"GATE GREEN: AVX2 1-thread ({rows[-1]['avx2']:.1f} GFLOPS)"
                  f" >= NumPy 1-thread ({rows[-1]['np1']:.1f} GFLOPS) "
                  f"@ {GATE_SIZE}^3")
            return 0
        print(f"GATE RED: AVX2 1-thread ({rows[-1]['avx2']:.1f}) < "
              f"NumPy 1-thread ({rows[-1]['np1']:.1f}) @ {GATE_SIZE}^3")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
