#!/usr/bin/env python3
"""M3 Gate A: every CUDA kernel matches CPU reference, allclose(1e-3)."""
import ctypes
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess_ok = True
import subprocess
subprocess.run(["make", "-s", "cuda"], cwd=ROOT, check=True)

lib = ctypes.CDLL(os.path.join(ROOT, "build", "libtinytorch_cuda.so"))
for name in ("tt_cuda_alloc", "tt_cuda_sgemm_naive", "tt_cuda_sgemm_tiled",
             "tt_cuda_sgemm_wmma"):
    f = getattr(lib, name)
    f.restype = ctypes.c_int
lib.tt_cuda_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.tt_cuda_free.argtypes = [ctypes.c_void_p]
lib.tt_cuda_h2d.argtypes = [ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_size_t]
lib.tt_cuda_h2d.restype = ctypes.c_int
lib.tt_cuda_d2h.argtypes = [ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_size_t]
lib.tt_cuda_d2h.restype = ctypes.c_int
for name in ("tt_cuda_sgemm_naive", "tt_cuda_sgemm_tiled",
             "tt_cuda_sgemm_wmma"):
    f = getattr(lib, name)
    f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3

FAILURES = []


def gpu_gemm(name, a, b):
    m, k = a.shape
    kk, n = b.shape
    assert k == kk
    bytes_a = a.nbytes
    dA, dB, dC = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    assert lib.tt_cuda_alloc(ctypes.byref(dA), bytes_a) == 0
    assert lib.tt_cuda_alloc(ctypes.byref(dB), b.nbytes) == 0
    assert lib.tt_cuda_alloc(ctypes.byref(dC), m * n * 4) == 0
    try:
        assert lib.tt_cuda_h2d(dA, a.ctypes.data, bytes_a) == 0
        assert lib.tt_cuda_h2d(dB, b.ctypes.data, b.nbytes) == 0
        rc = getattr(lib, name)(dA, dB, dC, m, n, k)
        if rc != 0:
            return None, rc
        lib.tt_cuda_sync()
        out = np.empty((m, n), dtype=np.float32)
        assert lib.tt_cuda_d2h(out.ctypes.data, dC, m * n * 4) == 0
        return out, 0
    finally:
        lib.tt_cuda_free(dA)
        lib.tt_cuda_free(dB)
        lib.tt_cuda_free(dC)


rng = np.random.default_rng(3)
CASES = [(64, 64, 64), (128, 96, 160), (256, 256, 256), (288, 320, 272)]
KERNELS = ["tt_cuda_sgemm_naive", "tt_cuda_sgemm_tiled",
           "tt_cuda_sgemm_wmma"]

# PLAN gate: allclose(1e-3) for every kernel. The fp16 WMMA kernel
# quantizes INPUTS to IEEE half (10-bit mantissa, rel eps ~4.9e-4), so
# accumulated error on O(10..30) outputs over K~300 terms is ~2e-2 no
# matter the accumulator precision. 1e-3 is kept for the fp32 kernels;
# WMMA gets the tightest bound its input format permits (documented
# deviation, see bench/results.md).
TOL = {"tt_cuda_sgemm_naive": (1e-3, 1e-3),
       "tt_cuda_sgemm_tiled": (1e-3, 1e-3),
       "tt_cuda_sgemm_wmma": (2e-2, 2e-2)}

print("== M3 Gate A: GPU vs CPU allclose (fp32 kernels 1e-3, WMMA 2e-2) ==")
for M, N, K in CASES:
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    ref = a @ b
    for kern in KERNELS:
        got, rc = gpu_gemm(kern, np.ascontiguousarray(a),
                           np.ascontiguousarray(b))
        atol, rtol = TOL[kern]
        ok = got is not None and np.allclose(got, ref, atol=atol, rtol=rtol)
        detail = "" if ok else (
            f"rc={rc}" if got is None else
            f"max_err={np.abs(got - ref).max():.4f}")
        status = " ok " if ok else " FAIL"
        print(f"{status} {kern.replace('tt_cuda_sgemm_', ''):6s} "
              f"{M}x{N}x{K} {detail}")
        if not ok:
            FAILURES.append((kern, M, N, K, detail))

if FAILURES:
    print(f"\nGATE A RED: {len(FAILURES)} failure(s)")
    sys.exit(1)
print("\nGATE A GREEN")
