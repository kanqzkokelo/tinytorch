#!/usr/bin/env python3
"""Golden + bench test for tt_gemm_batched (kernels/gemv_typed.cu).

Batched q4_0 GEMM for prompt prefill: Y[M,T] = W[M,K] @ X[K,T], one weight
stream per T-token tile instead of per token (prototype:
tests/proto_batched_gemv.cu).

Correctness: real tensor blk.0.attn_q.weight [896,896] Q4_0 dumped from
data/models/qwen2.5-0.5b-instruct-q4_0.gguf via build/dequant_ref (the CPU
reference dequantizer), X random, T in {1,8,16}; compare vs numpy dot,
rel < 1e-2 (quantization tolerance).

Bench: synthetic q4_0 weights at (1536,1536) and (4096,1536); us/token for
T=1 vs T=8/16.

GPU is shared (~3GB E2B process): all device allocations retry on OOM x5.
Footprint < 10 MB.

Run:  python3 tests/test_gemm_batched.py   (builds build/libtinytorch_cuda.so
via `make cuda` first)
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = os.path.join(REPO, "data", "models", "qwen2.5-0.5b-instruct-q4_0.gguf")
TENSOR = "blk.0.attn_q.weight"
MAX_T = 16          # TT_GEMM_BATCHED_MAX_T
REL_TOL = 1e-2

lib = None


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def build_lib():
    r = sh("make cuda", cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError("make cuda failed:\n" + r.stderr[-4000:])


def load_lib():
    import ctypes
    lib = ctypes.CDLL(os.path.join(REPO, "build", "libtinytorch_cuda.so"))
    for name in ("tt_cuda_alloc", "tt_gemm_batched"):
        assert hasattr(lib, name), f"symbol {name} missing from .so"
    lib.tt_cuda_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.tt_cuda_alloc.restype = ctypes.c_int
    lib.tt_cuda_free.argtypes = [ctypes.c_void_p]
    lib.tt_cuda_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.tt_cuda_h2d.restype = ctypes.c_int
    lib.tt_cuda_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.tt_cuda_d2h.restype = ctypes.c_int
    lib.tt_cuda_sync.restype = ctypes.c_int
    lib.tt_gemm_batched.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.tt_gemm_batched.restype = ctypes.c_int
    return lib


def cuda_alloc(bytes_):
    """cudaMalloc with OOM retry x5 (GPU shared with a ~3GB E2B process)."""
    import ctypes
    for attempt in range(5):
        p = ctypes.c_void_p()
        rc = lib.tt_cuda_alloc(ctypes.byref(p), bytes_)
        if rc == 0:
            return p
        print(f"  alloc({bytes_} B) failed rc={rc}, retry {attempt + 1}/5")
        time.sleep(2.0 * (attempt + 1))
        try:
            lib.tt_cuda_sync()
        except Exception:
            pass
    raise RuntimeError("device OOM after 5 retries")


def dequant_dump():
    """Dump fp32-dequantized real tensor via the C reference CLI."""
    out = os.path.join(tempfile.mkdtemp(prefix="tt_gemm_b_"), "w.f32")
    r = sh(f"{os.path.join(REPO, 'build', 'dequant_ref')} {GGUF} {TENSOR} {out}")
    if r.returncode != 0:
        raise RuntimeError("dequant_ref failed:\n" + r.stderr)
    w = np.fromfile(out, dtype=np.float32)
    n = w.size
    side = int(round(n ** 0.5))
    assert side * side == n, f"{TENSOR} not square-ish: {n}"
    return w.reshape(side, side)          # GGUF row-major W[M,K]


def run_gemm(w_q_bytes, w_np_deq, x, M, K, T):
    """Upload q4_0 W + X, run tt_gemm_batched, return Y[M,T] host copy."""
    import ctypes
    dw, dx, dy = cuda_alloc(M * K // 2), cuda_alloc(K * T * 4), cuda_alloc(M * T * 4)
    try:
        assert lib.tt_cuda_h2d(dw, w_q_bytes.ctypes.data, M * K // 2) == 0
        assert lib.tt_cuda_h2d(dx, x.ctypes.data, K * T * 4) == 0
        rc = lib.tt_gemm_batched(dw, 2, dx, dy, M, K, T, None)  # 2 = TTQ_Q4_0
        assert rc == 0, f"tt_gemm_batched rc={rc}"
        assert lib.tt_cuda_sync() == 0
        y = np.zeros(M * T, dtype=np.float32)
        assert lib.tt_cuda_d2h(y.ctypes.data, dy, M * T * 4) == 0
        return y.reshape(M, T)
    finally:
        lib.tt_cuda_free(dw)
        lib.tt_cuda_free(dx)
        lib.tt_cuda_free(dy)


def make_synthetic_w(M, K):
    """Random-but-valid q4_0 block bytes (fp16 d in normal range + nibbles)."""
    nb = K // 32
    rng = np.random.default_rng(42)
    d = rng.uniform(0.02, 0.06, size=(M, nb)).astype(np.float16)
    qs = rng.integers(0, 256, size=(M, nb, 16), dtype=np.uint8)
    w = np.concatenate([d.view(np.uint8).reshape(M, nb, 2), qs], axis=2)
    return w.reshape(M, nb * 18).copy()


def ref_dot(w_deq_f32, x, M, K, T):
    """numpy reference over the dequantized weights (float64 accum)."""
    return (w_deq_f32.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


class TestGemmbBatched(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global lib
        build_lib()
        lib = load_lib()

    def test_golden_real_tensor(self):
        w_deq = dequant_dump()               # [M,K] fp32, C-reference dequant
        M, K = w_deq.shape
        # raw q4_0 bytes straight from the GGUF for the same tensor
        import struct
        with open(GGUF, "rb") as f:
            f.seek(229100544)                # offset from --sizes
            w_q = np.frombuffer(f.read(M * K // 2), dtype=np.uint8)
        rng = np.random.default_rng(7)
        for T in (1, 8, 16):
            x = rng.standard_normal((K, T)).astype(np.float32)
            y = run_gemm(w_q, None, x, M, K, T)
            r = ref_dot(w_deq, x, M, K, T)
            e = rel_err(y, r)
            self.assertLess(e, REL_TOL, f"T={T} rel err {e:.3e}")
            print(f"golden {TENSOR} [{M},{K}] T={T}: rel={e:.2e} PASS")

    def test_error_codes(self):
        import ctypes
        w = make_synthetic_w(64, 256)
        x = np.zeros((256, 8), dtype=np.float32)
        dw, dx, dy = cuda_alloc(64 * 128), cuda_alloc(256 * 8 * 4), cuda_alloc(64 * 8 * 4)
        try:
            lib.tt_cuda_h2d(dw, w.ctypes.data, 64 * 128)
            lib.tt_cuda_h2d(dx, x.ctypes.data, 256 * 8 * 4)
            self.assertEqual(lib.tt_gemm_batched(None, 2, dx, dy, 64, 256, 8, None), -50)
            self.assertEqual(lib.tt_gemm_batched(dw, 2, dx, dy, 64, 256, 0, None), -50)
            self.assertEqual(lib.tt_gemm_batched(dw, 2, dx, dy, 64, 256, MAX_T + 1, None), -50)
            self.assertEqual(lib.tt_gemm_batched(dw, 99, dx, dy, 64, 256, 8, None), -100)
            self.assertEqual(lib.tt_gemm_batched(dw, 2, dx, dy, 64, 250, 8, None), -101)
        finally:
            lib.tt_cuda_free(dw); lib.tt_cuda_free(dx); lib.tt_cuda_free(dy)

    def test_bench(self):
        rng = np.random.default_rng(1)
        print(f"\n{'shape':>12} {'T':>3} {'ms':>9} {'us/tok':>8} {'tok/s':>9} {'GB/s':>7}")
        for (M, K) in ((1536, 1536), (4096, 1536)):
            w = make_synthetic_w(M, K)
            wb = M * (K // 32) * 18
            us_per_tok = {}
            for T in (1, 8, 16):
                x = rng.standard_normal((K, T)).astype(np.float32)
                y = run_gemm(w, None, x, M, K, T)     # warmup + correctness sanity
                dw, dx, dy = cuda_alloc(wb), cuda_alloc(K * T * 4), cuda_alloc(M * T * 4)
                try:
                    lib.tt_cuda_h2d(dw, w.ctypes.data, wb)
                    lib.tt_cuda_h2d(dx, x.ctypes.data, K * T * 4)
                    iters = 50 if M <= 2048 else 20
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        rc = lib.tt_gemm_batched(dw, 2, dx, dy, M, K, T, None)
                        assert rc == 0
                    lib.tt_cuda_sync()
                    ms = (time.perf_counter() - t0) / iters * 1e3
                finally:
                    lib.tt_cuda_free(dw); lib.tt_cuda_free(dx); lib.tt_cuda_free(dy)
                gbs = (wb + K * T * 4 + M * T * 4) / (ms * 1e-3) / 1e9
                us_per_tok[T] = ms * 1e3 / T
                print(f"{f'({M},{K})':>12} {T:>3} {ms:>9.4f} {us_per_tok[T]:>8.1f} "
                      f"{T/(ms*1e-3):>9.0f} {gbs:>7.1f}")
            print(f"             -> us/token vs T=1: T=8 {us_per_tok[1]/us_per_tok[8]:.2f}x cheaper, "
                  f"T=16 {us_per_tok[1]/us_per_tok[16]:.2f}x cheaper")


if __name__ == "__main__":
    unittest.main(verbosity=2)
