#ifndef CPU_BACKEND_H
#define CPU_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * M11 CPU backend: quant-aware threaded GEMV  y[M] = W[M,K] @ x[K].
 *
 * W is raw GGUF block data, row-major, K-contiguous rows — byte-identical
 * layout to what kernels/gemv_typed.cu consumes on device. Math mirrors
 * tt_gemv_typed semantics (fp32 accumulation over 32-value blocks, same
 * dequant formulas as src/dequant_ref.c, inlined here for speed).
 *
 * Supported dtypes (TTQ_* codes from dequant_ref.h):
 *   TTQ_Q4_0 (2):  18 B per 32 values
 *   TTQ_Q8_0 (8):  34 B per 32 values
 *   TTQ_Q4_K (12): 144 B per 256 values
 *   TTQ_Q5_K (13): 176 B per 256 values
 *   TTQ_Q6_K (14): 210 B per 256 values
 * Requires K % 32 == 0 (legacy) / K % 256 == 0 (K-quants) — same host-side
 * contract as GPU. Unsupported dtype -> -100 (mirrors tt_gemv_typed).
 *
 * Dispatch: on x86 with AVX2+FMA, all 5 dtypes run through intrinsics
 * kernels (dequant-with-FMA, per-block scale folded into the accumulator);
 * K-quants use per-sub-block incremental dot product (not the q*q pair-
 * sum trick — raw fp32 x has no quantization budget for the cross-term).
 * CPU_BACKEND_SCALAR=1 env forces the scalar path for A/B verification.
 * cb_using_avx2() reports the active path.
 *
 * Threading: OpenMP row-parallel when compiled with -fopenmp and
 * n_threads > 1; serial fallback otherwise. Rows are independent, so the
 * result is thread-count invariant bit-for-bit.
 *
 * This file is intentionally NOT wired into the Makefile (M11 CREATE-ONLY
 * constraint). Compile standalone:
 *
 *   gcc -O3 -mavx2 -mfma -fopenmp -std=c11 -Iinclude \
 *       -DCPU_BACKEND_MAIN -o build/cpu_backend src/cpu_backend.c -lm
 *
 * (drop -DCPU_BACKEND_MAIN to get a linkable object without main(); drop
 * -fopenmp for a pure-serial build).
 *
 * Standalone driver (CPU_BACKEND_MAIN):
 *   build/cpu_backend <W.bin> <dtype> <M> <K> <threads> <x.bin> [y.out]
 * Reads raw quantized W bytes + fp32 x, prints timing (ms, effective GB/s),
 * optionally writes fp32 y. Used by tests/test_cpu_backend.py.
 */

/* Returns 0 on success, negative on error:
 *   -1   NULL pointer or M/K <= 0
 *   -100 unsupported dtype (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K implemented)
 *   -101 K not a multiple of the required block size
 */
long tt_cpu_gemv(const void *W, int dtype, const float *x, float *y,
                 int M, int K, int n_threads);

/* 1 if the AVX2+FMA fast path is active (all 5 dtypes), 0 if scalar.
 * Honors CPU_BACKEND_SCALAR=1 override. */
int cb_using_avx2(void);

#ifdef __cplusplus
}
#endif

#endif /* CPU_BACKEND_H */
