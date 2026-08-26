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
 * Supported dtypes today (TTQ_* codes from dequant_ref.h):
 *   TTQ_Q4_0 (2): 18 B per 32 values
 *   TTQ_Q8_0 (8): 34 B per 32 values
 * Requires K % 32 == 0 for these types (same host-side contract as GPU).
 * Unsupported dtype -> -100 (mirrors tt_gemv_typed).
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
 *   -100 unsupported dtype (only Q4_0/Q8_0 implemented)
 *   -101 K not a multiple of 32 for the requested block type
 */
long tt_cpu_gemv(const void *W, int dtype, const float *x, float *y,
                 int M, int K, int n_threads);

#ifdef __cplusplus
}
#endif

#endif /* CPU_BACKEND_H */
