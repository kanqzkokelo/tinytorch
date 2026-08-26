/*
 * cpu_backend.c - M11 quant-aware threaded CPU GEMV (Q4_0 / Q8_0).
 *
 * y[M] = W[M,K] @ x[K], W in raw GGUF block layout (row-major, K-contiguous),
 * same bytes the CUDA kernels consume. Dequant math is the executable spec
 * from src/dequant_ref.c (ported from llama.cpp ggml-quants.c), inlined into
 * the accumulation loops — no per-element calls into dequant_ref.
 *
 * Accumulation: fp32 per row, sequential over blocks. Thread-count invariant
 * (rows are independent); matches tt_gemv_typed within fp32 accumulation
 * tolerance (~1e-2 rel at K=1536 quantized).
 *
 * See include/cpu_backend.h for the compile line and standalone-driver CLI.
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c11 */
#include "cpu_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* TTQ_* type codes (dequant_ref.h values; duplicated to stay header-only) */
#define TTQ_Q4_0 2
#define TTQ_Q8_0 8

#define QK 32                 /* values per block, both types */
#define QK4_0_BS 18           /* bytes per q4_0 block: fp16 d + 16 nibbles */
#define QK8_0_BS 34           /* bytes per q8_0 block: fp16 d + 32 int8    */

/* IEEE half -> float, bit-exact incl. subnormals (same as dequant_ref.c). */
static inline float cb_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7c00u) >> 10;
    uint32_t man  = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else { /* subnormal: normalize */
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x0400u));
            man &= 0x03ffu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    union { unsigned int u; float f; } cvt;
    cvt.u = bits;
    return cvt.f;
}

static inline float row_dot_q4_0(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK;
    /* 4 independent accumulators break the fp32 add dependency chain and
     * give the compiler SLP-vectorizable pairs. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK4_0_BS;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cb_fp16_to_fp32(dh);
        const uint8_t *qs = blk + 2;
        const float *xb = x + (long)b * QK;
        for (int j = 0; j < QK / 2; j += 4) {
            a0 += ((qs[j]   & 0x0F) - 8) * d * xb[j];
            a1 += ((qs[j]   >>   4) - 8) * d * xb[j + QK / 2];
            a2 += ((qs[j+1] & 0x0F) - 8) * d * xb[j+1];
            a3 += ((qs[j+1] >>   4) - 8) * d * xb[j+1 + QK / 2];
            a0 += ((qs[j+2] & 0x0F) - 8) * d * xb[j+2];
            a1 += ((qs[j+2] >>   4) - 8) * d * xb[j+2 + QK / 2];
            a2 += ((qs[j+3] & 0x0F) - 8) * d * xb[j+3];
            a3 += ((qs[j+3] >>   4) - 8) * d * xb[j+3 + QK / 2];
        }
    }
    return (a0 + a2) + (a1 + a3);
}

static inline float row_dot_q8_0(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK8_0_BS;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cb_fp16_to_fp32(dh);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + (long)b * QK;
        for (int j = 0; j < QK; j += 4) {
            a0 += qs[j]   * d * xb[j];
            a1 += qs[j+1] * d * xb[j+1];
            a2 += qs[j+2] * d * xb[j+2];
            a3 += qs[j+3] * d * xb[j+3];
        }
    }
    return (a0 + a2) + (a1 + a3);
}

long tt_cpu_gemv(const void *W, int dtype, const float *x, float *y,
                 int M, int K, int n_threads) {
    if (!W || !x || !y || M <= 0 || K <= 0) return -1;
    if (dtype != TTQ_Q4_0 && dtype != TTQ_Q8_0) {
        fprintf(stderr, "[cpu-backend] unsupported dtype %d (only Q4_0/Q8_0)\n",
                dtype);
        return -100;
    }
    if (K % QK != 0) {
        fprintf(stderr, "[cpu-backend] K=%d not a multiple of %d\n", K, QK);
        return -101;
    }
    if (n_threads <= 0) n_threads = 1;

    const uint8_t *w = (const uint8_t *)W;
    const long bs = (dtype == TTQ_Q4_0) ? QK4_0_BS : QK8_0_BS;

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int row = 0; row < M; row++) {
        const uint8_t *rw = w + (long)row * (K / QK) * bs;
        y[row] = (dtype == TTQ_Q4_0) ? row_dot_q4_0(rw, x, K)
                                     : row_dot_q8_0(rw, x, K);
    }
    return 0;
}

/* ------------------------- standalone driver ----------------------------- */
#ifdef CPU_BACKEND_MAIN

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <W.bin> <dtype> <M> <K> <threads> <x.bin> [y.out]\n",
                argv[0]);
        return 1;
    }
    const char *wpath = argv[1];
    int dtype = atoi(argv[2]), M = atoi(argv[3]), K = atoi(argv[4]);
    int threads = atoi(argv[5]);
    const char *xpath = argv[6];
    const char *ypath = argc > 7 ? argv[7] : NULL;

    long bs = (dtype == TTQ_Q4_0) ? QK4_0_BS : QK8_0_BS;
    long wbytes = (long)M * (K / QK) * bs;
    FILE *f = fopen(wpath, "rb");
    if (!f) { perror("open W"); return 1; }
    void *W = malloc((size_t)wbytes);
    if (fread(W, 1, (size_t)wbytes, f) != (size_t)wbytes) {
        fprintf(stderr, "short read on W (%ld bytes)\n", wbytes);
        return 1;
    }
    fclose(f);

    float *x = malloc((size_t)K * sizeof(float));
    f = fopen(xpath, "rb");
    if (!f || fread(x, sizeof(float), (size_t)K, f) != (size_t)K) {
        fprintf(stderr, "cannot read x (%d floats)\n", K);
        return 1;
    }
    fclose(f);

    float *y = malloc((size_t)M * sizeof(float));
    long rc = tt_cpu_gemv(W, dtype, x, y, M, K, threads);
    if (rc != 0) { fprintf(stderr, "tt_cpu_gemv -> %ld\n", rc); return 1; }

    /* timing: warmup + median of reps */
    int reps = 20;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        tt_cpu_gemv(W, dtype, x, y, M, K, threads);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    double gbs = (double)wbytes / (best * 1e-3) / 1e9;
    printf("dtype=%d M=%d K=%d threads=%d  %.3f ms  %.1f GB/s\n",
           dtype, M, K, threads, best, gbs);

    if (ypath) {
        f = fopen(ypath, "wb");
        if (!f || fwrite(y, sizeof(float), (size_t)M, f) != (size_t)M) {
            perror("write y"); return 1;
        }
        fclose(f);
    }
    free(W); free(x); free(y);
    return 0;
}
#endif /* CPU_BACKEND_MAIN */
