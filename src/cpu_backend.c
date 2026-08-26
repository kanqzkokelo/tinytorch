/*
 * cpu_backend.c - M11 quant-aware threaded CPU GEMV
 * (Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K; AVX2 fast path for Q4_0/Q8_0).
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
#define TTQ_Q4_K 12
#define TTQ_Q5_K 13
#define TTQ_Q6_K 14

#define QK 32                 /* values per legacy block */
#define QK_K 256              /* values per K-quant super-block */
#define K_SCALE_SIZE 12       /* packed 6-bit scale/min bytes (q4_K/q5_K) */
#define QK4_0_BS 18           /* bytes per q4_0 block: fp16 d + 16 nibbles */
#define QK8_0_BS 34           /* bytes per q8_0 block: fp16 d + 32 int8    */
#define QK4_K_BS 144          /* d,dmin | scales[12] | qs[128]            */
#define QK5_K_BS 176          /* d,dmin | scales[12] | qh[32] | qs[128]   */
#define QK6_K_BS 210          /* ql[128] | qh[64] | sc[16] | fp16 d       */

#if defined(__x86_64__) || defined(__i386__)
#define CB_X86 1
#include <immintrin.h>
#endif

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

/* Port of get_scale_min_k4 (ggml-quants.c): 6-bit scale/min pairs packed
 * into 12 bytes at blk+4. j in [0,8) selects the pair. */
static inline void cb_get_scale_min_k4(int j, const uint8_t *q,
                                       uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j + 0] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
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

/* ------------------------- K-quant dot products -------------------------- */

/* block_q4_K: fp16 d | fp16 dmin | scales[12] | qs[128]  (144 B / 256 vals).
 * Value math: d*sc*(nib) - min*m per sub-block (dequant_ref.c dq_q4_K). */
static inline float row_dot_q4_K(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK_K;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK4_K_BS;
        const uint8_t *q   = blk + 4 + K_SCALE_SIZE;
        uint16_t dh, dmh;
        memcpy(&dh,  blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const float d   = cb_fp16_to_fp32(dh);
        const float min = cb_fp16_to_fp32(dmh);
        const float *xb = x + (long)b * QK_K;
        float sum = 0.0f;
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            cb_get_scale_min_k4(is + 0, blk + 4, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            cb_get_scale_min_k4(is + 1, blk + 4, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l)
                sum += (d1 * (q[l] & 0xF) - m1) * xb[l];
            for (int l = 0; l < 32; ++l)
                sum += (d2 * (q[l] >> 4) - m2) * xb[l + 32];
            q += 32; xb += 64; is += 2;
        }
        acc += sum;
    }
    return acc;
}

/* block_q5_K: d,dmin | scales[12] | qh[32] | qs[128]  (176 B / 256 vals).
 * Field order per dequant_ref.c: high bits live in qh, masks rotate u1=1,u2=2
 * with <<2 per 64-value chunk. */
static inline float row_dot_q5_K(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK_K;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK5_K_BS;
        const uint8_t *qh  = blk + 4 + K_SCALE_SIZE;
        const uint8_t *ql  = qh + QK_K / 8;
        uint16_t dh, dmh;
        memcpy(&dh,  blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const float d   = cb_fp16_to_fp32(dh);
        const float min = cb_fp16_to_fp32(dmh);
        const float *xb = x + (long)b * QK_K;
        float sum = 0.0f;
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            cb_get_scale_min_k4(is + 0, blk + 4, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            cb_get_scale_min_k4(is + 1, blk + 4, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l)
                sum += (d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1) * xb[l];
            for (int l = 0; l < 32; ++l)
                sum += (d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2) * xb[l + 32];
            ql += 32; xb += 64; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
        acc += sum;
    }
    return acc;
}

/* block_q6_K: ql[128] | qh[64] | sc[16] int8 | fp16 d  (210 B / 256 vals).
 * No min term; per-16-value int8 scales, values biased by -32. */
static inline float row_dot_q6_K(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK_K;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK6_K_BS;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + QK_K / 2;
        const int8_t *sc = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        uint16_t dh;
        memcpy(&dh, blk + QK_K / 2 + QK_K / 4 + QK_K / 16, 2);
        const float d = cb_fp16_to_fp32(dh);
        const float *xb = x + (long)b * QK_K;
        float sum = 0.0f;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * sc[is + 0] * q1 * xb[n + l +  0]
                     + d * sc[is + 2] * q2 * xb[n + l + 32]
                     + d * sc[is + 4] * q3 * xb[n + l + 64]
                     + d * sc[is + 6] * q4 * xb[n + l + 96];
            }
            ql += 64; qh += 32; sc += 8;
        }
        acc += sum;
    }
    return acc;
}

/* ------------------------- AVX2 paths (q4_0 / q8_0) ---------------------- */

#ifdef CB_X86

/* FMADD 16 signed int8 lanes against 16 floats at xp, accumulate into acc.
 * Used by both AVX2 kernels: q values already fit int8 (-8..7 or full range). */
__attribute__((target("avx2,fma")))
static inline __m256 cb_fma16(__m256 acc, __m128i q8, const float *xp,
                              __m256 dv) {
    const __m256i qa = _mm256_cvtepi8_epi32(q8);
    const __m128i qb8 = _mm_bsrli_si128(q8, 8);
    const __m256i qb = _mm256_cvtepi8_epi32(qb8);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(qa), _mm256_loadu_ps(xp), acc);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(qb), _mm256_loadu_ps(xp + 8), acc);
    (void)dv;
    return acc;
}

__attribute__((target("avx2,fma")))
static inline float cb_hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
static inline float row_dot_q4_0_avx2(const uint8_t *rw, const float *x,
                                      int K) {
    /* Dequant-with-MAC: nibbles -> int8 (val-8) -> int32 -> fp32 -> FMA with x;
     * per-block fp16 scale folded into the accumulator once per block:
     * sum_b d_b * (q_b . x_b). */
    const int nb = K / QK;
    const __m128i f0 = _mm_set1_epi8(0x0F);
    const __m128i e8 = _mm_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK4_0_BS;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cb_fp16_to_fp32(dh);
        const __m128i raw = _mm_loadu_si128((const __m128i *)(blk + 2));
        const __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, f0), e8);
        const __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), f0), e8);
        const float *xb = x + (long)b * QK;
        __m256 pacc = _mm256_setzero_ps();
        pacc = cb_fma16(pacc, lo, xb, _mm256_setzero_ps());
        pacc = cb_fma16(pacc, hi, xb + 16, _mm256_setzero_ps());
        acc = _mm256_fmadd_ps(pacc, _mm256_set1_ps(d), acc);
    }
    return cb_hsum256(acc);
}

__attribute__((target("avx2,fma")))
static inline float row_dot_q8_0_avx2(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK;
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK8_0_BS;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cb_fp16_to_fp32(dh);
        const __m128i r0 = _mm_loadu_si128((const __m128i *)(blk + 2));
        const __m128i r1 = _mm_loadu_si128((const __m128i *)(blk + 18));
        const float *xb = x + (long)b * QK;
        __m256 pacc = _mm256_setzero_ps();
        pacc = cb_fma16(pacc, r0, xb, _mm256_setzero_ps());
        pacc = cb_fma16(pacc, r1, xb + 16, _mm256_setzero_ps());
        acc = _mm256_fmadd_ps(pacc, _mm256_set1_ps(d), acc);
    }
    return cb_hsum256(acc);
}

#endif /* CB_X86 */

/* ------------------------- dispatch -------------------------------------- */

static int g_cb_path = -1;   /* 0 = scalar, 1 = AVX2+FMA */

int cb_using_avx2(void) {
    if (g_cb_path < 0) {
        const char *e = getenv("CPU_BACKEND_SCALAR");
        if (e && e[0] && e[0] != '0') {
            g_cb_path = 0;
        } else {
#ifdef CB_X86
            __builtin_cpu_init();
            g_cb_path = __builtin_cpu_supports("avx2") &&
                        __builtin_cpu_supports("fma");
#else
            g_cb_path = 0;
#endif
        }
    }
    return g_cb_path;
}

static inline float row_dot(const uint8_t *rw, int dtype, const float *x,
                            int K, int use_avx2) {
    switch (dtype) {
        case TTQ_Q4_0:
#ifdef CB_X86
            if (use_avx2) return row_dot_q4_0_avx2(rw, x, K);
#endif
            return row_dot_q4_0(rw, x, K);
        case TTQ_Q8_0:
#ifdef CB_X86
            if (use_avx2) return row_dot_q8_0_avx2(rw, x, K);
#endif
            return row_dot_q8_0(rw, x, K);
        case TTQ_Q4_K: return row_dot_q4_K(rw, x, K);
        case TTQ_Q5_K: return row_dot_q5_K(rw, x, K);
        case TTQ_Q6_K: return row_dot_q6_K(rw, x, K);
        default: return 0.0f;
    }
}

long tt_cpu_gemv(const void *W, int dtype, const float *x, float *y,
                 int M, int K, int n_threads) {
    if (!W || !x || !y || M <= 0 || K <= 0) return -1;
    if (dtype != TTQ_Q4_0 && dtype != TTQ_Q8_0 &&
        dtype != TTQ_Q4_K && dtype != TTQ_Q5_K && dtype != TTQ_Q6_K) {
        fprintf(stderr, "[cpu-backend] unsupported dtype %d "
                "(Q4_0/Q8_0/Q4_K/Q5_K/Q6_K only)\n", dtype);
        return -100;
    }
    if (K % QK != 0 ||
        ((dtype == TTQ_Q4_K || dtype == TTQ_Q5_K || dtype == TTQ_Q6_K) &&
         K % QK_K != 0)) {
        fprintf(stderr, "[cpu-backend] K=%d not a multiple of %d%s\n", K, QK,
                (dtype == TTQ_Q4_K || dtype == TTQ_Q5_K ||
                 dtype == TTQ_Q6_K) ? " (K-quants need %256)" : "");
        return -101;
    }
    if (n_threads <= 0) n_threads = 1;

    long bs;
    switch (dtype) {
        case TTQ_Q4_0: bs = QK4_0_BS; break;
        case TTQ_Q8_0: bs = QK8_0_BS; break;
        case TTQ_Q4_K: bs = QK4_K_BS; break;
        case TTQ_Q5_K: bs = QK5_K_BS; break;
        default:       bs = QK6_K_BS; break;
    }
    const int row_vals = (bs >= QK4_K_BS) ? QK_K : QK;
    const int use_avx2 = cb_using_avx2();

    const uint8_t *w = (const uint8_t *)W;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int row = 0; row < M; row++) {
        const uint8_t *rw = w + (long)row * (K / row_vals) * bs;
        y[row] = row_dot(rw, dtype, x, K, use_avx2);
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

    long bs;
    switch (dtype) {
        case TTQ_Q4_0: bs = QK4_0_BS; break;
        case TTQ_Q8_0: bs = QK8_0_BS; break;
        case TTQ_Q4_K: bs = QK4_K_BS; break;
        case TTQ_Q5_K: bs = QK5_K_BS; break;
        default:       bs = QK6_K_BS; break;
    }
    long wbytes = (long)M * (K / ((bs >= QK4_K_BS) ? QK_K : QK)) * bs;
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
    const char *path_name = cb_using_avx2() ? "avx2" : "scalar";

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
    printf("path=%s dtype=%d M=%d K=%d threads=%d  %.3f ms  %.1f GB/s\n",
           path_name, dtype, M, K, threads, best, gbs);

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
