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
#define TTQ_Q2_K 10
#define TTQ_Q3_K 11
#define TTQ_Q4_K 12
#define TTQ_Q5_K 13
#define TTQ_Q6_K 14

#define QK 32                 /* values per legacy block */
#define QK_K 256              /* values per K-quant super-block */
#define K_SCALE_SIZE 12       /* packed 6-bit scale/min bytes (q4_K/q5_K) */
#define QK4_0_BS 18           /* bytes per q4_0 block: fp16 d + 16 nibbles */
#define QK8_0_BS 34           /* bytes per q8_0 block: fp16 d + 32 int8    */
#define QK2_K_BS 84           /* scales[16] | qs[64] | fp16 d, fp16 dmin   */
#define QK3_K_BS 110          /* hmask[32] | qs[64] | scales[12] | fp16 d  */
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
static inline float row_dot_q2_K(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK_K;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK2_K_BS;
        uint16_t dh, dmh;
        memcpy(&dh, blk + 80, 2);
        memcpy(&dmh, blk + 82, 2);
        const float d = cb_fp16_to_fp32(dh);
        const float min = cb_fp16_to_fp32(dmh);
        const uint8_t *q = blk + 16;
        const float *xb = x + (long)b * QK_K;
        float sum = 0.0f;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc0 = blk[is++];
                float dl0 = d * (float)(sc0 & 0xF);
                float ml0 = min * (float)(sc0 >> 4);
                for (int l = 0; l < 16; ++l)
                    sum += (dl0 * ((int8_t)((q[l] >> shift) & 3)) - ml0) * xb[l];

                uint8_t sc1 = blk[is++];
                float dl1 = d * (float)(sc1 & 0xF);
                float ml1 = min * (float)(sc1 >> 4);
                for (int l = 0; l < 16; ++l)
                    sum += (dl1 * ((int8_t)((q[l + 16] >> shift) & 3)) - ml1) * xb[l + 16];

                shift += 2;
                xb += 32;
            }
            q += 32;
        }
        acc += sum;
    }
    return acc;
}
static inline float row_dot_q3_K(const uint8_t *rw, const float *x, int K) {
    const int nb = K / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;
    float acc = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK3_K_BS;
        uint16_t dh;
        memcpy(&dh, blk + 108, 2);
        const float d_all = cb_fp16_to_fp32(dh);
        const uint8_t *q = blk + 32;
        const uint8_t *hm = blk;
        uint8_t m = 1;

        memcpy(aux, blk + 96, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const float *xb = x + (long)b * QK_K;
        float sum = 0.0f;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl0 = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    float w = dl0 * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                    sum += w * xb[l];
                }
                float dl1 = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    float w = dl1 * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    sum += w * xb[l + 16];
                }
                shift += 2;
                m <<= 1;
                xb += 32;
            }
            q += 32;
        }
        acc += sum;
    }
    return acc;
}

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

/* K-quant AVX2 strategy
 * ---------------------
 * Each K-quant super-block (256 values) decomposes into 32-element sub-blocks
 * each with a single fp32 scale factor. We don't try to dequantize the whole
 * 256-element block into fp32 in registers (would need 8 ymm of fp32 plus all
 * the nibble plumbing, and would force 4 KB of register state per row). Instead
 * we do the dot-product *incrementally per sub-block* — for each sub-block:
 *   1. load 16-32 bytes of nibble/quant data into 1-2 ymm,
 *   2. extract the int8 values (0..15 for Q4/Q5_K, -32..31 for Q6_K),
 *   3. compute 16 int16 pair-sums via _mm256_maddubs_epi16 (sums q[2i]+q[2i+1]),
 *   4. precompute x_pair[i] = x[2i]+x[2i+1] as 16 fp32, FMA q_pair·x_pair in fp32,
 *   5. fold the per-sub-block scale (d*sc for Q6_K, d*sc / -min*m for Q4/Q5_K)
 *      and the per-row fp32 accumulator.
 * The Q4_K/Q5_K subtraction `d*sc*q - min*m*1` is split: FMA the q·x part
 * with d*sc, then a single `min*m*hsum(x)` subtraction at the sub-block end.
 * For Q6_K there's no min term; per-16 sub-block we just FMA `d*sc*q` against x.
 * Memory: 8 ymm for two 32-elt q loads + 2 ymm of x_pairs = 10 ymm working +
 * 1-2 ymm acc — fits comfortably in the 16-ymm AVX2 budget. The fp16 d / dmin
 * loads are scalar and amortized over the whole super-block.
 */

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

/* ---- K-quant AVX2 kernels ---------------------------------------------- */

/* hsum of 32 consecutive floats (one sub-block of x). */
__attribute__((target("avx2,fma")))
static inline float cb_hsum32(const float *xp) {
    __m256 a = _mm256_loadu_ps(xp);
    __m256 b = _mm256_loadu_ps(xp + 8);
    __m256 c = _mm256_loadu_ps(xp + 16);
    __m256 d = _mm256_loadu_ps(xp + 24);
    __m256 s = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 lo = _mm256_castps256_ps128(s);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(lo);
}

/* Q4_K AVX2: fp16 d, fp16 dmin, scales[12], qs[128] -> 144 B per 256.
 * Per super-block: 4 outer iterations, each emitting 64 values via 2 sub-blocks
 * of 32 (lo-nibble and hi-nibble, each with its own d*sc and min*m).
 *
 * Correctness note: we cannot use the "pair sum" trick (`(q[2i]+q[2i+1]) *
 * (x[2i]+x[2i+1])` as a substitute for `q[2i]*x[2i] + q[2i+1]*x[2i+1]`) —
 * that introduces a cross-term that's only negligible when the second operand
 * is itself quantized. For raw fp32 x we must do the full dot product.
 *
 * Strategy: per sub-block of 32 vals, load 32 fp32 x's, convert 32 int8 nibbles
 * to 32 fp32 via cvtepi8_epi32+cvtepi32_ps (4 ymm each: 8 from each quarter),
 * FMA into a partial dot, then apply d*sc / -min*m at sub-block end.
 */
__attribute__((target("avx2,fma")))
static inline float row_dot_q4_K_avx2(const uint8_t *rw, const float *x,
                                      int K) {
    const int nb = K / QK_K;
    const __m128i f0 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK4_K_BS;
        const uint8_t *q   = blk + 4 + K_SCALE_SIZE;
        const uint8_t *sc  = blk + 4;
        uint16_t dh, dmh;
        memcpy(&dh,  blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const float d   = cb_fp16_to_fp32(dh);
        const float min = cb_fp16_to_fp32(dmh);
        const float *xb = x + (long)b * QK_K;
        for (int j = 0; j < QK_K; j += 64) {
            const __m256i qb = _mm256_loadu_si256((const __m256i *)q);
            /* 32 bytes of q: 32 lo-nibble bytes AND 32 hi-nibble bytes. */
            const __m128i qb_lo = _mm256_castsi256_si128(qb);
            const __m128i qb_hi = _mm256_extracti128_si256(qb, 1);
            const __m128i lo32_0 = _mm_and_si128(qb_lo, f0);  /* 16 lo vals, l=0..16 */
            const __m128i lo32_1 = _mm_and_si128(qb_hi, f0);  /* 16 lo vals, l=16..32 */
            const __m128i hi32_0 = _mm_and_si128(_mm_srli_epi16(qb_lo, 4), f0);  /* 16 hi vals, l=0..16 */
            const __m128i hi32_1 = _mm_and_si128(_mm_srli_epi16(qb_hi, 4), f0);  /* 16 hi vals, l=16..32 */
            /* Convert each 16-int8 to 16 fp32 (2 ymm: lower 8 + upper 8). */
            const __m256 lo0_lf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo32_0));
            const __m256 lo0_hf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(lo32_0, 8)));
            const __m256 lo1_lf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo32_1));
            const __m256 lo1_hf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(lo32_1, 8)));
            const __m256 hi0_lf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi32_0));
            const __m256 hi0_hf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(hi32_0, 8)));
            const __m256 hi1_lf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi32_1));
            const __m256 hi1_hf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(hi32_1, 8)));
            /* Full 32-elt dot products: lo · x[0..32] and hi · x[32..64]. */
            __m256 dot_lo = _mm256_mul_ps(lo0_lf, _mm256_loadu_ps(xb +  0));
            dot_lo = _mm256_fmadd_ps(lo0_hf, _mm256_loadu_ps(xb +  8), dot_lo);
            dot_lo = _mm256_fmadd_ps(lo1_lf, _mm256_loadu_ps(xb + 16), dot_lo);
            dot_lo = _mm256_fmadd_ps(lo1_hf, _mm256_loadu_ps(xb + 24), dot_lo);
            __m256 dot_hi = _mm256_mul_ps(hi0_lf, _mm256_loadu_ps(xb + 32));
            dot_hi = _mm256_fmadd_ps(hi0_hf, _mm256_loadu_ps(xb + 40), dot_hi);
            dot_hi = _mm256_fmadd_ps(hi1_lf, _mm256_loadu_ps(xb + 48), dot_hi);
            dot_hi = _mm256_fmadd_ps(hi1_hf, _mm256_loadu_ps(xb + 56), dot_hi);
            const float v_lo = cb_hsum256(dot_lo);
            const float v_hi = cb_hsum256(dot_hi);
            const int isb = (j >> 6) * 2;
            uint8_t sc0, m0, sc1, m1;
            cb_get_scale_min_k4(isb + 0, sc, &sc0, &m0);
            cb_get_scale_min_k4(isb + 1, sc, &sc1, &m1);
            const float d1 = d * (float)sc0, d2 = d * (float)sc1;
            const float m1c = min * (float)m0, m2c = min * (float)m1;
            const float sx1 = cb_hsum32(xb);
            const float sx2 = cb_hsum32(xb + 32);
            const __m256 pack = _mm256_set_ps(d2 * v_hi,  d1 * v_lo,
                                              -m2c * sx2, -m1c * sx1,
                                              0.f, 0.f, 0.f, 0.f);
            acc = _mm256_add_ps(acc, pack);
            q   += 32;
            xb  += 64;
        }
    }
    return cb_hsum256(acc);
}

/* Q5_K AVX2: same dequant math as Q4_K but each nibble has a 5th bit in
 * qh[32]. The qh byte at index l holds two 1-bit masks: bit 0 (lo mask) and
 * bit 1 (hi mask). Value becomes ql_nibble + (qh_bit << 4) in [0, 31].
 * 4 outer iters × 2 sub-blocks of 32 vals = 8 sub-blocks per super-block.
 * Same fp32 dot structure as Q4_K AVX2 (no pair-sum trick). */
__attribute__((target("avx2,fma")))
static inline float row_dot_q5_K_avx2(const uint8_t *rw, const float *x,
                                      int K) {
    const int nb = K / QK_K;
    const __m128i f0 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK5_K_BS;
        const uint8_t *sc  = blk + 4;
        const uint8_t *qh  = blk + 4 + K_SCALE_SIZE;
        const uint8_t *ql  = qh + QK_K / 8;
        uint16_t dh, dmh;
        memcpy(&dh,  blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const float d   = cb_fp16_to_fp32(dh);
        const float min = cb_fp16_to_fp32(dmh);
        const float *xb = x + (long)b * QK_K;
        for (int j = 0; j < QK_K; j += 64) {
            const __m256i qlb = _mm256_loadu_si256((const __m256i *)ql);
            const __m256i qhb = _mm256_loadu_si256((const __m256i *)qh);
            const __m128i qlb_lo = _mm256_castsi256_si128(qlb);
            const __m128i qlb_hi = _mm256_extracti128_si256(qlb, 1);
            const __m128i qhb_lo = _mm256_castsi256_si128(qhb);
            const __m128i qhb_hi = _mm256_extracti128_si256(qhb, 1);
            /* lo_value[l] = (qlb[l] & 0xF) | ((qhb[l] & 1) << 4)
             * hi_value[l] = (qlb[l] >> 4) & 0xF | ((qhb[l] & 2) << 3) */
            const __m128i lo32_0_raw = _mm_and_si128(qlb_lo, f0);
            const __m128i lo32_1_raw = _mm_and_si128(qlb_hi, f0);
            const __m128i hi32_0_raw = _mm_and_si128(_mm_srli_epi16(qlb_lo, 4), f0);
            const __m128i hi32_1_raw = _mm_and_si128(_mm_srli_epi16(qlb_hi, 4), f0);
            /* qh bit position depends on chunk: chunk 0 -> bit 0 (lo) / 1 (hi),
             * chunk 1 -> 2/3, chunk 2 -> 4/5, chunk 3 -> 6/7. We shift qh right
             * by qh_shift to bring the relevant bit to position 0, then AND 0x01
             * and shift left by 4 to land at value 16. The shift-left of 0x10
             * must be within 16-bit lanes to avoid byte-cross for chunks 2/3. */
            const int qh_shift = (j >> 6) * 2;  /* 0, 2, 4, 6 */
            const __m128i qh_lo_s = _mm_srli_epi16(qhb_lo, qh_shift);
            const __m128i qh_hi_s = _mm_srli_epi16(qhb_hi, qh_shift);
            const __m128i lo_bit0 = _mm_and_si128(qh_lo_s, _mm_set1_epi8(0x01));
            const __m128i lo_bit1 = _mm_and_si128(qh_hi_s, _mm_set1_epi8(0x01));
            const __m128i hi_bit0 = _mm_and_si128(_mm_srli_epi16(qh_lo_s, 1), _mm_set1_epi8(0x01));
            const __m128i hi_bit1 = _mm_and_si128(_mm_srli_epi16(qh_hi_s, 1), _mm_set1_epi8(0x01));
            const __m128i lo_mask_0 = _mm_slli_epi16(lo_bit0, 4);
            const __m128i lo_mask_1 = _mm_slli_epi16(lo_bit1, 4);
            const __m128i hi_mask_0 = _mm_slli_epi16(hi_bit0, 4);
            const __m128i hi_mask_1 = _mm_slli_epi16(hi_bit1, 4);
            const __m128i lo0 = _mm_or_si128(lo32_0_raw, lo_mask_0);
            const __m128i lo1 = _mm_or_si128(lo32_1_raw, lo_mask_1);
            const __m128i hi0 = _mm_or_si128(hi32_0_raw, hi_mask_0);
            const __m128i hi1 = _mm_or_si128(hi32_1_raw, hi_mask_1);
#define CB5K_QF(NAME, INP) do {                                                 \
                NAME##_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(INP));      \
                NAME##_hi = _mm256_cvtepi32_ps(                                 \
                    _mm256_cvtepi8_epi32(_mm_bsrli_si128(INP, 8)));             \
            } while (0)
            __m256 l0_lo, l0_hi, l1_lo, l1_hi;
            __m256 h0_lo, h0_hi, h1_lo, h1_hi;
            CB5K_QF(l0, lo0);
            CB5K_QF(l1, lo1);
            CB5K_QF(h0, hi0);
            CB5K_QF(h1, hi1);
#undef CB5K_QF
            const __m256 x0 = _mm256_loadu_ps(xb +  0);
            const __m256 x1 = _mm256_loadu_ps(xb +  8);
            const __m256 x2 = _mm256_loadu_ps(xb + 16);
            const __m256 x3 = _mm256_loadu_ps(xb + 24);
            const __m256 x4 = _mm256_loadu_ps(xb + 32);
            const __m256 x5 = _mm256_loadu_ps(xb + 40);
            const __m256 x6 = _mm256_loadu_ps(xb + 48);
            const __m256 x7 = _mm256_loadu_ps(xb + 56);
            __m256 dot_lo = _mm256_mul_ps(l0_lo, x0);
            dot_lo = _mm256_fmadd_ps(l0_hi, x1, dot_lo);
            dot_lo = _mm256_fmadd_ps(l1_lo, x2, dot_lo);
            dot_lo = _mm256_fmadd_ps(l1_hi, x3, dot_lo);
            __m256 dot_hi = _mm256_mul_ps(h0_lo, x4);
            dot_hi = _mm256_fmadd_ps(h0_hi, x5, dot_hi);
            dot_hi = _mm256_fmadd_ps(h1_lo, x6, dot_hi);
            dot_hi = _mm256_fmadd_ps(h1_hi, x7, dot_hi);
            const float v_lo = cb_hsum256(dot_lo);
            const float v_hi = cb_hsum256(dot_hi);
            const int isb = (j >> 6) * 2;
            uint8_t sc0, m0, sc1, m1;
            cb_get_scale_min_k4(isb + 0, sc, &sc0, &m0);
            cb_get_scale_min_k4(isb + 1, sc, &sc1, &m1);
            const float d1 = d * (float)sc0, d2 = d * (float)sc1;
            const float m1c = min * (float)m0, m2c = min * (float)m1;
            const float sx1 = cb_hsum32(xb);
            const float sx2 = cb_hsum32(xb + 32);
            const __m256 pack = _mm256_set_ps(d2 * v_hi,  d1 * v_lo,
                                              -m2c * sx2, -m1c * sx1,
                                              0.f, 0.f, 0.f, 0.f);
            acc = _mm256_add_ps(acc, pack);
            ql += 32;
            /* qh is NOT advanced within the super-block: same 32 bytes are
             * reused for all 4 chunks with rotating u1/u2 masks. */
            xb += 64;
        }
    }
    return cb_hsum256(acc);
}

/* Q6_K AVX2: ql[128], qh[64], sc[16] int8, fp16 d -> 210 B per 256.
 * 16 sub-blocks of 16 values, each with its own int8 scale. Two outer iters of
 * 128 vals, each spanning l=0..32. Per l: 4 sub-blocks, all in int8 [-32,31].
 * No min term: value = d*sc*q, contribution = d*sc * (q · x).
 *
 * Same correctness note as Q4_K: we must compute the full 16-elt dot product,
 * not the pair-sum approximation (raw fp32 x has no quantization budget for
 * the cross term). 16 int8 values -> cvtepi8_epi32 + cvtepi32_ps -> 16 fp32
 * q-values; mul against 16 fp32 x-values; horizontal sum.
 */
__attribute__((target("avx2,fma")))
static inline float row_dot_q6_K_avx2(const uint8_t *rw, const float *x,
                                      int K) {
    const int nb = K / QK_K;
    const __m128i f0    = _mm_set1_epi8(0x0F);
    const __m128i three = _mm_set1_epi8(0x03);
    const __m128i bias  = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rw + (long)b * QK6_K_BS;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + QK_K / 2;
        const int8_t  *sc = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        uint16_t dh;
        memcpy(&dh, blk + QK_K / 2 + QK_K / 4 + QK_K / 16, 2);
        const float d = cb_fp16_to_fp32(dh);
        const float *xb = x + (long)b * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            /* 64 bytes of ql, 32 bytes of qh. Per outer iter, l runs 0..32. */
            const __m256i qlA = _mm256_loadu_si256((const __m256i *)(ql + 0));
            const __m256i qlB = _mm256_loadu_si256((const __m256i *)(ql + 32));
            const __m256i qh_full = _mm256_loadu_si256((const __m256i *)qh);
            const __m128i qlA_lo = _mm256_castsi256_si128(qlA);  /* ql[l+ 0..16] */
            const __m128i qlA_hi = _mm256_extracti128_si256(qlA, 1);  /* ql[l+16..32] */
            const __m128i qlB_lo = _mm256_castsi256_si128(qlB);  /* ql[l+32..48] */
            const __m128i qlB_hi = _mm256_extracti128_si256(qlB, 1);  /* ql[l+48..64] */
            const __m128i qhA = _mm256_castsi256_si128(qh_full);  /* qh[l+ 0..16] */
            const __m128i qhB = _mm256_extracti128_si256(qh_full, 1);  /* qh[l+16..32] */
            /* Build 8 sub-blocks of 16 int8 each:
             *  q1_l (l=0..16, sc[0])  q1_h (l=16..32, sc[1])
             *  q2_l (sc[2])            q2_h (sc[3])
             *  q3_l (sc[4])            q3_h (sc[5])
             *  q4_l (sc[6])            q4_h (sc[7]) */
#define CB6K_BUILD(NAME, QH_BITS) do {                                          \
                const __m128i raw = _mm_and_si128(QL, f0);                      \
                const __m128i sh  = _mm_slli_epi16(QH_BITS, 4);                 \
                NAME = _mm_sub_epi8(_mm_or_si128(raw, sh), bias);               \
            } while (0)
            __m128i q1_l, q1_h, q2_l, q2_h, q3_l, q3_h, q4_l, q4_h;
            {
                const __m128i QL = qlA_lo;
                const __m128i qh_b = _mm_and_si128(qhA, three);
                CB6K_BUILD(q1_l, qh_b);
            }
            {
                const __m128i QL = qlA_hi;
                const __m128i qh_b = _mm_and_si128(qhB, three);
                CB6K_BUILD(q1_h, qh_b);
            }
            {
                const __m128i QL = qlB_lo;
                const __m128i qh_b = _mm_and_si128(_mm_srli_epi16(qhA, 2), three);
                CB6K_BUILD(q2_l, qh_b);
            }
            {
                const __m128i QL = qlB_hi;
                const __m128i qh_b = _mm_and_si128(_mm_srli_epi16(qhB, 2), three);
                CB6K_BUILD(q2_h, qh_b);
            }
            {
                const __m128i QL = _mm_srli_epi16(qlA_lo, 4);
                const __m128i qh_b = _mm_and_si128(_mm_srli_epi16(qhA, 4), three);
                CB6K_BUILD(q3_l, qh_b);
            }
            {
                const __m128i QL = _mm_srli_epi16(qlA_hi, 4);
                const __m128i qh_b = _mm_and_si128(_mm_srli_epi16(qhB, 4), three);
                CB6K_BUILD(q3_h, qh_b);
            }
            {
                const __m128i QL = _mm_srli_epi16(qlB_lo, 4);
                const __m128i qh_b = _mm_and_si128(_mm_srli_epi16(qhA, 6), three);
                CB6K_BUILD(q4_l, qh_b);
            }
            {
                const __m128i QL = _mm_srli_epi16(qlB_hi, 4);
                const __m128i qh_b = _mm_and_si128(_mm_srli_epi16(qhB, 6), three);
                CB6K_BUILD(q4_h, qh_b);
            }
#undef CB6K_BUILD
            /* Convert each 16-int8 q to two ymm of fp32 (8+8). The cvtepi8_epi32
             * extends signed int8 to int32; we need two 8-wide pieces per
             * 16-byte input, by splitting with _mm_bsrli_si128(..., 8). */
#define CB6K_QF(NAME, INP) do {                                                 \
                NAME##_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(INP));      \
                NAME##_hi = _mm256_cvtepi32_ps(                                 \
                    _mm256_cvtepi8_epi32(_mm_bsrli_si128(INP, 8)));             \
            } while (0)
            __m256 q1l_lo, q1l_hi, q1h_lo, q1h_hi;
            __m256 q2l_lo, q2l_hi, q2h_lo, q2h_hi;
            __m256 q3l_lo, q3l_hi, q3h_lo, q3h_hi;
            __m256 q4l_lo, q4l_hi, q4h_lo, q4h_hi;
            CB6K_QF(q1l, q1_l);
            CB6K_QF(q1h, q1_h);
            CB6K_QF(q2l, q2_l);
            CB6K_QF(q2h, q2_h);
            CB6K_QF(q3l, q3_l);
            CB6K_QF(q3h, q3_h);
            CB6K_QF(q4l, q4_l);
            CB6K_QF(q4h, q4_h);
#undef CB6K_QF
            /* 8 sub-dots: each is 16-elt dot product of int8-as-fp32 vs x. */
            const float *xbase = xb + n;
            const __m256 x0  = _mm256_loadu_ps(xbase +  0);
            const __m256 x1  = _mm256_loadu_ps(xbase +  8);
            const __m256 x2  = _mm256_loadu_ps(xbase + 16);
            const __m256 x3  = _mm256_loadu_ps(xbase + 24);
            const __m256 x4  = _mm256_loadu_ps(xbase + 32);
            const __m256 x5  = _mm256_loadu_ps(xbase + 40);
            const __m256 x6  = _mm256_loadu_ps(xbase + 48);
            const __m256 x7  = _mm256_loadu_ps(xbase + 56);
            const __m256 x8  = _mm256_loadu_ps(xbase + 64);
            const __m256 x9  = _mm256_loadu_ps(xbase + 72);
            const __m256 x10 = _mm256_loadu_ps(xbase + 80);
            const __m256 x11 = _mm256_loadu_ps(xbase + 88);
            const __m256 x12 = _mm256_loadu_ps(xbase + 96);
            const __m256 x13 = _mm256_loadu_ps(xbase +104);
            const __m256 x14 = _mm256_loadu_ps(xbase +112);
            const __m256 x15 = _mm256_loadu_ps(xbase +120);
            const float v1l = cb_hsum256(_mm256_fmadd_ps(q1l_lo, x0,
                                       _mm256_mul_ps(q1l_hi, x1)));
            const float v1h = cb_hsum256(_mm256_fmadd_ps(q1h_lo, x2,
                                       _mm256_mul_ps(q1h_hi, x3)));
            const float v2l = cb_hsum256(_mm256_fmadd_ps(q2l_lo, x4,
                                       _mm256_mul_ps(q2l_hi, x5)));
            const float v2h = cb_hsum256(_mm256_fmadd_ps(q2h_lo, x6,
                                       _mm256_mul_ps(q2h_hi, x7)));
            const float v3l = cb_hsum256(_mm256_fmadd_ps(q3l_lo, x8,
                                       _mm256_mul_ps(q3l_hi, x9)));
            const float v3h = cb_hsum256(_mm256_fmadd_ps(q3h_lo, x10,
                                       _mm256_mul_ps(q3h_hi, x11)));
            const float v4l = cb_hsum256(_mm256_fmadd_ps(q4l_lo, x12,
                                       _mm256_mul_ps(q4l_hi, x13)));
            const float v4h = cb_hsum256(_mm256_fmadd_ps(q4h_lo, x14,
                                       _mm256_mul_ps(q4h_hi, x15)));
            const float ds0 = d * (float)sc[0], ds1 = d * (float)sc[1];
            const float ds2 = d * (float)sc[2], ds3 = d * (float)sc[3];
            const float ds4 = d * (float)sc[4], ds5 = d * (float)sc[5];
            const float ds6 = d * (float)sc[6], ds7 = d * (float)sc[7];
            const __m256 pack = _mm256_set_ps(ds7 * v4h, ds6 * v4l,
                                              ds5 * v3h, ds4 * v3l,
                                              ds3 * v2h, ds2 * v2l,
                                              ds1 * v1h, ds0 * v1l);
            acc = _mm256_add_ps(acc, pack);
            ql += 64;
            qh += 32;
            sc += 8;
        }
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
        case TTQ_Q2_K:
            return row_dot_q2_K(rw, x, K);
        case TTQ_Q3_K:
            return row_dot_q3_K(rw, x, K);
        case TTQ_Q4_K:
#ifdef CB_X86
            if (use_avx2) return row_dot_q4_K_avx2(rw, x, K);
#endif
            return row_dot_q4_K(rw, x, K);
        case TTQ_Q5_K:
#ifdef CB_X86
            if (use_avx2) return row_dot_q5_K_avx2(rw, x, K);
#endif
            return row_dot_q5_K(rw, x, K);
        case TTQ_Q6_K:
#ifdef CB_X86
            if (use_avx2) return row_dot_q6_K_avx2(rw, x, K);
#endif
            return row_dot_q6_K(rw, x, K);
        default: return 0.0f;
    }
}

long tt_cpu_gemv(const void *W, int dtype, const float *x, float *y,
                 int M, int K, int n_threads) {
    if (!W || !x || !y || M <= 0 || K <= 0) return -1;
    if (dtype != TTQ_Q4_0 && dtype != TTQ_Q8_0 &&
        dtype != TTQ_Q2_K && dtype != TTQ_Q3_K && dtype != TTQ_Q4_K &&
        dtype != TTQ_Q5_K && dtype != TTQ_Q6_K) {
        fprintf(stderr, "[cpu-backend] unsupported dtype %d "
                "(Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K only)\n", dtype);
        return -100;
    }
    if (K % QK != 0 ||
        ((dtype == TTQ_Q2_K || dtype == TTQ_Q3_K || dtype == TTQ_Q4_K || dtype == TTQ_Q5_K || dtype == TTQ_Q6_K) &&
         K % QK_K != 0)) {
        fprintf(stderr, "[cpu-backend] K=%d not a multiple of %d%s\n", K, QK,
                (dtype == TTQ_Q2_K || dtype == TTQ_Q3_K || dtype == TTQ_Q4_K || dtype == TTQ_Q5_K ||
                 dtype == TTQ_Q6_K) ? " (K-quants need %256)" : "");
        return -101;
    }
    if (n_threads <= 0) n_threads = 1;

    long bs;
    switch (dtype) {
        case TTQ_Q4_0: bs = QK4_0_BS; break;
        case TTQ_Q8_0: bs = QK8_0_BS; break;
        case TTQ_Q2_K: bs = QK2_K_BS; break;
        case TTQ_Q3_K: bs = QK3_K_BS; break;
        case TTQ_Q4_K: bs = QK4_K_BS; break;
        case TTQ_Q5_K: bs = QK5_K_BS; break;
        default:       bs = QK6_K_BS; break;
    }
    const int row_vals = (bs >= QK2_K_BS) ? QK_K : QK;
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
        case TTQ_Q2_K: bs = QK2_K_BS; break;
        case TTQ_Q3_K: bs = QK3_K_BS; break;
        case TTQ_Q4_K: bs = QK4_K_BS; break;
        case TTQ_Q5_K: bs = QK5_K_BS; break;
        default:       bs = QK6_K_BS; break;
    }
    long wbytes = (long)M * (K / ((bs >= QK2_K_BS) ? QK_K : QK)) * bs;
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
