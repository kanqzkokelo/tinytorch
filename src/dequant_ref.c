/*
 * dequant_ref.c - CPU golden dequantization for all Tier-1 GGML quant types.
 *
 * Dequantization math is a direct port of llama.cpp ggml/src/ggml-quants.c
 * (dequantize_row_q4_0/q4_1/q5_0/q5_1/q8_0/q4_K/q5_K/q6_K); block layouts from
 * ggml/src/ggml-common.h. Credit: GGML / llama.cpp authors.
 * No llama.cpp code is linked; only the math semantics are reproduced.
 */
#include "dequant_ref.h"
#include "loader_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK_K  256
#define K_SCALE_SIZE 12

/* IEEE half -> float, bit-exact (weights are normal numbers in practice). */
static float fp16_to_fp32(uint16_t h) {
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
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ---------------- legacy block quants (math from ggml-quants.c) --------- */

static void dq_f16(const void *x, float *y, long k) {
    const uint16_t *p = (const uint16_t *)x;
    for (long i = 0; i < k; i++) y[i] = fp16_to_fp32(p[i]);
}

static void dq_f32(const void *x, float *y, long k) {
    memcpy(y, x, (size_t)k * sizeof(float));
}

static void dq_q4_0(const void *x, float *y, long k) {
    const uint8_t *b = (const uint8_t *)x;
    const long nb = k / QK4_0;
    for (long i = 0; i < nb; i++) {
        uint16_t dh;
        memcpy(&dh, b + i * 18, 2);
        const float d = fp16_to_fp32(dh);
        const uint8_t *qs = b + i * 18 + 2;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int x0 = (qs[j] & 0x0F) - 8;
            const int x1 = (qs[j] >>   4) - 8;
            y[i * QK4_0 + j]        = x0 * d;
            y[i * QK4_0 + j + QK4_0 / 2] = x1 * d;
        }
    }
}

static void dq_q4_1(const void *x, float *y, long k) {
    const uint8_t *b = (const uint8_t *)x;
    const long nb = k / QK4_1;
    for (long i = 0; i < nb; i++) {
        uint16_t dh, mh;
        memcpy(&dh, b + i * 20, 2);
        memcpy(&mh, b + i * 20 + 2, 2);
        const float d = fp16_to_fp32(dh);
        const float m = fp16_to_fp32(mh);
        const uint8_t *qs = b + i * 20 + 4;
        for (int j = 0; j < QK4_1 / 2; ++j) {
            const int x0 = (qs[j] & 0x0F);
            const int x1 = (qs[j] >>   4);
            y[i * QK4_1 + j]        = x0 * d + m;
            y[i * QK4_1 + j + QK4_1 / 2] = x1 * d + m;
        }
    }
}

static void dq_q5_0(const void *x, float *y, long k) {
    const uint8_t *b = (const uint8_t *)x;
    const long nb = k / QK5_0;
    for (long i = 0; i < nb; i++) {
        uint16_t dh;
        memcpy(&dh, b + i * 22, 2);
        const float d = fp16_to_fp32(dh);
        const uint8_t *qh = b + i * 22 + 2;
        const uint8_t *qs = b + i * 22 + 6;
        uint32_t qhv;
        memcpy(&qhv, qh, sizeof(qhv));
        for (int j = 0; j < QK5_0 / 2; ++j) {
            const uint8_t xh_0 = ((qhv >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qhv >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((qs[j] >>   4) | xh_1) - 16;
            y[i * QK5_0 + j]        = x0 * d;
            y[i * QK5_0 + j + QK5_0 / 2] = x1 * d;
        }
    }
}

static void dq_q5_1(const void *x, float *y, long k) {
    const uint8_t *b = (const uint8_t *)x;
    const long nb = k / QK5_1;
    for (long i = 0; i < nb; i++) {
        uint16_t dh, mh;
        memcpy(&dh, b + i * 24, 2);
        memcpy(&mh, b + i * 24 + 2, 2);
        const float d = fp16_to_fp32(dh);
        const float m = fp16_to_fp32(mh);
        const uint8_t *qh = b + i * 24 + 4;
        const uint8_t *qs = b + i * 24 + 8;
        uint32_t qhv;
        memcpy(&qhv, qh, sizeof(qhv));
        for (int j = 0; j < QK5_1 / 2; ++j) {
            const uint8_t xh_0 = ((qhv >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qhv >> (j + 12))     ) & 0x10;
            const int x0 = (qs[j] & 0x0F) | xh_0;
            const int x1 = (qs[j] >>   4) | xh_1;
            y[i * QK5_1 + j]        = x0 * d + m;
            y[i * QK5_1 + j + QK5_1 / 2] = x1 * d + m;
        }
    }
}

static void dq_q8_0(const void *x, float *y, long k) {
    const uint8_t *b = (const uint8_t *)x;
    const long nb = k / QK8_0;
    for (long i = 0; i < nb; i++) {
        uint16_t dh;
        memcpy(&dh, b + i * 34, 2);
        const float d = fp16_to_fp32(dh);
        const int8_t *qs = (const int8_t *)(b + i * 34 + 2);
        for (int j = 0; j < QK8_0; ++j) y[i * QK8_0 + j] = qs[j] * d;
    }
}

/* ---------------- K-quants (math from ggml-quants.c) --------------------- */

/* Port of get_scale_min_k4: decode 6-bit scale/min pairs packed into 12 bytes. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j + 0] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dq_q4_K(const void *x, float *y, long k) {
    /* block_q4_K: fp16 d, fp16 dmin, scales[12], qs[128] -> 144 B per 256 */
    const uint8_t *xb = (const uint8_t *)x;
    const long nb = k / QK_K;
    for (long i = 0; i < nb; i++) {
        const uint8_t *blk = xb + i * 144;
        const uint8_t *q   = blk + 4 + K_SCALE_SIZE;
        uint16_t dh, dmh;
        memcpy(&dh,  blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const float d   = fp16_to_fp32(dh);
        const float min = fp16_to_fp32(dmh);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, blk + 4, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, blk + 4, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void dq_q5_K(const void *x, float *y, long k) {
    /* block_q5_K: fp16 d, fp16 dmin, scales[12], qh[32], qs[128] -> 176 B/256 */
    const uint8_t *xb = (const uint8_t *)x;
    const long nb = k / QK_K;
    for (long i = 0; i < nb; i++) {
        const uint8_t *blk = xb + i * 176;
        /* block_q5_K field order: d,dmin | scales[12] | qh[32] | qs[128] */
        const uint8_t *qh  = blk + 4 + K_SCALE_SIZE;
        const uint8_t *ql  = qh + QK_K / 8;
        uint16_t dh, dmh;
        memcpy(&dh,  blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const float d   = fp16_to_fp32(dh);
        const float min = fp16_to_fp32(dmh);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, blk + 4, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, blk + 4, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dq_q6_K(const void *x, float *y, long k) {
    /* block_q6_K: ql[128], qh[64], scales[16] (int8), fp16 d -> 210 B per 256 */
    const uint8_t *xb = (const uint8_t *)x;
    const long nb = k / QK_K;
    for (long i = 0; i < nb; i++) {
        const uint8_t *blk = xb + i * 210;
        const uint8_t *ql = blk;             /* lower 4 bits  */
        const uint8_t *qh = blk + QK_K / 2;  /* upper 2 bits  */
        const int8_t *sc = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        uint16_t dh;
        memcpy(&dh, blk + QK_K / 2 + QK_K / 4 + QK_K / 16, 2);
        const float d = fp16_to_fp32(dh);

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ---------------- public API --------------------------------------------- */

long ttq_roundtrip(const char *gguf_path, const char *tensor_name,
                   int type_code, float *out, long out_cap) {
    GGUFModel *m = gguf_load(gguf_path);
    if (!m) return -1;
    GGUFTensor *t = gguf_get_tensor(m, tensor_name);
    if (!t) { fprintf(stderr, "[dequant_ref] tensor '%s' not found\n", tensor_name); gguf_free(m); return -2; }
    if ((int)t->type != type_code) {
        fprintf(stderr, "[dequant_ref] type mismatch: tensor '%s' is %d, expected %d\n",
                tensor_name, (int)t->type, type_code);
        gguf_free(m);
        return -3;
    }
    long numel = 1;
    for (int d = 0; d < t->ndim; d++) numel *= (long)t->shape[d];
    if (out_cap < numel) { fprintf(stderr, "[dequant_ref] out_cap %ld < %ld\n", out_cap, numel); gguf_free(m); return -4; }

    switch (t->type) {
        case TTQ_F32:  dq_f32(t->data, out, numel); break;
        case TTQ_F16:  dq_f16(t->data, out, numel); break;
        case TTQ_Q4_0: dq_q4_0(t->data, out, numel); break;
        case TTQ_Q4_1: dq_q4_1(t->data, out, numel); break;
        case TTQ_Q5_0: dq_q5_0(t->data, out, numel); break;
        case TTQ_Q5_1: dq_q5_1(t->data, out, numel); break;
        case TTQ_Q8_0: dq_q8_0(t->data, out, numel); break;
        case TTQ_Q4_K:
            if (numel % QK_K) { gguf_free(m); return -5; }
            dq_q4_K(t->data, out, numel); break;
        case TTQ_Q5_K:
            if (numel % QK_K) { gguf_free(m); return -5; }
            dq_q5_K(t->data, out, numel); break;
        case TTQ_Q6_K:
            if (numel % QK_K) { gguf_free(m); return -5; }
            dq_q6_K(t->data, out, numel); break;
        default:
            fprintf(stderr, "[dequant_ref] unsupported type %d\n", (int)t->type);
            gguf_free(m);
            return -5;
    }
    gguf_free(m);
    return numel;
}

#ifdef TTQ_MAIN
/* CLI: build/dequant_ref [--sizes] <gguf> [tensor] [out.bin|-]
 *   --sizes <gguf>          print "name type size_bytes offset" per tensor
 *   <gguf> <tensor> <out>   write dequantized float32 tensor to file ('-'=stdout)
 */
static const char *tt_type_name(int t) {
    switch (t) {
        case TTQ_F32: return "F32"; case TTQ_F16: return "F16";
        case TTQ_Q4_0: return "Q4_0"; case TTQ_Q4_1: return "Q4_1";
        case TTQ_Q5_0: return "Q5_0"; case TTQ_Q5_1: return "Q5_1";
        case TTQ_Q8_0: return "Q8_0"; case TTQ_Q4_K: return "Q4_K";
        case TTQ_Q5_K: return "Q5_K"; case TTQ_Q6_K: return "Q6_K";
        default: return "?";
    }
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--sizes") == 0) {
        GGUFModel *m = gguf_load(argv[2]);
        if (!m) return 1;
        for (int i = 0; i < m->tensor_count; i++)
            printf("%s\t%s\t%zu\t%llu\n", m->tensors[i].name,
                   tt_type_name((int)m->tensors[i].type),
                   m->tensors[i].size_bytes,
                   (unsigned long long)m->tensors[i].offset);
        gguf_free(m);
        return 0;
    }
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s [--sizes] <gguf> <tensor> [out.bin|-]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1], *name = argv[2];
    FILE *fo = stdout;
    if (argc == 4 && strcmp(argv[3], "-") != 0) {
        fo = fopen(argv[3], "wb");
        if (!fo) { perror("open output"); return 1; }
    }

    /* Peek tensor type to validate the requested code before allocating. */
    GGUFModel *m = gguf_load(path);
    if (!m) return 1;
    GGUFTensor *t = gguf_get_tensor(m, name);
    if (!t) { fprintf(stderr, "tensor '%s' not found\n", name); return 1; }
    int type_code = (int)t->type;
    long numel = 1;
    for (int d = 0; d < t->ndim; d++) numel *= (long)t->shape[d];
    gguf_free(m);

    float *out = malloc((size_t)numel * sizeof(float));
    if (!out) { fprintf(stderr, "oom (%ld floats)\n", numel); return 1; }
    long n = ttq_roundtrip(path, name, type_code, out, numel);
    if (n < 0) return 1;
    fwrite(out, sizeof(float), (size_t)n, fo);
    if (fo != stdout) fclose(fo);
    fprintf(stderr, "[dequant_ref] %s (%s): %ld floats written\n",
            name, tt_type_name(type_code), n);
    free(out);
    return 0;
}
#endif /* TTQ_MAIN */
