// Typed GEMV kernels for every Tier-1 GGML quant type (M7 task 2).
//
// One warp per output row, same pattern as k_gemv_q4_0 (kernels/gemv_q4_cuda.cu)
// but scalar byte loads: correctness first, the M6-tuned q4_0/q8_0 fast paths
// stay untouched and remain the ones the parity gate exercises.
//
// Math is a direct port of src/dequant_ref.c (itself ported from llama.cpp
// ggml/src/ggml-quants.c) — the CPU reference is the executable spec:
//   q4_1: 20B/32   d,m fp16; v = d*q + m
//   q5_0: 22B/32   d fp16, qh[4]; v = d*((qh_bit<<4|nib)-16)
//   q5_1: 24B/32   d,m fp16, qh[4]; v = d*(qh_bit<<4|nib) + m
//   q4_K: 144B/256 d,dmin fp16, scales[12], qs[128]; sub-block s of 32:
//         v = d*sc_s*q - dmin*m_s   (6-bit scale/min pairs, get_scale_min_k4)
//   q5_K: 176B/256 like q4_K + qh[32] high-bit plane (bit index = sub-block)
//   q6_K: 210B/256 ql[128], qh[64], int8 sc[16], d fp16;
//         v = d*sc[n/16]*((lo|hi6)-32)
//
// K-quants require n_per_row % 256 == 0 (asserted host-side in
// tt_gemv_typed with a clear error); legacy quants require % 32.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "dequant_ref.h"   /* TTQ_* type codes */

/* get_scale_min_k4 port (dequant_ref.c): decode 6-bit scale/min pair j. */
__device__ __forceinline__ void k4_scale_min(int j, const uint8_t *q,
                                             int *sc, int *mn) {
    if (j < 4) {
        *sc = q[j] & 63; *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ float half_at(const uint8_t *p) {
    return __half2float(*(const __half *)p);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/* ---------------- legacy 32-value block types ------------------------------ */

/* q4_1: lane strides 20B blocks; v = d*q + m */
__global__ void k_gemv_q4_1(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint8_t *rw = W + (long)row * nb * 20;
    float s = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = rw + b * 20;
        const float d = half_at(blk), m = half_at(blk + 2);
        const uint8_t *qs = blk + 4;
        const float *xb = x + b * 32;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            s += ((qs[j] & 0x0F) * d + m) * xb[j];
            s += ((qs[j] >>   4) * d + m) * xb[j + 16];
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* q5_0: 22B blocks; v = d*(((qh_bit<<4)|nib)-16).
 * qh bit mapping straight from dequant_ref.c dq_q5_0: value j uses bit j,
 * value j+16 uses bit j+12 (of the little-endian u32 at offset 2). */
__global__ void k_gemv_q5_0(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint8_t *rw = W + (long)row * nb * 22;
    float s = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = rw + b * 22;
        const float d = half_at(blk);
        uint32_t qhv;
        memcpy(&qhv, blk + 2, sizeof(qhv));
        const uint8_t *qs = blk + 6;
        const float *xb = x + b * 32;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int xh0 = (int)(((qhv >> (j +  0)) << 4) & 0x10);
            const int xh1 = (int)(((qhv >> (j + 12))     ) & 0x10);
            s += (((qs[j] & 0x0F) | xh0) - 16) * d * xb[j];
            s += (((qs[j] >>   4) | xh1) - 16) * d * xb[j + 16];
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* q5_1: 24B blocks; v = d*((qh_bit<<4)|nib) + m */
__global__ void k_gemv_q5_1(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint8_t *rw = W + (long)row * nb * 24;
    float s = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = rw + b * 24;
        const float d = half_at(blk), m = half_at(blk + 2);
        uint32_t qhv;
        memcpy(&qhv, blk + 4, sizeof(qhv));
        const uint8_t *qs = blk + 8;
        const float *xb = x + b * 32;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int xh0 = (int)(((qhv >> (j +  0)) << 4) & 0x10);
            const int xh1 = (int)(((qhv >> (j + 12))     ) & 0x10);
            s += (((qs[j] & 0x0F) | xh0) * d + m) * xb[j];
            s += (((qs[j] >>   4) | xh1) * d + m) * xb[j + 16];
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* q8_0 scalar twin of k_logits_q8_0 (the tuned one stays for the lm head). */
__global__ void k_gemv_q8_0(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint8_t *rw = W + (long)row * nb * 34;
    float s = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = rw + b * 34;
        const float d = half_at(blk);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + b * 32;
#pragma unroll
        for (int j = 0; j < 32; j++)
            s += (float)qs[j] * d * xb[j];
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* ---------------- K-quants: unit = sub-block of 32 values ------------------ *
 * Lane strides units u = sb*8 + s (sb super-block, s sub-block), so all
 * K/32 lanes are busy regardless of how many super-blocks fit in K.       */

/* q4_K: v = d*sc*nib - dmin*mn ; nibble low/high alternates by sub-block */
__global__ void k_gemv_q4_K(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nu = (K / 256) * 8;             /* sub-blocks per row */
    const uint8_t *rw = W + (long)row * (K / 256) * 144;
    float s = 0.0f;
    for (int u = lane; u < nu; u += 32) {
        const int sb = u >> 3, sub = u & 7;
        const uint8_t *blk = rw + sb * 144;
        const float d = half_at(blk), dmin = half_at(blk + 2);
        int sc, mn;
        k4_scale_min(sub, blk + 4, &sc, &mn);
        /* qs[128]: sub-block `sub` lives at bytes [sub/2*32, +32), low nibbles
         * for even sub, high nibbles for odd sub (dequant_ref.c dq_q4_K). */
        const uint8_t *q = blk + 16 + (sub >> 1) * 32;
        const float *xb = x + (long)sb * 256 + sub * 32;
        const float ds = d * sc, dm = dmin * mn;
        const int low = (sub & 1) == 0;
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const uint8_t byte = q[l];
            s += (ds * (low ? (byte & 0xF) : (byte >> 4)) - dm) * xb[l];
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* q5_K: q4_K + 5th bit plane qh[32]; value n's bit index == its sub-block id
 * (bit 2*(sub/2)+(sub&1) == sub — verified against dequant_ref.c dq_q5_K). */
__global__ void k_gemv_q5_K(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nu = (K / 256) * 8;
    const uint8_t *rw = W + (long)row * (K / 256) * 176;
    float s = 0.0f;
    for (int u = lane; u < nu; u += 32) {
        const int sb = u >> 3, sub = u & 7;
        const uint8_t *blk = rw + sb * 176;
        const float d = half_at(blk), dmin = half_at(blk + 2);
        int sc, mn;
        k4_scale_min(sub, blk + 4, &sc, &mn);
        const uint8_t *qh = blk + 16;                 /* qh[32] after scales */
        const uint8_t *ql = blk + 48 + (sub >> 1) * 32; /* qs after qh */
        const float *xb = x + (long)sb * 256 + sub * 32;
        const float ds = d * sc, dm = dmin * mn;
        const int low = (sub & 1) == 0;
        const int hmask = 16;                          /* added when bit set */
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const uint8_t byte = ql[l];
            const int hi = (qh[l] >> sub) & 1;
            s += (ds * ((low ? (byte & 0xF) : (byte >> 4)) + hi * hmask) - dm) * xb[l];
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* q6_K: v = d*sc[n/16]*((lo|hi6)-32).
 * Per-value layout derivation from dequant_ref.c dq_q6_K, with r = n%128,
 * c = chunk (n/128): ql byte c*64+(r&63) low nibble for r<64 / high for
 * r>=64; qh byte c*32+(r&31), 2-bit field shifted 2*(r>>5); scale idx
 * c*8+(r>>4). */
__global__ void k_gemv_q6_K(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nu = (K / 256) * 8;
    const uint8_t *rw = W + (long)row * (K / 256) * 210;
    float s = 0.0f;
    for (int u = lane; u < nu; u += 32) {
        const int sb = u >> 3, sub = u & 7;
        const uint8_t *blk = rw + sb * 210;
        const uint8_t *ql = blk;                       /* ql[128] @0      */
        const uint8_t *qh = blk + 128;                 /* qh[64] @128     */
        const int8_t *sc = (const int8_t *)(blk + 192);/* int8 scales @192*/
        const float d = half_at(blk + 208);            /* fp16 d @208     */
        const float *xb = x + (long)sb * 256 + sub * 32;
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const int n = sub * 32 + l;                /* within super-block */
            const int c = n >> 7, r = n & 127;
            const uint8_t qlb = ql[c * 64 + (r & 63)];
            const int lo = (r < 64) ? (qlb & 0xF) : (qlb >> 4);
            const int hi = (qh[c * 32 + (r & 31)] >> (2 * (r >> 5))) & 3;
            const int q = (lo | (hi << 4)) - 32;
            s += d * (float)sc[c * 8 + (r >> 4)] * (float)q * xb[l];
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* ---------------- f16 / f32 (Tier-1 completeness) -------------------------- */

__global__ void k_gemv_f16(const uint8_t *__restrict__ W,
                           const float *__restrict__ x, float *__restrict__ y,
                           int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const __half *rw = (const __half *)((const char *)W + (long)row * K * 2);
    float s = 0.0f;
    for (int i = lane; i < K; i += 32)
        s += __half2float(rw[i]) * x[i];
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

__global__ void k_gemv_f32(const float *__restrict__ W,
                           const float *__restrict__ x, float *__restrict__ y,
                           int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const float *rw = W + (long)row * K;
    float s = 0.0f;
    for (int i = lane; i < K; i += 32)
        s += rw[i] * x[i];
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

/* ---------------- embedding row dequant (typed) ---------------------------- *
 * One thread per 32-value unit writes 32 outputs; same math as above.       */

__global__ void k_embed_q4_1(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int b = threadIdx.x + blockIdx.x * blockDim.x;
    const int nb = dim / 32;
    if (b >= nb) return;
    const uint8_t *blk = W + (long)tok * nb * 20 + b * 20;
    const float d = half_at(blk), m = half_at(blk + 2);
    const uint8_t *qs = blk + 4;
    float *out = dx + b * 32;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        out[j]      = (qs[j] & 0x0F) * d + m;
        out[j + 16] = (qs[j] >>   4) * d + m;
    }
}

__global__ void k_embed_q5_0(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int b = threadIdx.x + blockIdx.x * blockDim.x;
    const int nb = dim / 32;
    if (b >= nb) return;
    const uint8_t *blk = W + (long)tok * nb * 22 + b * 22;
    const float d = half_at(blk);
    uint32_t qhv; memcpy(&qhv, blk + 2, sizeof(qhv));
    const uint8_t *qs = blk + 6;
    float *out = dx + b * 32;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        out[j]      = (((qs[j] & 0x0F) | (int)(((qhv >> (j +  0)) << 4) & 0x10)) - 16) * d;
        out[j + 16] = (((qs[j] >>   4) | (int)(((qhv >> (j + 12))     ) & 0x10)) - 16) * d;
    }
}

__global__ void k_embed_q5_1(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int b = threadIdx.x + blockIdx.x * blockDim.x;
    const int nb = dim / 32;
    if (b >= nb) return;
    const uint8_t *blk = W + (long)tok * nb * 24 + b * 24;
    const float d = half_at(blk), m = half_at(blk + 2);
    uint32_t qhv; memcpy(&qhv, blk + 4, sizeof(qhv));
    const uint8_t *qs = blk + 8;
    float *out = dx + b * 32;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        out[j]      = ((qs[j] & 0x0F) | (int)(((qhv >> (j +  0)) << 4) & 0x10)) * d + m;
        out[j + 16] = ((qs[j] >>   4) | (int)(((qhv >> (j + 12))     ) & 0x10)) * d + m;
    }
}

__global__ void k_embed_q8_0(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int b = threadIdx.x + blockIdx.x * blockDim.x;
    const int nb = dim / 32;
    if (b >= nb) return;
    const uint8_t *blk = W + (long)tok * nb * 34 + b * 34;
    const float d = half_at(blk);
    const int8_t *qs = (const int8_t *)(blk + 2);
    float *out = dx + b * 32;
#pragma unroll
    for (int j = 0; j < 32; j++) out[j] = (float)qs[j] * d;
}

__global__ void k_embed_q4_K(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int u = threadIdx.x + blockIdx.x * blockDim.x;
    const int nu = (dim / 256) * 8;
    if (u >= nu) return;
    const int sb = u >> 3, sub = u & 7;
    const uint8_t *blk = W + (long)tok * (dim / 256) * 144 + sb * 144;
    const float d = half_at(blk), dmin = half_at(blk + 2);
    int sc, mn;
    k4_scale_min(sub, blk + 4, &sc, &mn);
    const uint8_t *q = blk + 16 + (sub >> 1) * 32;
    float *out = dx + (long)sb * 256 + sub * 32;
    const float ds = d * sc, dm = dmin * mn;
    const int low = (sub & 1) == 0;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        const uint8_t byte = q[l];
        out[l] = ds * (low ? (byte & 0xF) : (byte >> 4)) - dm;
    }
}

__global__ void k_embed_q5_K(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int u = threadIdx.x + blockIdx.x * blockDim.x;
    const int nu = (dim / 256) * 8;
    if (u >= nu) return;
    const int sb = u >> 3, sub = u & 7;
    const uint8_t *blk = W + (long)tok * (dim / 256) * 176 + sb * 176;
    const float d = half_at(blk), dmin = half_at(blk + 2);
    int sc, mn;
    k4_scale_min(sub, blk + 4, &sc, &mn);
    const uint8_t *qh = blk + 16;
    const uint8_t *ql = blk + 48 + (sub >> 1) * 32;
    float *out = dx + (long)sb * 256 + sub * 32;
    const float ds = d * sc, dm = dmin * mn;
    const int low = (sub & 1) == 0;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        const uint8_t byte = ql[l];
        const int hi = (qh[l] >> sub) & 1;
        out[l] = ds * ((low ? (byte & 0xF) : (byte >> 4)) + hi * 16) - dm;
    }
}

__global__ void k_embed_q6_K(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int u = threadIdx.x + blockIdx.x * blockDim.x;
    const int nu = (dim / 256) * 8;
    if (u >= nu) return;
    const int sb = u >> 3, sub = u & 7;
    const uint8_t *blk = W + (long)tok * (dim / 256) * 210 + sb * 210;
    const uint8_t *ql = blk, *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    const float d = half_at(blk + 208);
    float *out = dx + (long)sb * 256 + sub * 32;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        const int n = sub * 32 + l;
        const int c = n >> 7, r = n & 127;
        const uint8_t qlb = ql[c * 64 + (r & 63)];
        const int lo = (r < 64) ? (qlb & 0xF) : (qlb >> 4);
        const int hi = (qh[c * 32 + (r & 31)] >> (2 * (r >> 5))) & 3;
        out[l] = d * (float)sc[c * 8 + (r >> 4)] * (float)((lo | (hi << 4)) - 32);
    }
}

__global__ void k_embed_f16(const uint8_t *__restrict__ W, int tok,
                            float *__restrict__ dx, int dim) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= dim) return;
    dx[i] = __half2float(((const __half *)((const char *)W + (long)tok * dim * 2))[i]);
}

__global__ void k_embed_f32(const float *__restrict__ W, int tok,
                            float *__restrict__ dx, int dim) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= dim) return;
    dx[i] = W[(long)tok * dim + i];
}

/* ---------------- host launchers ------------------------------------------ */

extern "C" {

static int gemv_dims(int M, dim3 *grid, dim3 *block) {
    if (M <= 0 || M > (1 << 22)) return -50;
    block->x = 32; block->y = 16; block->z = 1;
    grid->x = (M + block->y - 1) / block->y; grid->y = 1; grid->z = 1;
    return 0;
}

static int kquant_aligned(int dtype, long K) { return K % 256 == 0; }

/*
 * y[M] = W[M,K] @ x[K] with W stored in any supported dtype.
 * Returns 0 or a cuda error code; -100 on unsupported dtype;
 * -101 on K-quant alignment violation (K % 256 != 0, message on stderr).
 */
int tt_gemv_typed(const void *W, int dtype, const float *x, float *y,
                  int M, int K, cudaStream_t stream) {
    dim3 g, b;
    int rc = gemv_dims(M, &g, &b);
    if (rc) return rc;
    if (getenv("TT_DEBUG2"))
        fprintf(stderr, "[gemv] W=%p dt=%d x=%p y=%p M=%d K=%d\n",
                W, dtype, (const void *)x, (void *)y, M, K);

    switch (dtype) {
        case TTQ_Q4_0: {
            /* delegate to the M6.3b-tuned kernel (same launch contract) */
            extern int tt_gemv_q4_0(const void *, const float *, float *,
                                    int, int, cudaStream_t);
            return tt_gemv_q4_0(W, x, y, M, K, stream);
        }
        case TTQ_Q4_1:
            k_gemv_q4_1<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        case TTQ_Q5_0:
            k_gemv_q5_0<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        case TTQ_Q5_1:
            k_gemv_q5_1<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        case TTQ_Q8_0:
            k_gemv_q8_0<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        case TTQ_Q4_K:
        case TTQ_Q5_K:
        case TTQ_Q6_K:
            if (!kquant_aligned(dtype, K)) {
                fprintf(stderr, "[gemv-typed] K=%d not a multiple of 256 — "
                                "K-quant GEMV needs n_per_row %% 256 == 0 "
                                "(dtype %d)\n", K, dtype);
                return -101;
            }
            if (dtype == TTQ_Q4_K)
                k_gemv_q4_K<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            else if (dtype == TTQ_Q5_K)
                k_gemv_q5_K<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            else
                k_gemv_q6_K<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        case TTQ_F16:
            k_gemv_f16<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        case TTQ_F32:
            k_gemv_f32<<<g, b, 0, stream>>>((const float *)W, x, y, M, K);
            break;
        default:
            fprintf(stderr, "[gemv-typed] unsupported dtype %d\n", dtype);
            return -100;
    }
    return (int)cudaGetLastError();
}

/* Typed logits projection for everything that is NOT q4_0/q8_0 (those two
 * keep their tuned kernels inside tt_logits_dispatch). Logits IS a gemv:
 * identical kernels, one row per vocab entry. */
int tt_logits_typed(const void *dW, int dtype, const float *dx,
                    float *dlogits, int vocab, int K, cudaStream_t stream) {
    return tt_gemv_typed(dW, dtype, dx, dlogits, vocab, K, stream);
}

/* Typed embedding row lookup. Returns 0 or cuda err; -100 unsupported. */
int tt_embed_typed(const void *dW, int dtype, int tok, float *dx, int dim,
                   cudaStream_t stream) {
    switch (dtype) {
        case TTQ_Q4_0: {
            extern int tt_embed_q4_0(const void *, int, float *, int, cudaStream_t);
            return tt_embed_q4_0(dW, tok, dx, dim, stream);
        }
        case TTQ_Q4_1:
            k_embed_q4_1<<<(dim / 32 + 255) / 256, 256, 0, stream>>>(
                (const uint8_t *)dW, tok, dx, dim);
            break;
        case TTQ_Q5_0:
            k_embed_q5_0<<<(dim / 32 + 255) / 256, 256, 0, stream>>>(
                (const uint8_t *)dW, tok, dx, dim);
            break;
        case TTQ_Q5_1:
            k_embed_q5_1<<<(dim / 32 + 255) / 256, 256, 0, stream>>>(
                (const uint8_t *)dW, tok, dx, dim);
            break;
        case TTQ_Q8_0:
            k_embed_q8_0<<<(dim / 32 + 255) / 256, 256, 0, stream>>>(
                (const uint8_t *)dW, tok, dx, dim);
            break;
        case TTQ_Q4_K:
        case TTQ_Q5_K:
        case TTQ_Q6_K: {
            if (!kquant_aligned(dtype, dim)) {
                fprintf(stderr, "[gemv-typed] embed dim=%d not multiple of 256 "
                                "for K-quant dtype %d\n", dim, dtype);
                return -101;
            }
            const int nu = (dim / 256) * 8;
            if (dtype == TTQ_Q4_K)
                k_embed_q4_K<<<(nu + 255) / 256, 256, 0, stream>>>(
                    (const uint8_t *)dW, tok, dx, dim);
            else if (dtype == TTQ_Q5_K)
                k_embed_q5_K<<<(nu + 255) / 256, 256, 0, stream>>>(
                    (const uint8_t *)dW, tok, dx, dim);
            else
                k_embed_q6_K<<<(nu + 255) / 256, 256, 0, stream>>>(
                    (const uint8_t *)dW, tok, dx, dim);
            break;
        }
        case TTQ_F16:
            k_embed_f16<<<(dim + 255) / 256, 256, 0, stream>>>(
                (const uint8_t *)dW, tok, dx, dim);
            break;
        case TTQ_F32:
            k_embed_f32<<<(dim + 255) / 256, 256, 0, stream>>>(
                (const float *)dW, tok, dx, dim);
            break;
        default:
            fprintf(stderr, "[gemv-typed] embed: unsupported dtype %d\n", dtype);
            return -100;
    }
    return (int)cudaGetLastError();
}

} /* extern "C" */
