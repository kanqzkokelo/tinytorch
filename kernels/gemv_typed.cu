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
#include <cuda_bf16.h>
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

/* q3_K: 110B super-blocks of 256 values (16 sub-blocks of 16 values) */
__global__ void k_gemv_q3_K(const uint8_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nsb = K / 256;
    const int nu = nsb * 16;
    const uint8_t *rw = W + (long)row * nsb * 110;
    float s = 0.0f;

    for (int u = lane; u < nu; u += 32) {
        const int sb  = u >> 4;
        const int is  = u & 15;
        const int n   = is >> 3;
        const int j   = (is & 7) >> 1;
        const int is0 = is & 1;
        const int shift = j << 1;
        const uint8_t m = 1 << (4 * n + j);

        const uint8_t *blk = rw + sb * 110;
        const uint8_t *sc_raw = blk + 96;
        int8_t us = is <  4 ? (sc_raw[is-0] & 0xF) | (((sc_raw[is+8] >> 0) & 3) << 4) :
                    is <  8 ? (sc_raw[is-0] & 0xF) | (((sc_raw[is+4] >> 2) & 3) << 4) :
                    is < 12 ? (sc_raw[is-8] >>  4) | (((sc_raw[is+0] >> 4) & 3) << 4) :
                              (sc_raw[is-8] >>  4) | (((sc_raw[is-4] >> 6) & 3) << 4);

        const float d = half_at(blk + 108);
        const float dl = d * (float)(us - 32);

        const uint8_t *q  = blk + 32 + 32 * n + 16 * is0;
        const uint8_t *hm = blk + 16 * is0;
        const float *xb = x + (long)sb * 256 + is * 16;

#pragma unroll
        for (int l = 0; l < 16; l++) {
            int8_t w = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
            s += dl * (float)w * xb[l];
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

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

/* ---------------- K-quant V2 (M9.5) ---------------------------------------- *
 * 2-rows-per-warp vectorized variants of the K-quants above. Same dequant
 * math (port of dequant_ref.c dq_q{4,5,6}_K), same 32-lane-per-warp shape,
 * but each warp computes TWO output rows and the two rows share every x
 * read (halves x re-read pressure for the bandwidth-bound M=4864 hidden
 * matrices). Inner loop hoists sub-block scale/min/ds/dm once per sub-block
 * (vs per-element in the scalar version), so each sub-block contributes
 * `ds*sum(w*x) - dm*sum(x)` -- 1 fma pair instead of 32 (matches the
 * llama.cpp vec_dot_q4_K_q8_1_impl_vmmq shape without requiring Q8_1
 * activation quantization). x is read as 4 float4 chunks (32 floats ==
 * 1 sub-block), so each lane issues 4 float4 loads per sub-block; the two
 * rows share the loads and only the weight bytes double in flight.
 *
 * Contracts (caller must check via tt_gemv_typed gate):
 *   - K must be a multiple of 256 (sub-block = 32, super-block = 256).
 *   - M padded to even (caller does it, see tt_gemv_typed).
 *   - M >= TT_KQUANT_V2_MIN_M (small-M shapes still go to scalar, same
 *     reasoning as the q4_0/q8_0 V2 gate -- the per-warp setup overhead
 *     exceeds the saved x re-reads for K/V projections).
 *   - x 32-float chunks (sub-blocks) are 16-byte aligned: x base is
 *     16-byte aligned (engine guarantee), and any 32-element offset is
 *     a multiple of 128 bytes which is 16-byte aligned.
 *   - qs byte chunks are 4-byte aligned for the uint32 word reads below
 *     (Q4_K qs[128] @ super-block offset 16, Q5_K ql[128] @48, Q6_K
 *      ql[128] @0; super-block stride is 144/176/210, so byte offsets
 *      are 16*N*144, 16*N*176, 16*N*210 -- all multiples of 16, hence
 *      4-byte aligned when 0 <= 16*N < stride; the 4-byte alignment of
 *      the qs chunk address = (16 + 16*N*stride) mod 4 = 0 for any
 *      N >= 0, even for Q6_K's 210 stride).
 */

/* 2-rows-per-warp V2 for q3_K. 16 sub-blocks of 16 weights = 256 weights. */
__global__ void k_gemv_q3_K_v2(const uint8_t *__restrict__ W,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nsb = K / 256;
    const int nu  = nsb * 16;

    const uint8_t *rw0 = W + (long)row0 * nsb * 110;
    const uint8_t *rw1 = W + (long)row1 * nsb * 110;
    float s0 = 0.0f, s1 = 0.0f;

    for (int u = lane; u < nu; u += 32) {
        const int sb  = u >> 4;
        const int is  = u & 15;
        const int n   = is >> 3;
        const int j   = (is & 7) >> 1;
        const int is0 = is & 1;
        const int shift = j << 1;
        const uint8_t m = 1 << (4 * n + j);

        const uint8_t *blk0 = rw0 + sb * 110;
        const uint8_t *blk1 = rw1 + sb * 110;

        const uint8_t *sc0_raw = blk0 + 96;
        const uint8_t *sc1_raw = blk1 + 96;

        int8_t us0 = is <  4 ? (sc0_raw[is-0] & 0xF) | (((sc0_raw[is+8] >> 0) & 3) << 4) :
                     is <  8 ? (sc0_raw[is-0] & 0xF) | (((sc0_raw[is+4] >> 2) & 3) << 4) :
                     is < 12 ? (sc0_raw[is-8] >>  4) | (((sc0_raw[is+0] >> 4) & 3) << 4) :
                               (sc0_raw[is-8] >>  4) | (((sc0_raw[is-4] >> 6) & 3) << 4);

        int8_t us1 = is <  4 ? (sc1_raw[is-0] & 0xF) | (((sc1_raw[is+8] >> 0) & 3) << 4) :
                     is <  8 ? (sc1_raw[is-0] & 0xF) | (((sc1_raw[is+4] >> 2) & 3) << 4) :
                     is < 12 ? (sc1_raw[is-8] >>  4) | (((sc1_raw[is+0] >> 4) & 3) << 4) :
                               (sc1_raw[is-8] >>  4) | (((sc1_raw[is-4] >> 6) & 3) << 4);

        const float d0 = half_at(blk0 + 108);
        const float d1 = half_at(blk1 + 108);
        const float dl0 = d0 * (float)(us0 - 32);
        const float dl1 = d1 * (float)(us1 - 32);

        const uint8_t *q0  = blk0 + 32 + 32 * n + 16 * is0;
        const uint8_t *q1  = blk1 + 32 + 32 * n + 16 * is0;
        const uint8_t *hm0 = blk0 + 16 * is0;
        const uint8_t *hm1 = blk1 + 16 * is0;

        const float *xb = x + (long)sb * 256 + is * 16;
        const float4 *x4 = (const float4 *)xb;

#pragma unroll
        for (int c = 0; c < 4; c++) {
            const float4 xv = x4[c];
            const int base_l = c * 4;

            int8_t w0_0 = ((q0[base_l + 0] >> shift) & 3) - ((hm0[base_l + 0] & m) ? 0 : 4);
            int8_t w0_1 = ((q0[base_l + 1] >> shift) & 3) - ((hm0[base_l + 1] & m) ? 0 : 4);
            int8_t w0_2 = ((q0[base_l + 2] >> shift) & 3) - ((hm0[base_l + 2] & m) ? 0 : 4);
            int8_t w0_3 = ((q0[base_l + 3] >> shift) & 3) - ((hm0[base_l + 3] & m) ? 0 : 4);

            int8_t w1_0 = ((q1[base_l + 0] >> shift) & 3) - ((hm1[base_l + 0] & m) ? 0 : 4);
            int8_t w1_1 = ((q1[base_l + 1] >> shift) & 3) - ((hm1[base_l + 1] & m) ? 0 : 4);
            int8_t w1_2 = ((q1[base_l + 2] >> shift) & 3) - ((hm1[base_l + 2] & m) ? 0 : 4);
            int8_t w1_3 = ((q1[base_l + 3] >> shift) & 3) - ((hm1[base_l + 3] & m) ? 0 : 4);

            s0 += dl0 * (w0_0 * xv.x + w0_1 * xv.y + w0_2 * xv.z + w0_3 * xv.w);
            s1 += dl1 * (w1_0 * xv.x + w1_1 * xv.y + w1_2 * xv.z + w1_3 * xv.w);
        }
    }

    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

/* 2-rows-per-warp V2 for q4_K. Hoists (d*sc, dmin*m) per sub-block and
 * uses the ds*sum(w*x) - dm*sum(x) form so each sub-block's two scale
 * multiplies amortize over 32 weight-x MACs. Weight bytes loaded via
 * __ldg (read-only cache). Mirrors the k_gemv_q4_0 V2 shape (one warp,
 * two rows, lane-stride) so existing grid/block helpers apply. */
__global__ void k_gemv_q4_K_v2(const uint8_t *__restrict__ W,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;                    // caller pads M to even
    const int lane = threadIdx.x;
    const int nsb = K / 256;                      // super-blocks per row
    const int nu  = nsb * 8;                      // sub-blocks per row
    const uint8_t *rw0 = W + (long)row0 * nsb * 144;
    const uint8_t *rw1 = W + (long)row1 * nsb * 144;
    float s0 = 0.0f, s1 = 0.0f;

    for (int u = lane; u < nu; u += 32) {
        const int sb = u >> 3, sub = u & 7;
        const uint8_t *blk0 = rw0 + sb * 144;
        const uint8_t *blk1 = rw1 + sb * 144;
        const float d0  = half_at(blk0);
        const float dm0 = half_at(blk0 + 2);
        const float d1  = half_at(blk1);
        const float dm1 = half_at(blk1 + 2);
        int sc0, mn0, sc1, mn1;
        k4_scale_min(sub, blk0 + 4, &sc0, &mn0);
        k4_scale_min(sub, blk1 + 4, &sc1, &mn1);
        /* qs[128]: sub-block `sub` at bytes [sub/2*32, +32). Even sub =
         * low nibble, odd sub = high nibble (dequant_ref.c dq_q4_K). */
        const uint8_t *q0 = blk0 + 16 + (sub >> 1) * 32;
        const uint8_t *q1 = blk1 + 16 + (sub >> 1) * 32;
        const float *xb = x + (long)sb * 256 + sub * 32;
        const float ds0 = d0 * sc0, dmd0 = dm0 * mn0;
        const float ds1 = d1 * sc1, dmd1 = dm1 * mn1;
        const int low = (sub & 1) == 0;
        const float4 *x4 = (const float4 *)xb;
        float swx = 0.0f, sx = 0.0f;     /* shared x stats; both rows reuse */
        /* 2 chunks of 16 bytes (8 nibbles) per sub-block = 32 nibbles.
         * uint32 (4 bytes) packs 4 nibbles paired with one float4. */
#pragma unroll
        for (int q = 0; q < 2; q++) {
            const uint32_t w0 = __ldg((const uint32_t *)(q0 + q * 16));
            const uint32_t w1 = __ldg((const uint32_t *)(q1 + q * 16));
            const float4 xv0 = x4[q * 2 + 0];
            const float4 xv1 = x4[q * 2 + 1];
            /* low: byte 0 = bits 0..3, byte 1 = bits 8..11, ...  */
            /* high: byte 0 = bits 4..7, byte 1 = bits 12..15, ... */
            int a0, a1, a2, a3, b0, b1, b2, b3;
            if (low) {
                a0 = (int)( w0        & 0xFu);
                a1 = (int)((w0 >>  8) & 0xFu);
                a2 = (int)((w0 >> 16) & 0xFu);
                a3 = (int)((w0 >> 24)       );
                b0 = (int)( w1        & 0xFu);
                b1 = (int)((w1 >>  8) & 0xFu);
                b2 = (int)((w1 >> 16) & 0xFu);
                b3 = (int)((w1 >> 24)       );
            } else {
                a0 = (int)((w0 >>  4) & 0xFu);
                a1 = (int)((w0 >> 12) & 0xFu);
                a2 = (int)((w0 >> 20) & 0xFu);
                a3 = (int)( w0 >> 28       );
                b0 = (int)((w1 >>  4) & 0xFu);
                b1 = (int)((w1 >> 12) & 0xFu);
                b2 = (int)((w1 >> 20) & 0xFu);
                b3 = (int)( w1 >> 28       );
            }
            swx += a0 * xv0.x + a1 * xv0.y + a2 * xv0.z + a3 * xv0.w
                 + b0 * xv1.x + b1 * xv1.y + b2 * xv1.z + b3 * xv1.w;
            sx  += xv0.x + xv0.y + xv0.z + xv0.w
                 + xv1.x + xv1.y + xv1.z + xv1.w;
        }
        s0 += ds0 * swx - dmd0 * sx;
        s1 += ds1 * swx - dmd1 * sx;
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

/* 2-rows-per-warp V2 for q5_K. Same shape as q4_K V2, plus the 5th-bit
 * qh plane. For sub-block `sub`, value l's high bit is `(qh[l] >> sub) & 1`
 * and adds 16 to the 4-bit value (so the 5-bit integer weight is in
 * [0, 31]). Each sub-block uses 32 qh bits = 4 uint32 words for q=0..1.
 * Note the qh bytes for sub-block `sub` are at super-block+16+l (byte l,
 * bit sub). Reading qh[q*16..q*16+4] as uint32 gives bits for indices
 * [q*16, q*16+3]; bit positions within each byte are `sub`. */
__global__ void k_gemv_q5_K_v2(const uint8_t *__restrict__ W,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nsb = K / 256;
    const int nu  = nsb * 8;
    const uint8_t *rw0 = W + (long)row0 * nsb * 176;
    const uint8_t *rw1 = W + (long)row1 * nsb * 176;
    float s0 = 0.0f, s1 = 0.0f;

    for (int u = lane; u < nu; u += 32) {
        const int sb = u >> 3, sub = u & 7;
        const uint8_t *blk0 = rw0 + sb * 176;
        const uint8_t *blk1 = rw1 + sb * 176;
        const float d0  = half_at(blk0);
        const float dm0 = half_at(blk0 + 2);
        const float d1  = half_at(blk1);
        const float dm1 = half_at(blk1 + 2);
        int sc0, mn0, sc1, mn1;
        k4_scale_min(sub, blk0 + 4, &sc0, &mn0);
        k4_scale_min(sub, blk1 + 4, &sc1, &mn1);
        const uint8_t *qh0 = blk0 + 16;                       /* qh[32] @16  */
        const uint8_t *qh1 = blk1 + 16;
        const uint8_t *ql0 = blk0 + 48 + (sub >> 1) * 32;    /* ql[128] @48 */
        const uint8_t *ql1 = blk1 + 48 + (sub >> 1) * 32;
        const float *xb = x + (long)sb * 256 + sub * 32;
        const float ds0 = d0 * sc0, dmd0 = dm0 * mn0;
        const float ds1 = d1 * sc1, dmd1 = dm1 * mn1;
        const int low = (sub & 1) == 0;
        const float4 *x4 = (const float4 *)xb;
        float swx = 0.0f, sx = 0.0f;
#pragma unroll
        for (int q = 0; q < 2; q++) {
            const uint32_t w0  = __ldg((const uint32_t *)(ql0 + q * 16));
            const uint32_t w1  = __ldg((const uint32_t *)(ql1 + q * 16));
            const uint32_t h0  = __ldg((const uint32_t *)(qh0 + q * 16));
            const uint32_t h1  = __ldg((const uint32_t *)(qh1 + q * 16));
            const float4 xv0 = x4[q * 2 + 0];
            const float4 xv1 = x4[q * 2 + 1];
            /* High bit for value l: (qh[l] >> sub) & 1. For 4 packed bytes
             * loaded as uint32, bit `sub` of byte l = bit (l*8 + sub) of
             * the uint32. */
            int a0, a1, a2, a3, b0, b1, b2, b3;
            const unsigned s0_ = (unsigned)sub;
            if (low) {
                a0 = (int)( w0        & 0xFu) | (int)((h0      ) >> s0_ & 1u) << 4;
                a1 = (int)((w0 >>  8) & 0xFu) | (int)((h0 >>  8) >> s0_ & 1u) << 4;
                a2 = (int)((w0 >> 16) & 0xFu) | (int)((h0 >> 16) >> s0_ & 1u) << 4;
                a3 = (int)((w0 >> 24)       ) | (int)((h0 >> 24) >> s0_ & 1u) << 4;
                b0 = (int)( w1        & 0xFu) | (int)((h1      ) >> s0_ & 1u) << 4;
                b1 = (int)((w1 >>  8) & 0xFu) | (int)((h1 >>  8) >> s0_ & 1u) << 4;
                b2 = (int)((w1 >> 16) & 0xFu) | (int)((h1 >> 16) >> s0_ & 1u) << 4;
                b3 = (int)((w1 >> 24)       ) | (int)((h1 >> 24) >> s0_ & 1u) << 4;
            } else {
                a0 = (int)((w0 >>  4) & 0xFu) | (int)((h0 >>  4) >> s0_ & 1u) << 4;
                a1 = (int)((w0 >> 12) & 0xFu) | (int)((h0 >> 12) >> s0_ & 1u) << 4;
                a2 = (int)((w0 >> 20) & 0xFu) | (int)((h0 >> 20) >> s0_ & 1u) << 4;
                a3 = (int)( w0 >> 28       ) | (int)((h0 >> 28) >> s0_ & 1u) << 4;
                b0 = (int)((w1 >>  4) & 0xFu) | (int)((h1 >>  4) >> s0_ & 1u) << 4;
                b1 = (int)((w1 >> 12) & 0xFu) | (int)((h1 >> 12) >> s0_ & 1u) << 4;
                b2 = (int)((w1 >> 20) & 0xFu) | (int)((h1 >> 20) >> s0_ & 1u) << 4;
                b3 = (int)( w1 >> 28       ) | (int)((h1 >> 28) >> s0_ & 1u) << 4;
            }
            swx += a0 * xv0.x + a1 * xv0.y + a2 * xv0.z + a3 * xv0.w
                 + b0 * xv1.x + b1 * xv1.y + b2 * xv1.z + b3 * xv1.w;
            sx  += xv0.x + xv0.y + xv0.z + xv0.w
                 + xv1.x + xv1.y + xv1.z + xv1.w;
        }
        s0 += ds0 * swx - dmd0 * sx;
        s1 += ds1 * swx - dmd1 * sx;
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

/* 2-rows-per-warp V2 for q6_K. Per-value: v = d * sc[n/16] * ((lo|hi6)-32)
 * where r = n%128, c = n/128; ql byte c*64+(r&63), qh byte c*32+(r&31) with
 * 2-bit field shifted 2*(r>>5); scale idx c*8+(r>>4). Within a sub-block
 * of 32 (sub in [0,7]) values r = sub*32..sub*32+31, all in the same
 * super-block chunk when sub<4, c=0; sub>=4, c=1. q6_K's scales are int8
 * (one per 16 values, so 2 per sub-block of 32 -- indices q and q+1 for
 * chunk q in [0,4) ... wait, re-check: per_value scale is sc[n/16], so
 * for sub-block of 32 values starting at r0=sub*32 within a chunk of 128,
 * the 4 sub-chunks of 8 each use scales sc[c*8 + (r0>>4) + 0..3]).
 *
 * The (q-32) values are the same for both rows (same x), but the scales
 * and d differ per row. We compute sum_qx[k] = sum over chunk k of
 * (q-32)*x for the 8 values using scale sc_k, then accumulate
 * d * sc_k * sum_qx[k] per row. This keeps the per-element work but
 * amortizes the q extraction (uint32 + nibble shift + qh look-up) into
 * 4 chunks per sub-block (8 values each).
 *
 * Note: r within a chunk ranges in [0,128); for sub<4 r0 in {0,32,64,96}
 * and the sub-block is r0..r0+32. Within sub-block the r mod 64 byte
 * addressing means low (r&63<64) vs high (r&63>=64) nibble selection
 * alternates at r=r0+64. r0 in {0,32,64,96} -> high nibble starts at
 * r=r0+64: r0+64 in {64,96,128,160} -- but r is chunk-local, so
 * r=64 wraps to the next ql byte (c*64+0)? NO: ql is 128 bytes laid
 * out as c*64+(r&63), so r=64 -> ql[c*64+0], r=65 -> ql[c*64+1], etc.
 * r in [0,64) reads ql[c*64+0..63] low nibbles; r in [64,128) reads
 * ql[c*64+0..63] HIGH nibbles. Same byte, opposite nibble. */
__global__ void k_gemv_q6_K_v2(const uint8_t *__restrict__ W,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nsb = K / 256;
    const int nu  = nsb * 8;
    const uint8_t *rw0 = W + (long)row0 * nsb * 210;
    const uint8_t *rw1 = W + (long)row1 * nsb * 210;
    float s0 = 0.0f, s1 = 0.0f;

    for (int u = lane; u < nu; u += 32) {
        const int sb = u >> 3, sub = u & 7;
        const uint8_t *blk0 = rw0 + sb * 210;
        const uint8_t *blk1 = rw1 + sb * 210;
        const uint8_t *ql0 = blk0,        *qh0 = blk0 + 128;
        const int8_t  *sc0 = (const int8_t *)(blk0 + 192);
        const float   d0  = half_at(blk0 + 208);
        const uint8_t *ql1 = blk1,        *qh1 = blk1 + 128;
        const int8_t  *sc1 = (const int8_t *)(blk1 + 192);
        const float   d1  = half_at(blk1 + 208);
        const float *xb = x + (long)sb * 256 + sub * 32;
        const int c = sub >> 2;                  /* chunk: 0 or 1 */
        const int r0 = (sub & 3) << 5;           /* r-base in chunk [0,32,64,96] */
        const int sc_idx_base = c * 8 + (r0 >> 4);
        /* For sub-block of 32 values starting at r0 (chunk-local), the 4
         * 8-value sub-chunks use scales at indices sc_idx_base+0..+3.
         * r advances as r0+0, r0+8, r0+16, r0+24 -- each sub-chunk
         * spans r0+q*8 .. r0+(q+1)*8. */
        const float4 *x4 = (const float4 *)xb;
        float sum_qx[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + q * 8;
            /* ql address: ql0[c*64 + (r&63)]. Load 8 bytes covering r..r+7. */
            const uint64_t ql8 = __ldg((const uint64_t *)(ql0 + c * 64 + (r & 63)));
            /* qh address: qh0[c*32 + (r&31)] byte; bits 2*(r>>5)..2*(r>>5)+1
             * of that byte. Load 4 bytes covering r..r+3 (then r+4..r+7
             * from same +4 offset). */
            const uint32_t qh_lo = __ldg((const uint32_t *)(qh0 + c * 32 + (r & 31)));
            const uint32_t qh_hi = __ldg((const uint32_t *)(qh0 + c * 32 + (r & 31) + 4));
            /* Extract 8 (q-32) values. */
            const float4 xv0 = x4[q * 2 + 0];
            const float4 xv1 = x4[q * 2 + 1];
            const int shift0 = 2 * (r >> 5);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ql_byte = (int)((ql8 >> (j * 8)) & 0xFFu);
                const int lo = (j < 4 && (r + j) < 64) ? (ql_byte & 0xF) : (ql_byte >> 4);
                const int qh_byte = (j < 4) ? (int)((qh_lo >> (j * 8)) & 0xFFu)
                                            : (int)((qh_hi >> ((j - 4) * 8)) & 0xFFu);
                const int hi = (qh_byte >> shift0) & 3;
                const int q6 = (lo | (hi << 4)) - 32;
                const float xv = (j < 4) ? ((j == 0) ? xv0.x : (j == 1) ? xv0.y : (j == 2) ? xv0.z : xv0.w)
                                         : ((j == 4) ? xv1.x : (j == 5) ? xv1.y : (j == 6) ? xv1.z : xv1.w);
                sum_qx[q] += (float)q6 * xv;
            }
        }
        /* Apply per-row scales. Both rows' (q-32) values are the same
         * (same x, same block bytes), only d and sc differ. */
        const float dsc0_0 = d0 * (float)sc0[sc_idx_base + 0];
        const float dsc0_1 = d0 * (float)sc0[sc_idx_base + 1];
        const float dsc0_2 = d0 * (float)sc0[sc_idx_base + 2];
        const float dsc0_3 = d0 * (float)sc0[sc_idx_base + 3];
        const float dsc1_0 = d1 * (float)sc1[sc_idx_base + 0];
        const float dsc1_1 = d1 * (float)sc1[sc_idx_base + 1];
        const float dsc1_2 = d1 * (float)sc1[sc_idx_base + 2];
        const float dsc1_3 = d1 * (float)sc1[sc_idx_base + 3];
        s0 += dsc0_0 * sum_qx[0] + dsc0_1 * sum_qx[1]
            + dsc0_2 * sum_qx[2] + dsc0_3 * sum_qx[3];
        s1 += dsc1_0 * sum_qx[0] + dsc1_1 * sum_qx[1]
            + dsc1_2 * sum_qx[2] + dsc1_3 * sum_qx[3];
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

/* ---------------- f16 / f32 (Tier-1 completeness) -------------------------- */

// M9.5 V2: two rows per warp, float4 x reads. Each lane reads 4 weights
// (8 bytes from the f16 row) per inner step, accumulating into s0 and
// s1; the two rows share the float4 x load (halves x re-read pressure
// for the M=1536/4864 hidden matrices). Requires K to be a multiple of
// 4 (every block reads 4 halfs; 4*2 = 8-byte alignment = 16-byte stride
// = 4 floats of x in float4) and the row stride K*2 = multiple of 8
// (i.e. K a multiple of 4) so the f16 row base is 8-byte aligned; on
// smollm2 (K=576) and tinyllama (K=2048) both hold. The 2-rows-per-warp
// layout matches gemv_dims2 used by tt_gemv_q4_0 (same math shape).
__global__ void k_gemv_f16(const uint8_t *__restrict__ W,
                           const float *__restrict__ x, float *__restrict__ y,
                           int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;                    // caller pads M to even
    const int lane = threadIdx.x;
    const int K4 = K >> 2;                        // 4-weight groups per row
    const __half *rw0 = (const __half *)((const char *)W + (long)row0 * K * 2);
    const __half *rw1 = (const __half *)((const char *)W + (long)row1 * K * 2);
    const float4 *x4 = (const float4 *)x;
    float s0 = 0.0f, s1 = 0.0f;
    for (int g = lane; g < K4; g += 32) {
        const float4 xg = x4[g];
        const __half2 *h0 = (const __half2 *)(rw0 + (g << 2));
        const __half2 *h1 = (const __half2 *)(rw1 + (g << 2));
        const float2 w0a = __half22float2(h0[0]);
        const float2 w0b = __half22float2(h0[1]);
        const float2 w1a = __half22float2(h1[0]);
        const float2 w1b = __half22float2(h1[1]);
        s0 = fmaf(w0a.x, xg.x, s0);
        s0 = fmaf(w0a.y, xg.y, s0);
        s0 = fmaf(w0b.x, xg.z, s0);
        s0 = fmaf(w0b.y, xg.w, s0);
        s1 = fmaf(w1a.x, xg.x, s1);
        s1 = fmaf(w1a.y, xg.y, s1);
        s1 = fmaf(w1b.x, xg.z, s1);
        s1 = fmaf(w1b.y, xg.w, s1);
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
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

/* Scalar F16 fallback (M9.5): only used when K is NOT a multiple of 4 (the
 * V2 path's 4-half group alignment contract). Same shape as the pre-M9.5
 * scalar: one warp per row, one half per cycle. */
__global__ void k_gemv_f16_scalar(const uint8_t *__restrict__ W,
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

/* M9.5 V2 BF16: same shape as k_gemv_f16 V2 but for BF16 weights. The
 * dequant is a no-op (weights are already bf16), just a bf16->float dot
 * product. Uses __nv_bfloat162 (2 bf16s packed into a uint32) and the
 * __bfloat1622float2 conversion intrinsic. Inner loop reads 4 bf16s
 * (8 bytes = 2 bfloat162) per row per cycle, accumulating 4 fmaf into
 * s0 and 4 fmaf into s1. The two rows share every float4 x load (halves
 * x re-read pressure, same win shape as the F16 V2 above).
 *
 * Contracts (caller checks via tt_gemv_typed gate):
 *   - K must be a multiple of 4 (4 bf16s = 8 bytes; float4 x is 16 bytes
 *     = 4 floats; one float4 of x per 4-bf16 group).
 *   - M padded to even (caller does it, see tt_gemv_typed).
 *   - M >= 2 (V2 needs two rows per warp; scalar fallback for M == 1
 *     is the existing k_gemv_bf16 below).
 *   - W base is 8-byte aligned; row stride K*2 = 8*(K/4) is a multiple
 *     of 8 for any K divisible by 4 — engine loads full rows, alignment
 *     follows from the row base. */
__global__ void k_gemv_bf16_v2(const uint8_t *__restrict__ W,
                              const float *__restrict__ x, float *__restrict__ y,
                              int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int K4 = K >> 2;                          // 4-bf16 groups per row
    const __nv_bfloat16 *rw0 = (const __nv_bfloat16 *)((const char *)W + (long)row0 * K * 2);
    const __nv_bfloat16 *rw1 = (const __nv_bfloat16 *)((const char *)W + (long)row1 * K * 2);
    const float4 *x4 = (const float4 *)x;
    float s0 = 0.0f, s1 = 0.0f;
    for (int g = lane; g < K4; g += 32) {
        const float4 xg = x4[g];
        const __nv_bfloat162 *h0 = (const __nv_bfloat162 *)(rw0 + (g << 2));
        const __nv_bfloat162 *h1 = (const __nv_bfloat162 *)(rw1 + (g << 2));
        const float2 w0a = __bfloat1622float2(h0[0]);
        const float2 w0b = __bfloat1622float2(h0[1]);
        const float2 w1a = __bfloat1622float2(h1[0]);
        const float2 w1b = __bfloat1622float2(h1[1]);
        s0 = fmaf(w0a.x, xg.x, s0);
        s0 = fmaf(w0a.y, xg.y, s0);
        s0 = fmaf(w0b.x, xg.z, s0);
        s0 = fmaf(w0b.y, xg.w, s0);
        s1 = fmaf(w1a.x, xg.x, s1);
        s1 = fmaf(w1a.y, xg.y, s1);
        s1 = fmaf(w1b.x, xg.z, s1);
        s1 = fmaf(w1b.y, xg.w, s1);
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
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

__global__ void k_embed_q3_K(const uint8_t *__restrict__ W, int tok,
                             float *__restrict__ x, int dim) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int nu = (dim / 256) * 16;
    if (u >= nu) return;
    const int sb  = u >> 4;
    const int is  = u & 15;
    const int n   = is >> 3;
    const int j   = (is & 7) >> 1;
    const int is0 = is & 1;
    const int shift = j << 1;
    const uint8_t m = 1 << (4 * n + j);

    const uint8_t *blk = W + (long)tok * (dim / 256) * 110 + sb * 110;
    const uint8_t *sc_raw = blk + 96;
    int8_t us = is <  4 ? (sc_raw[is-0] & 0xF) | (((sc_raw[is+8] >> 0) & 3) << 4) :
                is <  8 ? (sc_raw[is-0] & 0xF) | (((sc_raw[is+4] >> 2) & 3) << 4) :
                is < 12 ? (sc_raw[is-8] >>  4) | (((sc_raw[is+0] >> 4) & 3) << 4) :
                          (sc_raw[is-8] >>  4) | (((sc_raw[is-4] >> 6) & 3) << 4);
    const float d = half_at(blk + 108);
    const float dl = d * (float)(us - 32);

    const uint8_t *q  = blk + 32 + 32 * n + 16 * is0;
    const uint8_t *hm = blk + 16 * is0;
    float *dst = x + (long)sb * 256 + is * 16;

#pragma unroll
    for (int l = 0; l < 16; l++) {
        int8_t w = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
        dst[l] = dl * (float)w;
    }
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

/* 2-rows-per-warp V2 grid helper: each warp covers TWO output rows, so the
 * number of grid blocks is half of gemv_dims (rounded up). Used by the F16
 * V2 and BF16 V2 dispatchers below; the kernels themselves early-return
 * when row0 >= M (handles odd M cleanly, and any leftover rows when M is
 * even-but-not-divisible-by-blockDim.y*2 just go unused -- the math shape
 * is identical to gemv_q4_cuda.cu's gemv_dims2). */
static int gemv_dims2(int M, dim3 *grid, dim3 *block) {
    if (M <= 0 || M > (1 << 22)) return -50;
    block->x = 32; block->y = 16; block->z = 1;
    grid->x = (M + block->y * 2 - 1) / (block->y * 2); grid->y = 1; grid->z = 1;
    return 0;
}

static int kquant_aligned(int dtype, long K) { return K % 256 == 0; }

/*
 * y[M] = W[M,K] @ x[K] with W stored in any supported dtype.
 * Returns 0 or a cuda error code; -100 on unsupported dtype;
 * -101 on K-quant alignment violation (K % 256 != 0, message on stderr).
 */
__global__ void k_gemv_bf16(const uint16_t *__restrict__ W,
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const uint16_t *rw = W + (long)row * K;
    float sum = 0.0f;
    for (int k = lane; k < K; k += 32) {
        union { unsigned int u; float f; } cvt;
        cvt.u = ((unsigned int)rw[k]) << 16;
        sum += cvt.f * x[k];
    }
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[row] = sum;
}

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
            /* M9.5: optional WMMA tensor-core path. Opt-in via TT_USE_WMMA=1.
             * V2 (default) is faster on Ampere consumer for single-token
             * decode because m16n16k16 wastes 15/16 of the N dim on the
             * GEMV-as-1x1-GEMM shape. WMMA wins only when N>=8 of REAL
             * x's are available (batched decode / prefill) -- not in
             * scope for this engine, but the kernel + dispatcher is
             * shipped for future batched-decode work. The launcher is
             * in gemv_q4_cuda.cu next to the kernel (k_gemv_wmma_q4_0). */
            if (getenv("TT_USE_WMMA") && M >= 16 && K % 16 == 0) {
                extern int tt_gemv_wmma_q4_0(const void *, const float *,
                                             float *, int, int, cudaStream_t);
                int wrc = tt_gemv_wmma_q4_0(W, x, y, M, K, stream);
                if (wrc == 0) return 0;
                /* fall through to V2 if WMMA returned bad dims */
            }
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
        case TTQ_Q8_0: {
            /* M9.5: V2 path closes the qwen3-0.6b-q8_0 0.25x gap to
             * llama.cpp CUDA. Delegate to the 2-rows-per-warp kernel in
             * gemv_q4_cuda.cu when K/32 (nb) is even — the alignment
             * contract of the uint32-streaming inner loop. Otherwise fall
             * back to the scalar k_gemv_q8_0 above. */
            if ((K & 31) == 0 && (((K >> 5)) & 1) == 0
                && M >= 256 /* TT_GEMV_Q4_0_V2_MIN_M: same small-M gate as
                              * q4_0 V2 in gemv_q4_cuda.cu; V2 carries 2x
                              * d/qs/accumulator state and loses to scalar
                              * for the tiny M_kv shapes. */) {
                extern int tt_gemv_q8_0(const void *, const float *, float *,
                                        int, int, cudaStream_t);
                return tt_gemv_q8_0(W, x, y, M, K, stream);
            }
            k_gemv_q8_0<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        }
        case TTQ_Q3_K:
        case TTQ_Q4_K:
        case TTQ_Q5_K:
        case TTQ_Q6_K:
            if (!kquant_aligned(dtype, K)) {
                fprintf(stderr, "[gemv-typed] K=%d not a multiple of 256 — "
                                "K-quant GEMV needs n_per_row %% 256 == 0 "
                                "(dtype %d)\n", K, dtype);
                return -101;
            }
            if (M >= 2) {
                dim3 g2, b2;
                if (gemv_dims2(M, &g2, &b2) == 0) {
                    if (dtype == TTQ_Q3_K)
                        k_gemv_q3_K_v2<<<g2, b2, 0, stream>>>((const uint8_t *)W, x, y, M, K);
                    else if (dtype == TTQ_Q4_K)
                        k_gemv_q4_K_v2<<<g2, b2, 0, stream>>>((const uint8_t *)W, x, y, M, K);
                    else if (dtype == TTQ_Q5_K)
                        k_gemv_q5_K_v2<<<g2, b2, 0, stream>>>((const uint8_t *)W, x, y, M, K);
                    else
                        k_gemv_q6_K_v2<<<g2, b2, 0, stream>>>((const uint8_t *)W, x, y, M, K);
                    break;
                }
            }
            if (dtype == TTQ_Q3_K)
                k_gemv_q3_K<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            else if (dtype == TTQ_Q4_K)
                k_gemv_q4_K<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            else if (dtype == TTQ_Q5_K)
                k_gemv_q5_K<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            else
                k_gemv_q6_K<<<g, b, 0, stream>>>((const uint8_t *)W, x, y, M, K);
            break;
        case TTQ_F16: {
            /* M9.5: V2 (2-rows-per-warp) is the existing k_gemv_f16, but the
             * prior dispatcher launched it with gemv_dims -- a one-warp-per-
             * row grid -- so 50% of warps early-returned. Use gemv_dims2
             * (2 rows per warp) so grid.x*b.y*2 == M. The V2 path requires
             * K%4 == 0 (4-half group alignment); the scalar fallback
             * handles K%4 != 0 (rare; all our LLM dims are div-by-4). */
            if (M >= 2 && (K & 3) == 0) {
                dim3 g2, b2;
                if (gemv_dims2(M, &g2, &b2) == 0)
                    k_gemv_f16<<<g2, b2, 0, stream>>>(
                        (const uint8_t *)W, x, y, M, K);
                else
                    k_gemv_f16<<<g, b, 0, stream>>>(
                        (const uint8_t *)W, x, y, M, K);
            } else {
                k_gemv_f16_scalar<<<g, b, 0, stream>>>(
                    (const uint8_t *)W, x, y, M, K);
            }
            break;
        }
        case TTQ_F32:
            k_gemv_f32<<<g, b, 0, stream>>>((const float *)W, x, y, M, K);
            break;
        case 30: /* TTQ_BF16 */ {
            /* M9.5: V2 (2-rows-per-warp, __nv_bfloat162). Same V2 contract
             * as F16 above: K%4 == 0 (4-bf16 group alignment) and M >= 2.
             * Falls back to the scalar k_gemv_bf16 (1 half-word per cycle)
             * for the K%4 != 0 case. */
            if (M >= 2 && (K & 3) == 0) {
                dim3 g2, b2;
                if (gemv_dims2(M, &g2, &b2) == 0)
                    k_gemv_bf16_v2<<<g2, b2, 0, stream>>>(
                        (const uint8_t *)W, x, y, M, K);
                else
                    k_gemv_bf16_v2<<<g, b, 0, stream>>>(
                        (const uint8_t *)W, x, y, M, K);
            } else {
                k_gemv_bf16<<<g, b, 0, stream>>>(
                    (const uint16_t *)W, x, y, M, K);
            }
            break;
        }
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

/* templates need C++ linkage: temporarily leave the extern "C" block */
} /* extern "C" (resumed below) */

/* ---------------- batched prefill GEMM (M12 task: production) -------------- *
 *
 * tt_gemm_batched: Y[M,T] = W[M,K] @ X[K,T], W q4_0 (GGUF raw blocks),
 * X row-major [K][T] (k-major), Y row-major [M][T]. Batches T tokens per
 * weight stream so prefill pays DRAM for W once per tile instead of once per
 * token (prototype tests/proto_batched_gemv.cu measured ~9x on 1536-wide
 * layers at T=8).
 *
 * Kernel: one warp per output row, lanes stride the K/32 blocks (same shape
 * as k_gemv_q4_0 above, identical dequant-inline math v=(q-8)*d), each lane
 * keeps a TT_GEMM_BATCHED_MAX_T-wide accumulator so all T columns ride one
 * weight pass. Accumulators stay in registers via full unroll; columns past
 * T are predicated off (memory-bound kernel, wasted MACs are free).
 *
 * No header file exists for this TU (tt_gemv_typed itself is declared only
 * via extern in its callers), so callers declare:
 *
 *   extern "C" int tt_gemm_batched(const void *W, int dtype,
 *       const float *X (K*T floats), float *Y (M*T floats),
 *       int M, int K, int T, cudaStream_t stream);
 *
 * Contract / error codes (mirrors tt_gemv_typed conventions):
 *   returns 0 or a cuda error code;
 *   -50   bad dims (NULL ptr, M<=0 or M > 2^22, T<=0 or T > MAX_T)
 *   -100  unsupported dtype (only TTQ_Q4_0 implemented — the dominant
 *         prompt-prefill case; extend with more kernels as needed)
 *   -101  K not a multiple of 32 (legacy quant alignment)
 *   T must be <= TT_GEMM_BATCHED_MAX_T (16): caller loops over column tiles
 *   of <=16 tokens when n_tokens is larger.
 */

#define TT_GEMM_BATCHED_MAX_T 16
#define TT_GEMM_BATCHED_WARPS 8

template <int TILE>
__global__ void k_gemm_batched_q4_0(const uint8_t *__restrict__ W,
                                    const float *__restrict__ X,
                                    float *__restrict__ Y,
                                    int M, int K, int T) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int col0 = blockIdx.y * TILE;
    const int nt = min(TILE, T - col0);      /* live columns in this tile */
    const int nb = K >> 5;
    const uint8_t *rw = W + (long)row * nb * 18;

    float acc[TILE];
#pragma unroll
    for (int t = 0; t < TILE; ++t) acc[t] = 0.f;

    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = rw + b * 18;
        const float d = half_at(blk);
        const uint8_t *qs = blk + 2;
        const float *xb = X + (size_t)b * 32 * T + col0;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const float vlo = (float)(qs[j] & 0x0F) - 8.0f;
            const float vhi = (float)(qs[j] >>   4) - 8.0f;
#pragma unroll
            for (int t = 0; t < TILE; ++t) {
                if (t >= nt) break;          /* straight-line predication */
                acc[t] += d * (vlo * xb[(size_t)j * T + t]
                             + vhi * xb[(size_t)(j + 16) * T + t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TILE; ++t) {
        if (t >= nt) break;
        const float v = warp_reduce_sum(acc[t]);
        if (lane == 0) Y[(size_t)row * T + col0 + t] = v;
    }
}

extern "C" {

int tt_gemm_batched(const void *W, int dtype, const float *X, float *Y,
                    int M, int K, int T, cudaStream_t stream) {
    if (!W || !X || !Y || M <= 0 || M > (1 << 22) ||
        T <= 0 || T > TT_GEMM_BATCHED_MAX_T)
        return -50;
    if (dtype != TTQ_Q4_0) {
        fprintf(stderr, "[gemm-batched] unsupported dtype %d (only Q4_0)\n",
                dtype);
        return -100;
    }
    if (K <= 0 || K % 32 != 0) {
        fprintf(stderr, "[gemm-batched] K=%d not a multiple of 32\n", K);
        return -101;
    }
    dim3 block(32, TT_GEMM_BATCHED_WARPS), g;
    g.x = (M + TT_GEMM_BATCHED_WARPS - 1) / TT_GEMM_BATCHED_WARPS;
    g.y = (T + TT_GEMM_BATCHED_MAX_T - 1) / TT_GEMM_BATCHED_MAX_T;
    g.z = 1;
    k_gemm_batched_q4_0<TT_GEMM_BATCHED_MAX_T>
        <<<g, block, 0, stream>>>((const uint8_t *)W, X, Y, M, K, T);
    return (int)cudaGetLastError();
}

/* Typed embedding row lookup. Returns 0 or cuda err; -100 unsupported. */
__global__ void k_embed_bf16(const uint16_t *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= dim) return;
    unsigned int bits = (unsigned int)W[(long)tok * dim + i] << 16;
    union { unsigned int u; float f; } cvt; cvt.u = bits;
    dx[i] = cvt.f;
}

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
        case TTQ_Q3_K:
        case TTQ_Q4_K:
        case TTQ_Q5_K:
        case TTQ_Q6_K: {
            if (!kquant_aligned(dtype, dim)) {
                fprintf(stderr, "[gemv-typed] embed dim=%d not multiple of 256 "
                                "for K-quant dtype %d\n", dim, dtype);
                return -101;
            }
            const int nu = (dim / 256) * 8;
            if (dtype == TTQ_Q3_K) {
                const int nu3 = (dim / 256) * 16;
                k_embed_q3_K<<<(nu3 + 255) / 256, 256, 0, stream>>>(
                    (const uint8_t *)dW, tok, dx, dim);
            } else if (dtype == TTQ_Q4_K)
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
        
        case 30: /* TTQ_BF16 */
            k_embed_bf16<<<(dim + 255) / 256, 256, 0, stream>>>(
                (const uint16_t *)dW, tok, dx, dim);
            break;
        default:
            fprintf(stderr, "[gemv-typed] embed: unsupported dtype %d\n", dtype);
            return -100;
    }
    return (int)cudaGetLastError();
}

} /* extern "C" */
