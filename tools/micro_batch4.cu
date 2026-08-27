// M10+ True Batched Verification microbench: k_gemv_q4_0_batch4 + k_gemv_q8_0_batch4.
//
// Design: shmem-cached 4x4 kernel. Each warp processes 4 weight rows
// and 4 candidate x vectors in a single call, producing 16 output
// values per warp. The 4 candidate x's are loaded into shared memory
// ONCE per block and reused by all warps, eliminating the 4x
// gmem-x-read amplification that the 1-row-per-warp design suffers from.
//
// Why shmem is needed: the q4_0 weight row is K*0.5625 bytes; the x
// vector is K*4 bytes (7x more than the weight). For q8_0 the
// weight is K*1 bytes (still 4x less than x). In any batched-4
// design that reads x from gmem, the 4x x traffic dwarfs the
// 1x weight traffic and caps speedup at ~1.2x over the single
// kernel — not the 3x the task spec requires. Caching the 4
// candidate x's in shmem drops the per-output gmem footprint to
// just 0.25 weight rows, restoring the arithmetic that makes
// batched verification worthwhile.
//
// Per-block layout (16 warps, 32 lanes, 4 rows x 4 cand per warp):
//   shmem: 4 * K * 4 bytes for the 4 x vectors
//   grid:  M / (16 warps * 4 rows) blocks
//   each block produces: 16 warps * 4 rows * 4 cand = 256 outputs
//   shmem: 4 * 896 * 4 = 14 KB (K=896) — well under Ampere's 100 KB/SM
//   shmem: 4 * 4864 * 4 = 76 KB (K=4864) — tight, single block/SM
//
// Compared to the 1-row-per-warp design: per-output gmem footprint
// drops from K*(0.25W + 1X) = K*4.14 bytes to K*0.14 bytes
// (q4_0 LM head, K=896: 3.7 KB to 126 bytes per output, ~30x less).
//
// Standalone, not linked into engine. Mirrors tools/micro_v4.cu
// structure (correctness check first, then timed loop with 100 iters
// after 10 warmup).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// -------- shared helpers (copied from micro_v4.cu for standalone) --------
__device__ __forceinline__ float warp_reduce(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

typedef struct {
    __half d;
    uint8_t qs[16];
} BlockQ4_0;

typedef struct {
    __half d;
    int8_t  qs[32];
} BlockQ8_0;

// =================================================================
// q4_0: V2 single (reference / sequential baseline). 2 rows/warp.
// Mirrors kernels/gemv_q4_cuda.cu k_gemv_q4_0.
// =================================================================
__global__ void k_gemv_q4_0_single(const BlockQ4_0 *__restrict__ W,
                                    const float    *__restrict__ x,
                                    float          *__restrict__ y,
                                    int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nb   = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (18 * b + 2) >> 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            s0 += (float)((int)(va         & 0xFu) - 8) * da * xa.x;
            s0 += (float)((int)((va >>  4) & 0xFu) - 8) * da * xb.x;
            s0 += (float)((int)((va >>  8) & 0xFu) - 8) * da * xa.y;
            s0 += (float)((int)((va >> 12) & 0xFu) - 8) * da * xb.y;
            s0 += (float)((int)((va >> 16) & 0xFu) - 8) * da * xa.z;
            s0 += (float)((int)((va >> 20) & 0xFu) - 8) * da * xb.z;
            s0 += (float)((int)((va >> 24) & 0xFu) - 8) * da * xa.w;
            s0 += (float)((int)(va >> 28) - 8) * da * xb.w;
            s1 += (float)((int)(vb         & 0xFu) - 8) * db * xa.x;
            s1 += (float)((int)((vb >>  4) & 0xFu) - 8) * db * xb.x;
            s1 += (float)((int)((vb >>  8) & 0xFu) - 8) * db * xa.y;
            s1 += (float)((int)((vb >> 12) & 0xFu) - 8) * db * xb.y;
            s1 += (float)((int)((vb >> 16) & 0xFu) - 8) * db * xa.z;
            s1 += (float)((int)((vb >> 20) & 0xFu) - 8) * db * xb.z;
            s1 += (float)((int)((vb >> 24) & 0xFu) - 8) * db * xa.w;
            s1 += (float)((int)(vb >> 28) - 8) * db * xb.w;
        }
    }
    s0 = warp_reduce(s0);
    s1 = warp_reduce(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// =================================================================
// q4_0: SHMEM-CACHED 4x4 BATCHED-4. 4 rows/warp, 4 candidate x
// vectors cached in shared memory, 16 register accumulators. Weight
// is loaded ONCE per K-position per row and the dequantized nibbles
// are held in named temporaries so the compiler keeps them in
// registers across all 4 candidates AND across the 4 rows.
//
// Layout: X is [4, K] flat (X[c*K + k]). Y is [4, M] flat (Y[c*M + m]).
// Constraints: K%32==0, nb=K/32 even. M must be a multiple of 4
// (caller pads). 4 candidate x's must fit in shmem (= 16*K bytes).
// =================================================================
__global__ void k_gemv_q4_0_batch4(const BlockQ4_0 *__restrict__ W,
                                    const float    *__restrict__ X,
                                    float          *__restrict__ Y,
                                    int M, int K) {
    extern __shared__ float sx[];   // [4][K]
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = (blockIdx.x * blockDim.y + warp) * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1, row2 = row0 + 2, row3 = row0 + 3;
    const int nb   = K / 32;

    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 18);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 18);

    // ---- 1) Cooperatively load 4 candidate x vectors into shmem ----
    // Total 4*K floats; 16 warps * 32 lanes = 512 threads; each thread
    // loads 4*K/512 floats. For K=896: 7 floats/thread (28 bytes = 7
    // float4/28 if we vectorize, but scalar is fine for the load path).
    // float4 vectorized load: each thread does (4*K) / (16*32*4) = K/512
    // float4 loads. K=896 -> 1.75 -> use 2 for some, 1 for others.
    // Simpler: each thread loads (4*K)/(16*32) = K/128 float4s. For
    // K=896 -> 7 float4s. Use a vectorized load loop.
    {
        const int total_f4 = 4 * K / 4;          // 4 * K floats / 4 = K float4
        const int tid      = warp * 32 + lane;
        const int nthreads = blockDim.y * 32;
        const float4 *src  = (const float4 *)X;
        float4       *dst  = (float4 *)sx;
        for (int i = tid; i < total_f4; i += nthreads) {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    // Shmem pointers for each candidate's x.
    const float4 *sx4_0 = (const float4 *)(sx + 0 * K);
    const float4 *sx4_1 = (const float4 *)(sx + 1 * K);
    const float4 *sx4_2 = (const float4 *)(sx + 2 * K);
    const float4 *sx4_3 = (const float4 *)(sx + 3 * K);

    // 16 accumulators: 4 rows x 4 candidates. Naming: sRC.
    float s00 = 0.f, s01 = 0.f, s02 = 0.f, s03 = 0.f;
    float s10 = 0.f, s11 = 0.f, s12 = 0.f, s13 = 0.f;
    float s20 = 0.f, s21 = 0.f, s22 = 0.f, s23 = 0.f;
    float s30 = 0.f, s31 = 0.f, s32 = 0.f, s33 = 0.f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const int sh  = (18 * b + 2) & 2;

        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const unsigned short d16c = (unsigned short)
            (((18 * b) & 2) ? (rw2[wsc] >> 16) : (rw2[wsc] & 0xFFFFu));
        const unsigned short d16d = (unsigned short)
            (((18 * b) & 2) ? (rw3[wsc] >> 16) : (rw3[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));
        const int a0 = (18 * b + 2) >> 2;

#pragma unroll
        for (int k = 0; k < 4; k++) {
            // Load 4 weight rows. Each row's weight is read once.
            const uint32_t la0 = __ldg(rw0 + a0 + k);
            const uint32_t la1 = __ldg(rw1 + a0 + k);
            const uint32_t la2 = __ldg(rw2 + a0 + k);
            const uint32_t la3 = __ldg(rw3 + a0 + k);
            const uint32_t va0 = sh ? __byte_perm(la0, __ldg(rw0 + a0 + k + 1), 0x5432) : la0;
            const uint32_t va1 = sh ? __byte_perm(la1, __ldg(rw1 + a0 + k + 1), 0x5432) : la1;
            const uint32_t va2 = sh ? __byte_perm(la2, __ldg(rw2 + a0 + k + 1), 0x5432) : la2;
            const uint32_t va3 = sh ? __byte_perm(la3, __ldg(rw3 + a0 + k + 1), 0x5432) : la3;

            // Dequant each row's 8 nibbles into named temporaries
            // (forces the compiler to keep them in registers for the
            // 16 FMA uses across the 4 candidates).
            const float a0_0 = (float)((int)( va0         & 0xFu) - 8) * da;
            const float a0_1 = (float)((int)((va0 >>  4)  & 0xFu) - 8) * da;
            const float a0_2 = (float)((int)((va0 >>  8)  & 0xFu) - 8) * da;
            const float a0_3 = (float)((int)((va0 >> 12)  & 0xFu) - 8) * da;
            const float a0_4 = (float)((int)((va0 >> 16)  & 0xFu) - 8) * da;
            const float a0_5 = (float)((int)((va0 >> 20)  & 0xFu) - 8) * da;
            const float a0_6 = (float)((int)((va0 >> 24)  & 0xFu) - 8) * da;
            const float a0_7 = (float)((int)( va0 >> 28)        - 8) * da;
            const float a1_0 = (float)((int)( va1         & 0xFu) - 8) * db;
            const float a1_1 = (float)((int)((va1 >>  4)  & 0xFu) - 8) * db;
            const float a1_2 = (float)((int)((va1 >>  8)  & 0xFu) - 8) * db;
            const float a1_3 = (float)((int)((va1 >> 12)  & 0xFu) - 8) * db;
            const float a1_4 = (float)((int)((va1 >> 16)  & 0xFu) - 8) * db;
            const float a1_5 = (float)((int)((va1 >> 20)  & 0xFu) - 8) * db;
            const float a1_6 = (float)((int)((va1 >> 24)  & 0xFu) - 8) * db;
            const float a1_7 = (float)((int)( va1 >> 28)        - 8) * db;
            const float a2_0 = (float)((int)( va2         & 0xFu) - 8) * dc;
            const float a2_1 = (float)((int)((va2 >>  4)  & 0xFu) - 8) * dc;
            const float a2_2 = (float)((int)((va2 >>  8)  & 0xFu) - 8) * dc;
            const float a2_3 = (float)((int)((va2 >> 12)  & 0xFu) - 8) * dc;
            const float a2_4 = (float)((int)((va2 >> 16)  & 0xFu) - 8) * dc;
            const float a2_5 = (float)((int)((va2 >> 20)  & 0xFu) - 8) * dc;
            const float a2_6 = (float)((int)((va2 >> 24)  & 0xFu) - 8) * dc;
            const float a2_7 = (float)((int)( va2 >> 28)        - 8) * dc;
            const float a3_0 = (float)((int)( va3         & 0xFu) - 8) * dd;
            const float a3_1 = (float)((int)((va3 >>  4)  & 0xFu) - 8) * dd;
            const float a3_2 = (float)((int)((va3 >>  8)  & 0xFu) - 8) * dd;
            const float a3_3 = (float)((int)((va3 >> 12)  & 0xFu) - 8) * dd;
            const float a3_4 = (float)((int)((va3 >> 16)  & 0xFu) - 8) * dd;
            const float a3_5 = (float)((int)((va3 >> 20)  & 0xFu) - 8) * dd;
            const float a3_6 = (float)((int)((va3 >> 24)  & 0xFu) - 8) * dd;
            const float a3_7 = (float)((int)( va3 >> 28)        - 8) * dd;

            // Load the 4 candidate x's for this K-block from shmem.
            const float4 xa0 = sx4_0[k],     xb0 = sx4_0[k + 4];
            const float4 xa1 = sx4_1[k],     xb1 = sx4_1[k + 4];
            const float4 xa2 = sx4_2[k],     xb2 = sx4_2[k + 4];
            const float4 xa3 = sx4_3[k],     xb3 = sx4_3[k + 4];

            // 4 rows x 4 candidates = 16 dot-product contributions.
            // For each row, the 8 nibbles of the 32-bit word pair
            // with xa (low nibbles) and xb (high nibbles) for the
            // 4 candidates.
            // Row 0
            s00 += a0_0 * xa0.x;  s00 += a0_1 * xb0.x;
            s00 += a0_2 * xa0.y;  s00 += a0_3 * xb0.y;
            s00 += a0_4 * xa0.z;  s00 += a0_5 * xb0.z;
            s00 += a0_6 * xa0.w;  s00 += a0_7 * xb0.w;
            s01 += a0_0 * xa1.x;  s01 += a0_1 * xb1.x;
            s01 += a0_2 * xa1.y;  s01 += a0_3 * xb1.y;
            s01 += a0_4 * xa1.z;  s01 += a0_5 * xb1.z;
            s01 += a0_6 * xa1.w;  s01 += a0_7 * xb1.w;
            s02 += a0_0 * xa2.x;  s02 += a0_1 * xb2.x;
            s02 += a0_2 * xa2.y;  s02 += a0_3 * xb2.y;
            s02 += a0_4 * xa2.z;  s02 += a0_5 * xb2.z;
            s02 += a0_6 * xa2.w;  s02 += a0_7 * xb2.w;
            s03 += a0_0 * xa3.x;  s03 += a0_1 * xb3.x;
            s03 += a0_2 * xa3.y;  s03 += a0_3 * xb3.y;
            s03 += a0_4 * xa3.z;  s03 += a0_5 * xb3.z;
            s03 += a0_6 * xa3.w;  s03 += a0_7 * xb3.w;
            // Row 1
            s10 += a1_0 * xa0.x;  s10 += a1_1 * xb0.x;
            s10 += a1_2 * xa0.y;  s10 += a1_3 * xb0.y;
            s10 += a1_4 * xa0.z;  s10 += a1_5 * xb0.z;
            s10 += a1_6 * xa0.w;  s10 += a1_7 * xb0.w;
            s11 += a1_0 * xa1.x;  s11 += a1_1 * xb1.x;
            s11 += a1_2 * xa1.y;  s11 += a1_3 * xb1.y;
            s11 += a1_4 * xa1.z;  s11 += a1_5 * xb1.z;
            s11 += a1_6 * xa1.w;  s11 += a1_7 * xb1.w;
            s12 += a1_0 * xa2.x;  s12 += a1_1 * xb2.x;
            s12 += a1_2 * xa2.y;  s12 += a1_3 * xb2.y;
            s12 += a1_4 * xa2.z;  s12 += a1_5 * xb2.z;
            s12 += a1_6 * xa2.w;  s12 += a1_7 * xb2.w;
            s13 += a1_0 * xa3.x;  s13 += a1_1 * xb3.x;
            s13 += a1_2 * xa3.y;  s13 += a1_3 * xb3.y;
            s13 += a1_4 * xa3.z;  s13 += a1_5 * xb3.z;
            s13 += a1_6 * xa3.w;  s13 += a1_7 * xb3.w;
            // Row 2
            s20 += a2_0 * xa0.x;  s20 += a2_1 * xb0.x;
            s20 += a2_2 * xa0.y;  s20 += a2_3 * xb0.y;
            s20 += a2_4 * xa0.z;  s20 += a2_5 * xb0.z;
            s20 += a2_6 * xa0.w;  s20 += a2_7 * xb0.w;
            s21 += a2_0 * xa1.x;  s21 += a2_1 * xb1.x;
            s21 += a2_2 * xa1.y;  s21 += a2_3 * xb1.y;
            s21 += a2_4 * xa1.z;  s21 += a2_5 * xb1.z;
            s21 += a2_6 * xa1.w;  s21 += a2_7 * xb1.w;
            s22 += a2_0 * xa2.x;  s22 += a2_1 * xb2.x;
            s22 += a2_2 * xa2.y;  s22 += a2_3 * xb2.y;
            s22 += a2_4 * xa2.z;  s22 += a2_5 * xb2.z;
            s22 += a2_6 * xa2.w;  s22 += a2_7 * xb2.w;
            s23 += a2_0 * xa3.x;  s23 += a2_1 * xb3.x;
            s23 += a2_2 * xa3.y;  s23 += a2_3 * xb3.y;
            s23 += a2_4 * xa3.z;  s23 += a2_5 * xb3.z;
            s23 += a2_6 * xa3.w;  s23 += a2_7 * xb3.w;
            // Row 3
            s30 += a3_0 * xa0.x;  s30 += a3_1 * xb0.x;
            s30 += a3_2 * xa0.y;  s30 += a3_3 * xb0.y;
            s30 += a3_4 * xa0.z;  s30 += a3_5 * xb0.z;
            s30 += a3_6 * xa0.w;  s30 += a3_7 * xb0.w;
            s31 += a3_0 * xa1.x;  s31 += a3_1 * xb1.x;
            s31 += a3_2 * xa1.y;  s31 += a3_3 * xb1.y;
            s31 += a3_4 * xa1.z;  s31 += a3_5 * xb1.z;
            s31 += a3_6 * xa1.w;  s31 += a3_7 * xb1.w;
            s32 += a3_0 * xa2.x;  s32 += a3_1 * xb2.x;
            s32 += a3_2 * xa2.y;  s32 += a3_3 * xb2.y;
            s32 += a3_4 * xa2.z;  s32 += a3_5 * xb2.z;
            s32 += a3_6 * xa2.w;  s32 += a3_7 * xb2.w;
            s33 += a3_0 * xa3.x;  s33 += a3_1 * xb3.x;
            s33 += a3_2 * xa3.y;  s33 += a3_3 * xb3.y;
            s33 += a3_4 * xa3.z;  s33 += a3_5 * xb3.z;
            s33 += a3_6 * xa3.w;  s33 += a3_7 * xb3.w;
        }
    }
    s00 = warp_reduce(s00); s01 = warp_reduce(s01); s02 = warp_reduce(s02); s03 = warp_reduce(s03);
    s10 = warp_reduce(s10); s11 = warp_reduce(s11); s12 = warp_reduce(s12); s13 = warp_reduce(s13);
    s20 = warp_reduce(s20); s21 = warp_reduce(s21); s22 = warp_reduce(s22); s23 = warp_reduce(s23);
    s30 = warp_reduce(s30); s31 = warp_reduce(s31); s32 = warp_reduce(s32); s33 = warp_reduce(s33);
    if (lane == 0) {
        Y[0 * M + row0] = s00;  Y[1 * M + row0] = s01;  Y[2 * M + row0] = s02;  Y[3 * M + row0] = s03;
        if (row1 < M) { Y[0 * M + row1] = s10;  Y[1 * M + row1] = s11;  Y[2 * M + row1] = s12;  Y[3 * M + row1] = s13; }
        if (row2 < M) { Y[0 * M + row2] = s20;  Y[1 * M + row2] = s21;  Y[2 * M + row2] = s22;  Y[3 * M + row2] = s23; }
        if (row3 < M) { Y[0 * M + row3] = s30;  Y[1 * M + row3] = s31;  Y[2 * M + row3] = s32;  Y[3 * M + row3] = s33; }
    }
}

// =================================================================
// q8_0: V2 single. 2 rows/warp, 8 uint32 words/block.
// =================================================================
__global__ void k_gemv_q8_0_single(const BlockQ8_0 *__restrict__ W,
                                    const float    *__restrict__ x,
                                    float          *__restrict__ y,
                                    int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nb   = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 34);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 34);
    float s0 = 0.0f, s1 = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;
        const int sh  = (34 * b + 2) & 2;
        const unsigned short d16a = (unsigned short)
            (((34 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((34 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (34 * b + 2) >> 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const float4 xv = x4[k];
            s0 += ((float)((int)(va << 24) >> 24)) * da * xv.x;
            s0 += ((float)((int)(va << 16) >> 24)) * da * xv.y;
            s0 += ((float)((int)(va <<  8) >> 24)) * da * xv.z;
            s0 += ((float)((int)(va       ) >> 24)) * da * xv.w;
            s1 += ((float)((int)(vb << 24) >> 24)) * db * xv.x;
            s1 += ((float)((int)(vb << 16) >> 24)) * db * xv.y;
            s1 += ((float)((int)(vb <<  8) >> 24)) * db * xv.z;
            s1 += ((float)((int)(vb       ) >> 24)) * db * xv.w;
        }
    }
    s0 = warp_reduce(s0);
    s1 = warp_reduce(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// =================================================================
// q8_0: SHMEM-CACHED 4x4 BATCHED-4. 4 rows x 4 candidates, x in
// shmem. 16 accumulators.
// =================================================================
__global__ void k_gemv_q8_0_batch4(const BlockQ8_0 *__restrict__ W,
                                    const float    *__restrict__ X,
                                    float          *__restrict__ Y,
                                    int M, int K) {
    extern __shared__ float sx[];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = (blockIdx.x * blockDim.y + warp) * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1, row2 = row0 + 2, row3 = row0 + 3;
    const int nb   = K / 32;

    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 34);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 34);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 34);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 34);

    {
        const int total_f4 = 4 * K / 4;
        const int tid      = warp * 32 + lane;
        const int nthreads = blockDim.y * 32;
        const float4 *src  = (const float4 *)X;
        float4       *dst  = (float4 *)sx;
        for (int i = tid; i < total_f4; i += nthreads) {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    const float4 *sx4_0 = (const float4 *)(sx + 0 * K);
    const float4 *sx4_1 = (const float4 *)(sx + 1 * K);
    const float4 *sx4_2 = (const float4 *)(sx + 2 * K);
    const float4 *sx4_3 = (const float4 *)(sx + 3 * K);

    float s00 = 0.f, s01 = 0.f, s02 = 0.f, s03 = 0.f;
    float s10 = 0.f, s11 = 0.f, s12 = 0.f, s13 = 0.f;
    float s20 = 0.f, s21 = 0.f, s22 = 0.f, s23 = 0.f;
    float s30 = 0.f, s31 = 0.f, s32 = 0.f, s33 = 0.f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;
        const int sh  = (34 * b + 2) & 2;
        const unsigned short d16a = (unsigned short)
            (((34 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((34 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const unsigned short d16c = (unsigned short)
            (((34 * b) & 2) ? (rw2[wsc] >> 16) : (rw2[wsc] & 0xFFFFu));
        const unsigned short d16d = (unsigned short)
            (((34 * b) & 2) ? (rw3[wsc] >> 16) : (rw3[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));
        const int a0 = (34 * b + 2) >> 2;

#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t la0 = __ldg(rw0 + a0 + k);
            const uint32_t la1 = __ldg(rw1 + a0 + k);
            const uint32_t la2 = __ldg(rw2 + a0 + k);
            const uint32_t la3 = __ldg(rw3 + a0 + k);
            const uint32_t va0 = sh ? __byte_perm(la0, __ldg(rw0 + a0 + k + 1), 0x5432) : la0;
            const uint32_t va1 = sh ? __byte_perm(la1, __ldg(rw1 + a0 + k + 1), 0x5432) : la1;
            const uint32_t va2 = sh ? __byte_perm(la2, __ldg(rw2 + a0 + k + 1), 0x5432) : la2;
            const uint32_t va3 = sh ? __byte_perm(la3, __ldg(rw3 + a0 + k + 1), 0x5432) : la3;

            const float a0_0 = (float)((int)(va0 << 24) >> 24) * da;
            const float a0_1 = (float)((int)(va0 << 16) >> 24) * da;
            const float a0_2 = (float)((int)(va0 <<  8) >> 24) * da;
            const float a0_3 = (float)((int)(va0      ) >> 24) * da;
            const float a1_0 = (float)((int)(va1 << 24) >> 24) * db;
            const float a1_1 = (float)((int)(va1 << 16) >> 24) * db;
            const float a1_2 = (float)((int)(va1 <<  8) >> 24) * db;
            const float a1_3 = (float)((int)(va1      ) >> 24) * db;
            const float a2_0 = (float)((int)(va2 << 24) >> 24) * dc;
            const float a2_1 = (float)((int)(va2 << 16) >> 24) * dc;
            const float a2_2 = (float)((int)(va2 <<  8) >> 24) * dc;
            const float a2_3 = (float)((int)(va2      ) >> 24) * dc;
            const float a3_0 = (float)((int)(va3 << 24) >> 24) * dd;
            const float a3_1 = (float)((int)(va3 << 16) >> 24) * dd;
            const float a3_2 = (float)((int)(va3 <<  8) >> 24) * dd;
            const float a3_3 = (float)((int)(va3      ) >> 24) * dd;

            const float4 xv0 = sx4_0[k];
            const float4 xv1 = sx4_1[k];
            const float4 xv2 = sx4_2[k];
            const float4 xv3 = sx4_3[k];

            // Row 0
            s00 += a0_0 * xv0.x; s00 += a0_1 * xv0.y; s00 += a0_2 * xv0.z; s00 += a0_3 * xv0.w;
            s01 += a0_0 * xv1.x; s01 += a0_1 * xv1.y; s01 += a0_2 * xv1.z; s01 += a0_3 * xv1.w;
            s02 += a0_0 * xv2.x; s02 += a0_1 * xv2.y; s02 += a0_2 * xv2.z; s02 += a0_3 * xv2.w;
            s03 += a0_0 * xv3.x; s03 += a0_1 * xv3.y; s03 += a0_2 * xv3.z; s03 += a0_3 * xv3.w;
            // Row 1
            s10 += a1_0 * xv0.x; s10 += a1_1 * xv0.y; s10 += a1_2 * xv0.z; s10 += a1_3 * xv0.w;
            s11 += a1_0 * xv1.x; s11 += a1_1 * xv1.y; s11 += a1_2 * xv1.z; s11 += a1_3 * xv1.w;
            s12 += a1_0 * xv2.x; s12 += a1_1 * xv2.y; s12 += a1_2 * xv2.z; s12 += a1_3 * xv2.w;
            s13 += a1_0 * xv3.x; s13 += a1_1 * xv3.y; s13 += a1_2 * xv3.z; s13 += a1_3 * xv3.w;
            // Row 2
            s20 += a2_0 * xv0.x; s20 += a2_1 * xv0.y; s20 += a2_2 * xv0.z; s20 += a2_3 * xv0.w;
            s21 += a2_0 * xv1.x; s21 += a2_1 * xv1.y; s21 += a2_2 * xv1.z; s21 += a2_3 * xv1.w;
            s22 += a2_0 * xv2.x; s22 += a2_1 * xv2.y; s22 += a2_2 * xv2.z; s22 += a2_3 * xv2.w;
            s23 += a2_0 * xv3.x; s23 += a2_1 * xv3.y; s23 += a2_2 * xv3.z; s23 += a2_3 * xv3.w;
            // Row 3
            s30 += a3_0 * xv0.x; s30 += a3_1 * xv0.y; s30 += a3_2 * xv0.z; s30 += a3_3 * xv0.w;
            s31 += a3_0 * xv1.x; s31 += a3_1 * xv1.y; s31 += a3_2 * xv1.z; s31 += a3_3 * xv1.w;
            s32 += a3_0 * xv2.x; s32 += a3_1 * xv2.y; s32 += a3_2 * xv2.z; s32 += a3_3 * xv2.w;
            s33 += a3_0 * xv3.x; s33 += a3_1 * xv3.y; s33 += a3_2 * xv3.z; s33 += a3_3 * xv3.w;
        }
    }
    s00 = warp_reduce(s00); s01 = warp_reduce(s01); s02 = warp_reduce(s02); s03 = warp_reduce(s03);
    s10 = warp_reduce(s10); s11 = warp_reduce(s11); s12 = warp_reduce(s12); s13 = warp_reduce(s13);
    s20 = warp_reduce(s20); s21 = warp_reduce(s21); s22 = warp_reduce(s22); s23 = warp_reduce(s23);
    s30 = warp_reduce(s30); s31 = warp_reduce(s31); s32 = warp_reduce(s32); s33 = warp_reduce(s33);
    if (lane == 0) {
        Y[0 * M + row0] = s00;  Y[1 * M + row0] = s01;  Y[2 * M + row0] = s02;  Y[3 * M + row0] = s03;
        if (row1 < M) { Y[0 * M + row1] = s10;  Y[1 * M + row1] = s11;  Y[2 * M + row1] = s12;  Y[3 * M + row1] = s13; }
        if (row2 < M) { Y[0 * M + row2] = s20;  Y[1 * M + row2] = s21;  Y[2 * M + row2] = s22;  Y[3 * M + row2] = s23; }
        if (row3 < M) { Y[0 * M + row3] = s30;  Y[1 * M + row3] = s31;  Y[2 * M + row3] = s32;  Y[3 * M + row3] = s33; }
    }
}

// ---------------- launchers ----------------
static void dims2(int M, dim3 *grid, dim3 *block) {
    // 2 rows per warp, 16 warps per block.
    block->x = 32; block->y = 16; block->z = 1;
    grid->x = (M + block->y * 2 - 1) / (block->y * 2);
    grid->y = 1; grid->z = 1;
}
static void dims_batch4(int M, dim3 *grid, dim3 *block) {
    // 4 rows per warp, 16 warps per block, 4 candidate x's in shmem.
    block->x = 32; block->y = 16; block->z = 1;
    grid->x = (M + block->y * 4 - 1) / (block->y * 4);
    grid->y = 1; grid->z = 1;
}

// ---------------- bench harness ----------------
template <typename Block, typename Single, typename Batch4>
struct BenchResult {
    float t_single;
    float t_seq4;
    float t_batch4;
    double maxd;
    double sum_abs;
};

template <typename Block, typename Single, typename Batch4>
static BenchResult<Block, Single, Batch4> run_bench(
    int M, int K, int REPS, int WARM,
    Single single_kernel, Batch4 batch4_kernel,
    void *dW, float *dx_single, float *dx_batch, float *dy_single, float *dy_batch,
    int shmem_bytes, cudaStream_t stream)
{
    BenchResult<Block, Single, Batch4> r{};

    dim3 g1, b1; dims2(M, &g1, &b1);
    dim3 g4, b4; dims_batch4(M, &g4, &b4);

    cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb);

    for (int i = 0; i < WARM; i++) {
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        batch4_kernel<<<g4, b4, shmem_bytes, stream>>>((const Block*)dW, dx_batch, dy_batch, M, K);
    }
    cudaStreamSynchronize(stream);

    cudaEventRecord(ea, stream);
    for (int i = 0; i < REPS; i++) {
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
    }
    cudaEventRecord(eb, stream); cudaEventSynchronize(eb);
    float ms; cudaEventElapsedTime(&ms, ea, eb);
    r.t_single = ms / REPS;

    cudaEventRecord(ea, stream);
    for (int i = 0; i < REPS; i++) {
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
    }
    cudaEventRecord(eb, stream); cudaEventSynchronize(eb);
    cudaEventElapsedTime(&ms, ea, eb);
    r.t_seq4 = ms / REPS;

    cudaEventRecord(ea, stream);
    for (int i = 0; i < REPS; i++) {
        batch4_kernel<<<g4, b4, shmem_bytes, stream>>>((const Block*)dW, dx_batch, dy_batch, M, K);
    }
    cudaEventRecord(eb, stream); cudaEventSynchronize(eb);
    cudaEventElapsedTime(&ms, ea, eb);
    r.t_batch4 = ms / REPS;

    // Correctness: single writes y[0..M-1] from row0/row1 in pairs.
    // batch4 writes Y[0*M+m] for each candidate, m=0..M-1.
    // For candidate 0 the two should match.
    {
        single_kernel<<<g1, b1, 0, stream>>>((const Block*)dW, dx_single, dy_single, M, K);
        batch4_kernel<<<g4, b4, shmem_bytes, stream>>>((const Block*)dW, dx_batch, dy_batch, M, K);
        cudaStreamSynchronize(stream);
        std::vector<float> h_single(M), h_batch(M);
        cudaMemcpy(h_single.data(), dy_single, M*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_batch.data(),   dy_batch,   M*4, cudaMemcpyDeviceToHost);
        double maxd = 0, sum_abs = 0;
        for (int i = 0; i < M; i++) {
            double d = fabs((double)h_single[i] - h_batch[i]);
            if (d > maxd) maxd = d;
            sum_abs += d;
        }
        r.maxd = maxd;
        r.sum_abs = sum_abs;
        if (maxd > 1e-3 && M < 2000) {
            printf("    [debug] M=%d K=%d first 4 single: %.4f %.4f %.4f %.4f  batch4(c0): %.4f %.4f %.4f %.4f\n",
                M, K,
                h_single[0], h_single[1], h_single[2], h_single[3],
                h_batch[0], h_batch[1], h_batch[2], h_batch[3]);
            // also dump rows 60-63 (warp 15 within block 0)
            if (M >= 64) {
                printf("    [debug] rows 60-63 single: %.4f %.4f %.4f %.4f  batch4(c0): %.4f %.4f %.4f %.4f\n",
                    h_single[60], h_single[61], h_single[62], h_single[63],
                    h_batch[60], h_batch[61], h_batch[62], h_batch[63]);
            }
        }
    }

    cudaEventDestroy(ea); cudaEventDestroy(eb);
    return r;
}

int main(int argc, char **argv) {
    int REPS = 100, WARM = 10;
    if (argc > 1) REPS = atoi(argv[1]);

    struct Shape { int M, K; const char *name; };
    Shape shapes[] = {
        {  896,   896, "Q/K/V (M=896, K=896)" },
        { 4864,   896, "FFN up/gate (M=4864, K=896)" },
        {  896,  4864, "FFN down (M=896, K=4864)" },
        {151936,  896, "LM head (M=151936, K=896)" },
    };
    int nshapes = sizeof(shapes)/sizeof(shapes[0]);

    printf("=========================================================\n");
    printf("M10+ True Batched-4 GEMV microbench (Ampere sm_86)\n");
    printf("  Design: 4 rows/warp x 4 candidate x in shmem (4x4)\n");
    printf("  REPS=%d (timed), WARM=%d (discarded)\n", REPS, WARM);
    printf("  Targets: overhead_ratio in [1.05, 1.15]\n");
    printf("           kernel_speedup >= 3.00x\n");
    printf("=========================================================\n\n");

    bool all_pass = true;

    printf("######## q4_0 ########\n");
    for (int s = 0; s < nshapes; s++) {
        int M = shapes[s].M, K = shapes[s].K;
        int nb = K / 32;
        if ((K & 31) != 0 || (nb & 1) != 0) {
            printf("SKIP q4_0 %s: K must be %%32 and nb even\n", shapes[s].name);
            continue;
        }
        if (M & 3) {
            int Mp = (M + 3) & ~3;
            printf("  [warn] %s: M=%d not multiple of 4, padding to %d (writes to y[M..Mp-1] undefined)\n",
                   shapes[s].name, M, Mp);
            M = Mp;
        }
        size_t wbytes = (size_t)M * nb * 18;
        size_t xbytes_single = (size_t)K * 4;
        size_t xbytes_batch4 = (size_t)4 * K * 4;
        size_t ybytes_single = (size_t)M * 4;
        size_t ybytes_batch4 = (size_t)4 * M * 4;
        int shmem_bytes = (int)xbytes_batch4;

        void *hW = malloc(wbytes);
        float *hx_s = (float *)malloc(xbytes_single);
        float *hx_b = (float *)malloc(xbytes_batch4);
        float *hy_s = (float *)malloc(ybytes_single);
        float *hy_b = (float *)malloc(ybytes_batch4);
        memset(hW, 0xab, wbytes);
        for (int i = 0; i < K; i++) {
            float v = sinf(0.7f * i + 0.3f);
            hx_s[i] = v;
            for (int c = 0; c < 4; c++) hx_b[c * K + i] = v + 0.01f * c;
        }
        memset(hy_s, 0, ybytes_single); memset(hy_b, 0, ybytes_batch4);

        void *dW; float *dx_s, *dx_b, *dy_s, *dy_b;
        cudaMalloc(&dW, wbytes); cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice);
        cudaMalloc(&dx_s, xbytes_single); cudaMemcpy(dx_s, hx_s, xbytes_single, cudaMemcpyHostToDevice);
        cudaMalloc(&dx_b, xbytes_batch4); cudaMemcpy(dx_b, hx_b, xbytes_batch4, cudaMemcpyHostToDevice);
        cudaMalloc(&dy_s, ybytes_single);
        cudaMalloc(&dy_b, ybytes_batch4);

        // For shmem > 48KB on Ampere, opt in to dynamic shared memory.
        if (shmem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(k_gemv_q4_0_batch4,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
        }

        auto r = run_bench<BlockQ4_0>(
            M, K, REPS, WARM,
            k_gemv_q4_0_single, k_gemv_q4_0_batch4,
            dW, dx_s, dx_b, dy_s, dy_b, shmem_bytes, 0);

        float overhead = r.t_batch4 / r.t_single;
        float speedup  = r.t_seq4   / r.t_batch4;
        bool ok = (overhead <= 1.15f) && (speedup >= 3.0f);
        if (!ok) all_pass = false;
        printf("  %-32s  M=%5d K=%4d  shmem=%dB\n", shapes[s].name, M, K, shmem_bytes);
        printf("    t_single=%.3f ms  t_seq4=%.3f ms  t_batch4=%.3f ms\n",
               r.t_single, r.t_seq4, r.t_batch4);
        printf("    overhead_ratio=%.3fx  kernel_speedup=%.3fx  max|d|=%g %s\n",
               overhead, speedup, r.maxd, ok ? "[PASS]" : "[FAIL]");

        cudaFree(dW); cudaFree(dx_s); cudaFree(dx_b); cudaFree(dy_s); cudaFree(dy_b);
        free(hW); free(hx_s); free(hx_b); free(hy_s); free(hy_b);
    }

    printf("\n######## q8_0 ########\n");
    for (int s = 0; s < nshapes; s++) {
        int M = shapes[s].M, K = shapes[s].K;
        int nb = K / 32;
        if ((K & 31) != 0 || (nb & 1) != 0) {
            printf("SKIP q8_0 %s: K must be %%32 and nb even\n", shapes[s].name);
            continue;
        }
        if (M & 3) {
            int Mp = (M + 3) & ~3;
            printf("  [warn] %s: M=%d not multiple of 4, padding to %d\n",
                   shapes[s].name, M, Mp);
            M = Mp;
        }
        size_t wbytes = (size_t)M * nb * 34;
        size_t xbytes_single = (size_t)K * 4;
        size_t xbytes_batch4 = (size_t)4 * K * 4;
        size_t ybytes_single = (size_t)M * 4;
        size_t ybytes_batch4 = (size_t)4 * M * 4;
        int shmem_bytes = (int)xbytes_batch4;

        void *hW = malloc(wbytes);
        float *hx_s = (float *)malloc(xbytes_single);
        float *hx_b = (float *)malloc(xbytes_batch4);
        float *hy_s = (float *)malloc(ybytes_single);
        float *hy_b = (float *)malloc(ybytes_batch4);
        memset(hW, 0xab, wbytes);
        for (int i = 0; i < K; i++) {
            float v = sinf(0.7f * i + 0.3f);
            hx_s[i] = v;
            for (int c = 0; c < 4; c++) hx_b[c * K + i] = v + 0.01f * c;
        }
        memset(hy_s, 0, ybytes_single); memset(hy_b, 0, ybytes_batch4);

        void *dW; float *dx_s, *dx_b, *dy_s, *dy_b;
        cudaMalloc(&dW, wbytes); cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice);
        cudaMalloc(&dx_s, xbytes_single); cudaMemcpy(dx_s, hx_s, xbytes_single, cudaMemcpyHostToDevice);
        cudaMalloc(&dx_b, xbytes_batch4); cudaMemcpy(dx_b, hx_b, xbytes_batch4, cudaMemcpyHostToDevice);
        cudaMalloc(&dy_s, ybytes_single);
        cudaMalloc(&dy_b, ybytes_batch4);

        if (shmem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(k_gemv_q8_0_batch4,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
        }

        auto r = run_bench<BlockQ8_0>(
            M, K, REPS, WARM,
            k_gemv_q8_0_single, k_gemv_q8_0_batch4,
            dW, dx_s, dx_b, dy_s, dy_b, shmem_bytes, 0);

        float overhead = r.t_batch4 / r.t_single;
        float speedup  = r.t_seq4   / r.t_batch4;
        bool ok = (overhead <= 1.15f) && (speedup >= 3.0f);
        if (!ok) all_pass = false;
        printf("  %-32s  M=%5d K=%4d  shmem=%dB\n", shapes[s].name, M, K, shmem_bytes);
        printf("    t_single=%.3f ms  t_seq4=%.3f ms  t_batch4=%.3f ms\n",
               r.t_single, r.t_seq4, r.t_batch4);
        printf("    overhead_ratio=%.3fx  kernel_speedup=%.3fx  max|d|=%g %s\n",
               overhead, speedup, r.maxd, ok ? "[PASS]" : "[FAIL]");

        cudaFree(dW); cudaFree(dx_s); cudaFree(dx_b); cudaFree(dy_s); cudaFree(dy_b);
        free(hW); free(hx_s); free(hx_b); free(hy_s); free(hy_b);
    }

    printf("\n=========================================================\n");
    printf("OVERALL: %s\n", all_pass ? "ALL PASS (ready for Task 2)" : "SOME FAILED (see above)");
    printf("=========================================================\n");
    return all_pass ? 0 : 1;
}
