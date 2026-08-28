// Task 2 microbench: k_gemv_q4_0_ffn_fused (one launch for Gate + Up +
// SwiGLU).
//
// Each FFN block in qwen2 currently launches:
//   1. k_gemv_q4_0(W_gate, X)        — 1 launch
//   2. k_gemv_q4_0(W_up,   X)        — 1 launch
//   3. k_swiglu_apply(g, u, h)       — 1 launch
// = 3 launches per layer. Each launch costs ~3 microseconds of
// overhead; 24 layers × 3 launches × 3 μs ≈ 216 μs. The fused kernel
// collapses this to ONE launch (load X into shmem once, compute
// both Gate and Up for the same 4 output rows in the same block,
// then apply SwiGLU before writing) — saving ~2 × 3 μs × 24
// layers = 144 μs/token ≈ +30 tok/s at 250 tok/s.
//
// Design (mirrors micro_qkv_fused.cu):
//   - 4 rows per warp (same as k_gemv_q4_0_v4 / k_fused_swiglu_q4_0
//     extended to 4 rows).
//   - Per block: 4 warps × 32 lanes = 128 threads.
//   - Input X of size K floats is loaded into shmem ONCE at the
//     start, then reused for both Gate and Up GEMV streams in the
//     same warp.
//   - After both GEMVs, SiLU(G) * U is computed in registers and
//     written to H. SiLU matches the existing k_swiglu_apply
//     formula:  x / (1 + exp(-x)).
//
// Per-block shmem: K * 4 bytes (K=896 -> 3.5 KB; trivial vs the
// 100 KB/SM Ampere limit).
//
// This file is a STANDALONE MICROBENCH. It does not modify the
// engine. It defines its own copies of BlockQ4_0,
// k_fused_swiglu_q4_0 (the existing 2-rows-per-warp path, used as
// the "3 sequential" baseline — one for the GEMV and a tiny
// elementwise kernel for SwiGLU), and the new
// k_gemv_q4_0_ffn_fused. Same pattern as micro_qkv_fused.cu /
// micro_v4.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Block layout (matches kernels/gemv_q4_cuda.cu).
// ---------------------------------------------------------------------------
typedef struct {
    __half d;
    uint8_t qs[16];
} BlockQ4_0;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float silu_host(float x) {
    return x / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Reference 2-rows-per-warp fused (Gate + Up + SwiGLU). Used as part
// of the "3 sequential" baseline. Mirrors k_fused_swiglu_q4_0 in
// kernels/gemv_q4_cuda.cu.
// ---------------------------------------------------------------------------
__global__ void k_fused_swiglu_q4_0_ref(const BlockQ4_0 *__restrict__ W_gate,
                                        const BlockQ4_0 *__restrict__ W_up,
                                        const float    *__restrict__ X,
                                        float          *__restrict__ H,
                                        int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nb   = K / 32;
    const uint32_t *gw0 = (const uint32_t *)((const char *)W_gate + (long)row0 * nb * 18);
    const uint32_t *gw1 = (const uint32_t *)((const char *)W_gate + (long)row1 * nb * 18);
    const uint32_t *uw0 = (const uint32_t *)((const char *)W_up   + (long)row0 * nb * 18);
    const uint32_t *uw1 = (const uint32_t *)((const char *)W_up   + (long)row1 * nb * 18);
    float sg0 = 0.0f, su0 = 0.0f, sg1 = 0.0f, su1 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const unsigned short dga = (unsigned short)
            (((18 * b) & 2) ? (gw0[wsc] >> 16) : (gw0[wsc] & 0xFFFFu));
        const unsigned short dgb = (unsigned short)
            (((18 * b) & 2) ? (gw1[wsc] >> 16) : (gw1[wsc] & 0xFFFFu));
        const unsigned short dua = (unsigned short)
            (((18 * b) & 2) ? (uw0[wsc] >> 16) : (uw0[wsc] & 0xFFFFu));
        const unsigned short dub = (unsigned short)
            (((18 * b) & 2) ? (uw1[wsc] >> 16) : (uw1[wsc] & 0xFFFFu));
        const float dga_ = __half2float(__ushort_as_half(dga));
        const float dgb_ = __half2float(__ushort_as_half(dgb));
        const float dua_ = __half2float(__ushort_as_half(dua));
        const float dub_ = __half2float(__ushort_as_half(dub));
        const int g0 = (18 * b + 2) >> 2;
        const float4 *x4 = (const float4 *)(X + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t gla = gw0[g0 + k], glb = gw1[g0 + k];
            const uint32_t ula = uw0[g0 + k], ulb = uw1[g0 + k];
            const uint32_t gva = sh ? __byte_perm(gla, gw0[g0 + k + 1], 0x5432) : gla;
            const uint32_t gvb = sh ? __byte_perm(glb, gw1[g0 + k + 1], 0x5432) : glb;
            const uint32_t uva = sh ? __byte_perm(ula, uw0[g0 + k + 1], 0x5432) : ula;
            const uint32_t uvb = sh ? __byte_perm(ulb, uw1[g0 + k + 1], 0x5432) : ulb;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            sg0 += (float)((int)(gva         & 0xFu) - 8) * dga_ * xa.x;
            sg0 += (float)((int)((gva >>  4) & 0xFu) - 8) * dga_ * xb.x;
            sg0 += (float)((int)((gva >>  8) & 0xFu) - 8) * dga_ * xa.y;
            sg0 += (float)((int)((gva >> 12) & 0xFu) - 8) * dga_ * xb.y;
            sg0 += (float)((int)((gva >> 16) & 0xFu) - 8) * dga_ * xa.z;
            sg0 += (float)((int)((gva >> 20) & 0xFu) - 8) * dga_ * xb.z;
            sg0 += (float)((int)((gva >> 24) & 0xFu) - 8) * dga_ * xa.w;
            sg0 += (float)((int)(gva >> 28) - 8) * dga_ * xb.w;
            su0 += (float)((int)(uva         & 0xFu) - 8) * dua_ * xa.x;
            su0 += (float)((int)((uva >>  4) & 0xFu) - 8) * dua_ * xb.x;
            su0 += (float)((int)((uva >>  8) & 0xFu) - 8) * dua_ * xa.y;
            su0 += (float)((int)((uva >> 12) & 0xFu) - 8) * dua_ * xb.y;
            su0 += (float)((int)((uva >> 16) & 0xFu) - 8) * dua_ * xa.z;
            su0 += (float)((int)((uva >> 20) & 0xFu) - 8) * dua_ * xb.z;
            su0 += (float)((int)((uva >> 24) & 0xFu) - 8) * dua_ * xa.w;
            su0 += (float)((int)(uva >> 28) - 8) * dua_ * xb.w;
            sg1 += (float)((int)(gvb         & 0xFu) - 8) * dgb_ * xa.x;
            sg1 += (float)((int)((gvb >>  4) & 0xFu) - 8) * dgb_ * xb.x;
            sg1 += (float)((int)((gvb >>  8) & 0xFu) - 8) * dgb_ * xa.y;
            sg1 += (float)((int)((gvb >> 12) & 0xFu) - 8) * dgb_ * xb.y;
            sg1 += (float)((int)((gvb >> 16) & 0xFu) - 8) * dgb_ * xa.z;
            sg1 += (float)((int)((gvb >> 20) & 0xFu) - 8) * dgb_ * xb.z;
            sg1 += (float)((int)((gvb >> 24) & 0xFu) - 8) * dgb_ * xa.w;
            sg1 += (float)((int)(gvb >> 28) - 8) * dgb_ * xb.w;
            su1 += (float)((int)(uvb         & 0xFu) - 8) * dub_ * xa.x;
            su1 += (float)((int)((uvb >>  4) & 0xFu) - 8) * dub_ * xb.x;
            su1 += (float)((int)((uvb >>  8) & 0xFu) - 8) * dub_ * xa.y;
            su1 += (float)((int)((uvb >> 12) & 0xFu) - 8) * dub_ * xb.y;
            su1 += (float)((int)((uvb >> 16) & 0xFu) - 8) * dub_ * xa.z;
            su1 += (float)((int)((uvb >> 20) & 0xFu) - 8) * dub_ * xb.z;
            su1 += (float)((int)((uvb >> 24) & 0xFu) - 8) * dub_ * xa.w;
            su1 += (float)((int)(uvb >> 28) - 8) * dub_ * xb.w;
        }
    }
    sg0 = warp_reduce_sum(sg0);
    su0 = warp_reduce_sum(su0);
    sg1 = warp_reduce_sum(sg1);
    su1 = warp_reduce_sum(su1);
    if (lane == 0) {
        H[row0] = silu(sg0) * su0;
        if (row1 < M) H[row1] = silu(sg1) * su1;
    }
}

// ---------------------------------------------------------------------------
// 3-sequential elementwise SwiGLU apply (the "third" launch). Same
// math as k_swiglu_apply in qwen2_cuda.cu.
// ---------------------------------------------------------------------------
__global__ void k_swiglu_apply_ref(const float *__restrict__ g,
                                   const float *__restrict__ u,
                                   float       *__restrict__ h,
                                   int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    const float gi = g[i];
    h[i] = silu(gi) * u[i];
}

// ---------------------------------------------------------------------------
// Fused FFN kernel: ONE launch covers (Gate, Up, SwiGLU). Per block:
// 2 warps × 32 lanes = 64 threads; each warp covers 2 output rows
// (same parallelism as the 3-seq plain GEMV). X is read from
// global memory twice (once for Gate, once for Up) — Ampere L1
// caches the second read for free. The whole point of fusion
// is to eliminate the per-launch overhead of the 3-seq path.
// ---------------------------------------------------------------------------
__global__ void k_gemv_q4_0_ffn_fused(const BlockQ4_0 *__restrict__ W_gate,
                                      const BlockQ4_0 *__restrict__ W_up,
                                      const float    *__restrict__ X,
                                      float          *__restrict__ H,
                                      int M, int K) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int row0 = (blockIdx.x * blockDim.y + warp) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;

    const int nb = K / 32;
    const uint32_t *gw0 = (const uint32_t *)((const char *)W_gate + (long)row0 * nb * 18);
    const uint32_t *gw1 = (const uint32_t *)((const char *)W_gate + (long)row1 * nb * 18);
    const uint32_t *uw0 = (const uint32_t *)((const char *)W_up   + (long)row0 * nb * 18);
    const uint32_t *uw1 = (const uint32_t *)((const char *)W_up   + (long)row1 * nb * 18);
    float sg0 = 0.0f, sg1 = 0.0f;
    float su0 = 0.0f, su1 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const unsigned short dga = (unsigned short)
            (((18 * b) & 2) ? (gw0[wsc] >> 16) : (gw0[wsc] & 0xFFFFu));
        const unsigned short dgb = (unsigned short)
            (((18 * b) & 2) ? (gw1[wsc] >> 16) : (gw1[wsc] & 0xFFFFu));
        const unsigned short dua = (unsigned short)
            (((18 * b) & 2) ? (uw0[wsc] >> 16) : (uw0[wsc] & 0xFFFFu));
        const unsigned short dub = (unsigned short)
            (((18 * b) & 2) ? (uw1[wsc] >> 16) : (uw1[wsc] & 0xFFFFu));
        const float dga_ = __half2float(__ushort_as_half(dga));
        const float dgb_ = __half2float(__ushort_as_half(dgb));
        const float dua_ = __half2float(__ushort_as_half(dua));
        const float dub_ = __half2float(__ushort_as_half(dub));
        const int g0 = (18 * b + 2) >> 2;
        const float4 *x4 = (const float4 *)(X + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t gla = __ldg(gw0 + g0 + k);
            const uint32_t glb = __ldg(gw1 + g0 + k);
            const uint32_t ula = __ldg(uw0 + g0 + k);
            const uint32_t ulb = __ldg(uw1 + g0 + k);
            const uint32_t gva = sh ? __byte_perm(gla, __ldg(gw0 + g0 + k + 1), 0x5432) : gla;
            const uint32_t gvb = sh ? __byte_perm(glb, __ldg(gw1 + g0 + k + 1), 0x5432) : glb;
            const uint32_t uva = sh ? __byte_perm(ula, __ldg(uw0 + g0 + k + 1), 0x5432) : ula;
            const uint32_t uvb = sh ? __byte_perm(ulb, __ldg(uw1 + g0 + k + 1), 0x5432) : ulb;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            sg0 += (float)((int)(gva         & 0xFu) - 8) * dga_ * xa.x;
            sg0 += (float)((int)((gva >>  4) & 0xFu) - 8) * dga_ * xb.x;
            sg0 += (float)((int)((gva >>  8) & 0xFu) - 8) * dga_ * xa.y;
            sg0 += (float)((int)((gva >> 12) & 0xFu) - 8) * dga_ * xb.y;
            sg0 += (float)((int)((gva >> 16) & 0xFu) - 8) * dga_ * xa.z;
            sg0 += (float)((int)((gva >> 20) & 0xFu) - 8) * dga_ * xb.z;
            sg0 += (float)((int)((gva >> 24) & 0xFu) - 8) * dga_ * xa.w;
            sg0 += (float)((int)(gva >> 28) - 8) * dga_ * xb.w;
            su0 += (float)((int)(uva         & 0xFu) - 8) * dua_ * xa.x;
            su0 += (float)((int)((uva >>  4) & 0xFu) - 8) * dua_ * xb.x;
            su0 += (float)((int)((uva >>  8) & 0xFu) - 8) * dua_ * xa.y;
            su0 += (float)((int)((uva >> 12) & 0xFu) - 8) * dua_ * xb.y;
            su0 += (float)((int)((uva >> 16) & 0xFu) - 8) * dua_ * xa.z;
            su0 += (float)((int)((uva >> 20) & 0xFu) - 8) * dua_ * xb.z;
            su0 += (float)((int)((uva >> 24) & 0xFu) - 8) * dua_ * xa.w;
            su0 += (float)((int)(uva >> 28) - 8) * dua_ * xb.w;
            sg1 += (float)((int)(gvb         & 0xFu) - 8) * dgb_ * xa.x;
            sg1 += (float)((int)((gvb >>  4) & 0xFu) - 8) * dgb_ * xb.x;
            sg1 += (float)((int)((gvb >>  8) & 0xFu) - 8) * dgb_ * xa.y;
            sg1 += (float)((int)((gvb >> 12) & 0xFu) - 8) * dgb_ * xb.y;
            sg1 += (float)((int)((gvb >> 16) & 0xFu) - 8) * dgb_ * xa.z;
            sg1 += (float)((int)((gvb >> 20) & 0xFu) - 8) * dgb_ * xb.z;
            sg1 += (float)((int)((gvb >> 24) & 0xFu) - 8) * dgb_ * xa.w;
            sg1 += (float)((int)(gvb >> 28) - 8) * dgb_ * xb.w;
            su1 += (float)((int)(uvb         & 0xFu) - 8) * dub_ * xa.x;
            su1 += (float)((int)((uvb >>  4) & 0xFu) - 8) * dub_ * xb.x;
            su1 += (float)((int)((uvb >>  8) & 0xFu) - 8) * dub_ * xa.y;
            su1 += (float)((int)((uvb >> 12) & 0xFu) - 8) * dub_ * xb.y;
            su1 += (float)((int)((uvb >> 16) & 0xFu) - 8) * dub_ * xa.z;
            su1 += (float)((int)((uvb >> 20) & 0xFu) - 8) * dub_ * xb.z;
            su1 += (float)((int)((uvb >> 24) & 0xFu) - 8) * dub_ * xa.w;
            su1 += (float)((int)(uvb >> 28) - 8) * dub_ * xb.w;
        }
    }
    sg0 = warp_reduce_sum(sg0);
    sg1 = warp_reduce_sum(sg1);
    su0 = warp_reduce_sum(su0);
    su1 = warp_reduce_sum(su1);
    if (lane == 0) {
        H[row0] = silu(sg0) * su0;
        if (row1 < M) H[row1] = silu(sg1) * su1;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: scalar q4_0 dequant + dot for one (gate, up) row.
// ---------------------------------------------------------------------------
static float cpu_dot(const BlockQ4_0 *Wrow, const float *x, int K) {
    const int nb = K / 32;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const BlockQ4_0 blk = Wrow[b];
        const float d = __half2float(blk.d);
        const float *xb = x + b * 32;
        for (int i = 0; i < 16; i++) {
            const int lo = (int)(blk.qs[i] & 0x0F) - 8;
            const int hi = (int)(blk.qs[i] >> 4) - 8;
            sum += (float)lo * d * xb[i] + (float)hi * d * xb[i + 16];
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Host launchers.
// ---------------------------------------------------------------------------
// (3-launch path moved to launch_3seq_faithful below)

// Plain 2-rows/warp q4_0 GEMV (no fused activation), used as the
// "Gate GEMV" and "Up GEMV" in the 3-seq baseline. Mirrors the
// production k_gemv_q4_0 kernel in gemv_q4_cuda.cu.
__global__ void k_gemv_q4_0_plain(const BlockQ4_0 *__restrict__ W,
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
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

static void launch_3seq_faithful(const BlockQ4_0 *Wgate, const BlockQ4_0 *Wup,
                                 const float *X, float *G, float *U, float *H,
                                 int M, int K, cudaStream_t s) {
    // 1) Gate GEMV (2 rows/warp, 4 warps/block, 8 rows/block)
    {
        const int warps = 4;
        const int blocks = (M + warps * 2 - 1) / (warps * 2);
        dim3 grid(blocks, 1, 1);
        dim3 block(32, warps, 1);
        k_gemv_q4_0_plain<<<grid, block, 0, s>>>(Wgate, X, G, M, K);
    }
    // 2) Up GEMV
    {
        const int warps = 4;
        const int blocks = (M + warps * 2 - 1) / (warps * 2);
        dim3 grid(blocks, 1, 1);
        dim3 block(32, warps, 1);
        k_gemv_q4_0_plain<<<grid, block, 0, s>>>(Wup, X, U, M, K);
    }
    // 3) SwiGLU apply (elementwise)
    {
        const int threads = 256;
        const int blocks = (M + threads - 1) / threads;
        k_swiglu_apply_ref<<<blocks, threads, 0, s>>>(G, U, H, M);
    }
}

static void launch_fused(const BlockQ4_0 *Wgate, const BlockQ4_0 *Wup,
                         const float *X, float *H,
                         int M, int K, cudaStream_t s) {
    const int warps = 4;
    const int blocks = (M + warps * 2 - 1) / (warps * 2);
    dim3 grid(blocks, 1, 1);
    dim3 block(32, warps, 1);
    k_gemv_q4_0_ffn_fused<<<grid, block, 0, s>>>(Wgate, Wup, X, H, M, K);
}

// ---------------------------------------------------------------------------
// Microbench driver.
// ---------------------------------------------------------------------------
struct Shape {
    const char *name;
    int M, K;
};

static float time_path(int which, const Shape &sh, const BlockQ4_0 *dWg,
                       const BlockQ4_0 *dWu, const float *dX,
                       float *dG, float *dU, float *dH,
                       int iters, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 10; i++) {
        if (which == 0) launch_3seq_faithful(dWg, dWu, dX, dG, dU, dH, sh.M, sh.K, s);
        else            launch_fused(dWg, dWu, dX, dH, sh.M, sh.K, s);
    }
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; i++) {
        if (which == 0) launch_3seq_faithful(dWg, dWu, dX, dG, dU, dH, sh.M, sh.K, s);
        else            launch_fused(dWg, dWu, dX, dH, sh.M, sh.K, s);
    }
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

int main(int argc, char **argv) {
    int iters = 200;
    if (argc > 1) iters = atoi(argv[1]);

    printf("micro_ffn_fused: %d timed iters per path\n", iters);

    // qwen2.5-0.5b FFN shape: M=4864, K=896 (gate/up projections).
    std::vector<Shape> shapes = {
        {"qwen2.5-0.5b FFN", 4864, 896},
    };

    // RNG seed for reproducible activations/weights.
    srand(42);

    for (const Shape &sh : shapes) {
        printf("\n=== shape: %s  M=%d K=%d ===\n", sh.name, sh.M, sh.K);

        const int nb = sh.K / 32;
        std::vector<BlockQ4_0> hWg(sh.M * nb);
        std::vector<BlockQ4_0> hWu(sh.M * nb);
        for (auto &b : hWg) { b.d = __float2half(0.05f); for (int j = 0; j < 16; j++) b.qs[j] = (uint8_t)rand(); }
        for (auto &b : hWu) { b.d = __float2half(0.05f); for (int j = 0; j < 16; j++) b.qs[j] = (uint8_t)rand(); }
        std::vector<float> hX(sh.K);
        for (auto &v : hX) v = (float)(rand() % 1000) / 1000.0f - 0.5f;

        // Device buffers.
        BlockQ4_0 *dWg, *dWu;
        float *dX, *dG, *dU, *dH;
        cudaMalloc(&dWg, hWg.size() * sizeof(BlockQ4_0));
        cudaMalloc(&dWu, hWu.size() * sizeof(BlockQ4_0));
        cudaMalloc(&dX,  hX.size() * sizeof(float));
        cudaMalloc(&dG,  sh.M * sizeof(float));
        cudaMalloc(&dU,  sh.M * sizeof(float));
        cudaMalloc(&dH,  sh.M * sizeof(float));
        cudaMemcpy(dWg, hWg.data(), hWg.size() * sizeof(BlockQ4_0), cudaMemcpyHostToDevice);
        cudaMemcpy(dWu, hWu.data(), hWu.size() * sizeof(BlockQ4_0), cudaMemcpyHostToDevice);
        cudaMemcpy(dX,  hX.data(),  hX.size()  * sizeof(float),        cudaMemcpyHostToDevice);
        cudaMemset(dG, 0, sh.M * sizeof(float));
        cudaMemset(dU, 0, sh.M * sizeof(float));
        cudaMemset(dH, 0, sh.M * sizeof(float));

        cudaStream_t s;
        cudaStreamCreate(&s);
        float t3 = time_path(0, sh, dWg, dWu, dX, dG, dU, dH, iters, s);
        float tf = time_path(1, sh, dWg, dWu, dX, dG, dU, dH, iters, s);
        cudaStreamDestroy(s);

        // Correctness check: compare fused output to CPU scalar ref
        // of SwiGLU(Gate, Up) for a small subset of rows.
        std::vector<float> hH3(sh.M), hHf(sh.M);
        launch_3seq_faithful(dWg, dWu, dX, dG, dU, dH, sh.M, sh.K, 0);
        cudaMemcpy(hH3.data(), dH, sh.M * sizeof(float), cudaMemcpyDeviceToHost);
        launch_fused(dWg, dWu, dX, dH, sh.M, sh.K, 0);
        cudaMemcpy(hHf.data(), dH, sh.M * sizeof(float), cudaMemcpyDeviceToHost);

        double max_err_fused = 0;
        double max_err_3seq  = 0;
        double max_diff_paths = 0;
        for (int m = 0; m < sh.M; m++) {
            float g = cpu_dot(hWg.data() + (long)m * nb, hX.data(), sh.K);
            float u = cpu_dot(hWu.data() + (long)m * nb, hX.data(), sh.K);
            float ref = silu_host(g) * u;
            max_err_fused = std::max(max_err_fused, (double)std::abs(hHf[m] - ref));
            max_err_3seq  = std::max(max_err_3seq,  (double)std::abs(hH3[m] - ref));
            max_diff_paths = std::max(max_diff_paths, (double)std::abs(hHf[m] - hH3[m]));
        }

        printf("time_3seq  = %.3f us/iter\n", t3 * 1e3f);
        printf("time_fused = %.3f us/iter\n", tf * 1e3f);
        printf("savings    = %.3f us/iter (%.1f%%)\n",
               (t3 - tf) * 1e3f, (t3 - tf) / t3 * 100.0);
        printf("max |fused - cpu|  = %.2e\n", max_err_fused);
        printf("max |3seq  - cpu|  = %.2e\n", max_err_3seq);
        printf("max |fused - 3seq| = %.2e\n", max_diff_paths);

        cudaFree(dWg); cudaFree(dWu);
        cudaFree(dX);
        cudaFree(dG); cudaFree(dU); cudaFree(dH);
    }

    return 0;
}
