// q4_0 GEMV primitives for the Qwen2 engine (M6 correctness rewrite).
//
// GGML q4_0 block layout: 32 values per block = fp16 scale d + 16 bytes qs.
// Dequant: x[j]   = ((qs[j]   & 0x0F) - 8) * d   for j in [0,15]
//          x[j+16]= ((qs[j] >> 4)    - 8) * d   for j in [0,15]
// All kernels here use that pairing (the old k_gemv_q4_0 paired nibble-low
// with x[2i], which is NOT the GGML layout and produced wrong dot products).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    half d;            // fp16 scale
    uint8_t qs[16];    // 32 packed nibbles
} BlockQ4_0;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// One warp computes one output row: y[m] = W[m,:] @ x[:].
// V2: vectorized weight streaming, port of the k_logits_q8_0 winner.
// Alignment reasoning (same approach as k_logits_q8_0): row stride is
// nb*18 bytes, which is 4-byte aligned iff nb is even (18*nb = 4*9*nb/2).
// Both layer shapes satisfy that: K=896 -> nb=28, K=4864 -> nb=152.
// Block scale d lives at byte 18*b (high half of word 18b>>2 iff b odd);
// the 16-byte qs payload starts at byte 18b+2, which is 4-byte aligned
// only for odd b, so even-b blocks merge two adjacent aligned words with
// __byte_perm. When merging, one extra trailing word is touched (next
// block's scale) - in-row for every block except possibly the last;
// nb even guarantees the last block is odd-b (aligned path), so no
// out-of-row access. Requires nb even.
// V2: two rows per warp. Each warp computes output rows 2r and 2r+1,
// sharing every x float4 load between them (x re-read pressure halves;
// weight bytes in flight double per lane). Same nb-even contract as V2.
__global__ void k_gemv_q4_0(const BlockQ4_0 *__restrict__ W,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;                    // caller pads M to even

    const int lane = threadIdx.x;
    const int nb = K / 32;                    // q4_0 blocks per row
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;                 // word holding blk.d
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (18 * b + 2) >> 2;              // first qs word
        const int sh  = (18 * b + 2) & 2;              // 2 => misaligned merge
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const float4 xa = x4[k];       // x[4k .. 4k+3]   <- low nibbles of bytes 4k..4k+3
            const float4 xb = x4[k + 4];   // x[16+4k .. +3]  <- high nibbles of bytes 4k..4k+3
            /* q4_0 byte j packs weight j (low nibble) and weight j+16 (high
             * nibble); word k holds bytes 4k..4k+3, so each byte's LOW nibble
             * pairs with xa and the SAME BYTE's HIGH nibble pairs with xb.
             * The previous version paired consecutive nibbles against
             * xa.x..xb.w, scrambling weight->x mapping => garbage output. */
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

// GELU tanh approximation (Hendrycks), used by gemma/gemma2 FFN epilogue.
__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Fused MLP up-projection: h[m] = act(W_gate[m,:] @ x) * (W_up[m,:] @ x).
// act = 0: silu (qwen2/llama/qwen3 SwiGLU); act = 1: gelu-tanh (gemma GeGLU):
//   gelu(x) = 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
// V2: two rows per warp (same sharing scheme as k_gemv_q4_0; nb-even contract).
__global__ void k_fused_swiglu_q4_0(const BlockQ4_0 *__restrict__ W_gate,
                                    const BlockQ4_0 *__restrict__ W_up,
                                    const float *__restrict__ x,
                                    float *__restrict__ out,
                                    int M, int K, int act) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;

    const int lane = threadIdx.x;
    const int nb = K / 32;
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
        const float4 *x4 = (const float4 *)(x + b * 32);
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
            /* same per-byte low->xa / high->xb pairing fix as k_gemv_q4_0 */
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
        out[row0] = (act ? gelu_tanh(sg0) : sg0 / (1.0f + expf(-sg0))) * su0;
        if (row1 < M)
            out[row1] = (act ? gelu_tanh(sg1) : sg1 / (1.0f + expf(-sg1))) * su1;
    }
}

// LM head / tied-embedding projection over vocab rows (same math as GEMV).
// Scalar fallback. Kept for the odd-nb case where the V2 uint32 streaming
// cannot satisfy its alignment contract; tt_logits_dispatch routes here then.
__global__ void k_logits_q4_0(const BlockQ4_0 *__restrict__ W,
                              const float *__restrict__ x,
                              float *__restrict__ logits,
                              int vocab, int K) {
    const int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= vocab) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const BlockQ4_0 *rowW = W + (long)v * nb;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        BlockQ4_0 blk = rowW[b];
        const float d = __half2float(blk.d);
        const float *xb = x + b * 32;
#pragma unroll
        for (int i = 0; i < 16; i++)
            sum += (((blk.qs[i] & 0x0F) - 8) * d) * xb[i] +
                   (((blk.qs[i] >> 4) - 8) * d) * xb[i + 16];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) logits[v] = sum;
}

// V2 LM head over q4_0 weights. Two vocab rows per warp, sharing the
// float4 x loads. Same uint32-streaming + __byte_perm merge scheme as
// k_gemv_q4_0 / k_logits_q8_0; the same per-byte low-nibble->xa /
// high-nibble->xb pairing (byte j: low nibble -> x[4k+(j%4)], high
// nibble -> x[16+4k+(j%4)] when j in [4k, 4k+3]) is required for the
// dot product to match scalar k_logits_q4_0. Requires nb = K/32 even
// (caller must check; otherwise fall back to scalar). Caller pads vocab
// to even (vocab is always even in our LLM heads).
__global__ void k_logits_q4_0_v2(const BlockQ4_0 *__restrict__ W,
                                 const float *__restrict__ x,
                                 float *__restrict__ logits,
                                 int vocab, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= vocab) return;
    const int row1 = row0 + 1;

    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;                 // word holding blk.d
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (18 * b + 2) >> 2;              // first qs word
        const int sh  = (18 * b + 2) & 2;              // 2 => misaligned merge
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const float4 xa = x4[k];       // x[4k .. 4k+3]   <- low nibbles of bytes 4k..4k+3
            const float4 xb = x4[k + 4];   // x[16+4k .. +3]  <- high nibbles of bytes 4k..4k+3
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
        logits[row0] = s0;
        if (row1 < vocab) logits[row1] = s1;
    }
}

// M9.5+ V4: 4-rows-per-warp q4_0 LM head. Same uint32-streaming +
// __byte_perm merge scheme as k_logits_q4_0_v2; extends to 4 rows per
// warp so each lane keeps more weight bytes in flight and amortizes
// the float4 x loads across 4 outputs (vs 2 in V2). Bit-exact against
// V2 (verified in tools/micro_v4.cu; max |V2-V4| = 0 for the LM head
// shape and across the FFN shapes in the microbench). Requires
//   1. M >= 4 and M is a multiple of 4 (caller pads; vocab is
//      even and 151936 for the qwen2 LM head is divisible by 4).
//   2. K % 32 == 0 (q4_0 contract; standard for our LLM shapes).
//   3. nb = K/32 even (same as V2; qwen2.5 K=896 -> nb=28 even,
//      llama-3.2-1b K=2048 -> nb=64 even, gemma q4_0 K=1024 -> nb=32).
// When M is not a multiple of 4 the trailing 1..3 rows fall through
// the row>=M guard. When M is not a multiple of 8 there can be 1..3
// unprocessed rows at the tail; the caller (dispatcher) MUST fall back
// to V2 in that case so those rows are still produced.
__global__ void k_logits_q4_0_v4(const BlockQ4_0 *__restrict__ W,
                                 const float *__restrict__ x,
                                 float *__restrict__ logits,
                                 int vocab, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= vocab) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;

    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 18);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;                 // word holding blk.d
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
        const int a0 = (18 * b + 2) >> 2;              // first qs word
        const int sh  = (18 * b + 2) & 2;              // 2 => misaligned merge
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t lc = rw2[a0 + k];
            const uint32_t ld = rw3[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const uint32_t vc = sh ? __byte_perm(lc, rw2[a0 + k + 1], 0x5432) : lc;
            const uint32_t vd = sh ? __byte_perm(ld, rw3[a0 + k + 1], 0x5432) : ld;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            // row0
            s0 += (float)((int)(va         & 0xFu) - 8) * da * xa.x;
            s0 += (float)((int)((va >>  4) & 0xFu) - 8) * da * xb.x;
            s0 += (float)((int)((va >>  8) & 0xFu) - 8) * da * xa.y;
            s0 += (float)((int)((va >> 12) & 0xFu) - 8) * da * xb.y;
            s0 += (float)((int)((va >> 16) & 0xFu) - 8) * da * xa.z;
            s0 += (float)((int)((va >> 20) & 0xFu) - 8) * da * xb.z;
            s0 += (float)((int)((va >> 24) & 0xFu) - 8) * da * xa.w;
            s0 += (float)((int)(va >> 28) - 8) * da * xb.w;
            // row1
            s1 += (float)((int)(vb         & 0xFu) - 8) * db * xa.x;
            s1 += (float)((int)((vb >>  4) & 0xFu) - 8) * db * xb.x;
            s1 += (float)((int)((vb >>  8) & 0xFu) - 8) * db * xa.y;
            s1 += (float)((int)((vb >> 12) & 0xFu) - 8) * db * xb.y;
            s1 += (float)((int)((vb >> 16) & 0xFu) - 8) * db * xa.z;
            s1 += (float)((int)((vb >> 20) & 0xFu) - 8) * db * xb.z;
            s1 += (float)((int)((vb >> 24) & 0xFu) - 8) * db * xa.w;
            s1 += (float)((int)(vb >> 28) - 8) * db * xb.w;
            // row2
            s2 += (float)((int)(vc         & 0xFu) - 8) * dc * xa.x;
            s2 += (float)((int)((vc >>  4) & 0xFu) - 8) * dc * xb.x;
            s2 += (float)((int)((vc >>  8) & 0xFu) - 8) * dc * xa.y;
            s2 += (float)((int)((vc >> 12) & 0xFu) - 8) * dc * xb.y;
            s2 += (float)((int)((vc >> 16) & 0xFu) - 8) * dc * xa.z;
            s2 += (float)((int)((vc >> 20) & 0xFu) - 8) * dc * xb.z;
            s2 += (float)((int)((vc >> 24) & 0xFu) - 8) * dc * xa.w;
            s2 += (float)((int)(vc >> 28) - 8) * dc * xb.w;
            // row3
            s3 += (float)((int)(vd         & 0xFu) - 8) * dd * xa.x;
            s3 += (float)((int)((vd >>  4) & 0xFu) - 8) * dd * xb.x;
            s3 += (float)((int)((vd >>  8) & 0xFu) - 8) * dd * xa.y;
            s3 += (float)((int)((vd >> 12) & 0xFu) - 8) * dd * xb.y;
            s3 += (float)((int)((vd >> 16) & 0xFu) - 8) * dd * xa.z;
            s3 += (float)((int)((vd >> 20) & 0xFu) - 8) * dd * xb.z;
            s3 += (float)((int)((vd >> 24) & 0xFu) - 8) * dd * xa.w;
            s3 += (float)((int)(vd >> 28) - 8) * dd * xb.w;
        }
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    s2 = warp_reduce_sum(s2);
    s3 = warp_reduce_sum(s3);
    if (lane == 0) {
        logits[row0] = s0;
        if (row1 < vocab) logits[row1] = s1;
        if (row2 < vocab) logits[row2] = s2;
        if (row3 < vocab) logits[row3] = s3;
    }
}

// M9.5+ V4: 4-rows-per-warp q4_0 layer GEMV. Same shape contract as
// k_gemv_q4_0 (V2) but with 4 rows per warp instead of 2. M must be
// a multiple of 4; caller pads and falls back to V2 when not. Bit-
// exact against V2 (microbench verified across LM head + FFN shapes).
__global__ void k_gemv_q4_0_v4(const BlockQ4_0 *__restrict__ W,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;

    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 18);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
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
        const int sh  = (18 * b + 2) & 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t lc = rw2[a0 + k];
            const uint32_t ld = rw3[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const uint32_t vc = sh ? __byte_perm(lc, rw2[a0 + k + 1], 0x5432) : lc;
            const uint32_t vd = sh ? __byte_perm(ld, rw3[a0 + k + 1], 0x5432) : ld;
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
            s2 += (float)((int)(vc         & 0xFu) - 8) * dc * xa.x;
            s2 += (float)((int)((vc >>  4) & 0xFu) - 8) * dc * xb.x;
            s2 += (float)((int)((vc >>  8) & 0xFu) - 8) * dc * xa.y;
            s2 += (float)((int)((vc >> 12) & 0xFu) - 8) * dc * xb.y;
            s2 += (float)((int)((vc >> 16) & 0xFu) - 8) * dc * xa.z;
            s2 += (float)((int)((vc >> 20) & 0xFu) - 8) * dc * xb.z;
            s2 += (float)((int)((vc >> 24) & 0xFu) - 8) * dc * xa.w;
            s2 += (float)((int)(vc >> 28) - 8) * dc * xb.w;
            s3 += (float)((int)(vd         & 0xFu) - 8) * dd * xa.x;
            s3 += (float)((int)((vd >>  4) & 0xFu) - 8) * dd * xb.x;
            s3 += (float)((int)((vd >>  8) & 0xFu) - 8) * dd * xa.y;
            s3 += (float)((int)((vd >> 12) & 0xFu) - 8) * dd * xb.y;
            s3 += (float)((int)((vd >> 16) & 0xFu) - 8) * dd * xa.z;
            s3 += (float)((int)((vd >> 20) & 0xFu) - 8) * dd * xb.z;
            s3 += (float)((int)((vd >> 24) & 0xFu) - 8) * dd * xa.w;
            s3 += (float)((int)(vd >> 28) - 8) * dd * xb.w;
        }
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    s2 = warp_reduce_sum(s2);
    s3 = warp_reduce_sum(s3);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
        if (row2 < M) y[row2] = s2;
        if (row3 < M) y[row3] = s3;
    }
}

// q8_0 block: fp16 scale d + 32 int8 values.
typedef struct {
    half d;
    int8_t qs[32];
} BlockQ8_0;

// LM head over q8_0 weights (output.weight in some GGUF conversions).
// V2: vectorized weight streaming. Row stride is nb*34 bytes; for K=896
// (nb=28) that is 952 bytes = 4-byte aligned, so the whole row can be
// viewed as an array of uint32 words. A block's qs payload starts at byte
// 34*b+2, which is 4-byte aligned only for odd b, so even-b blocks merge
// two adjacent aligned words with __byte_perm. When merging, one extra
// trailing word is touched (next block's scale) - in-row for every block
// except possibly the last; nb even guarantees the last block is odd-b
// (aligned path), so no out-of-row access. Requires nb even.
__global__ void k_logits_q8_0(const BlockQ8_0 *__restrict__ W,
                              const float *__restrict__ x,
                              float *__restrict__ logits,
                              int vocab, int K) {
    const int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= vocab) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *roww = (const uint32_t *)((const char *)W + (long)v * nb * 34);
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;                 // word holding blk.d
        const unsigned short d16 = (unsigned short)
            (((34 * b) & 2) ? (roww[wsc] >> 16) : (roww[wsc] & 0xFFFFu));
        const float d = __half2float(__ushort_as_half(d16));
        const int a0 = (34 * b + 2) >> 2;              // first qs word
        const int sh  = (34 * b + 2) & 2;              // 2 => misaligned merge
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t lo = roww[a0 + k];
            const uint32_t vv = sh ? __byte_perm(lo, roww[a0 + k + 1], 0x5432) : lo;
            const float4 xv = x4[k];
            sum += ((float)((int)(vv << 24) >> 24)) * d * xv.x;
            sum += ((float)((int)(vv << 16) >> 24)) * d * xv.y;
            sum += ((float)((int)(vv <<  8) >> 24)) * d * xv.z;
            sum += ((float)((int)(vv       ) >> 24)) * d * xv.w;
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) logits[v] = sum;
}

// V4: 4 rows per warp for Q8_0 LM head (pre-loads x vector into registers)
__global__ void k_logits_q8_0_v4(const BlockQ8_0 *__restrict__ W,
                                 const float *__restrict__ x,
                                 float *__restrict__ logits,
                                 int vocab, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= vocab) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;

    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 34);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 34);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 34);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 34);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;                 // word holding blk.d
        const int sh  = (34 * b + 2) & 2;              // 2 => misaligned merge
        const unsigned short d16a = (unsigned short)(((34 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)(((34 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const unsigned short d16c = (unsigned short)(((34 * b) & 2) ? (rw2[wsc] >> 16) : (rw2[wsc] & 0xFFFFu));
        const unsigned short d16d = (unsigned short)(((34 * b) & 2) ? (rw3[wsc] >> 16) : (rw3[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));
        const int a0 = (34 * b + 2) >> 2;              // first qs word
        const float4 *x4 = (const float4 *)(x + b * 32);

        float4 xv[8];
#pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x4[k];

#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t lc = rw2[a0 + k];
            const uint32_t ld = rw3[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const uint32_t vc = sh ? __byte_perm(lc, rw2[a0 + k + 1], 0x5432) : lc;
            const uint32_t vd = sh ? __byte_perm(ld, rw3[a0 + k + 1], 0x5432) : ld;
            const float4 xk = xv[k];

            s0 += ((float)((int)(va << 24) >> 24)) * da * xk.x;
            s0 += ((float)((int)(va << 16) >> 24)) * da * xk.y;
            s0 += ((float)((int)(va <<  8) >> 24)) * da * xk.z;
            s0 += ((float)((int)(va       ) >> 24)) * da * xk.w;

            s1 += ((float)((int)(vb << 24) >> 24)) * db * xk.x;
            s1 += ((float)((int)(vb << 16) >> 24)) * db * xk.y;
            s1 += ((float)((int)(vb <<  8) >> 24)) * db * xk.z;
            s1 += ((float)((int)(vb       ) >> 24)) * db * xk.w;

            s2 += ((float)((int)(vc << 24) >> 24)) * dc * xk.x;
            s2 += ((float)((int)(vc << 16) >> 24)) * dc * xk.y;
            s2 += ((float)((int)(vc <<  8) >> 24)) * dc * xk.z;
            s2 += ((float)((int)(vc       ) >> 24)) * dc * xk.w;

            s3 += ((float)((int)(vd << 24) >> 24)) * dd * xk.x;
            s3 += ((float)((int)(vd << 16) >> 24)) * dd * xk.y;
            s3 += ((float)((int)(vd <<  8) >> 24)) * dd * xk.z;
            s3 += ((float)((int)(vd       ) >> 24)) * dd * xk.w;
        }
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    s2 = warp_reduce_sum(s2);
    s3 = warp_reduce_sum(s3);

    if (lane == 0) {
        logits[row0] = s0;
        if (row1 < vocab) logits[row1] = s1;
        if (row2 < vocab) logits[row2] = s2;
        if (row3 < vocab) logits[row3] = s3;
    }
}

// V2 layer GEMV over q8_0 weights (closes the qwen3-0.6b-q8_0 0.25x gap
// to llama.cpp CUDA). Same shape as k_gemv_q4_0: two rows per warp, each
// lane strides the K/32 q8_0 blocks, each block loads 8 uint32 words of
// qs and sign-extends in-line to compute `sum += (int8)q * d * x` per
// element. x is re-read once per row but loaded as float4, so two rows
// per warp halve the x bandwidth pressure vs one-row-per-warp. Requires
// nb even (same contract as k_gemv_q4_0; qwen3-0.6b q8_0 K values are
// 1024/3072 -> nb=32/96, both even).
__global__ void k_gemv_q8_0(const BlockQ8_0 *__restrict__ W,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;                    // caller pads M to even

    const int lane = threadIdx.x;
    const int nb = K / 32;                        // q8_0 blocks per row
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 34);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 34);
    float s0 = 0.0f, s1 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;                 // word holding blk.d
        const int sh  = (34 * b + 2) & 2;              // 2 => misaligned merge
        const unsigned short d16a = (unsigned short)
            (((34 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((34 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (34 * b + 2) >> 2;              // first qs word
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
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// M10+ True Batched-4 GEMV (q4_0 + q8_0).
//
// Design: each warp processes 4 weight rows x 4 candidate x vectors in
// a single launch, producing 16 outputs per warp. The 4 candidate x
// vectors are loaded into shared memory ONCE per block and reused by
// all 16 warps; weights are streamed from gmem (4 rows in parallel so
// each q4_0/q8_0 word is read once per K-position). This is the
// 3.2-4.4x speedup kernel verified in tools/micro_batch4.cu.
//
// Constraints (caller must check):
//   - K % 32 == 0
//   - nb = K/32 even (same uint32 streaming + __byte_perm merge contract
//     as k_gemv_q4_0 / k_gemv_q8_0 V2 kernels)
//   - M multiple of 4 (4 rows per warp)
//   - shmem budget: 4 * K * 4 bytes for the 4 candidate x's; for K=4864
//     that is 76 KB (single block/SM on Ampere; opt in via
//     cudaFuncSetAttribute(MaxDynamicSharedMemorySize) when > 48 KB).
//
// Layout:
//   X is [4, K] flat, indexed as X[c*K + k]
//   Y is [4, M] flat, indexed as Y[c*M + m]
//   W is [M, K] (q4_0 or q8_0 blocks)
//
// Output[ c ][ m ] = sum_k dequant( W[m, k] ) * X[c, k]
//
// Returns -1 from the launcher if any constraint is violated (caller
// should fall back to 4 sequential single calls).

// M10+ q4_0 BATCHED-4: 4 rows/warp x 4 candidate x in shmem.
__global__ void k_gemv_q4_0_batch4(const BlockQ4_0 *__restrict__ W,
                                    const float    *__restrict__ X,
                                    float          *__restrict__ Y,
                                    int M, int K) {
    extern __shared__ float sx[];   // [4][K]
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = (blockIdx.x * blockDim.y + warp) * 4;
    const int row1 = row0 + 1, row2 = row0 + 2, row3 = row0 + 3;
    const int nb   = K / 32;

    // Cooperatively load 4 candidate x vectors into shmem (float4 vectorized).
    // The load + __syncthreads MUST execute on all warps; the row-out-of-
    // range early-return is deferred until after the barrier so the
    // collective sync is well-defined even when only a few warps have
    // real work (e.g. M < blockDim.y * 4). Use a per-warp strided load
    // (32 lanes cover the buffer in stride-32 chunks); this works for
    // any K >= 32 (when K < 32, the launcher rejects the call).
    {
        const int total_f4 = 4 * K / 4;          // K float4 per K floats
        const float4 *src  = (const float4 *)X;
        float4       *dst  = (float4 *)sx;
        for (int i = lane; i < total_f4; i += 32) {
            dst[i] = src[i];
        }
    }
    __syncthreads();
    if (row0 >= M) return;       // safe: barrier already passed

    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 18);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 18);

    const float4 *sx4_0 = (const float4 *)(sx + 0 * K);
    const float4 *sx4_1 = (const float4 *)(sx + 1 * K);
    const float4 *sx4_2 = (const float4 *)(sx + 2 * K);
    const float4 *sx4_3 = (const float4 *)(sx + 3 * K);

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
        /* x float4 offset for block b: each q4_0 block covers 32 values
         * = 8 float4s, so the float4 stride is 8 per b. */
        const int x_off = b * 8;

#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la0 = __ldg(rw0 + a0 + k);
            const uint32_t la1 = __ldg(rw1 + a0 + k);
            const uint32_t la2 = __ldg(rw2 + a0 + k);
            const uint32_t la3 = __ldg(rw3 + a0 + k);
            const uint32_t va0 = sh ? __byte_perm(la0, __ldg(rw0 + a0 + k + 1), 0x5432) : la0;
            const uint32_t va1 = sh ? __byte_perm(la1, __ldg(rw1 + a0 + k + 1), 0x5432) : la1;
            const uint32_t va2 = sh ? __byte_perm(la2, __ldg(rw2 + a0 + k + 1), 0x5432) : la2;
            const uint32_t va3 = sh ? __byte_perm(la3, __ldg(rw3 + a0 + k + 1), 0x5432) : la3;

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

            const float4 xa0 = sx4_0[x_off + k],     xb0 = sx4_0[x_off + k + 4];
            const float4 xa1 = sx4_1[x_off + k],     xb1 = sx4_1[x_off + k + 4];
            const float4 xa2 = sx4_2[x_off + k],     xb2 = sx4_2[x_off + k + 4];
            const float4 xa3 = sx4_3[x_off + k],     xb3 = sx4_3[x_off + k + 4];

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
    s00 = warp_reduce_sum(s00); s01 = warp_reduce_sum(s01); s02 = warp_reduce_sum(s02); s03 = warp_reduce_sum(s03);
    s10 = warp_reduce_sum(s10); s11 = warp_reduce_sum(s11); s12 = warp_reduce_sum(s12); s13 = warp_reduce_sum(s13);
    s20 = warp_reduce_sum(s20); s21 = warp_reduce_sum(s21); s22 = warp_reduce_sum(s22); s23 = warp_reduce_sum(s23);
    s30 = warp_reduce_sum(s30); s31 = warp_reduce_sum(s31); s32 = warp_reduce_sum(s32); s33 = warp_reduce_sum(s33);
    if (lane == 0) {
        Y[0 * M + row0] = s00;  Y[1 * M + row0] = s01;  Y[2 * M + row0] = s02;  Y[3 * M + row0] = s03;
        if (row1 < M) { Y[0 * M + row1] = s10;  Y[1 * M + row1] = s11;  Y[2 * M + row1] = s12;  Y[3 * M + row1] = s13; }
        if (row2 < M) { Y[0 * M + row2] = s20;  Y[1 * M + row2] = s21;  Y[2 * M + row2] = s22;  Y[3 * M + row2] = s23; }
        if (row3 < M) { Y[0 * M + row3] = s30;  Y[1 * M + row3] = s31;  Y[2 * M + row3] = s32;  Y[3 * M + row3] = s33; }
    }
}

// M10+ q8_0 BATCHED-4: 4 rows/warp x 4 candidate x in shmem.
__global__ void k_gemv_q8_0_batch4(const BlockQ8_0 *__restrict__ W,
                                    const float    *__restrict__ X,
                                    float          *__restrict__ Y,
                                    int M, int K) {
    extern __shared__ float sx[];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = (blockIdx.x * blockDim.y + warp) * 4;
    const int row1 = row0 + 1, row2 = row0 + 2, row3 = row0 + 3;
    const int nb   = K / 32;

    // Load + __syncthreads first (collective, all warps must participate);
    // defer the row-out-of-range early-return until after the barrier.
    {
        const int total_f4 = 4 * K / 4;
        const float4 *src  = (const float4 *)X;
        float4       *dst  = (float4 *)sx;
        // Per-warp strided load covers every slot in stride-32 chunks.
        for (int i = lane; i < total_f4; i += 32) {
            dst[i] = src[i];
        }
    }
    __syncthreads();
    if (row0 >= M) return;

    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 34);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 34);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 34);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 34);

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
        /* x float4 offset for block b: 8 float4s per q8_0 block. */
        const int x_off = b * 8;

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

            const float4 xv0 = sx4_0[x_off + k];
            const float4 xv1 = sx4_1[x_off + k];
            const float4 xv2 = sx4_2[x_off + k];
            const float4 xv3 = sx4_3[x_off + k];

            s00 += a0_0 * xv0.x; s00 += a0_1 * xv0.y; s00 += a0_2 * xv0.z; s00 += a0_3 * xv0.w;
            s01 += a0_0 * xv1.x; s01 += a0_1 * xv1.y; s01 += a0_2 * xv1.z; s01 += a0_3 * xv1.w;
            s02 += a0_0 * xv2.x; s02 += a0_1 * xv2.y; s02 += a0_2 * xv2.z; s02 += a0_3 * xv2.w;
            s03 += a0_0 * xv3.x; s03 += a0_1 * xv3.y; s03 += a0_2 * xv3.z; s03 += a0_3 * xv3.w;
            s10 += a1_0 * xv0.x; s10 += a1_1 * xv0.y; s10 += a1_2 * xv0.z; s10 += a1_3 * xv0.w;
            s11 += a1_0 * xv1.x; s11 += a1_1 * xv1.y; s11 += a1_2 * xv1.z; s11 += a1_3 * xv1.w;
            s12 += a1_0 * xv2.x; s12 += a1_1 * xv2.y; s12 += a1_2 * xv2.z; s12 += a1_3 * xv2.w;
            s13 += a1_0 * xv3.x; s13 += a1_1 * xv3.y; s13 += a1_2 * xv3.z; s13 += a1_3 * xv3.w;
            s20 += a2_0 * xv0.x; s20 += a2_1 * xv0.y; s20 += a2_2 * xv0.z; s20 += a2_3 * xv0.w;
            s21 += a2_0 * xv1.x; s21 += a2_1 * xv1.y; s21 += a2_2 * xv1.z; s21 += a2_3 * xv1.w;
            s22 += a2_0 * xv2.x; s22 += a2_1 * xv2.y; s22 += a2_2 * xv2.z; s22 += a2_3 * xv2.w;
            s23 += a2_0 * xv3.x; s23 += a2_1 * xv3.y; s23 += a2_2 * xv3.z; s23 += a2_3 * xv3.w;
            s30 += a3_0 * xv0.x; s30 += a3_1 * xv0.y; s30 += a3_2 * xv0.z; s30 += a3_3 * xv0.w;
            s31 += a3_0 * xv1.x; s31 += a3_1 * xv1.y; s31 += a3_2 * xv1.z; s31 += a3_3 * xv1.w;
            s32 += a3_0 * xv2.x; s32 += a3_1 * xv2.y; s32 += a3_2 * xv2.z; s32 += a3_3 * xv2.w;
            s33 += a3_0 * xv3.x; s33 += a3_1 * xv3.y; s33 += a3_2 * xv3.z; s33 += a3_3 * xv3.w;
        }
    }
    s00 = warp_reduce_sum(s00); s01 = warp_reduce_sum(s01); s02 = warp_reduce_sum(s02); s03 = warp_reduce_sum(s03);
    s10 = warp_reduce_sum(s10); s11 = warp_reduce_sum(s11); s12 = warp_reduce_sum(s12); s13 = warp_reduce_sum(s13);
    s20 = warp_reduce_sum(s20); s21 = warp_reduce_sum(s21); s22 = warp_reduce_sum(s22); s23 = warp_reduce_sum(s23);
    s30 = warp_reduce_sum(s30); s31 = warp_reduce_sum(s31); s32 = warp_reduce_sum(s32); s33 = warp_reduce_sum(s33);
    if (lane == 0) {
        Y[0 * M + row0] = s00;  Y[1 * M + row0] = s01;  Y[2 * M + row0] = s02;  Y[3 * M + row0] = s03;
        if (row1 < M) { Y[0 * M + row1] = s10;  Y[1 * M + row1] = s11;  Y[2 * M + row1] = s12;  Y[3 * M + row1] = s13; }
        if (row2 < M) { Y[0 * M + row2] = s20;  Y[1 * M + row2] = s21;  Y[2 * M + row2] = s22;  Y[3 * M + row2] = s23; }
        if (row3 < M) { Y[0 * M + row3] = s30;  Y[1 * M + row3] = s31;  Y[2 * M + row3] = s32;  Y[3 * M + row3] = s33; }
    }
}

// ---------------- M9.5 WMMA tensor-core MMQ path ----------------------- *
// Tensor-core mma.sync.aligned.m16n16k16.row.col.f16.f16.f32.f32 path
// for q4_0 GEMV. The N=16 dim of the mma is filled with 16 COPIES of
// the same single-token x (decode is GEMV, N=1), so 15/16 of the
// tensor-core work is wasted -- this kernel is INTENTIONALLY slower than
// the scalar V2 above on Ampere consumer (RTX 3050 sm_86). It exists
// to (a) ship a working tensor-core MMQ path for the parity gate,
// (b) provide infrastructure for future batched-decode (T>=8 real x's
// in N) which WILL win. Dispatcher gates it behind TT_USE_WMMA=1 so
// the production decode path stays on V2 (measured ~150us for
// 4864x4864 q4_0 GEMV on RTX 3050; WMMA path ~1.7ms there).
//
// Block: 1 warp = 32 threads. Per mma: M=16 output rows, K=16 reduction,
// N=16 (wasted 15/16 for single-token decode). Each tile: cooperatively
// dequant 16 rows of q4_0 to fp16 in shmem, then mma_sync. After K loop:
// store 16x16 c_frag to shmem, take N=0 column as the 16 output values.
//
// Layout proven by the smoke test in /tmp/wmma_clean: row-major A * row-
// major B = correct full 16x16 result. col-major B with ldb=16 has been
// observed to write only 2/16 columns on this driver, so we use row-major
// B with the B matrix filled as 16 horizontal copies of x (B[k*16 + n] =
// x_sh[k0 + k] for all n in 0..15).
//
// The kernel is unconditionally compiled (the wmma intrinsics are sm_70+
// only; this file is built with -gencode arch=compute_86,code=sm_86
// already, so the arch gate is satisfied). The launcher checks at
// runtime whether to dispatch to it (TT_USE_WMMA=1) or fall back to V2.
#include <mma.h>
__global__ void k_gemv_wmma_q4_0(const BlockQ4_0 *__restrict__ W,
                                  const float *__restrict__ x,
                                  float *__restrict__ y,
                                  int M, int K) {
    using namespace nvcuda;
    const int row0 = blockIdx.x * 16;
    if (row0 >= M) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;                       // q4_0 blocks per row
    const int n_tiles = K / 16;                  // K-tiles per warp

    // Shmem: x as fp16 (K elements) + per-tile working space.
    extern __shared__ __half smem[];
    __half *sx = smem;                           // K fp16

    // Load x as fp16 cooperatively.
    for (int i = lane; i < K; i += 32) sx[i] = __float2half(x[i]);
    __syncwarp();

    // Accumulator: 16x16 fp32. Initialize to zero.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    // Per-K-tile working memory in shmem.
    __shared__ __half sW[16 * 16];               // 16 rows x 16 K fp16
    __shared__ __half sB[16 * 16];               // 16 K x 16 N fp16

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 16;

        // Build A tile: 16 rows of W, K=16 elements starting at k0.
        // Dequant q4_0 to fp16 in row-major 16x16. Each lane handles
        // 256/32 = 8 elements (i, row = i/16, col = i%16).
        for (int i = lane; i < 16 * 16; i += 32) {
            const int row = i / 16;
            const int col = i % 16;
            const int kk = k0 + col;             // absolute K index
            const int blk_idx = kk / 32;         // q4_0 block within row
            const int in_blk = kk & 31;          // 0..31
            const BlockQ4_0 *blk = W + (long)(row0 + row) * nb + blk_idx;
            const float d = __half2float(blk->d);
            // in_blk < 16 -> low nibble of qs[in_blk]; in_blk >= 16 -> high nibble of qs[in_blk - 16]
            const int q_byte = blk->qs[in_blk & 15];
            const int nib = (in_blk < 16) ? (q_byte & 0xF) : ((q_byte >> 4) & 0xF);
            const float w = ((float)nib - 8.f) * d;
            sW[i] = __float2half(w);
        }
        __syncwarp();

        // Build B tile: 16x16 where each row k = sx[k0 + k], replicated 16x in N.
        // B[k][n] = x_sh[k0 + k] for all n in 0..15 (so 15/16 of the
        // tensor-core mma work is wasted on single-token decode; this is
        // the fundamental cost of doing GEMV on mma which is GEMM-shaped).
        for (int i = lane; i < 16 * 16; i += 32) {
            const int k = i / 16;                // 0..15
            sB[i] = sx[k0 + k];                  // all 16 N cols = same x value
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, sW, 16);
        wmma::load_matrix_sync(b_frag, sB, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store 16x16 accumulator to shmem, take N=0 column as 16 outputs.
    __shared__ float sC[16 * 16];
    wmma::store_matrix_sync(sC, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    if (lane < 16) y[row0 + lane] = sC[lane * 16 + 0];
}

// Embedding lookup: dequantize row `tok` of a q4_0 matrix into dx[0..dim).
__global__ void k_embed_q4_0(const BlockQ4_0 *__restrict__ W, int tok,
                             float *__restrict__ dx, int dim) {
    const int b = threadIdx.x + blockIdx.x * blockDim.x;
    const int nb = dim / 32;
    if (b >= nb) return;
    BlockQ4_0 blk = W[(long)tok * nb + b];
    const float d = __half2float(blk.d);
    float *out = dx + b * 32;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        out[i]      = ((blk.qs[i] & 0x0F) - 8) * d;
        out[i + 16] = ((blk.qs[i] >> 4) - 8) * d;
    }
}

/*
 * 2D Tiled Q4_0 Batched GEMM Kernel for Prompt Prefill (Conflict-Free Shmem)
 * Matrix math: Y = X * W^T
 *   X: [N, K] float (row-major activations)
 *   W: [M, K] BlockQ4_0 (row-major quantized weights)
 *   Y: [N, M] float (row-major output)
 *
 * Tile layout: BLOCK_M = 64, BLOCK_N = 32, BLOCK_K = 32
 * Threads per CTA: dim3 block(16, 16) = 256 threads
 * Register tiling: 2 tokens (N) x 4 rows (M) per thread => 8 accumulators per thread
 * Shared memory:
 *   __shared__ float sX[32][33] (padding +1 to eliminate 32-bank conflicts)
 *   __shared__ float sW_d[65]   (padding +1)
 *   __shared__ uint32_t sW_v[64][5] (padding +1 to eliminate bank conflicts)
 */
__global__ __launch_bounds__(256, 4)
void k_gemm_q4_0_prefill(
    const void *__restrict__ dW,
    const float *__restrict__ dX,
    float *__restrict__ dY,
    int M, int K, int N)
{
    const int tx = threadIdx.x; // 0..15 (M dimension)
    const int ty = threadIdx.y; // 0..15 (N dimension)
    const int tid = ty * 16 + tx; // 0..255

    const int m_base = blockIdx.x * 64 + tx * 4;
    const int n_base = blockIdx.y * 32 + ty * 2;

    const int nb = K / 32; // blocks per row

    __shared__ float sX[32][33];
    __shared__ float sW_d[65];
    __shared__ uint32_t sW_v[64][5];

    float acc[2][4];
    #pragma unroll
    for (int in = 0; in < 2; in++) {
        #pragma unroll
        for (int im = 0; im < 4; im++) {
            acc[in][im] = 0.0f;
        }
    }

    // Cooperative loading X: 256 threads load 32 tokens x 32 floats = 1024 floats (1 float4 per thread)
    const int n_load = tid / 8;     // 0..31
    const int k_vec_load = tid % 8; // 0..7
    const int n_global = blockIdx.y * 32 + n_load;

    for (int k_tile = 0; k_tile < nb; k_tile++) {
        // 1. Cooperative load X tile into sX
        const int k_global = k_tile * 32 + k_vec_load * 4;
        float4 x_vec;
        if (n_global < N && (k_global + 3) < K) {
            x_vec = *reinterpret_cast<const float4*>(&dX[n_global * K + k_global]);
        } else {
            x_vec.x = (n_global < N && (k_global + 0) < K) ? dX[n_global * K + k_global + 0] : 0.0f;
            x_vec.y = (n_global < N && (k_global + 1) < K) ? dX[n_global * K + k_global + 1] : 0.0f;
            x_vec.z = (n_global < N && (k_global + 2) < K) ? dX[n_global * K + k_global + 2] : 0.0f;
            x_vec.w = (n_global < N && (k_global + 3) < K) ? dX[n_global * K + k_global + 3] : 0.0f;
        }
        sX[n_load][k_vec_load * 4 + 0] = x_vec.x;
        sX[n_load][k_vec_load * 4 + 1] = x_vec.y;
        sX[n_load][k_vec_load * 4 + 2] = x_vec.z;
        sX[n_load][k_vec_load * 4 + 3] = x_vec.w;

        // 2. Cooperative load W tile into sW_d and sW_v (64 blocks loaded once per CTA)
        if (tid < 64) {
            int m_row = blockIdx.x * 64 + tid;
            if (m_row < M) {
                const uint32_t *rw = (const uint32_t *)((const char *)dW + (long)m_row * nb * 18);
                const int wsc = (18 * k_tile) >> 2;
                const unsigned short d16 = (unsigned short)(((18 * k_tile) & 2) ? (rw[wsc] >> 16) : (rw[wsc] & 0xFFFFu));
                sW_d[tid] = __half2float(__ushort_as_half(d16));

                const int a0 = (18 * k_tile + 2) >> 2;
                const int sh = (18 * k_tile + 2) & 2;

                #pragma unroll
                for (int k_sub = 0; k_sub < 4; k_sub++) {
                    uint32_t la = rw[a0 + k_sub];
                    uint32_t la_next = sh ? rw[a0 + k_sub + 1] : 0;
                    sW_v[tid][k_sub] = sh ? __byte_perm(la, la_next, 0x5432) : la;
                }
            } else {
                sW_d[tid] = 0.0f;
                #pragma unroll
                for (int k_sub = 0; k_sub < 4; k_sub++) sW_v[tid][k_sub] = 0;
            }
        }

        __syncthreads();

        // 3. Read weights from shared memory for 4 rows handled by thread
        float da[4];
        uint32_t va[4][4];
        #pragma unroll
        for (int im = 0; im < 4; im++) {
            int m_local = tx * 4 + im;
            da[im] = sW_d[m_local];
            #pragma unroll
            for (int k_sub = 0; k_sub < 4; k_sub++) {
                va[im][k_sub] = sW_v[m_local][k_sub];
            }
        }

        // 4. Compute dot products for 2 tokens x 4 rows
        #pragma unroll
        for (int k_sub = 0; k_sub < 4; k_sub++) {
            const int n0_local = ty * 2;
            const int n1_local = ty * 2 + 1;

            float x0_low0 = sX[n0_local][4 * k_sub + 0];
            float x0_low1 = sX[n0_local][4 * k_sub + 1];
            float x0_low2 = sX[n0_local][4 * k_sub + 2];
            float x0_low3 = sX[n0_local][4 * k_sub + 3];

            float x0_high0 = sX[n0_local][16 + 4 * k_sub + 0];
            float x0_high1 = sX[n0_local][16 + 4 * k_sub + 1];
            float x0_high2 = sX[n0_local][16 + 4 * k_sub + 2];
            float x0_high3 = sX[n0_local][16 + 4 * k_sub + 3];

            float x1_low0 = sX[n1_local][4 * k_sub + 0];
            float x1_low1 = sX[n1_local][4 * k_sub + 1];
            float x1_low2 = sX[n1_local][4 * k_sub + 2];
            float x1_low3 = sX[n1_local][4 * k_sub + 3];

            float x1_high0 = sX[n1_local][16 + 4 * k_sub + 0];
            float x1_high1 = sX[n1_local][16 + 4 * k_sub + 1];
            float x1_high2 = sX[n1_local][16 + 4 * k_sub + 2];
            float x1_high3 = sX[n1_local][16 + 4 * k_sub + 3];

            #pragma unroll
            for (int im = 0; im < 4; im++) {
                uint32_t v = va[im][k_sub];
                float d = da[im];

                int q0 = (int)(v & 0x0Fu) - 8;
                int q1 = (int)((v >> 4) & 0x0Fu) - 8;
                int q2 = (int)((v >> 8) & 0x0Fu) - 8;
                int q3 = (int)((v >> 12) & 0x0Fu) - 8;
                int q4 = (int)((v >> 16) & 0x0Fu) - 8;
                int q5 = (int)((v >> 20) & 0x0Fu) - 8;
                int q6 = (int)((v >> 24) & 0x0Fu) - 8;
                int q7 = (int)(v >> 28) - 8;

                float sum0 = (float)q0 * x0_low0  + (float)q1 * x0_high0
                           + (float)q2 * x0_low1  + (float)q3 * x0_high1
                           + (float)q4 * x0_low2  + (float)q5 * x0_high2
                           + (float)q6 * x0_low3  + (float)q7 * x0_high3;

                float sum1 = (float)q0 * x1_low0  + (float)q1 * x1_high0
                           + (float)q2 * x1_low1  + (float)q3 * x1_high1
                           + (float)q4 * x1_low2  + (float)q5 * x1_high2
                           + (float)q6 * x1_low3  + (float)q7 * x1_high3;

                acc[0][im] += sum0 * d;
                acc[1][im] += sum1 * d;
            }
        }

        __syncthreads();
    }

    // 5. Store Y accumulators
    #pragma unroll
    for (int in = 0; in < 2; in++) {
        int n_g = n_base + in;
        if (n_g < N) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int m_g = m_base + im;
                if (m_g < M) {
                    dY[n_g * M + m_g] = acc[in][im];
                }
            }
        }
    }
}

/*
 * Tensor Core WMMA Batched Q4_0 Prefill GEMM Kernel
 * Matrix math: Y = X * W^T
 *   X: [N, K] float (row-major activations)
 *   W: [M, K] BlockQ4_0 (row-major quantized weights)
 *   Y: [N, M] float (row-major output)
 *
 * Tile layout: BLOCK_M = 128, BLOCK_N = 32, BLOCK_K = 32
 * CTA layout: 256 threads (8 warps: 2 warps in N, 4 warps in M)
 *   warp_n = warp_id / 4 (0..1, each warp computes 16 tokens)
 *   warp_m = warp_id % 4 (0..3, each warp computes 32 output rows = 2 WMMA tiles)
 * WMMA fragment sizes: 16x16x16 (matrix_a FP16, matrix_b FP16, accumulator FP32)
 */
__global__ __launch_bounds__(256)
void k_gemm_wmma_q4_0_prefill(
    const void *__restrict__ dW,
    const float *__restrict__ dX,
    float *__restrict__ dY,
    int M, int K, int N)
{
    using namespace nvcuda;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int warp_n = warp_id / 4; // 0..1
    const int warp_m = warp_id % 4; // 0..3

    const int cta_m_base = blockIdx.x * 128;
    const int cta_n_base = blockIdx.y * 32;

    const int my_m_base = cta_m_base + warp_m * 32;
    const int my_n_base = cta_n_base + warp_n * 16;

    __align__(16) __shared__ half s_X[32][32];
    __align__(16) __shared__ half s_W[128][32];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;
    wmma::fill_fragment(c_frag0, 0.0f);
    wmma::fill_fragment(c_frag1, 0.0f);

    const int nb = K / 32;
    const int n_k_tiles = K / 32;

    for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
        const int k_base = k_tile * 32;

        // 1. Cooperative load X [32 x 32] = 1024 floats into s_X (Row-Major layout).
        // 256 threads load 4 floats each.
        #pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            int r = i / 32;
            int c = i % 32;
            int g_n = cta_n_base + r;
            int g_k = k_base + c;
            s_X[r][c] = (g_n < N && g_k < K) ? __float2half(dX[g_n * K + g_k]) : __float2half(0.0f);
        }

        // 2. Cooperative load W [128 x 32] = 128 Q4_0 blocks into s_W (Column-Major layout: s_W[m][k]).
        // 256 threads: 2 threads per row (128 rows). Each thread loads 16 elements.
        int r_w = tid / 2;    // m row 0..127
        int sub_k = tid % 2;  // sub_k 0..1 (16 elements: sub_k*16 .. sub_k*16+15)
        int g_m = cta_m_base + r_w;
        int blk_idx = k_tile;

        if (g_m < M && k_base < K) {
            const BlockQ4_0 *blk = (const BlockQ4_0 *)dW + (long)g_m * nb + blk_idx;
            half d = blk->d;
            int c_start = sub_k * 16;
            int is_high = sub_k;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int q_byte = blk->qs[j];
                int nib = is_high ? ((q_byte >> 4) & 0xF) : (q_byte & 0xF);
                s_W[r_w][c_start + j] = __hmul(__int2half_rn(nib - 8), d);
            }
        } else {
            int c_start = sub_k * 16;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                s_W[r_w][c_start + j] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Accumulate over 2 sub-tiles along K (16 elements each)
        #pragma unroll
        for (int k_sub = 0; k_sub < 2; k_sub++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag0;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag1;

            wmma::load_matrix_sync(a_frag, (half*)&s_X[warp_n * 16][k_sub * 16], 32);
            wmma::load_matrix_sync(b_frag0, (half*)&s_W[warp_m * 32][k_sub * 16], 32);
            wmma::load_matrix_sync(b_frag1, (half*)&s_W[warp_m * 32 + 16][k_sub * 16], 32);

            wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
            wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        }
        __syncthreads();
    }

    // Store output accumulators to DRAM Y [N x M]
    if (my_n_base + 15 < N) {
        if (my_m_base + 15 < M) {
            wmma::store_matrix_sync(dY + my_n_base * M + my_m_base, c_frag0, M, wmma::mem_row_major);
        }
        if (my_m_base + 31 < M) {
            wmma::store_matrix_sync(dY + my_n_base * M + my_m_base + 16, c_frag1, M, wmma::mem_row_major);
        }
    } else {
        __align__(16) __shared__ float s_C[32][128];
        wmma::store_matrix_sync((float*)&s_C[warp_n * 16][warp_m * 32], c_frag0, 128, wmma::mem_row_major);
        wmma::store_matrix_sync((float*)&s_C[warp_n * 16][warp_m * 32 + 16], c_frag1, 128, wmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (int i = lane; i < 512; i += 32) {
            int r = i / 32;
            int c = i % 32;
            int g_n = my_n_base + r;
            int g_m = my_m_base + c;
            if (g_n < N && g_m < M) {
                dY[g_n * M + g_m] = s_C[warp_n * 16 + r][warp_m * 32 + c];
            }
        }
    }
}

// ---------------- host launchers ----------------
extern "C" {

static void gemv_dims(int M, dim3 *grid, dim3 *block) {
    block->x = 32; block->y = 16; block->z = 1;
    grid->x = (M + block->y - 1) / block->y; grid->y = 1; grid->z = 1;
}

// V2 kernels compute TWO rows per warp -> halve the grid accordingly
// (used by the layer GEMV launchers only; logits stays one-row-per-warp).
static void gemv_dims2(int M, dim3 *grid, dim3 *block) {
    block->x = 32; block->y = 16; block->z = 1;
    grid->x = (M + block->y * 2 - 1) / (block->y * 2); grid->y = 1; grid->z = 1;
}

/* M9.5: WMMA tensor-core launcher for q4_0 GEMV. Gated behind TT_USE_WMMA=1
 * (default off: V2 scalar fp16 is faster on Ampere consumer for single-token
 * decode because m16n16k16 wastes 15/16 of the N dim). See k_gemv_wmma_q4_0
 * kernel comment for the math. Returns 0 on success, -50 on bad dims. */
int tt_gemv_wmma_q4_0(const void *dW, const float *dx, float *dy, int M, int K,
                      cudaStream_t stream) {
    if (M <= 0 || M > (1 << 22) || K <= 0 || K % 16 != 0) return -50;
    dim3 g((M + 15) / 16), b(32);
    int shmem = K * 2;                          // sx as fp16
    k_gemv_wmma_q4_0<<<g, b, shmem, stream>>>(
        (const BlockQ4_0 *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

/* stream-aware launchers: keeps the whole step on one non-blocking stream */
int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K,
                 cudaStream_t stream) {
    dim3 g, b; gemv_dims2(M, &g, &b);
    k_gemv_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

/* M9.5: V2 layer GEMV for q8_0 — used by the qwen3-0.6b-q8_0 decode path.
 * Same 2-rows-per-warp shape as tt_gemv_q4_0; requires K/32 (nb) even. */
int tt_gemv_q8_0(const void *dW, const float *dx, float *dy, int M, int K,
                 cudaStream_t stream) {
    dim3 g, b; gemv_dims2(M, &g, &b);
    k_gemv_q8_0<<<g, b, 0, stream>>>((const BlockQ8_0 *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

int tt_swiglu_q4_0(const void *dGate, const void *dUp, const float *dx,
                   float *dh, int M, int K, cudaStream_t stream) {
    dim3 g, b; gemv_dims2(M, &g, &b);
    k_fused_swiglu_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dGate, (const BlockQ4_0 *)dUp,
                                             dx, dh, M, K, 0);
    return (int)cudaGetLastError();
}

/* M7 task 3: activation-selecting fused q4_0 FFN kernel.
 * act: 0 = SiLU (SwiGLU), 1 = GELU-tanh (GeGLU, gemma families). */
int tt_ffn_q4_0(const void *dGate, const void *dUp, const float *dx,
                float *dh, int M, int K, int act, cudaStream_t stream) {
    dim3 g, b; gemv_dims2(M, &g, &b);
    k_fused_swiglu_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dGate, (const BlockQ4_0 *)dUp,
                                             dx, dh, M, K, act);
    return (int)cudaGetLastError();
}

int tt_logits_q4_0(const void *dW, const float *dx, float *dlogits,
                   int vocab, int K, cudaStream_t stream) {
    dim3 g, b; gemv_dims(vocab, &g, &b);
    k_logits_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
    return (int)cudaGetLastError();
}

/* M9.5: V2 LM head for q4_0 (2-rows-per-warp, uint32 word streaming).
 * Closes the qwen2.5-q4_0 0.72x gap by vectorizing the 1.7 ms scalar
 * logits path. Requires K/32 even; otherwise fall back to scalar via
 * tt_logits_dispatch. Caller pads vocab to even. */
int tt_logits_q4_0_v2(const void *dW, const float *dx, float *dlogits,
                      int vocab, int K, cudaStream_t stream) {
    dim3 g, b; gemv_dims2(vocab, &g, &b);
    k_logits_q4_0_v2<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
    return (int)cudaGetLastError();
}

/* M9.5+ V4 launcher: 4-rows-per-warp q4_0 LM head. Caller (dispatcher)
 * must already have verified (K%32==0) and (vocab%4==0). Same nb-even
 * contract as tt_logits_q4_0_v2. Returns the cuda error code (0 on ok). */
int tt_logits_q4_0_v4(const void *dW, const float *dx, float *dlogits,
                      int vocab, int K, cudaStream_t stream) {
    /* 4 rows per warp -> blockDim.y = 8 warps; grid covers vocab/4/8 tiles. */
    dim3 g, b; b.x = 32; b.y = 8; b.z = 1;
    g.x = (vocab + b.y * 4 - 1) / (b.y * 4); g.y = 1; g.z = 1;
    k_logits_q4_0_v4<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
    return (int)cudaGetLastError();
}

int tt_logits_q8_0(const void *dW, const float *dx, float *dlogits,
                   int vocab, int K, cudaStream_t stream) {
    if (!dW || !dx || !dlogits) return -1;
    dim3 b(32, 1);
    dim3 g((vocab + b.y - 1) / b.y, 1, 1);
    k_logits_q8_0<<<g, b, 0, stream>>>((const BlockQ8_0 *)dW, dx, dlogits, vocab, K);
    return (int)cudaGetLastError();
}

// 4-rows-per-warp Q8_0 LM head: logits [vocab] = X [K] * W^T [vocab, K]
// W is Q8_0 quantized. Evaluates 4 vocab rows per warp in parallel.
int tt_logits_q8_0_v4(const void *dW, const float *dx, float *dlogits,
                      int vocab, int K, cudaStream_t stream) {
    if (!dW || !dx || !dlogits) return -1;
    dim3 b4(32, 1);
    dim3 g4((vocab + b4.y * 4 - 1) / (b4.y * 4), 1, 1);
    k_logits_q8_0_v4<<<g4, b4, 0, stream>>>(
        (const BlockQ8_0 *)dW, dx, dlogits, vocab, K);
    return (int)cudaGetLastError();
}

/* M9.5+ V4 launcher: 4-rows-per-warp q4_0 layer GEMV. Caller (dispatcher)
 * must have verified (K%32==0) and (M%4==0). Same nb-even contract. */
int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy,
                    int M, int K, cudaStream_t stream) {
    dim3 g, b; b.x = 32; b.y = 8; b.z = 1;
    g.x = (M + b.y * 4 - 1) / (b.y * 4); g.y = 1; g.z = 1;
    k_gemv_q4_0_v4<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

/* M9.5+ V4 dispatch helper for q4_0 layer GEMVs. Returns 0 on success
 * (V4 or V2 selected, row coverage complete), 1 if the call fell
 * through to the scalar fallback (caller must then dispatch to
 * tt_gemv_typed or a scalar kernel). Bit-exact against the scalar
 * path; no precision delta vs V2 alone (V4 == V2 == scalar up to
 * the FMA order, which is identical in V2 and V4 so outputs match
 * bit-exact). M must be a multiple of 4 for V4 to be eligible; we
 * pad to 4 by passing padded_M = (M+3)&~3 internally and writing
 * y[i] for i < M only.
 *
 * M9.5+ conditional dispatch: route to V4 only when M >= 128. Microbench
 * (tools/micro_v4.cu, K=896) shows V4 ≈ V2 below M=128 (both at the
 * ~0.005 ms noise floor) and V4 wins 1.14-1.49x for M >= 128. M=1 cases
 * (Q/K/V projections, head dim projections in small models) now fall
 * through to V2, which is at worst equal to V4 and at best modestly
 * faster. Threshold matches the microbench data; the q4_0 LM head
 * (M=151936) and FFN shapes (M=4864) remain on V4. */
int tt_gemv_q4_0_dispatch(const void *dW, const float *dx, float *dy,
                           int M, int K, cudaStream_t stream) {
    const int nb = K / 32;
    if ((K & 31) == 0 && (nb & 1) == 0 && (M & 3) == 0 && M >= 128) {
        int rc = tt_gemv_q4_0_v4(dW, dx, dy, M, K, stream);
        if (rc == 0) return 0;
        /* fall through to V2 on launch failure */
    }
    dim3 g, b; gemv_dims2(M, &g, &b);
    k_gemv_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

/* M10+ Batched-4 GEMV launchers (q4_0 + q8_0). 4 candidate x vectors
 * are processed in a single launch against the same weight matrix,
 * producing 4*Y[M] outputs. Inputs:
 *   dW      - quantized weight matrix (q4_0 or q8_0)
 *   dX_4xK  - device buffer of 4 candidate x vectors, shape [4, K],
 *             row-major (X[c*K + k])
 *   dY_4xM  - device buffer of 4 candidate outputs, shape [4, M],
 *             row-major (Y[c*M + m])
 *   M, K    - output rows and inner dim
 *
 * Constraints (caller must check; otherwise fall back to 4 sequential
 * single calls):
 *   - K % 32 == 0
 *   - nb = K/32 even (uint32 streaming + __byte_perm merge contract)
 *   - M multiple of 4 (4 rows per warp)
 *   - non-NULL dW/dX/dY
 *
 * Returns 0 on success, -1 if constraints are violated. Caller pads
 * M internally to a multiple of 4 before calling if needed.
 *
 * Shmem = 4 * K * 4 bytes. For K=896 -> 14 KB (K=4864 -> 76 KB; opt
 * in to >48KB dynamic shmem on Ampere via cudaFuncSetAttribute).
 */
int tt_gemv_q4_0_batch4(const void *dW, const float *dX_4xK, float *dY_4xM,
                        int M, int K, cudaStream_t stream) {
    if (!dW || !dX_4xK || !dY_4xM) return -1;
    const int nb = K / 32;
    if ((K & 31) != 0 || (nb & 1) != 0 || (M & 3) != 0 || M <= 0) return -1;
    dim3 g, b; b.x = 32; b.y = 16; b.z = 1;
    g.x = (M + b.y * 4 - 1) / (b.y * 4); g.y = 1; g.z = 1;
    int shmem = 4 * K * 4;
    if (shmem > 48 * 1024) {
        cudaFuncSetAttribute(k_gemv_q4_0_batch4,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    k_gemv_q4_0_batch4<<<g, b, shmem, stream>>>(
        (const BlockQ4_0 *)dW, dX_4xK, dY_4xM, M, K);
    return (int)cudaGetLastError();
}

int tt_gemv_q8_0_batch4(const void *dW, const float *dX_4xK, float *dY_4xM,
                        int M, int K, cudaStream_t stream) {
    if (!dW || !dX_4xK || !dY_4xM) return -1;
    const int nb = K / 32;
    if ((K & 31) != 0 || (nb & 1) != 0 || (M & 3) != 0 || M <= 0) return -1;
    dim3 g, b; b.x = 32; b.y = 16; b.z = 1;
    g.x = (M + b.y * 4 - 1) / (b.y * 4); g.y = 1; g.z = 1;
    int shmem = 4 * K * 4;
    if (shmem > 48 * 1024) {
        cudaFuncSetAttribute(k_gemv_q8_0_batch4,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    k_gemv_q8_0_batch4<<<g, b, shmem, stream>>>(
        (const BlockQ8_0 *)dW, dX_4xK, dY_4xM, M, K);
    return (int)cudaGetLastError();
}

int tt_embed_q4_0(const void *dW, int tok, float *dx, int dim, cudaStream_t stream) {
    const int threads = dim / 32;
    k_embed_q4_0<<<(threads + 255) / 256, 256, 0, stream>>>((const BlockQ4_0 *)dW, tok, dx, dim);
    return (int)cudaGetLastError();
}

/* dtype-dispatching logits projection. M7: takes the GGML type code
 * directly (q8_0 keeps its tuned one-warp-per-row grid; q4_0 the default;
 * every other type delegates to kernels/gemv_typed.cu). */
int tt_logits_dispatch(const void *dW, int dtype, const float *dx,
                       float *dlogits, int vocab, int K, cudaStream_t stream) {
    if (dtype != 2 /*q4_0*/ && dtype != 8 /*q8_0*/) {
        extern int tt_logits_typed(const void *, int, const float *,
                                   float *, int, int, cudaStream_t);
        return tt_logits_typed(dW, dtype, dx, dlogits, vocab, K, stream);
    }
    dim3 g, b; gemv_dims(vocab, &g, &b);
    if (dtype == 8) {
        if ((vocab & 3) == 0 && (K & 31) == 0) {
            return tt_logits_q8_0_v4(dW, dx, dlogits, vocab, K, stream);
        }
        /* M6.3b: one warp per block for the head (y sweep 16->8->4->2->1
         * monotone win). Grid MUST be recomputed for the smaller block or
         * rows >= grid.x*b.y never get written (stale logits -> degenerate
         * sampling). The original sweep shipped without this and silently
         * zeroed every token id >= vocab/16. */
        b.y = 1;
        g.x = (vocab + b.y - 1) / b.y;
        k_logits_q8_0<<<g, b, 0, stream>>>((const BlockQ8_0 *)dW, dx, dlogits, vocab, K);
    } else {
        /* M9.5+: route q4_0 to V4 (4 rows/warp) when eligible, else V2,
         * else scalar. V4 wins 1.7x on the LM head shape and 1.5-1.6x
         * on FFN shapes vs V2 in the microbench (bit-exact).
         *   V4 needs: K%32==0, nb=K/32 even, vocab%4==0 (vocab is always
         *     even in our heads; check 4 explicitly so any future odd-vocab
         *     model falls through cleanly to V2).
         *   V2 needs: K%32==0, nb even (same uint32 streaming contract).
         *   scalar k_logits_q4_0 is the fallback for odd-nb or odd-K. */
        if ((K & 31) == 0 && ((K >> 5) & 1) == 0 && (vocab & 3) == 0) {
            b.x = 32; b.y = 8; b.z = 1;
            g.x = (vocab + b.y * 4 - 1) / (b.y * 4); g.y = 1; g.z = 1;
            k_logits_q4_0_v4<<<g, b, 0, stream>>>(
                (const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
        } else if (((K >> 5) & 1) == 0 && (K & 31) == 0) {
            /* V2 path: grid.x * blockDim.y * 2 >= vocab. Re-derive via
             * gemv_dims2 (same as tt_gemv_q4_0) so we don't waste warps. */
            gemv_dims2(vocab, &g, &b);
            k_logits_q4_0_v2<<<g, b, 0, stream>>>(
                (const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
        } else {
            k_logits_q4_0<<<g, b, 0, stream>>>(
                (const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
        }
    }
    return (int)cudaGetLastError();
}

int tt_gemm_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM,
                         int M, int K, int N, cudaStream_t stream)
{
    dim3 grid((M + 63) / 64, (N + 31) / 32);
    dim3 block(16, 16);
    k_gemm_q4_0_prefill<<<grid, block, 0, stream>>>(dW, dX_NxK, dY_NxM, M, K, N);
    return (int)cudaGetLastError();
}

int tt_gemm_wmma_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM,
                              int M, int K, int N, cudaStream_t stream)
{
    dim3 grid((M + 127) / 128, (N + 31) / 32);
    dim3 block(256);
    k_gemm_wmma_q4_0_prefill<<<grid, block, 0, stream>>>(dW, dX_NxK, dY_NxM, M, K, N);
    return (int)cudaGetLastError();
}

} /* extern "C" */
