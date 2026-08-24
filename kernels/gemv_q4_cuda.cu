// q4_0 GEMV primitives for the Qwen2 engine (M6 correctness rewrite).
//
// GGML q4_0 block layout: 32 values per block = fp16 scale d + 16 bytes qs.
// Dequant: x[j]   = ((qs[j]   & 0x0F) - 8) * d   for j in [0,15]
//          x[j+16]= ((qs[j] >> 4)    - 8) * d   for j in [0,15]
// All kernels here use that pairing (the old k_gemv_q4_0 paired nibble-low
// with x[2i], which is NOT the GGML layout and produced wrong dot products).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
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
            const float4 xa = x4[k];       // x[4k .. 4k+3]   <- low nibbles
            const float4 xb = x4[k + 4];   // x[16+4k .. +3]  <- high nibbles
            s0 += (float)((va         & 0xFu) - 8u) * da * xa.x;
            s0 += (float)(((va >>  4) & 0xFu) - 8u) * da * xa.y;
            s0 += (float)(((va >>  8) & 0xFu) - 8u) * da * xa.z;
            s0 += (float)(((va >> 12) & 0xFu) - 8u) * da * xa.w;
            s0 += (float)(((va >> 16) & 0xFu) - 8u) * da * xb.x;
            s0 += (float)(((va >> 20) & 0xFu) - 8u) * da * xb.y;
            s0 += (float)(((va >> 24) & 0xFu) - 8u) * da * xb.z;
            s0 += (float)( (va >> 28)         - 8u) * da * xb.w;
            s1 += (float)((vb         & 0xFu) - 8u) * db * xa.x;
            s1 += (float)(((vb >>  4) & 0xFu) - 8u) * db * xa.y;
            s1 += (float)(((vb >>  8) & 0xFu) - 8u) * db * xa.z;
            s1 += (float)(((vb >> 12) & 0xFu) - 8u) * db * xa.w;
            s1 += (float)(((vb >> 16) & 0xFu) - 8u) * db * xb.x;
            s1 += (float)(((vb >> 20) & 0xFu) - 8u) * db * xb.y;
            s1 += (float)(((vb >> 24) & 0xFu) - 8u) * db * xb.z;
            s1 += (float)( (vb >> 28)         - 8u) * db * xb.w;
        }
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// Fused MLP up-projection: h[m] = silu(W_gate[m,:] @ x) * (W_up[m,:] @ x).
// V2: two rows per warp (same sharing scheme as k_gemv_q4_0; nb-even contract).
__global__ void k_fused_swiglu_q4_0(const BlockQ4_0 *__restrict__ W_gate,
                                    const BlockQ4_0 *__restrict__ W_up,
                                    const float *__restrict__ x,
                                    float *__restrict__ out,
                                    int M, int K) {
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
            sg0 += (float)((gva         & 0xFu) - 8u) * dga_ * xa.x;
            sg0 += (float)(((gva >>  4) & 0xFu) - 8u) * dga_ * xa.y;
            sg0 += (float)(((gva >>  8) & 0xFu) - 8u) * dga_ * xa.z;
            sg0 += (float)(((gva >> 12) & 0xFu) - 8u) * dga_ * xa.w;
            sg0 += (float)(((gva >> 16) & 0xFu) - 8u) * dga_ * xb.x;
            sg0 += (float)(((gva >> 20) & 0xFu) - 8u) * dga_ * xb.y;
            sg0 += (float)(((gva >> 24) & 0xFu) - 8u) * dga_ * xb.z;
            sg0 += (float)( (gva >> 28)         - 8u) * dga_ * xb.w;
            su0 += (float)((uva         & 0xFu) - 8u) * dua_ * xa.x;
            su0 += (float)(((uva >>  4) & 0xFu) - 8u) * dua_ * xa.y;
            su0 += (float)(((uva >>  8) & 0xFu) - 8u) * dua_ * xa.z;
            su0 += (float)(((uva >> 12) & 0xFu) - 8u) * dua_ * xa.w;
            su0 += (float)(((uva >> 16) & 0xFu) - 8u) * dua_ * xb.x;
            su0 += (float)(((uva >> 20) & 0xFu) - 8u) * dua_ * xb.y;
            su0 += (float)(((uva >> 24) & 0xFu) - 8u) * dua_ * xb.z;
            su0 += (float)( (uva >> 28)         - 8u) * dua_ * xb.w;
            sg1 += (float)((gvb         & 0xFu) - 8u) * dgb_ * xa.x;
            sg1 += (float)(((gvb >>  4) & 0xFu) - 8u) * dgb_ * xa.y;
            sg1 += (float)(((gvb >>  8) & 0xFu) - 8u) * dgb_ * xa.z;
            sg1 += (float)(((gvb >> 12) & 0xFu) - 8u) * dgb_ * xa.w;
            sg1 += (float)(((gvb >> 16) & 0xFu) - 8u) * dgb_ * xb.x;
            sg1 += (float)(((gvb >> 20) & 0xFu) - 8u) * dgb_ * xb.y;
            sg1 += (float)(((gvb >> 24) & 0xFu) - 8u) * dgb_ * xb.z;
            sg1 += (float)( (gvb >> 28)         - 8u) * dgb_ * xb.w;
            su1 += (float)((uvb         & 0xFu) - 8u) * dub_ * xa.x;
            su1 += (float)(((uvb >>  4) & 0xFu) - 8u) * dub_ * xa.y;
            su1 += (float)(((uvb >>  8) & 0xFu) - 8u) * dub_ * xa.z;
            su1 += (float)(((uvb >> 12) & 0xFu) - 8u) * dub_ * xa.w;
            su1 += (float)(((uvb >> 16) & 0xFu) - 8u) * dub_ * xb.x;
            su1 += (float)(((uvb >> 20) & 0xFu) - 8u) * dub_ * xb.y;
            su1 += (float)(((uvb >> 24) & 0xFu) - 8u) * dub_ * xb.z;
            su1 += (float)( (uvb >> 28)         - 8u) * dub_ * xb.w;
        }
    }
    sg0 = warp_reduce_sum(sg0);
    su0 = warp_reduce_sum(su0);
    sg1 = warp_reduce_sum(sg1);
    su1 = warp_reduce_sum(su1);
    if (lane == 0) {
        const float ga = sg0;
        out[row0] = (ga / (1.0f + expf(-ga))) * su0;
        if (row1 < M) {
            const float gb = sg1;
            out[row1] = (gb / (1.0f + expf(-gb))) * su1;
        }
    }
}

// LM head / tied-embedding projection over vocab rows (same math as GEMV).
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

/* stream-aware launchers: keeps the whole step on one non-blocking stream */
int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K,
                 cudaStream_t stream) {
    dim3 g, b; gemv_dims2(M, &g, &b);
    k_gemv_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

int tt_swiglu_q4_0(const void *dGate, const void *dUp, const float *dx,
                   float *dh, int M, int K, cudaStream_t stream) {
    dim3 g, b; gemv_dims2(M, &g, &b);
    k_fused_swiglu_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dGate, (const BlockQ4_0 *)dUp,
                                             dx, dh, M, K);
    return (int)cudaGetLastError();
}

int tt_logits_q4_0(const void *dW, const float *dx, float *dlogits,
                   int vocab, int K, cudaStream_t stream) {
    dim3 g, b; gemv_dims(vocab, &g, &b);
    k_logits_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
    return (int)cudaGetLastError();
}

int tt_embed_q4_0(const void *dW, int tok, float *dx, int dim, cudaStream_t stream) {
    const int threads = dim / 32;
    k_embed_q4_0<<<(threads + 255) / 256, 256, 0, stream>>>((const BlockQ4_0 *)dW, tok, dx, dim);
    return (int)cudaGetLastError();
}

/* dtype-dispatching logits projection: is_q8 selects the q8_0 kernel */
int tt_logits_dispatch(const void *dW, int is_q8, const float *dx,
                       float *dlogits, int vocab, int K, cudaStream_t stream) {
    dim3 g, b; gemv_dims(vocab, &g, &b);
    if (is_q8)
        k_logits_q8_0<<<g, b, 0, stream>>>((const BlockQ8_0 *)dW, dx, dlogits, vocab, K);
    else
        k_logits_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dlogits, vocab, K);
    return (int)cudaGetLastError();
}

} /* extern "C" */
