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
__global__ void k_gemv_q4_0(const BlockQ4_0 *__restrict__ W,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    const int nb = K / 32;                    // q4_0 blocks per row
    const uint32_t *roww = (const uint32_t *)((const char *)W + (long)row * nb * 18);
    float sum = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;                 // word holding blk.d
        const unsigned short d16 = (unsigned short)
            (((18 * b) & 2) ? (roww[wsc] >> 16) : (roww[wsc] & 0xFFFFu));
        const float d = __half2float(__ushort_as_half(d16));
        const int a0 = (18 * b + 2) >> 2;              // first qs word
        const int sh  = (18 * b + 2) & 2;              // 2 => misaligned merge
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t lo = roww[a0 + k];
            const uint32_t vv = sh ? __byte_perm(lo, roww[a0 + k + 1], 0x5432) : lo;
            const float4 xa = x4[k];       // x[4k .. 4k+3]   <- low nibbles
            const float4 xb = x4[k + 4];   // x[16+4k .. +3]  <- high nibbles
            sum += (float)((vv         & 0xFu) - 8u) * d * xa.x;
            sum += (float)(((vv >>  4) & 0xFu) - 8u) * d * xa.y;
            sum += (float)(((vv >>  8) & 0xFu) - 8u) * d * xa.z;
            sum += (float)(((vv >> 12) & 0xFu) - 8u) * d * xa.w;
            sum += (float)(((vv >> 16) & 0xFu) - 8u) * d * xb.x;
            sum += (float)(((vv >> 20) & 0xFu) - 8u) * d * xb.y;
            sum += (float)(((vv >> 24) & 0xFu) - 8u) * d * xb.z;
            sum += (float)( (vv >> 28)         - 8u) * d * xb.w;
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Fused MLP up-projection: h[m] = silu(W_gate[m,:] @ x) * (W_up[m,:] @ x).
// V2: same vectorized streaming as k_gemv_q4_0 above (same nb-even contract).
__global__ void k_fused_swiglu_q4_0(const BlockQ4_0 *__restrict__ W_gate,
                                    const BlockQ4_0 *__restrict__ W_up,
                                    const float *__restrict__ x,
                                    float *__restrict__ out,
                                    int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *gw = (const uint32_t *)((const char *)W_gate + (long)row * nb * 18);
    const uint32_t *uw = (const uint32_t *)((const char *)W_up   + (long)row * nb * 18);
    float sg = 0.0f, su = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const unsigned short dg16 = (unsigned short)
            (((18 * b) & 2) ? (gw[wsc] >> 16) : (gw[wsc] & 0xFFFFu));
        const unsigned short du16 = (unsigned short)
            (((18 * b) & 2) ? (uw[wsc] >> 16) : (uw[wsc] & 0xFFFFu));
        const float dg = __half2float(__ushort_as_half(dg16));
        const float du = __half2float(__ushort_as_half(du16));
        const int g0 = (18 * b + 2) >> 2;
        const int u0 = g0;                           // same layout both rows
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t glo = gw[g0 + k];
            const uint32_t ulo = uw[u0 + k];
            const uint32_t gvv = sh ? __byte_perm(glo, gw[g0 + k + 1], 0x5432) : glo;
            const uint32_t uvv = sh ? __byte_perm(ulo, uw[u0 + k + 1], 0x5432) : ulo;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            sg += (float)((gvv         & 0xFu) - 8u) * dg * xa.x;
            sg += (float)(((gvv >>  4) & 0xFu) - 8u) * dg * xa.y;
            sg += (float)(((gvv >>  8) & 0xFu) - 8u) * dg * xa.z;
            sg += (float)(((gvv >> 12) & 0xFu) - 8u) * dg * xa.w;
            sg += (float)(((gvv >> 16) & 0xFu) - 8u) * dg * xb.x;
            sg += (float)(((gvv >> 20) & 0xFu) - 8u) * dg * xb.y;
            sg += (float)(((gvv >> 24) & 0xFu) - 8u) * dg * xb.z;
            sg += (float)( (gvv >> 28)         - 8u) * dg * xb.w;
            su += (float)((uvv         & 0xFu) - 8u) * du * xa.x;
            su += (float)(((uvv >>  4) & 0xFu) - 8u) * du * xa.y;
            su += (float)(((uvv >>  8) & 0xFu) - 8u) * du * xa.z;
            su += (float)(((uvv >> 12) & 0xFu) - 8u) * du * xa.w;
            su += (float)(((uvv >> 16) & 0xFu) - 8u) * du * xb.x;
            su += (float)(((uvv >> 20) & 0xFu) - 8u) * du * xb.y;
            su += (float)(((uvv >> 24) & 0xFu) - 8u) * du * xb.z;
            su += (float)( (uvv >> 28)         - 8u) * du * xb.w;
        }
    }
    sg = warp_reduce_sum(sg);
    su = warp_reduce_sum(su);
    if (lane == 0) {
        const float g = sg;
        out[row] = (g / (1.0f + expf(-g))) * su;
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

/* stream-aware launchers: keeps the whole step on one non-blocking stream */
int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K,
                 cudaStream_t stream) {
    dim3 g, b; gemv_dims(M, &g, &b);
    k_gemv_q4_0<<<g, b, 0, stream>>>((const BlockQ4_0 *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

int tt_swiglu_q4_0(const void *dGate, const void *dUp, const float *dx,
                   float *dh, int M, int K, cudaStream_t stream) {
    dim3 g, b; gemv_dims(M, &g, &b);
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
