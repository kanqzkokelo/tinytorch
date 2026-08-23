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
// blockDim=(32, warps_per_block); gridDim.x = ceil(M / warps_per_block).
__global__ void k_gemv_q4_0(const BlockQ4_0 *__restrict__ W,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    const int nb = K / 32;                    // q4_0 blocks per row
    const BlockQ4_0 *rowW = W + (long)row * nb;
    float sum = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        BlockQ4_0 blk = rowW[b];
        const float d = __half2float(blk.d);
        const float *xb = x + b * 32;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int q0 = (blk.qs[i] & 0x0F) - 8;
            const int q1 = (blk.qs[i] >> 4) - 8;
            sum += ((float)q0 * d) * xb[i] + ((float)q1 * d) * xb[i + 16];
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Fused MLP up-projection: h[m] = silu(W_gate[m,:] @ x) * (W_up[m,:] @ x).
__global__ void k_fused_swiglu_q4_0(const BlockQ4_0 *__restrict__ W_gate,
                                    const BlockQ4_0 *__restrict__ W_up,
                                    const float *__restrict__ x,
                                    float *__restrict__ out,
                                    int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    const int nb = K / 32;
    const BlockQ4_0 *gRow = W_gate + (long)row * nb;
    const BlockQ4_0 *uRow = W_up + (long)row * nb;
    float sg = 0.0f, su = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        BlockQ4_0 bg = gRow[b];
        BlockQ4_0 bu = uRow[b];
        const float dg = __half2float(bg.d);
        const float du = __half2float(bu.d);
        const float *xb = x + b * 32;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float x0 = xb[i], x1 = xb[i + 16];
            sg += (((bg.qs[i] & 0x0F) - 8) * dg) * x0 + (((bg.qs[i] >> 4) - 8) * dg) * x1;
            su += (((bu.qs[i] & 0x0F) - 8) * du) * x0 + (((bu.qs[i] >> 4) - 8) * du) * x1;
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
__global__ void k_logits_q8_0(const BlockQ8_0 *__restrict__ W,
                              const float *__restrict__ x,
                              float *__restrict__ logits,
                              int vocab, int K) {
    const int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= vocab) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const BlockQ8_0 *rowW = W + (long)v * nb;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        BlockQ8_0 blk = rowW[b];
        const float d = __half2float(blk.d);
        const float *xb = x + b * 32;
#pragma unroll
        for (int i = 0; i < 32; i++)
            sum += ((float)blk.qs[i]) * d * xb[i];
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
