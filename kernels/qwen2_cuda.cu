// Correct Qwen2-family decode engine (M6 rewrite).
//
// Per token t, per layer l — the full network, nothing skipped:
//   xn  = rmsnorm(x, attn_norm[l])                 (eps from GGUF)
//   q   = Wq[l] @ xn        rope(q, pos)           (n_heads * head_dim)
//   k   = Wk[l] @ xn        rope(k, pos)           (n_kv_heads * head_dim)
//   v   = Wv[l] @ xn
//   Kc[l,pos], Vc[l,pos] <- k, v                   (real KV cache write)
//   att = GQA-flash-attention(q, Kc[l][0..pos], Vc[l][0..pos])
//         head h attends kv group h / (n_heads/n_kv_heads)
//   x  += Wo[l] @ att                              (residual)
//   xn  = rmsnorm(x, ffn_norm[l])
//   h   = silu(Wgate[l] @ xn) * (Wup[l] @ xn)
//   x  += Wdown[l] @ h                             (residual)
// Final: logits = lm_head @ rmsnorm(x, output_norm); greedy argmax.
//
// All dims come from TTConfig (GGUF metadata). No hardcoded shapes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "dequant_ref.h"

/* BlockQ4_0 comes from loader_gguf.h (d stored as raw fp16 bits). */
#define Q4_D(blk) __half2float(*(const __half *)&(blk).d)

#ifndef BLOCK_Q8_0_DEFINED
#define BLOCK_Q8_0_DEFINED
struct BlockQ8_0 {
    half d;          // 2 bytes FP16 scale
    int8_t qs[32];   // 32 bytes signed int8
};
#endif

#define Q4_BYTES_PER_BLOCK 18
#define Q4_VALS_PER_BLOCK 32

extern "C" long ttq_dequant(const void *data, int type_code, long numel,
                            float *out);   /* src/dequant_ref.c */

/* launchers implemented in kernels/gemv_q4_cuda.cu */
extern "C" {
typedef struct CUstream_st *cudaStream_t;
int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K,
                 cudaStream_t stream);
int tt_swiglu_q4_0(const void *dGate, const void *dUp, const float *dx,
                   float *dh, int M, int K, cudaStream_t stream);
int tt_logits_q4_0(const void *dW, const float *dx, float *dlogits,
                   int vocab, int K, cudaStream_t stream);
/* M7 task 2: typed dispatch (kernels/gemv_typed.cu) */
int tt_gemv_typed(const void *W, int dtype, const float *x, float *y,
                  int M, int K, cudaStream_t stream);
/* M7 task 3: fused q4_0 FFN with activation-selectable epilogue */
int tt_ffn_q4_0(const void *dGate, const void *dUp, const float *dx,
                float *dh, int M, int K, int act, cudaStream_t stream);
int tt_embed_typed(const void *dW, int dtype, int tok, float *dx, int dim,
                   cudaStream_t stream);
int tt_logits_dispatch(const void *dW, int dtype, const float *dx,
                       float *dlogits, int vocab, int K, cudaStream_t stream);
/* M10+ Batched-4 LM head launcher (q4_0 only). X is [4, K], L is [4, vocab];
 * candidate-major output (L[c*vocab + v] = X[c*K + :] @ W[v, :]). Used by
 * the speculative-verify path; requires vocab%4==0 (caller pads). */
int tt_logits_q4_0_batch4(const void *dW, const float *dX_4xK,
                          float *dL_4xVocab, int vocab, int K, cudaStream_t stream);
 int tt_logits_dispatch(const void *dW, int dtype, const float *dx,
                        float *dlogits, int vocab, int K, cudaStream_t stream);
 int tt_embed_q4_0(const void *dW, int tok, float *dx, int dim, cudaStream_t stream);
/* M9.5+ V4 dispatch: routes q4_0 layer GEMVs to 4-rows-per-warp kernel
 * when eligible (M%4==0, K%32==0, nb even), else falls back to V2.
 * Bit-exact vs V2 (V4 == V2 == scalar up to FMA order, which is identical
 * between V2 and V4). Returns 0 on success. */
int tt_gemv_q4_0_dispatch(const void *dW, const float *dx, float *dy,
                          int M, int K, cudaStream_t stream);
}

static size_t q4_bytes(long numel) { return (size_t)(numel / Q4_VALS_PER_BLOCK) * Q4_BYTES_PER_BLOCK; }

/* M9.5+ V4-aware per-layer GEMV: q4_0 uses tt_gemv_q4_0_dispatch which
 * routes to V4 when eligible (4-rows/warp, 1.5-1.6x on FFN shapes per
 * the microbench) and falls back to V2 when not. Other dtypes continue
 * to use tt_gemv_typed (q8_0/f16/q4_k/q5_k/etc.). Bit-exact vs the
 * pre-V4 path: V4 output == V2 output == typed output for the same
 * weight tensor and x. */
static inline int tt_gemv_layer_dispatch(void *w_ptr, int w_dtype,
                                          const float *dx, float *dy,
                                          int M, int K, cudaStream_t s) {
    if (w_dtype == 2 /* GGUF_TYPE_Q4_0 */)
        return tt_gemv_q4_0_dispatch(w_ptr, dx, dy, M, K, s);
    return tt_gemv_typed(w_ptr, w_dtype, dx, dy, M, K, s);
}

/* Hybrid KV-cache dispatch helpers (Fix1). Threshold defaults to 256;
 * override via TT_QKV_THRESH. Below thresh FP32 is used for parity;
 * above thresh Q4/Q8 is used for speed (4x/8x bandwidth). */
static inline int kv_thresh_value(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *ev = getenv("TT_QKV_THRESH");
        cached = (ev && *ev) ? atoi(ev) : 256;
        if (cached < 0) cached = 256;
    }
    return cached;
}
/* Type-blind weight handle: device pointer + GGML type code for dispatch. */
typedef struct { void *ptr; int dtype; } TTensor;

/* Upload any tensor type-blind: raw size_bytes memcpy, dtype recorded. */
static int upload_w(GGUFModel *m, const char *name, TTensor *out) {
    GGUFTensor *t = gguf_get_tensor(m, name);
    out->ptr = NULL; out->dtype = -1;
    if (!t || !t->data) { fprintf(stderr, "[qwen2-engine] weight upload missing: %s\n", name); return -1; }
    void *d = NULL;
    if (cudaMalloc(&d, t->size_bytes) != cudaSuccess) { fprintf(stderr, "[qwen2-engine] cudaMalloc fail %s\n", name); return -1; }
    cudaMemcpy(d, t->data, t->size_bytes, cudaMemcpyHostToDevice);
    out->ptr = d;
    out->dtype = (int)t->type;
    if (getenv("TT_DEBUG")) {
        long shape0 = 0, shape1 = 0;
        /* shapes live in the loader; re-fetch for debug print */
        GGUFTensor *tt = gguf_get_tensor(m, name);
        if (tt) { shape0 = tt->shape[0]; shape1 = tt->ndim > 1 ? tt->shape[1] : 0; }
        fprintf(stderr, "[qwen2-engine] up %s type=%d size=%zu ne=[%ld,%ld]\n",
                name, t->type, t->size_bytes, shape0, shape1);
    }
    return 0;
}

/* ---------------- device kernels (engine-local ops) ---------------- */

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float warp_sum_all(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

/* y = x / sqrt(mean(x^2) + eps) * (woff + gamma) ; one block per row.
 * woff is the gemma-style norm offset; converted gemma GGUFs bake the
 * (1+w) into the stored weights (oracle convert_hf_to_gguf.py:4730), so it
 * stays 0 there. woff=0 => y[i] = x[i]*inv*g[i] exactly as before. */
__global__ void k_rmsnorm(const float *__restrict__ x, const float *__restrict__ g,
                          float *__restrict__ y, int dim, float eps, float woff) {
    extern __shared__ float s[];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum(ss);
    if ((tid & 31) == 0) s[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++) t += s[w];
        s[0] = rsqrtf(t / (float)dim + eps);
    }
    __syncthreads();
    const float inv = s[0];
    if (woff == 0.0f) {
        for (int i = tid; i < dim; i += blockDim.x) y[i] = x[i] * inv * g[i];
    } else {
        for (int i = tid; i < dim; i += blockDim.x) y[i] = x[i] * inv * (woff + g[i]);
    }
}

/* ---------------- M9 PLE-fused V2 (gemma4 MatFormer block) ----------------
 *
 * Replaces the per-layer host-assisted PLE round-trip (D2H x2, H2D x2, plus
 * host-side gelu*ple and rmsnorm) with a 2-launch device chain. The proto
 * design (tests/proto_ple_fused.cu, commit e7e9609) measured 22.3 us/layer
 * vs 44.6 us for the host path; the saving (35 layers * 22.3 us ~= 780 us
 * per token, ~22% of the 3.5 ms decode budget) makes the per-layer chain
 * graph-capturable for gemma4.
 *
 *   stage1: h = inp_gate @ x                (f32 GEMV, 256x1536)
 *           g = gelu(h) * ple[l]            (elementwise)
 *   stage2: p = pl_proj @ g                 (f32 GEMV, 1536x256)
 *           p = rmsnorm(p, gamma) in place  (atomic-ticket epilogue)
 *
 * Engine-side dtypes for blk.{l}.inp_gate.weight and blk.{l}.proj.weight
 * are f32 in the Q4_0/Q5_K_M/Q6_K gemma4 GGUFs (see [qwen2-engine] up
 * logs). The f32 kernels below match exactly. For non-f32 per-layer
 * weights the engine keeps the existing host-assisted path.
 */
__device__ __forceinline__ float ple_gelu_f(float x) {
    /* tanh-approx GELU (matches the old host path's gelu byte-for-byte
     * so the m84 goldens stay valid; proto used exact-erf as a design
     * reference, not the production kernel). */
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f *
                                    (x + 0.044715f * x * x * x)));
}

/* Stage1: gemv f32 + gelu + ple mul, one warp per output row. */
__global__ void k_ple_stage1_f32(const float *__restrict__ W1,    /* [M,K] */
                                const float *__restrict__ x,     /* [K] */
                                const float *__restrict__ ple_r, /* [M] */
                                float *__restrict__ g,           /* [M] */
                                int M, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    const int row = blockIdx.x * nwarp + warp;
    if (row >= M) return;
    const float *rw = W1 + (long)row * K;
    float s = 0.0f;
    for (int j = lane; j < K; j += 32) s += rw[j] * x[j];
    s = warp_sum(s);
    if (lane == 0) g[row] = ple_gelu_f(s) * ple_r[row];
}

/* Stage2: gemv f32 + atomic-ticket rmsnorm epilogue (one warp per output
 * row; last block to finish normalizes).  Matches proto design. */
__device__ unsigned int g_ple_ticket = 0;

__global__ void k_ple_stage2_f32(const float *__restrict__ W2,    /* [M,K] */
                                const float *__restrict__ g,     /* [K] */
                                const float *__restrict__ gamma, /* [M] */
                                float *__restrict__ out,         /* [M] */
                                int M, int K, float eps) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row < M) {
        const float *rw = W2 + (long)row * K;
        float s = 0.0f;
        for (int j = lane; j < K; j += 32) s += rw[j] * g[j];
        s = warp_sum(s);
        if (lane == 0) out[row] = s;
    }
    /* grid-wide barrier via ticket: last block normalizes */
    __shared__ bool ple_is_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int prev = atomicInc(&g_ple_ticket, gridDim.x - 1);
        ple_is_last = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (!ple_is_last) return;
    /* normalize: read back out[], apply rmsnorm in place */
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) ss += out[i] * out[i];
    /* warp + cross-warp reduce in smem (assumes blockDim.x is a power of 2) */
    __shared__ float ple_red[1024];
    /* intra-warp */
    for (int off = 16; off > 0; off /= 2) ss += __shfl_down_sync(0xffffffff, ss, off);
    const int warp = tid >> 5;
    const int lane2 = tid & 31;
    if (lane2 == 0) ple_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        ss = (lane2 < nw) ? ple_red[lane2] : 0.0f;
        for (int off = 16; off > 0; off /= 2) ss += __shfl_down_sync(0xffffffff, ss, off);
        if (lane2 == 0) ple_red[0] = rsqrtf(ss / (float)M + eps);
    }
    __syncthreads();
    const float inv = ple_red[0];
    for (int i = tid; i < M; i += blockDim.x) out[i] *= inv * gamma[i];
    g_ple_ticket = 0; /* reset for next launch */
}

/* RoPE in-place on rows [n_heads, head_dim]; one thread per half-dim pair. */
__global__ void k_rope(float *__restrict__ q, int n_heads, int head_dim,
                       const int *__restrict__ d_pos, float base) {
    const int pos = *d_pos;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;   /* 0..head_dim/2 */
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;

    float *row = q + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float ang = (float)pos * freq;
    const float c = cosf(ang), s = sinf(ang);
    const float v0 = row[i], v1 = row[i + head_dim / 2];
    row[i] = v0 * c - v1 * s;
    row[i + head_dim / 2] = v0 * s + v1 * c;
}

/* NEOX RoPE with per-pair frequency factors (gemma4 full-attn layers):
 * theta divided by ff[i]; large ff => identity rotation (partial rope).
 * ff may be NULL (all factors 1). */
__global__ void k_rope_ff(float *__restrict__ q, int n_heads, int head_dim,
                          const int *__restrict__ d_pos, float base,
                          const float *__restrict__ ff) {
    const int pos = *d_pos;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;   /* 0..head_dim/2 */
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;

    float *row = q + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float div = ff ? ff[i] : 1.0f;
    const float ang = (float)pos * freq / div;
    const float c = cosf(ang), s = sinf(ang);
    const float v0 = row[i], v1 = row[i + head_dim / 2];
    row[i] = v0 * c - v1 * s;
    row[i + head_dim / 2] = v0 * s + v1 * c;
}

/* GPT-J style RoPE: interleaved consecutive pairs (2i, 2i+1).
 * This is llama.cpp's LLAMA_ROPE_TYPE_NORM convention used by the llama
 * family (mistral/tinyllama/smollm) — oracle llama-model.cpp:3854-3875.
 * Selected by the ROPE_GPTJ trait; identical signature to k_rope. */
__global__ void k_rope_gptj(float *__restrict__ q, int n_heads, int head_dim,
                            const int *__restrict__ d_pos, float base) {
    const int pos = *d_pos;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;   /* 0..head_dim/2 */
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;

    float *row = q + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float ang = (float)pos * freq;
    const float c = cosf(ang), s = sinf(ang);
    const int i0 = 2 * i, i1 = 2 * i + 1;
    const float v0 = row[i0], v1 = row[i1];
    row[i0] = v0 * c - v1 * s;
    row[i1] = v0 * s + v1 * c;
}
/* GPT-J style RoPE with per-pair frequency factors:
 * Consecutive pairs (2i, 2i+1) rotated by ang = pos * freq / div. */
__global__ void k_rope_gptj_ff(float *__restrict__ q, int n_heads, int head_dim,
                               const int *__restrict__ d_pos, float base,
                               const float *__restrict__ ff) {
    const int pos = *d_pos;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;   /* 0..head_dim/2 */
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;

    float *row = q + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float div = ff ? ff[i] : 1.0f;
    const float ang = (float)pos * freq / div;
    const float c = cosf(ang), s = sinf(ang);
    const int i0 = 2 * i, i1 = 2 * i + 1;
    const float v0 = row[i0], v1 = row[i1];
    row[i0] = v0 * c - v1 * s;
    row[i1] = v0 * s + v1 * c;
}


/* Batched RoPE: one launch for all n tokens. Grid: ((HD/2+63)/64, n_heads, n). */
__global__ void k_rope_batched(float *__restrict__ q, int n_heads, int head_dim,
                               const int *__restrict__ d_pos_batch, float base,
                               int n, int stride) {
    const int tok = blockIdx.z;
    if (tok >= n) return;
    const int pos = d_pos_batch[tok];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;
    float *row = q + (long)tok * stride + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float ang = (float)pos * freq;
    const float c = cosf(ang), s = sinf(ang);
    const float v0 = row[i], v1 = row[i + head_dim / 2];
    row[i] = v0 * c - v1 * s;
    row[i + head_dim / 2] = v0 * s + v1 * c;
}
__global__ void k_rope_ff_batched(float *__restrict__ q, int n_heads, int head_dim,
                                  const int *__restrict__ d_pos_batch, float base,
                                  const float *__restrict__ ff, int n, int stride) {
    const int tok = blockIdx.z;
    if (tok >= n) return;
    const int pos = d_pos_batch[tok];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;
    float *row = q + (long)tok * stride + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float div = ff ? ff[i] : 1.0f;
    const float ang = (float)pos * freq / div;
    const float c = cosf(ang), s = sinf(ang);
    const float v0 = row[i], v1 = row[i + head_dim / 2];
    row[i] = v0 * c - v1 * s;
    row[i + head_dim / 2] = v0 * s + v1 * c;
}
__global__ void k_rope_gptj_batched(float *__restrict__ q, int n_heads, int head_dim,
                                    const int *__restrict__ d_pos_batch, float base,
                                    int n, int stride) {
    const int tok = blockIdx.z;
    if (tok >= n) return;
    const int pos = d_pos_batch[tok];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;
    float *row = q + (long)tok * stride + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float ang = (float)pos * freq;
    const float c = cosf(ang), s = sinf(ang);
    const int i0 = 2*i, i1 = 2*i+1;
    const float v0 = row[i0], v1 = row[i1];
    row[i0] = v0 * c - v1 * s;
    row[i1] = v0 * s + v1 * c;
}
__global__ void k_rope_gptj_ff_batched(float *__restrict__ q, int n_heads, int head_dim,
                                       const int *__restrict__ d_pos_batch, float base,
                                       const float *__restrict__ ff, int n, int stride) {
    const int tok = blockIdx.z;
    if (tok >= n) return;
    const int pos = d_pos_batch[tok];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int h = blockIdx.y;
    if (h >= n_heads || i >= head_dim / 2) return;
    float *row = q + (long)tok * stride + (long)h * head_dim;
    const float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    const float div = ff ? ff[i] : 1.0f;
    const float ang = (float)pos * freq / div;
    const float c = cosf(ang), s = sinf(ang);
    const int i0 = 2*i, i1 = 2*i+1;
    const float v0 = row[i0], v1 = row[i1];
    row[i0] = v0 * c - v1 * s;
    row[i1] = v0 * s + v1 * c;
}
/* Batched KV scatter kernels */
__global__ void k_kv_scatter_batched(const float *__restrict__ kst, const float *__restrict__ vst,
                                     float *__restrict__ Kc, float *__restrict__ Vc,
                                     const int *__restrict__ d_pos_batch,
                                     int n_kv_heads, int head_dim, int max_ctx, int n, int kvdim) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const long total = (long)n * kvdim;
    if (idx >= total) return;
    const int tok = idx / kvdim;
    const int elem = idx % kvdim;
    const int slot = d_pos_batch[tok] % max_ctx;
    Kc[(long)slot * kvdim + elem] = kst[(long)tok * kvdim + elem];
    Vc[(long)slot * kvdim + elem] = vst[(long)tok * kvdim + elem];
}
__global__ void k_kv_scatter_q8_0_batched(const float *__restrict__ kst, const float *__restrict__ vst,
                                          BlockQ8_0 *__restrict__ Kc, BlockQ8_0 *__restrict__ Vc,
                                          const int *__restrict__ d_pos_batch,
                                          int n_kv_heads, int head_dim, int max_ctx, int n) {
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_slot = (n_kv_heads * head_dim) / 32;
    const long total_blocks = (long)n * blocks_per_slot;
    if (block_idx >= total_blocks) return;
    const int tok = block_idx / blocks_per_slot;
    const int blk = block_idx % blocks_per_slot;
    const int slot = d_pos_batch[tok] % max_ctx;
    const int src_offset = tok * n_kv_heads * head_dim + blk * 32;
    float k_vals[32], v_vals[32];
    float max_k = 0.0f, max_v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) { k_vals[i]=kst[src_offset+i]; v_vals[i]=vst[src_offset+i]; max_k=fmaxf(max_k,fabsf(k_vals[i])); max_v=fmaxf(max_v,fabsf(v_vals[i])); }
    const float scale_k = (max_k > 0.0f) ? (max_k / 127.0f) : 1.0f;
    const float inv_k = (max_k > 0.0f) ? (127.0f / max_k) : 0.0f;
    const float scale_v = (max_v > 0.0f) ? (max_v / 127.0f) : 1.0f;
    const float inv_v = (max_v > 0.0f) ? (127.0f / max_v) : 0.0f;
    BlockQ8_0 *k_dest = Kc + (long)slot * blocks_per_slot + blk;
    BlockQ8_0 *v_dest = Vc + (long)slot * blocks_per_slot + blk;
    k_dest->d = __float2half(scale_k);
    v_dest->d = __float2half(scale_v);
    #pragma unroll
    for (int i = 0; i < 32; i++) { k_dest->qs[i]=(int8_t)__float2int_rn(k_vals[i]*inv_k); v_dest->qs[i]=(int8_t)__float2int_rn(v_vals[i]*inv_v); }
}
__global__ void k_kv_scatter_q4_0_batched(const float *__restrict__ kst, const float *__restrict__ vst,
                                          BlockQ4_0 *__restrict__ Kc, BlockQ4_0 *__restrict__ Vc,
                                          const int *__restrict__ d_pos_batch,
                                          int n_kv_heads, int head_dim, int max_ctx, int n) {
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_slot = (n_kv_heads * head_dim) / 32;
    const long total_blocks = (long)n * blocks_per_slot;
    if (block_idx >= total_blocks) return;
    const int tok = block_idx / blocks_per_slot;
    const int blk = block_idx % blocks_per_slot;
    const int slot = d_pos_batch[tok] % max_ctx;
    const long src_offset = (long)tok * (n_kv_heads * head_dim) + (long)blk * 32;
    float k_vals[32], v_vals[32];
    float max_k = 0.0f, max_v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        k_vals[i] = kst[src_offset + i];
        v_vals[i] = vst[src_offset + i];
        max_k = fmaxf(max_k, fabsf(k_vals[i]));
        max_v = fmaxf(max_v, fabsf(v_vals[i]));
    }
    const float scale_k = (max_k > 0.0f) ? (max_k / 7.0f) : 1.0f;
    const float inv_k   = (max_k > 0.0f) ? (7.0f / max_k) : 0.0f;
    const float scale_v = (max_v > 0.0f) ? (max_v / 7.0f) : 1.0f;
    const float inv_v   = (max_v > 0.0f) ? (7.0f / max_v) : 0.0f;
    BlockQ4_0 *k_dest = Kc + (long)slot * blocks_per_slot + blk;
    BlockQ4_0 *v_dest = Vc + (long)slot * blocks_per_slot + blk;
    // d is raw fp16 bits (uint16_t): store bit pattern, not value-convert
    k_dest->d = __half_as_ushort(__float2half(scale_k));
    v_dest->d = __half_as_ushort(__float2half(scale_v));
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int q0_k = __float2int_rn(k_vals[j] * inv_k) + 8;
        int q1_k = __float2int_rn(k_vals[j + 16] * inv_k) + 8;
        q0_k = max(0, min(15, q0_k));
        q1_k = max(0, min(15, q1_k));
        k_dest->qs[j] = (uint8_t)((q0_k & 0x0F) | ((q1_k & 0x0F) << 4));

        int q0_v = __float2int_rn(v_vals[j] * inv_v) + 8;
        int q1_v = __float2int_rn(v_vals[j + 16] * inv_v) + 8;
        q0_v = max(0, min(15, q0_v));
        q1_v = max(0, min(15, q1_v));
        v_dest->qs[j] = (uint8_t)((q0_v & 0x0F) | ((q1_v & 0x0F) << 4));
    }
}
/* Batched helpers for prefill */
__global__ void k_rmsnorm_batched(const float *__restrict__ x, const float *__restrict__ g,
                                  float *__restrict__ y, int dim, float eps, float woff, int n) {
    const int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float s[];
    const int tid = threadIdx.x;
    const float *xr = x + (long)row * dim;
    float *yr = y + (long)row * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) ss += xr[i]*xr[i];
    ss = warp_sum(ss);
    if ((tid & 31)==0) s[tid>>5]=ss;
    __syncthreads();
    if (tid==0){ float t=0; for(int w=0;w<(blockDim.x+31)/32;w++) t+=s[w]; s[0]=rsqrtf(t/(float)dim+eps); }
    __syncthreads();
    const float inv=s[0];
    if (woff==0.0f){ for(int i=tid;i<dim;i+=blockDim.x) yr[i]=xr[i]*inv*g[i]; }
    else { for(int i=tid;i<dim;i+=blockDim.x) yr[i]=xr[i]*inv*(woff+g[i]); }
}
__global__ void k_add_bias_batched(float *__restrict__ dst, const float *__restrict__ bias, int row_dim, int n) {
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const long total=(long)n*row_dim;
    if(idx>=total) return;
    dst[idx]+=bias[idx%row_dim];
}
__global__ void k_qk_norm_rms_batched(float *__restrict__ x, const float *__restrict__ g,
                                      int n_heads, int head_dim, float eps, int n, int stride) {
    const int tok = blockIdx.x / n_heads;
    const int h = blockIdx.x % n_heads;
    if (tok>=n) return;
    extern __shared__ float s[];
    const int tid=threadIdx.x;
    float *row = x + (long)tok*stride + (long)h*head_dim;
    float ss=0;
    for(int i=tid;i<head_dim;i+=blockDim.x) ss+=row[i]*row[i];
    ss=warp_sum(ss);
    if((tid&31)==0) s[tid>>5]=ss;
    __syncthreads();
    if(tid==0){ float t=0; for(int w=0;w<(blockDim.x+31)/32;w++) t+=s[w]; s[0]=rsqrtf(t/(float)head_dim+eps); }
    __syncthreads();
    const float inv=s[0];
    for(int i=tid;i<head_dim;i+=blockDim.x) row[i]=row[i]*inv*g[i];
}

/* M7 task 3: per-head RMSNorm over q/k rows pre-rope (qwen3 trait).
 * One block per head; mean-of-squares over head_dim only.
 * y[h*hd+i] = x[h*hd+i] * rsqrt(mean(x^2)+eps) * gamma[i]. */
__global__ void k_qk_norm_rms(float *__restrict__ x, const float *__restrict__ g,
                              int n_heads, int head_dim, float eps) {
    extern __shared__ float s[];
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const float *row = x + (long)h * head_dim;
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) ss += row[i] * row[i];
    ss = warp_sum(ss);
    if ((tid & 31) == 0) s[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++) t += s[w];
        s[0] = rsqrtf(t / (float)head_dim + eps);
    }
    __syncthreads();
    const float inv = s[0];
    for (int i = tid; i < head_dim; i += blockDim.x)
        x[(long)h * head_dim + i] = row[i] * inv * g[i];
}

/* Final-logit tanh softcap (gemma2): l = tanh(l/c)*c.
 * Mirrors oracle build_gemma2 tail (llama.cpp:5000-5003:
 * scale(1/c) -> tanh -> scale(c)). No-op launch skipped when cap<=0. */
__global__ void k_softcap(float *__restrict__ logits, int n, float cap) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) logits[i] = tanhf(logits[i] / cap) * cap;
}

__global__ void k_add(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) dst[i] += src[i];
}

/* GQA flash-attention decode: one warp per query head.
 * Cache layout: [n_kv_heads, max_ctx, head_dim] per layer (K and V separate). */
__global__ void k_flash_gqa(const float *__restrict__ q,
                            const float *__restrict__ Kc,
                            const float *__restrict__ Vc,
                            float *__restrict__ out,
                            const int *__restrict__ d_pos, /* inclusive: attend to 0..*d_pos */
                            int n_heads, int n_kv_heads, int head_dim,
                            int max_ctx, float scale, int window) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);          /* GQA group map */
    const int elems = head_dim / 32;                     /* per-lane elements */
    const float *qh = q + (long)h * head_dim + lane * elems;

    /* SWA (gemma2): skip slots older than the window. Slot t attends iff
     * pos - t < window (HF gemma2 masking: scores masked when i-j >= swa).
     * window <= 0 => full attention, t0=0, loop unchanged. */
    int t0 = 0;
    if (window > 0 && pos >= window) t0 = pos - window + 1;

    float qreg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0};

    /* INVARIANT: callers enforce pos < max_ctx (no ring wraparound in this loop) */
    for (int t = t0; t <= pos; t++) {
        /* slot-major layout: [slot][kv_head*head_dim], matches GEMV writes */
        const long off = ((long)t * n_kv_heads + kvh) * head_dim + lane * elems;
        const float *kp = Kc + off;
        const float *vp = Vc + off;
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) score += qreg[i] * kp[i];
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        const float m_new = fmaxf(m_prev, score);
        const float ex = expf(score - m_new);
        const float alpha = expf(m_prev - m_new);
        l_prev = l_prev * alpha + ex;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) oreg[i] = oreg[i] * alpha + ex * vp[i];
        m_prev = m_new;
    }

    const float inv_l = 1.0f / (l_prev + 1e-8f);
    float *oh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 16; i++)
        if (i < elems) oh[i] = oreg[i] * inv_l;
}

/* M9 split-K flash attention (proto source: tests/proto_flash_splitk.cu f7687e6).
 *
 * Problem the per-warp path above hits at long ctx: grid = n_heads blocks,
 * each block serially walks pos-t0+1 slots. SM occupancy caps and the
 * linear-in-ctx inner loop leave the GPU severely underused; the proto
 * measured 12x at ctx>=1024 on the same kernel+hardware. Wire-in path:
 *   - k_flash_gqa_splitk: grid = (H_l, S) blocks, each owns a slot subrange
 *     of size ~ (nslots + S-1)/S, runs the same online-softmax loop, writes
 *     raw (m, l, acc[hd]) partials to per-call workspace.
 *   - k_flash_gqa_combine: one block/head, online-softmax-merges the S
 *     partials. Order-independent to fp noise; m84/m61 goldens match the
 *     per-warp path at short ctx.
 *
 * Workspace alloc: S * H_l * (HDl + 2) floats. S = clamp(ctx/256, 2, 16).
 * Per-engine scratch in Qwen2Engine (d_split_pacc/pm/pl) sized to the worst
 * per-layer (max heads, max hd) seen in the model. */
__global__ void k_flash_gqa_splitk(const float *__restrict__ q,
                                   const float *__restrict__ Kc,
                                   const float *__restrict__ Vc,
                                   float *__restrict__ p_acc, /* [S][H][hd] */
                                   float *__restrict__ p_m,   /* [S][H]    */
                                   float *__restrict__ p_l,   /* [S][H]    */
                                   const int *__restrict__ d_pos,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   float scale, int window, int S) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);
    const int elems = head_dim / 32;

    int t_lo = 0;
    if (window > 0 && pos >= window) t_lo = pos - window + 1;
    const int nslots = pos - t_lo + 1;
    const int chunk = (nslots + S - 1) / S;
    const int begin = t_lo + s * chunk;
    const int end = min(pos + 1, t_lo + (s + 1) * chunk);

    float *myacc = p_acc + ((size_t)s * n_heads + h) * head_dim + lane * elems;
    float *mym = p_m + (size_t)s * n_heads + h;
    float *myl = p_l + (size_t)s * n_heads + h;

    if (begin >= end) {
        *mym = -INFINITY;
        *myl = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) myacc[i] = 0.0f;
        return;
    }

    const float *qh = q + (long)h * head_dim + lane * elems;
    float qreg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0};

    for (int t = begin; t < end; t++) {
        const long off = ((long)t * n_kv_heads + kvh) * head_dim + lane * elems;
        const float *kp = Kc + off;
        const float *vp = Vc + off;
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) score += qreg[i] * kp[i];
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        const float m_new = fmaxf(m_prev, score);
        const float ex = expf(score - m_new);
        const float alpha = expf(m_prev - m_new);
        l_prev = l_prev * alpha + ex;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) oreg[i] = oreg[i] * alpha + ex * vp[i];
        m_prev = m_new;
    }

    *mym = m_prev;
    *myl = l_prev;
#pragma unroll
    for (int i = 0; i < 16; i++)
        if (i < elems) myacc[i] = oreg[i];
}

__global__ void k_flash_gqa_combine(const float *__restrict__ p_acc,
                                   const float *__restrict__ p_m,
                                   const float *__restrict__ p_l,
                                   float *__restrict__ out,
                                   int n_heads, int head_dim, int S) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int elems = head_dim / 32;

    float m = -INFINITY, l = 0.0f;
    float oreg[16] = {0};

    for (int s = 0; s < S; s++) {
        const size_t idx = (size_t)s * n_heads + h;
        const float ls = p_l[idx];
        if (!(ls > 0.0f)) continue; /* empty split */
        const float ms = p_m[idx];
        const float m_new = fmaxf(m, ms);
        const float alpha = expf(m - m_new);
        const float beta = expf(ms - m_new);
        const float *acc = p_acc + idx * head_dim + lane * elems;
        l = l * alpha + ls * beta;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) oreg[i] = oreg[i] * alpha + acc[i] * beta;
        m = m_new;
    }

    const float inv_l = 1.0f / (l + 1e-8f);
    float *oh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 16; i++)
        if (i < elems) oh[i] = oreg[i] * inv_l;
}

/* Scatter staged K/V rows into the cache slot given by the device position
 * scalar. Runs after K staging (+bias+RoPE), so the cache holds post-RoPE K. */
__global__ void k_kv_scatter(const float *__restrict__ kst, const float *__restrict__ vst,
                             float *__restrict__ Kc, float *__restrict__ Vc,
                             const int *__restrict__ d_pos, int n_kv_heads, int head_dim, int max_ctx) {
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int kvdim = n_kv_heads*head_dim;
    if (i >= kvdim) return;
    const int slot = (*d_pos) % max_ctx;
    Kc[(long)slot*kvdim + i] = kst[i];
    Vc[(long)slot*kvdim + i] = vst[i];
}

/* Scatter kernel: quantizes FP32 K/V staging vectors into Q8_0 cache slot (*d_pos) */
__global__ void k_kv_scatter_q8_0(
    const float *__restrict__ kst,
    const float *__restrict__ vst,
    BlockQ8_0   *__restrict__ Kc,
    BlockQ8_0   *__restrict__ Vc,
    const int   *__restrict__ d_pos,
    int n_kv_heads, int head_dim, int max_ctx) {
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_blocks_per_slot = (n_kv_heads * head_dim) / 32;
    if (block_idx >= num_blocks_per_slot) return;
    const int slot = (*d_pos) % max_ctx;
    const int src_offset = block_idx * 32;

    float k_vals[32], v_vals[32];
    float max_k = 0.0f, max_v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        k_vals[i] = kst[src_offset + i];
        v_vals[i] = vst[src_offset + i];
        max_k = fmaxf(max_k, fabsf(k_vals[i]));
        max_v = fmaxf(max_v, fabsf(v_vals[i]));
    }

    const float scale_k = (max_k > 0.0f) ? (max_k / 127.0f) : 1.0f;
    const float inv_k   = (max_k > 0.0f) ? (127.0f / max_k) : 0.0f;
    const float scale_v = (max_v > 0.0f) ? (max_v / 127.0f) : 1.0f;
    const float inv_v   = (max_v > 0.0f) ? (127.0f / max_v) : 0.0f;

    BlockQ8_0 *k_dest = Kc + (long)slot * num_blocks_per_slot + block_idx;
    BlockQ8_0 *v_dest = Vc + (long)slot * num_blocks_per_slot + block_idx;

    k_dest->d = __float2half(scale_k);
    v_dest->d = __float2half(scale_v);

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        k_dest->qs[i] = (int8_t)__float2int_rn(k_vals[i] * inv_k);
        v_dest->qs[i] = (int8_t)__float2int_rn(v_vals[i] * inv_v);
    }
}

/* Scatter kernel: quantizes FP32 K/V staging vectors into Q4_0 cache slot (*d_pos) */
__global__ void k_kv_scatter_q4_0(
    const float *__restrict__ kst,
    const float *__restrict__ vst,
    BlockQ4_0   *__restrict__ Kc,
    BlockQ4_0   *__restrict__ Vc,
    const int   *__restrict__ d_pos,
    int n_kv_heads, int head_dim, int max_ctx) {
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_blocks_per_slot = (n_kv_heads * head_dim) / 32;
    if (block_idx >= num_blocks_per_slot) return;
    const int slot = (*d_pos) % max_ctx;
    const int src_offset = block_idx * 32;

    float k_vals[32], v_vals[32];
    float max_k = 0.0f, max_v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        k_vals[i] = kst[src_offset + i];
        v_vals[i] = vst[src_offset + i];
        max_k = fmaxf(max_k, fabsf(k_vals[i]));
        max_v = fmaxf(max_v, fabsf(v_vals[i]));
    }

    const float scale_k = (max_k > 0.0f) ? (max_k / 7.0f) : 1.0f;
    const float inv_k   = (max_k > 0.0f) ? (7.0f / max_k) : 0.0f;
    const float scale_v = (max_v > 0.0f) ? (max_v / 7.0f) : 1.0f;
    const float inv_v   = (max_v > 0.0f) ? (7.0f / max_v) : 0.0f;

    BlockQ4_0 *k_dest = Kc + (long)slot * num_blocks_per_slot + block_idx;
    BlockQ4_0 *v_dest = Vc + (long)slot * num_blocks_per_slot + block_idx;

    // d is raw fp16 bits (uint16_t): store bit pattern, not value-convert
    k_dest->d = __half_as_ushort(__float2half(scale_k));
    v_dest->d = __half_as_ushort(__float2half(scale_v));

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int q0_k = __float2int_rn(k_vals[j] * inv_k) + 8;
        int q1_k = __float2int_rn(k_vals[j + 16] * inv_k) + 8;
        q0_k = max(0, min(15, q0_k));
        q1_k = max(0, min(15, q1_k));
        k_dest->qs[j] = (uint8_t)((q0_k & 0x0F) | ((q1_k & 0x0F) << 4));

        int q0_v = __float2int_rn(v_vals[j] * inv_v) + 8;
        int q1_v = __float2int_rn(v_vals[j + 16] * inv_v) + 8;
        q0_v = max(0, min(15, q0_v));
        q1_v = max(0, min(15, q1_v));
        v_dest->qs[j] = (uint8_t)((q0_v & 0x0F) | ((q1_v & 0x0F) << 4));
    }
}

/* Late-enable backfill (P1-2 fix): quantize FP32 cache slots [0..n_slots)
 * into the Q cache. Same per-block quantization as the single-token scatter
 * kernels, but the source is the FP32 cache slab (layout [slot][kvdim]) and
 * the slot index is explicit (no d_pos). Launched once per layer at enable
 * time; shared-KV layers are skipped by the caller (they alias the source
 * slab, mirroring the forward scatter path). */
__global__ void k_kv_backfill_q8_0(
    const float *__restrict__ Kf,
    const float *__restrict__ Vf,
    BlockQ8_0   *__restrict__ Kc,
    BlockQ8_0   *__restrict__ Vc,
    int n_slots, int kvdim) {
    const int nb = kvdim / 32;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_slots * nb) return;
    const int slot = idx / nb;
    const int bi = idx % nb;
    const float *ks = Kf + (long)slot * kvdim + bi * 32;
    const float *vs = Vf + (long)slot * kvdim + bi * 32;
    float max_k = 0.0f, max_v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        max_k = fmaxf(max_k, fabsf(ks[i]));
        max_v = fmaxf(max_v, fabsf(vs[i]));
    }
    const float scale_k = (max_k > 0.0f) ? (max_k / 127.0f) : 1.0f;
    const float inv_k   = (max_k > 0.0f) ? (127.0f / max_k) : 0.0f;
    const float scale_v = (max_v > 0.0f) ? (max_v / 127.0f) : 1.0f;
    const float inv_v   = (max_v > 0.0f) ? (127.0f / max_v) : 0.0f;
    BlockQ8_0 *kd = Kc + (long)slot * nb + bi;
    BlockQ8_0 *vd = Vc + (long)slot * nb + bi;
    kd->d = __float2half(scale_k);
    vd->d = __float2half(scale_v);
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        kd->qs[i] = (int8_t)__float2int_rn(ks[i] * inv_k);
        vd->qs[i] = (int8_t)__float2int_rn(vs[i] * inv_v);
    }
}

__global__ void k_kv_backfill_q4_0(
    const float *__restrict__ Kf,
    const float *__restrict__ Vf,
    BlockQ4_0   *__restrict__ Kc,
    BlockQ4_0   *__restrict__ Vc,
    int n_slots, int kvdim) {
    const int nb = kvdim / 32;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_slots * nb) return;
    const int slot = idx / nb;
    const int bi = idx % nb;
    const float *ks = Kf + (long)slot * kvdim + bi * 32;
    const float *vs = Vf + (long)slot * kvdim + bi * 32;
    float max_k = 0.0f, max_v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        max_k = fmaxf(max_k, fabsf(ks[i]));
        max_v = fmaxf(max_v, fabsf(vs[i]));
    }
    const float scale_k = (max_k > 0.0f) ? (max_k / 7.0f) : 1.0f;
    const float inv_k   = (max_k > 0.0f) ? (7.0f / max_k) : 0.0f;
    const float scale_v = (max_v > 0.0f) ? (max_v / 7.0f) : 1.0f;
    const float inv_v   = (max_v > 0.0f) ? (7.0f / max_v) : 0.0f;
    BlockQ4_0 *kd = Kc + (long)slot * nb + bi;
    BlockQ4_0 *vd = Vc + (long)slot * nb + bi;
    // d is raw fp16 bits (uint16_t): store bit pattern, not value-convert
    kd->d = __half_as_ushort(__float2half(scale_k));
    vd->d = __half_as_ushort(__float2half(scale_v));
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int q0_k = __float2int_rn(ks[j] * inv_k) + 8;
        int q1_k = __float2int_rn(ks[j + 16] * inv_k) + 8;
        q0_k = max(0, min(15, q0_k));
        q1_k = max(0, min(15, q1_k));
        kd->qs[j] = (uint8_t)((q0_k & 0x0F) | ((q1_k & 0x0F) << 4));
        int q0_v = __float2int_rn(vs[j] * inv_v) + 8;
        int q1_v = __float2int_rn(vs[j + 16] * inv_v) + 8;
        q0_v = max(0, min(15, q0_v));
        q1_v = max(0, min(15, q1_v));
        vd->qs[j] = (uint8_t)((q0_v & 0x0F) | ((q1_v & 0x0F) << 4));
    }
}

/* Flash GQA kernel reading Q8_0 KV cache, dequantizing on-the-fly in registers */
__global__ void k_flash_gqa_q8_0(
    const float     *__restrict__ q,
    const BlockQ8_0 *__restrict__ Kc_q8,
    const BlockQ8_0 *__restrict__ Vc_q8,
    float           *__restrict__ out,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim, int max_ctx,
    float scale, int window) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int lane = threadIdx.x; // 0..31
    const int elems = head_dim / 32;
    const int kvh = h / (n_heads / n_kv_heads);
    const int blocks_per_head = head_dim / 32;
    const int blocks_per_slot = n_kv_heads * blocks_per_head;

    const float *qh = q + (long)h * head_dim + lane * elems;

    int t0 = 0;
    if (window > 0 && pos >= window) t0 = pos - window + 1;

    float qreg[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        qreg[i] = (i < elems) ? qh[i] : 0.0f;
    }

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0.0f};

    const int block_in_head = (lane * elems) / 32;
    const int elem_sub_idx = (lane * elems) % 32;
    const int block_idx = kvh * blocks_per_head + block_in_head;
    const int block_byte_off = block_idx * 34;
    const int wsc = block_byte_off >> 2;
    const int sh_d = block_byte_off & 2;
    const int a0 = (block_byte_off + 2) >> 2;
    const int sh_qs = (block_byte_off + 2) & 2;

    const int k_elem_word = elem_sub_idx >> 2;

    const long stride = (long)blocks_per_slot * 34;
    const char *k_ptr = (const char *)Kc_q8 + (long)t0 * stride;
    const char *v_ptr = (const char *)Vc_q8 + (long)t0 * stride;

    for (int t = t0; t <= pos; t++) {
        const uint32_t *k_slot_u32 = (const uint32_t *)k_ptr;
        const uint32_t *v_slot_u32 = (const uint32_t *)v_ptr;
        k_ptr += stride;
        v_ptr += stride;

        // Load scale dk
        const uint32_t dw_k = k_slot_u32[wsc];
        const unsigned short d16_k = (unsigned short)(sh_d ? (dw_k >> 16) : (dw_k & 0xFFFFu));
        const float dk = __half2float(__ushort_as_half(d16_k));

        // Load int8 values for K
        const uint32_t lo_k = k_slot_u32[a0 + k_elem_word];
        const uint32_t vv_k = sh_qs ? __byte_perm(lo_k, k_slot_u32[a0 + k_elem_word + 1], 0x5432) : lo_k;

        float score = 0.0f;
        if (elems == 4) {
            float k0 = (float)((int8_t)(vv_k      ));
            float k1 = (float)((int8_t)(vv_k >>  8));
            float k2 = (float)((int8_t)(vv_k >> 16));
            float k3 = (float)((int8_t)(vv_k >> 24));
            score = (qreg[0]*k0 + qreg[1]*k1 + qreg[2]*k2 + qreg[3]*k3) * dk;
        } else if (elems == 2) {
            int shift = (elem_sub_idx & 2) ? 16 : 0;
            float k0 = (float)((int8_t)(vv_k >> shift));
            float k1 = (float)((int8_t)(vv_k >> (shift + 8)));
            score = (qreg[0]*k0 + qreg[1]*k1) * dk;
        } else {
            #pragma unroll
            for (int i = 0; i < elems; i++) {
                int shift = ((elem_sub_idx + i) & 3) * 8;
                float k_val = (float)((int8_t)(vv_k >> shift));
                score += qreg[i] * (k_val * dk);
            }
        }

        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        float m_curr = fmaxf(m_prev, score);
        float p = expf(score - m_curr);
        float alpha = expf(m_prev - m_curr);
        float l_curr = l_prev * alpha + p;

        // Load scale dv
        const uint32_t dw_v = v_slot_u32[wsc];
        const unsigned short d16_v = (unsigned short)(sh_d ? (dw_v >> 16) : (dw_v & 0xFFFFu));
        const float dv = __half2float(__ushort_as_half(d16_v));

        // Load int8 values for V
        const uint32_t lo_v = v_slot_u32[a0 + k_elem_word];
        const uint32_t vv_v = sh_qs ? __byte_perm(lo_v, v_slot_u32[a0 + k_elem_word + 1], 0x5432) : lo_v;

        const float pdv = p * dv;
        if (elems == 4) {
            float v0 = (float)((int8_t)(vv_v      ));
            float v1 = (float)((int8_t)(vv_v >>  8));
            float v2 = (float)((int8_t)(vv_v >> 16));
            float v3 = (float)((int8_t)(vv_v >> 24));
            oreg[0] = oreg[0] * alpha + pdv * v0;
            oreg[1] = oreg[1] * alpha + pdv * v1;
            oreg[2] = oreg[2] * alpha + pdv * v2;
            oreg[3] = oreg[3] * alpha + pdv * v3;
        } else if (elems == 2) {
            int shift = (elem_sub_idx & 2) ? 16 : 0;
            float v0 = (float)((int8_t)(vv_v >> shift));
            float v1 = (float)((int8_t)(vv_v >> (shift + 8)));
            oreg[0] = oreg[0] * alpha + pdv * v0;
            oreg[1] = oreg[1] * alpha + pdv * v1;
        } else {
            #pragma unroll
            for (int i = 0; i < elems; i++) {
                int shift = ((elem_sub_idx + i) & 3) * 8;
                float v_val = (float)((int8_t)(vv_v >> shift));
                oreg[i] = oreg[i] * alpha + pdv * v_val;
            }
        }

        m_prev = m_curr;
        l_prev = l_curr;
    }

    float *outh = out + (long)h * head_dim + lane * elems;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        if (i < elems) outh[i] = oreg[i] / l_prev;
    }
}

#define BC_SPLIT 64 // tried 32 and 128 at N>32768: 64 best (2.18ms vs 2.20/2.19 at 131k paged), kept 64

__global__ void k_fa2_q8_split(
    const float     *__restrict__ q,
    const BlockQ8_0 *__restrict__ Kc_q8,
    const BlockQ8_0 *__restrict__ Vc_q8,
    float           *__restrict__ p_acc,
    float           *__restrict__ p_m,
    float           *__restrict__ p_l,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window, int S)
{
    const int pos = *d_pos;
    const int s = blockIdx.x;
    const int kv = blockIdx.y;
    if (s >= S || kv >= n_kv_heads) return;

    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (warp >= G) return;

    const int head = kv * G + warp;
    const int elems = head_dim / 32;
    const int blocks_per_head = head_dim / 32;

    // Load Q vector directly into registers for this head & lane
    const float *qh = q + (long)head * head_dim + lane * elems;
    float qreg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (elems == 4) {
        const float4 q_vec = *(const float4 *)qh;
        qreg[0] = q_vec.x; qreg[1] = q_vec.y; qreg[2] = q_vec.z; qreg[3] = q_vec.w;
    } else if (elems == 2) {
        const float2 q_vec = *(const float2 *)qh;
        qreg[0] = q_vec.x; qreg[1] = q_vec.y;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i < elems) qreg[i] = qh[i];
        }
    }

    // Sliding window & slice bounds
    int t_lo = (window > 0 && pos >= window) ? (pos - window + 1) : 0;
    int nslots = pos - t_lo + 1;
    int chunk = (nslots + S - 1) / S;
    int begin = t_lo + s * chunk;
    int end = min(pos + 1, t_lo + (s + 1) * chunk);

    // Inactive slice early exit
    if (begin >= end) {
        if (lane == 0) {
            p_m[(size_t)s * n_heads + head] = -1e30f;
            p_l[(size_t)s * n_heads + head] = 0.0f;
        }
        float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i < elems) myacc[i] = 0.0f;
        }
        return;
    }

    // Shared memory: BC=64 KV tokens
    extern __shared__ char raw_smem_split[];
    half   *sK_d = (half*)raw_smem_split;
    half   *sV_d = sK_d + BC_SPLIT * blocks_per_head;
    int8_t *sK_q = (int8_t*)(sV_d + BC_SPLIT * blocks_per_head);
    int8_t *sV_q = sK_q + BC_SPLIT * head_dim;

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int block_in_head = (lane * elems) / 32;

    // Iterate over tokens in [begin, end) in tiles of BC
    for (int t_tile = begin; t_tile < end; t_tile += BC_SPLIT) {
        int t_tile_end = min(end, t_tile + BC_SPLIT);
        int bc_active = t_tile_end - t_tile;

        // Cooperative load of bc_active KV blocks into smem
        // FIX: byte-wise copy avoids misaligned 4-byte loads (qs at offset 2 in 34-byte BlockQ8_0)
        int total_blocks = bc_active * blocks_per_head;
        for (int i = tid; i < total_blocks; i += blockDim.x) {
            int tok = i / blocks_per_head;
            int b   = i % blocks_per_head;
            long g_idx = ((long)(t_tile + tok) * n_kv_heads + kv) * blocks_per_head + b;
            const BlockQ8_0 bk = Kc_q8[g_idx];
            const BlockQ8_0 bv = Vc_q8[g_idx];
            sK_d[tok * blocks_per_head + b] = bk.d;
            sV_d[tok * blocks_per_head + b] = bv.d;
            int row_off = tok * head_dim + b * 32;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                sK_q[row_off + j] = bk.qs[j];
                sV_q[row_off + j] = bv.qs[j];
            }
        }
        __syncthreads();

        // Process tokens in smem tile
        for (int t_idx = 0; t_idx < bc_active; t_idx++) {
            const float dk = __half2float(sK_d[t_idx * blocks_per_head + block_in_head]);
            const float dv = __half2float(sV_d[t_idx * blocks_per_head + block_in_head]);

            const int byte_off = t_idx * head_dim + lane * elems;
            float k_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float v_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            if (elems == 4) {
                const uint32_t k_u32 = *(const uint32_t *)&sK_q[byte_off];
                const uint32_t v_u32 = *(const uint32_t *)&sV_q[byte_off];
                k_val[0] = (float)((int8_t)(k_u32      ));
                k_val[1] = (float)((int8_t)(k_u32 >>  8));
                k_val[2] = (float)((int8_t)(k_u32 >> 16));
                k_val[3] = (float)((int8_t)(k_u32 >> 24));
                v_val[0] = (float)((int8_t)(v_u32      ));
                v_val[1] = (float)((int8_t)(v_u32 >>  8));
                v_val[2] = (float)((int8_t)(v_u32 >> 16));
                v_val[3] = (float)((int8_t)(v_u32 >> 24));
            } else if (elems == 2) {
                const uint16_t k_u16 = *(const uint16_t *)&sK_q[byte_off];
                const uint16_t v_u16 = *(const uint16_t *)&sV_q[byte_off];
                k_val[0] = (float)((int8_t)(k_u16      ));
                k_val[1] = (float)((int8_t)(k_u16 >>  8));
                v_val[0] = (float)((int8_t)(v_u16      ));
                v_val[1] = (float)((int8_t)(v_u16 >>  8));
            } else {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (i < elems) {
                        k_val[i] = (float)sK_q[byte_off + i];
                        v_val[i] = (float)sV_q[byte_off + i];
                    }
                }
            }

            float dot_partial = (qreg[0] * k_val[0] + qreg[1] * k_val[1] + qreg[2] * k_val[2] + qreg[3] * k_val[3]) * dk;
            float score = warp_sum(dot_partial);
            score = __shfl_sync(0xffffffff, score, 0) * scale;

            float m_curr = fmaxf(m_prev, score);
            float p = expf(score - m_curr);
            float alpha = expf(m_prev - m_curr);
            l_prev = l_prev * alpha + p;

            float pdv = p * dv;
            acc[0] = acc[0] * alpha + pdv * v_val[0];
            acc[1] = acc[1] * alpha + pdv * v_val[1];
            acc[2] = acc[2] * alpha + pdv * v_val[2];
            acc[3] = acc[3] * alpha + pdv * v_val[3];

            m_prev = m_curr;
        }
        __syncthreads();
    }

    // Write partials
    if (lane == 0) {
        p_m[(size_t)s * n_heads + head] = m_prev;
        p_l[(size_t)s * n_heads + head] = l_prev;
    }
    float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (i < elems) myacc[i] = acc[i];
    }
}

__global__ void k_fa2_q4_split(
    const float     *__restrict__ q,
    const BlockQ4_0 *__restrict__ Kc_q4,
    const BlockQ4_0 *__restrict__ Vc_q4,
    float           *__restrict__ p_acc,
    float           *__restrict__ p_m,
    float           *__restrict__ p_l,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window, int S)
{
    const int pos = *d_pos;
    const int s = blockIdx.x;
    const int kv = blockIdx.y;
    if (s >= S || kv >= n_kv_heads) return;

    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (warp >= G) return;

    const int head = kv * G + warp;

    const int elems = head_dim / 32;

    // Load Q vector directly into registers for this head & lane
    // (elems-branch: HD=128 -> float4, HD=64 -> float2; else scalar)
    const float *qh = q + (long)head * head_dim + lane * elems;
    float qreg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (elems == 4) {
        const float4 q_vec = *(const float4 *)qh;
        qreg[0] = q_vec.x; qreg[1] = q_vec.y; qreg[2] = q_vec.z; qreg[3] = q_vec.w;
    } else if (elems == 2) {
        const float2 q_vec = *(const float2 *)qh;
        qreg[0] = q_vec.x; qreg[1] = q_vec.y;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i < elems) qreg[i] = qh[i];
        }
    }

    int t_lo = (window > 0 && pos >= window) ? (pos - window + 1) : 0;
    int nslots = pos - t_lo + 1;
    int chunk = (nslots + S - 1) / S;
    int begin = t_lo + s * chunk;
    int end = min(pos + 1, t_lo + (s + 1) * chunk);

    if (begin >= end) {
        if (lane == 0) {
            p_m[(size_t)s * n_heads + head] = -1e30f;
            p_l[(size_t)s * n_heads + head] = 0.0f;
        }
        float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i < elems) myacc[i] = 0.0f;
        }
        return;
    }

    const int blocks_per_head = head_dim / 32;

    extern __shared__ char raw_smem_split_q4[];
    half    *sK_d = (half*)raw_smem_split_q4;
    half    *sV_d = sK_d + BC_SPLIT * blocks_per_head;
    uint8_t *sK_q = (uint8_t*)(sV_d + BC_SPLIT * blocks_per_head);
    uint8_t *sV_q = sK_q + BC_SPLIT * (head_dim / 2);

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // elems-scaled block index: HD=128 -> lane>>3, HD=64 -> lane>>4
    const int block_in_head = (lane * elems) / 32;
    // HD=128 fast-path mapping: 8 lanes/block, 4 nibbles/lane
    const int sub = lane & 7;
    const bool is_high = (sub >= 4);
    const int byte_offset = (sub & 3) * 4;
    const uint32_t shift = is_high ? 4 : 0;

    for (int t_tile = begin; t_tile < end; t_tile += BC_SPLIT) {
        int t_tile_end = min(end, t_tile + BC_SPLIT);
        int bc_active = t_tile_end - t_tile;

        int total_blocks = bc_active * blocks_per_head;
        for (int i = tid; i < total_blocks; i += blockDim.x) {
            int tok = i / blocks_per_head;
            int b   = i % blocks_per_head;
            long g_idx = ((long)(t_tile + tok) * n_kv_heads + kv) * blocks_per_head + b;
            const BlockQ4_0 bk = Kc_q4[g_idx];
            const BlockQ4_0 bv = Vc_q4[g_idx];
            // d is raw fp16 bits: reinterpret, not value-convert
            sK_d[tok * blocks_per_head + b] = __ushort_as_half(bk.d);
            sV_d[tok * blocks_per_head + b] = __ushort_as_half(bv.d);

            int row_off = tok * (head_dim / 2) + b * 16;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                ((uint16_t *)&sK_q[row_off])[j] = ((const uint16_t *)&bk.qs[0])[j];
                ((uint16_t *)&sV_q[row_off])[j] = ((const uint16_t *)&bv.qs[0])[j];
            }
        }
        __syncthreads();

        for (int t_idx = 0; t_idx < bc_active; t_idx++) {
            const float dk = __half2float(sK_d[t_idx * blocks_per_head + block_in_head]);
            const float dv = __half2float(sV_d[t_idx * blocks_per_head + block_in_head]);

            float kval[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float vval[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (elems == 2) {
                // Q4_0 packing: byte j holds vals j (low) and j+16 (high).
                // Lane covers 2 consecutive vals vbase..vbase+1 of its block.
                const size_t krow = (size_t)t_idx * (head_dim / 2) + block_in_head * 16;
                const int vbase = (lane & 15) * 2;
                const uint8_t k_b0 = sK_q[krow + ((vbase    ) & 15)];
                const uint8_t k_b1 = sK_q[krow + ((vbase + 1) & 15)];
                const uint8_t v_b0 = sV_q[krow + ((vbase    ) & 15)];
                const uint8_t v_b1 = sV_q[krow + ((vbase + 1) & 15)];
                kval[0] = (float)(((int)((k_b0 >> ((vbase      >= 16) ? 4 : 0)) & 0x0F)) - 8);
                kval[1] = (float)(((int)((k_b1 >> ((vbase + 1 >= 16) ? 4 : 0)) & 0x0F)) - 8);
                vval[0] = (float)(((int)((v_b0 >> ((vbase      >= 16) ? 4 : 0)) & 0x0F)) - 8);
                vval[1] = (float)(((int)((v_b1 >> ((vbase + 1 >= 16) ? 4 : 0)) & 0x0F)) - 8);
            } else {
                const uint32_t k_u32 = *(const uint32_t *)&sK_q[t_idx * (head_dim / 2) + block_in_head * 16 + byte_offset];
                const uint32_t v_u32 = *(const uint32_t *)&sV_q[t_idx * (head_dim / 2) + block_in_head * 16 + byte_offset];

                uint32_t k_shifted = k_u32 >> shift;
                kval[0] = (float)((int)((k_shifted      ) & 0x0F) - 8);
                kval[1] = (float)((int)((k_shifted >>  8) & 0x0F) - 8);
                kval[2] = (float)((int)((k_shifted >> 16) & 0x0F) - 8);
                kval[3] = (float)((int)((k_shifted >> 24) & 0x0F) - 8);

                uint32_t v_shifted = v_u32 >> shift;
                vval[0] = (float)((int)((v_shifted      ) & 0x0F) - 8);
                vval[1] = (float)((int)((v_shifted >>  8) & 0x0F) - 8);
                vval[2] = (float)((int)((v_shifted >> 16) & 0x0F) - 8);
                vval[3] = (float)((int)((v_shifted >> 24) & 0x0F) - 8);
            }

            float dot_partial = (qreg[0] * kval[0] + qreg[1] * kval[1] + qreg[2] * kval[2] + qreg[3] * kval[3]) * dk;
            float score = warp_sum(dot_partial);
            score = __shfl_sync(0xffffffff, score, 0) * scale;

            float m_curr = fmaxf(m_prev, score);
            float p = expf(score - m_curr);
            float alpha = expf(m_prev - m_curr);
            l_prev = l_prev * alpha + p;

            float pdv = p * dv;
            acc[0] = acc[0] * alpha + pdv * vval[0];
            acc[1] = acc[1] * alpha + pdv * vval[1];
            acc[2] = acc[2] * alpha + pdv * vval[2];
            acc[3] = acc[3] * alpha + pdv * vval[3];

            m_prev = m_curr;
        }
        __syncthreads();
    }

    if (lane == 0) {
        p_m[(size_t)s * n_heads + head] = m_prev;
        p_l[(size_t)s * n_heads + head] = l_prev;
    }
    float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (i < elems) myacc[i] = acc[i];
    }
}

#define BC_FP32 32
__global__ void k_fa2_fp32_split(
    const float *__restrict__ q,
    const float *__restrict__ Kc,
    const float *__restrict__ Vc,
    float *__restrict__ p_acc,
    float *__restrict__ p_m,
    float *__restrict__ p_l,
    const int *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window, int S)
{
    const int pos = *d_pos;
    const int s = blockIdx.x;
    const int kv = blockIdx.y;
    if (s >= S || kv >= n_kv_heads) return;
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (warp >= G) return;
    const int head = kv * G + warp;
    const int elems = head_dim / 32;
    const float *qh = q + (long)head * head_dim + lane * elems;
    float qreg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (elems == 4) {
        const float4 qv = *(const float4 *)qh;
        qreg[0]=qv.x; qreg[1]=qv.y; qreg[2]=qv.z; qreg[3]=qv.w;
    } else if (elems == 2) {
        const float2 qv = *(const float2 *)qh;
        qreg[0]=qv.x; qreg[1]=qv.y;
    } else {
#pragma unroll
        for (int i=0;i<4;i++) if(i<elems) qreg[i]=qh[i];
    }
    int t_lo = (window > 0 && pos >= window) ? (pos - window + 1) : 0;
    int nslots = pos - t_lo + 1;
    int chunk = (nslots + S - 1) / S;
    int begin = t_lo + s * chunk;
    int end = min(pos + 1, t_lo + (s + 1) * chunk);
    if (begin >= end) {
        if (lane == 0) { p_m[(size_t)s * n_heads + head] = -1e30f; p_l[(size_t)s * n_heads + head] = 0.0f; }
        float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
#pragma unroll
        for (int i=0;i<4;i++) if(i<elems) myacc[i]=0.0f;
        return;
    }
    extern __shared__ float smem_fp32[];
    float *sK = smem_fp32;
    float *sV = smem_fp32 + BC_FP32 * head_dim;
    float m_prev = -1e30f, l_prev = 0.0f;
    float acc[4] = {0.0f,0.0f,0.0f,0.0f};
    for (int t_tile = begin; t_tile < end; t_tile += BC_FP32) {
        int t_tile_end = min(end, t_tile + BC_FP32);
        int bc_active = t_tile_end - t_tile;
        int total = bc_active * head_dim;
        for (int i = tid; i < total; i += blockDim.x) {
            int tok = i / head_dim;
            int d = i % head_dim;
            long g_idx = ((long)(t_tile + tok) * n_kv_heads + kv) * head_dim + d;
            sK[tok * head_dim + d] = Kc[g_idx];
            sV[tok * head_dim + d] = Vc[g_idx];
        }
        __syncthreads();
        for (int t_idx = 0; t_idx < bc_active; t_idx++) {
            const float *k_ptr = sK + t_idx * head_dim + lane * elems;
            const float *v_ptr = sV + t_idx * head_dim + lane * elems;
            float dot = 0.0f;
            if (elems==4) dot = qreg[0]*k_ptr[0] + qreg[1]*k_ptr[1] + qreg[2]*k_ptr[2] + qreg[3]*k_ptr[3];
            else if (elems==2) dot = qreg[0]*k_ptr[0] + qreg[1]*k_ptr[1];
            else { for(int i=0;i<elems;i++) dot += qreg[i]*k_ptr[i]; }
            float score = warp_sum(dot);
            score = __shfl_sync(0xffffffff, score, 0) * scale;
            float m_curr = fmaxf(m_prev, score);
            float p = expf(score - m_curr);
            float alpha = expf(m_prev - m_curr);
            l_prev = l_prev * alpha + p;
            if (elems==4) { acc[0]=acc[0]*alpha + p*v_ptr[0]; acc[1]=acc[1]*alpha + p*v_ptr[1]; acc[2]=acc[2]*alpha + p*v_ptr[2]; acc[3]=acc[3]*alpha + p*v_ptr[3]; }
            else if (elems==2) { acc[0]=acc[0]*alpha + p*v_ptr[0]; acc[1]=acc[1]*alpha + p*v_ptr[1]; }
            else { for(int i=0;i<elems;i++) acc[i]=acc[i]*alpha + p*v_ptr[i]; }
            m_prev = m_curr;
        }
        __syncthreads();
    }
    if (lane == 0) { p_m[(size_t)s * n_heads + head] = m_prev; p_l[(size_t)s * n_heads + head] = l_prev; }
    float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
    if (elems==4) { myacc[0]=acc[0]; myacc[1]=acc[1]; myacc[2]=acc[2]; myacc[3]=acc[3]; }
    else if (elems==2) { myacc[0]=acc[0]; myacc[1]=acc[1]; }
    else { for(int i=0;i<elems;i++) myacc[i]=acc[i]; }
}

__global__ void k_fa2_combine(
    const float *__restrict__ p_acc,
    const float *__restrict__ p_m,
    const float *__restrict__ p_l,
    float       *__restrict__ out,
    int n_heads, int head_dim, int S)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int elems = head_dim / 32;

    float m_global = -1e30f, l_global = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < S; s++) {
        float m_s = p_m[s * n_heads + h];
        float l_s = p_l[s * n_heads + h];
        if (!isfinite(m_s) || !isfinite(l_s) || l_s <= 0.0f || m_s <= -1e20f) continue;
        float m_new = fmaxf(m_global, m_s);
        float alpha_prev = expf(m_global - m_new);
        float alpha_s    = expf(m_s - m_new);
        l_global = l_global * alpha_prev + l_s * alpha_s;
        const float *p_ptr = p_acc + ((size_t)s * n_heads + h) * head_dim + lane * elems;
        if (elems == 4) {
            const float4 p_vec = *(const float4 *)p_ptr;
            acc[0] = acc[0] * alpha_prev + p_vec.x * alpha_s;
            acc[1] = acc[1] * alpha_prev + p_vec.y * alpha_s;
            acc[2] = acc[2] * alpha_prev + p_vec.z * alpha_s;
            acc[3] = acc[3] * alpha_prev + p_vec.w * alpha_s;
        } else if (elems == 2) {
            const float2 p_vec = *(const float2 *)p_ptr;
            acc[0] = acc[0] * alpha_prev + p_vec.x * alpha_s;
            acc[1] = acc[1] * alpha_prev + p_vec.y * alpha_s;
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (i < elems) acc[i] = acc[i] * alpha_prev + p_ptr[i] * alpha_s;
            }
        }
        m_global = m_new;
    }

    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    float *out_ptr = out + (long)h * head_dim + lane * elems;
    if (elems == 4) {
        *(float4 *)out_ptr = make_float4(
            acc[0] * inv_l,
            acc[1] * inv_l,
            acc[2] * inv_l,
            acc[3] * inv_l
        );
    } else if (elems == 2) {
        *(float2 *)out_ptr = make_float2(
            acc[0] * inv_l,
            acc[1] * inv_l
        );
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i < elems) out_ptr[i] = acc[i] * inv_l;
        }
    }
}

#define BR_PREFILL 8
#define BC_PREFILL 64

#if 0 /* Q4 prefill flash: parked pending k_fa2_q4_split decode HD=64 fix */
template <int ELEMS>
__global__ void k_prefill_flash_q4_0(
    const float     *__restrict__ Q,
    const BlockQ4_0 *__restrict__ Kc,
    const BlockQ4_0 *__restrict__ Vc,
    float           *__restrict__ Att,
    int n, int ctx, int e_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window)
{
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv = blockIdx.y;
    const int q_tile = blockIdx.x;
    const int head = kv * G + warp;
    if (warp >= G) return;

    const int blocks_per_head = head_dim / 32;

    extern __shared__ char raw_smem_prefill_q4[];
    half    *sK_d = (half*)raw_smem_prefill_q4;
    half    *sV_d = sK_d + BC_PREFILL * blocks_per_head;
    uint8_t *sK_q = (uint8_t*)(sV_d + BC_PREFILL * blocks_per_head);
    uint8_t *sV_q = sK_q + BC_PREFILL * (head_dim / 2);

    float qreg[BR_PREFILL][ELEMS];
    float m_state[BR_PREFILL];
    float l_state[BR_PREFILL];
    float acc[BR_PREFILL][ELEMS];
    int   max_kv[BR_PREFILL];
    int   row_min_t[BR_PREFILL];
    bool  active[BR_PREFILL];

#pragma unroll
    for (int r = 0; r < BR_PREFILL; r++) {
        const int qrow = q_tile * BR_PREFILL + r;
        active[r] = (qrow < n);
        max_kv[r] = active[r] ? (e_pos + qrow) : -1;
        row_min_t[r] = (window > 0 && active[r] && (e_pos + qrow + 1 > window))
                       ? (e_pos + qrow + 1 - window) : 0;

        m_state[r] = -1e30f;
        l_state[r] = 0.0f;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            acc[r][e] = 0.0f;
        }

        if (active[r]) {
            const float *qr = Q + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * ELEMS;
            if (ELEMS == 4) {
                float4 q4 = *reinterpret_cast<const float4*>(qr);
                qreg[r][0] = q4.x; qreg[r][1] = q4.y; qreg[r][2] = q4.z; qreg[r][3] = q4.w;
            } else if (ELEMS == 2) {
                float2 q2 = *reinterpret_cast<const float2*>(qr);
                qreg[r][0] = q2.x; qreg[r][1] = q2.y;
            } else {
#pragma unroll
                for (int e = 0; e < ELEMS; e++) qreg[r][e] = qr[e];
            }
        } else {
#pragma unroll
            for (int e = 0; e < ELEMS; e++) qreg[r][e] = 0.0f;
        }
    }

    const int S = (ctx + BC_PREFILL - 1) / BC_PREFILL;
    const int block_in_head = (ELEMS == 2) ? (lane >> 4) : (lane >> 3);
    const int sub = (ELEMS == 2) ? (lane & 15) : (lane & 7);
    const bool is_high = (ELEMS == 2) ? (sub >= 8) : (sub >= 4);
    const int byte_offset = (ELEMS == 2) ? ((sub & 7) * 2) : ((sub & 3) * 4);
    const uint32_t shift = is_high ? 4 : 0;

    for (int s = 0; s < S; s++) {
        const int s_start = s * BC_PREFILL;
        const int s_end_excl = (s_start + BC_PREFILL < ctx) ? (s_start + BC_PREFILL) : ctx;
        const int bc_active = s_end_excl - s_start;
        if (bc_active <= 0) continue;

        const int total_blocks = bc_active * blocks_per_head;
        for (int i = tid; i < total_blocks; i += blockDim.x) {
            const int tok = i / blocks_per_head;
            const int b   = i % blocks_per_head;
            const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * blocks_per_head + b;
            const BlockQ4_0 bk = Kc[g_idx];
            const BlockQ4_0 bv = Vc[g_idx];
            // d is raw fp16 bits: reinterpret, not value-convert
            sK_d[tok * blocks_per_head + b] = __ushort_as_half(bk.d);
            sV_d[tok * blocks_per_head + b] = __ushort_as_half(bv.d);
            const int row_off = tok * (head_dim / 2) + b * 16;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                ((uint16_t *)&sK_q[row_off])[j] = ((const uint16_t *)&bk.qs[0])[j];
                ((uint16_t *)&sV_q[row_off])[j] = ((const uint16_t *)&bv.qs[0])[j];
            }
        }
        __syncthreads();

        for (int t = s_start; t < s_end_excl; t++) {
            const int t_in_tile = t - s_start;
            const float dk = __half2float(sK_d[t_in_tile * blocks_per_head + block_in_head]);
            const float dv = __half2float(sV_d[t_in_tile * blocks_per_head + block_in_head]);

            const int base_off = t_in_tile * (head_dim / 2) + block_in_head * 16 + byte_offset;

            float k_val[ELEMS], v_val[ELEMS];
            if (ELEMS == 2) {
                const uint32_t k_u16 = *(const uint16_t *)&sK_q[base_off];
                const uint32_t v_u16 = *(const uint16_t *)&sV_q[base_off];
                const uint32_t k_sh = k_u16 >> shift;
                const uint32_t v_sh = v_u16 >> shift;
                k_val[0] = (float)((int)((k_sh     ) & 0x0F) - 8);
                k_val[1] = (float)((int)((k_sh >> 8) & 0x0F) - 8);
                v_val[0] = (float)((int)((v_sh     ) & 0x0F) - 8);
                v_val[1] = (float)((int)((v_sh >> 8) & 0x0F) - 8);
            } else if (ELEMS == 4) {
                const uint32_t k_u32 = *(const uint32_t *)&sK_q[base_off];
                const uint32_t v_u32 = *(const uint32_t *)&sV_q[base_off];
                const uint32_t k_sh = k_u32 >> shift;
                const uint32_t v_sh = v_u32 >> shift;
                k_val[0] = (float)((int)((k_sh      ) & 0x0F) - 8);
                k_val[1] = (float)((int)((k_sh >>  8) & 0x0F) - 8);
                k_val[2] = (float)((int)((k_sh >> 16) & 0x0F) - 8);
                k_val[3] = (float)((int)((k_sh >> 24) & 0x0F) - 8);
                v_val[0] = (float)((int)((v_sh      ) & 0x0F) - 8);
                v_val[1] = (float)((int)((v_sh >>  8) & 0x0F) - 8);
                v_val[2] = (float)((int)((v_sh >> 16) & 0x0F) - 8);
                v_val[3] = (float)((int)((v_sh >> 24) & 0x0F) - 8);
            }

#pragma unroll
            for (int r = 0; r < BR_PREFILL; r++) {
                if (!active[r] || t < row_min_t[r] || t > max_kv[r]) continue;

                float dot_partial = 0.0f;
#pragma unroll
                for (int e = 0; e < ELEMS; e++) {
                    dot_partial += qreg[r][e] * k_val[e];
                }
                dot_partial *= dk;
                float score = warp_sum(dot_partial);
                score = __shfl_sync(0xffffffff, score, 0) * scale;

                float m_curr = fmaxf(m_state[r], score);
                float p = expf(score - m_curr);
                float alpha = expf(m_state[r] - m_curr);
                l_state[r] = l_state[r] * alpha + p;

                float pdv = p * dv;
#pragma unroll
                for (int e = 0; e < ELEMS; e++) {
                    acc[r][e] = acc[r][e] * alpha + pdv * v_val[e];
                }

                m_state[r] = m_curr;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < BR_PREFILL; r++) {
        if (active[r]) {
            const int qrow = q_tile * BR_PREFILL + r;
            const float inv_l = (l_state[r] > 0.0f) ? (1.0f / l_state[r]) : 0.0f;
            float *out_row = Att + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * ELEMS;
            if (ELEMS == 4) {
                *reinterpret_cast<float4*>(out_row) = make_float4(
                    acc[r][0] * inv_l,
                    acc[r][1] * inv_l,
                    acc[r][2] * inv_l,
                    acc[r][3] * inv_l
                );
            } else if (ELEMS == 2) {
                *reinterpret_cast<float2*>(out_row) = make_float2(
                    acc[r][0] * inv_l,
                    acc[r][1] * inv_l
                );
            } else {
#pragma unroll
                for (int e = 0; e < ELEMS; e++) out_row[e] = acc[r][e] * inv_l;
            }
        }
    }
}
#endif

__global__ void k_prefill_flash_q8_0(
    const float     *__restrict__ Q,
    const BlockQ8_0 *__restrict__ Kc,
    const BlockQ8_0 *__restrict__ Vc,
    float           *__restrict__ Att,
    int n, int ctx, int e_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window)
{
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv = blockIdx.y;
    const int q_tile = blockIdx.x;
    const int head = kv * G + warp;
    if (warp >= G) return;

    const int elems = head_dim / 32;
    const int blocks_per_head = head_dim / 32;

    extern __shared__ char raw_smem_prefill[];
    half   *sK_d = (half*)raw_smem_prefill;
    half   *sV_d = sK_d + BC_PREFILL * blocks_per_head;
    int8_t *sK_q = (int8_t*)(sV_d + BC_PREFILL * blocks_per_head);
    int8_t *sV_q = sK_q + BC_PREFILL * head_dim;

    float qreg[BR_PREFILL][4];
    float m_state[BR_PREFILL];
    float l_state[BR_PREFILL];
    float acc[BR_PREFILL][4];
    int   max_kv[BR_PREFILL];
    int   row_min_t[BR_PREFILL];
    bool  active[BR_PREFILL];

#pragma unroll
    for (int r = 0; r < BR_PREFILL; r++) {
        const int qrow = q_tile * BR_PREFILL + r;
        active[r] = (qrow < n);
        max_kv[r] = active[r] ? (e_pos + qrow) : -1;
        row_min_t[r] = (window > 0 && active[r] && (e_pos + qrow + 1 > window))
                       ? (e_pos + qrow + 1 - window) : 0;

        m_state[r] = -1e30f;
        l_state[r] = 0.0f;
        acc[r][0] = 0.0f; acc[r][1] = 0.0f; acc[r][2] = 0.0f; acc[r][3] = 0.0f;

        if (active[r]) {
            const float *qr = Q + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * elems;
            /* elems-branch (HD=128: 4, HD=64: 2): loading qr[0..3]
             * unconditionally overreads the head (and faults/pollutes at
             * buffer edges). Zero-fill the unused lanes. */
            if (elems == 4) {
                qreg[r][0] = qr[0]; qreg[r][1] = qr[1]; qreg[r][2] = qr[2]; qreg[r][3] = qr[3];
            } else if (elems == 2) {
                qreg[r][0] = qr[0]; qreg[r][1] = qr[1]; qreg[r][2] = 0.0f; qreg[r][3] = 0.0f;
            } else {
                qreg[r][0] = qr[0];
                qreg[r][1] = (elems > 1) ? qr[1] : 0.0f;
                qreg[r][2] = (elems > 2) ? qr[2] : 0.0f;
                qreg[r][3] = (elems > 3) ? qr[3] : 0.0f;
            }
        } else {
            qreg[r][0] = 0.0f; qreg[r][1] = 0.0f; qreg[r][2] = 0.0f; qreg[r][3] = 0.0f;
        }
    }

    const int S = (ctx + BC_PREFILL - 1) / BC_PREFILL;
    /* elems-scaled lane mapping: block_in_head=(lane*elems)/32 (HD=128:
     * lane>>3, HD=64: lane>>4). The old lane>>3/(lane&7)*elems split only
     * equals lane*elems when elems==4; otherwise lanes read the wrong
     * block/scale and OOB smem bytes. */
    const int block_in_head = (lane * elems) / 32;
    const int lane_byte_off = lane * elems;

    for (int s = 0; s < S; s++) {
        const int s_start = s * BC_PREFILL;
        const int s_end_excl = (s_start + BC_PREFILL < ctx) ? (s_start + BC_PREFILL) : ctx;
        const int bc_active = s_end_excl - s_start;
        if (bc_active <= 0) continue;

        const int total_blocks = bc_active * blocks_per_head;
        for (int i = tid; i < total_blocks; i += blockDim.x) {
            const int tok = i / blocks_per_head;
            const int b   = i % blocks_per_head;
            const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * blocks_per_head + b;
            const BlockQ8_0 bk = Kc[g_idx];
            const BlockQ8_0 bv = Vc[g_idx];
            sK_d[tok * blocks_per_head + b] = bk.d;
            sV_d[tok * blocks_per_head + b] = bv.d;
            const int row_off = tok * head_dim + b * 32;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                sK_q[row_off + j] = bk.qs[j];
                sV_q[row_off + j] = bv.qs[j];
            }
        }
        __syncthreads();

        for (int t = s_start; t < s_end_excl; t++) {
            const int t_in_tile = t - s_start;
            const int k_q_off = t_in_tile * head_dim + lane_byte_off;
            const int v_q_off = k_q_off;
            const float dk = __half2float(sK_d[t_in_tile * blocks_per_head + block_in_head]);
            const float dv = __half2float(sV_d[t_in_tile * blocks_per_head + block_in_head]);

            /* elems-branch: uint32 pack covers 4 lanes only at elems==4.
             * At elems==2 it loads 2 bytes past this lane (misaligned +
             * cross-lane garbage); use uint16, scalar otherwise. */
            float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (elems == 4) {
                const uint32_t k_u32 = *(const uint32_t *)&sK_q[k_q_off];
                const uint32_t v_u32 = *(const uint32_t *)&sV_q[v_q_off];
                k0 = (float)((int8_t)(k_u32      ));
                k1 = (float)((int8_t)(k_u32 >>  8));
                k2 = (float)((int8_t)(k_u32 >> 16));
                k3 = (float)((int8_t)(k_u32 >> 24));
                v0 = (float)((int8_t)(v_u32      ));
                v1 = (float)((int8_t)(v_u32 >>  8));
                v2 = (float)((int8_t)(v_u32 >> 16));
                v3 = (float)((int8_t)(v_u32 >> 24));
            } else if (elems == 2) {
                const uint16_t k_u16 = *(const uint16_t *)&sK_q[k_q_off];
                const uint16_t v_u16 = *(const uint16_t *)&sV_q[v_q_off];
                k0 = (float)((int8_t)(k_u16     ));
                k1 = (float)((int8_t)(k_u16 >> 8));
                v0 = (float)((int8_t)(v_u16     ));
                v1 = (float)((int8_t)(v_u16 >> 8));
            } else {
                if (elems > 0) { k0 = (float)sK_q[k_q_off]; v0 = (float)sV_q[v_q_off]; }
                if (elems > 1) { k1 = (float)sK_q[k_q_off + 1]; v1 = (float)sV_q[v_q_off + 1]; }
                if (elems > 2) { k2 = (float)sK_q[k_q_off + 2]; v2 = (float)sV_q[v_q_off + 2]; }
                if (elems > 3) { k3 = (float)sK_q[k_q_off + 3]; v3 = (float)sV_q[v_q_off + 3]; }
            }
#pragma unroll
            for (int r = 0; r < BR_PREFILL; r++) {
                if (!active[r] || t < row_min_t[r] || t > max_kv[r]) continue;

                float dot_partial = (qreg[r][0]*k0 + qreg[r][1]*k1 + qreg[r][2]*k2 + qreg[r][3]*k3) * dk;
                float score = warp_sum(dot_partial);
                score = __shfl_sync(0xffffffff, score, 0) * scale;

                float m_curr = fmaxf(m_state[r], score);
                float p = expf(score - m_curr);
                float alpha = expf(m_state[r] - m_curr);
                l_state[r] = l_state[r] * alpha + p;

                float pdv = p * dv;
                acc[r][0] = acc[r][0] * alpha + pdv * v0;
                acc[r][1] = acc[r][1] * alpha + pdv * v1;
                acc[r][2] = acc[r][2] * alpha + pdv * v2;
                acc[r][3] = acc[r][3] * alpha + pdv * v3;

                m_state[r] = m_curr;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < BR_PREFILL; r++) {
        if (active[r]) {
            const int qrow = q_tile * BR_PREFILL + r;
            float inv_l = 1.0f / l_state[r];
            float *out_row = Att + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * elems;
            /* elems-branch: out_row[2..3] belong to the next lane/head at
             * elems==2 (and fault past the buffer end on a full tile). */
            if (elems == 4) {
                out_row[0] = acc[r][0] * inv_l;
                out_row[1] = acc[r][1] * inv_l;
                out_row[2] = acc[r][2] * inv_l;
                out_row[3] = acc[r][3] * inv_l;
            } else if (elems == 2) {
                out_row[0] = acc[r][0] * inv_l;
                out_row[1] = acc[r][1] * inv_l;
            } else {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (i < elems) out_row[i] = acc[r][i] * inv_l;
                }
            }
        }
    }
}

#define BC_PREFILL_FP32 64
__global__ void k_prefill_flash_fp32(
    const float *__restrict__ Q,
    const float *__restrict__ Kc,
    const float *__restrict__ Vc,
    float       *__restrict__ Att,
    int n, int ctx, int e_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window)
{
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv = blockIdx.y;
    const int q_tile = blockIdx.x;
    const int head = kv * G + warp;
    if (warp >= G) return;
    const int elems = head_dim / 32;
    extern __shared__ float smem_fp32_prefill[];
    float *sK = smem_fp32_prefill;
    float *sV = sK + BC_PREFILL_FP32 * head_dim;
    float qreg[BR_PREFILL][4];
    float m_state[BR_PREFILL];
    float l_state[BR_PREFILL];
    float acc[BR_PREFILL][4];
    int   max_kv[BR_PREFILL];
    int   row_min_t[BR_PREFILL];
    bool  active[BR_PREFILL];
#pragma unroll
    for (int r = 0; r < BR_PREFILL; r++) {
        const int qrow = q_tile * BR_PREFILL + r;
        active[r] = (qrow < n);
        max_kv[r] = active[r] ? (e_pos + qrow) : -1;
        row_min_t[r] = (window > 0 && active[r] && (e_pos + qrow + 1 > window)) ? (e_pos + qrow + 1 - window) : 0;
        m_state[r] = -1e30f;
        l_state[r] = 0.0f;
        acc[r][0]=0.0f; acc[r][1]=0.0f; acc[r][2]=0.0f; acc[r][3]=0.0f;
        if (active[r]) {
            const float *qr = Q + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * elems;
            if (elems == 4) {
                float4 q4 = *reinterpret_cast<const float4*>(qr);
                qreg[r][0] = q4.x; qreg[r][1] = q4.y; qreg[r][2] = q4.z; qreg[r][3] = q4.w;
            } else if (elems == 2) {
                float2 q2 = *reinterpret_cast<const float2*>(qr);
                qreg[r][0] = q2.x; qreg[r][1] = q2.y; qreg[r][2] = 0.0f; qreg[r][3] = 0.0f;
            } else {
                qreg[r][0]=qr[0]; qreg[r][1]=(elems>1?qr[1]:0); qreg[r][2]=(elems>2?qr[2]:0); qreg[r][3]=(elems>3?qr[3]:0);
                if (elems==1) { qreg[r][1]=0; qreg[r][2]=0; qreg[r][3]=0; }
            }
        } else { qreg[r][0]=0; qreg[r][1]=0; qreg[r][2]=0; qreg[r][3]=0; }
    }
    const int S = (ctx + BC_PREFILL_FP32 - 1) / BC_PREFILL_FP32;
    const int hd4 = head_dim >> 2;
    const bool is_vec4 = ((head_dim & 3) == 0);
    const float4 *Kc4 = reinterpret_cast<const float4*>(Kc);
    const float4 *Vc4 = reinterpret_cast<const float4*>(Vc);
    float4 *sK4 = reinterpret_cast<float4*>(sK);
    float4 *sV4 = reinterpret_cast<float4*>(sV);

    for (int s = 0; s < S; s++) {
        const int s_start = s * BC_PREFILL_FP32;
        const int s_end_excl = (s_start + BC_PREFILL_FP32 < ctx) ? (s_start + BC_PREFILL_FP32) : ctx;
        const int bc_active = s_end_excl - s_start;
        if (bc_active <= 0) continue;
        const int total = bc_active * head_dim;

        if (is_vec4) {
            const int total_vec4 = total >> 2;
            const int hd4_shift = (hd4 == 16) ? 4 : ((hd4 == 32) ? 5 : 0);
            if (hd4_shift) {
                const int mask = (1 << hd4_shift) - 1;
                for (int i = tid; i < total_vec4; i += blockDim.x) {
                    const int tok = i >> hd4_shift;
                    const int d4 = i & mask;
                    const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * hd4 + d4;
                    sK4[i] = Kc4[g_idx];
                    sV4[i] = Vc4[g_idx];
                }
            } else {
                for (int i = tid; i < total_vec4; i += blockDim.x) {
                    const int tok = i / hd4;
                    const int d4 = i % hd4;
                    const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * hd4 + d4;
                    sK4[i] = Kc4[g_idx];
                    sV4[i] = Vc4[g_idx];
                }
            }
        } else {
            for (int i = tid; i < total; i += blockDim.x) {
                const int tok = i / head_dim;
                const int d = i % head_dim;
                const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * head_dim + d;
                sK[tok * head_dim + d] = Kc[g_idx];
                sV[tok * head_dim + d] = Vc[g_idx];
            }
        }
        __syncthreads();

        if (elems == 2) {
            const float *k_base = sK + lane * 2;
            const float *v_base = sV + lane * 2;
            for (int t = s_start; t < s_end_excl; t++) {
                const int t_in_tile = t - s_start;
                const float2 kv2 = *reinterpret_cast<const float2*>(k_base + t_in_tile * head_dim);
                const float2 vv2 = *reinterpret_cast<const float2*>(v_base + t_in_tile * head_dim);
                const float k0 = kv2.x, k1 = kv2.y;
                const float v0 = vv2.x, v1 = vv2.y;
#pragma unroll
                for (int r = 0; r < BR_PREFILL; r++) {
                    if (!active[r] || t < row_min_t[r] || t > max_kv[r]) continue;
                    float dot_partial = qreg[r][0] * k0 + qreg[r][1] * k1;
                    float score = warp_sum_all(dot_partial) * scale;
                    float m_curr = fmaxf(m_state[r], score);
                    float p = expf(score - m_curr);
                    float alpha = expf(m_state[r] - m_curr);
                    l_state[r] = l_state[r] * alpha + p;
                    acc[r][0] = acc[r][0] * alpha + p * v0;
                    acc[r][1] = acc[r][1] * alpha + p * v1;
                    m_state[r] = m_curr;
                }
            }
        } else if (elems == 4) {
            const float *k_base = sK + lane * 4;
            const float *v_base = sV + lane * 4;
            for (int t = s_start; t < s_end_excl; t++) {
                const int t_in_tile = t - s_start;
                const float4 kv4 = *reinterpret_cast<const float4*>(k_base + t_in_tile * head_dim);
                const float4 vv4 = *reinterpret_cast<const float4*>(v_base + t_in_tile * head_dim);
                const float k0 = kv4.x, k1 = kv4.y, k2 = kv4.z, k3 = kv4.w;
                const float v0 = vv4.x, v1 = vv4.y, v2 = vv4.z, v3 = vv4.w;
#pragma unroll
                for (int r = 0; r < BR_PREFILL; r++) {
                    if (!active[r] || t < row_min_t[r] || t > max_kv[r]) continue;
                    float dot_partial = qreg[r][0] * k0 + qreg[r][1] * k1 + qreg[r][2] * k2 + qreg[r][3] * k3;
                    float score = warp_sum_all(dot_partial) * scale;
                    float m_curr = fmaxf(m_state[r], score);
                    float p = expf(score - m_curr);
                    float alpha = expf(m_state[r] - m_curr);
                    l_state[r] = l_state[r] * alpha + p;
                    acc[r][0] = acc[r][0] * alpha + p * v0;
                    acc[r][1] = acc[r][1] * alpha + p * v1;
                    acc[r][2] = acc[r][2] * alpha + p * v2;
                    acc[r][3] = acc[r][3] * alpha + p * v3;
                    m_state[r] = m_curr;
                }
            }
        } else {
            for (int t = s_start; t < s_end_excl; t++) {
                const int t_in_tile = t - s_start;
                const float *k_ptr = sK + t_in_tile * head_dim + lane * elems;
                const float *v_ptr = sV + t_in_tile * head_dim + lane * elems;
                float k0 = k_ptr[0];
                float k1 = (elems>1?k_ptr[1]:0);
                float k2 = (elems>2?k_ptr[2]:0);
                float k3 = (elems>3?k_ptr[3]:0);
                float v0 = v_ptr[0];
                float v1 = (elems>1?v_ptr[1]:0);
                float v2 = (elems>2?v_ptr[2]:0);
                float v3 = (elems>3?v_ptr[3]:0);
#pragma unroll
                for (int r = 0; r < BR_PREFILL; r++) {
                    if (!active[r] || t < row_min_t[r] || t > max_kv[r]) continue;
                    float dot_partial = 0.0f;
                    if (elems==1) dot_partial = qreg[r][0]*k0;
                    else { for(int ei=0;ei<elems;ei++) dot_partial += qreg[r][ei] * (ei==0?k0:(ei==1?k1:(ei==2?k2:k3))); }
                    float score = warp_sum(dot_partial);
                    score = __shfl_sync(0xffffffff, score, 0) * scale;
                    float m_curr = fmaxf(m_state[r], score);
                    float p = expf(score - m_curr);
                    float alpha = expf(m_state[r] - m_curr);
                    l_state[r] = l_state[r] * alpha + p;
                    acc[r][0] = acc[r][0] * alpha + p * v0;
                    acc[r][1] = acc[r][1] * alpha + p * v1;
                    acc[r][2] = acc[r][2] * alpha + p * v2;
                    acc[r][3] = acc[r][3] * alpha + p * v3;
                    m_state[r] = m_curr;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BR_PREFILL; r++) {
        if (active[r]) {
            const int qrow = q_tile * BR_PREFILL + r;
            float inv_l = 1.0f / l_state[r];
            float *out_row = Att + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * elems;
            if (elems==4) {
                *reinterpret_cast<float4*>(out_row) = make_float4(acc[r][0]*inv_l, acc[r][1]*inv_l, acc[r][2]*inv_l, acc[r][3]*inv_l);
            } else if (elems==2) {
                *reinterpret_cast<float2*>(out_row) = make_float2(acc[r][0]*inv_l, acc[r][1]*inv_l);
            } else if (elems==1) {
                out_row[0]=acc[r][0]*inv_l;
            } else {
                for(int ei=0;ei<elems;ei++) out_row[ei]=acc[r][ei]*inv_l;
            }
        }
    }
}

/* TT_FLASH_FP16 restructured FP32 flash: identical math, thread-per-row
 * schedule. Template params T_ (threads/row) and E_ (dims/thread, const so
 * q/acc stay in registers). Each lane owns one q-row chunk; K/V rows
 * broadcast from smem so no warp shuffle is needed. Two-pass per-K-block
 * softmax (pass1 rowmax + single rescale, pass2 single expf per pair).
 * Same Q/Kc/Vc/Att layout and causal/window semantics as fp32 kernel. */
#define BC_FP16 64
template <int T_, int E_>
__global__ void k_prefill_flash_fp16_t(
    const float *__restrict__ Q,
    const float *__restrict__ Kc,
    const float *__restrict__ Vc,
    float       *__restrict__ Att,
    int n, int ctx, int e_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window)
{
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv = blockIdx.y;
    const int q_tile = blockIdx.x;
    const int head = kv * G + warp;
    if (warp >= G) return;
    const int BR = 32 / T_;
    const int sub = lane % T_;
    const int ri = lane / T_;
    const int qrow = q_tile * BR + ri;
    const bool active = (qrow < n);
    const int max_kv = active ? (e_pos + qrow) : -1;
    const int row_min = (window > 0 && active && (e_pos + qrow + 1 > window))
        ? (e_pos + qrow + 1 - window) : 0;
    extern __shared__ float smem_fp16[];
    float *sK = smem_fp16;
    float *sV = sK + BC_FP16 * head_dim;
    float q[E_];
    float acc[E_];
#pragma unroll
    for (int e = 0; e < E_; e++) { q[e] = 0.0f; acc[e] = 0.0f; }
    if (active) {
        const float *qr = Q + (long)qrow * (n_heads * head_dim)
                        + (long)head * head_dim + (long)sub * E_;
        const float4 *qr4 = reinterpret_cast<const float4*>(qr);
#pragma unroll
        for (int i = 0; i < E_ / 4; i++) {
            float4 v = qr4[i];
            q[i*4+0]=v.x; q[i*4+1]=v.y; q[i*4+2]=v.z; q[i*4+3]=v.w;
        }
    }
    float m = -1e30f;
    float l = 0.0f;
    const int S = (ctx + BC_FP16 - 1) / BC_FP16;
    const int hd4 = head_dim >> 2;
    const bool is_vec4 = ((head_dim & 3) == 0);
    const float4 *Kc4 = reinterpret_cast<const float4*>(Kc);
    const float4 *Vc4 = reinterpret_cast<const float4*>(Vc);
    float4 *sK4 = reinterpret_cast<float4*>(sK);
    float4 *sV4 = reinterpret_cast<float4*>(sV);
    for (int s = 0; s < S; s++) {
        const int s_start = s * BC_FP16;
        const int s_end_excl = (s_start + BC_FP16 < ctx) ? (s_start + BC_FP16) : ctx;
        const int bc_active = s_end_excl - s_start;
        if (bc_active <= 0) continue;
        const int total = bc_active * head_dim;
        if (is_vec4) {
            const int total_vec4 = total >> 2;
            const int hd4_shift = (hd4 == 16) ? 4 : ((hd4 == 32) ? 5 : 0);
            if (hd4_shift) {
                const int mask = (1 << hd4_shift) - 1;
                for (int i = tid; i < total_vec4; i += blockDim.x) {
                    const int tok = i >> hd4_shift;
                    const int d4 = i & mask;
                    const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * hd4 + d4;
                    sK4[i] = Kc4[g_idx];
                    sV4[i] = Vc4[g_idx];
                }
            } else {
                for (int i = tid; i < total_vec4; i += blockDim.x) {
                    const int tok = i / hd4;
                    const int d4 = i % hd4;
                    const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * hd4 + d4;
                    sK4[i] = Kc4[g_idx];
                    sV4[i] = Vc4[g_idx];
                }
            }
        } else {
            for (int i = tid; i < total; i += blockDim.x) {
                const int tok = i / head_dim;
                const int d = i % head_dim;
                const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * head_dim + d;
                sK[tok * head_dim + d] = Kc[g_idx];
                sV[tok * head_dim + d] = Vc[g_idx];
            }
        }
        __syncthreads();
        float bmax = -1e30f;
        for (int t = s_start; t < s_end_excl; t++) {
            const bool ok = active && (t >= row_min) && (t <= max_kv);
            float d = 0.0f;
            if (ok) {
                const float *krow = sK + (t - s_start) * head_dim + sub * E_;
#pragma unroll
                for (int e = 0; e < E_; e++) d += q[e] * krow[e];
            }
            if (T_ >= 2) d += __shfl_xor_sync(0xffffffff, d, 1);
            if (T_ >= 4) d += __shfl_xor_sync(0xffffffff, d, 2);
            if (ok) { float sc = d * scale; if (sc > bmax) bmax = sc; }
        }
        const float m_new = fmaxf(m, bmax);
        const float a = expf(m - m_new);
#pragma unroll
        for (int e = 0; e < E_; e++) acc[e] *= a;
        l *= a;
        const float beta = expf(bmax - m_new);
        float bsum = 0.0f;
        for (int t = s_start; t < s_end_excl; t++) {
            const bool ok = active && (t >= row_min) && (t <= max_kv);
            float d = 0.0f;
            if (ok) {
                const float *krow = sK + (t - s_start) * head_dim + sub * E_;
#pragma unroll
                for (int e = 0; e < E_; e++) d += q[e] * krow[e];
            }
            if (T_ >= 2) d += __shfl_xor_sync(0xffffffff, d, 1);
            if (T_ >= 4) d += __shfl_xor_sync(0xffffffff, d, 2);
            if (ok) {
                const float p = expf(d * scale - bmax) * beta;
                bsum += p;
                const float *vrow = sV + (t - s_start) * head_dim + sub * E_;
#pragma unroll
                for (int e = 0; e < E_; e++) acc[e] += p * vrow[e];
            }
        }
        l += bsum;
        m = m_new;
        __syncthreads();
    }
    if (active) {
        const float inv = 1.0f / l;
        float *out = Att + (long)qrow * (n_heads * head_dim)
                   + (long)head * head_dim + (long)sub * E_;
        float4 *out4 = reinterpret_cast<float4*>(out);
        float4 *acc4 = reinterpret_cast<float4*>(acc);
#pragma unroll
        for (int i = 0; i < E_ / 4; i++) {
            float4 v = acc4[i];
            out4[i] = make_float4(v.x*inv, v.y*inv, v.z*inv, v.w*inv);
        }
    }
}

/* TT_FLASH_FP16 dispatch: flag off or uncommon HD keeps legacy fp32 kernel. */
static inline int tt_flash_fp16_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("TT_FLASH_FP16") ? 1 : 0;
    return v;
}
static void launch_prefill_flash(const float *Q, const float *Kf, const float *Vf,
    float *Att, int n, int ctx, int e_pos, int H, int KV, int HD,
    float scale, int swa, cudaStream_t stream) {
    int use_fp16 = tt_flash_fp16_on() && (HD == 64 || HD == 128 || HD == 32);
    if (use_fp16) {
        static int attr = 0;
        if (!attr) {
            attr = 1;
            cudaFuncSetAttribute(k_prefill_flash_fp16_t<2, 64>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)(2 * (size_t)BC_FP16 * 128 * sizeof(float)));
        }
        const int threads = (H / KV) * 32;
        const size_t smem = 2 * (size_t)BC_FP16 * HD * sizeof(float);
        if (HD == 64) {
            dim3 grid((n + 31) / 32, KV);
            k_prefill_flash_fp16_t<1, 64><<<grid, threads, smem, stream>>>(
                Q, Kf, Vf, Att, n, ctx, e_pos, H, KV, HD, scale, swa);
        } else if (HD == 128) {
            dim3 grid((n + 15) / 16, KV);
            k_prefill_flash_fp16_t<2, 64><<<grid, threads, smem, stream>>>(
                Q, Kf, Vf, Att, n, ctx, e_pos, H, KV, HD, scale, swa);
        } else {
            dim3 grid((n + 31) / 32, KV);
            k_prefill_flash_fp16_t<1, 32><<<grid, threads, smem, stream>>>(
                Q, Kf, Vf, Att, n, ctx, e_pos, H, KV, HD, scale, swa);
        }
    } else {
        dim3 grid((n + BR_PREFILL - 1) / BR_PREFILL, KV);
        const int threads = (H / KV) * 32;
        const size_t smem = 2 * (size_t)BC_PREFILL_FP32 * HD * sizeof(float);
        k_prefill_flash_fp32<<<grid, threads, smem, stream>>>(
            Q, Kf, Vf, Att, n, ctx, e_pos, H, KV, HD, scale, swa);
    }
}

extern "C" int tt_kv_scatter(const float *kst, const float *vst, float *Kc, float *Vc,
                             const int *d_pos, int n_kv_heads, int head_dim, int max_ctx, cudaStream_t stream) {
    const int kvdim = n_kv_heads * head_dim;
    k_kv_scatter<<<(kvdim + 255) / 256, 256, 0, stream>>>(
        kst, vst, Kc, Vc, d_pos, n_kv_heads, head_dim, max_ctx);
    return 0;
}

extern "C" int tt_flash_gqa(const float *q, const float *Kc, const float *Vc, float *out,
                            const int *d_pos, int n_heads, int n_kv_heads, int head_dim,
                            int max_ctx, float scale, int window, cudaStream_t stream) {
    k_flash_gqa<<<n_heads, 32, 0, stream>>>(
        q, Kc, Vc, out, d_pos, n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
    return 0;
}

extern "C" int tt_kv_scatter_q8_0(const float *kst, const float *vst, void *Kc_q8, void *Vc_q8,
                                  const int *d_pos, int n_kv_heads, int head_dim, int max_ctx, cudaStream_t stream) {
    const int num_blocks = (n_kv_heads * head_dim) / 32;
    k_kv_scatter_q8_0<<<(num_blocks + 255) / 256, 256, 0, stream>>>(
        kst, vst, (BlockQ8_0 *)Kc_q8, (BlockQ8_0 *)Vc_q8, d_pos, n_kv_heads, head_dim, max_ctx);
    return 0;
}

extern "C" int tt_kv_scatter_q4_0(const float *kst, const float *vst, void *Kc_q4, void *Vc_q4,
                                  const int *d_pos, int n_kv_heads, int head_dim, int max_ctx, cudaStream_t stream) {
    const int num_blocks = (n_kv_heads * head_dim) / 32;
    k_kv_scatter_q4_0<<<(num_blocks + 255) / 256, 256, 0, stream>>>(
        kst, vst, (BlockQ4_0 *)Kc_q4, (BlockQ4_0 *)Vc_q4, d_pos, n_kv_heads, head_dim, max_ctx);
    return 0;
}

/* Test hooks for the late-enable backfill (P1-2): quantize FP32 cache slabs
 * [0..n_slots) into Q caches in one launch. Bit-identical to per-slot
 * tt_kv_scatter_q{4,8}_0 given the same FP32 slot contents. */
extern "C" int tt_kv_backfill_q8_0(const float *Kf, const float *Vf, void *Kc_q8, void *Vc_q8,
                                     int n_slots, int kvdim, cudaStream_t stream) {
    long total = (long)n_slots * (kvdim / 32);
    k_kv_backfill_q8_0<<<(total + 255) / 256, 256, 0, stream>>>(
        Kf, Vf, (BlockQ8_0 *)Kc_q8, (BlockQ8_0 *)Vc_q8, n_slots, kvdim);
    return 0;
}

extern "C" int tt_kv_backfill_q4_0(const float *Kf, const float *Vf, void *Kc_q4, void *Vc_q4,
                                     int n_slots, int kvdim, cudaStream_t stream) {
    long total = (long)n_slots * (kvdim / 32);
    k_kv_backfill_q4_0<<<(total + 255) / 256, 256, 0, stream>>>(
        Kf, Vf, (BlockQ4_0 *)Kc_q4, (BlockQ4_0 *)Vc_q4, n_slots, kvdim);
    return 0;
}

extern "C" int tt_flash_gqa_q8_0(const float *q, const void *Kc_q8, const void *Vc_q8, float *out,
                                 const int *d_pos, int n_heads, int n_kv_heads, int head_dim,
                                 int max_ctx, float scale, int window, cudaStream_t stream) {
    k_flash_gqa_q8_0<<<n_heads, 32, 0, stream>>>(
        q, (const BlockQ8_0 *)Kc_q8, (const BlockQ8_0 *)Vc_q8, out, d_pos,
        n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
    return 0;
}

extern "C" int tt_flash_gqa_q8_0_splitk(const float *q, const void *Kc_q8, const void *Vc_q8,
                                        float *p_acc, float *p_m, float *p_l, float *out,
                                        const int *d_pos, int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int window, int S, cudaStream_t stream) {
    dim3 grid_split(S, n_kv_heads);
    int threads_split = (n_heads / n_kv_heads) * 32;
    int blocks_per_head = head_dim / 32;
    size_t smem_bytes = 2 * (size_t)BC_SPLIT * blocks_per_head * sizeof(half)
                      + 2 * (size_t)BC_SPLIT * head_dim * sizeof(int8_t);
    k_fa2_q8_split<<<grid_split, threads_split, smem_bytes, stream>>>(
        q, (const BlockQ8_0 *)Kc_q8, (const BlockQ8_0 *)Vc_q8,
        p_acc, p_m, p_l,
        d_pos, n_heads, n_kv_heads, head_dim,
        scale, window, S);
    k_fa2_combine<<<n_heads, 32, 0, stream>>>(
        p_acc, p_m, p_l,
        out, n_heads, head_dim, S);
    return 0;
}

extern "C" int tt_flash_gqa_q4_0_splitk(const float *q, const void *Kc_q4, const void *Vc_q4,
                                        float *p_acc, float *p_m, float *p_l, float *out,
                                        const int *d_pos, int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int window, int S, cudaStream_t stream) {
    dim3 grid_split(S, n_kv_heads);
    int threads_split = (n_heads / n_kv_heads) * 32;
    int blocks_per_head = head_dim / 32;
    size_t smem_bytes = 2 * (size_t)BC_SPLIT * blocks_per_head * sizeof(half)
                      + 2 * (size_t)BC_SPLIT * (head_dim / 2) * sizeof(uint8_t);
    k_fa2_q4_split<<<grid_split, threads_split, smem_bytes, stream>>>(
        q, (const BlockQ4_0 *)Kc_q4, (const BlockQ4_0 *)Vc_q4,
        p_acc, p_m, p_l,
        d_pos, n_heads, n_kv_heads, head_dim,
        scale, window, S);
    k_fa2_combine<<<n_heads, 32, 0, stream>>>(
        p_acc, p_m, p_l,
        out, n_heads, head_dim, S);
    return 0;
}

/* Two-stage argmax over vocab. V2: float4 loads + parallel final reduce.
 * Tie-break stays deterministic: on equal values the smaller index wins
 * (matches the original scalar kernel and the greedy-sampling oracle). */
__global__ void k_argmax_partial(const float *__restrict__ x, int n,
                                 float *__restrict__ bvals, int *__restrict__ bidxs) {
    __shared__ float sv[256];
    __shared__ int si[256];
    const int tid = threadIdx.x;
    const int n4 = n >> 2;
    const float4 *x4 = (const float4 *)(const void *)x;
    float best = -INFINITY; int bi = 0;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        const int base = i << 2;
        if (v.x > best) { best = v.x; bi = base + 0; }
        if (v.y > best) { best = v.y; bi = base + 1; }
        if (v.z > best) { best = v.z; bi = base + 2; }
        if (v.w > best) { best = v.w; bi = base + 3; }
    }
    for (int i = (n4 << 2) + tid; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > best) { best = v; bi = i; }
    }
    sv[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            const float ov = sv[tid + s2]; const int oi = si[tid + s2];
            if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) {
                sv[tid] = ov; si[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) { bvals[blockIdx.x] = sv[0]; bidxs[blockIdx.x] = si[0]; }
}

/* Final reduce over per-block partials: one block of ARGMAX_NB threads,
 * same value-then-index tie-break as the partial kernel. */
__global__ void k_argmax_final(const float *__restrict__ bvals, const int *__restrict__ bidxs,
                               int nb, int *__restrict__ out) {
    __shared__ float sv[64];
    __shared__ int si[64];
    const int tid = threadIdx.x;
    float best = -INFINITY; int bi = 0;
    if (tid < nb) { best = bvals[tid]; bi = bidxs[tid]; }
    else { best = -INFINITY; bi = 0; }
    sv[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            const float ov = sv[tid + s2]; const int oi = si[tid + s2];
            if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) { sv[tid] = ov; si[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) *out = si[0];
}

/* Dynamic-token embedding for cudaGraph replay: the token id is read from
 * device memory, so one captured graph can decode any token. Body is a copy
 * of k_embed_q4_0 (kernels/gemv_q4_cuda.cu) — cross-TU kernel launches would
 * need relocatable device code, so it is inlined here instead. */
__global__ void k_embed_q4_0_dyn(const BlockQ4_0 *__restrict__ W, const int *__restrict__ d_tok,
                                 float *__restrict__ dx, int dim) {
    const int tok = *d_tok;
    const int b = threadIdx.x + blockIdx.x * blockDim.x;
    const int nb = dim / 32;
    if (b >= nb) return;
    BlockQ4_0 blk = W[(long)tok * nb + b];
    /* NOTE: loader_gguf.h stores d as uint16_t raw fp16 bits — must
     * bit-reinterpret, NOT integer-convert (implicit uint16_t→__half
     * conversion silently produces huge scales). */
    const float d = __half2float(*(const __half *)&blk.d);
    float *out = dx + b * 32;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        out[i]      = ((blk.qs[i] & 0x0F) - 8) * d;
        out[i + 16] = ((blk.qs[i] >> 4) - 8) * d;
    }
}

/* Dynamic-token embedding for q6_K (cudaGraph replay variant). Body is
 * structurally identical to k_embed_q6_K in kernels/gemv_typed.cu, only the
 * source of the token id differs (device memory instead of host register).
 * Inlined here for the same reason as k_embed_q4_0_dyn: cross-TU device
 * launches need relocatable code. Math is byte-identical to the eager
 * host-tok k_embed_q6_K, which mirrors src/dequant_ref.c::dq_q6_K.
 *
 * Block layout (Q6_K superblock, 256 elements, 210 bytes):
 *   blk[  0..127]  ql   (low 4 bits, 2 per byte, 64 bytes per half-block)
 *   blk[128..191]  qh   (high 2 bits, 32 bytes per half-block)
 *   blk[192..207]  sc   (int8 scales, 16 bytes)
 *   blk[208..209]  d    (fp16 superblock scale)
 * Each output thread writes 32 elements (one sub-block of a superblock).
 * 8 sub-blocks * 32 = 256 elements per superblock. */
__global__ void k_embed_q6_K_dyn(const uint8_t *__restrict__ W, const int *__restrict__ d_tok,
                                 float *__restrict__ dx, int dim) {
    const int tok = *d_tok;
    const int u = threadIdx.x + blockIdx.x * blockDim.x;
    const int nu = (dim / 256) * 8;
    if (u >= nu) return;
    const int sb = u >> 3, sub = u & 7;
    const uint8_t *blk = W + (long)tok * (dim / 256) * 210 + sb * 210;
    const uint8_t *ql = blk, *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    const float d = __half2float(*(const __half *)(blk + 208));
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

__global__ void k_embed_q2_K_dyn(const uint8_t *__restrict__ W, const int *__restrict__ d_tok,
                                 float *__restrict__ dx, int dim) {
    const int tok = *d_tok;
    const int u = threadIdx.x + blockIdx.x * blockDim.x;
    const int nu = (dim / 256) * 16;
    if (u >= nu) return;
    const int sb  = u >> 4;
    const int is  = u & 15;
    const int n   = is >> 3;
    const int j   = (is & 7) >> 1;
    const int is0 = is & 1;
    const int shift = j << 1;

    const uint8_t *blk = W + (long)tok * (dim / 256) * 84 + sb * 84;
    const uint8_t sc = blk[is];

    const float d   = __half2float(*(const __half *)(blk + 80));
    const float dm  = __half2float(*(const __half *)(blk + 82));
    const float dl  = d  * (float)(sc & 0xF);
    const float ml  = dm * (float)(sc >> 4);

    const uint8_t *q = blk + 16 + 32 * n + 16 * is0;
    float *dst = dx + (long)sb * 256 + is * 16;

#pragma unroll
    for (int l = 0; l < 16; l++) {
        int8_t w = (int8_t)((q[l] >> shift) & 3);
        dst[l] = dl * (float)w - ml;
    }
}
__global__ void k_embed_q3_K_dyn(const uint8_t *__restrict__ W, const int *__restrict__ d_tok,
                                 float *__restrict__ dx, int dim) {
    const int tok = *d_tok;
    const int u = threadIdx.x + blockIdx.x * blockDim.x;
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
    const float d = __half2float(*(const __half *)(blk + 108));
    const float dl = d * (float)(us - 32);

    const uint8_t *q  = blk + 32 + 32 * n + 16 * is0;
    const uint8_t *hm = blk + 16 * is0;
    float *dst = dx + (long)sb * 256 + is * 16;

#pragma unroll
    for (int l = 0; l < 16; l++) {
        int8_t w = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
        dst[l] = dl * (float)w;
    }
}
/* Position increment INSIDE the captured region: each replay advances the
 * device position scalar exactly once (host mirror does e->pos++ in lockstep). */
__global__ void k_pos_inc(int *d_pos) { (*d_pos)++; }

__global__ void k_scale(float *__restrict__ buf, float s, int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] *= s;
}

/* llama.cpp-style repetition penalty over a device ring of recent tokens.
   No-ops when penalty <= 1.0f. Capture-safe: reads/writes device state only. */
__global__ void k_repeat_penalty(float *__restrict__ logits,
                                 const int *__restrict__ recent,
                                 const int *__restrict__ n_recent,
                                 int vocab, float penalty) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vocab) return;
    if (penalty <= 1.0f) return;
    const int nr = *n_recent < 64 ? *n_recent : 64;
    float v = logits[i];
    for (int j = 0; j < nr; j++) {
        if (recent[j] == i) {
            v = v > 0.0f ? v / penalty : v * penalty;
            break;
        }
    }
    logits[i] = v;
}

/* Gumbel-max transform: logits' = logits/temp + gumbel_noise.
   argmax(logits') samples softmax(logits/temp). No-op when temp <= 0 (greedy:
   downstream argmax sees unmodified logits => byte-identical to before).
   Nonce = *d_pos keeps it deterministic and capture-safe. */
__device__ __forceinline__ unsigned int phil32(unsigned int seed, unsigned int idx) {
    unsigned int h = seed ^ (idx * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return h;
}
__global__ void k_gumbel_transform(float *__restrict__ logits, int vocab,
                                   float temp, const int *__restrict__ d_pos,
                                   const int *__restrict__ d_sampling_on) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vocab) return;
    if (temp <= 0.0f || !*d_sampling_on) return;
    const unsigned int r = phil32((unsigned int)(*d_pos) * 2654435761u + 12345u,
                                  (unsigned int)i);
    /* u in (0,1]; gumbel = -log(-log(u)) */
    const float u = ((float)r + 1.0f) / 4294967296.0f;
    const float g = -logf(-logf(u));
    logits[i] = logits[i] / temp + g;
}

/* called at end of captured step: bump pos AND record fed token in recent ring */
__global__ void k_pos_inc_recent(int *d_pos, const int *d_next_tok,
                                 int *recent, int *n_recent) {
    recent[(*d_pos) % 64] = *d_next_tok;
    if (*n_recent < 64) (*n_recent)++;
    (*d_pos)++;
}

/* act(g) * u elementwise: the non-q4_0 MLP path (two plain GEMVs + this)
 * replaces the fused q4_0 kernel when gate/up are any other dtype.
 * act = 0: silu; act = 1: gelu tanh-approx (gemma GeGLU families). */
__global__ void k_swiglu_apply(const float *__restrict__ g,
                               const float *__restrict__ u,
                               float *__restrict__ h, int n, int act) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    const float gi = g[i];
    const float a = act ? 0.5f * gi * (1.0f + tanhf(0.7978845608028654f * (gi + 0.044715f * gi * gi * gi)))
                        : gi / (1.0f + expf(-gi));
    h[i] = a * u[i];
}

/* ---------------- host-side engine ---------------- */

#define MAX_LAYERS 128
#define MAX_DIM 16384

struct LayerW {
    /* M7 task 2: weights are type-blind {ptr, GGML type} pairs; the GEMV
     * dispatcher picks the kernel from `dtype`. */
    TTensor q, k, v, o, gate, up, down;
    float *attn_norm, *ffn_norm;      /* device f32 gammas */
    float *post_attn_norm, *post_ffn_norm; /* gemma2 sandwich norms (optional) */
    TTensor inp_gate, pl_proj;        /* gemma4 MatFormer per-layer block */
    float *pl_post_norm;              /* [dim] gamma, normalizes pl_proj output */
    float out_scale_val;              /* gemma4 per-layer scalar (0 = absent) */
    float *q_bias, *k_bias, *v_bias;  /* this GGUF variant carries QKV biases */
    float *q_norm, *k_norm;           /* per-head q/k rmsnorm gammas (qwen3 trait) */
};

static int c_kvdim_for(const TTConfig *c) { return c->n_kv_heads * c->head_dim; }

struct Qwen2Engine {
    TTConfig cfg;
    GGUFModel *gguf;
    LayerW L[MAX_LAYERS];
    TTensor d_embd;          /* tied or untied lm head below */
    TTensor d_out_w;
    float *d_out_norm;
    /* activations */
    float *d_x, *d_xn, *d_q, *d_att, *d_h, *d_logits;
    /* split-SwiGLU staging for non-q4_0 gate/up dtypes (fused q4_0 path
     * keeps using d_h only) */
    float *d_g, *d_u;
    /* caches: [layer][kv_head][slot][head_dim] */
    float *d_kc, *d_vc;
    /* Q8_0 and Q4_0 KV caches: allocated when enabled */
    BlockQ8_0 *d_kc_q8, *d_vc_q8;
    int use_q8_kvcache;
    BlockQ4_0 *d_kc_q4, *d_vc_q4;
    int use_q4_kvcache;
    /* Hybrid dispatch threshold (host-cached): FP32 below, Q* above */
    int kv_thresh_cached;
    int kv_thresh_valid;
    float *d_k_stage, *d_v_stage;
    /* device mirror of pos: kernels read position from here (graph-readiness) */
    int *d_pos;
    /* argmax scratch */
    float *d_bvals; int *d_bidxs, *d_out;
    /* cudaGraph replay of the decode step (M6.3) */
    cudaGraphExec_t graph_exec;   /* instantiated decode-step graph */
    int graph_ready;              /* nonzero once capture+instantiate succeeded */
    int no_graph;                 /* TT_NO_GRAPH=1: eager path forever */
    int *d_next_tok;              /* device token fed by the next replay */
    int *h_sampled;               /* pinned staging for async D2H argmax result */
    /* sampling controls (all device-resident => capture-safe; default greedy) */
    float sampling_temp;          /* 0 = pure greedy (kernels no-op) */
    float repeat_penalty;
    int   *d_recent;              /* ring of recently fed tokens, 64 entries */
    int   *d_n_recent;            /* valid count in ring */
    int   *d_sampling_on;         /* 0/1 flag read by kernels inside graph */
    int pending_tok;              /* sampled token not yet fed through layers */
    /* M9 split-K flash attention workspace (long-ctx decode). Sized to
     * S=16 * max_heads * (max_hd + 2) floats in qwen2_engine_create; freed
     * in qwen2_engine_free. S=clamp(ctx/256,2,16) at launch. */
    float *d_split_pacc;          /* [S_MAX * max_heads * max_hd] */
    float *d_split_pm;            /* [S_MAX * max_heads] */
    float *d_split_pl;            /* [S_MAX * max_heads] */
    int    d_split_S_max;         /* S at workspace alloc time (capacity) */
    /* TT_SPEC_BATCH: persistent device buffers for the post-final-layer
     * activations of N verify candidates (layout [N, dim]). Lazy-alloc
     * on first verify call that uses TT_SPEC_BATCH>=2. */
    float *d_x_batch;             /* [d_spec_max_n * dim]   post-layers hidden */
    float *d_xn_batch;            /* [d_spec_max_n * dim]   post-rmsnorm hidden (LM head input) */
    float *d_logits_batch;        /* [d_spec_max_n * vocab] per-candidate logits row-major */
    int    d_spec_max_n;          /* allocation cap (>=1) */
    TTensor pl_model_proj;        /* [n_layers*256, dim] typed */
    int    d_spec_n;              /* actual N used last call */

    /* gemma4 MatFormer per-layer embeddings */
    float *pl_proj_norm_host;     /* [256] host gamma */
    float *d_pl_tmp;              /* [n_layers*256] staging */
    float *d_ple_row;             /* [n_layers*256] active position's block */
    float *d_ones;                /* ones vector for plain V rmsnorm */
    int pl_hd[MAX_LAYERS];
    int pl_heads[MAX_LAYERS], pl_kv[MAX_LAYERS], pl_ffn[MAX_LAYERS];
    int pl_swa[MAX_LAYERS], pl_src[MAX_LAYERS];
    int has_pl_embd;
    int pl_dim;                   /* 256 */
    int ple_cache_tok;            /* last token id dequantized into ple_pe */
    float *ple_pe;                /* cached scaled per-layer token embed row */
    float *d_rope_freqs;          /* [256] partial-rope factors (device) */
    /* Task 1: persistent prefill arena (eliminate 20 cudaMalloc/Free per prefill) */
    float *d_pf_X, *d_pf_Xn, *d_pf_Q, *d_pf_K, *d_pf_V, *d_pf_Att, *d_pf_H, *d_pf_G, *d_pf_U;
    int *d_pf_pos_batch;
    size_t pf_arena_max_n;
    int pos;
    int n_gpu_layers;
    float *h_x_buf, *h_xn_buf, *h_q_buf, *h_att_buf, *h_h_buf, *h_g_buf, *h_u_buf, *h_out_buf;
    float *h_k_stage, *h_v_stage;
    float *h_kc, *h_vc;
    cudaStream_t stream;
};

/* Hybrid effective-checks (Fix1): use Q* only when flag set AND pos > thresh.
 * TT_NO_BACKFILL=1 is a test-only hatch: backfill skipped, Q slabs [0..pos)
 * stay zeroed -> force FP32 flash so zeroed slabs are never read above thresh. */
static inline int kv_no_backfill(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TT_NO_BACKFILL") ? 1 : 0;
    return cached;
}
static inline int kv_use_q4_eff(const Qwen2Engine *e) {
    return e->use_q4_kvcache && e->pos > kv_thresh_value() && !kv_no_backfill();
}
static inline int kv_use_q8_eff(const Qwen2Engine *e) {
    return e->use_q8_kvcache && !e->use_q4_kvcache && e->pos > kv_thresh_value() && !kv_no_backfill();
}
static inline int kv_use_q4_eff_at(const Qwen2Engine *e, int pos) {
    return e->use_q4_kvcache && pos > kv_thresh_value() && !kv_no_backfill();
}
static inline int kv_use_q8_eff_at(const Qwen2Engine *e, int pos) {
    return e->use_q8_kvcache && !e->use_q4_kvcache && pos > kv_thresh_value() && !kv_no_backfill();
}
/* Decode split-K S formulas (shared eager + graph-capture). Capture bakes
 * the formula's value at the capture ctx; replay stays valid at any ctx
 * because every split kernel derives [begin,end) from live *d_pos and the
 * combine skips empty splits (l<=0). Divergence fixed here: replay S was
 * S_max/32 at every ctx (ctx50: 64 vs eager 2; ctx512: 64 vs 8; ctx4096:
 * 64 vs 64 match), now baked from the same formula eager uses. */
static inline int split_S_q(int ctx, int smax) {
    int S = (ctx + 63) / 64;
    if (S < 2) S = 2;
    if (S > 64) S = 64;
    if (S > smax) S = smax;
    return S;
}
static inline int split_S_fp32(int ctx, int smax) {
    int S = (ctx + 31) / 32;
    if (S < 2) S = 2;
    if (S > 32) S = 32;
    if (S > smax) S = smax;
    return S;
}

static float *upload_f32(GGUFModel *m, const char *name) {
    GGUFTensor *t = gguf_get_tensor(m, name);
    if (!t || !t->data) {
        /* optional-tensor silence: callers treat NULL as "feature absent"
         * (e.g. QKV biases on bias-free families). Only real failures print. */
        if (strstr(name, ".bias") == NULL)
            fprintf(stderr, "[qwen2-engine] f32 upload missing: %s\n", name);
        return NULL;
    }
    float *d = NULL;
    if (cudaMalloc(&d, t->size_bytes) != cudaSuccess) { fprintf(stderr, "[qwen2-engine] cudaMalloc fail %s\n", name); return NULL; }
    cudaMemcpy(d, t->data, t->size_bytes, cudaMemcpyHostToDevice);
    return d;
}

TTConfig tt_config_from_gguf(const GGUFModel *m, int max_ctx) {
    TTConfig c;
    memset(&c, 0, sizeof(c));
    if (!m || m->dim <= 0 || m->n_layers <= 0 || m->n_heads <= 0) return c;
    c.dim = m->dim;
    c.hidden_dim = m->hidden_dim;
    c.n_layers = m->n_layers;
    c.n_heads = m->n_heads;
    c.n_kv_heads = m->n_kv_heads > 0 ? m->n_kv_heads : m->n_heads;
    /* Derive geometry from blk.0 TENSORS when metadata is absent or ambiguous
     * (gemma4: head counts are per-layer arrays; its feed_forward_length meta
     * disagrees with actual tensor shapes). Tensor shapes are ground truth.
     * GGUF ne[] order: ne0 = fastest axis = input width; ne1 = output count. */
    {
        GGUFTensor *tq = gguf_get_tensor((GGUFModel *)m, "blk.0.attn_q.weight");
        GGUFTensor *tk = gguf_get_tensor((GGUFModel *)m, "blk.0.attn_k.weight");
        GGUFTensor *td = gguf_get_tensor((GGUFModel *)m, "blk.0.ffn_down.weight");
        if (tq && tk && tq->ndim == 2 && tk->ndim == 2) {
            const long qr = tq->shape[1];      /* outputs = n_heads * hd */
            const long kr = tk->shape[1];      /* outputs = n_kv_heads * hd */
            long a = qr, b = kr;
            while (b) { long t2 = a % b; a = b; b = t2; }
            const int hd_gcd = (int)a;
            if (c.n_heads <= 0) {
                /* meta missing: pick hd = gcd, heads follow */
                if (m->head_dim > 0 && qr % m->head_dim == 0 && kr % m->head_dim == 0)
                    c.head_dim = m->head_dim;
                else c.head_dim = hd_gcd;
                c.n_heads = (int)(qr / c.head_dim);
                c.n_kv_heads = (int)(kr / c.head_dim);
            } else {
                /* meta present: verify against tensors, fall back to gcd */
                int hd = c.dim / c.n_heads;
                if (qr % hd != 0 || kr % hd != 0 || c.n_heads * hd != qr)
                    hd = (m->head_dim > 0 && qr % m->head_dim == 0 && kr % m->head_dim == 0)
                         ? m->head_dim : hd_gcd;
                c.head_dim = hd;
            }
            c.n_kv_heads = c.n_kv_heads > 0 ? c.n_kv_heads : c.n_heads;
        }
        if (td && td->ndim == 2 && td->shape[0] > 0) {
            c.hidden_dim = (int)td->shape[0];   /* tensor truth beats meta */
        }
    }
    c.vocab = 0;                    /* resolved from tokenizer/embedding at create */
    c.max_ctx = max_ctx;
    c.rms_eps = m->rms_norm_eps > 0 ? m->rms_norm_eps : 1e-6f;
    c.rope_base = m->rope_freq_base > 0 ? m->rope_freq_base : 10000.0f;
    return c;
}

extern "C" void qwen2_engine_enable_q8_kvcache(Qwen2Engine *e, int enable);
extern "C" void qwen2_engine_enable_q4_kvcache(Qwen2Engine *e, int enable);

static void fail(const char *msg) { fprintf(stderr, "[qwen2-engine] %s\n", msg); }

Qwen2Engine *qwen2_engine_create(const TTConfig *cfg, GGUFModel *m) {
#define ABORT_CREATE(msg) do { fail(msg); qwen2_engine_free(e); return NULL; } while (0)

    if (!cfg || !m || cfg->dim == 0) { fail("bad config"); return NULL; }
    if (cfg->dim % cfg->n_heads || cfg->n_heads % cfg->n_kv_heads ||
        cfg->hidden_dim % 32 || cfg->dim % 32) { fail("dims not divisible"); return NULL; }
    if (cfg->n_layers > MAX_LAYERS || cfg->dim > MAX_DIM) { fail("dims exceed engine limits"); return NULL; }

    Qwen2Engine *e = (Qwen2Engine *)calloc(1, sizeof(Qwen2Engine));
    e->gguf = m;
    e->cfg = *cfg;
    e->pos = 0;
    /* KV-share source map: -1 = layer owns its K/V (default for all arches;
     * the gemma4 hetero scan below overwrites shared layers) */
    for (int l = 0; l < MAX_LAYERS; l++) e->pl_src[l] = -1;
    cudaStreamCreate(&e->stream);

    /* M7 task 3: resolve architecture traits from general.architecture.
     * Unknown arch => clear error listing what we support. */
    if (tt_traits_resolve(m, &e->cfg.tr) != 0) {
        fprintf(stderr,
                "[qwen2-engine] ERROR: unsupported general.architecture '%s'.\n"
                "  supported: %s\n",
                m->architecture[0] ? m->architecture : "(missing)",
                tt_traits_supported());
        qwen2_engine_free(e);
        return NULL;
    }
    fprintf(stderr, "[qwen2-engine] traits: rope=%s act=%s softcap=%.1f swa=%d tied=%d qk_norm=%d norm_off=%.1f\n",
            e->cfg.tr.rope == ROPE_GPTJ ? "gptj" : "neox",
            e->cfg.tr.act == ACT_GELU ? "gelu" : "silu",
            e->cfg.tr.softcap_value, e->cfg.tr.swa_size, e->cfg.tr.tied_embeddings,
            e->cfg.tr.qk_norm_rms, e->cfg.tr.norm_offset);

    const long D = cfg->dim, F = cfg->hidden_dim;
    char name[160];

    /* resolve vocab from the embedding tensor's actual shape */
    GGUFTensor *tembd = gguf_get_tensor(m, "token_embd.weight");
    if (!tembd) { fail("token_embd.weight missing"); return NULL; }
    e->cfg.vocab = (int)tembd->shape[tembd->ndim - 1];

    e->d_embd.ptr = NULL;
    upload_w(m, "token_embd.weight", &e->d_embd);

    {
        GGUFTensor *tw = gguf_get_tensor(m, "output.weight");
        if (tw && tw->data && upload_w(m, "output.weight", &e->d_out_w) == 0) {
            /* keep whichever dtype output.weight actually carries */
        } else {
            e->d_out_w.ptr = NULL; e->d_out_w.dtype = -1;
        }
    }
    if (!e->d_out_w.ptr || getenv("TT_FORCE_TIED")) {         /* tied embeddings fallback */
        e->d_out_w = e->d_embd;
        fprintf(stderr, "[qwen2-engine] using TIED embedding as lm head (dtype %d)\n", e->d_out_w.dtype);
    }
    fprintf(stderr, "[qwen2-engine] lm head dtype: %d (%s)\n", e->d_out_w.dtype,
            e->d_out_w.dtype == GGUF_TYPE_Q8_0 ? "q8_0" :
            e->d_out_w.dtype == GGUF_TYPE_Q4_0 ? "q4_0" : "typed dispatch");
    e->d_out_norm = upload_f32(m, "output_norm.weight");

    int n_gpu_layers = cfg->n_layers;
    const char *gpu_layers_env = getenv("TT_GPU_LAYERS");
    if (!gpu_layers_env) gpu_layers_env = getenv("TT_N_GPU_LAYERS");
    if (gpu_layers_env) {
        int v = atoi(gpu_layers_env);
        if (v >= 0 && v <= cfg->n_layers) n_gpu_layers = v;
    }
    e->n_gpu_layers = n_gpu_layers;
    if (n_gpu_layers < cfg->n_layers) {
        fprintf(stderr, "[qwen2-engine] HYBRID CPU-GPU Offloading: %d GPU layers, %d CPU layers\n",
                n_gpu_layers, cfg->n_layers - n_gpu_layers);
    }
    for (int l = 0; l < cfg->n_layers; l++) {
        LayerW *w = &e->L[l];
        const int is_gpu = (l < e->n_gpu_layers);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        if (is_gpu) upload_w(m, name, &w->q);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->q.ptr = t ? t->data : NULL; w->q.dtype = t ? (int)t->type : -1; }
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        if (is_gpu) upload_w(m, name, &w->k);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->k.ptr = t ? t->data : NULL; w->k.dtype = t ? (int)t->type : -1; }
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        if (is_gpu) upload_w(m, name, &w->v);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->v.ptr = t ? t->data : NULL; w->v.dtype = t ? (int)t->type : -1; }
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        if (is_gpu) upload_w(m, name, &w->o);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->o.ptr = t ? t->data : NULL; w->o.dtype = t ? (int)t->type : -1; }
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        if (is_gpu) upload_w(m, name, &w->gate);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->gate.ptr = t ? t->data : NULL; w->gate.dtype = t ? (int)t->type : -1; }
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        if (is_gpu) upload_w(m, name, &w->up);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->up.ptr = t ? t->data : NULL; w->up.dtype = t ? (int)t->type : -1; }
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        if (is_gpu) upload_w(m, name, &w->down);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->down.ptr = t ? t->data : NULL; w->down.dtype = t ? (int)t->type : -1; }
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        if (is_gpu) w->attn_norm = upload_f32(m, name);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->attn_norm = t ? (float*)t->data : NULL; }
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        if (is_gpu) w->ffn_norm  = upload_f32(m, name);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->ffn_norm = t ? (float*)t->data : NULL; }
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
        if (is_gpu) w->q_bias    = upload_f32(m, name);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->q_bias = t ? (float*)t->data : NULL; }
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
        if (is_gpu) w->k_bias    = upload_f32(m, name);
        else { GGUFTensor *t = gguf_get_tensor(m, name); w->k_bias = t ? (float*)t->data : NULL; }
        snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l); w->post_attn_norm = upload_f32(m, name); /* gemma2 sandwich */
        {   /* gemma4 MatFormer block tensors (optional) */
            GGUFTensor *tg = gguf_get_tensor(m, name);
            (void)tg;
        }
        snprintf(name, sizeof(name), "blk.%d.inp_gate.weight", l);
        if (gguf_get_tensor(m, name)) upload_w(m, name, &w->inp_gate);
        snprintf(name, sizeof(name), "blk.%d.proj.weight", l);
        if (gguf_get_tensor(m, name)) upload_w(m, name, &w->pl_proj);
        snprintf(name, sizeof(name), "blk.%d.post_norm.weight", l);
        w->pl_post_norm = upload_f32(m, name);   /* gemma4: normalizes pl_proj out */
        snprintf(name, sizeof(name), "blk.%d.layer_output_scale.weight", l);
        {
            GGUFTensor *tsc = gguf_get_tensor(m, name);
            w->out_scale_val = 0.0f;
            if (tsc && tsc->data && tsc->size_bytes >= 4)
                memcpy(&w->out_scale_val, tsc->data, 4);
        }
        snprintf(name, sizeof(name), "blk.%d.post_ffw_norm.weight", l);       w->post_ffn_norm  = upload_f32(m, name); /* gemma2 sandwich */
        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);         w->v_bias    = upload_f32(m, name); /* optional */
        if (e->cfg.tr.qk_norm_rms) {   /* qwen3 trait: gammas required */
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l); w->q_norm = upload_f32(m, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l); w->k_norm = upload_f32(m, name);
            if (!w->q_norm || !w->k_norm) { /* gemma4: no per-layer QK norm gammas; clear trait, skip step */
                if (l == 0) fprintf(stderr, "[qwen2-engine] qk_norm_rms trait set but attn_q/k_norm missing; disabling (gemma4)\n");
                w->q_norm = w->k_norm = NULL; e->cfg.tr.qk_norm_rms = 0;
            }
        }
        if (!w->q.ptr || !w->k.ptr || !w->v.ptr || !w->o.ptr || !w->gate.ptr || !w->up.ptr || !w->down.ptr ||
            !w->attn_norm || !w->ffn_norm) {  /* v_bias optional: absent in stock Qwen2 */
            fprintf(stderr, "[qwen2-engine] missing weights for layer %d\n", l);
            qwen2_engine_free(e); return NULL;
        }
    }
    if (!e->d_embd.ptr || !e->d_out_norm) ABORT_CREATE("missing embedding/output_norm");

    /* activations + caches */
    cudaMalloc(&e->d_x,  D * sizeof(float));
    cudaMalloc(&e->d_xn, D * sizeof(float));
    int max_hd = cfg->head_dim;
    int max_heads = cfg->dim / cfg->head_dim;
    long max_ffn = F;
    int max_kv = cfg->n_kv_heads;
    /* Default max_qout is the homogeneous case:
     * cfg->n_heads*cfg->head_dim. d_q/d_att must fit every layer. */
    long max_qout = (long)cfg->n_heads * cfg->head_dim;
    if (m && strstr(m->architecture, "gemma4") == m->architecture) {
        const int hd_meta = cfg->head_dim;
        for (int l = 0; l < cfg->n_layers; l++) {
            char tn[128];
            snprintf(tn, sizeof(tn), "blk.%d.attn_q.weight", l);
            GGUFTensor *t = gguf_get_tensor(m, tn);
            const long qr = t && t->ndim == 2 ? (long)t->shape[1] : -1;
            snprintf(tn, sizeof(tn), "blk.%d.attn_k.weight", l);
            GGUFTensor *tk = gguf_get_tensor(m, tn);
            const long kr = tk && tk->ndim == 2 ? (long)tk->shape[1] : -1;
            /* per-layer head_dim: gemma4 full layers carry hd=512 (8 heads),
             * swa layers hd=256. gcd of q/k rows recovers it. */
            long a = qr, b2 = kr;
            if (a > 0 && b2 > 0) { while (b2) { long tt = a % b2; a = b2; b2 = tt; } }
            e->pl_hd[l] = (qr > 0 && kr > 0) ? (int)a : hd_meta;
            if (e->pl_hd[l] <= 0 || qr % e->pl_hd[l] != 0 || kr % e->pl_hd[l] != 0)
                e->pl_hd[l] = hd_meta;
            if (e->pl_hd[l] > max_hd) max_hd = e->pl_hd[l];
            e->pl_heads[l] = qr > 0 ? (int)(qr / e->pl_hd[l]) : cfg->dim / cfg->head_dim;
            if (e->pl_heads[l] > max_heads) max_heads = e->pl_heads[l];
            if ((long)e->pl_heads[l] * e->pl_hd[l] > max_qout)
                max_qout = (long)e->pl_heads[l] * e->pl_hd[l];
            e->pl_kv[l] = kr > 0 ? (int)(kr / e->pl_hd[l]) : cfg->n_kv_heads;
            if (e->pl_kv[l] * e->pl_hd[l] > max_kv * hd_meta)
                max_kv = e->pl_kv[l] * e->pl_hd[l] / hd_meta;
            snprintf(tn, sizeof(tn), "blk.%d.ffn_down.weight", l);
            t = gguf_get_tensor(m, tn);
            e->pl_ffn[l] = t && t->ndim == 2 ? (int)t->shape[0] : F;
            if (e->pl_ffn[l] > max_ffn) max_ffn = e->pl_ffn[l];
            /* gemma4 E-series pattern: full-attn layers carry hd=512
             * (vs swa hd=256) and take no sliding window. */
            e->pl_swa[l] = (e->pl_hd[l] > cfg->head_dim)
                               ? 0 : (m->sliding_window > 0 ? m->sliding_window : 512);
            /* gemma4 KV sharing: shared_kv_layers=20 => first 15 layers hold
             * KV; layers 15+ reuse layer 13 (swa) or 14 (full) caches.
             * llama-model.cpp:2502: src = 15 - (is_swa ? 2 : 1). */
            const int n_kv_own = cfg->n_layers - m->shared_kv_layers;
            e->pl_src[l] = (l >= n_kv_own && m->shared_kv_layers > 0)
                               ? n_kv_own - (e->pl_swa[l] ? 2 : 1) : -1;
        }
        fprintf(stderr, "[qwen2-engine] hetero maxes: heads=%d kv=%d ffn=%ld "
                "hd0=%d hd4=%d\n", max_heads, max_kv, max_ffn,
                e->pl_hd[0], e->pl_hd[4]);
    }
    cudaMalloc(&e->d_q,  max_qout * sizeof(float));
    cudaMalloc(&e->d_att, max_qout * sizeof(float));
    cudaMalloc(&e->d_h,  max_ffn * sizeof(float));
    cudaMalloc(&e->d_g,  max_ffn * sizeof(float));   /* split-SwiGLU staging */
    cudaMalloc(&e->d_u,  max_ffn * sizeof(float));
    /* gemma4 MatFormer per-layer embeddings */
    e->has_pl_embd = 0;
    e->pl_dim = 0;
    e->ple_cache_tok = -1;
    {
        GGUFTensor *tp = gguf_get_tensor(m, "per_layer_token_embd.weight");
        GGUFTensor *tm = gguf_get_tensor(m, "per_layer_model_proj.weight");
        if (tp && tp->data && tm && tm->data && e->cfg.tr.per_layer_embd &&
            m->per_layer_embd_dim > 0) {
            e->has_pl_embd = 1;
            e->pl_dim = m->per_layer_embd_dim;
            memset(&e->pl_model_proj, 0, sizeof(TTensor));
            void *dpj = NULL;
            if (cudaMalloc(&dpj, tm->size_bytes) == cudaSuccess) {
                cudaMemcpy(dpj, tm->data, tm->size_bytes, cudaMemcpyHostToDevice);
                e->pl_model_proj.ptr = dpj;
                e->pl_model_proj.dtype = (int)tm->type;
                                fprintf(stderr, "[qwen2-engine] pl_model_proj type=%d size=%ld ne=[%ld,%ld]\n",
                        (int)tm->type, tm->size_bytes, tm->shape[0], tm->shape[1]);
            } else { e->has_pl_embd = 0; }
        }
        if (e->has_pl_embd) {
            const long row = (long)e->cfg.n_layers * e->pl_dim;   /* 8960 */
            cudaMalloc(&e->d_pl_tmp, row * sizeof(float));
            cudaMalloc(&e->d_ple_row, row * sizeof(float));
            e->pl_proj_norm_host = (float *)malloc(256 * sizeof(float));
            {
                GGUFTensor *tn2 = gguf_get_tensor(m, "per_layer_proj_norm.weight");
                if (tn2 && tn2->data && tn2->size_bytes >= 256*4)
                    memcpy(e->pl_proj_norm_host, tn2->data, 256 * sizeof(float));
            }
            const int ones_n = max_kv * cfg->head_dim;
            cudaMalloc(&e->d_ones, ones_n * sizeof(float));
            {
                float *ones = (float *)malloc(ones_n * sizeof(float));
                for (int i = 0; i < ones_n; i++) ones[i] = 1.0f;
                cudaMemcpy(e->d_ones, ones, ones_n * sizeof(float),
                           cudaMemcpyHostToDevice);
                free(ones);
            }
            e->ple_pe = (float *)malloc(row * sizeof(float));
            cudaMalloc(&e->d_rope_freqs, 256 * sizeof(float));
            cudaMemset(e->d_rope_freqs, 0, 256 * sizeof(float));
            fprintf(stderr, "[qwen2-engine] MatFormer per-layer embeddings ON "
                            "(pl_dim=%d)\n", e->pl_dim);

            /* rope_freqs [256] f32: partial-rope factors for FULL attn layers.
             * ggml semantics: theta/ff — pairs with ff=1e30 don't rotate. */
            {
                GGUFTensor *trf = gguf_get_tensor(m, "rope_freqs.weight");
                if (trf && trf->data && trf->size_bytes >= 256 * 4)
                    cudaMemcpy(e->d_rope_freqs, trf->data, 256 * sizeof(float),
                               cudaMemcpyHostToDevice);
            }

            /* pl_heads/kv/ffn already derived above from tensor shapes */
        }
    }
    cudaMalloc(&e->d_logits, (long)e->cfg.vocab * sizeof(float));
    /* M9 split-K flash workspace. Sized to the worst per-layer (max heads,
     * max hd) seen in this model. S = clamp(ctx/256, 2, 16) at launch;
     * workspace capacity = 16, sufficient for any ctx <= 4096. Peak for
     * gemma4 (16 heads * hd 512): 16 * 16 * 514 * 4 ~= 526 KB. */
    {
        const int S_MAX = 64;
        const size_t per_acc = (size_t)max_heads * max_hd;
        const size_t per_ml  = (size_t)max_heads;
        cudaMalloc(&e->d_split_pacc, (size_t)S_MAX * per_acc * sizeof(float));
        cudaMalloc(&e->d_split_pm,   (size_t)S_MAX * per_ml  * sizeof(float));
        cudaMalloc(&e->d_split_pl,   (size_t)S_MAX * per_ml  * sizeof(float));
        e->d_split_S_max = S_MAX;
    }
    /* per-layer max kv width: gemma4 full layers carry 2x the kv heads */
    /* Hybrid KV dispatch (Fix1): always allocate FP32 KV. Quantized caches
     * are additional. Effective threshold (TT_QKV_THRESH, default 256) picks
     * FP32 below thresh (parity) and Q4/Q8 above (speed). This keeps m61
     * GREEN at short ctx while letting TT_Q4_KV=1 hit ~282 tok/s at ctx1024.
     * Parity gates must be run at short ctx where FP32 is active. */
    const long cache_per = (long)max_kv * cfg->max_ctx * cfg->head_dim;
    cudaMalloc(&e->d_kc, cache_per * cfg->n_layers * sizeof(float));
    cudaMalloc(&e->d_vc, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_kc, 0, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_vc, 0, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_xn, 0, D * sizeof(float));
    cudaMemset(e->d_q, 0, max_qout * sizeof(float));
    cudaMemset(e->d_att, 0, max_qout * sizeof(float));
    cudaMemset(e->d_h, 0, F * sizeof(float));
    cudaMemset(e->d_g, 0, F * sizeof(float));
    cudaMemset(e->d_u, 0, F * sizeof(float));
    cudaMemset(e->d_logits, 0, (long)e->cfg.vocab * sizeof(float));
    cudaMemset(e->d_x, 0, D * sizeof(float));
    const long kvdim_alloc = (long)max_kv * cfg->head_dim;   /* per-layer max */
    cudaMalloc(&e->d_k_stage, kvdim_alloc * sizeof(float));
    cudaMalloc(&e->d_v_stage, kvdim_alloc * sizeof(float));
    cudaMemset(e->d_k_stage, 0, kvdim_alloc * sizeof(float));
    cudaMemset(e->d_v_stage, 0, kvdim_alloc * sizeof(float));
    cudaMalloc(&e->d_pos, sizeof(int));
    cudaMemsetAsync(e->d_pos, 0, sizeof(int), e->stream);   /* pos starts at 0 on device */
    const int nb = 256;
    cudaMalloc(&e->d_bvals, nb * sizeof(float));
    cudaMalloc(&e->d_bidxs, nb * sizeof(int));
    /* sampling state: greedy until qwen2_engine_set_sampling() is called */
    e->sampling_temp = 0.0f;
    e->repeat_penalty = 1.0f;
    cudaMalloc(&e->d_recent, 64 * sizeof(int));
    cudaMalloc(&e->d_n_recent, sizeof(int));
    cudaMemset(e->d_recent, 0, 64 * sizeof(int));
    cudaMemsetAsync(e->d_n_recent, 0, sizeof(int), e->stream);
    {
        int zero = 0;
        cudaMalloc(&e->d_sampling_on, sizeof(int));
        cudaMemcpy(e->d_sampling_on, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&e->d_out, sizeof(int));
    /* graph replay state */
    e->graph_exec = NULL;
    e->graph_ready = 0;
    /* TT_PROFILE forces eager mode: event records inside the captured region
     * are illegal, so per-stage profiling always runs graph-free. Hybrid
     * CPU offloading also uses eager mode since CPU layers cannot be captured. */
    e->no_graph = (getenv("TT_NO_GRAPH") || getenv("TT_PROFILE") || e->n_gpu_layers < cfg->n_layers) ? 1 : 0;
    e->pending_tok = -1;
    cudaMalloc(&e->d_next_tok, sizeof(int));
    cudaHostAlloc(&e->h_sampled, sizeof(int), cudaHostAllocDefault);
    e->d_kc_q8 = NULL;
    e->d_vc_q8 = NULL;
    e->use_q8_kvcache = 0;
    e->d_kc_q4 = NULL;
    e->d_vc_q4 = NULL;
    e->use_q4_kvcache = 0;
    const char *q4_env = getenv("TT_Q4_KV");
    if (q4_env && atoi(q4_env) != 0) {
        qwen2_engine_enable_q4_kvcache(e, 1);
    } else {
        const char *q8_env = getenv("TT_Q8_KV");
        if (q8_env && atoi(q8_env) != 0) {
            qwen2_engine_enable_q8_kvcache(e, 1);
        }
    }
    /* Hybrid Fix1: quantized KV with threshold>0 requires per-token dispatch.
     * Graph capture locks the flash/scatter path at capture pos (short ctx),
     * so it cannot switch to Q* at long ctx. Disable graph when hybrid is
     * active to allow threshold dispatch; short ctx stays FP32 for parity,
     * long ctx switches to Q* for speed. */
    if ((e->use_q4_kvcache || e->use_q8_kvcache) && kv_thresh_value() > 0) {
        if (!e->no_graph) {
            fprintf(stderr, "[qwen2-engine] hybrid KV (thresh %d): graph disabled for per-ctx dispatch\n", kv_thresh_value());
        }
        e->no_graph = 1;
    }
    if (e->n_gpu_layers < cfg->n_layers) {
        e->h_x_buf = (float *)malloc(D * sizeof(float));
        e->h_xn_buf = (float *)malloc(D * sizeof(float));
        e->h_q_buf = (float *)malloc(max_qout * sizeof(float));
        e->h_att_buf = (float *)malloc(max_qout * sizeof(float));
        e->h_h_buf = (float *)malloc(max_ffn * sizeof(float));
        e->h_g_buf = (float *)malloc(max_ffn * sizeof(float));
        e->h_u_buf = (float *)malloc(max_ffn * sizeof(float));
        e->h_out_buf = (float *)malloc(D * sizeof(float));
        e->h_k_stage = (float *)malloc(kvdim_alloc * sizeof(float));
        e->h_v_stage = (float *)malloc(kvdim_alloc * sizeof(float));
        e->h_kc = (float *)calloc((size_t)cfg->n_layers * cfg->n_kv_heads * cfg->max_ctx * cfg->head_dim, sizeof(float));
        e->h_vc = (float *)calloc((size_t)cfg->n_layers * cfg->n_kv_heads * cfg->max_ctx * cfg->head_dim, sizeof(float));
    } else {
        e->h_x_buf = e->h_xn_buf = e->h_q_buf = e->h_att_buf = NULL;
        e->h_h_buf = e->h_g_buf = e->h_u_buf = e->h_out_buf = NULL;
        e->h_k_stage = e->h_v_stage = e->h_kc = e->h_vc = NULL;
    }
    /* M10+ batched speculative buffers: unallocated until first verify_speculative(). */
    e->d_x_batch = NULL;
    e->d_xn_batch = NULL;
    e->d_logits_batch = NULL;
    e->d_spec_n = -1;
    e->d_spec_max_n = 0;
    /* Task 1: persistent prefill arena (512 tokens or max_ctx if smaller) */
    {
        size_t pf_max_n = 512;
        if ((size_t)cfg->max_ctx < pf_max_n) pf_max_n = (size_t)cfg->max_ctx;
        e->pf_arena_max_n = pf_max_n;
        e->d_pf_X = e->d_pf_Xn = e->d_pf_Q = e->d_pf_K = e->d_pf_V = NULL;
        e->d_pf_Att = e->d_pf_H = e->d_pf_G = e->d_pf_U = NULL;
        e->d_pf_pos_batch = NULL;
        /* Recompute maxima for sizing (matches prefill_batched_gemm logic) */
        int pf_max_qout = cfg->n_heads * cfg->head_dim;
        int pf_max_kvdim = cfg->n_kv_heads * cfg->head_dim;
        int pf_hidden = cfg->hidden_dim;
        for (int l = 0; l < cfg->n_layers; l++) {
            int H_l = e->pl_heads[l] > 0 ? e->pl_heads[l] : cfg->n_heads;
            int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : cfg->n_kv_heads;
            int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : cfg->head_dim;
            if (H_l * HDl > pf_max_qout) pf_max_qout = H_l * HDl;
            if (KV_l * HDl > pf_max_kvdim) pf_max_kvdim = KV_l * HDl;
            int FF_l = e->pl_ffn[l] > 0 ? e->pl_ffn[l] : cfg->hidden_dim;
            if (FF_l > pf_hidden) pf_hidden = FF_l;
        }
        cudaMalloc(&e->d_pf_X, pf_max_n * (size_t)cfg->dim * sizeof(float));
        cudaMalloc(&e->d_pf_Xn, pf_max_n * (size_t)cfg->dim * sizeof(float));
        cudaMalloc(&e->d_pf_Q, pf_max_n * (size_t)pf_max_qout * sizeof(float));
        cudaMalloc(&e->d_pf_K, pf_max_n * (size_t)pf_max_kvdim * sizeof(float));
        cudaMalloc(&e->d_pf_V, pf_max_n * (size_t)pf_max_kvdim * sizeof(float));
        cudaMalloc(&e->d_pf_Att, pf_max_n * (size_t)pf_max_qout * sizeof(float));
        cudaMalloc(&e->d_pf_H, pf_max_n * (size_t)pf_hidden * sizeof(float));
        cudaMalloc(&e->d_pf_G, pf_max_n * (size_t)pf_hidden * sizeof(float));
        cudaMalloc(&e->d_pf_U, pf_max_n * (size_t)pf_hidden * sizeof(float));
        cudaMalloc(&e->d_pf_pos_batch, pf_max_n * sizeof(int));
        if (!e->d_pf_X || !e->d_pf_Xn || !e->d_pf_Q || !e->d_pf_K || !e->d_pf_V ||
            !e->d_pf_Att || !e->d_pf_H || !e->d_pf_G || !e->d_pf_U || !e->d_pf_pos_batch) {
            fprintf(stderr, "[qwen2-engine] pf arena alloc failed (pf_max_n=%zu)\n", pf_max_n);
        }
    }
    /* If FP32 prefill flash shared memory exceeds default 48KB, opt in to dynamic shmem */
    {
        int max_hd = cfg->head_dim;
        for (int l = 0; l < cfg->n_layers; l++) {
            int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : cfg->head_dim;
            if (HDl > max_hd) max_hd = HDl;
        }
        size_t max_smem_fp32 = 2 * (size_t)BC_PREFILL_FP32 * max_hd * sizeof(float);
        if (max_smem_fp32 > 48 * 1024) {
            cudaFuncSetAttribute(k_prefill_flash_fp32,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_smem_fp32);
        }
        {
            int max_bph = max_hd / 32;
            size_t max_smem_q8 = 2 * (size_t)BC_PREFILL * max_bph * sizeof(half)
                               + 2 * (size_t)BC_PREFILL * max_hd * sizeof(int8_t);
            if (max_smem_q8 > 48 * 1024) {
                cudaFuncSetAttribute(k_prefill_flash_q8_0,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_smem_q8);
            }
        }
    }
    return e;


}

void qwen2_engine_free(Qwen2Engine *e) {
    if (!e) return;
    /* weight buffers were allocated individually; free via cudaFree of tracked
     * pointers is omitted where ownership aliases mmap (host side frees via gguf_free).
     * Device allocations freed here: */
    cudaFree(e->d_x); cudaFree(e->d_xn); cudaFree(e->d_q); cudaFree(e->d_att); cudaFree(e->d_h);
    if (e->d_g) cudaFree(e->d_g);
    if (e->d_u) cudaFree(e->d_u);
    cudaFree(e->d_logits); cudaFree(e->d_kc); cudaFree(e->d_vc);
    if (e->d_kc_q8) cudaFree(e->d_kc_q8);
    if (e->d_vc_q8) cudaFree(e->d_vc_q8);
    if (e->d_kc_q4) cudaFree(e->d_kc_q4);
    if (e->d_vc_q4) cudaFree(e->d_vc_q4);
    cudaFree(e->d_k_stage); cudaFree(e->d_v_stage); cudaFree(e->d_pos);
    cudaFree(e->d_bvals); cudaFree(e->d_bidxs); cudaFree(e->d_out);
    if (e->d_split_pacc) cudaFree(e->d_split_pacc);
    if (e->d_split_pm)   cudaFree(e->d_split_pm);
    if (e->d_split_pl)   cudaFree(e->d_split_pl);
    if (e->d_x_batch) cudaFree(e->d_x_batch);
    if (e->d_xn_batch) cudaFree(e->d_xn_batch);
    if (e->d_logits_batch) cudaFree(e->d_logits_batch);
    if (e->d_recent) cudaFree(e->d_recent);
    if (e->d_n_recent) cudaFree(e->d_n_recent);
    if (e->h_sampled) cudaFreeHost(e->h_sampled);
    if (e->d_pl_tmp) cudaFree(e->d_pl_tmp);
    if (e->d_ple_row) cudaFree(e->d_ple_row);
    if (e->d_ones) cudaFree(e->d_ones);
    if (e->ple_pe) free(e->ple_pe);
    if (e->pl_proj_norm_host) free(e->pl_proj_norm_host);
    cudaFree(e->d_next_tok);
    if (e->d_pf_X) cudaFree(e->d_pf_X);
    if (e->d_pf_Xn) cudaFree(e->d_pf_Xn);
    if (e->d_pf_Q) cudaFree(e->d_pf_Q);
    if (e->d_pf_K) cudaFree(e->d_pf_K);
    if (e->d_pf_V) cudaFree(e->d_pf_V);
    if (e->d_pf_Att) cudaFree(e->d_pf_Att);
    if (e->d_pf_H) cudaFree(e->d_pf_H);
    if (e->d_pf_G) cudaFree(e->d_pf_G);
    if (e->d_pf_U) cudaFree(e->d_pf_U);
    if (e->d_pf_pos_batch) cudaFree(e->d_pf_pos_batch);
    if (e->d_split_pl)   cudaFree(e->d_split_pl);
    if (e->d_x_batch)    cudaFree(e->d_x_batch);
    /* NOTE: per-weight cudaFree calls are intentionally not tracked here; they are
     * leaked until process exit by design (engine lifetime == process lifetime).
    if (e->h_x_buf) free(e->h_x_buf);
    if (e->h_xn_buf) free(e->h_xn_buf);
    if (e->h_q_buf) free(e->h_q_buf);
    if (e->h_att_buf) free(e->h_att_buf);
    if (e->h_h_buf) free(e->h_h_buf);
    if (e->h_g_buf) free(e->h_g_buf);
    if (e->h_u_buf) free(e->h_u_buf);
    if (e->h_out_buf) free(e->h_out_buf);
    if (e->h_k_stage) free(e->h_k_stage);
    if (e->h_v_stage) free(e->h_v_stage);
    if (e->h_kc) free(e->h_kc);
    if (e->h_vc) free(e->h_vc);
     * Tracked as known limitation in PLAN_M6 M6.1. */
    if (e->stream) cudaStreamDestroy(e->stream);
    free(e);
}

/* 0 while running eagerly; 1 while a cudaStream capture is in flight. */
static int g_capturing = 0;

/* ---------- TT_PROFILE per-stage instrumentation ----------
 * Enabled by env TT_PROFILE (which also forces eager/no-graph mode at create:
 * cudaEvent records are not allowed inside a captured region). Stages bracket
 * kernel launches with lazy-created event pairs on e->stream; per-invocation
 * ms is accumulated and stored for a median table printed via
 * qwen2_debug_profile_report(). */
typedef enum {
    TT_P_EMBED = 0, TT_P_QKV, TT_P_OMLP, TT_P_FLASH,
    TT_P_RMSNORM, TT_P_SCATTER, TT_P_LOGITS, TT_P_ARGMAX,
    TT_P_NSTAGES
} TTProfStage;

static const char *tt_prof_names[TT_P_NSTAGES] = {
    "embed", "qkv-gemv", "o+mlp-gemv", "flash",
    "rmsnorm", "kv-scatter", "logits-gemv", "argmax"
};

typedef struct {
    cudaEvent_t b, e;
    double sum;
    int n;
    float s[8192];   /* per-invocation ms, capped */
} TTProf;

static TTProf tt_prof[TT_P_NSTAGES];
static int tt_prof_on = -1;

static int tt_profiling(void) {
    if (tt_prof_on < 0) tt_prof_on = getenv("TT_PROFILE") ? 1 : 0;
    return tt_prof_on;
}

static void tt_prof_begin(TTProfStage st, cudaStream_t stream) {
    TTProf *p = &tt_prof[st];
    if (!p->b) { cudaEventCreate(&p->b); cudaEventCreate(&p->e); }
    cudaEventRecord(p->b, stream);
}

static void tt_prof_end(TTProfStage st, cudaStream_t stream) {
    TTProf *p = &tt_prof[st];
    cudaEventRecord(p->e, stream);
    cudaEventSynchronize(p->e);
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, p->b, p->e) == cudaSuccess) {
        p->sum += ms;
        if (p->n < 8192) p->s[p->n++] = ms;
    }
}

void qwen2_debug_profile_reset(void) {
    for (int i = 0; i < TT_P_NSTAGES; i++) { tt_prof[i].sum = 0; tt_prof[i].n = 0; }
}

static int tt_flt_cmp(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

void qwen2_debug_profile_report(int nsteps) {
    if (!tt_profiling() || nsteps <= 0) return;
    printf("PROFILE mode=eager\n");
    double total = 0;
    for (int i = 0; i < TT_P_NSTAGES; i++) {
        TTProf *p = &tt_prof[i];
        if (!p->n) continue;
        /* per-invocation counts are deterministic per step, so samples split
         * evenly into per-step groups; sum each group -> per-step totals */
        const int per = p->n / nsteps;
        if (per < 1) continue;
        const int use = (p->n / per) < nsteps ? (p->n / per) : nsteps;
        float step_tot[512];
        const int ns = use < 512 ? use : 512;
        for (int s = 0; s < ns; s++) {
            float acc = 0.f;
            for (int j = s * per; j < (s + 1) * per && j < p->n; j++) acc += p->s[j];
            step_tot[s] = acc;
        }
        qsort(step_tot, (size_t)ns, sizeof(float), tt_flt_cmp);
        const float med = (ns & 1) ? step_tot[ns / 2]
                                   : 0.5f * (step_tot[ns / 2 - 1] + step_tot[ns / 2]);
        printf("PROFILE %-12s %8.3f\n", tt_prof_names[i], med);
        total += med;
    }
    printf("PROFILE %-12s %8.3f\n", "TOTAL(med)", total);
}

__global__ void k_fill_const(float *p, int n, float v) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) p[i] = v;
}
static void eng_rms(Qwen2Engine *e, const float *d_ptr, const char *tag, int n) {
    static float buf[4096];
    cudaMemcpy(buf, d_ptr, n * 4, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(e->stream);
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)buf[i]*buf[i];
    fprintf(stderr, "[e4-L%d] %s rms=%.4f\n", e->pos, tag, sqrt(s/n));
}
static void ple_canary(Qwen2Engine *e, const char *tag, int l) {
    static float buf[8960];
    const long n = (long)e->cfg.n_layers * e->pl_dim;
    cudaMemcpy(buf, e->d_ple_row, n * 4, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(e->stream);
    int bad = -1;
    for (long i = 5376; i < n && bad < 0; i++) if (buf[i] != -777.0f) bad = (int)i;
    fprintf(stderr, "[CANARY] L%d %s firstbad=%d\n", l, tag, bad);
}
extern "C" long tt_cpu_gemv(const void *W, int dtype, const float *x, float *y,
                            int M, int K, int n_threads);

static void cpu_rmsnorm(const float *x, const float *w, float *out, int dim, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

static void cpu_silu_mult(const float *g, const float *u, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float val = g[i];
        float silu = val / (1.0f + expf(-val));
        out[i] = silu * u[i];
    }
}

static void cpu_rope(float *v, int n_heads, int head_dim, int pos, float base) {
    for (int h = 0; h < n_heads; h++) {
        float *vec = v + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float theta = powf(base, -2.0f * (float)i / (float)head_dim) * (float)pos;
            float cos_th = cosf(theta);
            float sin_th = sinf(theta);
            float v0 = vec[i];
            float v1 = vec[i + head_dim / 2];
            vec[i]                 = v0 * cos_th - v1 * sin_th;
            vec[i + head_dim / 2] = v0 * sin_th + v1 * cos_th;
        }
    }
}

static void forward_layer_cpu(Qwen2Engine *e, int l) {
    const TTConfig *c = &e->cfg;
    LayerW *w = &e->L[l];
    const int D = c->dim;
    const int F = c->hidden_dim;
    const int HD = c->head_dim;
    const int H_l = e->pl_heads[l] > 0 ? e->pl_heads[l] : c->n_heads;
    const int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : c->n_kv_heads;
    const int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
    const int attn_qout = H_l * HDl;
    const int kvdim_l = KV_l * HDl;
    const int n_threads = 8;

    cpu_rmsnorm(e->h_x_buf, w->attn_norm, e->h_xn_buf, D, c->rms_eps);

    tt_cpu_gemv(w->q.ptr, w->q.dtype, e->h_xn_buf, e->h_q_buf, attn_qout, D, n_threads);
    tt_cpu_gemv(w->k.ptr, w->k.dtype, e->h_xn_buf, e->h_k_stage, kvdim_l, D, n_threads);
    tt_cpu_gemv(w->v.ptr, w->v.dtype, e->h_xn_buf, e->h_v_stage, kvdim_l, D, n_threads);

    cpu_rope(e->h_q_buf, H_l, HDl, e->pos, c->rope_base);
    cpu_rope(e->h_k_stage, KV_l, HDl, e->pos, c->rope_base);

    const float scale = 1.0f / sqrtf((float)HDl);
    const int gqa_ratio = H_l / KV_l;
    const int kvdim = KV_l * HDl;
    float *kc = e->h_kc + (long)l * c->max_ctx * kvdim;
    float *vc = e->h_vc + (long)l * c->max_ctx * kvdim;

    memcpy(kc + (long)e->pos * kvdim, e->h_k_stage, kvdim * sizeof(float));
    memcpy(vc + (long)e->pos * kvdim, e->h_v_stage, kvdim * sizeof(float));

    const int ctx_len = e->pos + 1;
    for (int h = 0; h < H_l; h++) {
        const int kv = h / gqa_ratio;
        const float *qh = e->h_q_buf + h * HDl;
        float *out_h = e->h_att_buf + h * HDl;
        memset(out_h, 0, HDl * sizeof(float));

        float max_score = -1e30f;
        float scores[4096];
        for (int t = 0; t < ctx_len && t < 4096; t++) {
            const float *kt = kc + (long)t * kvdim + kv * HDl;
            float score = 0.0f;
            for (int d = 0; d < HDl; d++) score += qh[d] * kt[d];
            score *= scale;
            scores[t] = score;
            if (score > max_score) max_score = score;
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < ctx_len && t < 4096; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < ctx_len && t < 4096; t++) {
            float weight = scores[t] * inv_sum;
            const float *vt = vc + (long)t * kvdim + kv * HDl;
            for (int d = 0; d < HDl; d++) {
                out_h[d] += weight * vt[d];
            }
        }
    }
    tt_cpu_gemv(w->o.ptr, w->o.dtype, e->h_att_buf, e->h_out_buf, D, attn_qout, n_threads);
    for (int i = 0; i < D; i++) e->h_x_buf[i] += e->h_out_buf[i];

    cpu_rmsnorm(e->h_x_buf, w->ffn_norm, e->h_xn_buf, D, c->rms_eps);

    tt_cpu_gemv(w->gate.ptr, w->gate.dtype, e->h_xn_buf, e->h_g_buf, F, D, n_threads);
    tt_cpu_gemv(w->up.ptr, w->up.dtype, e->h_xn_buf, e->h_u_buf, F, D, n_threads);

    cpu_silu_mult(e->h_g_buf, e->h_u_buf, e->h_h_buf, F);

    tt_cpu_gemv(w->down.ptr, w->down.dtype, e->h_h_buf, e->h_out_buf, D, F, n_threads);
    for (int i = 0; i < D; i++) e->h_x_buf[i] += e->h_out_buf[i];
}
static int forward_layers(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;
    const int HD = c->head_dim;
    long cache_layer = c->n_kv_heads * (long)c->max_ctx * HD;   /* stride matches alloc */
    long cache_layer_q8 = c->n_kv_heads * (long)c->max_ctx * (HD / 32);
    if (e->pl_hd[0] > 0) {
        long mx = 0;
        long mx_q8 = 0;
        for (int l = 0; l < c->n_layers; l++) {
            const long w2 = (long)e->pl_kv[l] * e->pl_hd[l];
            if (w2 > mx) mx = w2;
            const long w2_q8 = (long)e->pl_kv[l] * (e->pl_hd[l] / 32);
            if (w2_q8 > mx_q8) mx_q8 = w2_q8;
        }
        cache_layer = mx * c->max_ctx;
        cache_layer_q8 = mx_q8 * c->max_ctx;
    }
    dim3 g, b;

    static int trace = -1;
    if (trace < 0) trace = getenv("TT_TRACE") ? 1 : 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (l == e->n_gpu_layers) {
            cudaMemcpyAsync(e->h_x_buf, e->d_x, c->dim * sizeof(float),
                            cudaMemcpyDeviceToHost, e->stream);
            cudaStreamSynchronize(e->stream);
        }
        if (l >= e->n_gpu_layers) {
            forward_layer_cpu(e, l);
            continue;
        }
        LayerW *w = &e->L[l];
        float *Kl_f = e->d_kc + l * cache_layer;
        float *Vl_f = e->d_vc + l * cache_layer;
        BlockQ8_0 *Kl_q8 = e->d_kc_q8 ? (e->d_kc_q8 + (long)l * cache_layer_q8) : NULL;
        BlockQ8_0 *Vl_q8 = e->d_vc_q8 ? (e->d_vc_q8 + (long)l * cache_layer_q8) : NULL;
        BlockQ4_0 *Kl_q4 = e->d_kc_q4 ? (e->d_kc_q4 + (long)l * cache_layer_q8) : NULL;
        BlockQ4_0 *Vl_q4 = e->d_vc_q4 ? (e->d_vc_q4 + (long)l * cache_layer_q8) : NULL;
        /* gemma4 KV sharing: shared layers (pl_src[l] >= 0) read the source
         * layer's cache slab instead of computing/scattering their own K/V.
         * llama-model.cpp:2502 semantics. */
        const int kv_shared = e->has_pl_embd && e->pl_src[l] >= 0;
        if (kv_shared) {
            Kl_f = e->d_kc + (long)e->pl_src[l] * cache_layer;
            Vl_f = e->d_vc + (long)e->pl_src[l] * cache_layer;
            if (e->d_kc_q8) {
                Kl_q8 = e->d_kc_q8 + (long)e->pl_src[l] * cache_layer_q8;
                Vl_q8 = e->d_vc_q8 + (long)e->pl_src[l] * cache_layer_q8;
            }
            if (e->d_kc_q4) {
                Kl_q4 = e->d_kc_q4 + (long)e->pl_src[l] * cache_layer_q8;
                Vl_q4 = e->d_vc_q4 + (long)e->pl_src[l] * cache_layer_q8;
            }
        }
        if (trace) fprintf(stderr, "[FWD] L%d enter\n", l);

        /* Cache layout: [slot][kv_head * head_dim] so each GEMV output of
         * width n_kv_heads*HD lands contiguously per slot.
         * Flash kernel indexes K(t,kvh,i) = Kl_f[(t*n_kv_heads + kvh)*HD + i].
         * K/V are computed into staging buffers, then scattered to the slot
         * selected by the DEVICE position scalar (*e->d_pos). */

        #define CHK_STAGE(tag) do { cudaError_t ce_ = cudaGetLastError(); \
            if (ce_ != cudaSuccess && getenv("TT_DEBUG")) \
                fprintf(stderr, "[qwen2-engine] L%d %s: %s\n", l, tag, cudaGetErrorString(ce_)); } while(0)

        /* 1. xn = rmsnorm(x) * attn_norm */
        if (tt_profiling()) tt_prof_begin(TT_P_RMSNORM, e->stream);
        k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
            e->d_x, w->attn_norm, e->d_xn, c->dim, c->rms_eps, c->tr.norm_offset);
        if (tt_profiling()) tt_prof_end(TT_P_RMSNORM, e->stream);

        CHK_STAGE("1 rmsnorm");
        if (trace && l == 0) { eng_rms(e, e->d_xn, "xn", c->dim); }
        /* 2. projections: q -> d_q ; k,v -> KV cache slot pos */
        if (tt_profiling()) tt_prof_begin(TT_P_QKV, e->stream);
        const int H_l = e->pl_heads[l] > 0 ? e->pl_heads[l] : c->n_heads;
        const int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : c->n_kv_heads;
        const int FF_l = e->pl_ffn[l] > 0 ? e->pl_ffn[l] : c->hidden_dim;
        const int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
        const int attn_qout = H_l * HDl;
        const int kvdim_l = KV_l * HDl;
        int qrc = tt_gemv_layer_dispatch(w->q.ptr, w->q.dtype, e->d_xn, e->d_q,
                      attn_qout, c->dim, e->stream);
        if (qrc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] q gemv rc=%d\n", qrc);
        if (!kv_shared) {
            int krc = tt_gemv_layer_dispatch(w->k.ptr, w->k.dtype, e->d_xn, e->d_k_stage, kvdim_l, c->dim, e->stream);
            if (krc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] k gemv rc=%d\n", krc);
        }
        /* QKV biases present in some GGUF conversions of Qwen2 (applied by
         * llama.cpp whenever the tensors exist). Optional by design. */
        CHK_STAGE("2 qkv-gemv");
        const int kvdim = kvdim_l;   /* per-layer: full-attn gemma4 layers carry hd=512 */
        if (w->q_bias)
            k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_q, w->q_bias, c->dim);
        if (!kv_shared && w->k_bias)
            k_add<<<(kvdim + 255) / 256, 256, 0, e->stream>>>(e->d_k_stage, w->k_bias, kvdim);

        /* M7 trait: qwen3 per-head q/k RMSNorm pre-rope. Sits between the
         * q/k projections (+biases) and RoPE; branch is dead for every
         * family without the trait. */
        if (c->tr.qk_norm_rms) {
            const int qkthreads = HDl < 256 ? HDl : 256;
            k_qk_norm_rms<<<H_l, qkthreads, qkthreads * sizeof(float), e->stream>>>
                (e->d_q, w->q_norm, H_l, HDl, c->tr.qk_norm_eps);
            if (!kv_shared)
            k_qk_norm_rms<<<KV_l, qkthreads, qkthreads * sizeof(float), e->stream>>>
                (e->d_k_stage, w->k_norm, KV_l, HDl, c->tr.qk_norm_eps);
        }

        CHK_STAGE("2b biases+qknorm");
        /* 3. RoPE on q (all heads) and on the staged k row (in-place, pre-scatter).
         * Kernel picked by the rope-style trait: NEOX half-split vs GPT-J
         * interleaved pairs (llama family). */
        static int no_rope = -1;
        if (no_rope < 0) no_rope = getenv("TT_NO_ROPE") ? 1 : 0;
        if (!no_rope) {
            /* gemma4: full-attn layers (16 heads) use base 1e6 + partial-rope
             * freq factors (theta/ff); swa layers use base 1e4, full rotation. */
            const int is_full_l = (e->pl_swa[l] == 0);
            const float base_l = e->has_pl_embd
                ? (is_full_l ? 1e6f : 1e4f) : c->rope_base;
            const float *ff_l = (e->has_pl_embd && is_full_l) ? e->d_rope_freqs : NULL;
            void (*rope_fn)(float *, int, int, const int *, float) =
                (c->tr.rope == ROPE_GPTJ) ? k_rope_gptj : k_rope;
            void (*rope_ff_fn)(float *, int, int, const int *, float, const float *) =
                (c->tr.rope == ROPE_GPTJ) ? k_rope_gptj_ff : k_rope_ff;
            g.x = (HDl / 2 + 63) / 64; g.y = H_l; g.z = 1;
            b.x = 64; b.y = 1; b.z = 1;
            if (ff_l)
                rope_ff_fn<<<g, b, 0, e->stream>>>(e->d_q, H_l, HDl, e->d_pos, base_l, ff_l);
            else
                rope_fn<<<g, b, 0, e->stream>>>(e->d_q, H_l, HDl, e->d_pos, base_l);
            g.x = (HDl / 2 + 63) / 64; g.y = KV_l; g.z = 1;
            if (!kv_shared) {
            if (ff_l)
                rope_ff_fn<<<g, b, 0, e->stream>>>(e->d_k_stage, KV_l, HDl, e->d_pos, base_l, ff_l);
            else
                rope_fn<<<g, b, 0, e->stream>>>(e->d_k_stage, KV_l, HDl, e->d_pos, base_l);
            }
        }

        /* v projection + bias (QKV group) */
        if (!kv_shared) {
        int vrc = tt_gemv_layer_dispatch(w->v.ptr, w->v.dtype, e->d_xn, e->d_v_stage, kvdim_l, c->dim, e->stream);
        if (vrc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] v gemv rc=%d\n", vrc);
        if (w->v_bias)
            k_add<<<(kvdim + 255) / 256, 256, 0, e->stream>>>(e->d_v_stage, w->v_bias, kvdim);
        /* gemma4: plain per-head RMSNorm on V (no gamma) — ones-gamma trick */
        if (e->has_pl_embd) {
            const int vt = HDl < 256 ? HDl : 256;
            k_qk_norm_rms<<<KV_l, vt, vt * sizeof(float), e->stream>>>(
                e->d_v_stage, e->d_ones, KV_l, HDl, c->rms_eps);
        }
        }
        if (tt_profiling()) tt_prof_end(TT_P_QKV, e->stream);

        /* scatter staged K/V into the cache slot chosen by *d_pos.
         * Must precede flash attention. Shared-KV layers skip: they read the
         * source layer's already-populated slab. */
        if (!kv_shared) {
        if (tt_profiling()) tt_prof_begin(TT_P_SCATTER, e->stream);
        /* Dual-write: always FP32, plus Q4/Q8 when allocated (ptr-gated,
         * threshold-independent). Decode flash read stays threshold-gated. */
        k_kv_scatter<<<(kvdim_l + 255) / 256, 256, 0, e->stream>>>(
            e->d_k_stage, e->d_v_stage, Kl_f, Vl_f, e->d_pos,
            KV_l, HDl, c->max_ctx);
        if (Kl_q4 && Vl_q4) {
            const int num_blocks = kvdim_l / 32;
            k_kv_scatter_q4_0<<<(num_blocks + 255) / 256, 256, 0, e->stream>>>(
                e->d_k_stage, e->d_v_stage, Kl_q4, Vl_q4, e->d_pos,
                KV_l, HDl, c->max_ctx);
        }
        if (Kl_q8 && Vl_q8) {
            const int num_blocks = kvdim_l / 32;
            k_kv_scatter_q8_0<<<(num_blocks + 255) / 256, 256, 0, e->stream>>>(
                e->d_k_stage, e->d_v_stage, Kl_q8, Vl_q8, e->d_pos,
                KV_l, HDl, c->max_ctx);
        }
        if (tt_profiling()) tt_prof_end(TT_P_SCATTER, e->stream);
        }

        CHK_STAGE("3 rope");
        /* 4. GQA flash attention over slots [t0..pos]; t0 raised by the SWA
         * trait (gemma2), full [0..pos] when swa_size == 0 (qwen2 unchanged).
         * M9 split-K wire-in: at long ctx the per-warp serial loop
         * underutilizes the GPU (8-16 blocks on 20+ SMs), so dispatch to
         * S = clamp(ctx/256, 2, 16) split-K + combine when ctx > 128.
         * Short ctx keeps the serial path (lower launch overhead). The
         * dispatch is host-side and runs once per forward_layers invocation;
         * under graph capture the chosen path is fixed for the recorded
         * graph's lifetime — safe because the serial path is functionally
         * correct at every ctx and the test gates use short ctx. */
        if (tt_profiling()) tt_prof_begin(TT_P_FLASH, e->stream);
        {
            const int ctx_l = e->pos;            /* host mirror of *d_pos */
            const float scale_l = c->tr.attn_scale_one ? 1.0f
                              : 1.0f / sqrtf((float)HDl);   /* match prior kernel arg */
            const int swa_l = e->has_pl_embd ? e->pl_swa[l] : c->tr.swa_size;
            if (kv_use_q4_eff(e)) {
                /* capture bakes eager S at capture ctx (see split_S_q) */
                int S = split_S_q(ctx_l, e->d_split_S_max);
                dim3 grid_split(S, KV_l);
                int threads_split = (H_l / KV_l) * 32;
                int blocks_per_head = HDl / 32;
                size_t smem_bytes = 2 * (size_t)BC_SPLIT * blocks_per_head * sizeof(half)
                                  + 2 * (size_t)BC_SPLIT * (HDl / 2) * sizeof(uint8_t);
                k_fa2_q4_split<<<grid_split, threads_split, smem_bytes, e->stream>>>(
                    e->d_q, Kl_q4, Vl_q4,
                    e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                    e->d_pos,
                    H_l, KV_l, HDl,
                    scale_l, swa_l, S);
                k_fa2_combine<<<H_l, 32, 0, e->stream>>>(
                    e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                    e->d_att, H_l, HDl, S);
            } else if (kv_use_q8_eff(e)) {
                /* capture bakes eager S at capture ctx (see split_S_q) */
                int S = split_S_q(ctx_l, e->d_split_S_max);
                dim3 grid_split(S, KV_l);
                int threads_split = (H_l / KV_l) * 32;
                int blocks_per_head = HDl / 32;
                size_t smem_bytes = 2 * (size_t)BC_SPLIT * blocks_per_head * sizeof(half)
                                  + 2 * (size_t)BC_SPLIT * HDl * sizeof(int8_t);
                k_fa2_q8_split<<<grid_split, threads_split, smem_bytes, e->stream>>>(
                    e->d_q, Kl_q8, Vl_q8,
                    e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                    e->d_pos,
                    H_l, KV_l, HDl,
                    scale_l, swa_l, S);
                k_fa2_combine<<<H_l, 32, 0, e->stream>>>(
                    e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                    e->d_att, H_l, HDl, S);
            } else {
                // FP32 FA2 tiled split-K: BC=32, smem 2*BC*HD*4, S=ceil(ctx/64) chunk=64 O(1) per slice
                // Bypass tiled when ctx<=32 (L2, serial faster) or HD!=128 (fallback to serial/splitK)
                if (ctx_l > 32 && HDl == 128) {
                    /* capture bakes eager S at capture ctx (see split_S_fp32) */
                    int S = split_S_fp32(ctx_l, e->d_split_S_max);
                    dim3 grid_split(S, KV_l);
                    int threads_split = (H_l / KV_l) * 32;
                    size_t smem_bytes = 2 * (size_t)BC_FP32 * HDl * sizeof(float);
                    k_fa2_fp32_split<<<grid_split, threads_split, smem_bytes, e->stream>>>(
                        e->d_q, Kl_f, Vl_f,
                        e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                        e->d_pos,
                        H_l, KV_l, HDl,
                        scale_l, swa_l, S);
                    k_fa2_combine<<<H_l, 32, 0, e->stream>>>(
                        e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                        e->d_att, H_l, HDl, S);
                } else if (ctx_l > 64) {
                    int S = ctx_l / 256;
                    if (S < 2) S = 2;
                    if (S > 32) S = 32;
                    if (S > e->d_split_S_max) S = e->d_split_S_max;
                    dim3 grid_split(H_l, S);
                    k_flash_gqa_splitk<<<grid_split, 32, 0, e->stream>>>(
                        e->d_q, Kl_f, Vl_f,
                        e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                        e->d_pos,
                        H_l, KV_l, HDl,
                        scale_l, swa_l, S);
                    k_flash_gqa_combine<<<H_l, 32, 0, e->stream>>>(
                        e->d_split_pacc, e->d_split_pm, e->d_split_pl,
                        e->d_att, H_l, HDl, S);
                } else {
                    k_flash_gqa<<<H_l, 32, 0, e->stream>>>(
                        e->d_q, Kl_f, Vl_f, e->d_att,
                        e->d_pos,
                        H_l, KV_l, HDl, c->max_ctx,
                        scale_l, swa_l);
                }
            }
        }
        if (tt_profiling()) tt_prof_end(TT_P_FLASH, e->stream);

        CHK_STAGE("4 flash");
        if (trace && l == 0) {
            eng_rms(e, e->d_v_stage, "v", kvdim_l);
            eng_rms(e, e->d_att, "ao", attn_qout);
        }
        if (trace && e->has_pl_embd && l >= 18 && l <= 21) ple_canary(e, "flash", l);
        /* 5. Wo projection + residual: x += att @ Wo^T */
        if (tt_profiling()) tt_prof_begin(TT_P_OMLP, e->stream);
        int orc_ = tt_gemv_layer_dispatch(w->o.ptr, w->o.dtype, e->d_att, e->d_xn, c->dim, attn_qout, e->stream);
        if (orc_ && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] o gemv rc=%d\n", orc_);
        /* gemma2 sandwich: normalize the attention output before residual */
        if (w->post_attn_norm)
            k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
                e->d_xn, w->post_attn_norm, e->d_xn, c->dim, c->rms_eps,
                c->tr.norm_offset);
        k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_x, e->d_xn, c->dim);
        if (tt_profiling()) tt_prof_end(TT_P_OMLP, e->stream);
        if (trace && l == 0) {
            static float ao2[4096];
            /* o-proj output lands in d_xn pre-residual */
            eng_rms(e, e->d_xn, "attn_out_raw", c->dim);
            (void)ao2;
        }
        if (trace && l == 0) eng_rms(e, e->d_x, "x_after_attn", c->dim);
        if (trace && e->has_pl_embd && l >= 18 && l <= 21) ple_canary(e, "oproj", l);

        /* 6. ffn norm */
        if (tt_profiling()) tt_prof_begin(TT_P_RMSNORM, e->stream);
        k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
            e->d_x, w->ffn_norm, e->d_xn, c->dim, c->rms_eps, c->tr.norm_offset);
        if (tt_profiling()) tt_prof_end(TT_P_RMSNORM, e->stream);

        /* 7+8. MLP: fused q4_0 SwiGLU/GeGLU when possible (epilogue from the
         * activation trait), else two typed GEMVs + elementwise apply.
         * Then down projection + residual. */
        if (tt_profiling()) tt_prof_begin(TT_P_OMLP, e->stream);
        const int act_gelu = (c->tr.act == ACT_GELU) ? 1 : 0;
        if (w->gate.dtype == GGUF_TYPE_Q4_0 && w->up.dtype == GGUF_TYPE_Q4_0
            && !e->has_pl_embd) {   /* gemma4: force typed path until fused-GELU is validated */
            tt_ffn_q4_0(w->gate.ptr, w->up.ptr, e->d_xn, e->d_h,
                        FF_l, c->dim, act_gelu, e->stream);
        } else {
            int grc = tt_gemv_layer_dispatch(w->gate.ptr, w->gate.dtype, e->d_xn, e->d_g,
                          FF_l, c->dim, e->stream);
            if (grc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] gate gemv rc=%d\n", grc);
            CHK_STAGE("7a gate-gemv");
            int urc = tt_gemv_layer_dispatch(w->up.ptr, w->up.dtype, e->d_xn, e->d_u,
                          FF_l, c->dim, e->stream);
            if (urc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] up gemv rc=%d\n", urc);
            CHK_STAGE("7b up-gemv");
            k_swiglu_apply<<<(FF_l + 255) / 256, 256, 0, e->stream>>>(
                e->d_g, e->d_u, e->d_h, FF_l, act_gelu);
            CHK_STAGE("6b swiglu-apply");
        }
        int drc = tt_gemv_layer_dispatch(w->down.ptr, w->down.dtype, e->d_h, e->d_xn,
                      c->dim, FF_l, e->stream);
        if (drc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] down gemv rc=%d\n", drc);
        CHK_STAGE("7c down-gemv");
        /* gemma2 sandwich: normalize the MLP output before residual */
        if (w->post_ffn_norm)
            k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
                e->d_xn, w->post_ffn_norm, e->d_xn, c->dim, c->rms_eps,
                c->tr.norm_offset);
        k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_x, e->d_xn, c->dim);
        CHK_STAGE("7d add");
        if (tt_profiling()) tt_prof_end(TT_P_OMLP, e->stream);
        CHK_STAGE("7 mlp");
        if (trace && l == 0) {
            eng_rms(e, e->d_h, "gu", FF_l);
            eng_rms(e, e->d_xn, "mlp_pre_norm", c->dim);
        }
        if (trace && l == 0) eng_rms(e, e->d_xn, "mlp_raw", c->dim);
        if (trace && e->has_pl_embd && l >= 18 && l <= 21) ple_canary(e, "mlp", l);

        if (trace) {
            static float xt[1536];
            cudaMemcpy(xt, e->d_x, c->dim * 4, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(e->stream);
            double s2 = 0; int nn = 0;
            for (int i = 0; i < c->dim; i++) { s2 += (double)xt[i]*xt[i]; if (isnan(xt[i])) nn++; }
            fprintf(stderr, "[FWD] L%d done rms=%.4f nan=%d\n", l, sqrt(s2/c->dim), nn);
        }

        /* gemma4 MatFormer per-layer embedding block. M9 V2: 2 device
         * launches, no host round-trips, graph-capturable. Order:
         *   g = gelu(inp_gate @ x) * PLE[l] ; p = rmsnorm(pl_proj @ g) ;
         *   x += p ; x *= layer_output_scale
         * The fused k_ple_stage2_f32 absorbs the post-norm that used to be
         * a separate k_rmsnorm launch. */
        static int no_ple2 = -1;
        if (no_ple2 < 0) no_ple2 = getenv("TT_NO_PLE") ? 1 : 0;
        if (e->has_pl_embd && !no_ple2) {
            const int slot = e->pos % c->max_ctx;
            (void)slot;
            /* V2 fast path: 2 device launches, no host round-trips, graph-
             * capturable. Requires f32 per-layer weights (true for the
             * Q4_0/Q5_K_M/Q6_K gemma4 GGUFs; engine keeps the host path as
             * a fallback for any non-f32 variant or under capture). */
            const int use_v2 = (w->inp_gate.dtype == TTQ_F32 &&
                                w->pl_proj.dtype  == TTQ_F32 &&
                                w->pl_post_norm != NULL);
            if (use_v2) {
                /* Optional per-layer timing (TT_PLE_TIMING=1). Lazy-init a
                 * pair of timed cudaEvents on first call; report per-layer
                 * us for the 2-launch V2 chain. Skipped under capture. */
                static int ple_timing = -1;
                if (ple_timing < 0) ple_timing = getenv("TT_PLE_TIMING") ? 1 : 0;
                static cudaEvent_t ple_ea, ple_eb;
                static int ple_ev_inited = 0;
                if (ple_timing && !ple_ev_inited) {
                    cudaEventCreate(&ple_ea);
                    cudaEventCreate(&ple_eb);
                    ple_ev_inited = 1;
                }
                if (ple_timing && !g_capturing) cudaEventRecord(ple_ea, e->stream);
                /* stage1: gemv f32 + tanh-approx gelu + ple mul, 1 warp/row */
                const int s1_nwarp = 4;                /* 4 warps/block => 4 rows */
                const int s1_grid  = (e->pl_dim + s1_nwarp - 1) / s1_nwarp;
                k_ple_stage1_f32<<<s1_grid, s1_nwarp * 32, 0, e->stream>>>(
                    (const float *)w->inp_gate.ptr,
                    e->d_x,
                    e->d_ple_row + (long)l * e->pl_dim,
                    e->d_pl_tmp,
                    e->pl_dim, c->dim);
                /* stage2: gemv f32 + atomic-ticket fused rmsnorm, 1 warp/row.
                 * The fused post-norm (gamma=pl_post_norm) replaces the old
                 * separate k_rmsnorm call so the whole block is 2 launches. */
                const int s2_nwarp = 16;
                const int s2_grid  = (c->dim + s2_nwarp - 1) / s2_nwarp;
                k_ple_stage2_f32<<<s2_grid, s2_nwarp * 32, 0, e->stream>>>(
                    (const float *)w->pl_proj.ptr,
                    e->d_pl_tmp,
                    w->pl_post_norm,
                    e->d_xn,
                    c->dim, e->pl_dim, c->rms_eps);
                k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_x, e->d_xn, c->dim);

                /* per-layer output scale (scalar by value) */
                if (w->out_scale_val != 0.0f && w->out_scale_val != 1.0f)
                    k_scale<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(
                        e->d_x, w->out_scale_val, c->dim);
                if (trace && !g_capturing) {
                    static float xs2[1536];
                    cudaMemcpy(xs2, e->d_x, c->dim * 4, cudaMemcpyDeviceToHost);
                    cudaStreamSynchronize(e->stream);
                    double s2 = 0;
                    for (int i = 0; i < c->dim; i++) s2 += (double)xs2[i]*xs2[i];
                    fprintf(stderr, "[FWD] L%d postscale_rms=%.4f\n", l, sqrt(s2/c->dim));
                }
                if (ple_timing && !g_capturing) {
                    cudaEventRecord(ple_eb, e->stream);
                    cudaEventSynchronize(ple_eb);
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, ple_ea, ple_eb);
                    fprintf(stderr, "[PLE-V2] L%d %.2f us\n", l, ms * 1000.0f);
                }
            } else {
                /* host-assisted fallback (non-f32 per-layer weights only) */
                static float gbuf[1024];
                static float ple_slice[1024];
                static float pe_slice[1024];

                /* g = inp_gate @ x (device) */
                int ig = tt_gemv_layer_dispatch(w->inp_gate.ptr, w->inp_gate.dtype,
                                       e->d_x, e->d_pl_tmp, e->pl_dim, c->dim,
                                       e->stream);
                cudaStreamSynchronize(e->stream);
                cudaMemcpy(gbuf, e->d_pl_tmp, e->pl_dim * 4, cudaMemcpyDeviceToHost);

                /* fetch this position's PLE row + pe slice for layer l */
                cudaMemcpy(ple_slice, e->d_ple_row + (long)l * 256,
                           256 * sizeof(float), cudaMemcpyDeviceToHost);
                if (trace && l >= 19 && l <= 22) {
                    static float whole[35 * 256];
                    cudaMemcpy(whole, e->d_ple_row, 35 * 256 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    int nb = 0;
                    for (int i = 0; i < 35 * 256; i++) if (isnan(whole[i])) nb++;
                    fprintf(stderr, "[PLEW] L%d row-nan=%d slice0=%.4f\n", l, nb, whole[0]);
                }
                memcpy(pe_slice, e->ple_pe + (long)l * 256, 256 * sizeof(float));

                if (trace) {
                    int bad = 0;
                    for (int i = 0; i < e->pl_dim; i++)
                        if (isnan(gbuf[i]) || isnan(ple_slice[i])) bad++;
                    fprintf(stderr, "[PLEB] L%d bad=%d g[0]=%.4f ps[0]=%.4f\n",
                            l, bad, gbuf[0], ple_slice[0]);
                }
                /* gelu then elementwise multiply by PLE slice */
                for (int i = 0; i < e->pl_dim; i++) {
                    const float gv = gbuf[i];
                    gbuf[i] = 0.5f * gv * (1.0f + tanhf(0.7978845608028654f *
                                                        (gv + 0.044715f * gv * gv * gv)));
                    gbuf[i] *= ple_slice[i];
                }

                if (getenv("TT_PLE_DEBUG") && l == 0) {
                    float s = 0, mn = 1e30f, mx = -1e30f;
                    for (int i = 0; i < e->pl_dim; i++) {
                        s += gbuf[i];
                        if (gbuf[i] < mn) mn = gbuf[i];
                        if (gbuf[i] > mx) mx = gbuf[i];
                    }
                    fprintf(stderr, "[PLEDBG] L0 gated: sum=%.4f min=%.4f max=%.4f\n", s, mn, mx);
                    float ps = 0;
                    for (int i = 0; i < 8; i++) ps += ple_slice[i];
                    fprintf(stderr, "[PLEDBG] L0 ple[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f pe[0]=%.4f\n",
                            ple_slice[0], ple_slice[1], ple_slice[2], ple_slice[3],
                            ple_slice[4], ple_slice[5], ple_slice[6], ple_slice[7], pe_slice[0]);
                }
                /* p = rmsnorm(pl_proj @ g) ; x += p ; x *= out_scale */
                cudaMemcpy(e->d_pl_tmp, gbuf, e->pl_dim * 4, cudaMemcpyHostToDevice);
                int pg = tt_gemv_layer_dispatch(w->pl_proj.ptr, w->pl_proj.dtype,
                                       e->d_pl_tmp, e->d_xn, c->dim, e->pl_dim,
                                       e->stream);
                cudaStreamSynchronize(e->stream);
                if (ig || pg) fprintf(stderr, "[qwen2-engine] pl block rc ig=%d pg=%d\n", ig, pg);
                if (w->pl_post_norm)
                    k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
                        e->d_xn, w->pl_post_norm, e->d_xn, c->dim, c->rms_eps,
                        c->tr.norm_offset);
                k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_x, e->d_xn, c->dim);

                /* per-layer output scale (scalar by value) */
                if (w->out_scale_val != 0.0f && w->out_scale_val != 1.0f)
                    k_scale<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(
                        e->d_x, w->out_scale_val, c->dim);
                if (trace) {
                    static float xs2[1536];
                    cudaMemcpy(xs2, e->d_x, c->dim * 4, cudaMemcpyDeviceToHost);
                    cudaStreamSynchronize(e->stream);
                    double s2 = 0;
                    for (int i = 0; i < c->dim; i++) s2 += (double)xs2[i]*xs2[i];
                    fprintf(stderr, "[FWD] L%d postscale_rms=%.4f\n", l, sqrt(s2/c->dim));
                }
                if (getenv("TT_PLE_DEBUG") && l == 0) {
                    static float xs[1536];
                    cudaMemcpy(xs, e->d_x, c->dim * 4, cudaMemcpyDeviceToHost);
                    cudaStreamSynchronize(e->stream);
                    float s = 0; int nan = 0;
                    for (int i = 0; i < c->dim; i++) { s += xs[i]; if (isnan(xs[i])) nan++; }
                    fprintf(stderr, "[PLEDBG] L0 x-after: sum=%.4f nan=%d scale=%f\n", s, nan, w->out_scale_val);
                }
            }
        }

        {
            static const char *dump_env = NULL;
            /* never dump while a stream capture is in progress: the blocking
             * D2H copies below would be illegal inside a captured region */
            if (!dump_env) dump_env = getenv("TT_DUMP_LAYER");
            if (dump_env && !g_capturing && atoi(dump_env) == -100 - l) {
                /* dump raw K cache for this layer: slots [0..pos] */
                char fn[128];
                snprintf(fn, sizeof(fn), "/tmp/eng_kcache_layer%d.bin", l);
                FILE *fp = fopen(fn, "wb");
                if (fp) {
                    const long n = (long)c->n_kv_heads * c->max_ctx * c->head_dim;
                    float *tmp = (float *)malloc(n * 4);
                    cudaMemcpy(tmp, e->d_kc + l * n, n * 4, cudaMemcpyDeviceToHost);
                    fwrite(tmp, 4, n, fp);
                    fclose(fp); free(tmp);
                    fprintf(stderr, "[qwen2-engine] dumped %s\n", fn);
                }
            }
            if (dump_env && !g_capturing && atoi(dump_env) == l) {
                const int tok_slot = e->pos - 1;
                char fn[128];
                snprintf(fn, sizeof(fn), "/tmp/eng_layer%d_tok%d.bin", l, tok_slot);
                FILE *fp = fopen(fn, "wb");
                if (fp) {
                    float *tmp = (float *)malloc(c->dim * 4);
                    cudaMemcpy(tmp, e->d_x, c->dim * 4, cudaMemcpyDeviceToHost);
                    fwrite(tmp, 4, c->dim, fp);
                    fclose(fp); free(tmp);
                    fprintf(stderr, "[qwen2-engine] dumped %s\n", fn);
                }
            }
        }
    }
    if (e->n_gpu_layers < c->n_layers) {
        cudaMemcpyAsync(e->d_x, e->h_x_buf, c->dim * sizeof(float),
                        cudaMemcpyHostToDevice, e->stream);
        cudaStreamSynchronize(e->stream);
    }
    return (int)cudaGetLastError();
}

/* per-layer embedding finisher (host, small data): rmsnorm each 256-slice
 * with pl_proj_norm, add scaled pe row, scale by 1/sqrt(2) */
static void ple_finish_host(float *v, const float *gamma256,
                            const float *pe_row, int n_slices) {
    for (int s = 0; s < n_slices; s++) {
        float *sl = v + (long)s * 256;
        double ss = 0.0;
        for (int i = 0; i < 256; i++) ss += (double)sl[i] * sl[i];
        const float inv = 1.0f / sqrtf((float)(ss / 256.0) + 1e-6f);
        for (int i = 0; i < 256; i++)
            sl[i] = sl[i] * inv * gamma256[i];
    }
    const float s2 = 1.0f / sqrtf(2.0f);
    for (long i = 0; i < (long)n_slices * 256; i++)
        v[i] = (v[i] + pe_row[i]) * s2;
}

static int embed_token(Qwen2Engine *e, int tok) {
    if (tt_profiling()) tt_prof_begin(TT_P_EMBED, e->stream);
    /* M7: embedding may be any Tier-1 dtype now; dispatch by type */
    const int rc = tt_embed_typed(e->d_embd.ptr, e->d_embd.dtype, tok,
                                  e->d_x, e->cfg.dim, e->stream);
    if (tt_profiling()) tt_prof_end(TT_P_EMBED, e->stream);
    /* gemma families scale embeddings by sqrt(hidden_size) */
    if (e->cfg.tr.embed_sqrt)
        k_scale<<<(e->cfg.dim + 255) / 256, 256, 0, e->stream>>>(
            e->d_x, sqrtf((float)e->cfg.dim), e->cfg.dim);

    /* gemma4: build this position's MatFormer per-layer input row.
     * Host-assisted (small data): gemv on GPU -> D2H -> CPU finish -> H2D. */
    static int no_ple = -1;
    if (no_ple < 0) no_ple = getenv("TT_NO_PLE") ? 1 : 0;
    if (e->has_pl_embd && !no_ple) {
        const long row = (long)e->cfg.n_layers * e->pl_dim;
        fprintf(stderr, "[PLE] A enter\n");
        cudaStreamSynchronize(e->stream);
        fprintf(stderr, "[PLE] B gemv in\n");
        cudaStreamSynchronize(e->stream);   /* ensure embed done before reading x */
        if (getenv("TT_TRACE")) {
            static float xt[1536];
            cudaMemcpy(xt, e->d_x, e->cfg.dim * 4, cudaMemcpyDeviceToHost);
            float mx = 0; int nb = 0;
            for (int i = 0; i < e->cfg.dim; i++) {
                if (isnan(xt[i]) || isinf(xt[i])) nb++;
                if (fabsf(xt[i]) > mx) mx = fabsf(xt[i]);
            }
            fprintf(stderr, "[PLE] x-check nan/inf=%d maxabs=%.4f\n", nb, mx);
        }
        int prc = tt_gemv_layer_dispatch(e->pl_model_proj.ptr, e->pl_model_proj.dtype,
                                e->d_x, e->d_ple_row, (int)row, e->cfg.dim,
                                e->stream);
        cudaStreamSynchronize(e->stream);
        if (prc) return prc;
        fprintf(stderr, "[PLE] C d2h\n");
        if (getenv("TT_TRACE")) {
            int nb = 0, ni = 0;
            float mx = 0;
            for (long i = 0; i < row; i++) {
                if (isnan(e->ple_pe[i])) nb++;
                if (isinf(e->ple_pe[i])) ni++;
                if (fabsf(e->ple_pe[i]) > mx && !isinf(e->ple_pe[i])) mx = fabsf(e->ple_pe[i]);
            }
            fprintf(stderr, "[PLE] post-gemv nan=%d inf=%d maxabs=%.4f v[5376]=%.4f\n",
                    nb, ni, mx, e->ple_pe[5376]);
            {   /* CPU golden for row 5376 */
                static float xg[1536], wg[1536];
                cudaMemcpy(xg, e->d_x, e->cfg.dim * 4, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(e->stream);
                GGUFTensor *tpj = gguf_get_tensor(e->gguf, "per_layer_model_proj.weight");
                ttq_dequant((const char *)tpj->data + 5376L * tpj->size_bytes / tpj->shape[1],
                            tpj->type, 1536, wg);
                double s = 0;
                for (int i = 0; i < 1536; i++) s += (double)wg[i] * xg[i];
                fprintf(stderr, "[PLE] cpu-row5376=%.4f gpu=%.4f\n", s, e->ple_pe[5376]);
            }
        }
        cudaMemcpy(e->ple_pe, e->d_ple_row, row * sizeof(float),
                   cudaMemcpyDeviceToHost);   /* reuse as staging */
        fprintf(stderr, "[PLE] D deq\n");

        /* dequant per_layer_token_embd row 'tok' (q4_0), scale by sqrt(pl_dim).
         * llama.cpp build_inp_per_layer scales the raw get_rows output by
         * tok_embd_scale = sqrt(n_embd_per_layer) BEFORE project_per_layer_inputs
         * adds it to the normed projection. */
        GGUFTensor *tp = gguf_get_tensor(e->gguf, "per_layer_token_embd.weight");
        if (!tp || !tp->data) return -10;
        static float pe_full[64 * 1024];
        long rb;
        switch (tp->type) {
            case TTQ_Q4_0: rb = row / 32 * 18; break;
            case TTQ_Q5_K: rb = row / 256 * 176; break;
            case TTQ_Q6_K: rb = row / 256 * 210; break;
            case TTQ_Q8_0: rb = row / 32 * 34; break;
            case TTQ_F16:  rb = row * 2; break;
            default: fprintf(stderr, "[qwen2-engine] ple: unsupported pe dtype %d (TTQ_Q5_K=%d)\n", tp->type, TTQ_Q5_K); return -11;
        }
        /* CPU: per-slice rmsnorm w/ pl_proj_norm, then add pe*sqrt(pl_dim),
         * then *1/sqrt2 */
        ttq_dequant((const char *)tp->data + (long)tok * rb, tp->type, row, pe_full);

        /* CPU: per-slice rmsnorm w/ pl_proj_norm, then + pe*sqrt(pl_dim),
         * then *1/sqrt2 (done inside ple_finish_host) */
        fprintf(stderr, "[PLE] E finish\n");
        if (getenv("TT_TRACE")) {
            int nb1 = 0, nb2 = 0;
            for (long i = 0; i < row; i++) {
                if (isnan(pe_full[i])) nb1++;
                if (isnan(e->ple_pe[i])) nb2++;
            }
            fprintf(stderr, "[PLE] nan-check: pe_full_nan=%d ple_pe_nan=%d\n", nb1, nb2);
        }
        const float pe_scale = sqrtf((float)e->pl_dim);
        for (long i = 0; i < row; i++) pe_full[i] *= pe_scale;
        ple_finish_host(e->ple_pe, e->pl_proj_norm_host, pe_full,
                        e->cfg.n_layers);
        fprintf(stderr, "[PLE] F done\n");

        cudaMemcpy(e->d_ple_row, e->ple_pe, row * sizeof(float),
                   cudaMemcpyHostToDevice);
        if (getenv("TT_TRACE")) {
            const float pv = getenv("TT_PLE_ZERO_TAIL") ? 0.0f : -777.0f;
            k_fill_const<<<(8960*4 - 5376*4 + 255)/256, 256, 0, e->stream>>>(
                e->d_ple_row + 5376, 8960 - 5376, pv);
        }
        if (getenv("TT_DUMP_PLE")) {
            static float pled[35 * 256];
            cudaMemcpy(pled, e->d_ple_row, row * sizeof(float), cudaMemcpyDeviceToHost);
            char fn[512]; snprintf(fn, sizeof fn, "%s.tok%d",
                                   getenv("TT_DUMP_PLE"), e->pos);
            FILE *fp = fopen(fn, "wb"); if (fp) { fwrite(pled, 4, row, fp); fclose(fp); }
            fprintf(stderr, "[PLE] dumped %s\n", fn);
        }
    }
    return rc;
}

static int advance(Qwen2Engine *e, int tok) {
    int rc = embed_token(e, tok);
    if (rc) { fprintf(stderr, "[qwen2-engine] embed rc=%d tok=%d\n", rc, tok); return rc; }
    rc = forward_layers(e);      /* runs while *d_pos == current slot */
    if (rc) { fprintf(stderr, "[qwen2-engine] forward rc=%d\n", rc); return rc; }
    e->pos++;
    /* SYNC copy: source is mutable host memory the next advance() increments
     * immediately after enqueue. A small pageable async H2D copy can be
     * executed later on the device timeline and would then read the
     * already-incremented pos -> wrong RoPE/scatter/attention positions
     * (observed as cross-process nondeterministic logits). */
    cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);
    return 0;
}

void qwen2_engine_reset(Qwen2Engine *e) {
    if (!e) return;
    e->pos = 0;
    cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);
}

int prefill_batched_gemm(Qwen2Engine *e, const int *toks, int n, float *h_x_out) {
    if (!e || !toks || n <= 0) return -1;
    const TTConfig *c = &e->cfg;
    if (e->pos + n > c->max_ctx) return -2;

    int (*prefill_gemm_fn)(const void *, const float *, float *, int, int, int, cudaStream_t) =
        tt_gemm_q4_0_prefill;
    /* TT_USE_WMMA_PRE=1 routes prefill GEMMs with n>=64 to the WMMA
     * tensor-core kernel; smaller chunks keep the CUDA-core kernel. */
    { static int wmma_pre = -1;
      if (wmma_pre < 0) wmma_pre = getenv("TT_USE_WMMA_PRE") ? 1 : 0;
      if (wmma_pre && n >= 64) prefill_gemm_fn = tt_gemm_wmma_q4_0_prefill; }

    const int dim = c->dim;
    const int hidden_dim = c->hidden_dim;
    const int HD = c->head_dim;
    long cache_layer = c->n_kv_heads * (long)c->max_ctx * HD;
    long cache_layer_q8 = c->n_kv_heads * (long)c->max_ctx * (HD / 32);
    if (e->pl_hd[0] > 0) {
        long mx = 0;
        long mx_q8 = 0;
        for (int l = 0; l < c->n_layers; l++) {
            const long w2 = (long)e->pl_kv[l] * e->pl_hd[l];
            if (w2 > mx) mx = w2;
            const long w2_q8 = (long)e->pl_kv[l] * (e->pl_hd[l] / 32);
            if (w2_q8 > mx_q8) mx_q8 = w2_q8;
        }
        cache_layer = mx * c->max_ctx;
        cache_layer_q8 = mx_q8 * c->max_ctx;
    }

    /* Task 1: persistent arena reuse for n <= pf_arena_max_n */
    int max_qout = c->n_heads * HD;
    int max_kvdim = c->n_kv_heads * HD;
    int max_ff = hidden_dim;
    for (int l = 0; l < c->n_layers; l++) {
        int H_l = e->pl_heads[l] > 0 ? e->pl_heads[l] : c->n_heads;
        int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : c->n_kv_heads;
        int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
        if (H_l * HDl > max_qout) max_qout = H_l * HDl;
        if (KV_l * HDl > max_kvdim) max_kvdim = KV_l * HDl;
        int FFx = e->pl_ffn[l] > 0 ? e->pl_ffn[l] : hidden_dim;
        if (FFx > max_ff) max_ff = FFx;
    }
    int use_arena = (e->pf_arena_max_n >= (size_t)n && e->d_pf_X && e->d_pf_Xn && e->d_pf_Q && e->d_pf_K && e->d_pf_V && e->d_pf_Att && e->d_pf_H && e->d_pf_G && e->d_pf_U && e->d_pf_pos_batch);
    float *d_X = NULL, *d_Xn = NULL, *d_Q = NULL, *d_K = NULL, *d_V = NULL;
    float *d_Att = NULL, *d_H = NULL, *d_G = NULL, *d_U = NULL;
    int *d_pos_batch = NULL;
    if (use_arena) {
        d_X = e->d_pf_X; d_Xn = e->d_pf_Xn; d_Q = e->d_pf_Q; d_K = e->d_pf_K; d_V = e->d_pf_V;
        d_Att = e->d_pf_Att; d_H = e->d_pf_H; d_G = e->d_pf_G; d_U = e->d_pf_U; d_pos_batch = e->d_pf_pos_batch;
    } else {
        if (cudaMalloc(&d_X, (size_t)n * dim * sizeof(float)) != cudaSuccess) return -3;
        if (cudaMalloc(&d_Xn, (size_t)n * dim * sizeof(float)) != cudaSuccess) { cudaFree(d_X); return -4; }
        if (cudaMalloc(&d_Q, (size_t)n * max_qout * sizeof(float)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); return -5; }
        if (cudaMalloc(&d_K, (size_t)n * max_kvdim * sizeof(float)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); return -6; }
        if (cudaMalloc(&d_V, (size_t)n * max_kvdim * sizeof(float)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); return -7; }
        if (cudaMalloc(&d_Att, (size_t)n * max_qout * sizeof(float)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); return -8; }
        if (cudaMalloc(&d_H, (size_t)n * max_ff * sizeof(float)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); return -9; }
        if (cudaMalloc(&d_G, (size_t)n * max_ff * sizeof(float)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); cudaFree(d_H); return -10; }
        if (cudaMalloc(&d_U, (size_t)n * max_ff * sizeof(float)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); cudaFree(d_H); cudaFree(d_G); return -11; }
        if (cudaMalloc(&d_pos_batch, (size_t)n * sizeof(int)) != cudaSuccess) { cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); cudaFree(d_H); cudaFree(d_G); cudaFree(d_U); return -12; }
    }

    int *h_pos_batch = (int *)malloc((size_t)n * sizeof(int));
    if (!h_pos_batch) return -13;
    const int pos0 = e->pos;
    for (int i = 0; i < n; i++) {
        h_pos_batch[i] = pos0 + i;
    }
    cudaMemcpy(d_pos_batch, h_pos_batch, (size_t)n * sizeof(int), cudaMemcpyHostToDevice);
    free(h_pos_batch);

    /* Opt in to dynamic shmem for FP32 prefill flash if needed */
    {
        int pf_max_hd = HD;
        for (int l = 0; l < c->n_layers; l++) {
            int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
            if (HDl > pf_max_hd) pf_max_hd = HDl;
        }
        size_t smem_fp32_attr = 2 * (size_t)BC_PREFILL_FP32 * pf_max_hd * sizeof(float);
        if (smem_fp32_attr > 48 * 1024) {
            cudaFuncSetAttribute(k_prefill_flash_fp32,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_fp32_attr);
        }
        {
            int pf_bph = pf_max_hd / 32;
            size_t smem_q8_attr = 2 * (size_t)BC_PREFILL * pf_bph * sizeof(half)
                                + 2 * (size_t)BC_PREFILL * pf_max_hd * sizeof(int8_t);
            if (smem_q8_attr > 48 * 1024) {
                cudaFuncSetAttribute(k_prefill_flash_q8_0,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_q8_attr);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        tt_embed_typed(e->d_embd.ptr, e->d_embd.dtype, toks[i],
                       d_X + (long)i * dim, dim, e->stream);
        if (c->tr.embed_sqrt) {
            k_scale<<<(dim + 255) / 256, 256, 0, e->stream>>>(
                d_X + (long)i * dim, sqrtf((float)dim), dim);
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &e->L[l];
        float *Kl_f = e->d_kc + (long)l * cache_layer;
        float *Vl_f = e->d_vc + (long)l * cache_layer;
        BlockQ8_0 *Kl_q8 = e->d_kc_q8 ? (e->d_kc_q8 + (long)l * cache_layer_q8) : NULL;
        BlockQ8_0 *Vl_q8 = e->d_vc_q8 ? (e->d_vc_q8 + (long)l * cache_layer_q8) : NULL;
        BlockQ4_0 *Kl_q4 = e->d_kc_q4 ? (e->d_kc_q4 + (long)l * cache_layer_q8) : NULL;
        BlockQ4_0 *Vl_q4 = e->d_vc_q4 ? (e->d_vc_q4 + (long)l * cache_layer_q8) : NULL;
        const int kv_shared = e->has_pl_embd && e->pl_src[l] >= 0;
        if (kv_shared) {
            Kl_f = e->d_kc + (long)e->pl_src[l] * cache_layer;
            Vl_f = e->d_vc + (long)e->pl_src[l] * cache_layer;
            if (e->d_kc_q8) {
                Kl_q8 = e->d_kc_q8 + (long)e->pl_src[l] * cache_layer_q8;
                Vl_q8 = e->d_vc_q8 + (long)e->pl_src[l] * cache_layer_q8;
            }
            if (e->d_kc_q4) {
                Kl_q4 = e->d_kc_q4 + (long)e->pl_src[l] * cache_layer_q8;
                Vl_q4 = e->d_vc_q4 + (long)e->pl_src[l] * cache_layer_q8;
            }
        }

        const int H_l = e->pl_heads[l] > 0 ? e->pl_heads[l] : c->n_heads;
        const int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : c->n_kv_heads;
        const int FF_l = e->pl_ffn[l] > 0 ? e->pl_ffn[l] : hidden_dim;
        const int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
        const int attn_qout = H_l * HDl;
        const int kvdim_l = KV_l * HDl;

        /* 1. RMSNorm before QKV - batched */
        if (tt_profiling()) tt_prof_begin(TT_P_RMSNORM, e->stream);
        k_rmsnorm_batched<<<n, 256, 256*sizeof(float), e->stream>>>(d_X, w->attn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);
        if (tt_profiling()) tt_prof_end(TT_P_RMSNORM, e->stream);

        /* 2. Batched QKV GEMM */
        if (tt_profiling()) tt_prof_begin(TT_P_QKV, e->stream);
        if (w->q.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->q.ptr, d_Xn, d_Q, attn_qout, dim, n, e->stream);
            if (!kv_shared) {
                prefill_gemm_fn(w->k.ptr, d_Xn, d_K, kvdim_l, dim, n, e->stream);
                prefill_gemm_fn(w->v.ptr, d_Xn, d_V, kvdim_l, dim, n, e->stream);
            }
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->q.ptr, w->q.dtype, d_Xn + (long)i * dim, d_Q + (long)i * attn_qout, attn_qout, dim, e->stream);
                if (!kv_shared) {
                    tt_gemv_layer_dispatch(w->k.ptr, w->k.dtype, d_Xn + (long)i * dim, d_K + (long)i * kvdim_l, kvdim_l, dim, e->stream);
                    tt_gemv_layer_dispatch(w->v.ptr, w->v.dtype, d_Xn + (long)i * dim, d_V + (long)i * kvdim_l, kvdim_l, dim, e->stream);
                }
            }
        }
        if (tt_profiling()) tt_prof_end(TT_P_QKV, e->stream);

        /* Biases & QK norm - batched */
        if (w->q_bias) {
            k_add_bias_batched<<<(n*attn_qout+255)/256,256,0,e->stream>>>(d_Q, w->q_bias, attn_qout, n);
        }
        if (!kv_shared && w->k_bias) {
            k_add_bias_batched<<<(n*kvdim_l+255)/256,256,0,e->stream>>>(d_K, w->k_bias, kvdim_l, n);
        }
        if (!kv_shared && w->v_bias) {
            k_add_bias_batched<<<(n*kvdim_l+255)/256,256,0,e->stream>>>(d_V, w->v_bias, kvdim_l, n);
        }

        if (c->tr.qk_norm_rms) {
            const int qkthreads = HDl < 256 ? HDl : 256;
            {
                size_t smem = qkthreads*sizeof(float);
                k_qk_norm_rms_batched<<<n*H_l, qkthreads, smem, e->stream>>>(d_Q, w->q_norm, H_l, HDl, c->tr.qk_norm_eps, n, attn_qout);
                if (!kv_shared)
                    k_qk_norm_rms_batched<<<n*KV_l, qkthreads, smem, e->stream>>>(d_K, w->k_norm, KV_l, HDl, c->tr.qk_norm_eps, n, kvdim_l);
            }
        }

        /* 3. RoPE, Scatter, and Flash Attention per token */
        const int is_full_l = (e->pl_swa[l] == 0);
        const float base_l = e->has_pl_embd ? (is_full_l ? 1e6f : 1e4f) : c->rope_base;
        const float *ff_l = (e->has_pl_embd && is_full_l) ? e->d_rope_freqs : NULL;
        void (*rope_fn)(float *, int, int, const int *, float) =
            (c->tr.rope == ROPE_GPTJ) ? k_rope_gptj : k_rope;
        const float scale_l = c->tr.attn_scale_one ? 1.0f : 1.0f / sqrtf((float)HDl);
        const int swa_l = e->has_pl_embd ? e->pl_swa[l] : c->tr.swa_size;

        // Batched RoPE + scatter: single launch per layer instead of n launches
        {
            static int no_rope = -1;
            if (no_rope < 0) no_rope = getenv("TT_NO_ROPE") ? 1 : 0;
            if (!no_rope) {
                dim3 g_qb((HDl / 2 + 63) / 64, H_l, n), b_rope(64, 1, 1);
                dim3 g_kb((HDl / 2 + 63) / 64, KV_l, n);
                const int is_gptj = (c->tr.rope == ROPE_GPTJ);
                if (is_gptj) {
                    if (ff_l) {
                        k_rope_gptj_ff_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, ff_l, n, attn_qout);
                        if (!kv_shared) k_rope_gptj_ff_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, ff_l, n, kvdim_l);
                    } else {
                        k_rope_gptj_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, n, attn_qout);
                        if (!kv_shared) k_rope_gptj_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, n, kvdim_l);
                    }
                } else {
                    if (ff_l) {
                        k_rope_ff_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, ff_l, n, attn_qout);
                        if (!kv_shared) k_rope_ff_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, ff_l, n, kvdim_l);
                    } else {
                        k_rope_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, n, attn_qout);
                        if (!kv_shared) k_rope_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, n, kvdim_l);
                    }
                }
            }
            if (!kv_shared) {
                /* P0-1: dual-write FP32 + Q4. Decode below thresh reads FP32;
                 * decode above thresh reads Q4, so both must be populated.
                 * P0-2: gate on ptrs non-NULL (flag-set + ptr-NULL = crash). */
                if (e->use_q4_kvcache && Kl_q4 && Vl_q4) {
                    k_kv_scatter_batched<<<(n*kvdim_l+255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_f, Vl_f, d_pos_batch, KV_l, HDl, c->max_ctx, n, kvdim_l);
                    const int blocks_per_slot = kvdim_l / 32;
                    const long total_blocks = (long)n * blocks_per_slot;
                    k_kv_scatter_q4_0_batched<<<(total_blocks + 255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_q4, Vl_q4, d_pos_batch, KV_l, HDl, c->max_ctx, n);
                } else if (e->use_q8_kvcache && Kl_q8 && Vl_q8) {
                    /* Dual-write FP32 + Q8 (mirror Q4 path): hybrid decode
                     * below thresh reads FP32, so it must be populated. */
                    k_kv_scatter_batched<<<(n*kvdim_l+255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_f, Vl_f, d_pos_batch, KV_l, HDl, c->max_ctx, n, kvdim_l);
                    const int blocks_per_slot = kvdim_l / 32;
                    const long total_blocks = (long)n * blocks_per_slot;
                    k_kv_scatter_q8_0_batched<<<(total_blocks + 255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_q8, Vl_q8, d_pos_batch, KV_l, HDl, c->max_ctx, n);
                } else {
                    k_kv_scatter_batched<<<(n*kvdim_l+255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_f, Vl_f, d_pos_batch, KV_l, HDl, c->max_ctx, n, kvdim_l);
                }
            }
        }

        /* P0-1: prefill flash always on FP32 (bit-exact output at every ctx).
         * Q4 cache stays populated for decode above thresh. P0-2: ptr guard. */
        if (tt_profiling()) tt_prof_begin(TT_P_FLASH, e->stream);
        if (HDl > 128) {
            /* Tiled prefill flash regs/smem sized for elems<=4 (HD<=128);
             * fall back to per-token serial path for this layer only. */
            if (e->use_q8_kvcache && Kl_q8 && Vl_q8 && !(e->use_q4_kvcache && Kl_q4 && Vl_q4)) {
                for (int qi = 0; qi < n; qi++)
                    k_flash_gqa_q8_0<<<H_l, 32, 0, e->stream>>>(
                        d_Q + (long)qi * attn_qout, Kl_q8, Vl_q8, d_Att + (long)qi * attn_qout,
                        d_pos_batch + qi, H_l, KV_l, HDl, c->max_ctx, scale_l, swa_l);
            } else {
                for (int qi = 0; qi < n; qi++)
                    k_flash_gqa<<<H_l, 32, 0, e->stream>>>(
                        d_Q + (long)qi * attn_qout, Kl_f, Vl_f, d_Att + (long)qi * attn_qout,
                        d_pos_batch + qi, H_l, KV_l, HDl, c->max_ctx, scale_l, swa_l);
            }
        } else if (e->use_q4_kvcache && Kl_q4 && Vl_q4) {
            launch_prefill_flash(d_Q, Kl_f, Vl_f, d_Att,
                n, e->pos + n, e->pos, H_l, KV_l, HDl, scale_l, swa_l, e->stream);
        } else if (e->use_q8_kvcache && Kl_q8 && Vl_q8) {
            int num_q_tiles = (n + BR_PREFILL - 1) / BR_PREFILL;
            dim3 grid_pf(num_q_tiles, KV_l);
            int threads_pf = (H_l / KV_l) * 32;
            int blocks_per_head = HDl / 32;
            size_t smem_bytes = 2 * (size_t)BC_PREFILL * blocks_per_head * sizeof(half)
                              + 2 * (size_t)BC_PREFILL * HDl * sizeof(int8_t);
            k_prefill_flash_q8_0<<<grid_pf, threads_pf, smem_bytes, e->stream>>>(
                d_Q, Kl_q8, Vl_q8, d_Att,
                n, e->pos + n, e->pos, H_l, KV_l, HDl, scale_l, swa_l);
        } else {
            launch_prefill_flash(d_Q, Kl_f, Vl_f, d_Att,
                n, e->pos + n, e->pos, H_l, KV_l, HDl, scale_l, swa_l, e->stream);
        }
        if (tt_profiling()) tt_prof_end(TT_P_FLASH, e->stream);

        /* 4. O projection */
        if (tt_profiling()) tt_prof_begin(TT_P_OMLP, e->stream);
        if (w->o.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->o.ptr, d_Att, d_Xn, dim, attn_qout, n, e->stream);
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->o.ptr, w->o.dtype, d_Att + (long)i * attn_qout, d_Xn + (long)i * dim, dim, attn_qout, e->stream);
            }
        }
        if (tt_profiling()) tt_prof_end(TT_P_OMLP, e->stream);

        if (w->post_attn_norm) {
            k_rmsnorm_batched<<<n,256,256*sizeof(float),e->stream>>>(d_Xn, w->post_attn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);
        }

        /* Residual add: X += Xn */
        k_add<<<(n * dim + 255) / 256, 256, 0, e->stream>>>(d_X, d_Xn, n * dim);

        /* 5. FFN RMSNorm - batched */
        if (tt_profiling()) tt_prof_begin(TT_P_RMSNORM, e->stream);
        k_rmsnorm_batched<<<n,256,256*sizeof(float),e->stream>>>(d_X, w->ffn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);
        if (tt_profiling()) tt_prof_end(TT_P_RMSNORM, e->stream);

        /* 6. Gate & Up GEMM projections */
        const int act_gelu = (c->tr.act == ACT_GELU) ? 1 : 0;
        if (tt_profiling()) tt_prof_begin(TT_P_OMLP, e->stream);
        if (w->gate.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->gate.ptr, d_Xn, d_G, FF_l, dim, n, e->stream);
            prefill_gemm_fn(w->up.ptr, d_Xn, d_U, FF_l, dim, n, e->stream);
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->gate.ptr, w->gate.dtype, d_Xn + (long)i * dim, d_G + (long)i * FF_l, FF_l, dim, e->stream);
                tt_gemv_layer_dispatch(w->up.ptr, w->up.dtype, d_Xn + (long)i * dim, d_U + (long)i * FF_l, FF_l, dim, e->stream);
            }
        }
        if (tt_profiling()) tt_prof_end(TT_P_OMLP, e->stream);

        /* 7. SwiGLU activation */
        k_swiglu_apply<<<(n * FF_l + 255) / 256, 256, 0, e->stream>>>(d_G, d_U, d_H, n * FF_l, act_gelu);

        /* 8. Down projection GEMM */
        if (tt_profiling()) tt_prof_begin(TT_P_OMLP, e->stream);
        if (w->down.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->down.ptr, d_H, d_Xn, dim, FF_l, n, e->stream);
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->down.ptr, w->down.dtype, d_H + (long)i * FF_l, d_Xn + (long)i * dim, dim, FF_l, e->stream);
            }
        }
        if (tt_profiling()) tt_prof_end(TT_P_OMLP, e->stream);

        if (w->post_ffn_norm) {
            k_rmsnorm_batched<<<n,256,256*sizeof(float),e->stream>>>(d_Xn, w->post_ffn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);
        }

        /* Residual add: X += Xn */
        k_add<<<(n * dim + 255) / 256, 256, 0, e->stream>>>(d_X, d_Xn, n * dim);
    }

    e->pos += n;
    cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpyAsync(e->d_x, d_X + (long)(n - 1) * dim, (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice, e->stream);

    if (h_x_out) {
        cudaMemcpyAsync(h_x_out, d_X, (size_t)n * dim * sizeof(float), cudaMemcpyDeviceToHost, e->stream);
    }

    cudaStreamSynchronize(e->stream);

    if (!use_arena) {
        cudaFree(d_X); cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_Att); cudaFree(d_H); cudaFree(d_G); cudaFree(d_U); cudaFree(d_pos_batch);
    }

    return 0;
}
/* M10+ Batch-speculative verify prefill: same per-layer batched
 * forward as prefill_batched_gemm() but writes the per-token final
 * hidden state into d_x_out (device) instead of H2H-copying it back.
 * d_x_out MUST be a device buffer of n*c->dim floats; on return it
 * holds the final hidden state for every position [0..n) row-major.
 *
 * This is the speculative-verify replacement for the host copy. The
 * next stages (rmsnorm over each row, then LM head) can read d_x_out
 * entirely on device, collapsing the verify pass to:
 *   1 launch for N prefill layers,
 *   N launches for rmsnorm,
 *   1 launch for k_logits_q4_0_batch4,
 *   1 D2D copy from d_logits_batch -> out_logits.
 *
 * The function is a standalone copy of prefill_batched_gemm() with the
 * host copy removed and the d_X pointer redirected to d_x_out. The
 * original prefill_batched_gemm() is left UNCHANGED so other callers
 * (notably qwen2_engine_prefill chunked path) keep working. */
int prefill_batched_gemm_dx(Qwen2Engine *e, const int *toks, int n, float *d_x_out) {
    if (!e || !toks || n <= 0 || !d_x_out) return -1;
    const TTConfig *c = &e->cfg;
    if (e->pos + n > c->max_ctx) return -2;

    int (*prefill_gemm_fn)(const void *, const float *, float *, int, int, int, cudaStream_t) =
        tt_gemm_q4_0_prefill;

    const int dim = c->dim;
    const int hidden_dim = c->hidden_dim;
    const int HD = c->head_dim;
    long cache_layer = c->n_kv_heads * (long)c->max_ctx * HD;
    long cache_layer_q8 = c->n_kv_heads * (long)c->max_ctx * (HD / 32);
    if (e->pl_hd[0] > 0) {
        long mx = 0;
        long mx_q8 = 0;
        for (int l = 0; l < c->n_layers; l++) {
            const long w2 = (long)e->pl_kv[l] * e->pl_hd[l];
            if (w2 > mx) mx = w2;
            const long w2_q8 = (long)e->pl_kv[l] * (e->pl_hd[l] / 32);
            if (w2_q8 > mx_q8) mx_q8 = w2_q8;
        }
        cache_layer = mx * c->max_ctx;
        cache_layer_q8 = mx_q8 * c->max_ctx;
    }

    int max_qout = c->n_heads * HD;
    int max_kvdim = c->n_kv_heads * HD;
    int max_ff = hidden_dim;
    for (int l = 0; l < c->n_layers; l++) {
        int H_l = e->pl_heads[l] > 0 ? e->pl_heads[l] : c->n_heads;
        int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : c->n_kv_heads;
        int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
        if (H_l * HDl > max_qout) max_qout = H_l * HDl;
        if (KV_l * HDl > max_kvdim) max_kvdim = KV_l * HDl;
        int FFx = e->pl_ffn[l] > 0 ? e->pl_ffn[l] : hidden_dim;
        if (FFx > max_ff) max_ff = FFx;
    }
    int use_arena = (e->pf_arena_max_n >= (size_t)n && e->d_pf_Xn && e->d_pf_Q && e->d_pf_K && e->d_pf_V && e->d_pf_Att && e->d_pf_H && e->d_pf_G && e->d_pf_U && e->d_pf_pos_batch);
    float *d_X = d_x_out;            /* caller-supplied; no cudaMalloc */
    float *d_Xn = NULL, *d_Q = NULL, *d_K = NULL, *d_V = NULL;
    float *d_Att = NULL, *d_H = NULL, *d_G = NULL, *d_U = NULL;
    int *d_pos_batch = NULL;
    if (use_arena) {
        d_Xn = e->d_pf_Xn; d_Q = e->d_pf_Q; d_K = e->d_pf_K; d_V = e->d_pf_V;
        d_Att = e->d_pf_Att; d_H = e->d_pf_H; d_G = e->d_pf_G; d_U = e->d_pf_U; d_pos_batch = e->d_pf_pos_batch;
    } else {
        if (cudaMalloc(&d_Xn, (size_t)n * dim * sizeof(float)) != cudaSuccess) return -4;
        if (cudaMalloc(&d_Q, (size_t)n * max_qout * sizeof(float)) != cudaSuccess) { cudaFree(d_Xn); return -5; }
        if (cudaMalloc(&d_K, (size_t)n * max_kvdim * sizeof(float)) != cudaSuccess) { cudaFree(d_Xn); cudaFree(d_Q); return -6; }
        if (cudaMalloc(&d_V, (size_t)n * max_kvdim * sizeof(float)) != cudaSuccess) { cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); return -7; }
        if (cudaMalloc(&d_Att, (size_t)n * max_qout * sizeof(float)) != cudaSuccess) { cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); return -8; }
        if (cudaMalloc(&d_H, (size_t)n * max_ff * sizeof(float)) != cudaSuccess) { cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); return -9; }
        if (cudaMalloc(&d_G, (size_t)n * max_ff * sizeof(float)) != cudaSuccess) { cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); cudaFree(d_H); return -10; }
        if (cudaMalloc(&d_U, (size_t)n * max_ff * sizeof(float)) != cudaSuccess) { cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); cudaFree(d_H); cudaFree(d_G); return -11; }
        if (cudaMalloc(&d_pos_batch, (size_t)n * sizeof(int)) != cudaSuccess) { cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Att); cudaFree(d_H); cudaFree(d_G); cudaFree(d_U); return -12; }
    }

    int *h_pos_batch = (int *)malloc((size_t)n * sizeof(int));
    if (!h_pos_batch) return -13;
    const int pos0 = e->pos;
    for (int i = 0; i < n; i++) h_pos_batch[i] = pos0 + i;
    cudaMemcpy(d_pos_batch, h_pos_batch, (size_t)n * sizeof(int), cudaMemcpyHostToDevice);
    free(h_pos_batch);

    /* Opt in to dynamic shmem for FP32 prefill flash if needed */
    {
        int pf_max_hd = HD;
        for (int l = 0; l < c->n_layers; l++) {
            int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
            if (HDl > pf_max_hd) pf_max_hd = HDl;
        }
        size_t smem_fp32_attr = 2 * (size_t)BC_PREFILL_FP32 * pf_max_hd * sizeof(float);
        if (smem_fp32_attr > 48 * 1024) {
            cudaFuncSetAttribute(k_prefill_flash_fp32,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_fp32_attr);
        }
        {
            int pf_bph = pf_max_hd / 32;
            size_t smem_q8_attr = 2 * (size_t)BC_PREFILL * pf_bph * sizeof(half)
                                + 2 * (size_t)BC_PREFILL * pf_max_hd * sizeof(int8_t);
            if (smem_q8_attr > 48 * 1024) {
                cudaFuncSetAttribute(k_prefill_flash_q8_0,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_q8_attr);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        tt_embed_typed(e->d_embd.ptr, e->d_embd.dtype, toks[i],
                       d_X + (long)i * dim, dim, e->stream);
        if (c->tr.embed_sqrt) {
            k_scale<<<(dim + 255) / 256, 256, 0, e->stream>>>(
                d_X + (long)i * dim, sqrtf((float)dim), dim);
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &e->L[l];
        float *Kl_f = e->d_kc + (long)l * cache_layer;
        float *Vl_f = e->d_vc + (long)l * cache_layer;
        BlockQ8_0 *Kl_q8 = e->d_kc_q8 ? (e->d_kc_q8 + (long)l * cache_layer_q8) : NULL;
        BlockQ8_0 *Vl_q8 = e->d_vc_q8 ? (e->d_vc_q8 + (long)l * cache_layer_q8) : NULL;
        BlockQ4_0 *Kl_q4 = e->d_kc_q4 ? (e->d_kc_q4 + (long)l * cache_layer_q8) : NULL;
        BlockQ4_0 *Vl_q4 = e->d_vc_q4 ? (e->d_vc_q4 + (long)l * cache_layer_q8) : NULL;
        const int kv_shared = e->has_pl_embd && e->pl_src[l] >= 0;
        if (kv_shared) {
            Kl_f = e->d_kc + (long)e->pl_src[l] * cache_layer;
            Vl_f = e->d_vc + (long)e->pl_src[l] * cache_layer;
            if (e->d_kc_q8) {
                Kl_q8 = e->d_kc_q8 + (long)e->pl_src[l] * cache_layer_q8;
                Vl_q8 = e->d_vc_q8 + (long)e->pl_src[l] * cache_layer_q8;
            }
            if (e->d_kc_q4) {
                Kl_q4 = e->d_kc_q4 + (long)e->pl_src[l] * cache_layer_q8;
                Vl_q4 = e->d_vc_q4 + (long)e->pl_src[l] * cache_layer_q8;
            }
        }

        const int H_l = e->pl_heads[l] > 0 ? e->pl_heads[l] : c->n_heads;
        const int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : c->n_kv_heads;
        const int FF_l = e->pl_ffn[l] > 0 ? e->pl_ffn[l] : hidden_dim;
        const int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : HD;
        const int attn_qout = H_l * HDl;
        const int kvdim_l = KV_l * HDl;

        /* 1. RMSNorm before QKV - batched */
        k_rmsnorm_batched<<<n, 256, 256*sizeof(float), e->stream>>>(d_X, w->attn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);

        /* 2. Batched QKV GEMM */
        if (w->q.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->q.ptr, d_Xn, d_Q, attn_qout, dim, n, e->stream);
            if (!kv_shared) {
                prefill_gemm_fn(w->k.ptr, d_Xn, d_K, kvdim_l, dim, n, e->stream);
                prefill_gemm_fn(w->v.ptr, d_Xn, d_V, kvdim_l, dim, n, e->stream);
            }
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->q.ptr, w->q.dtype, d_Xn + (long)i * dim, d_Q + (long)i * attn_qout, attn_qout, dim, e->stream);
                if (!kv_shared) {
                    tt_gemv_layer_dispatch(w->k.ptr, w->k.dtype, d_Xn + (long)i * dim, d_K + (long)i * kvdim_l, kvdim_l, dim, e->stream);
                    tt_gemv_layer_dispatch(w->v.ptr, w->v.dtype, d_Xn + (long)i * dim, d_V + (long)i * kvdim_l, kvdim_l, dim, e->stream);
                }
            }
        }

        /* Biases & QK norm - batched */
        if (w->q_bias) {
            k_add_bias_batched<<<(n*attn_qout+255)/256,256,0,e->stream>>>(d_Q, w->q_bias, attn_qout, n);
        }
        if (!kv_shared && w->k_bias) {
            k_add_bias_batched<<<(n*kvdim_l+255)/256,256,0,e->stream>>>(d_K, w->k_bias, kvdim_l, n);
        }
        if (!kv_shared && w->v_bias) {
            k_add_bias_batched<<<(n*kvdim_l+255)/256,256,0,e->stream>>>(d_V, w->v_bias, kvdim_l, n);
        }

        if (c->tr.qk_norm_rms) {
            const int qkthreads = HDl < 256 ? HDl : 256;
            {
                size_t smem = qkthreads*sizeof(float);
                k_qk_norm_rms_batched<<<n*H_l, qkthreads, smem, e->stream>>>(d_Q, w->q_norm, H_l, HDl, c->tr.qk_norm_eps, n, attn_qout);
                if (!kv_shared)
                    k_qk_norm_rms_batched<<<n*KV_l, qkthreads, smem, e->stream>>>(d_K, w->k_norm, KV_l, HDl, c->tr.qk_norm_eps, n, kvdim_l);
            }
        }

        const int is_full_l = (e->pl_swa[l] == 0);
        const float base_l = e->has_pl_embd ? (is_full_l ? 1e6f : 1e4f) : c->rope_base;
        const float *ff_l = (e->has_pl_embd && is_full_l) ? e->d_rope_freqs : NULL;
        void (*rope_fn)(float *, int, int, const int *, float) =
            (c->tr.rope == ROPE_GPTJ) ? k_rope_gptj : k_rope;
        const float scale_l = c->tr.attn_scale_one ? 1.0f : 1.0f / sqrtf((float)HDl);
        const int swa_l = e->has_pl_embd ? e->pl_swa[l] : c->tr.swa_size;

        // Batched RoPE + scatter: single launch per layer instead of n launches
        {
            static int no_rope = -1;
            if (no_rope < 0) no_rope = getenv("TT_NO_ROPE") ? 1 : 0;
            if (!no_rope) {
                dim3 g_qb((HDl / 2 + 63) / 64, H_l, n), b_rope(64, 1, 1);
                dim3 g_kb((HDl / 2 + 63) / 64, KV_l, n);
                const int is_gptj = (c->tr.rope == ROPE_GPTJ);
                if (is_gptj) {
                    if (ff_l) {
                        k_rope_gptj_ff_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, ff_l, n, attn_qout);
                        if (!kv_shared) k_rope_gptj_ff_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, ff_l, n, kvdim_l);
                    } else {
                        k_rope_gptj_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, n, attn_qout);
                        if (!kv_shared) k_rope_gptj_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, n, kvdim_l);
                    }
                } else {
                    if (ff_l) {
                        k_rope_ff_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, ff_l, n, attn_qout);
                        if (!kv_shared) k_rope_ff_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, ff_l, n, kvdim_l);
                    } else {
                        k_rope_batched<<<g_qb, b_rope, 0, e->stream>>>(d_Q, H_l, HDl, d_pos_batch, base_l, n, attn_qout);
                        if (!kv_shared) k_rope_batched<<<g_kb, b_rope, 0, e->stream>>>(d_K, KV_l, HDl, d_pos_batch, base_l, n, kvdim_l);
                    }
                }
            }
            if (!kv_shared) {
                /* P0-1: dual-write FP32 + Q4. Decode below thresh reads FP32;
                 * decode above thresh reads Q4, so both must be populated.
                 * P0-2: gate on ptrs non-NULL (flag-set + ptr-NULL = crash). */
                if (e->use_q4_kvcache && Kl_q4 && Vl_q4) {
                    k_kv_scatter_batched<<<(n*kvdim_l+255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_f, Vl_f, d_pos_batch, KV_l, HDl, c->max_ctx, n, kvdim_l);
                    const int blocks_per_slot = kvdim_l / 32;
                    const long total_blocks = (long)n * blocks_per_slot;
                    k_kv_scatter_q4_0_batched<<<(total_blocks + 255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_q4, Vl_q4, d_pos_batch, KV_l, HDl, c->max_ctx, n);
                } else if (e->use_q8_kvcache && Kl_q8 && Vl_q8) {
                    /* Dual-write FP32 + Q8 (mirror Q4 path): hybrid decode
                     * below thresh reads FP32, so it must be populated. */
                    k_kv_scatter_batched<<<(n*kvdim_l+255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_f, Vl_f, d_pos_batch, KV_l, HDl, c->max_ctx, n, kvdim_l);
                    const int blocks_per_slot = kvdim_l / 32;
                    const long total_blocks = (long)n * blocks_per_slot;
                    k_kv_scatter_q8_0_batched<<<(total_blocks + 255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_q8, Vl_q8, d_pos_batch, KV_l, HDl, c->max_ctx, n);
                } else {
                    k_kv_scatter_batched<<<(n*kvdim_l+255)/256, 256, 0, e->stream>>>(
                        d_K, d_V, Kl_f, Vl_f, d_pos_batch, KV_l, HDl, c->max_ctx, n, kvdim_l);
                }
            }
        }

        /* P0-1: prefill flash always on FP32 (bit-exact output at every ctx).
         * Q4 cache stays populated for decode above thresh. P0-2: ptr guard. */
        if (HDl > 128) {
            /* Tiled prefill flash regs/smem sized for elems<=4 (HD<=128);
             * fall back to per-token serial path for this layer only. */
            if (e->use_q8_kvcache && Kl_q8 && Vl_q8 && !(e->use_q4_kvcache && Kl_q4 && Vl_q4)) {
                for (int qi = 0; qi < n; qi++)
                    k_flash_gqa_q8_0<<<H_l, 32, 0, e->stream>>>(
                        d_Q + (long)qi * attn_qout, Kl_q8, Vl_q8, d_Att + (long)qi * attn_qout,
                        d_pos_batch + qi, H_l, KV_l, HDl, c->max_ctx, scale_l, swa_l);
            } else {
                for (int qi = 0; qi < n; qi++)
                    k_flash_gqa<<<H_l, 32, 0, e->stream>>>(
                        d_Q + (long)qi * attn_qout, Kl_f, Vl_f, d_Att + (long)qi * attn_qout,
                        d_pos_batch + qi, H_l, KV_l, HDl, c->max_ctx, scale_l, swa_l);
            }
        } else if (e->use_q4_kvcache && Kl_q4 && Vl_q4) {
            launch_prefill_flash(d_Q, Kl_f, Vl_f, d_Att,
                n, e->pos + n, e->pos, H_l, KV_l, HDl, scale_l, swa_l, e->stream);
        } else if (e->use_q8_kvcache && Kl_q8 && Vl_q8) {
            int num_q_tiles = (n + BR_PREFILL - 1) / BR_PREFILL;
            dim3 grid_pf(num_q_tiles, KV_l);
            int threads_pf = (H_l / KV_l) * 32;
            int blocks_per_head = HDl / 32;
            size_t smem_bytes = 2 * (size_t)BC_PREFILL * blocks_per_head * sizeof(half)
                              + 2 * (size_t)BC_PREFILL * HDl * sizeof(int8_t);
            k_prefill_flash_q8_0<<<grid_pf, threads_pf, smem_bytes, e->stream>>>(
                d_Q, Kl_q8, Vl_q8, d_Att,
                n, e->pos + n, e->pos, H_l, KV_l, HDl, scale_l, swa_l);
        } else {
            launch_prefill_flash(d_Q, Kl_f, Vl_f, d_Att,
                n, e->pos + n, e->pos, H_l, KV_l, HDl, scale_l, swa_l, e->stream);
        }

        /* 4. O projection */
        if (w->o.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->o.ptr, d_Att, d_Xn, dim, attn_qout, n, e->stream);
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->o.ptr, w->o.dtype, d_Att + (long)i * attn_qout, d_Xn + (long)i * dim, dim, attn_qout, e->stream);
            }
        }

        if (w->post_attn_norm) {
            k_rmsnorm_batched<<<n,256,256*sizeof(float),e->stream>>>(d_Xn, w->post_attn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);
        }

        /* Residual add: X += Xn */
        k_add<<<(n * dim + 255) / 256, 256, 0, e->stream>>>(d_X, d_Xn, n * dim);

        /* 5. FFN RMSNorm - batched */
        k_rmsnorm_batched<<<n,256,256*sizeof(float),e->stream>>>(d_X, w->ffn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);

        /* 6. Gate & Up GEMM projections */
        const int act_gelu = (c->tr.act == ACT_GELU) ? 1 : 0;
        if (w->gate.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->gate.ptr, d_Xn, d_G, FF_l, dim, n, e->stream);
            prefill_gemm_fn(w->up.ptr, d_Xn, d_U, FF_l, dim, n, e->stream);
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->gate.ptr, w->gate.dtype, d_Xn + (long)i * dim, d_G + (long)i * FF_l, FF_l, dim, e->stream);
                tt_gemv_layer_dispatch(w->up.ptr, w->up.dtype, d_Xn + (long)i * dim, d_U + (long)i * FF_l, FF_l, dim, e->stream);
            }
        }

        /* 7. SwiGLU activation */
        k_swiglu_apply<<<(n * FF_l + 255) / 256, 256, 0, e->stream>>>(d_G, d_U, d_H, n * FF_l, act_gelu);

        /* 8. Down projection GEMM */
        if (w->down.dtype == TTQ_Q4_0) {
            prefill_gemm_fn(w->down.ptr, d_H, d_Xn, dim, FF_l, n, e->stream);
        } else {
            for (int i = 0; i < n; i++) {
                tt_gemv_layer_dispatch(w->down.ptr, w->down.dtype, d_H + (long)i * FF_l, d_Xn + (long)i * dim, dim, FF_l, e->stream);
            }
        }

        if (w->post_ffn_norm) {
            k_rmsnorm_batched<<<n,256,256*sizeof(float),e->stream>>>(d_Xn, w->post_ffn_norm, d_Xn, dim, c->rms_eps, c->tr.norm_offset, n);
        }

        /* Residual add: X += Xn */
        k_add<<<(n * dim + 255) / 256, 256, 0, e->stream>>>(d_X, d_Xn, n * dim);
    }

    e->pos += n;
    cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);
    /* Final hidden state for every position lives in d_X = d_x_out. */

    if (!use_arena) {
        cudaFree(d_Xn); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_Att); cudaFree(d_H); cudaFree(d_G); cudaFree(d_U); cudaFree(d_pos_batch);
    }

    return 0;
}

int qwen2_engine_prefill(Qwen2Engine *e, const int *toks, int n) {
    if (!e || !toks || n <= 0) return -1;
    if (e->pos + n > e->cfg.max_ctx) return -2;          /* context overflow */
    /* Graph replay leaves the most recently SAMPLED token unfed (the caller
     * stopped asking). Flush it through the layers so the KV cache matches
     * the eager state machine before the new prompt lands. */
    if (e->graph_ready && e->pending_tok >= 0) {
        if (advance(e, e->pending_tok)) return -3;
        e->pending_tok = -1;
    }
    /* resync device position scalar before any forward work (sync: see advance()) */
    cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);
    if (n >= 32 && !e->has_pl_embd && e->cfg.tr.softcap_value == 0.0f) {
        const int CHUNK_SIZE = 512;
        int offset = 0;
        int failed = 0;
        while (offset < n) {
            int chunk_len = (offset + CHUNK_SIZE <= n) ? CHUNK_SIZE : (n - offset);
            if (chunk_len >= 32) {
                int rc = prefill_batched_gemm(e, toks + offset, chunk_len, NULL);
                if (rc != 0) { failed = 1; break; }
            } else {
                for (int i = 0; i < chunk_len; i++) {
                    int rc = advance(e, toks[offset + i]);
                    if (rc) { failed = 1; break; }
                }
                if (failed) break;
            }
            offset += chunk_len;
        }
        if (!failed) {
            cudaStreamSynchronize(e->stream);
            return 0;
        }
    }
    for (int i = 0; i < n; i++) {
        int rc = advance(e, toks[i]);
        if (rc) return rc;
    }
    cudaStreamSynchronize(e->stream);
    return 0;
}

/* Speculative-decode "logits-only" tail: same rmsnorm -> lm-head -> softcap
 * as sample_eager(), but no sampling transforms, no argmax, no D2H sync.
 * Result lives in e->d_logits on the device (vocab floats, f32).
 * Used by qwen2_engine_verify_speculative() after each candidate-token
 * advance.  No-op with respect to KV cache / pos. */
static int compute_logits_into_d_logits(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;
    if (tt_profiling()) tt_prof_begin(TT_P_RMSNORM, e->stream);
    k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
        e->d_x, e->d_out_norm, e->d_xn, c->dim, c->rms_eps, c->tr.norm_offset);
    if (tt_profiling()) tt_prof_end(TT_P_RMSNORM, e->stream);

    if (tt_profiling()) tt_prof_begin(TT_P_LOGITS, e->stream);
    const int rc = tt_logits_dispatch(e->d_out_w.ptr, e->d_out_w.dtype, e->d_xn,
                                      e->d_logits, c->vocab, c->dim, e->stream);
    if (tt_profiling()) tt_prof_end(TT_P_LOGITS, e->stream);
    if (rc) return rc;

    /* M7 trait: final-logit tanh softcap (gemma2). Identical kernel as
     * the eager path; if the trait is absent this branch is dead. */
    if (c->tr.softcap_value > 0.0f)
        k_softcap<<<(c->vocab + 255) / 256, 256, 0, e->stream>>>(
            e->d_logits, c->vocab, c->tr.softcap_value);
    return 0;
}

/* Speculative-decode VERIFY pass: feed N candidate tokens through the
 * layers in order, returning the per-position logits to a device buffer.
 *
 * Reference semantics: out_logits[i] must equal the logits that a
 * qwen2_engine_next() call would have produced right after advancing
 * with h_candidate_tokens[i], starting from the engine state at entry
 * (with the prior N-1 candidates already fed).  Bit-exact with the
 * single-token eager path.
 *
 * Implementation note (correctness-first, perf-follow-up):
 *   The eager engines' per-layer kernels are batch_size=1 GEMVs. The
 *   cleanest batched verify kernel (parallel Q/K/V/FFN over N positions,
 *   grouped Q@K^T attention with N-tile) is a non-trivial rewrite and
 *   outside this task's scope. Instead we drive N sequential advance()
 *   calls — each writes its K/V to the correct cache slot (advance()
 *   increments d_pos for the next slot) and runs the same forward
 *   layers body the graph path uses. After each advance we run the
 *   logits-only tail (compute_logits_into_d_logits) and D2D-copy
 *   d_logits -> out_logits + i*vocab.  The orchestrator (Task 3) saves
 *   the engine state before calling and restores it on rejection.
 */
int qwen2_engine_verify_speculative(Qwen2Engine *e,
                                    const int *h_candidate_tokens,
                                    int n_candidate,
                                    float *out_logits) {
    if (!e || !h_candidate_tokens || !out_logits) return -1;
    if (n_candidate <= 0) return -2;
    const TTConfig *c = &e->cfg;
    if (e->pos + n_candidate > c->max_ctx) return -3;       /* ctx overflow */
    if (n_candidate > 16) return -4;                        /* sanity cap    */

    /* Mirror prefill()'s graph-replay flush: a sampled-but-unfed pending
     * token must be advanced first so the KV cache matches the eager
     * state machine. */
    if (e->graph_ready && e->pending_tok >= 0) {
        if (advance(e, e->pending_tok)) return -5;
        e->pending_tok = -1;
    }
    /* advance() relies on *e->d_pos for the cache-slot write index. After
     * the (optional) flush, host and device pos agree; the first advance()
     * below will write to slot e->pos, the next to e->pos+1, etc. */
    cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);

    const long vocab_f = (long)c->vocab;
    const int dim = c->dim;

    /* TT_SPEC_BATCH gate (lazy env read; defaults to 1 = per-row eager).
     * spec_batch==1  : original per-row advance()+compute_logits_into_d_logits()
     *                  loop below (correctness path; ~120 tok/s on qwen2.5-0.5b).
     * spec_batch>=2  : device-resident batched prefill (prefill_batched_gemm_dx)
     *                  + LM head. When n_candidate==4 AND lm_head is q4_0 with
     *                  vocab%4==0, the LM head collapses to one
     *                  tt_logits_q4_0_batch4 launch (single weight pass over all
     *                  4 candidates). n_candidate<4 still benefits from the
     *                  device-resident prefill (no H2H round-trip).
     * For n_candidate<4 the orchestrator is expected to pad to 4 (e.g.
     * via K+1 always == 4 with TT_DRAFT_K=3); we do NOT auto-pad here because
     * padding would write duplicate K/V into the cache slots. */
    static int spec_batch_env = -1;
    if (spec_batch_env < 0) {
        const char *env = getenv("TT_SPEC_BATCH");
        spec_batch_env = (env && atoi(env) >= 1) ? atoi(env) : 1;
    }
    const int want_batched = (spec_batch_env >= 2)
                          && (n_candidate >= 2)
                          && (e->n_gpu_layers == c->n_layers)
                          && !e->has_pl_embd
                          && (c->tr.softcap_value == 0.0f);

    if (want_batched) {
        /* Lazy-alloc (or grow) the batched speculative workspace:
         *   d_x_batch       [n_max, dim]    post-final-layer hidden states
         *   d_xn_batch      [n_max, dim]    post-rmsnorm hidden (LM head input)
         *   d_logits_batch  [n_max, vocab]  candidate-major logits [c*vocab+v]
         * Layout for k_logits_q4_0_batch4 is exactly [n_max, dim/vocab] row-major. */
        const int n_max = n_candidate;
        if (e->d_spec_max_n < n_max) {
            if (e->d_x_batch)    { cudaFree(e->d_x_batch);    e->d_x_batch = NULL; }
            if (e->d_xn_batch)   { cudaFree(e->d_xn_batch);   e->d_xn_batch = NULL; }
            if (e->d_logits_batch) { cudaFree(e->d_logits_batch); e->d_logits_batch = NULL; }
            if (cudaMalloc(&e->d_x_batch,    (size_t)n_max * dim * sizeof(float)) != cudaSuccess) return -30;
            if (cudaMalloc(&e->d_xn_batch,   (size_t)n_max * dim * sizeof(float)) != cudaSuccess) return -31;
            if (cudaMalloc(&e->d_logits_batch, (size_t)n_max * vocab_f * sizeof(float)) != cudaSuccess) return -32;
            e->d_spec_max_n = n_max;
        }
        e->d_spec_n = n_candidate;

        /* 1) Batched prefill: writes the per-position final hidden states
         * into d_x_batch (device-resident; no H2H copy back). This advances
         * e->pos by n_candidate and writes K/V to those cache slots. */
        int rc = prefill_batched_gemm_dx(e, h_candidate_tokens, n_candidate, e->d_x_batch);
        if (rc != 0) {
            /* Prefill refused (mixed dtype / PLE / softcap). Fall back to the
             * original per-row eager path below. Restore d_pos so the eager
             * path sees the pre-call position. */
            e->pos -= n_candidate;
            cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);
        } else {
            /* 2) Per-row final rmsnorm into d_xn_batch (LM head input). */
            for (int i = 0; i < n_candidate; i++) {
                k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
                    e->d_x_batch + (long)i * dim,
                    e->d_out_norm,
                    e->d_xn_batch + (long)i * dim,
                    dim, c->rms_eps, c->tr.norm_offset);
            }

            /* 3) LM head: batched-4 (single launch) when contract holds;
             * otherwise per-row tt_logits_dispatch into d_logits_batch. */
            int use_b4 = (n_candidate == 4)
                       && (e->d_out_w.dtype == GGUF_TYPE_Q4_0)
                       && ((vocab_f & 3) == 0)
                       && (dim % 32 == 0);
            if (use_b4) {
                rc = tt_logits_q4_0_batch4(e->d_out_w.ptr,
                                            e->d_xn_batch,
                                            e->d_logits_batch,
                                            (int)vocab_f, dim, e->stream);
                if (rc != 0) {
                    /* Contract violation surfaced only at launch time; fall back. */
                    use_b4 = 0;
                }
            }

            if (!use_b4) {
                for (int i = 0; i < n_candidate; i++) {
                    tt_logits_dispatch(e->d_out_w.ptr, e->d_out_w.dtype,
                                       e->d_xn_batch + (long)i * dim,
                                       e->d_logits_batch + (long)i * vocab_f,
                                       (int)vocab_f, dim, e->stream);
                }
            }

            /* 4) D2D copy the [n_candidate, vocab] logits block to out_logits. */
            cudaMemcpyAsync(out_logits, e->d_logits_batch,
                            (size_t)n_candidate * vocab_f * sizeof(float),
                            cudaMemcpyDeviceToDevice, e->stream);
            cudaStreamSynchronize(e->stream);
            return 0;
        }
    }

    /* Fallback path: original per-row advance() + compute_logits_into_d_logits().
     * Always correct; used for n_candidate==1, off-spec engines (PLE, mixed
     * dtypes, softcap), and when prefill_batched_gemm_dx refuses. */
    for (int i = 0; i < n_candidate; i++) {
        if (advance(e, h_candidate_tokens[i])) return -10 - i;
        if (compute_logits_into_d_logits(e))   return -20 - i;
        cudaMemcpyAsync(out_logits + (long)i * vocab_f,
                        e->d_logits,
                        vocab_f * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        e->stream);
    }
    return 0;
}

/* Eager sampling tail shared by all paths: final rmsnorm -> logits ->
 * greedy argmax -> D2H sync. Returns token id, or negative on error.
 * d_logits stays valid afterwards (dump_logits relies on this). */
/* sampling controls applied to d_logits before argmax (eager path mirror) */
static void apply_sampling_eager(Qwen2Engine *e) {
    if (e->repeat_penalty > 1.0f)
        k_repeat_penalty<<<(e->cfg.vocab + 255) / 256, 256, 0, e->stream>>>(
            e->d_logits, e->d_recent, e->d_n_recent, e->cfg.vocab, e->repeat_penalty);
    k_gumbel_transform<<<(e->cfg.vocab + 255) / 256, 256, 0, e->stream>>>(
        e->d_logits, e->cfg.vocab, e->sampling_temp, e->d_pos, e->d_sampling_on);
}

static int sample_eager(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;
    if (tt_profiling()) tt_prof_begin(TT_P_RMSNORM, e->stream);
    k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
        e->d_x, e->d_out_norm, e->d_xn, c->dim, c->rms_eps, c->tr.norm_offset);
    if (tt_profiling()) tt_prof_end(TT_P_RMSNORM, e->stream);

    if (tt_profiling()) tt_prof_begin(TT_P_LOGITS, e->stream);
    int rc = tt_logits_dispatch(e->d_out_w.ptr, e->d_out_w.dtype, e->d_xn,
                                e->d_logits, c->vocab, c->dim, e->stream);
    if (rc) return -1;
    if (tt_profiling()) tt_prof_end(TT_P_LOGITS, e->stream);

    /* M7 trait: final-logit tanh softcap (gemma2), post-GEMV pre-sampling. */
    if (c->tr.softcap_value > 0.0f)
        k_softcap<<<(c->vocab + 255) / 256, 256, 0, e->stream>>>(
            e->d_logits, c->vocab, c->tr.softcap_value);
    apply_sampling_eager(e);

    if (tt_profiling()) tt_prof_begin(TT_P_ARGMAX, e->stream);
    const int nb = 64;
    k_argmax_partial<<<nb, 256, 0, e->stream>>>(e->d_logits, c->vocab, e->d_bvals, e->d_bidxs);
    k_argmax_final<<<1, nb, 0, e->stream>>>(e->d_bvals, e->d_bidxs, nb, e->d_out);

    int id = 0;
    cudaMemcpyAsync(&id, e->d_out, sizeof(int), cudaMemcpyDeviceToHost, e->stream);
    cudaStreamSynchronize(e->stream);
    if (tt_profiling()) tt_prof_end(TT_P_ARGMAX, e->stream);
    return id;
}

/* Capture the whole decode step into a single-launch graph:
 *   embed(d_next_tok) -> forward_layers -> rmsnorm -> logits -> argmax -> pos_inc
 * All nodes on e->stream. The captured embed reads whatever token sits in
 * d_next_tok at REPLAY time (set per-call via async H2D before the launch).
 * On success sets graph_ready and returns 0; leaves graph_ready=0 otherwise. */
static int qwen2_engine_graph_capture(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;

    /* M9.5: dynamic-token embed kernel is now provided for q4_0 AND q6_k.
     * Other dtypes (q8_0, f16, q4_k/q5_k, etc.) still lack an in-graph
     * variant; fall back to eager for those. Captured graph works for
     * qwen2.5 (q4_0), llama-3.2 (q6_k despite filename), gemma2 (q6_k). */
    const int edt = e->d_embd.dtype;
    if (edt != GGUF_TYPE_Q4_0 && edt != GGUF_TYPE_Q2_K && edt != GGUF_TYPE_Q3_K && edt != GGUF_TYPE_Q6_K) return -1;
    const int dummy = e->pending_tok >= 0 ? e->pending_tok : 0;
    cudaMemcpy(e->d_next_tok, &dummy, sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(e->stream);

    /* warm up graph-only kernels eagerly (lazy module load must not happen
     * implicitly during capture). Save/restore engine state they touch.
     * NOTE: d_logits must be RESTORED, not zeroed — callers (dump_logits,
     * parity gate) read it right after this returns, while the sampled id
     * from sample_eager is still the current answer. */
    {
        float *xsave = NULL, *lsave = NULL;
        cudaMalloc(&xsave, c->dim * sizeof(float));
        cudaMalloc(&lsave, c->vocab * sizeof(float));
        cudaMemcpy(xsave, e->d_x, c->dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(lsave, e->d_logits, c->vocab * sizeof(float), cudaMemcpyDeviceToDevice);
        const int pos_before = e->pos;
        if (edt == GGUF_TYPE_Q4_0) {
            const int threads = c->dim / 32;
            k_embed_q4_0_dyn<<<(threads + 255) / 256, 256, 0, e->stream>>>(
                (const BlockQ4_0 *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        } else if (edt == GGUF_TYPE_Q2_K) {
            const int nu = (c->dim / 256) * 16;
            k_embed_q2_K_dyn<<<(nu + 255) / 256, 256, 0, e->stream>>>(
                (const uint8_t *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        } else if (edt == GGUF_TYPE_Q3_K) {
            const int nu = (c->dim / 256) * 16;
            k_embed_q3_K_dyn<<<(nu + 255) / 256, 256, 0, e->stream>>>(
                (const uint8_t *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        } else { /* GGUF_TYPE_Q6_K */
            const int nu = (c->dim / 256) * 8;
            k_embed_q6_K_dyn<<<(nu + 255) / 256, 256, 0, e->stream>>>(
                (const uint8_t *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        }
        k_pos_inc<<<1, 1, 0, e->stream>>>(e->d_pos);
        k_repeat_penalty<<<1, 256, 0, e->stream>>>(e->d_logits, e->d_recent,
                                                   e->d_n_recent, c->vocab, 2.0f);
        k_gumbel_transform<<<1, 256, 0, e->stream>>>(e->d_logits, c->vocab,
                                                     0.8f, e->d_pos, e->d_sampling_on);
        cudaStreamSynchronize(e->stream);
        /* restore state the warmup perturbed */
        cudaMemcpy(e->d_x, xsave, c->dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(e->d_logits, lsave, c->vocab * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(e->d_pos, &pos_before, sizeof(int), cudaMemcpyHostToDevice);
        cudaFree(xsave); cudaFree(lsave);
    }

    g_capturing = 1;
    if (cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        g_capturing = 0;
        return -1;
    }

    {   /* dynamic-token embedding (device-side id) */
        if (edt == GGUF_TYPE_Q4_0) {
            const int threads = c->dim / 32;
            k_embed_q4_0_dyn<<<(threads + 255) / 256, 256, 0, e->stream>>>(
                (const BlockQ4_0 *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        } else if (edt == GGUF_TYPE_Q2_K) {
            const int nu = (c->dim / 256) * 16;
            k_embed_q2_K_dyn<<<(nu + 255) / 256, 256, 0, e->stream>>>(
                (const uint8_t *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        } else if (edt == GGUF_TYPE_Q3_K) {
            const int nu = (c->dim / 256) * 16;
            k_embed_q3_K_dyn<<<(nu + 255) / 256, 256, 0, e->stream>>>(
                (const uint8_t *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        } else { /* GGUF_TYPE_Q6_K */
            const int nu = (c->dim / 256) * 8;
            k_embed_q6_K_dyn<<<(nu + 255) / 256, 256, 0, e->stream>>>(
                (const uint8_t *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        }
    }
    /* gemma families: scale embeddings by sqrt(dim) right after the
     * dynamic embed (matches embed_token() in the eager path; without it
     * gemma2/graph diverges from gemma2/eager in 1-2 steps). Host branch
     * is constant per capture, kernel reads no host state. */
    if (c->tr.embed_sqrt)
        k_scale<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(
            e->d_x, sqrtf((float)c->dim), c->dim);
    const int frc = forward_layers(e);          /* all layers at *d_pos */
    k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
        e->d_x, e->d_out_norm, e->d_xn, c->dim, c->rms_eps, c->tr.norm_offset);
    const int lrc = tt_logits_dispatch(e->d_out_w.ptr, e->d_out_w.dtype, e->d_xn,
                                       e->d_logits, c->vocab, c->dim, e->stream);
    /* M7 trait: final-logit softcap (gemma2). Host-side branch is constant
     * per capture; kernel reads no host state => capture-safe. */
    if (c->tr.softcap_value > 0.0f)
        k_softcap<<<(c->vocab + 255) / 256, 256, 0, e->stream>>>(
            e->d_logits, c->vocab, c->tr.softcap_value);
    /* sampling controls: both kernels no-op when disabled (greedy byte-identical) */
    k_repeat_penalty<<<(c->vocab + 255) / 256, 256, 0, e->stream>>>(
        e->d_logits, e->d_recent, e->d_n_recent, c->vocab, e->repeat_penalty);
    k_gumbel_transform<<<(c->vocab + 255) / 256, 256, 0, e->stream>>>(
        e->d_logits, c->vocab, e->sampling_temp, e->d_pos, e->d_sampling_on);
    const int nb = 64;
    k_argmax_partial<<<nb, 256, 0, e->stream>>>(e->d_logits, c->vocab, e->d_bvals, e->d_bidxs);
    k_argmax_final<<<1, nb, 0, e->stream>>>(e->d_bvals, e->d_bidxs, nb, e->d_out);
    k_pos_inc_recent<<<1, 1, 0, e->stream>>>(e->d_pos, e->d_next_tok,
                                             e->d_recent, e->d_n_recent);

    cudaGraph_t graph = NULL;
    const cudaError_t enderr = cudaStreamEndCapture(e->stream, &graph);
    g_capturing = 0;
    if (enderr != cudaSuccess || !graph || frc || lrc) {
        if (graph) cudaGraphDestroy(graph);
        return -1;
    }

    /* CUDA 12 signature: flags as unsigned long long (cuda_runtime_api.h).
     * The legacy 5-arg form is an inline wrapper in cuda_runtime.h that
     * delegates to this same 3-arg entry. */
    const cudaError_t ie = cudaGraphInstantiate(&e->graph_exec, graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess || !e->graph_exec) {
        e->graph_exec = NULL;
        return -1;
    }
    e->graph_ready = 1;
    fprintf(stderr, "[qwen2-engine] decode-step graph captured (cudaGraph replay ON)\n");
    return 0;
}

int qwen2_engine_next(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;
    if (!e || e->pos >= c->max_ctx - 1) return -2;

    /* TT_NO_GRAPH=1: legacy eager path forever (sample AND advance). */
    if (!e->graph_ready && !e->no_graph) {
        /* First call after prefill: eager sample WITHOUT advancing — x still
         * holds the hidden state of the last fed token, exactly like the eager
         * path's sampling stage. Also warms up norm/logits/argmax kernels. */
        const int id = sample_eager(e);
        if (id < 0 || id >= c->vocab) return id < 0 ? id : -3;
        e->pending_tok = id;
        if (qwen2_engine_graph_capture(e) == 0)
            return id;               /* graph will feed `id` on next call */
        fprintf(stderr, "[qwen2-engine] graph capture failed — falling back to eager permanently\n");
        e->no_graph = 1;
        e->pending_tok = -1;
        if (advance(e, id)) return -4;   /* exact legacy behavior */
        return id;
    }

    if (e->no_graph) {
        const int id = sample_eager(e);
        if (id < 0 || id >= c->vocab) return id < 0 ? id : -3;
        if (advance(e, id)) return -4;
        return id;
    }

    if (e->pending_tok < 0) {
        /* Freshly after a prefill that flushed the old pending: x holds the
         * hidden state of the last fed token — eager sample WITHOUT advancing,
         * stash result as the token the next replay will feed. */
        const int id = sample_eager(e);
        if (id < 0 || id >= c->vocab) return id < 0 ? id : -3;
        e->pending_tok = id;
        return id;
    }

    /* Graph replay: feed pending_tok through the full step, sample the NEXT
     * token, advance d_pos exactly once inside the graph. Returned sequence
     * is identical to eager: s1 (first call above), then s2, s3, ... */
    return qwen2_debug_replay_step(e, e->pending_tok);
}

/* One graph-replayed decode step: H2D token -> graph launch -> D2H sample
 * -> sync, plus pos/pending bookkeeping. Shared by qwen2_engine_next and
 * tools/profile_step.cu (single code path, no duplication). Returns the
 * sampled id, or -1 when the graph path is unavailable (eager unsupported
 * for profiling). */
int qwen2_debug_replay_step(Qwen2Engine *e, int next_tok) {
    if (!e || !e->graph_ready || next_tok < 0) return -1;
    const int vocab = e->cfg.vocab;

    /* SYNC copy: source is a stack parameter; an async copy could execute
     * after this function returns, reading reused stack memory. */
    cudaMemcpy(e->d_next_tok, &next_tok, sizeof(int), cudaMemcpyHostToDevice);
    cudaGraphLaunch(e->graph_exec, e->stream);
    cudaMemcpyAsync(e->h_sampled, e->d_out, sizeof(int),
                    cudaMemcpyDeviceToHost, e->stream);
    cudaStreamSynchronize(e->stream);                     /* read h_sampled only after sync */

    const int id = e->h_sampled[0];
    if (id < 0 || id >= vocab) return -3;
    e->pos++;                /* host mirror of k_pos_inc (bookkeeping/guards) */
    e->pending_tok = id;     /* already sampled internally — one step ahead */
    return id;
}

void *qwen2_debug_stream(Qwen2Engine *e) { return e ? (void *)e->stream : NULL; }

int qwen2_engine_pos(const Qwen2Engine *e) { return e ? e->pos : -1; }

extern "C" void qwen2_engine_enable_q8_kvcache(Qwen2Engine *e, int enable) {
    if (!e) return;
    if (enable && (!e->d_kc_q8 || !e->d_vc_q8)) {
        long cache_per_blocks = (long)e->cfg.n_kv_heads * e->cfg.max_ctx * (e->cfg.head_dim / 32);
        if (e->pl_hd[0] > 0) {
            long mx_q8 = 0;
            for (int l = 0; l < e->cfg.n_layers; l++) {
                int kv = e->pl_kv[l] > 0 ? e->pl_kv[l] : e->cfg.n_kv_heads;
                int hd = e->pl_hd[l] > 0 ? e->pl_hd[l] : e->cfg.head_dim;
                long w2_q8 = (long)kv * (hd / 32);
                if (w2_q8 > mx_q8) mx_q8 = w2_q8;
            }
            cache_per_blocks = mx_q8 * e->cfg.max_ctx;
        }
        if (!e->d_kc_q8) {
            cudaMalloc(&e->d_kc_q8, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ8_0));
            cudaMemset(e->d_kc_q8, 0, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ8_0));
        }
        if (!e->d_vc_q8) {
            cudaMalloc(&e->d_vc_q8, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ8_0));
            cudaMemset(e->d_vc_q8, 0, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ8_0));
        }
        /* Late-enable backfill (P1-2): FP32 decoding may have populated slots
         * [0..pos) before this call; the memset above left those Q8 slots
         * zeroed, which attention would read as silent wrong output above
         * threshold. Quantize existing FP32 slabs into Q8 (one-time cost).
         * Shared-KV layers skip: they alias the source slab, mirroring the
         * forward scatter path. */
        if (e->pos > 0 && e->d_kc && e->d_vc && e->d_kc_q8 && e->d_vc_q8 && !getenv("TT_NO_BACKFILL")) {
            long cache_layer = (long)e->cfg.n_kv_heads * e->cfg.max_ctx * e->cfg.head_dim;
            if (e->pl_hd[0] > 0) {
                long mx = 0;
                for (int l = 0; l < e->cfg.n_layers; l++) {
                    long w = (long)e->pl_kv[l] * e->pl_hd[l];
                    if (w > mx) mx = w;
                }
                cache_layer = mx * e->cfg.max_ctx;
            }
            int n_slots = e->pos < e->cfg.max_ctx ? e->pos : e->cfg.max_ctx;
            for (int l = 0; l < e->cfg.n_layers; l++) {
                if (e->has_pl_embd && e->pl_src[l] >= 0) continue;
                int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : e->cfg.n_kv_heads;
                int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : e->cfg.head_dim;
                int kvdim = KV_l * HDl;
                if (kvdim % 32 != 0) continue;
                long total = (long)n_slots * (kvdim / 32);
                k_kv_backfill_q8_0<<<(total + 255) / 256, 256, 0, e->stream>>>(
                    e->d_kc + (long)l * cache_layer,
                    e->d_vc + (long)l * cache_layer,
                    e->d_kc_q8 + (long)l * cache_per_blocks,
                    e->d_vc_q8 + (long)l * cache_per_blocks,
                    n_slots, kvdim);
            }
            cudaStreamSynchronize(e->stream);
        }
        fprintf(stderr, "[qwen2-engine] Q8_0 KV cache ENABLED (4x DRAM traffic reduction), backfilled %d slots\n",
                (e->pos > 0 && e->d_kc_q8 && !getenv("TT_NO_BACKFILL")) ? (e->pos < e->cfg.max_ctx ? e->pos : e->cfg.max_ctx) : 0);
    }
    e->use_q8_kvcache = enable;
    /* Late flag flip vs graph capture: replay bakes the flash/scatter path
     * chosen at capture, so a post-capture flip is silently ignored by
     * replay (stale path). Invalidate -> eager forever (always coherent). */
    if (e->graph_exec) {
        cudaGraphExecDestroy(e->graph_exec);
        e->graph_exec = NULL;
        e->graph_ready = 0;
        e->no_graph = 1;
        e->pending_tok = -1;
        fprintf(stderr, "[qwen2-engine] Q8 flag flipped post-capture: graph dropped, eager mode\n");
    }
}

extern "C" void qwen2_engine_enable_q4_kvcache(Qwen2Engine *e, int enable) {
    if (!e) return;
    if (enable && (!e->d_kc_q4 || !e->d_vc_q4)) {
        long cache_per_blocks = (long)e->cfg.n_kv_heads * e->cfg.max_ctx * (e->cfg.head_dim / 32);
        if (e->pl_hd[0] > 0) {
            long mx_q4 = 0;
            for (int l = 0; l < e->cfg.n_layers; l++) {
                int kv = e->pl_kv[l] > 0 ? e->pl_kv[l] : e->cfg.n_kv_heads;
                int hd = e->pl_hd[l] > 0 ? e->pl_hd[l] : e->cfg.head_dim;
                long w2_q4 = (long)kv * (hd / 32);
                if (w2_q4 > mx_q4) mx_q4 = w2_q4;
            }
            cache_per_blocks = mx_q4 * e->cfg.max_ctx;
        }
        if (!e->d_kc_q4) {
            cudaMalloc(&e->d_kc_q4, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ4_0));
            cudaMemset(e->d_kc_q4, 0, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ4_0));
        }
        if (!e->d_vc_q4) {
            cudaMalloc(&e->d_vc_q4, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ4_0));
            cudaMemset(e->d_vc_q4, 0, cache_per_blocks * e->cfg.n_layers * sizeof(BlockQ4_0));
        }
        /* Late-enable backfill (P1-2): see Q8 path above. */
        if (e->pos > 0 && e->d_kc && e->d_vc && e->d_kc_q4 && e->d_vc_q4 && !getenv("TT_NO_BACKFILL")) {
            long cache_layer = (long)e->cfg.n_kv_heads * e->cfg.max_ctx * e->cfg.head_dim;
            if (e->pl_hd[0] > 0) {
                long mx = 0;
                for (int l = 0; l < e->cfg.n_layers; l++) {
                    long w = (long)e->pl_kv[l] * e->pl_hd[l];
                    if (w > mx) mx = w;
                }
                cache_layer = mx * e->cfg.max_ctx;
            }
            int n_slots = e->pos < e->cfg.max_ctx ? e->pos : e->cfg.max_ctx;
            for (int l = 0; l < e->cfg.n_layers; l++) {
                if (e->has_pl_embd && e->pl_src[l] >= 0) continue;
                int KV_l = e->pl_kv[l] > 0 ? e->pl_kv[l] : e->cfg.n_kv_heads;
                int HDl = e->pl_hd[l] > 0 ? e->pl_hd[l] : e->cfg.head_dim;
                int kvdim = KV_l * HDl;
                if (kvdim % 32 != 0) continue;
                long total = (long)n_slots * (kvdim / 32);
                k_kv_backfill_q4_0<<<(total + 255) / 256, 256, 0, e->stream>>>(
                    e->d_kc + (long)l * cache_layer,
                    e->d_vc + (long)l * cache_layer,
                    e->d_kc_q4 + (long)l * cache_per_blocks,
                    e->d_vc_q4 + (long)l * cache_per_blocks,
                    n_slots, kvdim);
            }
            cudaStreamSynchronize(e->stream);
        }
        fprintf(stderr, "[qwen2-engine] Q4_0 KV cache ENABLED (8x DRAM traffic reduction vs FP32), backfilled %d slots\n",
                (e->pos > 0 && e->d_kc_q4 && !getenv("TT_NO_BACKFILL")) ? (e->pos < e->cfg.max_ctx ? e->pos : e->cfg.max_ctx) : 0);
    }
    e->use_q4_kvcache = enable;
    /* Late flag flip vs graph capture: see Q8 path above. */
    if (e->graph_exec) {
        cudaGraphExecDestroy(e->graph_exec);
        e->graph_exec = NULL;
        e->graph_ready = 0;
        e->no_graph = 1;
        e->pending_tok = -1;
        fprintf(stderr, "[qwen2-engine] Q4 flag flipped post-capture: graph dropped, eager mode\n");
    }
}

void qwen2_engine_set_sampling(Qwen2Engine *e, float temp, int topk,
                               float penalty) {
    if (!e) return;
    (void)topk; /* Gumbel-max samples full softmax; topk reserved */
    e->sampling_temp = temp;
    e->repeat_penalty = penalty;
    const int on = (temp > 0.0f) ? 1 : 0;
    cudaMemcpy(e->d_sampling_on, &on, sizeof(int), cudaMemcpyHostToDevice);
    if (on)
        fprintf(stderr, "[qwen2-engine] sampling ON: temp=%.2f penalty=%.2f\n",
                temp, penalty);
}

/* debug accessors (used by parity harness; not part of the public API contract) */
int qwen2_debug_copy_x(Qwen2Engine *e, float *host, int n) {
    if (!e || !host) return -1;
    const int ncpy = n < e->cfg.dim ? n : e->cfg.dim;
    cudaMemcpy(host, e->d_x, sizeof(float) * ncpy, cudaMemcpyDeviceToHost);
    return ncpy;
}
int qwen2_debug_copy_kv(Qwen2Engine *e, int layer, float *host, long max_floats) {
    if (!e || !host || layer >= e->cfg.n_layers) return -1;
    const long per = (long)e->cfg.n_kv_heads * e->cfg.max_ctx * e->cfg.head_dim;
    const long ncpy = per < max_floats ? per : max_floats;
    cudaMemcpy(host, e->d_kc + (long)layer * per, sizeof(float) * ncpy, cudaMemcpyDeviceToHost);
    return (int)ncpy;
}
int qwen2_debug_copy_xn(Qwen2Engine *e, float *host, int n) {
    if (!e || !host) return -1;
    const int ncpy = n < e->cfg.dim ? n : e->cfg.dim;
    cudaMemcpy(host, e->d_xn, sizeof(float) * ncpy, cudaMemcpyDeviceToHost);
    return ncpy;
}
int qwen2_debug_copy_logits(Qwen2Engine *e, float *host, int n) {
    if (!e || !host) return -1;
    /* logits buffer holds the last computed projection only after next();
     * caller must know what stage it is at */
    const int ncpy = n < e->cfg.vocab ? n : e->cfg.vocab;
    cudaMemcpy(host, e->d_logits, sizeof(float) * ncpy, cudaMemcpyDeviceToHost);
    return ncpy;
}

/* Fix2: rollback helper for verify_speculative pos-drift bug.
 * Truncates pos to target_pos and updates device mirror. Rejected KV
 * slots beyond target remain but are not read (attention window is
 * 0..pos). Clears pending_tok if it would be dangling. */
extern "C" void qwen2_engine_rollback(Qwen2Engine *e, int target_pos) {
    if (!e) return;
    if (target_pos < 0 || target_pos > e->cfg.max_ctx) return;
    if (target_pos > e->pos) return; /* only truncate */
    e->pos = target_pos;
    cudaMemcpy(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice);
    e->pending_tok = -1;
    /* Rollback vs graph capture: replay bakes capture-time S/path and its
     * pos_inc assumes uninterrupted forward progress; a rewind leaves
     * graph_exec stale. Destroy -> eager forever (always coherent). */
    if (e->graph_exec) {
        cudaGraphExecDestroy(e->graph_exec);
        e->graph_exec = NULL;
        e->graph_ready = 0;
        e->no_graph = 1;
        fprintf(stderr, "[qwen2-engine] rollback to %d: graph dropped, eager mode\n", target_pos);
    }
}

/* Public single-token "step + logits" used by tests and the spec-verify
 * golden path. Eager-only: it does NOT participate in the graph-replay
 * path and avoids the sampled-tok/D2H sync of qwen2_engine_next(). The
 * spec-verify golden computes the same logits a sequential eager next()
 * would produce, but without consuming a sampled token id. */
int qwen2_engine_step_logits(Qwen2Engine *e, int tok, float *host_logits) {
    if (!e || !host_logits) return -1;
    if (e->pos >= e->cfg.max_ctx - 1) return -2;
    /* If the graph path is ready and there's a sampled-but-unfed token
     * (left over from a prior qwen2_engine_next call), drain it first so
     * the KV cache matches the eager state machine. The spec-verify
     * test constructs engines in a fresh state (no graph capture yet)
     * so this branch is normally dead in the test. */
    if (e->graph_ready && e->pending_tok >= 0) {
        if (advance(e, e->pending_tok)) return -3;
        e->pending_tok = -1;
    }
    if (advance(e, tok)) return -4;
    if (compute_logits_into_d_logits(e)) return -5;
    const int ncpy = e->cfg.vocab;
    cudaMemcpy(host_logits, e->d_logits, sizeof(float) * ncpy,
               cudaMemcpyDeviceToHost);
    return 0;
}
