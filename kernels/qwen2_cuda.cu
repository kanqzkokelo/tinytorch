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
int tt_embed_q4_0(const void *dW, int tok, float *dx, int dim, cudaStream_t stream);
}

static size_t q4_bytes(long numel) { return (size_t)(numel / Q4_VALS_PER_BLOCK) * Q4_BYTES_PER_BLOCK; }

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
    /* KV staging rows (pre-scatter) */
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
    /* gemma4 MatFormer per-layer embeddings */
    TTensor pl_model_proj;        /* [n_layers*256, dim] typed */
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
    int pos;
    cudaStream_t stream;
};

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

    for (int l = 0; l < cfg->n_layers; l++) {
        LayerW *w = &e->L[l];
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);       upload_w(m, name, &w->q);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);       upload_w(m, name, &w->k);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);       upload_w(m, name, &w->v);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);  upload_w(m, name, &w->o);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);     upload_w(m, name, &w->gate);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);       upload_w(m, name, &w->up);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);     upload_w(m, name, &w->down);
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);    w->attn_norm = upload_f32(m, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);     w->ffn_norm  = upload_f32(m, name);
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);         w->q_bias    = upload_f32(m, name); /* optional */
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);         w->k_bias    = upload_f32(m, name); /* optional */
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
            if (!w->q_norm || !w->k_norm) {
                fprintf(stderr, "[qwen2-engine] qk_norm_rms trait set but attn_q/k_norm missing for layer %d\n", l);
                qwen2_engine_free(e); return NULL;
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
    /* heterogeneous layers (gemma4): size staging to per-layer maxima */
    int max_heads = cfg->n_heads, max_kv = cfg->n_kv_heads;
    long max_ffn = F;
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
            e->pl_heads[l] = qr > 0 ? (int)(qr / e->pl_hd[l]) : cfg->dim / cfg->head_dim;
            if (e->pl_heads[l] > max_heads) max_heads = e->pl_heads[l];
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
    cudaMalloc(&e->d_q,  (long)max_heads * cfg->head_dim * sizeof(float));
    cudaMalloc(&e->d_att, (long)max_heads * cfg->head_dim * sizeof(float));
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
    /* per-layer max kv width: gemma4 full layers carry 2x the kv heads */
    const long cache_per = (long)max_kv * cfg->max_ctx * cfg->head_dim;
    cudaMalloc(&e->d_kc, cache_per * cfg->n_layers * sizeof(float));
    cudaMalloc(&e->d_vc, cache_per * cfg->n_layers * sizeof(float));
    /* Zero every scratch/cache buffer: parity gates compare raw floats, so any
     * read of never-written device memory (garbage varies with physical page
     * assignment per process) shows up as cross-process nondeterminism.
     * All buffers are logically fully overwritten before use; this is a
     * deterministic-behavior belt-and-suspenders measure. */
    cudaMemset(e->d_kc, 0, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_vc, 0, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_xn, 0, D * sizeof(float));
    cudaMemset(e->d_q, 0, (long)cfg->n_heads * cfg->head_dim * sizeof(float));
    cudaMemset(e->d_att, 0, (long)cfg->n_heads * cfg->head_dim * sizeof(float));
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
     * are illegal, so per-stage profiling always runs graph-free. */
    e->no_graph = (getenv("TT_NO_GRAPH") || getenv("TT_PROFILE")) ? 1 : 0;
    e->pending_tok = -1;
    cudaMalloc(&e->d_next_tok, sizeof(int));
    cudaHostAlloc(&e->h_sampled, sizeof(int), cudaHostAllocDefault);
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
    cudaFree(e->d_k_stage); cudaFree(e->d_v_stage); cudaFree(e->d_pos);
    cudaFree(e->d_bvals); cudaFree(e->d_bidxs); cudaFree(e->d_out);
    if (e->d_recent) cudaFree(e->d_recent);
    if (e->d_n_recent) cudaFree(e->d_n_recent);
    if (e->h_sampled) cudaFreeHost(e->h_sampled);
    if (e->d_pl_tmp) cudaFree(e->d_pl_tmp);
    if (e->d_ple_row) cudaFree(e->d_ple_row);
    if (e->d_ones) cudaFree(e->d_ones);
    if (e->ple_pe) free(e->ple_pe);
    if (e->pl_proj_norm_host) free(e->pl_proj_norm_host);
    cudaFree(e->d_next_tok);
    if (e->graph_exec) cudaGraphExecDestroy(e->graph_exec);
    /* NOTE: per-weight cudaFree calls are intentionally not tracked here; they are
     * leaked until process exit by design (engine lifetime == process lifetime).
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
static int forward_layers(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;
    const int HD = c->head_dim;
    long cache_layer = c->n_kv_heads * (long)c->max_ctx * HD;   /* stride matches alloc */
    if (e->pl_hd[0] > 0) {
        long mx = 0;
        for (int l = 0; l < c->n_layers; l++) {
            const long w2 = (long)e->pl_kv[l] * e->pl_hd[l];
            if (w2 > mx) mx = w2;
        }
        cache_layer = mx * c->max_ctx;
    }
    dim3 g, b;

    static int trace = -1;
    if (trace < 0) trace = getenv("TT_TRACE") ? 1 : 0;
    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &e->L[l];
        float *Kl_f = e->d_kc + l * cache_layer;
        float *Vl_f = e->d_vc + l * cache_layer;
        /* gemma4 KV sharing: shared layers (pl_src[l] >= 0) read the source
         * layer's cache slab instead of computing/scattering their own K/V.
         * llama-model.cpp:2502 semantics. */
        const int kv_shared = e->has_pl_embd && e->pl_src[l] >= 0;
        if (kv_shared) {
            Kl_f = e->d_kc + (long)e->pl_src[l] * cache_layer;
            Vl_f = e->d_vc + (long)e->pl_src[l] * cache_layer;
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
        int qrc = tt_gemv_typed(w->q.ptr, w->q.dtype, e->d_xn, e->d_q,
                      attn_qout, c->dim, e->stream);
        if (qrc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] q gemv rc=%d\n", qrc);
        if (!kv_shared) {
            int krc = tt_gemv_typed(w->k.ptr, w->k.dtype, e->d_xn, e->d_k_stage, c->n_kv_heads * HD, c->dim, e->stream);
            if (krc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] k gemv rc=%d\n", krc);
        }
        /* QKV biases present in some GGUF conversions of Qwen2 (applied by
         * llama.cpp whenever the tensors exist). Optional by design. */
        CHK_STAGE("2 qkv-gemv");
        const int kvdim = KV_l * HD;
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
            g.x = (HDl / 2 + 63) / 64; g.y = H_l; g.z = 1;
            b.x = 64; b.y = 1; b.z = 1;
            if (ff_l)
                k_rope_ff<<<g, b, 0, e->stream>>>(e->d_q, H_l, HDl, e->d_pos, base_l, ff_l);
            else
                rope_fn<<<g, b, 0, e->stream>>>(e->d_q, H_l, HDl, e->d_pos, base_l);
            g.x = (HDl / 2 + 63) / 64; g.y = KV_l; g.z = 1;
            if (!kv_shared) {
            if (ff_l)
                k_rope_ff<<<g, b, 0, e->stream>>>(e->d_k_stage, KV_l, HDl, e->d_pos, base_l, ff_l);
            else
                rope_fn<<<g, b, 0, e->stream>>>(e->d_k_stage, KV_l, HDl, e->d_pos, base_l);
            }
        }

        /* v projection + bias (QKV group) */
        if (!kv_shared) {
        int vrc = tt_gemv_typed(w->v.ptr, w->v.dtype, e->d_xn, e->d_v_stage, kvdim_l, c->dim, e->stream);
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
        k_kv_scatter<<<(kvdim_l + 255) / 256, 256, 0, e->stream>>>(
            e->d_k_stage, e->d_v_stage, Kl_f, Vl_f, e->d_pos,
            KV_l, HD, c->max_ctx);
        if (tt_profiling()) tt_prof_end(TT_P_SCATTER, e->stream);
        }

        CHK_STAGE("3 rope");
        /* 4. GQA flash attention over slots [t0..pos]; t0 raised by the SWA
         * trait (gemma2), full [0..pos] when swa_size == 0 (qwen2 unchanged). */
        if (tt_profiling()) tt_prof_begin(TT_P_FLASH, e->stream);
        k_flash_gqa<<<H_l, 32, 0, e->stream>>>(
            e->d_q, Kl_f, Vl_f, e->d_att,
            e->d_pos,
            H_l, KV_l, HDl, c->max_ctx,
            (c->tr.attn_scale_one ? 1.0f : 1.0f / sqrtf((float)HD)),
            e->has_pl_embd ? e->pl_swa[l] : c->tr.swa_size);
        if (tt_profiling()) tt_prof_end(TT_P_FLASH, e->stream);

        CHK_STAGE("4 flash");
        if (trace && l == 0) {
            eng_rms(e, e->d_v_stage, "v", kvdim_l);
            eng_rms(e, e->d_att, "ao", attn_qout);
        }
        if (trace && e->has_pl_embd && l >= 18 && l <= 21) ple_canary(e, "flash", l);
        /* 5. Wo projection + residual: x += att @ Wo^T */
        if (tt_profiling()) tt_prof_begin(TT_P_OMLP, e->stream);
        int orc_ = tt_gemv_typed(w->o.ptr, w->o.dtype, e->d_att, e->d_xn, c->dim, attn_qout, e->stream);
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
            int grc = tt_gemv_typed(w->gate.ptr, w->gate.dtype, e->d_xn, e->d_g,
                          FF_l, c->dim, e->stream);
            if (grc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] gate gemv rc=%d\n", grc);
            CHK_STAGE("7a gate-gemv");
            int urc = tt_gemv_typed(w->up.ptr, w->up.dtype, e->d_xn, e->d_u,
                          FF_l, c->dim, e->stream);
            if (urc && getenv("TT_DEBUG")) fprintf(stderr, "[qwen2-engine] up gemv rc=%d\n", urc);
            CHK_STAGE("7b up-gemv");
            k_swiglu_apply<<<(FF_l + 255) / 256, 256, 0, e->stream>>>(
                e->d_g, e->d_u, e->d_h, FF_l, act_gelu);
            CHK_STAGE("6b swiglu-apply");
        }
        int drc = tt_gemv_typed(w->down.ptr, w->down.dtype, e->d_h, e->d_xn,
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

        /* gemma4 MatFormer per-layer embedding block (all host round-trips;
         * pl_dim=256 so data movement is trivial). Order per build_gemma4:
         *   g = gelu(inp_gate @ x) * PLE[l] ; p = rmsnorm(pl_proj @ g) ;
         *   x += p ; x *= layer_output_scale                        */
        static int no_ple2 = -1;
        if (no_ple2 < 0) no_ple2 = getenv("TT_NO_PLE") ? 1 : 0;
        if (e->has_pl_embd && !no_ple2) {
            const int slot = e->pos % c->max_ctx;
            static float gbuf[1024];
            static float ple_slice[1024];
            static float pe_slice[1024];

            /* g = inp_gate @ x (device) */
            int ig = tt_gemv_typed(w->inp_gate.ptr, w->inp_gate.dtype,
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
            /* gelu then elementwise multiply by PLE slice — NOTE: per
             * build_gemma4 the multiply happens AFTER gelu but the residual
             * uses pre-gate cur; also pe is added post-projection in
             * project_per_layer_inputs, already folded into ple_finish_host. */
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
            int pg = tt_gemv_typed(w->pl_proj.ptr, w->pl_proj.dtype,
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
        int prc = tt_gemv_typed(e->pl_model_proj.ptr, e->pl_model_proj.dtype,
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

        /* dequant per_layer_token_embd row 'tok' (q4_0), scale by sqrt(pl_dim) */
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
        /* NOTE: llama.cpp project_per_layer_inputs adds the RAW get_rows
         * output — no sqrt(pl_dim) scaling on the token-embedding side. */
        ttq_dequant((const char *)tp->data + (long)tok * rb, tp->type, row, pe_full);

        /* CPU: per-slice rmsnorm w/ pl_proj_norm, then add pe, then *1/sqrt2 */
        fprintf(stderr, "[PLE] E finish\n");
        if (getenv("TT_TRACE")) {
            int nb1 = 0, nb2 = 0;
            for (long i = 0; i < row; i++) {
                if (isnan(pe_full[i])) nb1++;
                if (isnan(e->ple_pe[i])) nb2++;
            }
            fprintf(stderr, "[PLE] nan-check: pe_full_nan=%d ple_pe_nan=%d\n", nb1, nb2);
        }
        ple_finish_host(e->ple_pe, e->pl_proj_norm_host, pe_full,
                        e->cfg.n_layers);
        fprintf(stderr, "[PLE] F done\n");

        cudaMemcpy(e->d_ple_row, e->ple_pe, row * sizeof(float),
                   cudaMemcpyHostToDevice);
        if (getenv("TT_TRACE"))
            k_fill_const<<<(8960*4 - 5376*4 + 255)/256, 256, 0, e->stream>>>(
                e->d_ple_row + 5376, 8960 - 5376, -777.0f);
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
    for (int i = 0; i < n; i++) {
        int rc = advance(e, toks[i]);
        if (rc) return rc;
    }
    cudaStreamSynchronize(e->stream);
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

    /* M7: the dynamic-token embed kernel is q4_0-only; other embedding
     * dtypes run eager (typed embed reads the host token id). */
    if (e->d_embd.dtype != GGUF_TYPE_Q4_0) return -1;

    /* dummy valid token before capture begins (plain, uncaptured copy) */
    const int dummy = e->pending_tok >= 0 ? e->pending_tok : 0;
    cudaMemcpy(e->d_next_tok, &dummy, sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(e->stream);

    /* warm up graph-only kernels eagerly (lazy module load must not happen
     * implicitly during capture). Save/restore engine state they touch.
     * NOTE: d_logits must be RESTORED, not zeroed — callers (dump_logits,
     * parity gate) read it right after this returns, while the sampled id
     * from sample_eager is still the current answer. */
    {
        const int threads = c->dim / 32;
        float *xsave = NULL, *lsave = NULL;
        cudaMalloc(&xsave, c->dim * sizeof(float));
        cudaMalloc(&lsave, c->vocab * sizeof(float));
        cudaMemcpy(xsave, e->d_x, c->dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(lsave, e->d_logits, c->vocab * sizeof(float), cudaMemcpyDeviceToDevice);
        const int pos_before = e->pos;
        k_embed_q4_0_dyn<<<(threads + 255) / 256, 256, 0, e->stream>>>(
            (const BlockQ4_0 *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
        k_pos_inc<<<1, 1, 0, e->stream>>>(e->d_pos);
        k_repeat_penalty<<<1, 256, 0, e->stream>>>(e->d_logits, e->d_recent,
                                                   e->d_n_recent, c->vocab, 2.0f);
        k_gumbel_transform<<<1, 256, 0, e->stream>>>(e->d_logits, c->vocab,
                                                     0.8f, e->d_pos, e->d_sampling_on);
        cudaStreamSynchronize(e->stream);
        /* restore state the warmup perturbed */
        cudaMemcpy(e->d_x, xsave, c->dim * sizeof(float), cudaMemcpyHostToDevice);
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
        const int threads = c->dim / 32;
        k_embed_q4_0_dyn<<<(threads + 255) / 256, 256, 0, e->stream>>>(
            (const BlockQ4_0 *)e->d_embd.ptr, e->d_next_tok, e->d_x, c->dim);
    }
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
