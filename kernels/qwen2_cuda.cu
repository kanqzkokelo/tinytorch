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

/* BlockQ4_0 comes from loader_gguf.h (d stored as raw fp16 bits). */
#define Q4_D(blk) __half2float(*(const __half *)&(blk).d)

#define Q4_BYTES_PER_BLOCK 18
#define Q4_VALS_PER_BLOCK 32

/* launchers implemented in kernels/gemv_q4_cuda.cu */
extern "C" {
typedef struct CUstream_st *cudaStream_t;
int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K,
                 cudaStream_t stream);
int tt_swiglu_q4_0(const void *dGate, const void *dUp, const float *dx,
                   float *dh, int M, int K, cudaStream_t stream);
int tt_logits_q4_0(const void *dW, const float *dx, float *dlogits,
                   int vocab, int K, cudaStream_t stream);
int tt_logits_dispatch(const void *dW, int is_q8, const float *dx,
                       float *dlogits, int vocab, int K, cudaStream_t stream);
int tt_embed_q4_0(const void *dW, int tok, float *dx, int dim, cudaStream_t stream);
}

static size_t q4_bytes(long numel) { return (size_t)(numel / Q4_VALS_PER_BLOCK) * Q4_BYTES_PER_BLOCK; }

/* ---------------- device kernels (engine-local ops) ---------------- */

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

/* y = x / sqrt(mean(x^2) + eps) * gamma ; one block per row. */
__global__ void k_rmsnorm(const float *__restrict__ x, const float *__restrict__ g,
                          float *__restrict__ y, int dim, float eps) {
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
    for (int i = tid; i < dim; i += blockDim.x) y[i] = x[i] * inv * g[i];
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
                            int max_ctx, float scale) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);          /* GQA group map */
    const int elems = head_dim / 32;                     /* per-lane elements */
    const float *qh = q + (long)h * head_dim + lane * elems;

    float qreg[8];
#pragma unroll
    for (int i = 0; i < 8; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    /* INVARIANT: callers enforce pos < max_ctx (no ring wraparound in this loop) */
    for (int t = 0; t <= pos; t++) {
        /* slot-major layout: [slot][kv_head*head_dim], matches GEMV writes */
        const long off = ((long)t * n_kv_heads + kvh) * head_dim + lane * elems;
        const float *kp = Kc + off;
        const float *vp = Vc + off;
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (i < elems) score += qreg[i] * kp[i];
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        const float m_new = fmaxf(m_prev, score);
        const float ex = expf(score - m_new);
        const float alpha = expf(m_prev - m_new);
        l_prev = l_prev * alpha + ex;
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (i < elems) oreg[i] = oreg[i] * alpha + ex * vp[i];
        m_prev = m_new;
    }

    const float inv_l = 1.0f / (l_prev + 1e-8f);
    float *oh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 8; i++)
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

/* Two-stage argmax over vocab. */
__global__ void k_argmax_partial(const float *__restrict__ x, int n,
                                 float *__restrict__ bvals, int *__restrict__ bidxs) {
    __shared__ float sv[128];
    __shared__ int si[128];
    const int tid = threadIdx.x;
    float best = -INFINITY; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    sv[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            if (sv[tid + s2] > sv[tid]) { sv[tid] = sv[tid + s2]; si[tid] = si[tid + s2]; }
        }
        __syncthreads();
    }
    if (tid == 0) { bvals[blockIdx.x] = sv[0]; bidxs[blockIdx.x] = si[0]; }
}

__global__ void k_argmax_final(const float *__restrict__ bvals, const int *__restrict__ bidxs,
                               int nb, int *__restrict__ out) {
    float best = -INFINITY; int bi = 0;
    for (int i = 0; i < nb; i++)
        if (bvals[i] > best) { best = bvals[i]; bi = bidxs[i]; }
    *out = bi;
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

/* ---------------- host-side engine ---------------- */

#define MAX_LAYERS 128
#define MAX_DIM 16384

struct LayerW {
    const BlockQ4_0 *q, *k, *v, *o, *gate, *up, *down;
    float *attn_norm, *ffn_norm;      /* device f32 gammas */
    float *q_bias, *k_bias, *v_bias;  /* this GGUF variant carries QKV biases */
};

struct Qwen2Engine {
    TTConfig cfg;
    LayerW L[MAX_LAYERS];
    const BlockQ4_0 *d_embd;          /* tied or untied lm head below */
    const BlockQ4_0 *d_out_w;
    float *d_out_norm;
    int out_is_q8;                    /* lm head stored as q8_0 */
    /* activations */
    float *d_x, *d_xn, *d_q, *d_att, *d_h, *d_logits;
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
    int pending_tok;              /* sampled token not yet fed through layers */
    int pos;
    cudaStream_t stream;
};

static float *upload_f32(GGUFModel *m, const char *name) {
    GGUFTensor *t = gguf_get_tensor(m, name);
    if (!t || !t->data) { fprintf(stderr, "[qwen2-engine] f32 upload missing: %s\n", name); return NULL; }
    float *d = NULL;
    if (cudaMalloc(&d, t->size_bytes) != cudaSuccess) { fprintf(stderr, "[qwen2-engine] cudaMalloc fail %s\n", name); return NULL; }
    cudaMemcpy(d, t->data, t->size_bytes, cudaMemcpyHostToDevice);
    return d;
}

static const BlockQ4_0 *upload_q4(GGUFModel *m, const char *name) {
    GGUFTensor *t = gguf_get_tensor(m, name);
    if (!t || !t->data) { fprintf(stderr, "[qwen2-engine] q4 upload missing: %s\n", name); return NULL; }
    void *d = NULL;
    if (cudaMalloc(&d, t->size_bytes) != cudaSuccess) { fprintf(stderr, "[qwen2-engine] cudaMalloc fail %s\n", name); return NULL; }
    cudaMemcpy(d, t->data, t->size_bytes, cudaMemcpyHostToDevice);
    return (const BlockQ4_0 *)d;
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
    c.head_dim = c.dim / c.n_heads;
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
    e->cfg = *cfg;
    e->pos = 0;
    cudaStreamCreate(&e->stream);

    const long D = cfg->dim, F = cfg->hidden_dim;
    char name[160];

    /* resolve vocab from the embedding tensor's actual shape */
    GGUFTensor *tembd = gguf_get_tensor(m, "token_embd.weight");
    if (!tembd) { fail("token_embd.weight missing"); return NULL; }
    e->cfg.vocab = (int)tembd->shape[tembd->ndim - 1];

    e->d_embd = upload_q4(m, "token_embd.weight");

    {
        GGUFTensor *tw = gguf_get_tensor(m, "output.weight");
        if (tw && tw->data && tw->type == GGUF_TYPE_Q8_0) {
            e->out_is_q8 = 1;
            void *d = NULL;
            cudaMalloc(&d, tw->size_bytes);
            cudaMemcpy(d, tw->data, tw->size_bytes, cudaMemcpyHostToDevice);
            e->d_out_w = (const BlockQ4_0 *)d;
        } else {
            e->d_out_w = upload_q4(m, "output.weight");
        }
    }
    if (!e->d_out_w || getenv("TT_FORCE_TIED")) {         /* tied embeddings fallback */
        e->d_out_w = e->d_embd;
        e->out_is_q8 = 0;
        fprintf(stderr, "[qwen2-engine] using TIED embedding as lm head\n");
    }
    fprintf(stderr, "[qwen2-engine] lm head: %s\n", e->out_is_q8 ? "output.weight(q8_0)" : "tied/token_embd(q4_0)");
    e->d_out_norm = upload_f32(m, "output_norm.weight");

    for (int l = 0; l < cfg->n_layers; l++) {
        LayerW *w = &e->L[l];
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);       w->q    = upload_q4(m, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);       w->k    = upload_q4(m, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);       w->v    = upload_q4(m, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);  w->o    = upload_q4(m, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);     w->gate = upload_q4(m, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);       w->up   = upload_q4(m, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);     w->down = upload_q4(m, name);
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);    w->attn_norm = upload_f32(m, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);     w->ffn_norm  = upload_f32(m, name);
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);         w->q_bias    = upload_f32(m, name); /* optional */
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);         w->k_bias    = upload_f32(m, name); /* optional */
        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);         w->v_bias    = upload_f32(m, name); /* optional */
        if (!w->q || !w->k || !w->v || !w->o || !w->gate || !w->up || !w->down ||
            !w->attn_norm || !w->ffn_norm) {  /* v_bias optional: absent in stock Qwen2 */
            fprintf(stderr, "[qwen2-engine] missing weights for layer %d\n", l);
            qwen2_engine_free(e); return NULL;
        }
    }
    if (!e->d_embd || !e->d_out_norm) ABORT_CREATE("missing embedding/output_norm");

    /* activations + caches */
    cudaMalloc(&e->d_x,  D * sizeof(float));
    cudaMalloc(&e->d_xn, D * sizeof(float));
    cudaMalloc(&e->d_q,  (long)cfg->n_heads * cfg->head_dim * sizeof(float));
    cudaMalloc(&e->d_att, (long)cfg->n_heads * cfg->head_dim * sizeof(float));
    cudaMalloc(&e->d_h,  F * sizeof(float));
    cudaMalloc(&e->d_logits, (long)e->cfg.vocab * sizeof(float));
    const long cache_per = (long)cfg->n_kv_heads * cfg->max_ctx * cfg->head_dim;
    cudaMalloc(&e->d_kc, cache_per * cfg->n_layers * sizeof(float));
    cudaMalloc(&e->d_vc, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_x, 0, D * sizeof(float));
    const long kvdim_alloc = (long)cfg->n_kv_heads * cfg->head_dim;
    cudaMalloc(&e->d_k_stage, kvdim_alloc * sizeof(float));
    cudaMalloc(&e->d_v_stage, kvdim_alloc * sizeof(float));
    cudaMalloc(&e->d_pos, sizeof(int));
    cudaMemsetAsync(e->d_pos, 0, sizeof(int), e->stream);   /* pos starts at 0 on device */
    const int nb = 256;
    cudaMalloc(&e->d_bvals, nb * sizeof(float));
    cudaMalloc(&e->d_bidxs, nb * sizeof(int));
    cudaMalloc(&e->d_out, sizeof(int));
    /* graph replay state */
    e->graph_exec = NULL;
    e->graph_ready = 0;
    e->no_graph = getenv("TT_NO_GRAPH") ? 1 : 0;
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
    cudaFree(e->d_logits); cudaFree(e->d_kc); cudaFree(e->d_vc);
    cudaFree(e->d_k_stage); cudaFree(e->d_v_stage); cudaFree(e->d_pos);
    cudaFree(e->d_bvals); cudaFree(e->d_bidxs); cudaFree(e->d_out);
    cudaFree(e->d_next_tok);
    if (e->h_sampled) cudaFreeHost(e->h_sampled);
    if (e->graph_exec) cudaGraphExecDestroy(e->graph_exec);
    /* NOTE: per-weight cudaFree calls are intentionally not tracked here; they are
     * leaked until process exit by design (engine lifetime == process lifetime).
     * Tracked as known limitation in PLAN_M6 M6.1. */
    if (e->stream) cudaStreamDestroy(e->stream);
    free(e);
}

/* 0 while running eagerly; 1 while a cudaStream capture is in flight. */
static int g_capturing = 0;

static int forward_layers(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;
    const int HD = c->head_dim;
    const long cache_layer = (long)c->n_kv_heads * c->max_ctx * HD;
    dim3 g, b;

    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &e->L[l];
        float *Kl_f = e->d_kc + l * cache_layer;
        float *Vl_f = e->d_vc + l * cache_layer;

        /* Cache layout: [slot][kv_head * head_dim] so each GEMV output of
         * width n_kv_heads*HD lands contiguously per slot.
         * Flash kernel indexes K(t,kvh,i) = Kl_f[(t*n_kv_heads + kvh)*HD + i].
         * K/V are computed into staging buffers, then scattered to the slot
         * selected by the DEVICE position scalar (*e->d_pos). */

        /* 1. xn = rmsnorm(x) * attn_norm */
        k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
            e->d_x, w->attn_norm, e->d_xn, c->dim, c->rms_eps);

        /* 2. projections: q -> d_q ; k,v -> KV cache slot pos */
        tt_gemv_q4_0(w->q, e->d_xn, e->d_q, c->dim, c->dim, e->stream);
        tt_gemv_q4_0(w->k, e->d_xn, e->d_k_stage, c->n_kv_heads * HD, c->dim, e->stream);
        /* QKV biases present in some GGUF conversions of Qwen2 (applied by
         * llama.cpp whenever the tensors exist). Optional by design. */
        const int kvdim = c->n_kv_heads * HD;
        if (w->q_bias)
            k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_q, w->q_bias, c->dim);
        if (w->k_bias)
            k_add<<<(kvdim + 255) / 256, 256, 0, e->stream>>>(e->d_k_stage, w->k_bias, kvdim);

        /* 3. RoPE on q (all heads) and on the staged k row (in-place, pre-scatter) */
        static int no_rope = -1;
        if (no_rope < 0) no_rope = getenv("TT_NO_ROPE") ? 1 : 0;
        if (!no_rope) {
            g.x = (HD / 2 + 63) / 64; g.y = c->n_heads; g.z = 1;
            b.x = 64; b.y = 1; b.z = 1;
            k_rope<<<g, b, 0, e->stream>>>(e->d_q, c->n_heads, HD, e->d_pos, c->rope_base);
            g.x = (HD / 2 + 63) / 64; g.y = c->n_kv_heads; g.z = 1;
            k_rope<<<g, b, 0, e->stream>>>(e->d_k_stage, c->n_kv_heads, HD, e->d_pos, c->rope_base);
        }

        /* v projection + bias, then scatter staged K/V into the cache slot
         * chosen by *d_pos. Must precede flash attention. */
        tt_gemv_q4_0(w->v, e->d_xn, e->d_v_stage, c->n_kv_heads * HD, c->dim, e->stream);
        if (w->v_bias)
            k_add<<<(kvdim + 255) / 256, 256, 0, e->stream>>>(e->d_v_stage, w->v_bias, kvdim);
        k_kv_scatter<<<(kvdim + 255) / 256, 256, 0, e->stream>>>(
            e->d_k_stage, e->d_v_stage, Kl_f, Vl_f, e->d_pos,
            c->n_kv_heads, HD, c->max_ctx);

        /* 4. GQA flash attention over slots [0..pos] */
        k_flash_gqa<<<c->n_heads, 32, 0, e->stream>>>(
            e->d_q, Kl_f, Vl_f, e->d_att,
            e->d_pos,
            c->n_heads, c->n_kv_heads, HD, c->max_ctx,
            1.0f / sqrtf((float)HD));

        /* 5. Wo projection + residual: x += att @ Wo^T */
        tt_gemv_q4_0(w->o, e->d_att, e->d_xn, c->dim, c->dim, e->stream);
        k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_x, e->d_xn, c->dim);

        /* 6. ffn norm */
        k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
            e->d_x, w->ffn_norm, e->d_xn, c->dim, c->rms_eps);

        /* 7. fused SwiGLU MLP */
        tt_swiglu_q4_0(w->gate, w->up, e->d_xn, e->d_h, c->hidden_dim, c->dim, e->stream);

        /* 8. down projection + residual */
        tt_gemv_q4_0(w->down, e->d_h, e->d_xn, c->dim, c->hidden_dim, e->stream);
        k_add<<<(c->dim + 255) / 256, 256, 0, e->stream>>>(e->d_x, e->d_xn, c->dim);

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

static int embed_token(Qwen2Engine *e, int tok) {
    return tt_embed_q4_0(e->d_embd, tok, e->d_x, e->cfg.dim, e->stream);
}

static int advance(Qwen2Engine *e, int tok) {
    int rc = embed_token(e, tok);
    if (rc) return rc;
    rc = forward_layers(e);      /* runs while *d_pos == current slot */
    if (rc) return rc;
    e->pos++;
    cudaMemcpyAsync(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice, e->stream);
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
    /* resync device position scalar before any forward work */
    cudaMemcpyAsync(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice, e->stream);
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
static int sample_eager(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;
    k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
        e->d_x, e->d_out_norm, e->d_xn, c->dim, c->rms_eps);
    int rc = tt_logits_dispatch(e->d_out_w, e->out_is_q8, e->d_xn,
                                e->d_logits, c->vocab, c->dim, e->stream);
    if (rc) return -1;

    const int nb = 256;
    k_argmax_partial<<<nb, 128, 0, e->stream>>>(e->d_logits, c->vocab, e->d_bvals, e->d_bidxs);
    k_argmax_final<<<1, 1, 0, e->stream>>>(e->d_bvals, e->d_bidxs, nb, e->d_out);

    int id = 0;
    cudaMemcpyAsync(&id, e->d_out, sizeof(int), cudaMemcpyDeviceToHost, e->stream);
    cudaStreamSynchronize(e->stream);
    return id;
}

/* Capture the whole decode step into a single-launch graph:
 *   embed(d_next_tok) -> forward_layers -> rmsnorm -> logits -> argmax -> pos_inc
 * All nodes on e->stream. The captured embed reads whatever token sits in
 * d_next_tok at REPLAY time (set per-call via async H2D before the launch).
 * On success sets graph_ready and returns 0; leaves graph_ready=0 otherwise. */
static int qwen2_engine_graph_capture(Qwen2Engine *e) {
    const TTConfig *c = &e->cfg;

    /* dummy valid token before capture begins (plain, uncaptured copy) */
    const int dummy = e->pending_tok >= 0 ? e->pending_tok : 0;
    cudaMemcpy(e->d_next_tok, &dummy, sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(e->stream);

    /* warm up graph-only kernels eagerly (lazy module load must not happen
     * implicitly during capture). Save/restore engine state they touch. */
    {
        const int threads = c->dim / 32;
        float *xsave = NULL;
        cudaMalloc(&xsave, c->dim * sizeof(float));
        cudaMemcpy(xsave, e->d_x, c->dim * sizeof(float), cudaMemcpyDeviceToHost);
        const int pos_before = e->pos;
        k_embed_q4_0_dyn<<<(threads + 255) / 256, 256, 0, e->stream>>>(
            e->d_embd, e->d_next_tok, e->d_x, c->dim);
        k_pos_inc<<<1, 1, 0, e->stream>>>(e->d_pos);
        cudaStreamSynchronize(e->stream);
        /* restore state the warmup perturbed */
        cudaMemcpy(e->d_x, xsave, c->dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(e->d_pos, &pos_before, sizeof(int), cudaMemcpyHostToDevice);
        cudaFree(xsave);
    }

    g_capturing = 1;
    if (cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        g_capturing = 0;
        return -1;
    }

    {   /* dynamic-token embedding (device-side id) */
        const int threads = c->dim / 32;
        k_embed_q4_0_dyn<<<(threads + 255) / 256, 256, 0, e->stream>>>(
            e->d_embd, e->d_next_tok, e->d_x, c->dim);
    }
    const int frc = forward_layers(e);          /* all layers at *d_pos */
    k_rmsnorm<<<1, 256, 256 * sizeof(float), e->stream>>>(
        e->d_x, e->d_out_norm, e->d_xn, c->dim, c->rms_eps);
    const int lrc = tt_logits_dispatch(e->d_out_w, e->out_is_q8, e->d_xn,
                                       e->d_logits, c->vocab, c->dim, e->stream);
    const int nb = 256;
    k_argmax_partial<<<nb, 128, 0, e->stream>>>(e->d_logits, c->vocab, e->d_bvals, e->d_bidxs);
    k_argmax_final<<<1, 1, 0, e->stream>>>(e->d_bvals, e->d_bidxs, nb, e->d_out);
    k_pos_inc<<<1, 1, 0, e->stream>>>(e->d_pos);

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
    cudaMemcpyAsync(e->d_next_tok, &e->pending_tok, sizeof(int),
                    cudaMemcpyHostToDevice, e->stream);   /* same stream, before launch */
    cudaGraphLaunch(e->graph_exec, e->stream);
    cudaMemcpyAsync(e->h_sampled, e->d_out, sizeof(int),
                    cudaMemcpyDeviceToHost, e->stream);
    cudaStreamSynchronize(e->stream);                     /* read h_sampled only after sync */

    const int id = e->h_sampled[0];
    if (id < 0 || id >= c->vocab) return -3;
    e->pos++;                /* host mirror of k_pos_inc (bookkeeping/guards) */
    e->pending_tok = id;     /* already sampled internally — one step ahead */
    return id;
}

int qwen2_engine_pos(const Qwen2Engine *e) { return e ? e->pos : -1; }

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
