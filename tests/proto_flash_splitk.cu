/* tests/proto_flash_splitk.cu
 *
 * PROTOTYPE: split-K flash attention decode for long context.
 *
 * Problem: k_flash_gqa (kernels/qwen2_cuda.cu) launches ONE warp PER HEAD
 * and iterates slots serially -> cost grows linearly with ctx while the
 * grid stays tiny (35 heads x 1 warp = 35 blocks on 20+ SMs). At long ctx
 * this underutilizes the GPU badly.
 *
 * V0 (baseline): verbatim serial warp-per-head kernel, for reference timing.
 * V1 (split-K):  S blocks per head, block s owns slot subrange
 *                [t_lo + s*chunk, min(pos+1, t_lo + (s+1)*chunk)) of the
 *                attended window, runs the same online-softmax loop, and
 *                writes fp32 partials (m, l, acc[hd]) to a workspace.
 *                A second combine kernel merges the S partials with the
 *                standard online-softmax merge (numerically stable, order
 *                independent up to fp rounding).
 *
 * Build:
 *   $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 tests/proto_flash_splitk.cu -o /tmp/proto_flash_splitk
 *   LD_LIBRARY_PATH=$HOME/mmcuda/lib /tmp/proto_flash_splitk
 *
 * kernels/ untouched. If qwen2_cuda.cu changes, re-copy V0 here.
 *
 * Covered:
 *  1. Correctness vs CPU double reference (gated rel < 1e-5) at several ctx,
 *     incl. SWA window and empty-split edge (ctx < S).
 *  2. Numerical-order independence: V1(S=4) vs V1(S=16) vs V1(S=64) and V0
 *     must agree within fp noise.
 *  3. Benchmark: us vs ctx in {512, 4096, 16384, 32768} (+ finer low-ctx
 *     sweep for the crossover point), S in {4, 16, 64}.
 *  4. Occupancy report (blocks/SM before vs after).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define CHECK(call)                                                            \
    do {                                                                        \
        cudaError_t e_ = (call);                                                \
        if (e_ != cudaSuccess) {                                                \
            printf("[CUDA FAIL] %s:%d %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(e_));                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* ==========================================================================
 * V0: verbatim copy of k_flash_gqa + warp_sum from kernels/qwen2_cuda.cu
 * (same rationale as test_flash_multi.cu: cannot #include engine headers)
 * ======================================================================== */

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void k_flash_gqa(const float *__restrict__ q,
                            const float *__restrict__ Kc,
                            const float *__restrict__ Vc,
                            float *__restrict__ out,
                            const int *__restrict__ d_pos,
                            int n_heads, int n_kv_heads, int head_dim,
                            int max_ctx, float scale, int window) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);
    const int elems = head_dim / 32;
    const float *qh = q + (long)h * head_dim + lane * elems;

    int t0 = 0;
    if (window > 0 && pos >= window) t0 = pos - window + 1;

    float qreg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0};

    for (int t = t0; t <= pos; t++) {
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

/* ==========================================================================
 * V1a: split-K partial kernel. grid = (n_heads, S), block = 32 lanes.
 * Each block reduces its slot subrange into an online-softmax partial
 * (raw, unnormalized acc). Empty subranges (when nslots < S) write
 * m = -inf, l = 0 so the combiner can skip them.
 * ======================================================================== */

__global__ void k_flash_split_partial(const float *__restrict__ q,
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

/* V1b: combine S partials per head via online-softmax merge. */
__global__ void k_flash_combine(const float *__restrict__ p_acc,
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
        const float alpha = expf(m - m_new); /* 0 when m == -inf: safe */
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

static void run_splitk(const float *d_q, const float *d_K, const float *d_V,
                       float *d_out, const int *d_pos, float *d_pacc,
                       float *d_pm, float *d_pl, int H, int KV, int HD,
                       float scale, int window, int S, int pos) {
    dim3 grid(H, S);
    k_flash_split_partial<<<grid, 32>>>(d_q, d_K, d_V, d_pacc, d_pm, d_pl,
                                        d_pos, H, KV, HD, scale, window, S);
    k_flash_combine<<<H, 32>>>(d_pacc, d_pm, d_pl, d_out, H, HD, S);
    CHECK(cudaGetLastError());
}

/* ============================ CPU reference ============================== */

static void ref_attention(const double *q, const double *Kc, const double *Vc,
                          int n_heads, int n_kv_heads, int head_dim,
                          double scale, int pos, int window, double *out) {
    for (int h = 0; h < n_heads; h++) {
        const int kvh = h / (n_heads / n_kv_heads);
        int t0 = 0;
        if (window > 0 && pos >= window) t0 = pos - window + 1;
        double maxs = -INFINITY;
        for (int t = t0; t <= pos; t++) {
            double s = 0.0;
            for (int d = 0; d < head_dim; d++)
                s += q[(size_t)h * head_dim + d] * Kc[((size_t)t * n_kv_heads + kvh) * head_dim + d];
            s *= scale;
            if (s > maxs) maxs = s;
        }
        double l = 0.0;
        for (int d = 0; d < head_dim; d++) out[(size_t)h * head_dim + d] = 0.0;
        for (int t = t0; t <= pos; t++) {
            double s = 0.0;
            for (int d = 0; d < head_dim; d++)
                s += q[(size_t)h * head_dim + d] * Kc[((size_t)t * n_kv_heads + kvh) * head_dim + d];
            s = exp(s * scale - maxs);
            l += s;
            for (int d = 0; d < head_dim; d++)
                out[(size_t)h * head_dim + d] += s * Vc[((size_t)t * n_kv_heads + kvh) * head_dim + d];
        }
        for (int d = 0; d < head_dim; d++) out[(size_t)h * head_dim + d] /= l;
    }
}

static unsigned long long rng_state = 0x9e3779b97f4a7c15ULL;
static double frand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return ((double)(rng_state >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

/* ============================ shared state =============================== */

struct Env {
    int H = 8, KV = 1, HD = 512;
    int MAXCTX = 32768;
    int MAXS = 64;
    float *d_q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_out = nullptr;
    int *d_pos = nullptr;
    float *d_pacc = nullptr, *d_pm = nullptr, *d_pl = nullptr;
    /* host mirrors */
    double *q = nullptr, *Kc = nullptr, *Vc = nullptr;

    size_t cvn() const { return (size_t)MAXCTX * KV * HD; }
    size_t qn() const { return (size_t)H * HD; }
};

static void env_init(Env &e) {
    const size_t qn = e.qn(), cvn = e.cvn();
    CHECK(cudaMalloc(&e.d_q, qn * sizeof(float)));
    CHECK(cudaMalloc(&e.d_K, cvn * sizeof(float))); /* 64 MB @ 32k */
    CHECK(cudaMalloc(&e.d_V, cvn * sizeof(float))); /* 64 MB @ 32k */
    CHECK(cudaMalloc(&e.d_out, qn * sizeof(float)));
    CHECK(cudaMalloc(&e.d_pos, sizeof(int)));
    CHECK(cudaMalloc(&e.d_pacc, (size_t)e.MAXS * e.H * e.HD * sizeof(float)));
    CHECK(cudaMalloc(&e.d_pm, (size_t)e.MAXS * e.H * sizeof(float)));
    CHECK(cudaMalloc(&e.d_pl, (size_t)e.MAXS * e.H * sizeof(float)));

    e.q = new double[qn];
    e.Kc = new double[cvn];
    e.Vc = new double[cvn];
    for (size_t i = 0; i < qn; i++) e.q[i] = frand();
    for (size_t i = 0; i < cvn; i++) { e.Kc[i] = frand(); e.Vc[i] = frand(); }

    float *hq = new float[qn], *hK = new float[cvn], *hV = new float[cvn];
    for (size_t i = 0; i < qn; i++) hq[i] = (float)e.q[i];
    for (size_t i = 0; i < cvn; i++) { hK[i] = (float)e.Kc[i]; hV[i] = (float)e.Vc[i]; }
    CHECK(cudaMemcpy(e.d_q, hq, qn * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(e.d_K, hK, cvn * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(e.d_V, hV, cvn * 4, cudaMemcpyHostToDevice));
    delete[] hq; delete[] hK; delete[] hV;
}

static void env_free(Env &e) {
    cudaFree(e.d_q); cudaFree(e.d_K); cudaFree(e.d_V); cudaFree(e.d_out);
    cudaFree(e.d_pos); cudaFree(e.d_pacc); cudaFree(e.d_pm); cudaFree(e.d_pl);
    delete[] e.q; delete[] e.Kc; delete[] e.Vc;
}

/* ============================ correctness ================================ */

struct ErrStat { double max_abs = 0.0; double max_rel = 0.0; };

/* Gate: scale-relative error |got-ref|/max|ref| < 1e-5. Per-element rel is
 * also reported but not gated: attention outputs average many +/-V terms and
 * individual true values cross zero, so per-element rel explodes on
 * near-zero entries purely from fp32-vs-double input rounding (~1e-8 abs).
 * Scale-relative error is the honest long-context accuracy metric here. */
static ErrStat compare(const float *got, const double *ref, size_t n) {
    ErrStat st;
    double refmax = 0.0;
    for (size_t i = 0; i < n; i++) refmax = std::max(refmax, fabs(ref[i]));
    const double floorv = std::max(refmax, 1e-30);
    for (size_t i = 0; i < n; i++) {
        const double dv = fabs((double)got[i] - ref[i]);
        if (!(dv == dv) || dv > INFINITY) { st.max_rel = INFINITY; return st; }
        st.max_abs = std::max(st.max_abs, dv);
        st.max_rel = std::max(st.max_rel, dv / floorv);
    }
    return st;
}

static void set_pos(Env &e, int pos) { CHECK(cudaMemcpy(e.d_pos, &pos, 4, cudaMemcpyHostToDevice)); }

static int correctness(void) {
    Env e;
    env_init(e);
    const float scale = 1.0f / sqrtf((float)e.HD);
    const size_t qn = e.qn();

    struct Case { int pos, window; int S; const char *name; };
    const Case cases[] = {
        {15,      0,  4, "ctx=16  full  S=4"},
        {255,     0, 16, "ctx=256 full  S=16"},
        {127,     0, 64, "ctx=128 full  S=64 (>ctx splits partly empty)"},
        {2047,    0, 64, "ctx=2048 full S=64"},
        {999, 100, 16, "swa win=100 pos=999 S=16"},
    };
    int fails = 0;
    printf("== correctness (vs CPU double, gated rel < 1e-5) ==\n");

    double *ref = new double[qn];
    float *h_out = new float[qn];

    for (const Case &c : cases) {
        set_pos(e, c.pos);
        const float ws = (float)c.S * e.H;
        /* poison workspace so we prove every cell written */
        CHECK(cudaMemset(e.d_pacc, 0xFF, ws * e.HD * sizeof(float)));
        CHECK(cudaMemset(e.d_pm, 0xFF, ws * sizeof(float)));
        CHECK(cudaMemset(e.d_pl, 0xFF, ws * sizeof(float)));

        run_splitk(e.d_q, e.d_K, e.d_V, e.d_out, e.d_pos, e.d_pacc, e.d_pm,
                   e.d_pl, e.H, e.KV, e.HD, scale, c.window, c.S, c.pos);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, e.d_out, qn * 4, cudaMemcpyDeviceToHost));

        ref_attention(e.q, e.Kc, e.Vc, e.H, e.KV, e.HD, (double)scale,
                      c.pos, c.window, ref);
        ErrStat st = compare(h_out, ref, qn);
        const bool ok = st.max_rel < 1e-5;
        fails += !ok;
        printf("  %-42s %s  max_abs=%.2e scale_rel=%.2e\n", c.name,
               ok ? "PASS" : "FAIL", st.max_abs, st.max_rel);
    }

    /* ---- order independence: S=4 vs 16 vs 64 and vs V0 ---- */
    printf("== numerical order independence (S=4 vs S=16 vs S=64 vs V0, ctx=4096) ==\n");
    const int P = 4095;
    set_pos(e, P);
    const size_t wsz = (size_t)e.MAXS * e.H * e.HD;
    float *outs[4];
    for (int k = 0; k < 3; k++) {
        const int S[] = {4, 16, 64};
        outs[k] = new float[qn];
        run_splitk(e.d_q, e.d_K, e.d_V, e.d_out, e.d_pos, e.d_pacc, e.d_pm,
                   e.d_pl, e.H, e.KV, e.HD, scale, 0, S[k], P);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(outs[k], e.d_out, qn * 4, cudaMemcpyDeviceToHost));
    }
    outs[3] = new float[qn];
    k_flash_gqa<<<e.H, 32>>>(e.d_q, e.d_K, e.d_V, e.d_out, e.d_pos, e.H, e.KV,
                             e.HD, e.MAXCTX, scale, 0);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(outs[3], e.d_out, qn * 4, cudaMemcpyDeviceToHost));
    (void)wsz;

    const char *nm[] = {"S=4  vs S=16", "S=4  vs S=64", "S=4  vs V0"};
    const int pa[] = {1, 2, 3};
    for (int k = 0; k < 3; k++) {
        double mx = 0.0;
        for (size_t i = 0; i < qn; i++)
            mx = std::max(mx, (double)fabs(outs[0][i] - outs[pa[k]][i]));
        const bool ok = mx < 1e-5;
        fails += !ok;
        printf("  %-42s %s  max_abs=%.2e\n", nm[k], ok ? "PASS" : "FAIL", mx);
    }
    for (int k = 0; k < 4; k++) delete[] outs[k];
    delete[] ref; delete[] h_out;
    env_free(e);
    return fails;
}

/* ============================ benchmark ================================== */

static float time_kernel_us(void (*fn)(void *), void *arg, int iters) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    /* warmup */
    for (int i = 0; i < 3; i++) fn(arg);
    CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; i++) fn(arg);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        best = std::min(best, ms * 1000.0f / iters);
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best; /* us per call, best-of-3 batches */
}

struct BenchArg {
    Env *e;
    float scale;
    int S;
    bool use_v1;
    int pos;
};

static void bench_once(void *p) {
    BenchArg *b = (BenchArg *)p;
    if (!b->use_v1) {
        k_flash_gqa<<<b->e->H, 32>>>(b->e->d_q, b->e->d_K, b->e->d_V, b->e->d_out,
                                     b->e->d_pos, b->e->H, b->e->KV, b->e->HD,
                                     b->e->MAXCTX, b->scale, 0);
    } else {
        run_splitk(b->e->d_q, b->e->d_K, b->e->d_V, b->e->d_out, b->e->d_pos,
                   b->e->d_pacc, b->e->d_pm, b->e->d_pl, b->e->H, b->e->KV,
                   b->e->HD, b->scale, 0, b->S, b->pos);
    }
}

static void occupancy_report(void) {
    int dev, sms;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int v0_blocks = 0, vp_blocks = 0, vc_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v0_blocks, k_flash_gqa, 32, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&vp_blocks, k_flash_split_partial, 32, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&vc_blocks, k_flash_combine, 32, 0);
    printf("\n== occupancy ==\n");
    printf("  SMs: %d\n", sms);
    printf("  V0 warp/head : grid=8 blocks  (%d block/SM cap) -> 8 blocks total, "
           "%d%% of SMs get >=1 block at ctx-independent grid\n",
           v0_blocks, 100 * std::min(8, sms) / sms);
    printf("  V1 partial   : grid=8*S blocks (%d block/SM cap)\n", vp_blocks);
    for (int S : {4, 16, 64})
        printf("    S=%2d -> %3d blocks, ~%.1f blocks/SM resident-capable, "
               "grid covers %.0f%% of SMs\n",
               S, 8 * S, (double)std::min(vp_blocks * 8 * S, sms * vp_blocks) / sms,
               100.0 * std::min(8 * S, sms) / sms);
    printf("  V1 combine   : grid=8 blocks (%d block/SM cap)\n", vc_blocks);
}

static int benchmark(void) {
    Env e;
    env_init(e);
    const float scale = 1.0f / sqrtf((float)e.HD);
    const int ctxs[] = {512, 4096, 16384, 32768};
    const int Ss[] = {4, 16, 64};

    printf("\n== benchmark (best-of-batches us/call, H=8 KV=1 hd=512, pos=ctx-1) ==\n");
    printf("%8s | %10s | %11s | %11s | %11s |  best S / speedup\n",
           "ctx", "V0", "V1 S=4", "V1 S=16", "V1 S=64");

    int found_cross = -1;

    for (int ctx : ctxs) {
        set_pos(e, ctx - 1);
        BenchArg base{&e, scale, 0, false, ctx - 1};
        const float t0us = time_kernel_us(bench_once, &base, ctx <= 4096 ? 50 : 20);
        printf("%8d | %10.2f |", ctx, t0us);
        float best = 1e30f; int bestS = 0;
        for (int S : Ss) {
            BenchArg ba{&e, scale, S, true, ctx - 1};
            const float tus = time_kernel_us(bench_once, &ba, ctx <= 4096 ? 50 : 20);
            printf(" %11.2f |", tus);
            if (tus < best) { best = tus; bestS = S; }
        }
        printf("  S=%d  %.2fx\n", bestS, t0us / best);
        if (found_cross < 0 && best < t0us) found_cross = ctx;
    }

    /* finer low-ctx sweep for the crossover */
    printf("\n== crossover sweep (low ctx) ==\n");
    const int lo_ctxs[] = {128, 256, 512, 1024, 2048, 4096};
    for (int ctx : lo_ctxs) {
        set_pos(e, ctx - 1);
        BenchArg base{&e, scale, 0, false, ctx - 1};
        const float t0us = time_kernel_us(bench_once, &base, 200);
        printf("  ctx=%5d V0=%8.2fus", ctx, t0us);
        for (int S : Ss) {
            BenchArg ba{&e, scale, S, true, ctx - 1};
            const float tus = time_kernel_us(bench_once, &ba, 200);
            printf("  S=%d:%8.2fus(%s)", S, tus, tus < t0us ? "WIN" : "-");
        }
        printf("\n");
    }

    occupancy_report();

    size_t freeB = 0, totB = 0;
    cudaMemGetInfo(&freeB, &totB);
    printf("\n  device mem: %.0f MB free / %.0f MB total; proto footprint "
           "(KV @32k + workspace) ~%zu MB\n",
           freeB / 1048576.0, totB / 1048576.0,
           (size_t)((2.0 * e.cvn() * 4 + e.MAXS * e.H * (e.HD + 2) * 4) / 1048576.0));

    env_free(e);
    return 0;
}

int main(void) {
    int fails = correctness();
    fails += benchmark();
    printf(fails == 0 ? "\nALL PASS\n" : "\n%d FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
