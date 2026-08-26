/* tests/test_flash_multi.cu
 *
 * Standalone unit test for the GQA flash-attention decode kernel
 * k_flash_gqa (kernels/qwen2_cuda.cu), against a double-precision CPU
 * naive softmax(QK^T*scale)@V reference.
 *
 * Manual build:
 *   $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 tests/test_flash_multi.cu -o /tmp/test_flash_multi
 *   LD_LIBRARY_PATH=$HOME/mmcuda/lib /tmp/test_flash_multi
 *
 * NOTE: kernel bodies below are copied VERBATIM from kernels/qwen2_cuda.cu
 * (k_flash_gqa + its helper warp_sum). They cannot be #include'd because
 * that file drags in engine/loader headers. kernels/ must NOT be modified;
 * if the kernel changes there, re-copy here and rebuild.
 *
 * Covered:
 *  1. uniform arch   H=8 KV=1 hd=128 ctx=16 pos=15, full window
 *  2. mixed head_dim H=8 KV=1 hd=512 and hd=256, pos=5
 *  3. GQA group map  H=8 KV=2 (heads 0-3 -> kv0, 4-7 -> kv1)
 *  4. SWA window     window=4 pos=9, distant K slots poisoned so a wrong
 *                    t0 (off-by-one) would dominate the output
 *  5. stability      Q/K ~ +/-30 (pre-scale scores up to ~ +/-900*hd)
 *  6. pos=0 identity out == V[0]
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ================= verbatim copies from kernels/qwen2_cuda.cu ============ */

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

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

/* ============================ end verbatim copies ======================== */

/* CPU reference: double precision, same masking semantics as kernel. */
static void ref_attention(const double *q, const double *Kc, const double *Vc,
                          int n_heads, int n_kv_heads, int head_dim,
                          double scale, int pos, int window, double *out) {
    for (int h = 0; h < n_heads; h++) {
        const int kvh = h / (n_heads / n_kv_heads);
        int t0 = 0;
        if (window > 0 && pos >= window) t0 = pos - window + 1;
        /* masked scores */
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
static double frand(void) { /* [-1, 1] deterministic */
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return ((double)(rng_state >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

struct Cfg {
    int n_heads, n_kv_heads, head_dim, max_ctx, pos, window;
    const char *name;
};

static int run_case(const Cfg &c, bool poison_distant, double amp) {
    const int H = c.n_heads, KV = c.n_kv_heads, HD = c.head_dim;
    const size_t qn = (size_t)H * HD;
    const size_t cvn = (size_t)c.max_ctx * KV * HD;

    double *q = new double[qn]();
    double *Kc = new double[cvn]();
    double *Vc = new double[cvn]();
    double *ref = new double[qn]();

    for (size_t i = 0; i < qn; i++) q[i] = frand() * amp;
    for (size_t i = 0; i < cvn; i++) { Kc[i] = frand() * amp; Vc[i] = frand() * amp; }

    if (poison_distant) {
        /* SWA case: make every slot t < pos-window+1 carry K[t]=100*e_1 and
         * Q[*] heavy on dim 1 => huge scores IF wrongly included. In-window
         * slots get small random K on dim 1 so correct path stays tame. */
        const int t0_correct = c.pos - c.window + 1;
        for (int h = 0; h < H; h++) q[(size_t)h * HD + 1] = 3.0;
        for (int t = 0; t < c.pos; t++) {
            if (t < t0_correct) {
                for (int kvh = 0; kvh < KV; kvh++) {
                    for (int d = 0; d < HD; d++) Kc[((size_t)t * KV + kvh) * HD + d] = 0.0;
                    Kc[((size_t)t * KV + kvh) * HD + 1] = 100.0;
                }
            } else {
                for (int kvh = 0; kvh < KV; kvh++) Kc[((size_t)t * KV + kvh) * HD + 1] = frand() * 0.5;
            }
        }
    }

    /* device buffers */
    float *d_q, *d_K, *d_V, *d_out;
    int *d_pos;
    float *h_q = new float[qn], *h_K = new float[cvn], *h_V = new float[cvn], *h_out = new float[qn];
    for (size_t i = 0; i < qn; i++) h_q[i] = (float)q[i];
    for (size_t i = 0; i < cvn; i++) { h_K[i] = (float)Kc[i]; h_V[i] = (float)Vc[i]; }

    cudaMalloc(&d_q, qn * sizeof(float));
    cudaMalloc(&d_K, cvn * sizeof(float));
    cudaMalloc(&d_V, cvn * sizeof(float));
    cudaMalloc(&d_out, qn * sizeof(float));
    cudaMalloc(&d_pos, sizeof(int));
    cudaMemcpy(d_q, h_q, qn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, cvn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, cvn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, &c.pos, sizeof(int), cudaMemcpyHostToDevice);

    const float scale = 1.0f / sqrtf((float)HD);
    k_flash_gqa<<<H, 32>>>(d_q, d_K, d_V, d_out, d_pos, H, KV, HD, c.max_ctx, scale, c.window);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FAIL] %s : CUDA error %s\n", c.name, cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_out, d_out, qn * sizeof(float), cudaMemcpyDeviceToHost);

    ref_attention(q, Kc, Vc, H, KV, HD, (double)scale, c.pos, c.window, ref);

    double maxdiff = 0.0;
    int bad = 0;
    for (size_t i = 0; i < qn; i++) {
        const double dv = fabs((double)h_out[i] - ref[i]);
        if (!(dv == dv) || dv > INFINITY) bad = 1; /* NaN/inf check */
        if (dv > maxdiff) maxdiff = dv;
    }
    const char *verdict = (!bad && maxdiff < 2e-4) ? "PASS" : "FAIL";
    printf("%-28s %s  maxdiff=%.3e%s\n", c.name, verdict, maxdiff, bad ? "  (NaN/inf!)" : "");

    cudaFree(d_q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out); cudaFree(d_pos);
    delete[] h_q; delete[] h_K; delete[] h_V; delete[] h_out;
    delete[] q; delete[] Kc; delete[] Vc; delete[] ref;
    return (!bad && maxdiff < 2e-4) ? 0 : 1;
}

int main(void) {
    int fails = 0;
    printf("k_flash_gqa unit tests (CPU double-precision reference)\n");

    /* 1. uniform arch, full context */
    Cfg c1 = {8, 1, 128, 16, 15, 0, "uniform hd=128 pos=15"};
    fails += run_case(c1, false, 1.0);

    /* 2. mixed head dims (gemma4 shapes) */
    Cfg c2a = {8, 1, 512, 8, 5, 0, "mixed hd=512 pos=5"};
    Cfg c2b = {8, 1, 256, 8, 5, 0, "mixed hd=256 pos=5"};
    fails += run_case(c2a, false, 1.0);
    fails += run_case(c2b, false, 1.0);

    /* 3. GQA group mapping: heads 0-3 -> kv0, heads 4-7 -> kv1 */
    Cfg c3 = {8, 2, 128, 8, 7, 0, "gqa nkv=2 hd=128 pos=7"};
    fails += run_case(c3, false, 1.0);

    /* 4. SWA: window=4, pos=9 -> only slots 6..9; poisoned distant K */
    Cfg c4 = {8, 1, 128, 16, 9, 4, "swa win=4 pos=9"};
    fails += run_case(c4, true, 1.0);

    /* 5. numerical stability: large magnitudes */
    Cfg c5 = {8, 1, 128, 8, 7, 0, "stability amp=30"};
    fails += run_case(c5, false, 30.0);

    /* 6. pos=0 identity: handled via dedicated explicit check below */
    {
        const int H = 8, KV = 2, HD = 64, CTX = 4;
        const size_t qn = (size_t)H * HD, cvn = (size_t)CTX * KV * HD;
        float h_q[qn], h_V[cvn], h_out[qn];
        for (size_t i = 0; i < qn; i++) h_q[i] = (float)(frand() * 10.0); /* Q irrelevant at pos=0 */
        for (size_t i = 0; i < cvn; i++) h_V[i] = (float)(frand() * 2.0);
        float *d_q, *d_K, *d_V, *d_out; int *d_pos; int pos0 = 0;
        cudaMalloc(&d_q, qn * 4); cudaMalloc(&d_K, cvn * 4); cudaMalloc(&d_V, cvn * 4);
        cudaMalloc(&d_out, qn * 4); cudaMalloc(&d_pos, 4);
        cudaMemcpy(d_q, h_q, qn * 4, cudaMemcpyHostToDevice);
        /* K zeroed implicitly? no: fill with junk to prove it's ignored weight-wise */
        float *h_Kjunk = new float[cvn];
        for (size_t i = 0; i < cvn; i++) h_Kjunk[i] = (float)(frand() * 100.0);
        cudaMemcpy(d_K, h_Kjunk, cvn * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, cvn * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, &pos0, 4, cudaMemcpyHostToDevice);
        k_flash_gqa<<<H, 32>>>(d_q, d_K, d_V, d_out, d_pos, H, KV, HD, CTX, 1.0f / sqrtf((float)HD), 0);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, qn * 4, cudaMemcpyDeviceToHost);
        double maxdiff = 0.0;
        for (int h = 0; h < H; h++) {
            const int kvh = h / (H / KV);
            for (int d = 0; d < HD; d++) {
                const double dv = fabs((double)h_out[(size_t)h * HD + d] -
                                       (double)h_V[(size_t)kvh * HD + d]);
                if (dv > maxdiff) maxdiff = dv;
            }
        }
        const char *verdict = maxdiff < 1e-6 ? "PASS" : "FAIL";
        if (maxdiff >= 1e-6) fails++;
        printf("%-28s %s  maxdiff(V0 identity)=%.3e\n", "pos=0 identity", verdict, maxdiff);
        cudaFree(d_q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out); cudaFree(d_pos);
        delete[] h_Kjunk;
    }

    printf(fails == 0 ? "\nALL PASS\n" : "\n%d CASE(S) FAILED\n", fails == 0 ? 0 : fails);
    return fails == 0 ? 0 : 1;
}
