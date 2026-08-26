/* tests/test_rope_ff.cu
 *
 * Standalone unit test for the RoPE freq-factor variant (k_rope_ff) and the
 * plain NEOX k_rope, against a double-precision ggml rope_yarn reference.
 *
 * Manual build:
 *   $HOME/mmcuda/bin/nvcc -arch=sm_86 -Iinclude tests/test_rope_ff.cu -o /tmp/test_rope_ff
 *   LD_LIBRARY_PATH=$HOME/mmcuda/lib /tmp/test_rope_ff
 *
 * NOTE: kernel bodies below are copied VERBATIM from kernels/qwen2_cuda.cu
 * (branch m6-correctness). They cannot be #include'd because that file drags
 * in engine/loader headers. If kernels change, re-copy here.
 *
 * Reference semantics (oracle llama.cpp ggml/src/ggml-cpu/ops.cpp:5843,
 * ggml rope_yarn): theta_i = pos * base^(-2i/head_dim) / ff[i]; NEOX layout
 * rotates halves x[i], x[i+head_dim/2].
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- verbatim copies from kernels/qwen2_cuda.cu ---- */

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

/* ---- end verbatim copies ---- */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err_), __FILE__, __LINE__);          \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* Deterministic LCG fill, matches test conventions (seed 42). */
static unsigned long long lcg_state;
static float next_rand(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((float)((lcg_state >> 40) & 0xFFFFFF)) / (float)0xFFFFFF - 0.5f;
}

static void fill_input(float *q, size_t n_elems) {
    lcg_state = 42ULL;
    for (size_t j = 0; j < n_elems; j++) q[j] = next_rand();
}

static double max_abs_diff(const float *a, const float *b, long n, long *worst) {
    double m = 0.0; long wj = -1;
    for (long j = 0; j < n; j++) {
        double d = fabs((double)a[j] - (double)b[j]);
        if (d > m) { m = d; wj = j; }
    }
    if (worst) *worst = wj;
    return m;
}

/* Run one GPU case: launch config copied from qwen2_cuda.cu forward_layers
 * rope section: grid=((hd/2+63)/64, n_heads), block=(64,1,1).
 * Returns max abs diff between GPU result and CPU reference. */
static double run_case(const char *name, int n_heads, int head_dim, int pos,
                       float base, const float *h_ff /* NULL or host ff */,
                       const double *ref_ff, bool expect_pass, double tol,
                       bool *passed) {
    const long total = (long)n_heads * head_dim;
    const int half = head_dim / 2;

    float *h_in = (float *)malloc(total * sizeof(float));
    float *h_out = (float *)malloc(total * sizeof(float));
    fill_input(h_in, total);

    float *d_q, *d_ff = NULL;
    int h_pos = pos, *d_pos;
    CUDA_CHECK(cudaMalloc(&d_q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));
    if (h_ff) { CUDA_CHECK(cudaMalloc(&d_ff, half * sizeof(float))); }
    CUDA_CHECK(cudaMemcpy(d_q, h_in, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, &h_pos, sizeof(int), cudaMemcpyHostToDevice));
    if (h_ff) CUDA_CHECK(cudaMemcpy(d_ff, h_ff, half * sizeof(float), cudaMemcpyHostToDevice));

    dim3 g((head_dim / 2 + 63) / 64, n_heads, 1);
    dim3 b(64, 1, 1);
    if (h_ff)
        k_rope_ff<<<g, b>>>(d_q, n_heads, head_dim, d_pos, base, d_ff);
    else
        k_rope<<<g, b>>>(d_q, n_heads, head_dim, d_pos, base);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_q, total * sizeof(float), cudaMemcpyDeviceToHost));

    /* CPU reference */
    float *h_ref = (float *)malloc(total * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        const float *src = h_in + (long)h * head_dim;
        float *dst = h_ref + (long)h * head_dim;
        for (int i = 0; i < half; i++) {
            const double freq = pow((double)base, -2.0 * (double)i / (double)head_dim);
            const double div = ref_ff ? ref_ff[i] : 1.0;
            const double theta = (double)pos * freq / div;
            const double c = cos(theta), s = sin(theta);
            const double v0 = src[i], v1 = src[i + half];
            dst[i] = (float)(v0 * c - v1 * s);
            dst[i + half] = (float)(v0 * s + v1 * c);
        }
    }

    long worst = -1;
    double md = max_abs_diff(h_out, h_ref, total, &worst);
    bool ok = md < tol;
    printf("%-42s pos=%-3d hd=%-4d base=%g  maxdiff=%.3e  [%s]\n",
           name, pos, head_dim, (double)base, md, ok ? "PASS" : "FAIL");
    if (!ok && worst >= 0) {
        long hh = worst / head_dim, jj = worst % head_dim;
        int pair = (jj < half) ? jj : jj - half;
        printf("    worst elem: head=%ld idx=%ld (pair %d, %s half)\n"
               "    gpu=%.7f ref=%.7f\n",
               hh, jj, pair, jj < half ? "first" : "second",
               (double)h_out[worst], (double)h_ref[worst]);
    }
    *passed = ok;

    free(h_in); free(h_out); free(h_ref);
    cudaFree(d_q); cudaFree(d_pos); if (d_ff) cudaFree(d_ff);
    return md;
}

int main(void) {
    int failures = 0;
    bool ok;
    double md;

    printf("== test_rope_ff: ggml-semantics check for k_rope / k_rope_ff ==\n");

    /* Case 1: gemma4-E2B full-attn layer shape. hd=512, base=1e6, pos=7,
     * ff[256]: 1.0 for i<32 else 1e30 (partial rope). */
    const int NH = 16, HD = 512, HALF = HD / 2;
    float *ff_h = (float *)malloc(HALF * sizeof(float));
    double *ff_d = (double *)malloc(HALF * sizeof(double));
    for (int i = 0; i < HALF; i++) {
        ff_h[i] = (i < 32) ? 1.0f : 1e30f;
        ff_d[i] = (i < 32) ? 1.0 : 1e30;
    }
    md = run_case("case1 ff partial-rope (hd=512 b=1e6)", NH, HD, 7, 1e6f,
                  ff_h, ff_d, true, 1e-3, &ok);
    if (!ok) failures++;

    /* Case 5: hd=256 swa layer, base=1e4, k_rope NULL path must equal
     * k_rope_ff fed all-ones factors. Compare GPU outputs against each other
     * AND both against CPU reference. */
    {
        const int nh2 = 16, hd2 = 256, half2 = hd2 / 2;
        const long total = (long)nh2 * hd2;

        float *h_in = (float *)malloc(total * sizeof(float));
        fill_input(h_in, total);

        float *d_a, *d_b2;
        int hpos = 7, *d_pos;
        CUDA_CHECK(cudaMalloc(&d_a, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b2, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_a, h_in, total * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, h_in, total * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos, &hpos, sizeof(int), cudaMemcpyHostToDevice));

        dim3 g((hd2 / 2 + 63) / 64, nh2, 1);
        dim3 b(64, 1, 1);
        k_rope<<<g, b>>>(d_a, nh2, hd2, d_pos, 1e4f);
        CUDA_CHECK(cudaGetLastError());
        k_rope_ff<<<g, b>>>(d_b2, nh2, hd2, d_pos, 1e4f, NULL); /* NULL path */
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float *h_a = (float *)malloc(total * sizeof(float));
        float *h_bb = (float *)malloc(total * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_a, d_a, total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bb, d_b2, total * sizeof(float), cudaMemcpyDeviceToHost));

        long w1 = -1, w2 = -1;
        double md_eq = max_abs_diff(h_a, h_bb, total, &w1);

        /* CPU reference for this case */
        float *h_ref = (float *)malloc(total * sizeof(float));
        for (int h = 0; h < nh2; h++) {
            const float *src = h_in + (long)h * hd2;
            float *dst = h_ref + (long)h * hd2;
            for (int i = 0; i < half2; i++) {
                const double theta = (double)hpos *
                    pow(1e4, -2.0 * (double)i / (double)hd2);
                const double c = cos(theta), s = sin(theta);
                dst[i] = (float)((double)src[i] * c - (double)src[i + half2] * s);
                dst[i + half2] = (float)((double)src[i] * s + (double)src[i + half2] * c);
            }
        }
        double md_kr = max_abs_diff(h_a, h_ref, total, &w2);
        bool eq_ok = md_eq == 0.0;   /* bit-exact expected: identical fp math */
        bool kr_ok = md_kr < 1e-3;
        printf("%-42s k_rope vs k_rope_ff(ones): %.3e [%s]\n",
               "case5 swa equivalence (hd=256)", md_eq, eq_ok ? "PASS" : "FAIL");
        printf("%-42s k_rope vs cpu ref:         %.3e [%s]\n",
               "", md_kr, kr_ok ? "PASS" : "FAIL");
        if (!eq_ok || !kr_ok) failures++;

        free(h_in); free(h_a); free(h_bb); free(h_ref);
        cudaFree(d_a); cudaFree(d_b2); cudaFree(d_pos);
    }

    /* Edge cases */
    {
        /* pos=0 identity regardless of ff */
        md = run_case("edge pos=0 identity w/ big ff", NH, HD, 0, 1e6f,
                      ff_h, ff_d, true, 1e-6, &ok);
        if (!ok) failures++;

        /* ff=1e30 everywhere -> ang ~ 0 -> identity rotation */
        for (int i = 0; i < HALF; i++) { ff_h[i] = 1e30f; ff_d[i] = 1e30; }
        md = run_case("edge ff=1e30 all pairs identity", NH, HD, 7, 1e6f,
                      ff_h, ff_d, true, 1e-5, &ok);
        if (!ok) failures++;
    }

    free(ff_h); free(ff_d);

    printf("\n%s (%d failure(s))\n", failures ? "OVERALL FAIL" : "OVERALL PASS",
           failures);
    return failures ? 1 : 0;
}
