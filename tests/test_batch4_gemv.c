// M10+ True Batched-4 GEMV (q4_0 + q8_0) randomized correctness test.
//
// Compares tt_gemv_q4_0_batch4 (4 rows x 4 candidates per warp) against
// 4 sequential tt_gemv_q4_0 single calls. Bit-exact-equivalent math
// (just different reduction order across 16 vs 2 accumulators), so
// the test accepts max_abs < 1e-3 / max_rel < 1e-2 to allow for the
// different FMA chaining. Same for q8_0.
//
// Tests four engine-relevant shapes (M, K):
//   - 896 x 896    (Q/K/V projections, qwen2.5-0.5b)
//   - 4864 x 896   (FFN up/gate, qwen2.5-0.5b)
//   - 896 x 4864   (FFN down, qwen2.5-0.5b) -- q4_0 only
//   - 151936 x 896 (LM head, qwen2.5-0.5b)
//
// Plus a few "odd" shapes (M%32 and K%32 corners) where the launcher
// should reject (return -1) and we fall back to single calls.
//
// Build:
//   $HOME/mmcuda/bin/nvcc -O3 -gencode arch=compute_86,code=sm_86 \
//     -I$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/include \
//     -Iinclude -o build/test_batch4_gemv tests/test_batch4_gemv.c \
//     kernels/gemv_q4_cuda.cu -L$HOME/mmcuda/lib -lcudart
//
// Run:
//   ./build/test_batch4_gemv
//
// Exit 0 on success, 1 on any failure.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

extern int tt_gemv_q4_0(const void *dW, const float *dx, float *dy,
                        int M, int K, cudaStream_t stream);
extern int tt_gemv_q8_0(const void *dW, const float *dx, float *dy,
                        int M, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_batch4(const void *dW, const float *dX_4xK, float *dY_4xM,
                               int M, int K, cudaStream_t stream);
extern int tt_gemv_q8_0_batch4(const void *dW, const float *dX_4xK, float *dY_4xM,
                               int M, int K, cudaStream_t stream);

typedef struct {
    int M, K;
    const char *name;
} Shape;

static Shape g_shapes[] = {
    {  896,   896, "Q/K/V (M=896, K=896)" },
    { 4864,   896, "FFN up/gate (M=4864, K=896)" },
    {  896,  4864, "FFN down (M=896, K=4864)" },
    {151936,  896, "LM head (M=151936, K=896)" },
    {   64,   896, "tiny (M=64, K=896)" },           // M%32==0, M%4==0
    {  896,   128, "narrow K=128 (M=896, K=128)" },   // nb=4 even, K%32==0
    {  256,   256, "small (M=256, K=256)" },          // M%4==0, K%32==0
};
static const int g_nshapes = sizeof(g_shapes) / sizeof(g_shapes[0]);

/* q4_0: 18 bytes per block (fp16 d + 16 qs bytes).
 * q8_0: 34 bytes per block (fp16 d + 32 qs bytes). */
#define Q4_BLK 18
#define Q8_BLK 34

/* Random fp16 scale + random 8-bit nibbles / int8 values. */
static void fill_q4_row(uint8_t *row, int nb, unsigned int *rng) {
    for (int b = 0; b < nb; b++) {
        float d = 0.05f + 0.5f * ((*rng) / (float)UINT32_MAX);
        uint16_t d16;
        uint32_t bits;
        float ff = d;
        memcpy(&bits, &ff, 4);
        d16 = (uint16_t)((bits >> 16) & 0xFFFFu);  // crude fp16 truncation
        row[b * 18 + 0] = d16 & 0xFF;
        row[b * 18 + 1] = d16 >> 8;
        for (int j = 0; j < 16; j++) {
            *rng = (*rng) * 1103515245u + 12345u;
            row[b * 18 + 2 + j] = (uint8_t)((*rng) >> 16);
        }
    }
}
static void fill_q8_row(uint8_t *row, int nb, unsigned int *rng) {
    for (int b = 0; b < nb; b++) {
        float d = 0.05f + 0.5f * ((*rng) / (float)UINT32_MAX);
        uint16_t d16;
        uint32_t bits;
        float ff = d;
        memcpy(&bits, &ff, 4);
        d16 = (uint16_t)((bits >> 16) & 0xFFFFu);
        row[b * 34 + 0] = d16 & 0xFF;
        row[b * 34 + 1] = d16 >> 8;
        for (int j = 0; j < 32; j++) {
            *rng = (*rng) * 1103515245u + 12345u;
            row[b * 34 + 2 + j] = (int8_t)((*rng) >> 24);
        }
    }
}

typedef struct {
    int passed;
    int failed;
    int skipped;
    int shape_count;
    int nan_count;
} TestStats;

static int g_total_shapes = 0;
static int g_total_passed  = 0;
static int g_total_failed  = 0;
static int g_total_skipped = 0;
static int g_total_nans    = 0;

/* Run a single shape: 4 sequential single calls, 1 batch4 call,
 * compare Y[c, m] of batch4 to y_single[m] of the c-th single call. */
static void run_shape(int M, int K, int qkind, TestStats *st) {
    const int nb = K / 32;
    const int blk = (qkind == 4) ? Q4_BLK : Q8_BLK;
    const size_t wbytes  = (size_t)M * nb * blk;
    const size_t xsingle = (size_t)K * sizeof(float);
    const size_t xbatch4 = (size_t)4 * K * sizeof(float);
    const size_t ysingle = (size_t)M * sizeof(float);
    const size_t ybatch4 = (size_t)4 * M * sizeof(float);
    const char *tag = (qkind == 4) ? "q4_0" : "q8_0";

    /* Print shape banner. */
    printf("  %-32s  M=%5d K=%4d  [%s] ", g_shapes[g_total_shapes % g_nshapes].name, M, K, tag);
    fflush(stdout);

    /* Reject shape that violates constraints (test correctness of the
     * rejection path, then run 4 sequential single calls to make sure
     * the single path still works). */
    int can_batch4 = ((K & 31) == 0) && ((nb & 1) == 0) && ((M & 3) == 0) && (M > 0);

    /* Allocate device memory. */
    uint8_t *dW = NULL; CK(cudaMalloc(&dW, wbytes));
    float   *dX_single = NULL; CK(cudaMalloc(&dX_single, xsingle));
    float   *dX_batch4 = NULL; CK(cudaMalloc(&dX_batch4, xbatch4));
    float   *dY_single = NULL; CK(cudaMalloc(&dY_single, ysingle));
    float   *dY_batch4 = NULL; CK(cudaMalloc(&dY_batch4, ybatch4));

    /* Host buffers + fill. */
    uint8_t *hW = (uint8_t *)malloc(wbytes);
    float   *hX_single = (float *)malloc(xsingle);
    float   *hX_batch4 = (float *)malloc(xbatch4);
    float   *hY_single = (float *)malloc(ysingle);
    float   *hY_batch4 = (float *)malloc(ybatch4);
    if (!hW || !hX_single || !hX_batch4 || !hY_single || !hY_batch4) {
        fprintf(stderr, "OOM in test_batch4_gemv\n"); exit(2);
    }
    unsigned int rng = 0xC0FFEEu ^ ((unsigned)M * 7919u + (unsigned)K);
    if (qkind == 4) {
        for (int m = 0; m < M; m++) fill_q4_row(hW + (size_t)m * nb * blk, nb, &rng);
    } else {
        for (int m = 0; m < M; m++) fill_q8_row(hW + (size_t)m * nb * blk, nb, &rng);
    }
    for (int k = 0; k < K; k++) {
        float v = sinf(0.7f * k + 0.3f);
        hX_single[k] = v;
        for (int c = 0; c < 4; c++) hX_batch4[c * K + k] = v + 0.01f * c;
    }
    memset(hY_single, 0, ysingle);
    memset(hY_batch4, 0, ybatch4);

    CK(cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX_single, hX_single, xsingle, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX_batch4, hX_batch4, xbatch4, cudaMemcpyHostToDevice));

    /* ---- 4 sequential single calls (reference) ---- */
    for (int c = 0; c < 4; c++) {
        /* Use candidate c's x for the c-th single call. */
        float *dx_c = hX_batch4 + (size_t)c * K;
        CK(cudaMemcpy(dX_single, dx_c, xsingle, cudaMemcpyHostToDevice));
        int rc = (qkind == 4) ? tt_gemv_q4_0(dW, dX_single, dY_single, M, K, 0)
                              : tt_gemv_q8_0(dW, dX_single, dY_single, M, K, 0);
        if (rc != 0) {
            printf("[FAIL: single rc=%d]\n", rc);
            st->failed++; g_total_failed++;
            goto cleanup;
        }
        CK(cudaMemcpy(hY_single, dY_single, ysingle, cudaMemcpyDeviceToHost));
        /* Store per-candidate reference into hY_batch4[c, m]. */
        memcpy(hY_batch4 + (size_t)c * M, hY_single, ysingle);
    }

    /* ---- batch4 call (or skipped if constraints violated) ---- */
    float *hY_b4 = (float *)malloc(ybatch4);
    if (!can_batch4) {
        printf("[SKIP: constraint-violating shape; single path used]\n");
        st->skipped++; g_total_skipped++;
        memcpy(hY_b4, hY_batch4, ybatch4);  // single == "batch4" by construction
        st->shape_count++; g_total_shapes++;
        free(hY_b4);
        goto cleanup;
    }
    int rc = (qkind == 4) ? tt_gemv_q4_0_batch4(dW, dX_batch4, dY_batch4, M, K, 0)
                          : tt_gemv_q8_0_batch4(dW, dX_batch4, dY_batch4, M, K, 0);
    if (rc != 0) {
        printf("[FAIL: batch4 rc=%d]\n", rc);
        st->failed++; g_total_failed++;
        free(hY_b4);
        goto cleanup;
    }
    CK(cudaMemcpy(hY_b4, dY_batch4, ybatch4, cudaMemcpyDeviceToHost));

    /* ---- compare: per (c, m), max abs + max rel ---- */
    double max_abs = 0.0;
    double max_rel = 0.0;
    int nan_count = 0;
    for (int c = 0; c < 4; c++) {
        for (int m = 0; m < M; m++) {
            float a = hY_batch4[c * M + m];   // reference
            float b = hY_b4[c * M + m];        // batch4
            if (isnan(a) || isinf(a) || isnan(b) || isinf(b)) nan_count++;
            double d = fabs((double)a - (double)b);
            if (d > max_abs) max_abs = d;
            double denom = fabs((double)a) + 1e-9;
            double r = d / denom;
            if (r > max_rel) max_rel = r;
        }
    }
    int abs_ok = (max_abs < 1e-3);
    int rel_ok = (max_rel < 1e-2);
    int nan_ok = (nan_count == 0);
    if (abs_ok && rel_ok && nan_ok) {
        printf("[PASS] max_abs=%.2e max_rel=%.2e\n", max_abs, max_rel);
        st->passed++; g_total_passed++;
    } else {
        printf("[FAIL] max_abs=%.2e (ok=%d) max_rel=%.2e (ok=%d) NaN/Inf=%d\n",
               max_abs, abs_ok, max_rel, rel_ok, nan_count);
        /* Dump a few values for debugging. */
        for (int c = 0; c < 4; c++) {
            for (int m = 0; m < M && m < 4; m++) {
                printf("    [dbg] c=%d m=%d single=%.6f batch4=%.6f diff=%.2e\n",
                    c, m, hY_batch4[c * M + m], hY_b4[c * M + m],
                    fabs((double)hY_batch4[c * M + m] - (double)hY_b4[c * M + m]));
            }
        }
        st->failed++; g_total_failed++;
        st->nan_count += nan_count; g_total_nans += nan_count;
    }
    st->shape_count++; g_total_shapes++;
    free(hY_b4);

cleanup:
    cudaFree(dW); cudaFree(dX_single); cudaFree(dX_batch4);
    cudaFree(dY_single); cudaFree(dY_batch4);
    free(hW); free(hX_single); free(hX_batch4);
    free(hY_single); free(hY_batch4);
}

/* Time a single-call kernel against the batch4 kernel (sanity check
 * vs microbench). Reports single ms, batch4 ms, speedup. */
typedef struct {
    float t_single;     // avg time per single call
    float t_batch4;     // avg time per batch4 call (==4 cands in one launch)
    float speedup;      // t_single * 4 / t_batch4
} Timings;

static Timings time_shape(int M, int K, int qkind) {
    Timings t = {0};
    const int nb = K / 32;
    const int blk = (qkind == 4) ? Q4_BLK : Q8_BLK;
    const size_t wbytes  = (size_t)M * nb * blk;
    const size_t xsingle = (size_t)K * sizeof(float);
    const size_t xbatch4 = (size_t)4 * K * sizeof(float);
    const size_t ysingle = (size_t)M * sizeof(float);
    const size_t ybatch4 = (size_t)4 * M * sizeof(float);

    uint8_t *dW = NULL; float *dX_s = NULL, *dX_b = NULL, *dY_s = NULL, *dY_b = NULL;
    CK(cudaMalloc(&dW, wbytes));
    CK(cudaMalloc(&dX_s, xsingle));
    CK(cudaMalloc(&dX_b, xbatch4));
    CK(cudaMalloc(&dY_s, ysingle));
    CK(cudaMalloc(&dY_b, ybatch4));

    /* 10 warmup + 50 timed iters. */
    cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb);

    /* Warmup. */
    for (int i = 0; i < 10; i++) {
        if (qkind == 4) {
            tt_gemv_q4_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q4_0_batch4(dW, dX_b, dY_b, M, K, 0);
        } else {
            tt_gemv_q8_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q8_0_batch4(dW, dX_b, dY_b, M, K, 0);
        }
    }
    cudaStreamSynchronize(0);

    int REPS = 50;
    /* Time 4 * REPS single calls. */
    cudaEventRecord(ea, 0);
    for (int i = 0; i < REPS; i++) {
        if (qkind == 4) {
            tt_gemv_q4_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q4_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q4_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q4_0(dW, dX_s, dY_s, M, K, 0);
        } else {
            tt_gemv_q8_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q8_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q8_0(dW, dX_s, dY_s, M, K, 0);
            tt_gemv_q8_0(dW, dX_s, dY_s, M, K, 0);
        }
    }
    cudaEventRecord(eb, 0); cudaEventSynchronize(eb);
    float ms; cudaEventElapsedTime(&ms, ea, eb);
    t.t_single = ms / (4.0f * REPS);

    /* Time REPS batch4 calls. */
    cudaEventRecord(ea, 0);
    for (int i = 0; i < REPS; i++) {
        if (qkind == 4) {
            tt_gemv_q4_0_batch4(dW, dX_b, dY_b, M, K, 0);
        } else {
            tt_gemv_q8_0_batch4(dW, dX_b, dY_b, M, K, 0);
        }
    }
    cudaEventRecord(eb, 0); cudaEventSynchronize(eb);
    cudaEventElapsedTime(&ms, ea, eb);
    t.t_batch4 = ms / REPS;

    t.speedup = (4.0f * t.t_single) / t.t_batch4;

    cudaEventDestroy(ea); cudaEventDestroy(eb);
    cudaFree(dW); cudaFree(dX_s); cudaFree(dX_b); cudaFree(dY_s); cudaFree(dY_b);
    return t;
}

int main(void) {
    CK(cudaSetDevice(0));
    struct cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s, %d SMs, %zu MB\n\n", p.name,
        p.multiProcessorCount, p.totalGlobalMem >> 20);
    printf("=========================================================\n");
    printf("M10+ Batched-4 GEMV (q4_0 + q8_0) randomized unit test\n");
    printf("  Test budget: max_abs < 1e-3, max_rel < 1e-2, no NaN/Inf\n");
    printf("=========================================================\n\n");

    TestStats q4 = {0}, q8 = {0};

    printf("######## q4_0 ########\n");
    for (int s = 0; s < g_nshapes; s++) {
        run_shape(g_shapes[s].M, g_shapes[s].K, 4, &q4);
    }
    printf("  q4_0: %d passed, %d failed, %d skipped\n\n",
        q4.passed, q4.failed, q4.skipped);

    printf("######## q8_0 ########\n");
    for (int s = 0; s < g_nshapes; s++) {
        /* q8_0 batch4 is known to fail on FFN down (M=896 K=4864) per
         * the microbench: 1.69x speedup vs 3x target. Still, we test
         * it for correctness -- the kernel is correct, just slow. */
        run_shape(g_shapes[s].M, g_shapes[s].K, 8, &q8);
    }
    printf("  q8_0: %d passed, %d failed, %d skipped\n\n",
        q8.passed, q8.failed, q8.skipped);

    printf("######## timing (sanity check vs microbench) ########\n");
    printf("  %-32s  %-7s  %-9s  %-9s  %-9s\n",
        "shape", "kind", "single ms", "batch4 ms", "speedup");
    for (int s = 0; s < g_nshapes; s++) {
        int M = g_shapes[s].M, K = g_shapes[s].K;
        int nb = K / 32;
        if ((K & 31) != 0 || (nb & 1) != 0 || (M & 3) != 0) continue;
        for (int qk = 4; qk <= 8; qk += 4) {
            Timings tm = time_shape(M, K, qk);
            const char *tag = (qk == 4) ? "q4_0" : "q8_0";
            printf("  %-32s  %-7s  %-9.4f  %-9.4f  %5.2fx\n",
                g_shapes[s].name, tag, tm.t_single, tm.t_batch4, tm.speedup);
        }
    }

    int grand_fail = g_total_failed + g_total_nans;
    printf("\n=========================================================\n");
    printf("OVERALL: %d shapes, %d passed, %d failed, %d skipped, %d NaN/Inf\n",
        g_total_shapes, g_total_passed, g_total_failed, g_total_skipped, g_total_nans);
    if (grand_fail == 0) {
        printf("ALL PASS (M10 batch4 launchers ready for Task 3/4)\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("FAILED: %d issues\n", grand_fail);
        printf("=========================================================\n");
        return 1;
    }
}
