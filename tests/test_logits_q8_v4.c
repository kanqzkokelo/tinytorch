#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "qwen2_engine.h"

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

extern int tt_logits_dispatch(const void *dW, int dtype, const float *dx,
                              float *dlogits, int vocab, int K, cudaStream_t stream);

static void fill_q8_row(uint8_t *row, int nb, unsigned int *rng) {
    for (int b = 0; b < nb; b++) {
        float d = 0.001f + 0.1f * ((*rng) / (float)UINT32_MAX);
        uint32_t bits;
        memcpy(&bits, &d, 4);
        uint16_t d16 = (uint16_t)((bits >> 16) & 0xFFFFu);
        row[b * 34 + 0] = d16 & 0xFF;
        row[b * 34 + 1] = d16 >> 8;
        for (int j = 0; j < 32; j++) {
            *rng = (*rng) * 1103515245u + 12345u;
            row[b * 34 + 2 + j] = (int8_t)((*rng) >> 24);
        }
    }
}

static void fill_x(float *x, int K, unsigned int *rng) {
    for (int i = 0; i < K; i++) {
        *rng = (*rng) * 1103515245u + 12345u;
        x[i] = ((int)((*rng) % 2000) - 1000) * 0.001f;
    }
}

int main(void) {
    printf("=========================================================================\n");
    printf("  Testing tt_logits_q8_0_v4 Bit-Exact Correctness (K=896)\n");
    printf("=========================================================================\n");

    int test_vocabs[] = {896, 4864, 151936};
    int num_vocabs = sizeof(test_vocabs) / sizeof(test_vocabs[0]);
    int K = 896;
    int nb = K / 32;

    int total_failures = 0;

    for (int vi = 0; vi < num_vocabs; vi++) {
        int vocab = test_vocabs[vi];
        size_t w_bytes = (size_t)vocab * nb * 34;
        size_t x_bytes = (size_t)K * sizeof(float);
        size_t logits_bytes = (size_t)vocab * sizeof(float);

        uint8_t *h_W = (uint8_t *)malloc(w_bytes);
        float *h_x = (float *)malloc(x_bytes);
        float *h_logits_ref = (float *)malloc(logits_bytes);
        float *h_logits_v4  = (float *)malloc(logits_bytes);
        float *h_logits_disp = (float *)malloc(logits_bytes);

        unsigned int rng = 42 + vi;
        fill_q8_row(h_W, vocab * nb, &rng);
        fill_x(h_x, K, &rng);

        void *d_W;
        float *d_x, *d_logits_ref, *d_logits_v4, *d_logits_disp;
        CK(cudaMalloc((void **)&d_W, w_bytes));
        CK(cudaMalloc((void **)&d_x, x_bytes));
        CK(cudaMalloc((void **)&d_logits_ref, logits_bytes));
        CK(cudaMalloc((void **)&d_logits_v4, logits_bytes));
        CK(cudaMalloc((void **)&d_logits_disp, logits_bytes));

        CK(cudaMemcpy(d_W, h_W, w_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_logits_ref, 0, logits_bytes));
        CK(cudaMemset(d_logits_v4, 0, logits_bytes));
        CK(cudaMemset(d_logits_disp, 0, logits_bytes));

        // 1. Run baseline (tt_logits_q8_0 v1)
        int err_ref = tt_logits_q8_0(d_W, d_x, d_logits_ref, vocab, K, 0);
        // 2. Run V4 directly (tt_logits_q8_0_v4)
        int err_v4 = tt_logits_q8_0_v4(d_W, d_x, d_logits_v4, vocab, K, 0);
        // 3. Run dispatch (tt_logits_dispatch with dtype 8)
        int err_disp = tt_logits_dispatch(d_W, 8, d_x, d_logits_disp, vocab, K, 0);

        CK(cudaDeviceSynchronize());

        if (err_ref != 0 || err_v4 != 0 || err_disp != 0) {
            fprintf(stderr, "Launcher error codes: ref=%d v4=%d disp=%d\n", err_ref, err_v4, err_disp);
            total_failures++;
        }

        CK(cudaMemcpy(h_logits_ref, d_logits_ref, logits_bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_logits_v4, d_logits_v4, logits_bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_logits_disp, d_logits_disp, logits_bytes, cudaMemcpyDeviceToHost));

        float max_abs_err_v4 = 0.0f;
        float max_abs_err_disp = 0.0f;
        int nan_inf_count = 0;

        for (int i = 0; i < vocab; i++) {
            float ref = h_logits_ref[i];
            float v4  = h_logits_v4[i];
            float disp = h_logits_disp[i];

            if (isnan(ref) || isinf(ref) || isnan(v4) || isinf(v4) || isnan(disp) || isinf(disp)) {
                nan_inf_count++;
            }

            float err_v = fabsf(ref - v4);
            if (err_v > max_abs_err_v4) max_abs_err_v4 = err_v;

            float err_d = fabsf(ref - disp);
            if (err_d > max_abs_err_disp) max_abs_err_disp = err_d;
        }

        printf("vocab = %7d | max_abs_err (v4): %11.4e | max_abs_err (disp): %11.4e | nan/inf: %d | %s\n",
               vocab, max_abs_err_v4, max_abs_err_disp, nan_inf_count,
               (max_abs_err_v4 < 1e-4f && max_abs_err_disp < 1e-4f && nan_inf_count == 0) ? "PASS (BIT-EXACT)" : "FAIL");

        if (max_abs_err_v4 >= 1e-4f || max_abs_err_disp >= 1e-4f || nan_inf_count > 0) {
            total_failures++;
        }

        CK(cudaFree(d_W));
        CK(cudaFree(d_x));
        CK(cudaFree(d_logits_ref));
        CK(cudaFree(d_logits_v4));
        CK(cudaFree(d_logits_disp));
        free(h_W);
        free(h_x);
        free(h_logits_ref);
        free(h_logits_v4);
        free(h_logits_disp);
    }

    printf("=========================================================================\n");
    if (total_failures == 0) {
        printf("ALL TESTS PASSED.\n");
        return 0;
    } else {
        printf("TESTS FAILED (%d failures).\n", total_failures);
        return 1;
    }
}
