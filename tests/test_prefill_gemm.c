// Task 2 Unit Test: Batched Q4_0 Prefill GEMM vs CPU Reference and Sequential GEMV.
//
// Tests tt_gemm_q4_0_prefill across boundary values of N and model dimensions (M, K).
// Compares bit-exact / floating-point outputs against:
//   1. CPU reference matmul + Q4 dequantization
//   2. Sequential single-token tt_gemv_q4_0 calls
//
// Checks:
//   - max_abs_error < 1e-3
//   - max_rel_error < 1e-2
//   - Zero NaN / Inf

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

extern int tt_gemv_q4_0(const void *dW, const float *dx, float *dy,
                        int M, int K, cudaStream_t stream);

static inline uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00);
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00);
    if (e <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exp = (int32_t)((h >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(h & 0x03FF);
    if (exp == 0) {
        if (mant == 0) {
            uint32_t res = sign;
            float f; memcpy(&f, &res, 4); return f;
        }
        while (!(mant & 0x0400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x0400;
    } else if (exp == 31) {
        uint32_t res = sign | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &res, 4); return f;
    }
    exp = exp - 15 + 127;
    uint32_t res = sign | ((uint32_t)exp << 23) | (mant << 13);
    float f; memcpy(&f, &res, 4); return f;
}

static void generate_synthetic_q4_0(BlockQ4_0 *W, long num_blocks, unsigned int *seed) {
    for (long i = 0; i < num_blocks; i++) {
        *seed = (*seed) * 1103515245u + 12345u;
        float d = 0.01f + 0.1f * ((*seed) / (float)UINT32_MAX);
        W[i].d = float_to_half(d);
        for (int j = 0; j < 16; j++) {
            *seed = (*seed) * 1103515245u + 12345u;
            W[i].qs[j] = (uint8_t)((*seed) >> 16);
        }
    }
}

static void generate_synthetic_float(float *arr, long count, unsigned int *seed) {
    for (long i = 0; i < count; i++) {
        *seed = (*seed) * 1103515245u + 12345u;
        arr[i] = ((*seed) / (float)UINT32_MAX) * 2.0f - 1.0f;
    }
}

static void dequantize_q4_0(const BlockQ4_0 *W, float *W_float, int M, int K) {
    int nb = K / 32;
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            const BlockQ4_0 *blk = &W[m * nb + b];
            float d = half_to_float(blk->d);
            for (int j = 0; j < 16; j++) {
                int q0 = (blk->qs[j] & 0x0F) - 8;
                int q1 = (blk->qs[j] >> 4) - 8;
                W_float[m * K + b * 32 + j] = (float)q0 * d;
                W_float[m * K + b * 32 + j + 16] = (float)q1 * d;
            }
        }
    }
}

static void cpu_matmul_q4_0(const float *W_float, const float *X, float *Y_cpu, int N, int M, int K) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            double sum = 0.0;
            const float *x_row = X + (long)n * K;
            const float *w_row = W_float + (long)m * K;
            for (int k = 0; k < K; k++) {
                sum += (double)x_row[k] * (double)w_row[k];
            }
            Y_cpu[(long)n * M + m] = (float)sum;
        }
    }
}

int main(void) {
    printf("========================================================================\n");
    printf("        Unit Test: Batched Q4_0 Prefill GEMM (tt_gemm_q4_0_prefill)\n");
    printf("========================================================================\n");

    int n_vals[] = {1, 31, 32, 33, 127, 128, 256, 512};
    int num_n = sizeof(n_vals) / sizeof(n_vals[0]);

    struct { int M, K; } dims[] = {
        { 896,  896},
        {4864,  896},
        { 896, 4864},
        {4864, 4864}
    };
    int num_dims = sizeof(dims) / sizeof(dims[0]);

    int total_tests = 0;
    int total_passed = 0;
    int total_failed = 0;
    unsigned int seed = 42;

    for (int d = 0; d < num_dims; d++) {
        int M = dims[d].M;
        int K = dims[d].K;
        long num_blocks = (long)M * (K / 32);

        BlockQ4_0 *hW = (BlockQ4_0 *)malloc(num_blocks * sizeof(BlockQ4_0));
        float *hW_float = (float *)malloc((long)M * K * sizeof(float));

        generate_synthetic_q4_0(hW, num_blocks, &seed);
        dequantize_q4_0(hW, hW_float, M, K);

        void *dW;
        CK(cudaMalloc(&dW, num_blocks * sizeof(BlockQ4_0)));
        CK(cudaMemcpy(dW, hW, num_blocks * sizeof(BlockQ4_0), cudaMemcpyHostToDevice));

        for (int ni = 0; ni < num_n; ni++) {
            int N = n_vals[ni];
            total_tests++;

            long x_count = (long)N * K;
            long y_count = (long)N * M;

            float *hX = (float *)malloc(x_count * sizeof(float));
            float *hY_cpu = (float *)malloc(y_count * sizeof(float));
            float *hY_seq = (float *)malloc(y_count * sizeof(float));
            float *hY_gemm = (float *)malloc(y_count * sizeof(float));

            generate_synthetic_float(hX, x_count, &seed);

            // Compute CPU reference
            cpu_matmul_q4_0(hW_float, hX, hY_cpu, N, M, K);

            float *dX, *dY_seq, *dY_gemm;
            CK(cudaMalloc((void**)&dX, x_count * sizeof(float)));
            CK(cudaMalloc((void**)&dY_seq, y_count * sizeof(float)));
            CK(cudaMalloc((void**)&dY_gemm, y_count * sizeof(float)));

            CK(cudaMemcpy(dX, hX, x_count * sizeof(float), cudaMemcpyHostToDevice));

            // Sequential GEMV calls
            for (int i = 0; i < N; i++) {
                int rc = tt_gemv_q4_0(dW, dX + (long)i * K, dY_seq + (long)i * M, M, K, 0);
                CK(rc);
            }

            // Batched prefill GEMM launcher
            int rc = tt_gemm_q4_0_prefill(dW, dX, dY_gemm, M, K, N, 0);
            CK(rc);

            CK(cudaDeviceSynchronize());

            CK(cudaMemcpy(hY_seq, dY_seq, y_count * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hY_gemm, dY_gemm, y_count * sizeof(float), cudaMemcpyDeviceToHost));

            // Verify outputs
            float max_abs_cpu = 0.0f;
            float max_rel_cpu = 0.0f;
            float max_abs_seq = 0.0f;
            float max_rel_seq = 0.0f;
            int nan_inf_count = 0;

            for (long i = 0; i < y_count; i++) {
                float val_gemm = hY_gemm[i];
                float val_cpu  = hY_cpu[i];
                float val_seq  = hY_seq[i];

                if (isnan(val_gemm) || isinf(val_gemm)) {
                    nan_inf_count++;
                }

                float abs_cpu = fabsf(val_gemm - val_cpu);
                float rel_cpu = abs_cpu / fmaxf(fabsf(val_cpu), 1e-2f);
                if (abs_cpu > max_abs_cpu) max_abs_cpu = abs_cpu;
                if (rel_cpu > max_rel_cpu) max_rel_cpu = rel_cpu;

                float abs_seq = fabsf(val_gemm - val_seq);
                float rel_seq = abs_seq / fmaxf(fabsf(val_seq), 1e-2f);
                if (abs_seq > max_abs_seq) max_abs_seq = abs_seq;
                if (rel_seq > max_rel_seq) max_rel_seq = rel_seq;
            }

            int pass = (max_abs_cpu < 1e-3f) && (max_rel_cpu < 1e-2f) &&
                       (max_abs_seq < 1e-3f) && (max_rel_seq < 1e-2f) &&
                       (nan_inf_count == 0);

            if (pass) {
                total_passed++;
                printf("[PASS] N=%3d M=%4d K=%4d | abs_cpu=%.2e rel_cpu=%.2e | abs_seq=%.2e rel_seq=%.2e | nan/inf=%d\n",
                       N, M, K, max_abs_cpu, max_rel_cpu, max_abs_seq, max_rel_seq, nan_inf_count);
            } else {
                total_failed++;
                printf("[FAIL] N=%3d M=%4d K=%4d | abs_cpu=%.2e rel_cpu=%.2e | abs_seq=%.2e rel_seq=%.2e | nan/inf=%d\n",
                       N, M, K, max_abs_cpu, max_rel_cpu, max_abs_seq, max_rel_seq, nan_inf_count);
            }

            CK(cudaFree(dX));
            CK(cudaFree(dY_seq));
            CK(cudaFree(dY_gemm));
            free(hX);
            free(hY_cpu);
            free(hY_seq);
            free(hY_gemm);
        }

        CK(cudaFree(dW));
        free(hW);
        free(hW_float);
    }

    printf("========================================================================\n");
    printf("Result Summary: %d / %d PASSED (%d FAILED)\n", total_passed, total_tests, total_failed);
    printf("========================================================================\n");

    return (total_failed == 0) ? 0 : 1;
}
