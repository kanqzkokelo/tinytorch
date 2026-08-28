#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "../kernels/gemv_q4_cuda.cu"

/*
 * 2D Tiled Q4_0 Batched GEMM Kernel for Prompt Prefill
 * Matrix math: Y = X * W^T
 *   X: [N, K] float (row-major activations)
 *   W: [M, K] BlockQ4_0 (row-major quantized weights)
 *   Y: [N, M] float (row-major output)
 *
 * Tile layout: BLOCK_M = 64, BLOCK_N = 32, BLOCK_K = 32
 * Threads per CTA: dim3 block(16, 16) = 256 threads
 * Register tiling: 2 tokens (N) x 4 rows (M) per thread => 8 accumulators per thread
 * Shared memory:
 *   __shared__ float sX[32][33]      (padding +1 to eliminate 32-bank conflicts)
 *   __shared__ float sW_d[65]        (padding +1)
 *   __shared__ uint32_t sW_v[64][5]  (padding +1 to eliminate 32-bank conflicts)
 */
__global__ __launch_bounds__(256, 4)
void k_gemm_q4_0_prefill(
    const void *__restrict__ dW,
    const float *__restrict__ dX,
    float *__restrict__ dY,
    int M, int K, int N)
{
    const int tx = threadIdx.x; // 0..15 (M dimension)
    const int ty = threadIdx.y; // 0..15 (N dimension)
    const int tid = ty * 16 + tx; // 0..255

    const int m_base = blockIdx.x * 64 + tx * 4;
    const int n_base = blockIdx.y * 32 + ty * 2;

    const int nb = K / 32; // q4_0 blocks per row

    __shared__ float sX[32][33];
    __shared__ float sW_d[65];
    __shared__ uint32_t sW_v[64][5];

    float acc[2][4];
    #pragma unroll
    for (int in = 0; in < 2; in++) {
        #pragma unroll
        for (int im = 0; im < 4; im++) {
            acc[in][im] = 0.0f;
        }
    }

    // Cooperative loading X: 256 threads load 32 tokens x 32 floats = 1024 floats (1 float4 per thread)
    const int n_load = tid / 8;     // 0..31
    const int k_vec_load = tid % 8; // 0..7
    const int n_global = blockIdx.y * 32 + n_load;

    for (int k_tile = 0; k_tile < nb; k_tile++) {
        // 1. Cooperative load X tile into sX (bank-conflict free: sX[32][33])
        const int k_global = k_tile * 32 + k_vec_load * 4;
        float4 x_vec;
        if (n_global < N && (k_global + 3) < K) {
            x_vec = *reinterpret_cast<const float4*>(&dX[n_global * K + k_global]);
        } else {
            x_vec.x = (n_global < N && (k_global + 0) < K) ? dX[n_global * K + k_global + 0] : 0.0f;
            x_vec.y = (n_global < N && (k_global + 1) < K) ? dX[n_global * K + k_global + 1] : 0.0f;
            x_vec.z = (n_global < N && (k_global + 2) < K) ? dX[n_global * K + k_global + 2] : 0.0f;
            x_vec.w = (n_global < N && (k_global + 3) < K) ? dX[n_global * K + k_global + 3] : 0.0f;
        }
        sX[n_load][k_vec_load * 4 + 0] = x_vec.x;
        sX[n_load][k_vec_load * 4 + 1] = x_vec.y;
        sX[n_load][k_vec_load * 4 + 2] = x_vec.z;
        sX[n_load][k_vec_load * 4 + 3] = x_vec.w;

        // 2. Cooperative load W tile into sW_d and sW_v (64 blocks loaded once per CTA)
        if (tid < 64) {
            int m_row = blockIdx.x * 64 + tid;
            if (m_row < M) {
                const uint32_t *rw = (const uint32_t *)((const char *)dW + (long)m_row * nb * 18);
                const int wsc = (18 * k_tile) >> 2;
                const unsigned short d16 = (unsigned short)(((18 * k_tile) & 2) ? (rw[wsc] >> 16) : (rw[wsc] & 0xFFFFu));
                sW_d[tid] = __half2float(__ushort_as_half(d16));

                const int a0 = (18 * k_tile + 2) >> 2;
                const int sh = (18 * k_tile + 2) & 2;

                #pragma unroll
                for (int k_sub = 0; k_sub < 4; k_sub++) {
                    uint32_t la = rw[a0 + k_sub];
                    uint32_t la_next = sh ? rw[a0 + k_sub + 1] : 0;
                    sW_v[tid][k_sub] = sh ? __byte_perm(la, la_next, 0x5432) : la;
                }
            } else {
                sW_d[tid] = 0.0f;
                #pragma unroll
                for (int k_sub = 0; k_sub < 4; k_sub++) sW_v[tid][k_sub] = 0;
            }
        }

        __syncthreads();

        // 3. Read weights from shared memory for 4 rows handled by thread
        float da[4];
        uint32_t va[4][4];
        #pragma unroll
        for (int im = 0; im < 4; im++) {
            int m_local = tx * 4 + im;
            da[im] = sW_d[m_local];
            #pragma unroll
            for (int k_sub = 0; k_sub < 4; k_sub++) {
                va[im][k_sub] = sW_v[m_local][k_sub];
            }
        }

        // 4. Compute dot products for 2 tokens x 4 rows
        #pragma unroll
        for (int k_sub = 0; k_sub < 4; k_sub++) {
            const int n0_local = ty * 2;
            const int n1_local = ty * 2 + 1;

            float x0_low0 = sX[n0_local][4 * k_sub + 0];
            float x0_low1 = sX[n0_local][4 * k_sub + 1];
            float x0_low2 = sX[n0_local][4 * k_sub + 2];
            float x0_low3 = sX[n0_local][4 * k_sub + 3];

            float x0_high0 = sX[n0_local][16 + 4 * k_sub + 0];
            float x0_high1 = sX[n0_local][16 + 4 * k_sub + 1];
            float x0_high2 = sX[n0_local][16 + 4 * k_sub + 2];
            float x0_high3 = sX[n0_local][16 + 4 * k_sub + 3];

            float x1_low0 = sX[n1_local][4 * k_sub + 0];
            float x1_low1 = sX[n1_local][4 * k_sub + 1];
            float x1_low2 = sX[n1_local][4 * k_sub + 2];
            float x1_low3 = sX[n1_local][4 * k_sub + 3];

            float x1_high0 = sX[n1_local][16 + 4 * k_sub + 0];
            float x1_high1 = sX[n1_local][16 + 4 * k_sub + 1];
            float x1_high2 = sX[n1_local][16 + 4 * k_sub + 2];
            float x1_high3 = sX[n1_local][16 + 4 * k_sub + 3];

            #pragma unroll
            for (int im = 0; im < 4; im++) {
                uint32_t v = va[im][k_sub];
                float d = da[im];

                int q0 = (int)(v & 0x0Fu) - 8;
                int q1 = (int)((v >> 4) & 0x0Fu) - 8;
                int q2 = (int)((v >> 8) & 0x0Fu) - 8;
                int q3 = (int)((v >> 12) & 0x0Fu) - 8;
                int q4 = (int)((v >> 16) & 0x0Fu) - 8;
                int q5 = (int)((v >> 20) & 0x0Fu) - 8;
                int q6 = (int)((v >> 24) & 0x0Fu) - 8;
                int q7 = (int)(v >> 28) - 8;

                float sum0 = (float)q0 * x0_low0  + (float)q1 * x0_high0
                           + (float)q2 * x0_low1  + (float)q3 * x0_high1
                           + (float)q4 * x0_low2  + (float)q5 * x0_high2
                           + (float)q6 * x0_low3  + (float)q7 * x0_high3;

                float sum1 = (float)q0 * x1_low0  + (float)q1 * x1_high0
                           + (float)q2 * x1_low1  + (float)q3 * x1_high1
                           + (float)q4 * x1_low2  + (float)q5 * x1_high2
                           + (float)q6 * x1_low3  + (float)q7 * x1_high3;

                acc[0][im] += sum0 * d;
                acc[1][im] += sum1 * d;
            }
        }

        __syncthreads();
    }

    // 5. Store Y accumulators
    #pragma unroll
    for (int in = 0; in < 2; in++) {
        int n_g = n_base + in;
        if (n_g < N) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int m_g = m_base + im;
                if (m_g < M) {
                    dY[(long)n_g * M + m_g] = acc[in][im];
                }
            }
        }
    }
}

int tt_gemm_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM,
                         int M, int K, int N, cudaStream_t stream)
{
    dim3 grid((M + 63) / 64, (N + 31) / 32);
    dim3 block(16, 16);
    k_gemm_q4_0_prefill<<<grid, block, 0, stream>>>(dW, dX_NxK, dY_NxM, M, K, N);
    return (int)cudaGetLastError();
}

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

static void generate_synthetic_q4_0(BlockQ4_0 *W, long num_blocks) {
    for (long i = 0; i < num_blocks; i++) {
        uint16_t h = float_to_half(0.05f);
        memcpy(&W[i].d, &h, 2);
        for (int j = 0; j < 16; j++) {
            W[i].qs[j] = (uint8_t)(rand() & 0xFF);
        }
    }
}

static void generate_synthetic_float(float *arr, long count) {
    for (long i = 0; i < count; i++) {
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void bench_shape(int N, int M, int K) {
    long num_blocks = (long)M * (K / 32);
    BlockQ4_0 *hW = (BlockQ4_0 *)malloc(num_blocks * sizeof(BlockQ4_0));
    float *hX = (float *)malloc((long)N * K * sizeof(float));
    float *hY_seq = (float *)malloc((long)N * M * sizeof(float));
    float *hY_gemm = (float *)malloc((long)N * M * sizeof(float));

    srand(42);
    generate_synthetic_q4_0(hW, num_blocks);
    generate_synthetic_float(hX, (long)N * K);

    void *dW;
    float *dX, *dY_seq, *dY_gemm;
    cudaMalloc(&dW, num_blocks * sizeof(BlockQ4_0));
    cudaMalloc(&dX, (long)N * K * sizeof(float));
    cudaMalloc(&dY_seq, (long)N * M * sizeof(float));
    cudaMalloc(&dY_gemm, (long)N * M * sizeof(float));

    cudaMemcpy(dW, hW, num_blocks * sizeof(BlockQ4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, (long)N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup sequential
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 0; i < N; i++) {
            tt_gemv_q4_0(dW, dX + (long)i * K, dY_seq + (long)i * M, M, K, 0);
        }
    }

    // Warmup batched GEMM
    for (int iter = 0; iter < 5; iter++) {
        tt_gemm_q4_0_prefill(dW, dX, dY_gemm, M, K, N, 0);
    }
    cudaDeviceSynchronize();

    // Verify correctness
    cudaMemcpy(hY_seq, dY_seq, (long)N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hY_gemm, dY_gemm, (long)N * M * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_error = 0.0f;
    for (long i = 0; i < (long)N * M; i++) {
        float err = fabsf(hY_seq[i] - hY_gemm[i]);
        if (err > max_abs_error) max_abs_error = err;
    }

    // Measure time_seq
    int num_iters = (N >= 512) ? 20 : 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int iter = 0; iter < num_iters; iter++) {
        for (int i = 0; i < N; i++) {
            tt_gemv_q4_0(dW, dX + (long)i * K, dY_seq + (long)i * M, M, K, 0);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms_seq = 0.0f;
    cudaEventElapsedTime(&ms_seq, start, stop);
    float time_seq = ms_seq / num_iters;

    // Measure time_gemm
    cudaEventRecord(start, 0);
    for (int iter = 0; iter < num_iters; iter++) {
        tt_gemm_q4_0_prefill(dW, dX, dY_gemm, M, K, N, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms_gemm = 0.0f;
    cudaEventElapsedTime(&ms_gemm, start, stop);
    float time_gemm = ms_gemm / num_iters;

    float speedup = time_seq / time_gemm;

    printf("| N=%3d | M=%4d | K=%4d | %8.3f ms | %8.3f ms | %7.2fx | %13.6f |\n",
           N, M, K, time_seq, time_gemm, speedup, max_abs_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dY_seq);
    cudaFree(dY_gemm);
    free(hW);
    free(hX);
    free(hY_seq);
    free(hY_gemm);
}

int main() {
    printf("========================================================================================\n");
    printf("                  Microbench Batched Q4_0 Prefill GEMM Kernel\n");
    printf("========================================================================================\n");
    printf("|   N   |   M  |   K  |   time_seq |  time_gemm | speedup | max_abs_error |\n");
    printf("|-------|------|------|------------|------------|---------|----------------|\n");

    int N_vals[] = {32, 128, 512};
    int shapes[][2] = {
        {896, 896},
        {4864, 896},
        {896, 4864},
        {4864, 4864}
    };

    for (int n_idx = 0; n_idx < 3; n_idx++) {
        int N = N_vals[n_idx];
        for (int s_idx = 0; s_idx < 4; s_idx++) {
            int M = shapes[s_idx][0];
            int K = shapes[s_idx][1];
            bench_shape(N, M, K);
        }
    }
    printf("========================================================================================\n");
    return 0;
}
