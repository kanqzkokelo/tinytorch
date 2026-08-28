#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <string.h>

#include "qwen2_engine.h"

using namespace nvcuda;

extern "C" {
int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
int tt_gemm_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM, int M, int K, int N, cudaStream_t s);
}

typedef struct {
    half d;            // fp16 scale
    uint8_t qs[16];    // 32 packed nibbles
} LocalBlockQ4_0;

/*
 * Tensor Core WMMA Batched Q4_0 Prefill GEMM Kernel
 * Matrix math: Y = X * W^T
 *   X: [N, K] float (row-major activations)
 *   W: [M, K] BlockQ4_0 (row-major quantized weights)
 *   Y: [N, M] float (row-major output)
 *
 * Tile layout: BLOCK_M = 128, BLOCK_N = 32, BLOCK_K = 32
 * CTA layout: 256 threads (8 warps: 2 warps in N, 4 warps in M)
 *   warp_n = warp_id / 4 (0..1, each warp computes 16 tokens)
 *   warp_m = warp_id % 4 (0..3, each warp computes 32 output rows = 2 WMMA tiles)
 * WMMA fragment sizes: 16x16x16 (matrix_a FP16, matrix_b FP16, accumulator FP32)
 */
__global__ __launch_bounds__(256)
void k_gemm_wmma_q4_0_prefill(
    const void *__restrict__ dW,
    const float *__restrict__ dX,
    float *__restrict__ dY,
    int M, int K, int N)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int warp_n = warp_id / 4; // 0..1
    const int warp_m = warp_id % 4; // 0..3

    const int cta_m_base = blockIdx.x * 128;
    const int cta_n_base = blockIdx.y * 32;

    const int my_m_base = cta_m_base + warp_m * 32;
    const int my_n_base = cta_n_base + warp_n * 16;

    __align__(16) __shared__ half s_X[32][32];
    __align__(16) __shared__ half s_W[128][32];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;
    wmma::fill_fragment(c_frag0, 0.0f);
    wmma::fill_fragment(c_frag1, 0.0f);

    const int nb = K / 32;
    const int n_k_tiles = K / 32;

    for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
        const int k_base = k_tile * 32;

        // 1. Cooperative load X [32 x 32] = 1024 floats into s_X (Row-Major layout).
        // 256 threads load 4 floats each.
        #pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            int r = i / 32;
            int c = i % 32;
            int g_n = cta_n_base + r;
            int g_k = k_base + c;
            s_X[r][c] = (g_n < N && g_k < K) ? __float2half(dX[g_n * K + g_k]) : __float2half(0.0f);
        }

        // 2. Cooperative load W [128 x 32] = 128 Q4_0 blocks into s_W (Column-Major layout: s_W[m][k]).
        // 256 threads: 2 threads per row (128 rows). Each thread loads 16 elements.
        int r_w = tid / 2;    // m row 0..127
        int sub_k = tid % 2;  // sub_k 0..1 (16 elements: sub_k*16 .. sub_k*16+15)
        int g_m = cta_m_base + r_w;
        int blk_idx = k_tile;

        if (g_m < M && k_base < K) {
            const LocalBlockQ4_0 *blk = (const LocalBlockQ4_0 *)dW + (long)g_m * nb + blk_idx;
            half d = blk->d;
            int c_start = sub_k * 16;
            int is_high = sub_k;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int q_byte = blk->qs[j];
                int nib = is_high ? ((q_byte >> 4) & 0xF) : (q_byte & 0xF);
                s_W[r_w][c_start + j] = __hmul(__int2half_rn(nib - 8), d);
            }
        } else {
            int c_start = sub_k * 16;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                s_W[r_w][c_start + j] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Accumulate over 2 sub-tiles along K (16 elements each)
        #pragma unroll
        for (int k_sub = 0; k_sub < 2; k_sub++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag0;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag1;

            wmma::load_matrix_sync(a_frag, (half*)&s_X[warp_n * 16][k_sub * 16], 32);
            wmma::load_matrix_sync(b_frag0, (half*)&s_W[warp_m * 32][k_sub * 16], 32);
            wmma::load_matrix_sync(b_frag1, (half*)&s_W[warp_m * 32 + 16][k_sub * 16], 32);

            wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
            wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        }
        __syncthreads();
    }

    // Store output accumulators to DRAM Y [N x M]
    if (my_n_base + 15 < N) {
        if (my_m_base + 15 < M) {
            wmma::store_matrix_sync(dY + my_n_base * M + my_m_base, c_frag0, M, wmma::mem_row_major);
        }
        if (my_m_base + 31 < M) {
            wmma::store_matrix_sync(dY + my_n_base * M + my_m_base + 16, c_frag1, M, wmma::mem_row_major);
        }
    } else {
        __align__(16) __shared__ float s_C[32][128];
        wmma::store_matrix_sync((float*)&s_C[warp_n * 16][warp_m * 32], c_frag0, 128, wmma::mem_row_major);
        wmma::store_matrix_sync((float*)&s_C[warp_n * 16][warp_m * 32 + 16], c_frag1, 128, wmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16;
            int c = i % 16;
            int g_n = my_n_base + r;
            int g_m = my_m_base + c;
            if (g_n < N && g_m < M) {
                dY[g_n * M + g_m] = s_C[warp_n * 16 + r][warp_m * 32 + c];
            }
        }
    }
}

int tt_gemm_wmma_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM,
                              int M, int K, int N, cudaStream_t stream)
{
    dim3 grid((M + 127) / 128, (N + 31) / 32);
    dim3 block(256);
    k_gemm_wmma_q4_0_prefill<<<grid, block, 0, stream>>>(dW, dX_NxK, dY_NxM, M, K, N);
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

static void generate_synthetic_q4_0(LocalBlockQ4_0 *W, long num_blocks) {
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
    LocalBlockQ4_0 *hW = (LocalBlockQ4_0 *)malloc(num_blocks * sizeof(LocalBlockQ4_0));
    float *hX = (float *)malloc((long)N * K * sizeof(float));
    float *hY_seq = (float *)malloc((long)N * M * sizeof(float));
    float *hY_cuda = (float *)malloc((long)N * M * sizeof(float));
    float *hY_wmma = (float *)malloc((long)N * M * sizeof(float));

    srand(42);
    generate_synthetic_q4_0(hW, num_blocks);
    generate_synthetic_float(hX, (long)N * K);

    void *dW;
    float *dX, *dY_seq, *dY_cuda, *dY_wmma;
    cudaMalloc(&dW, num_blocks * sizeof(LocalBlockQ4_0));
    cudaMalloc(&dX, (long)N * K * sizeof(float));
    cudaMalloc(&dY_seq, (long)N * M * sizeof(float));
    cudaMalloc(&dY_cuda, (long)N * M * sizeof(float));
    cudaMalloc(&dY_wmma, (long)N * M * sizeof(float));

    cudaMemcpy(dW, hW, num_blocks * sizeof(LocalBlockQ4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, (long)N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    for (int iter = 0; iter < 3; iter++) {
        for (int i = 0; i < N; i++) {
            tt_gemv_q4_0(dW, dX + (long)i * K, dY_seq + (long)i * M, M, K, 0);
        }
        tt_gemm_q4_0_prefill(dW, dX, dY_cuda, M, K, N, 0);
        tt_gemm_wmma_q4_0_prefill(dW, dX, dY_wmma, M, K, N, 0);
    }
    cudaDeviceSynchronize();

    // Copy results to host to verify correctness
    cudaMemcpy(hY_seq, dY_seq, (long)N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hY_wmma, dY_wmma, (long)N * M * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_error = 0.0f;
    for (long i = 0; i < (long)N * M; i++) {
        float err = fabsf(hY_seq[i] - hY_wmma[i]);
        if (err > max_abs_error) max_abs_error = err;
    }

    int num_iters = (N >= 512) ? 20 : 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. time_seq: N sequential single GEMV calls
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

    // 2. time_cuda_gemm: CUDA core 2D batched GEMM
    cudaEventRecord(start, 0);
    for (int iter = 0; iter < num_iters; iter++) {
        tt_gemm_q4_0_prefill(dW, dX, dY_cuda, M, K, N, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms_cuda = 0.0f;
    cudaEventElapsedTime(&ms_cuda, start, stop);
    float time_cuda_gemm = ms_cuda / num_iters;

    // 3. time_wmma_gemm: Tensor Core WMMA batched GEMM
    cudaEventRecord(start, 0);
    for (int iter = 0; iter < num_iters; iter++) {
        tt_gemm_wmma_q4_0_prefill(dW, dX, dY_wmma, M, K, N, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms_wmma = 0.0f;
    cudaEventElapsedTime(&ms_wmma, start, stop);
    float time_wmma_gemm = ms_wmma / num_iters;

    float speedup = time_seq / time_wmma_gemm;

    printf("| N=%3d | M=%4d | K=%4d | %8.3f ms | %8.3f ms | %8.3f ms | %7.2fx | %13.6f |\n",
           N, M, K, time_seq, time_cuda_gemm, time_wmma_gemm, speedup, max_abs_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dY_seq);
    cudaFree(dY_cuda);
    cudaFree(dY_wmma);
    free(hW);
    free(hX);
    free(hY_seq);
    free(hY_cuda);
    free(hY_wmma);
}

int main() {
    printf("=========================================================================================================\n");
    printf("                  Microbench Tensor Core WMMA Q4_0 Batched Prefill GEMM Kernel\n");
    printf("=========================================================================================================\n");
    printf("|   N   |   M  |   K  |   time_seq |  time_cuda |  time_wmma | speedup | max_abs_error |\n");
    printf("|-------|------|------|------------|------------|------------|---------|----------------|\n");

    int N_vals[] = {64, 128, 256, 512};
    int shapes[][2] = {
        {896, 896},
        {4864, 896},
        {896, 4864},
        {4864, 4864}
    };

    for (int n_idx = 0; n_idx < 4; n_idx++) {
        int N = N_vals[n_idx];
        for (int s_idx = 0; s_idx < 4; s_idx++) {
            int M = shapes[s_idx][0];
            int K = shapes[s_idx][1];
            bench_shape(N, M, K);
        }
    }
    printf("=========================================================================================================\n");
    return 0;
}
