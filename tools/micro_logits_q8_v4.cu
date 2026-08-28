#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <algorithm>

typedef struct {
    half d;
    int8_t qs[32];
} BlockQ8_0;

static __device__ __forceinline__ float micro_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// V1: scalar twin of k_logits_q8_0 (1 row per warp)
__global__ void k_logits_q8_0_v1(const BlockQ8_0 *__restrict__ W,
                                 const float *__restrict__ x,
                                 float *__restrict__ logits,
                                 int vocab, int K) {
    const int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= vocab) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *roww = (const uint32_t *)((const char *)W + (long)v * nb * 34);
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;                 // word holding blk.d
        const unsigned short d16 = (unsigned short)
            (((34 * b) & 2) ? (roww[wsc] >> 16) : (roww[wsc] & 0xFFFFu));
        const float d = __half2float(__ushort_as_half(d16));
        const int a0 = (34 * b + 2) >> 2;              // first qs word
        const int sh  = (34 * b + 2) & 2;              // 2 => misaligned merge
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t lo = roww[a0 + k];
            const uint32_t vv = sh ? __byte_perm(lo, roww[a0 + k + 1], 0x5432) : lo;
            const float4 xv = x4[k];
            sum += ((float)((int)(vv << 24) >> 24)) * d * xv.x;
            sum += ((float)((int)(vv << 16) >> 24)) * d * xv.y;
            sum += ((float)((int)(vv <<  8) >> 24)) * d * xv.z;
            sum += ((float)((int)(vv       ) >> 24)) * d * xv.w;
        }
    }
    sum = micro_warp_reduce_sum(sum);
    if (lane == 0) logits[v] = sum;
}

// V4: 4 rows per warp for Q8_0 LM head (pre-loads x vector into registers)
__global__ void k_logits_q8_0_v4(const BlockQ8_0 *__restrict__ W,
                                 const float *__restrict__ x,
                                 float *__restrict__ logits,
                                 int vocab, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= vocab) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;

    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 34);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 34);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 34);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 34);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;                 // word holding blk.d
        const int sh  = (34 * b + 2) & 2;              // 2 => misaligned merge
        const unsigned short d16a = (unsigned short)(((34 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)(((34 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const unsigned short d16c = (unsigned short)(((34 * b) & 2) ? (rw2[wsc] >> 16) : (rw2[wsc] & 0xFFFFu));
        const unsigned short d16d = (unsigned short)(((34 * b) & 2) ? (rw3[wsc] >> 16) : (rw3[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));
        const int a0 = (34 * b + 2) >> 2;              // first qs word
        const float4 *x4 = (const float4 *)(x + b * 32);

        float4 xv[8];
#pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x4[k];

#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t lc = rw2[a0 + k];
            const uint32_t ld = rw3[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const uint32_t vc = sh ? __byte_perm(lc, rw2[a0 + k + 1], 0x5432) : lc;
            const uint32_t vd = sh ? __byte_perm(ld, rw3[a0 + k + 1], 0x5432) : ld;
            const float4 xk = xv[k];

            s0 += ((float)((int)(va << 24) >> 24)) * da * xk.x;
            s0 += ((float)((int)(va << 16) >> 24)) * da * xk.y;
            s0 += ((float)((int)(va <<  8) >> 24)) * da * xk.z;
            s0 += ((float)((int)(va       ) >> 24)) * da * xk.w;

            s1 += ((float)((int)(vb << 24) >> 24)) * db * xk.x;
            s1 += ((float)((int)(vb << 16) >> 24)) * db * xk.y;
            s1 += ((float)((int)(vb <<  8) >> 24)) * db * xk.z;
            s1 += ((float)((int)(vb       ) >> 24)) * db * xk.w;

            s2 += ((float)((int)(vc << 24) >> 24)) * dc * xk.x;
            s2 += ((float)((int)(vc << 16) >> 24)) * dc * xk.y;
            s2 += ((float)((int)(vc <<  8) >> 24)) * dc * xk.z;
            s2 += ((float)((int)(vc       ) >> 24)) * dc * xk.w;

            s3 += ((float)((int)(vd << 24) >> 24)) * dd * xk.x;
            s3 += ((float)((int)(vd << 16) >> 24)) * dd * xk.y;
            s3 += ((float)((int)(vd <<  8) >> 24)) * dd * xk.z;
            s3 += ((float)((int)(vd       ) >> 24)) * dd * xk.w;
        }
    }
    s0 = micro_warp_reduce_sum(s0);
    s1 = micro_warp_reduce_sum(s1);
    s2 = micro_warp_reduce_sum(s2);
    s3 = micro_warp_reduce_sum(s3);

    if (lane == 0) {
        logits[row0] = s0;
        if (row1 < vocab) logits[row1] = s1;
        if (row2 < vocab) logits[row2] = s2;
        if (row3 < vocab) logits[row3] = s3;
    }
}

void bench_shape(int M, int K) {
    const int nb = K / 32;
    const size_t wbytes = (size_t)M * nb * sizeof(BlockQ8_0);
    const size_t xbytes = (size_t)K * sizeof(float);
    const size_t ybytes = (size_t)M * sizeof(float);

    BlockQ8_0 *hW = (BlockQ8_0 *)malloc(wbytes);
    float *hx = (float *)malloc(xbytes);
    float *hy_v1 = (float *)malloc(ybytes);
    float *hy_v4 = (float *)malloc(ybytes);

    srand(42);
    for (size_t b = 0; b < (size_t)M * nb; b++) {
        float scale = (float)(rand() % 100 + 1) * 0.001f;
        hW[b].d = __float2half(scale);
        for (int i = 0; i < 32; i++) {
            hW[b].qs[i] = (int8_t)(rand() % 255 - 127);
        }
    }
    for (int i = 0; i < K; i++) {
        hx[i] = (rand() % 2000 - 1000) * 0.001f;
    }

    BlockQ8_0 *dW;
    float *dx, *dy_v1, *dy_v4;
    cudaMalloc(&dW, wbytes);
    cudaMalloc(&dx, xbytes);
    cudaMalloc(&dy_v1, ybytes);
    cudaMalloc(&dy_v4, ybytes);

    cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, xbytes, cudaMemcpyHostToDevice);
    cudaMemset(dy_v1, 0, ybytes);
    cudaMemset(dy_v4, 0, ybytes);

    // Launch V1
    dim3 b1(32, 1);
    dim3 g1((M + b1.y - 1) / b1.y, 1, 1);

    // Launch V4
    dim3 b4(32, 1);
    dim3 g4((M + b4.y * 4 - 1) / (b4.y * 4), 1, 1);

    // Warmup
    for (int i = 0; i < 10; i++) {
        k_logits_q8_0_v1<<<g1, b1>>>(dW, dx, dy_v1, M, K);
        k_logits_q8_0_v4<<<g4, b4>>>(dW, dx, dy_v4, M, K);
    }
    cudaDeviceSynchronize();

    const int num_iters = 100;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1); cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    for (int i = 0; i < num_iters; i++) {
        k_logits_q8_0_v1<<<g1, b1>>>(dW, dx, dy_v1, M, K);
    }
    cudaEventRecord(stop1); cudaEventSynchronize(stop1);
    float ms_v1 = 0.0f; cudaEventElapsedTime(&ms_v1, start1, stop1);
    float time_v1 = ms_v1 / num_iters;

    cudaEvent_t start4, stop4;
    cudaEventCreate(&start4); cudaEventCreate(&stop4);

    cudaEventRecord(start4);
    for (int i = 0; i < num_iters; i++) {
        k_logits_q8_0_v4<<<g4, b4>>>(dW, dx, dy_v4, M, K);
    }
    cudaEventRecord(stop4); cudaEventSynchronize(stop4);
    float ms_v4 = 0.0f; cudaEventElapsedTime(&ms_v4, start4, stop4);
    float time_v4 = ms_v4 / num_iters;

    cudaMemcpy(hy_v1, dy_v1, ybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy_v4, dy_v4, ybytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < M; i++) {
        float err = fabsf(hy_v1[i] - hy_v4[i]);
        if (err > max_err) max_err = err;
    }

    float speedup = time_v1 / time_v4;

    printf("| %7d | %4d | %12.4f | %12.4f | %9.2fx | %15.4e |\n",
           M, K, time_v1, time_v4, speedup, max_err);

    cudaEventDestroy(start1); cudaEventDestroy(stop1);
    cudaEventDestroy(start4); cudaEventDestroy(stop4);
    cudaFree(dW); cudaFree(dx); cudaFree(dy_v1); cudaFree(dy_v4);
    free(hW); free(hx); free(hy_v1); free(hy_v4);
}

int main() {
    printf("=========================================================================================\n");
    printf("|       M |    K | time_v1 (ms) | time_v4 (ms) |   speedup |  max_abs_error  |\n");
    printf("=========================================================================================\n");
    bench_shape(896, 896);
    bench_shape(4864, 896);
    bench_shape(151936, 896);
    printf("=========================================================================================\n");
    return 0;
}
