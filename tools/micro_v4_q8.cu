// M9.5 micro-bench: V2 vs V4 for q8_0 layer shapes (M=1024 K=1024, M=3072 K=1024).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_reduce(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

typedef struct {
    __half d;
    int8_t qs[32];
} BlockQ8_0;

// ===== V2 (copied from gemv_q4_cuda.cu) =====
__global__ void k_gemv_q8_0_v2(const BlockQ8_0 *W, const float *x, float *y, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 34);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 34);
    float s0 = 0.0f, s1 = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (34 * b) >> 2;
        const int sh  = (34 * b + 2) & 2;
        const unsigned short d16a = (unsigned short)(((34 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)(((34 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (34 * b + 2) >> 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const float4 xv = x4[k];
            s0 += ((float)((int)(va << 24) >> 24)) * da * xv.x;
            s0 += ((float)((int)(va << 16) >> 24)) * da * xv.y;
            s0 += ((float)((int)(va <<  8) >> 24)) * da * xv.z;
            s0 += ((float)((int)(va       ) >> 24)) * da * xv.w;
            s1 += ((float)((int)(vb << 24) >> 24)) * db * xv.x;
            s1 += ((float)((int)(vb << 16) >> 24)) * db * xv.y;
            s1 += ((float)((int)(vb <<  8) >> 24)) * db * xv.z;
            s1 += ((float)((int)(vb       ) >> 24)) * db * xv.w;
        }
    }
    s0 = warp_reduce(s0);
    s1 = warp_reduce(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// ===== V4 (copied from gemv_q4_cuda.cu) =====
__global__ void k_gemv_q8_0_v4(const BlockQ8_0 *W, const float *x, float *y, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= M) return;
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
        const int wsc = (34 * b) >> 2;
        const int sh  = (34 * b + 2) & 2;
        const unsigned short d16a = (unsigned short)(((34 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)(((34 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const unsigned short d16c = (unsigned short)(((34 * b) & 2) ? (rw2[wsc] >> 16) : (rw2[wsc] & 0xFFFFu));
        const unsigned short d16d = (unsigned short)(((34 * b) & 2) ? (rw3[wsc] >> 16) : (rw3[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));
        const int a0 = (34 * b + 2) >> 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
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
            const float4 xv = x4[k];
            s0 += ((float)((int)(va << 24) >> 24)) * da * xv.x;
            s0 += ((float)((int)(va << 16) >> 24)) * da * xv.y;
            s0 += ((float)((int)(va <<  8) >> 24)) * da * xv.z;
            s0 += ((float)((int)(va       ) >> 24)) * da * xv.w;
            s1 += ((float)((int)(vb << 24) >> 24)) * db * xv.x;
            s1 += ((float)((int)(vb << 16) >> 24)) * db * xv.y;
            s1 += ((float)((int)(vb <<  8) >> 24)) * db * xv.z;
            s1 += ((float)((int)(vb       ) >> 24)) * db * xv.w;
            s2 += ((float)((int)(vc << 24) >> 24)) * dc * xv.x;
            s2 += ((float)((int)(vc << 16) >> 24)) * dc * xv.y;
            s2 += ((float)((int)(vc <<  8) >> 24)) * dc * xv.z;
            s2 += ((float)((int)(vc       ) >> 24)) * dc * xv.w;
            s3 += ((float)((int)(vd << 24) >> 24)) * dd * xv.x;
            s3 += ((float)((int)(vd << 16) >> 24)) * dd * xv.y;
            s3 += ((float)((int)(vd <<  8) >> 24)) * dd * xv.z;
            s3 += ((float)((int)(vd       ) >> 24)) * dd * xv.w;
        }
    }
    s0 = warp_reduce(s0);
    s1 = warp_reduce(s1);
    s2 = warp_reduce(s2);
    s3 = warp_reduce(s3);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
        if (row2 < M) y[row2] = s2;
        if (row3 < M) y[row3] = s3;
    }
}

void run_bench(int M, int K) {
    printf("\n== q8_0 GEMV M=%d K=%d ==\n", M, K);
    const int nb = K / 32;
    const size_t wbytes = (size_t)M * nb * 34;
    const size_t xbytes = (size_t)K * 4;
    const size_t ybytes = (size_t)M * 4;
    BlockQ8_0 *hW = (BlockQ8_0*)malloc(wbytes);
    float *hx = (float*)malloc(xbytes);
    float *hy = (float*)malloc(ybytes);
    srand(42);
    for (size_t i = 0; i < wbytes/2; i++) ((uint16_t*)hW)[i] = rand() & 0xFFFF;
    for (int i = 0; i < K; i++) hx[i] = (rand()%2000 - 1000) * 0.001f;

    BlockQ8_0 *dW; float *dx, *dy;
    cudaMalloc(&dW, wbytes); cudaMalloc(&dx, xbytes); cudaMalloc(&dy, ybytes);
    cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, xbytes, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0, ybytes);

    auto bench = [&](const char* name, int blockY, int gridX, auto fn) {
        // warmup
        for (int i = 0; i < 5; i++) fn(dW, dx, dy, M, K, blockY, gridX);
        cudaDeviceSynchronize();
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        const int N = 100;
        for (int i = 0; i < N; i++) fn(dW, dx, dy, M, K, blockY, gridX);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        printf("  %s: %.3f us/call\n", name, ms*1000.0f/N);
    };

    // V2: blockY=16, grid.x = (M + 16*2 -1) / 32
    {
        int blockY = 16;
        int gridX = (M + blockY*2 - 1) / (blockY*2);
        bench("V2 (2r/warp)", blockY, gridX, [](BlockQ8_0* W, float* x, float* y, int M, int K, int by, int gx) {
            dim3 g(gx), b(32, by);
            k_gemv_q8_0_v2<<<g, b>>>(W, x, y, M, K);
        });
    }
    // V4: blockY=8, grid.x = (M + 8*4 -1) / 32
    {
        int blockY = 8;
        int gridX = (M + blockY*4 - 1) / (blockY*4);
        bench("V4 (4r/warp)", blockY, gridX, [](BlockQ8_0* W, float* x, float* y, int M, int K, int by, int gx) {
            dim3 g(gx), b(32, by);
            k_gemv_q8_0_v4<<<g, b>>>(W, x, y, M, K);
        });
    }

    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    free(hW); free(hx); free(hy);
}

int main() {
    // qwen3 shapes: attn_q/k/v M=2048 K=1024; ffn_gate/up M=3072 K=1024; ffn_down M=1024 K=3072
    run_bench(1024, 1024);
    run_bench(2048, 1024);
    run_bench(3072, 1024);
    run_bench(1024, 3072);
    // LM head
    run_bench(151936, 1024);
    return 0;
}
