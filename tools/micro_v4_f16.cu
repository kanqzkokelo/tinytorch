// M9.5 micro-bench: V2 vs V4 for F16 layer shapes
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

// V2 (copied from gemv_typed.cu)
__global__ void k_gemv_f16_v2(const __half *W, const float *x, float *y, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int K4 = K >> 2;
    const __half *rw0 = (const __half *)((const char *)W + (long)row0 * K * 2);
    const __half *rw1 = (const __half *)((const char *)W + (long)row1 * K * 2);
    const float4 *x4 = (const float4 *)x;
    float s0 = 0.0f, s1 = 0.0f;
    for (int g = lane; g < K4; g += 32) {
        const float4 xg = x4[g];
        const __half2 *h0 = (const __half2 *)(rw0 + (g << 2));
        const __half2 *h1 = (const __half2 *)(rw1 + (g << 2));
        const float2 w0a = __half22float2(h0[0]);
        const float2 w0b = __half22float2(h0[1]);
        const float2 w1a = __half22float2(h1[0]);
        const float2 w1b = __half22float2(h1[1]);
        s0 = fmaf(w0a.x, xg.x, s0);
        s0 = fmaf(w0a.y, xg.y, s0);
        s0 = fmaf(w0b.x, xg.z, s0);
        s0 = fmaf(w0b.y, xg.w, s0);
        s1 = fmaf(w1a.x, xg.x, s1);
        s1 = fmaf(w1a.y, xg.y, s1);
        s1 = fmaf(w1b.x, xg.z, s1);
        s1 = fmaf(w1b.y, xg.w, s1);
    }
    s0 = warp_reduce(s0);
    s1 = warp_reduce(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// V4 (copied from gemv_q4_cuda.cu)
__global__ void k_gemv_f16_v4(const __half *W, const float *x, float *y, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const int lane = threadIdx.x;
    const int K4 = K >> 2;
    const __half *rw0 = W + (long)row0 * K;
    const __half *rw1 = W + (long)row1 * K;
    const __half *rw2 = W + (long)row2 * K;
    const __half *rw3 = W + (long)row3 * K;
    const float4 *x4 = (const float4 *)x;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int g = lane; g < K4; g += 32) {
        const float4 xg = x4[g];
        const __half2 *h0 = (const __half2 *)(rw0 + (g << 2));
        const __half2 *h1 = (const __half2 *)(rw1 + (g << 2));
        const __half2 *h2 = (const __half2 *)(rw2 + (g << 2));
        const __half2 *h3 = (const __half2 *)(rw3 + (g << 2));
        const float2 w0a = __half22float2(h0[0]);
        const float2 w0b = __half22float2(h0[1]);
        const float2 w1a = __half22float2(h1[0]);
        const float2 w1b = __half22float2(h1[1]);
        const float2 w2a = __half22float2(h2[0]);
        const float2 w2b = __half22float2(h2[1]);
        const float2 w3a = __half22float2(h3[0]);
        const float2 w3b = __half22float2(h3[1]);
        s0 = fmaf(w0a.x, xg.x, s0);
        s0 = fmaf(w0a.y, xg.y, s0);
        s0 = fmaf(w0b.x, xg.z, s0);
        s0 = fmaf(w0b.y, xg.w, s0);
        s1 = fmaf(w1a.x, xg.x, s1);
        s1 = fmaf(w1a.y, xg.y, s1);
        s1 = fmaf(w1b.x, xg.z, s1);
        s1 = fmaf(w1b.y, xg.w, s1);
        s2 = fmaf(w2a.x, xg.x, s2);
        s2 = fmaf(w2a.y, xg.y, s2);
        s2 = fmaf(w2b.x, xg.z, s2);
        s2 = fmaf(w2b.y, xg.w, s2);
        s3 = fmaf(w3a.x, xg.x, s3);
        s3 = fmaf(w3a.y, xg.y, s3);
        s3 = fmaf(w3b.x, xg.z, s3);
        s3 = fmaf(w3b.y, xg.w, s3);
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
    printf("\n== F16 GEMV M=%d K=%d ==\n", M, K);
    const size_t wbytes = (size_t)M * K * 2;
    const size_t xbytes = (size_t)K * 4;
    const size_t ybytes = (size_t)M * 4;
    __half *hW = (__half*)malloc(wbytes);
    float *hx = (float*)malloc(xbytes);
    float *hy = (float*)malloc(ybytes);
    srand(42);
    for (size_t i = 0; i < wbytes/2; i++) ((uint16_t*)hW)[i] = rand() & 0xFFFF;
    for (int i = 0; i < K; i++) hx[i] = (rand()%2000 - 1000) * 0.001f;

    __half *dW; float *dx, *dy;
    cudaMalloc(&dW, wbytes); cudaMalloc(&dx, xbytes); cudaMalloc(&dy, ybytes);
    cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, xbytes, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0, ybytes);

    auto bench = [&](const char* name, int blockY, int gridX, auto fn) {
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

    {
        int blockY = 16;
        int gridX = (M + blockY*2 - 1) / (blockY*2);
        bench("V2 (2r/warp)", blockY, gridX, [](__half* W, float* x, float* y, int M, int K, int by, int gx) {
            dim3 g(gx), b(32, by);
            k_gemv_f16_v2<<<g, b>>>(W, x, y, M, K);
        });
    }
    {
        int blockY = 8;
        int gridX = (M + blockY*4 - 1) / (blockY*4);
        bench("V4 (4r/warp)", blockY, gridX, [](__half* W, float* x, float* y, int M, int K, int by, int gx) {
            dim3 g(gx), b(32, by);
            k_gemv_f16_v4<<<g, b>>>(W, x, y, M, K);
        });
    }

    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    free(hW); free(hx); free(hy);
}

int main() {
    // smollm2-135m: dim=576 hidden=1536 layers=30; K=576, 1536
    run_bench(576, 576);
    run_bench(576, 1536);
    run_bench(1536, 576);
    run_bench(151936, 576);
    return 0;
}
