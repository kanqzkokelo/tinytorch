// M9.5 micro-bench: V2 vs V4 for q4_0 LM head shape (M=151936, K=896).
// Standalone (not linked into engine). Times only the kernel itself.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
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
    uint8_t qs[16];
} BlockQ4_0;

// ===== V2 reference (copied from gemv_q4_cuda.cu) =====
__global__ void k_logits_q4_0_v2(const BlockQ4_0 *W, const float *x,
                                 float *logits, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (18 * b + 2) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            s0 += (float)((int)(va         & 0xFu) - 8) * da * xa.x;
            s0 += (float)((int)((va >>  4) & 0xFu) - 8) * da * xb.x;
            s0 += (float)((int)((va >>  8) & 0xFu) - 8) * da * xa.y;
            s0 += (float)((int)((va >> 12) & 0xFu) - 8) * da * xb.y;
            s0 += (float)((int)((va >> 16) & 0xFu) - 8) * da * xa.z;
            s0 += (float)((int)((va >> 20) & 0xFu) - 8) * da * xb.z;
            s0 += (float)((int)((va >> 24) & 0xFu) - 8) * da * xa.w;
            s0 += (float)((int)(va >> 28) - 8) * da * xb.w;
            s1 += (float)((int)(vb         & 0xFu) - 8) * db * xa.x;
            s1 += (float)((int)((vb >>  4) & 0xFu) - 8) * db * xb.x;
            s1 += (float)((int)((vb >>  8) & 0xFu) - 8) * db * xa.y;
            s1 += (float)((int)((vb >> 12) & 0xFu) - 8) * db * xb.y;
            s1 += (float)((int)((vb >> 16) & 0xFu) - 8) * db * xa.z;
            s1 += (float)((int)((vb >> 20) & 0xFu) - 8) * db * xb.z;
            s1 += (float)((int)((vb >> 24) & 0xFu) - 8) * db * xa.w;
            s1 += (float)((int)(vb >> 28) - 8) * db * xb.w;
        }
    }
    s0 = warp_reduce(s0);
    s1 = warp_reduce(s1);
    if (lane == 0) {
        logits[row0] = s0;
        if (row1 < M) logits[row1] = s1;
    }
}

// ===== V4: 4 rows per warp =====
__global__ void k_logits_q4_0_v4(const BlockQ4_0 *W, const float *x,
                                 float *logits, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 18);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const unsigned short d16c = (unsigned short)
            (((18 * b) & 2) ? (rw2[wsc] >> 16) : (rw2[wsc] & 0xFFFFu));
        const unsigned short d16d = (unsigned short)
            (((18 * b) & 2) ? (rw3[wsc] >> 16) : (rw3[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));
        const int a0 = (18 * b + 2) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t lc = rw2[a0 + k];
            const uint32_t ld = rw3[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const uint32_t vc = sh ? __byte_perm(lc, rw2[a0 + k + 1], 0x5432) : lc;
            const uint32_t vd = sh ? __byte_perm(ld, rw3[a0 + k + 1], 0x5432) : ld;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];

            // row0
            s0 += (float)((int)(va         & 0xFu) - 8) * da * xa.x;
            s0 += (float)((int)((va >>  4) & 0xFu) - 8) * da * xb.x;
            s0 += (float)((int)((va >>  8) & 0xFu) - 8) * da * xa.y;
            s0 += (float)((int)((va >> 12) & 0xFu) - 8) * da * xb.y;
            s0 += (float)((int)((va >> 16) & 0xFu) - 8) * da * xa.z;
            s0 += (float)((int)((va >> 20) & 0xFu) - 8) * da * xb.z;
            s0 += (float)((int)((va >> 24) & 0xFu) - 8) * da * xa.w;
            s0 += (float)((int)(va >> 28) - 8) * da * xb.w;
            // row1
            s1 += (float)((int)(vb         & 0xFu) - 8) * db * xa.x;
            s1 += (float)((int)((vb >>  4) & 0xFu) - 8) * db * xb.x;
            s1 += (float)((int)((vb >>  8) & 0xFu) - 8) * db * xa.y;
            s1 += (float)((int)((vb >> 12) & 0xFu) - 8) * db * xb.y;
            s1 += (float)((int)((vb >> 16) & 0xFu) - 8) * db * xa.z;
            s1 += (float)((int)((vb >> 20) & 0xFu) - 8) * db * xb.z;
            s1 += (float)((int)((vb >> 24) & 0xFu) - 8) * db * xa.w;
            s1 += (float)((int)(vb >> 28) - 8) * db * xb.w;
            // row2
            s2 += (float)((int)(vc         & 0xFu) - 8) * dc * xa.x;
            s2 += (float)((int)((vc >>  4) & 0xFu) - 8) * dc * xb.x;
            s2 += (float)((int)((vc >>  8) & 0xFu) - 8) * dc * xa.y;
            s2 += (float)((int)((vc >> 12) & 0xFu) - 8) * dc * xb.y;
            s2 += (float)((int)((vc >> 16) & 0xFu) - 8) * dc * xa.z;
            s2 += (float)((int)((vc >> 20) & 0xFu) - 8) * dc * xb.z;
            s2 += (float)((int)((vc >> 24) & 0xFu) - 8) * dc * xa.w;
            s2 += (float)((int)(vc >> 28) - 8) * dc * xb.w;
            // row3
            s3 += (float)((int)(vd         & 0xFu) - 8) * dd * xa.x;
            s3 += (float)((int)((vd >>  4) & 0xFu) - 8) * dd * xb.x;
            s3 += (float)((int)((vd >>  8) & 0xFu) - 8) * dd * xa.y;
            s3 += (float)((int)((vd >> 12) & 0xFu) - 8) * dd * xb.y;
            s3 += (float)((int)((vd >> 16) & 0xFu) - 8) * dd * xa.z;
            s3 += (float)((int)((vd >> 20) & 0xFu) - 8) * dd * xb.z;
            s3 += (float)((int)((vd >> 24) & 0xFu) - 8) * dd * xa.w;
            s3 += (float)((int)(vd >> 28) - 8) * dd * xb.w;
        }
    }
    s0 = warp_reduce(s0);
    s1 = warp_reduce(s1);
    s2 = warp_reduce(s2);
    s3 = warp_reduce(s3);
    if (lane == 0) {
        logits[row0] = s0;
        if (row1 < M) logits[row1] = s1;
        if (row2 < M) logits[row2] = s2;
        if (row3 < M) logits[row3] = s3;
    }
}

static void gemv_dims2(int M, dim3 *grid, dim3 *block) {
    block->x = 32; block->y = 16; block->z = 1;
    grid->x = (M + block->y * 2 - 1) / (block->y * 2);
    grid->y = 1; grid->z = 1;
}
static void gemv_dims4(int M, dim3 *grid, dim3 *block) {
    block->x = 32; block->y = 8; block->z = 1;
    grid->x = (M + block->y * 4 - 1) / (block->y * 4);
    grid->y = 1; grid->z = 1;
}

int main(int argc, char **argv) {
    int M = 151936, K = 896;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if ((K & 31) != 0) { fprintf(stderr, "K must be %%-32\n"); return 1; }
    int nb = K / 32;
    if ((nb & 1) != 0) { fprintf(stderr, "nb must be even\n"); return 1; }
    if (M & 3) { M = (M + 3) & ~3; fprintf(stderr, "[warn] M padded to %d\n", M); }

    size_t wbytes_alloc = (size_t)M * nb * 18;
    size_t wbytes = wbytes_alloc; // honest weight bytes: M*nb*18 = M*K*0.5625 for q4_0 (not M*K*2)
    size_t xbytes = (size_t)K * 4;
    size_t ybytes = (size_t)M * 4;

    void *hW = malloc(wbytes_alloc);
    void *hx = malloc(xbytes);
    float *hy = (float *)malloc(ybytes);
    memset(hW, 0xab, wbytes_alloc);
    for (int i = 0; i < K; i++) ((float *)hx)[i] = sinf(0.7f * i + 0.3f);
    memset(hy, 0, ybytes);

    void *dW, *dx, *dy;
    cudaMalloc(&dW, wbytes_alloc); cudaMemcpy(dW, hW, wbytes_alloc, cudaMemcpyHostToDevice);
    cudaMalloc(&dx, xbytes); cudaMemcpy(dx, hx, xbytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dy, ybytes);

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

    // Warmup
    for (int i = 0; i < 5; i++) {
        dim3 g, bl;
        gemv_dims2(M, &g, &bl);
        k_logits_q4_0_v2<<<g, bl>>>((const BlockQ4_0 *)dW, (const float *)dx,
                                    (float *)dy, M, K);
        gemv_dims4(M, &g, &bl);
        k_logits_q4_0_v4<<<g, bl>>>((const BlockQ4_0 *)dW, (const float *)dx,
                                    (float *)dy, M, K);
    }
    cudaDeviceSynchronize();

    // Correctness check
    {
        dim3 g, bl; gemv_dims2(M, &g, &bl);
        k_logits_q4_0_v2<<<g, bl>>>((const BlockQ4_0 *)dW, (const float *)dx,
                                    (float *)dy, M, K);
        std::vector<float> y_v2(M);
        cudaMemcpy(y_v2.data(), dy, ybytes, cudaMemcpyDeviceToHost);
        gemv_dims4(M, &g, &bl);
        k_logits_q4_0_v4<<<g, bl>>>((const BlockQ4_0 *)dW, (const float *)dx,
                                    (float *)dy, M, K);
        std::vector<float> y_v4(M);
        cudaMemcpy(y_v4.data(), dy, ybytes, cudaMemcpyDeviceToHost);
        double maxd = 0; int at = -1;
        for (int i = 0; i < M; i++) {
            double d = fabs((double)y_v2[i] - y_v4[i]);
            if (d > maxd) { maxd = d; at = i; }
        }
        printf("correctness: max |V2-V4| = %.6e at row %d (v2=%.4f v4=%.4f)\n",
               maxd, at, y_v2[at], y_v4[at]);
    }

    // Honest per-iter 200 samples median/p95
    auto bench_one = [&](auto kernel_lambda, const char* tag){
        const int REPS=200;
        const int WARM=20;
        dim3 g, bl;
        if(tag[1]=='2') gemv_dims2(M,&g,&bl); else gemv_dims4(M,&g,&bl);
        for(int i=0;i<WARM;i++) kernel_lambda(g,bl);
        cudaDeviceSynchronize();
        std::vector<float> samples; samples.reserve(REPS);
        for(int i=0;i<REPS;i++){
            cudaEventRecord(a);
            kernel_lambda(g,bl);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms_i; cudaEventElapsedTime(&ms_i,a,b);
            samples.push_back(ms_i);
        }
        std::sort(samples.begin(), samples.end());
        float p50=samples[REPS/2], p95=samples[(int)(REPS*0.95)], mean=0; for(auto x:samples) mean+=x; mean/=REPS;
        double bw_p50=(double)wbytes/(p50*1e-3)/1e9;
        double bw_mean=(double)wbytes/(mean*1e-3)/1e9;
        printf("%s M=%d K=%d  mean %.3f ms  median(p50) %.3f ms  p95 %.3f ms  weight-BW median %.1f GB/s (mean %.1f)  honest per-iter %d samples\n", tag, M,K, mean,p50,p95,bw_p50,bw_mean,REPS);
    };
    bench_one([&](dim3 g,dim3 bl){ k_logits_q4_0_v2<<<g,bl>>>((const BlockQ4_0*)dW,(const float*)dx,(float*)dy,M,K); }, "V2");
    bench_one([&](dim3 g,dim3 bl){ k_logits_q4_0_v4<<<g,bl>>>((const BlockQ4_0*)dW,(const float*)dx,(float*)dy,M,K); }, "V4");
    return 0;
}
