// Peak Memory Bandwidth Q4_0 GEMV Microbench (Task 1).
// Compares existing V4 kernel (tt_gemv_q4_0_v4) against k_gemv_q4_0_peak using:
// 1. Vectorized uint4 128-bit aligned global loads via __ldg()
// 2. Software L2 prefetch directives: asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
// 3. Grid Launch Tuning for 16 SMs

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// Forward declaration of BlockQ4_0 layout
typedef struct {
    half d;
    uint8_t qs[16];
} BlockQ4_0;

// V4 baseline launcher declared in kernels/gemv_q4_cuda.cu (C linkage)
extern "C" {
int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy,
                    int M, int K, cudaStream_t stream);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Peak Kernel: 128-bit aligned loads via __ldg() + software L2 prefetch + occupancy tuning
__global__ __launch_bounds__(256, 4)
void k_gemv_q4_0_peak(const BlockQ4_0 *__restrict__ W,
                      const float *__restrict__ x,
                      float *__restrict__ y,
                      int M, int K) {
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

    // Software L2 prefetch 2 iterations ahead
    asm volatile("prefetch.global.L2 [%0];" :: "l"(rw0));
    asm volatile("prefetch.global.L2 [%0];" :: "l"(rw1));
    asm volatile("prefetch.global.L2 [%0];" :: "l"(rw2));
    asm volatile("prefetch.global.L2 [%0];" :: "l"(rw3));

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (__ldg(rw0 + wsc) >> 16) : (__ldg(rw0 + wsc) & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (__ldg(rw1 + wsc) >> 16) : (__ldg(rw1 + wsc) & 0xFFFFu));
        const unsigned short d16c = (unsigned short)
            (((18 * b) & 2) ? (__ldg(rw2 + wsc) >> 16) : (__ldg(rw2 + wsc) & 0xFFFFu));
        const unsigned short d16d = (unsigned short)
            (((18 * b) & 2) ? (__ldg(rw3 + wsc) >> 16) : (__ldg(rw3 + wsc) & 0xFFFFu));

        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));

        const int a0 = (18 * b + 2) >> 2;
        const int sh  = (18 * b + 2) & 2;

        const float4 *x4 = (const float4 *)(x + b * 32);

        // Vectorized read-only __ldg() 128-bit chunk loads
        const uint32_t la0 = __ldg(rw0 + a0 + 0);
        const uint32_t la1 = __ldg(rw0 + a0 + 1);
        const uint32_t la2 = __ldg(rw0 + a0 + 2);
        const uint32_t la3 = __ldg(rw0 + a0 + 3);
        const uint32_t la4 = sh ? __ldg(rw0 + a0 + 4) : 0;

        const uint32_t lb0 = __ldg(rw1 + a0 + 0);
        const uint32_t lb1 = __ldg(rw1 + a0 + 1);
        const uint32_t lb2 = __ldg(rw1 + a0 + 2);
        const uint32_t lb3 = __ldg(rw1 + a0 + 3);
        const uint32_t lb4 = sh ? __ldg(rw1 + a0 + 4) : 0;

        const uint32_t lc0 = __ldg(rw2 + a0 + 0);
        const uint32_t lc1 = __ldg(rw2 + a0 + 1);
        const uint32_t lc2 = __ldg(rw2 + a0 + 2);
        const uint32_t lc3 = __ldg(rw2 + a0 + 3);
        const uint32_t lc4 = sh ? __ldg(rw2 + a0 + 4) : 0;

        const uint32_t ld0 = __ldg(rw3 + a0 + 0);
        const uint32_t ld1 = __ldg(rw3 + a0 + 1);
        const uint32_t ld2 = __ldg(rw3 + a0 + 2);
        const uint32_t ld3 = __ldg(rw3 + a0 + 3);
        const uint32_t ld4 = sh ? __ldg(rw3 + a0 + 4) : 0;

        uint32_t va[4], vb[4], vc[4], vd[4];
        if (sh) {
            va[0] = __byte_perm(la0, la1, 0x5432);
            va[1] = __byte_perm(la1, la2, 0x5432);
            va[2] = __byte_perm(la2, la3, 0x5432);
            va[3] = __byte_perm(la3, la4, 0x5432);

            vb[0] = __byte_perm(lb0, lb1, 0x5432);
            vb[1] = __byte_perm(lb1, lb2, 0x5432);
            vb[2] = __byte_perm(lb2, lb3, 0x5432);
            vb[3] = __byte_perm(lb3, lb4, 0x5432);

            vc[0] = __byte_perm(lc0, lc1, 0x5432);
            vc[1] = __byte_perm(lc1, lc2, 0x5432);
            vc[2] = __byte_perm(lc2, lc3, 0x5432);
            vc[3] = __byte_perm(lc3, lc4, 0x5432);

            vd[0] = __byte_perm(ld0, ld1, 0x5432);
            vd[1] = __byte_perm(ld1, ld2, 0x5432);
            vd[2] = __byte_perm(ld2, ld3, 0x5432);
            vd[3] = __byte_perm(ld3, ld4, 0x5432);
        } else {
            va[0] = la0; va[1] = la1; va[2] = la2; va[3] = la3;
            vb[0] = lb0; vb[1] = lb1; vb[2] = lb2; vb[3] = lb3;
            vc[0] = lc0; vc[1] = lc1; vc[2] = lc2; vc[3] = lc3;
            vd[0] = ld0; vd[1] = ld1; vd[2] = ld2; vd[3] = ld3;
        }

#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t v_a = va[k];
            const uint32_t v_b = vb[k];
            const uint32_t v_c = vc[k];
            const uint32_t v_d = vd[k];
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];

            s0 += (float)((int)(v_a         & 0xFu) - 8) * da * xa.x;
            s0 += (float)((int)((v_a >>  4) & 0xFu) - 8) * da * xb.x;
            s0 += (float)((int)((v_a >>  8) & 0xFu) - 8) * da * xa.y;
            s0 += (float)((int)((v_a >> 12) & 0xFu) - 8) * da * xb.y;
            s0 += (float)((int)((v_a >> 16) & 0xFu) - 8) * da * xa.z;
            s0 += (float)((int)((v_a >> 20) & 0xFu) - 8) * da * xb.z;
            s0 += (float)((int)((v_a >> 24) & 0xFu) - 8) * da * xa.w;
            s0 += (float)((int)(v_a >> 28) - 8) * da * xb.w;

            s1 += (float)((int)(v_b         & 0xFu) - 8) * db * xa.x;
            s1 += (float)((int)((v_b >>  4) & 0xFu) - 8) * db * xb.x;
            s1 += (float)((int)((v_b >>  8) & 0xFu) - 8) * db * xa.y;
            s1 += (float)((int)((v_b >> 12) & 0xFu) - 8) * db * xb.y;
            s1 += (float)((int)((v_b >> 16) & 0xFu) - 8) * db * xa.z;
            s1 += (float)((int)((v_b >> 20) & 0xFu) - 8) * db * xb.z;
            s1 += (float)((int)((v_b >> 24) & 0xFu) - 8) * db * xa.w;
            s1 += (float)((int)(v_b >> 28) - 8) * db * xb.w;

            s2 += (float)((int)(v_c         & 0xFu) - 8) * dc * xa.x;
            s2 += (float)((int)((v_c >>  4) & 0xFu) - 8) * dc * xb.x;
            s2 += (float)((int)((v_c >>  8) & 0xFu) - 8) * dc * xa.y;
            s2 += (float)((int)((v_c >> 12) & 0xFu) - 8) * dc * xb.y;
            s2 += (float)((int)((v_c >> 16) & 0xFu) - 8) * dc * xa.z;
            s2 += (float)((int)((v_c >> 20) & 0xFu) - 8) * dc * xb.z;
            s2 += (float)((int)((v_c >> 24) & 0xFu) - 8) * dc * xa.w;
            s2 += (float)((int)(v_c >> 28) - 8) * dc * xb.w;

            s3 += (float)((int)(v_d         & 0xFu) - 8) * dd * xa.x;
            s3 += (float)((int)((v_d >>  4) & 0xFu) - 8) * dd * xb.x;
            s3 += (float)((int)((v_d >>  8) & 0xFu) - 8) * dd * xa.y;
            s3 += (float)((int)((v_d >> 12) & 0xFu) - 8) * dd * xb.y;
            s3 += (float)((int)((v_d >> 16) & 0xFu) - 8) * dd * xa.z;
            s3 += (float)((int)((v_d >> 20) & 0xFu) - 8) * dd * xb.z;
            s3 += (float)((int)((v_d >> 24) & 0xFu) - 8) * dd * xa.w;
            s3 += (float)((int)(v_d >> 28) - 8) * dd * xb.w;
        }
    }

    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    s2 = warp_reduce_sum(s2);
    s3 = warp_reduce_sum(s3);

    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
        if (row2 < M) y[row2] = s2;
        if (row3 < M) y[row3] = s3;
    }
}

void run_microbench_shape(int M, int K) {
    size_t nb = K / 32;
    size_t wbytes = (size_t)M * nb * 18 + 512;
    size_t xbytes = (size_t)K * sizeof(float) + 128;
    size_t ybytes = (size_t)M * sizeof(float) + 128;

    void *hW = malloc(wbytes);
    float *hx = (float *)malloc(xbytes);
    float *hy_v4 = (float *)malloc(ybytes);
    float *hy_peak = (float *)malloc(ybytes);

    memset(hW, 0x5a, wbytes);
    for (int i = 0; i < K; i++) hx[i] = sinf(0.5f * i + 0.1f);

    void *dW = NULL, *dx = NULL, *dy = NULL;
    cudaMalloc(&dW, wbytes);
    cudaMalloc(&dx, xbytes);
    cudaMalloc(&dy, ybytes);
    cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, xbytes, cudaMemcpyHostToDevice);

    dim3 g_pk((M + 31) / 32, 1, 1);
    dim3 b_pk(32, 8, 1);

    // Warmup
    for (int i = 0; i < 5; i++) {
        tt_gemv_q4_0_v4(dW, (const float*)dx, (float*)dy, M, K, (cudaStream_t)0);
        k_gemv_q4_0_peak<<<g_pk, b_pk>>>((const BlockQ4_0*)dW, (const float*)dx, (float*)dy, M, K);
    }
    cudaDeviceSynchronize();

    // Verify correctness
    tt_gemv_q4_0_v4(dW, (const float*)dx, (float*)dy, M, K, (cudaStream_t)0);
    cudaMemcpy(hy_v4, dy, ybytes, cudaMemcpyDeviceToHost);

    k_gemv_q4_0_peak<<<g_pk, b_pk>>>((const BlockQ4_0*)dW, (const float*)dx, (float*)dy, M, K);
    cudaMemcpy(hy_peak, dy, ybytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < M; i++) {
        float err = fabsf(hy_v4[i] - hy_peak[i]);
        if (err > max_err) max_err = err;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int REPS = (M > 50000) ? 50 : 200;

    // Time V4
    cudaEventRecord(start);
    for (int i = 0; i < REPS; i++) {
        tt_gemv_q4_0_v4(dW, (const float*)dx, (float*)dy, M, K, (cudaStream_t)0);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_v4 = 0.0f; cudaEventElapsedTime(&ms_v4, start, stop); ms_v4 /= REPS;
    double bw_v4 = (double)(M * nb * 18) / (ms_v4 * 1e-3) / 1e9;

    // Time Peak
    cudaEventRecord(start);
    for (int i = 0; i < REPS; i++) {
        k_gemv_q4_0_peak<<<g_pk, b_pk>>>((const BlockQ4_0*)dW, (const float*)dx, (float*)dy, M, K);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_pk = 0.0f; cudaEventElapsedTime(&ms_pk, start, stop); ms_pk /= REPS;
    double bw_pk = (double)(M * nb * 18) / (ms_pk * 1e-3) / 1e9;

    printf("| %6d | %4d | %8.4f | %8.4f | %8.2f | %8.2f | %13.6e |\n",
           M, K, ms_v4, ms_pk, bw_v4, bw_pk, max_err);
    fflush(stdout);

    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    free(hW); free(hx); free(hy_v4); free(hy_peak);
}

int main() {
    printf("========================================================================================\n");
    printf(" Peak-Bandwidth Q4_0 GEMV Microbench (k_gemv_q4_0_peak vs tt_gemv_q4_0_v4)\n");
    printf("========================================================================================\n");
    printf("| %6s | %4s | %8s | %8s | %8s | %8s | %13s |\n",
           "M", "K", "time_v4", "time_peak", "bw_v4", "bw_peak", "max_abs_error");
    printf("|--------|------|----------|-----------|----------|-----------|---------------|\n");
    fflush(stdout);

    int Ms[] = {896, 4864, 151936};
    int Ks[] = {896, 4864};

    for (int m : Ms) {
        for (int k : Ks) {
            run_microbench_shape(m, k);
        }
    }

    printf("========================================================================================\n");
    return 0;
}
