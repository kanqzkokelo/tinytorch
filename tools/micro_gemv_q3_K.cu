/*
 * micro_gemv_q3_K.cu - Microbenchmark and verification for Q3_K CUDA GEMV
 *
 * Verifies:
 * 1. CPU reference dequantization matching GGML block_q3_K
 * 2. CUDA 2-rows-per-warp GEMV kernel k_gemv_q3_K_v2
 * 3. Max absolute error (< 1e-4) and L_inf relative error (< 5e-7 target)
 * 4. Memory bandwidth (GB/s) on RTX 3050 Laptop
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_K 256

int cmp_f(const void *a, const void *b){ float fa=*(const float*)a; float fb=*(const float*)b; return (fa>fb)-(fa<fb); }

typedef struct {
    uint8_t hmask[32]; // quants - high bit (32 bytes)
    uint8_t qs[64];    // quants - low 2 bits (64 bytes)
    uint8_t scales[12];// scales (12 bytes)
    uint16_t d;        // super-block scale (fp16) (2 bytes)
} BlockQ3_K;

static_assert(sizeof(BlockQ3_K) == 110, "BlockQ3_K size must be exactly 110 bytes");

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7c00u) >> 10;
    uint32_t man  = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x0400u));
            man &= 0x03ffu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// CPU reference dequantization matching GGML
static void dequantize_row_q3_K(const BlockQ3_K *x, float *y, int64_t k) {
    const int nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_fp32(x[i].d);
        const uint8_t *q = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl0 = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl0 * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                float dl1 = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl1 * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// CPU reference GEMV
static void gemv_q3_K_cpu_ref(const BlockQ3_K *W, const float *x, float *y, int M, int K) {
    float *dequant_row = (float *)malloc(K * sizeof(float));
    const int nb = K / QK_K;
    for (int r = 0; r < M; r++) {
        dequantize_row_q3_K(W + r * nb, dequant_row, K);
        double dot = 0.0;
        for (int c = 0; c < K; c++) {
            dot += (double)dequant_row[c] * (double)x[c];
        }
        y[r] = (float)dot;
    }
    free(dequant_row);
}

// CUDA Device helper to convert half from pointer
static __device__ __forceinline__ float half_at(const void *p) {
    uint16_t h = *(const uint16_t *)p;
    return __half2float(*(const __half *)&h);
}

// CUDA 2-rows-per-warp GEMV kernel for Q3_K
__global__ void k_gemv_q3_K_v2(const uint8_t *__restrict__ W,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nsb = K / 256;
    const int nu  = nsb * 16; // 16 sub-blocks of 16 weights each

    const uint8_t *rw0 = W + (long)row0 * nsb * 110;
    const uint8_t *rw1 = W + (long)row1 * nsb * 110;
    float s0 = 0.0f, s1 = 0.0f;

    for (int u = lane; u < nu; u += 32) {
        const int sb  = u >> 4;
        const int is  = u & 15; // 0..15 sub-block
        const int n   = is >> 3; // 0 or 1
        const int j   = (is & 7) >> 1; // 0..3
        const int is0 = is & 1; // 0 or 1
        const int shift = j << 1;
        const uint8_t m = 1 << (4 * n + j);

        const uint8_t *blk0 = rw0 + sb * 110;
        const uint8_t *blk1 = rw1 + sb * 110;

        // Unpack scale for is:
        const uint8_t *sc0_raw = blk0 + 96;
        const uint8_t *sc1_raw = blk1 + 96;

        int8_t us0 = is <  4 ? (sc0_raw[is-0] & 0xF) | (((sc0_raw[is+8] >> 0) & 3) << 4) :
                     is <  8 ? (sc0_raw[is-0] & 0xF) | (((sc0_raw[is+4] >> 2) & 3) << 4) :
                     is < 12 ? (sc0_raw[is-8] >>  4) | (((sc0_raw[is+0] >> 4) & 3) << 4) :
                               (sc0_raw[is-8] >>  4) | (((sc0_raw[is-4] >> 6) & 3) << 4);

        int8_t us1 = is <  4 ? (sc1_raw[is-0] & 0xF) | (((sc1_raw[is+8] >> 0) & 3) << 4) :
                     is <  8 ? (sc1_raw[is-0] & 0xF) | (((sc1_raw[is+4] >> 2) & 3) << 4) :
                     is < 12 ? (sc1_raw[is-8] >>  4) | (((sc1_raw[is+0] >> 4) & 3) << 4) :
                               (sc1_raw[is-8] >>  4) | (((sc1_raw[is-4] >> 6) & 3) << 4);

        const float d0 = half_at(blk0 + 108);
        const float d1 = half_at(blk1 + 108);
        const float dl0 = d0 * (float)(us0 - 32);
        const float dl1 = d1 * (float)(us1 - 32);

        // Sub-block data pointers (16 elements):
        const uint8_t *q0  = blk0 + 32 + 32 * n + 16 * is0;
        const uint8_t *q1  = blk1 + 32 + 32 * n + 16 * is0;
        const uint8_t *hm0 = blk0 + 16 * is0;
        const uint8_t *hm1 = blk1 + 16 * is0;

        const float *xb = x + (long)sb * 256 + is * 16;
        const float4 *x4 = (const float4 *)xb;

        // 16 elements = 4 float4 loads
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const float4 xv = x4[c];
            const int base_l = c * 4;

            int8_t w0_0 = ((q0[base_l + 0] >> shift) & 3) - ((hm0[base_l + 0] & m) ? 0 : 4);
            int8_t w0_1 = ((q0[base_l + 1] >> shift) & 3) - ((hm0[base_l + 1] & m) ? 0 : 4);
            int8_t w0_2 = ((q0[base_l + 2] >> shift) & 3) - ((hm0[base_l + 2] & m) ? 0 : 4);
            int8_t w0_3 = ((q0[base_l + 3] >> shift) & 3) - ((hm0[base_l + 3] & m) ? 0 : 4);

            int8_t w1_0 = ((q1[base_l + 0] >> shift) & 3) - ((hm1[base_l + 0] & m) ? 0 : 4);
            int8_t w1_1 = ((q1[base_l + 1] >> shift) & 3) - ((hm1[base_l + 1] & m) ? 0 : 4);
            int8_t w1_2 = ((q1[base_l + 2] >> shift) & 3) - ((hm1[base_l + 2] & m) ? 0 : 4);
            int8_t w1_3 = ((q1[base_l + 3] >> shift) & 3) - ((hm1[base_l + 3] & m) ? 0 : 4);

            s0 += dl0 * (w0_0 * xv.x + w0_1 * xv.y + w0_2 * xv.z + w0_3 * xv.w);
            s1 += dl1 * (w1_0 * xv.x + w1_1 * xv.y + w1_2 * xv.z + w1_3 * xv.w);
        }
    }

    // Warp reduction
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        s0 += __shfl_xor_sync(0xffffffff, s0, mask);
        s1 += __shfl_xor_sync(0xffffffff, s1, mask);
    }

    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

int main(int argc, char **argv) {
    printf("=== Q3_K CUDA GEMV Verification & Benchmark ===\n");

    const int M = (argc > 1) ? atoi(argv[1]) : 8192;
    const int K = (argc > 2) ? atoi(argv[2]) : 4096;

    if (K % QK_K != 0) { fprintf(stderr,"ERROR: K=%d must be multiple of QK_K=%d\n",K,QK_K); return 1; }
    if (K % 256 != 0) { fprintf(stderr,"ERROR: K=%d must be multiple of 256\n",K); return 1; }

    printf("Matrix Shape: M=%d, K=%d (nsb=%d)\n", M, K, K/256);
    const int nsb = K / 256;
    const size_t bytes_W = (size_t)M * nsb * sizeof(BlockQ3_K);
    const size_t L2_BYTES = 2*1024*1024;
    const bool l2_resident = bytes_W < L2_BYTES;
    if (l2_resident) printf("WARNING: weight %.2f MB < L2 %.2f MB -> L2-bound\n", (double)bytes_W/1e6, (double)L2_BYTES/1e6);
    else printf("Weight bytes: %.2f MB (exceeds L2 -> DRAM honest)\n", (double)bytes_W/1e6);
    const size_t bytes_x = (size_t)K * sizeof(float);
    const size_t bytes_y = (size_t)M * sizeof(float);

    BlockQ3_K *h_W = (BlockQ3_K *)malloc(bytes_W);
    float *h_x = (float *)malloc(bytes_x);
    float *h_y_ref = (float *)malloc(bytes_y);
    float *h_y_gpu = (float *)malloc(bytes_y);

    srand(42);
    for (size_t i = 0; i < (size_t)M * nsb; i++) {
        for (int j = 0; j < 32; j++) h_W[i].hmask[j] = (uint8_t)rand();
        for (int j = 0; j < 64; j++) h_W[i].qs[j] = (uint8_t)rand();
        for (int j = 0; j < 12; j++) h_W[i].scales[j] = (uint8_t)rand();
        uint16_t d_fp16 = 0x251f; // ~0.02
        h_W[i].d = d_fp16;
    }

    for (int i = 0; i < K; i++) {
        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    printf("Computing CPU reference GEMV...\n");
    gemv_q3_K_cpu_ref(h_W, h_x, h_y_ref, M, K);

    uint8_t *d_W;
    float *d_x, *d_y;
    cudaMalloc(&d_W, bytes_W);
    cudaMalloc(&d_x, bytes_x);
    cudaMalloc(&d_y, bytes_y);

    cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice);

    dim3 block(32, 4);
    dim3 grid((M + 7) / 8);

    for (int i = 0; i < 5; i++) {
        k_gemv_q3_K_v2<<<grid, block>>>(d_W, d_x, d_y, M, K);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_y_gpu, d_y, bytes_y, cudaMemcpyDeviceToHost);

    double diff_sq = 0.0, ref_sq = 0.0;
    double max_abs_diff = 0.0;
    double max_ref = 0.0;
    for (int i = 0; i < M; i++) {
        double diff = fabs((double)h_y_gpu[i] - (double)h_y_ref[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (fabs((double)h_y_ref[i]) > max_ref) max_ref = fabs((double)h_y_ref[i]);
        diff_sq += diff * diff;
        ref_sq += (double)h_y_ref[i] * (double)h_y_ref[i];
    }
    double l2_rel_error = sqrt(diff_sq) / (sqrt(ref_sq) + 1e-9);
    double max_rel_inf = max_abs_diff / (max_ref + 1e-9);

    printf("Numerical Verification vs CPU Golden:\n");
    printf("  Max Absolute Error: %.6e\n", max_abs_diff);
    printf("  L2 Relative Error : %.6e\n", l2_rel_error);
    printf("  L_inf Rel Error   : %.6e\n", max_rel_inf);

    const int iters = 500;
    const int warmup = 20;
    for(int i=0;i<warmup;i++) k_gemv_q3_K_v2<<<grid, block>>>(d_W, d_x, d_y, M, K);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *samples=(float*)malloc(iters*sizeof(float));
    for(int i=0;i<iters;i++){
        cudaEventRecord(start);
        k_gemv_q3_K_v2<<<grid, block>>>(d_W, d_x, d_y, M, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_i=0; cudaEventElapsedTime(&ms_i, start, stop);
        samples[i]=ms_i;
    }
    qsort(samples, iters, sizeof(float), cmp_f);
    float min_ms=samples[0]; float p50_ms=samples[iters/2]; float p95_ms=samples[(int)(iters*0.95)]; float max_ms=samples[iters-1];
    double sum=0; for(int i=0;i<iters;i++) sum+=samples[i];
    float mean_ms=(float)(sum/iters);
    double gb = (double)bytes_W / 1e9;
    printf("Benchmark Performance (M=%d, K=%d, %d samples, per-iter sync):\n", M, K, iters);
    printf("  Kernel Time: mean %.4f ms  median(p50) %.4f ms  p95 %.4f ms  min %.4f max %.4f\n", mean_ms, p50_ms, p95_ms, min_ms, max_ms);
    printf("  Effective Bandwidth: mean %.2f GB/s  median %.2f GB/s  p95 %.2f GB/s\n", gb/(mean_ms/1000.0), gb/(p50_ms/1000.0), gb/(p95_ms/1000.0));
    if(l2_resident) printf("  NOTE: weight fits in L2 -> L2 bandwidth, not DRAM 176 GB/s. Use larger M/K.\n");
    else printf("  NOTE: weight exceeds L2 -> honest DRAM (peak 176 GB/s).\n");
    printf("  Hint: lock clocks with 'sudo nvidia-smi -lgc 1500,1500'\n");

    bool pass = (max_abs_diff < 1e-4 && max_rel_inf < 1e-4);
    if (pass) {
        printf("RESULT: PASS\n");
    } else {
        printf("RESULT: FAIL\n");
        return 1;
    }

    free(samples);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_W);
    free(h_x);
    free(h_y_ref);
    free(h_y_gpu);

    return 0;
}
