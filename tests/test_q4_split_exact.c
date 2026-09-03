#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "qwen2_engine.h"

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

static float frand(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

/* minimal half->float, bit-exact for these magnitudes */
static float half2float(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}

int main(void) {
    int HDs[] = {64, 128};
    int Ns[] = {5, 13, 70};
    int n_heads = 14, n_kv_heads = 2;
    float window = 0;
    int fails = 0;

    for (int hi = 0; hi < 2; hi++) {
        int head_dim = HDs[hi];
        float scale = 1.0f / sqrtf((float)head_dim);
        size_t q_size = (size_t)n_heads * head_dim * sizeof(float);
        size_t kv_dim = (size_t)n_kv_heads * head_dim;
        int max_ctx = 256;
        int max_S = 4;

        float *h_q = malloc(q_size);
        float *h_out = malloc(q_size);
        float *h_kst = malloc(kv_dim * sizeof(float));
        float *h_vst = malloc(kv_dim * sizeof(float));
        srand(100 + head_dim);
        for (size_t i = 0; i < (size_t)n_heads * head_dim; i++) h_q[i] = frand();

        float *d_q, *d_out, *d_kst, *d_vst, *d_Kf, *d_Vf;
        void *d_Kq, *d_Vq;
        float *d_pacc, *d_pm, *d_pl;
        int *d_pos;
        CK(cudaMalloc(&d_q, q_size));
        CK(cudaMalloc(&d_out, q_size));
        CK(cudaMalloc(&d_kst, kv_dim * sizeof(float)));
        CK(cudaMalloc(&d_vst, kv_dim * sizeof(float)));
        CK(cudaMalloc(&d_Kf, (size_t)max_ctx * kv_dim * sizeof(float)));
        CK(cudaMalloc(&d_Vf, (size_t)max_ctx * kv_dim * sizeof(float)));
        size_t nb = kv_dim / 32;
        CK(cudaMalloc(&d_Kq, (size_t)max_ctx * nb * 18));
        CK(cudaMalloc(&d_Vq, (size_t)max_ctx * nb * 18));
        CK(cudaMalloc(&d_pacc, (size_t)max_S * n_heads * head_dim * sizeof(float)));
        CK(cudaMalloc(&d_pm, (size_t)max_S * n_heads * sizeof(float)));
        CK(cudaMalloc(&d_pl, (size_t)max_S * n_heads * sizeof(float)));
        CK(cudaMalloc(&d_pos, sizeof(int)));
        CK(cudaMemcpy(d_q, h_q, q_size, cudaMemcpyHostToDevice));

        for (int ni = 0; ni < 3; ni++) {
            int N = Ns[ni];
            CK(cudaMemset(d_Kf, 0, (size_t)max_ctx * kv_dim * sizeof(float)));
            CK(cudaMemset(d_Vf, 0, (size_t)max_ctx * kv_dim * sizeof(float)));
            CK(cudaMemset(d_Kq, 0, (size_t)max_ctx * nb * 18));
            CK(cudaMemset(d_Vq, 0, (size_t)max_ctx * nb * 18));
            for (int t = 0; t < N; t++) {
                for (size_t i = 0; i < kv_dim; i++) {
                    h_kst[i] = frand();
                    h_vst[i] = frand();
                }
                CK(cudaMemcpy(d_kst, h_kst, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_vst, h_vst, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_pos, &t, sizeof(int), cudaMemcpyHostToDevice));
                tt_kv_scatter(d_kst, d_vst, d_Kf, d_Vf, d_pos, n_kv_heads, head_dim, max_ctx, 0);
                tt_kv_scatter_q4_0(d_kst, d_vst, d_Kq, d_Vq, d_pos, n_kv_heads, head_dim, max_ctx, 0);
            }
            CK(cudaDeviceSynchronize());
            int cur = N - 1;
            CK(cudaMemcpy(d_pos, &cur, sizeof(int), cudaMemcpyHostToDevice));

            /* read back Q4 cache, dequant on host, fp64 exact reference */
            uint8_t *h_Kq = malloc((size_t)max_ctx * nb * 18);
            uint8_t *h_Vq = malloc((size_t)max_ctx * nb * 18);
            float *h_Kf = malloc((size_t)max_ctx * kv_dim * sizeof(float));
            float *h_Vf = malloc((size_t)max_ctx * kv_dim * sizeof(float));
            CK(cudaMemcpy(h_Kq, d_Kq, (size_t)max_ctx * nb * 18, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_Vq, d_Vq, (size_t)max_ctx * nb * 18, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_Kf, d_Kf, (size_t)max_ctx * kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_Vf, d_Vf, (size_t)max_ctx * kv_dim * sizeof(float), cudaMemcpyDeviceToHost));

            double maxerr_split = 0, maxerr_qnoise = 0;
            for (int S = 1; S <= 4; S *= 2) {
                CK(cudaMemset(d_out, 0, q_size));
                tt_flash_gqa_q4_0_splitk(d_q, d_Kq, d_Vq, d_pacc, d_pm, d_pl, d_out,
                                         d_pos, n_heads, n_kv_heads, head_dim,
                                         scale, window, S, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(h_out, d_out, q_size, cudaMemcpyDeviceToHost));

                /* host fp64 reference from dequantized Q4 blocks */
                for (int h = 0; h < n_heads; h++) {
                    int kv = h / (n_heads / n_kv_heads);
                    for (int d = 0; d < head_dim; d++) {
                        double m = -1e100, l = 0;
                        double acc = 0;
                        /* first pass: max */
                        double *scores = malloc(N * sizeof(double));
                        for (int t = 0; t < N; t++) {
                            double dot = 0;
                            for (int dd = 0; dd < head_dim; dd++) {
                                size_t blk = ((size_t)kv * head_dim + dd) / 32;
                                int idx = ((size_t)kv * head_dim + dd) % 32;
                                uint8_t *bp = h_Kq + ((size_t)t * nb + blk) * 18;
                                uint16_t dh;
                                memcpy(&dh, bp, 2);
                                float dk = half2float(dh);
                                (void)dk;
                                int j = idx < 16 ? idx : idx - 16;
                                int byte = bp[2 + j];
                                int nib = (idx < 16) ? (byte & 0xF) : (byte >> 4);
                                double kval = (double)((nib - 8)) * (double)half2float(dh);
                                dot += (double)h_q[(size_t)h * head_dim + dd] * kval;
                            }
                            scores[t] = dot * (double)scale;
                            if (scores[t] > m) m = scores[t];
                        }
                        for (int t = 0; t < N; t++) {
                            double p = exp(scores[t] - m);
                            l += p;
                            double v = 0;
                            {
                                size_t blk = ((size_t)kv * head_dim + d) / 32;
                                int idx = ((size_t)kv * head_dim + d) % 32;
                                uint8_t *bp = h_Vq + ((size_t)t * nb + blk) * 18;
                                uint16_t dh;
                                memcpy(&dh, bp, 2);
                                int j = idx < 16 ? idx : idx - 16;
                                int byte = bp[2 + j];
                                int nib = (idx < 16) ? (byte & 0xF) : (byte >> 4);
                                v = (double)((nib - 8)) * (double)half2float(dh);
                            }
                            acc += p * v;
                        }
                        free(scores);
                        double ref = acc / l;
                        double got = (double)h_out[(size_t)h * head_dim + d];
                        double e = fabs(got - ref);
                        if (e > maxerr_split) maxerr_split = e;
                    }
                }
                printf("HD=%3d N=%3d S=%d | kernel-vs-exactQ4 maxerr=%.3e %s\n",
                       head_dim, N, S, maxerr_split, maxerr_split < 2e-5 ? "PASS" : "FAIL");
                if (maxerr_split >= 2e-5) fails++;
            }
            free(h_Kq);
            free(h_Vq);
            free(h_Kf);
            free(h_Vf);
        }
        CK(cudaFree(d_q));
        CK(cudaFree(d_out));
        CK(cudaFree(d_kst));
        CK(cudaFree(d_vst));
        CK(cudaFree(d_Kf));
        CK(cudaFree(d_Vf));
        CK(cudaFree(d_Kq));
        CK(cudaFree(d_Vq));
        CK(cudaFree(d_pacc));
        CK(cudaFree(d_pm));
        CK(cudaFree(d_pl));
        CK(cudaFree(d_pos));
        free(h_q);
        free(h_out);
        free(h_kst);
        free(h_vst);
    }
    printf(fails ? "EXACT-REF: FAILURES=%d\n" : "EXACT-REF: ALL PASS\n", fails);
    return fails ? 1 : 0;
}

#define __half2float_original half2float
