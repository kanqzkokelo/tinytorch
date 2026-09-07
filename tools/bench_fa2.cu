// Standalone FA2 prefill flash microbench: kernel-only ms + CPU reference check.
// Build: nvcc -O3 -gencode arch=compute_86,code=sm_86 -o /tmp/bench_fa2 tools/bench_fa2.cu -L$HOME/mmcuda/lib -lcudart
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define FA2_BN 32
#define FA2_GBM 16
#define FA2_GMAX 8

// ---- begin kernel copy (must match kernels/qwen2_cuda.cu k_fa2_gqa64) ----
#define FA2_BN 32
#define FA2_GBM 16
#define FA2_GMAX 8
#define FA2_SBN_S 36
#define FA2_SBN_P 40
/* TT_FA2_PRE grouped variant (HD=64): CTA per (16-row Q-tile, kv-head),
 * one warp per q-head in the group (G<=8). K/V tiles loaded ONCE per CTA
 * and shared by all G warps. O in per-lane registers (32 floats); P@V
 * done chunk-wise (16 cols) through the S scratch. smem ~43KB @G=7. */
#define FA2_GBM 16
#define FA2_GMAX 8
/* S/P scratch columns padded 32->36/40: row stride 144B/80B (16B-aligned)
 * spreads the 16 warp rows over banks (2-way vs 16-way conflict unpadded).
 * P (half) needs ldm%8==0 for wmma alignment -> 40. */
#define FA2_SBN_S 36
#define FA2_SBN_P 40
__global__ __launch_bounds__(256) void k_fa2_gqa64(
    const float *__restrict__ Q, const float *__restrict__ Kc,
    const float *__restrict__ Vc, float *__restrict__ Att,
    int n, int ctx, int e_pos, int n_heads, int n_kv_heads,
    float scale, int window)
{
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (warp >= G) return;
    const int kv = blockIdx.y;
    const int head = kv * G + warp;
    const int q0 = blockIdx.x * FA2_GBM;
    extern __shared__ half gsmem[];
    half *sQ  = gsmem;                                        /* [G][16][64] */
    half *sKT = sQ + (size_t)G * FA2_GBM * 64;                /* [64][32] */
    half *sV  = sKT + (size_t)64 * FA2_BN;                    /* [32][64] */
    float *Sbase = (float *)(sV + (size_t)FA2_BN * 64);
    float *S = Sbase + (size_t)warp * 16 * FA2_SBN_S;
    half *P = (half *)(Sbase + (size_t)G * 16 * FA2_SBN_S)
            + (size_t)warp * 16 * FA2_SBN_P;
    {
        const int perQ = G * FA2_GBM * 64;
        for (int i = tid; i < perQ; i += blockDim.x) {
            const int w = i / (FA2_GBM * 64), rr = (i / 64) % FA2_GBM, d = i % 64;
            const int qrow = q0 + rr;
            float v = 0.0f;
            if (qrow < n)
                v = Q[(long)qrow * (n_heads * 64) + (long)(kv * G + w) * 64 + d];
            sQ[i] = __float2half(v);
        }
    }
    __syncthreads();
    const int row = lane & 15;
    const int qrow = q0 + row;
    const bool valid = (qrow < n);
    const int max_kv = valid ? (e_pos + qrow) : -1;
    const int row_min = (window > 0 && valid && (e_pos + qrow + 1 > window))
        ? (e_pos + qrow + 1 - window) : 0;
    float m = -1e30f, l = 0.0f;
    /* Per-lane O tile (own 32-col half only): const-indexed so it stays
     * in registers (dynamic indexing would spill to local memory). */
    float Oreg[32];
    for (int d = 0; d < 32; d++) Oreg[d] = 0.0f;
    const int half0 = (lane >> 4) * 32;
    const int myHalf = (lane >> 4);
    const int qMax = e_pos + (q0 + FA2_GBM - 1 < n ? q0 + FA2_GBM - 1 : n - 1);
    const int rowMinGlob = (window > 0) ? (e_pos + q0 + 1 - window) : 0;
    const float4 *Kc4 = reinterpret_cast<const float4*>(Kc);
    const float4 *Vc4 = reinterpret_cast<const float4*>(Vc);
    const int hd4 = 16;
    half *sQw = sQ + (size_t)warp * FA2_GBM * 64;
    using namespace nvcuda;
    for (int ks = 0; ks < ctx; ks += FA2_BN) {
        if (ks > qMax) break;
        const int ke = (ks + FA2_BN < ctx) ? ks + FA2_BN : ctx;
        if (ke <= rowMinGlob) continue;
        const int kc = ke - ks;
        {
            const int perKV = FA2_BN * hd4;
            for (int i = tid; i < perKV; i += blockDim.x) {
                const int k = i / hd4, d4 = i % hd4;
                const int gt = ks + k;
                float4 kv4 = make_float4(0, 0, 0, 0), vv4 = make_float4(0, 0, 0, 0);
                if (k < kc) {
                    const long g = ((long)gt * n_kv_heads + kv) * hd4 + d4;
                    kv4 = Kc4[g]; vv4 = Vc4[g];
                }
                const float *kf = (const float *)&kv4, *vf = (const float *)&vv4;
                for (int e = 0; e < 4; e++) {
                    sKT[(d4 * 4 + e) * FA2_BN + k] = __float2half(kf[e]);
                    sV[k * 64 + d4 * 4 + e] = __float2half(vf[e]);
                }
            }
        }
        __syncthreads();
        for (int nt = 0; nt < FA2_BN / 16; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < 4; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, sQw + (size_t)kk * 16, 64);
                wmma::load_matrix_sync(fb, sKT + (size_t)kk * 16 * FA2_BN + (size_t)nt * 16, FA2_BN);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(S + (size_t)nt * 16, acc, FA2_SBN_S, wmma::mem_row_major);
        }
        __syncwarp();
        /* Softmax: lanes<16 own the row math (halves expf + S traffic);
         * upper lanes take a/bsum/m via shuffle, rescale own O half. */
        {
            float *Srow = S + (size_t)row * FA2_SBN_S;
            half *Prow = P + (size_t)row * FA2_SBN_P;
            float bmax = -1e30f, bsum = 0.0f, a = 1.0f, m_new = m;
            if (lane < 16) {
                for (int k = 0; k < kc; k++) {
                    const int gt = ks + k;
                    float sc = -1e30f;
                    if (valid && gt >= row_min && gt <= max_kv) sc = Srow[k] * scale;
                    if (sc > bmax) bmax = sc;
                }
                m_new = fmaxf(m, bmax);
                a = expf(m - m_new);
                const float beta = expf(bmax - m_new);
                for (int k = 0; k < kc; k++) {
                    const int gt = ks + k;
                    float p = 0.0f;
                    if (valid && gt >= row_min && gt <= max_kv) {
                        p = expf(Srow[k] * scale - bmax) * beta;
                        /* Self-consistent l: accumulate the ROUNDED weight
                         * actually staged to P (half flushes tiny p to 0;
                         * counting unrounded p in l biases O/l downward). */
                        p = __half2float(__float2half(p));
                        bsum += p;
                    }
                    Prow[k] = __float2half(p);
                }
                for (int k = kc; k < FA2_BN; k++) Prow[k] = __float2half(0.0f);
            }
            const float a_all = __shfl_sync(0xffffffff, a, lane & 15);
            const float bsum_all = __shfl_sync(0xffffffff, bsum, lane & 15);
            m = __shfl_sync(0xffffffff, m_new, lane & 15);
            for (int d = 0; d < 32; d++) Oreg[d] *= a_all;
            l = l * a_all + bsum_all;
        }
        __syncwarp();
        /* P@V in two 32-col halves through the S scratch (16x32 fp32).
         * T-aliasing sQ was sized wrong (1024 floats/warp vs 512 free). */
        for (int h = 0; h < 2; h++) {
            for (int nt = 0; nt < 2; nt++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int kk = 0; kk < FA2_BN / 16; kk++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fa, P + (size_t)kk * 16, FA2_SBN_P);
                    wmma::load_matrix_sync(fb, sV + (size_t)kk * 16 * 64 + (size_t)(h * 32 + nt * 16), 64);
                    wmma::mma_sync(acc, fa, fb, acc);
                }
                wmma::store_matrix_sync(S + (size_t)nt * 16, acc, FA2_SBN_S, wmma::mem_row_major);
            }
            __syncwarp();
            if (myHalf == h) {
                float *Srow = S + (size_t)row * FA2_SBN_S;
                for (int c = 0; c < 32; c++) Oreg[c] += Srow[c];
            }
            __syncwarp();
        }
        __syncwarp();
        __syncthreads();
    }
    if (valid && l > 0.0f) {
        const float inv = 1.0f / l;
        float *out = Att + (long)qrow * (n_heads * 64) + (long)head * 64;
        for (int d = 0; d < 32; d++) out[half0 + d] = Oreg[d] * inv;
    }
}

// ---- end kernel copy ----

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 759;
    int H = 14, KV = 2, HD = 64, ctx = n, epos = 0, window = 0;
    float scale = 1.0f / sqrtf(64.0f);
    size_t nQ = (size_t)n * H * HD, nKV = (size_t)ctx * KV * HD;
    std::vector<float> hQ(nQ), hK(nKV), hV(nKV), hO(nQ), hRef(nQ);
    srand(42);
    for (size_t i = 0; i < nQ; i++) hQ[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    for (size_t i = 0; i < nKV; i++) {
        hK[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        hV[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    }
    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, nQ * 4); cudaMalloc(&dK, nKV * 4);
    cudaMalloc(&dV, nKV * 4); cudaMalloc(&dO, nQ * 4);
    cudaMemcpy(dQ, hQ.data(), nQ * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), nKV * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), nKV * 4, cudaMemcpyHostToDevice);
    int G = H / KV;
    dim3 grid((n + FA2_GBM - 1) / FA2_GBM, KV);
    int threads = G * 32;
    size_t smem = ((size_t)G * FA2_GBM * 64 + 64 * FA2_BN + (size_t)FA2_BN * 64) * sizeof(half)
                + (size_t)G * 16 * FA2_SBN_S * sizeof(float)
                + (size_t)G * 16 * FA2_SBN_P * sizeof(half);
    printf("grid=(%d,%d) threads=%d smem=%zu\n", grid.x, grid.y, threads, smem);
    for (int i = 0; i < 3; i++) {
        k_fa2_gqa64<<<grid, threads, smem, 0>>>(dQ, dK, dV, dO, n, ctx, epos, H, KV, scale, window);
    }
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 20; i++) {
        k_fa2_gqa64<<<grid, threads, smem, 0>>>(dQ, dK, dV, dO, n, ctx, epos, H, KV, scale, window);
    }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    printf("kernel-only: %.3f ms/launch (n=%d)\n", ms / 20, n);
    cudaMemcpy(hO.data(), dO, nQ * 4, cudaMemcpyDeviceToHost);
    // CPU reference (double)
    double maxd = 0;
    for (int h = 0; h < H; h++) {
        int kv = h / G;
        for (int q = 0; q < n; q++) {
            double m = -1e30, l = 0; double acc[64] = {0};
            for (int t = 0; t <= q; t++) {
                double s = 0;
                for (int d = 0; d < 64; d++) s += (double)hQ[(q * H + h) * 64 + d] * hK[(t * KV + kv) * 64 + d];
                s *= scale;
                double mn = m > s ? m : s;
                double al = exp(m - mn);
                l = l * al + exp(s - mn);
                for (int d = 0; d < 64; d++) acc[d] = acc[d] * al + exp(s - mn) * hV[(t * KV + kv) * 64 + d];
                m = mn;
            }
            for (int d = 0; d < 64; d++) {
                double ref = acc[d] / l;
                double got = hO[(q * H + h) * 64 + d];
                double dd = fabs(ref - got);
                if (dd > maxd) maxd = dd;
            }
        }
    }
    printf("maxdiff vs FP64: %g\n", maxd);
    return 0;
}
