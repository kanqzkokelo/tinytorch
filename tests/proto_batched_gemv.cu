// PROTOTYPE (create-only study): batched q4_0 GEMV -> GEMM for prompt prefill.
//
// Problem: engine runs tt_gemv_typed one token per launch; prefill of N tokens
// costs N sequential weight streams. llama.cpp batches as mmvq/GEMM. This file
// measures when batching wins on our hardware.
//
//   W[M,K] q4_0 (18B blocks: fp16 d + 16B nibbles, v = (q-8)*d)
//   X[K,T] row-major, Y[M,T] row-major
//   V1: simple tiling - one warp per output row x all T cols, lanes stride K,
//       acc[T] per lane, shfl reduce. Naive baseline.
//   V2: block tiles rows (blockDim.y warps); T<=32: column per lane + K-split
//       across lane groups (xor-reduce); T>32: CP=T/32 columns per lane with
//       float4 x loads. Weight streamed exactly ONCE for any T.
//
// Build:
//   $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 -o proto_batched_gemv \
//       tests/proto_batched_gemv.cu
// Run:
//   LD_LIBRARY_PATH=$HOME/mmcuda/lib ./proto_batched_gemv
// Profile (optional):
//   ncu --set full ./proto_batched_gemv
//
// Baseline for comparison: measured single-token GEMV era ~213 tok/s on the
// 1536-wide layers ~= effective ~330 GB/s weight stream rate (BASE_GB_S).
//
// Self-contained: no repo headers. CPU double reference checks T=8 at
// rel < 1e-2 (quantization tolerance).
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BASE_GB_S 330.0          // single-token effective GB/s (measured era)
#define MAX_T 128
#define MAX_ROWS 262144

/* ---------------- fp16 bit conversion (host+device, no deps) --------------- */
static inline __host__ __device__ float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {                       // subnormal / zero
        if (!man) { bits = sign; }
        else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400));
            m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {             // inf / nan
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}
static inline uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  e    = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00);
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00);
    if (e <= 0) return (uint16_t)sign;    // scales are normal-range; no need more
    return (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
}

static __device__ __forceinline__ float warp_red(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

/* Load one q4_0 block via aligned u32 window + __byte_perm.
 * Row byte offset o=18b: b even -> scale in low half of word o>>2, qs payload
 * straddles words; b odd -> scale in high half, qs 16B-aligned. Either way 4-5
 * aligned word loads replace 18 scalar byte loads (the #1 bandwidth fix;
 * same reasoning as kernels/gemv_q4_cuda.cu V2). Reads may touch one word
 * past the row end -> caller allocates 16B slack.
 * Nibble j (0..15): lo=q[j>>2] byte j&3 low nibble; hi nibble = value j+16. */
static __device__ __forceinline__ void load_blk(const uint32_t *__restrict__ rw32,
                                                int b, float &d, uint32_t q[4]) {
    const int o = b * 18;
    const int w = o >> 2;
    uint32_t w0 = rw32[w];
    if (o & 2) {
        d = h2f((uint16_t)(w0 >> 16));
        q[0] = rw32[w + 1]; q[1] = rw32[w + 2];
        q[2] = rw32[w + 3]; q[3] = rw32[w + 4];
    } else {
        d = h2f((uint16_t)(w0 & 0xFFFFu));
        uint32_t w1 = rw32[w + 1], w2 = rw32[w + 2];
        uint32_t w3 = rw32[w + 3], w4 = rw32[w + 4];
        q[0] = __byte_perm(w0, w1, 0x5432);
        q[1] = __byte_perm(w1, w2, 0x5432);
        q[2] = __byte_perm(w2, w3, 0x5432);
        q[3] = __byte_perm(w3, w4, 0x5432);
    }
}
static __device__ __forceinline__ int nib(const uint32_t q[4], int j) {
    return (int)(q[j >> 2] >> ((j & 3) * 8)) & 0xF;
}
static __device__ __forceinline__ int nib_hi(const uint32_t q[4], int j) {
    return (int)(q[j >> 2] >> ((j & 3) * 8 + 4)) & 0xF;
}

/* Warp-cooperative W streaming: 32 consecutive q4_0 blocks = 576B = 144 u32.
 * Each lane loads 4-5 CONSECUTIVE words (perfect 128B transactions), then
 * reassembles its own block b0+lane from neighbours via __shfl_sync.
 * Naive lane-strides-blocks mapping makes every load touch 32 scattered
 * sectors -> ~9x DRAM amplification (measured 17 GB/s ceiling); this restores
 * full-line coalescing. Returns false if block >= nb (q,d undefined).
 * Caller guarantees 16B slack past row end. */
static __device__ __forceinline__ bool gather_block(const uint32_t *__restrict__ rw32,
                                                   int b0, int nb, int lane,
                                                   float &d, uint32_t q[4]) {
    const int nblk = min(32, nb - b0);
    if (nblk <= 0) return false;
    const int nwords = (nblk * 18 + 3) >> 2;      // words covering valid blocks
    uint32_t wr[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const int wi = i * 32 + lane;
        wr[i] = (wi < nwords + 1) ? rw32[(b0 * 18 >> 2) + wi] : 0u;
    }
    /* shuffles are unconditional: ALL lanes must converge (full mask),
     * even those whose block b0+lane is past nb */
    const int b = b0 + lane;
    const int o = 18 * (b - b0);                  // local byte offset 0..574
    const int w = o >> 2;
    uint32_t ww[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const int wi = w + i;                     // 0..144
        ww[i] = __shfl_sync(0xffffffffu, wr[wi >> 5], wi & 31);
    }
    if (b >= nb) return false;                    // now safe to diverge
    if (o & 2) {
        d = h2f((uint16_t)(ww[0] >> 16));
        q[0] = ww[1]; q[1] = ww[2]; q[2] = ww[3]; q[3] = ww[4];
    } else {
        d = h2f((uint16_t)(ww[0] & 0xFFFFu));
        q[0] = __byte_perm(ww[0], ww[1], 0x5432);
        q[1] = __byte_perm(ww[1], ww[2], 0x5432);
        q[2] = __byte_perm(ww[2], ww[3], 0x5432);
        q[3] = __byte_perm(ww[3], ww[4], 0x5432);
    }
    return true;
}

/* ---------------- V1: warp per row, acc[T] per lane ------------------------ */
template<int T_>
__global__ void k_v1(const uint8_t *__restrict__ W, const float *__restrict__ X,
                     float *__restrict__ Y, int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb = K >> 5;
    const uint32_t *rw = (const uint32_t *)(W + (size_t)row * nb * 18);

    float acc[T_];
#pragma unroll
    for (int t = 0; t < T_; ++t) acc[t] = 0.f;

    for (int b0 = 0; b0 < nb; b0 += 32) {
        float d; uint32_t q[4];
        const bool have = gather_block(rw, b0, nb, lane, d, q);
        /* no break here: loop trip count must stay warp-uniform */
        if (have) {
        const float *xb = X + (size_t)(b0 + lane) * 32 * T_;
#pragma unroll
        for (int t = 0; t < T_; ++t) {
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                s += (nib(q, j) - 8) * xb[(size_t)j * T_ + t];
                s += (nib_hi(q, j) - 8) * xb[(size_t)(j + 16) * T_ + t];
            }
            acc[t] += s * d;
        }
        }
    }
#pragma unroll
    for (int t = 0; t < T_; ++t) {
        float v = warp_red(acc[t]);
        if (!lane) Y[(size_t)row * T_ + t] = v;
    }
}

/* ---------------- V2: rows across blockDim.y, T split across lanes ---------
 * T<=32 : one column per lane (col=lane%T), K implicitly split across the
 *         lane groups covering that column; xor-reduce at the end.
 * T> 32 : CP=T/32 columns per lane (float4-able), no reduction needed.
 * Both paths stream W through gather_block (coalesced) and read X directly
 * (small enough to stay L2/L1 resident). */
template<int T_>
__global__ void k_v2(const uint8_t *__restrict__ W, const float *__restrict__ X,
                     float *__restrict__ Y, int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb = K >> 5;
    const uint32_t *rw = (const uint32_t *)(W + (size_t)row * nb * 18);

    constexpr int CP = (T_ <= 32) ? 1 : T_ / 32;     // columns per lane
    float acc[CP];
#pragma unroll
    for (int c = 0; c < CP; ++c) acc[c] = 0.f;
    const int col0 = (T_ <= 32) ? (lane % T_) : (lane * CP);

    for (int b0 = 0; b0 < nb; b0 += 32) {
        float d; uint32_t q[4];
        const bool have = gather_block(rw, b0, nb, lane, d, q);
        if (have) {
            const float *xb = X + (size_t)(b0 + lane) * 32 * T_ + col0;
            float s[CP];
#pragma unroll
            for (int c = 0; c < CP; ++c) s[c] = 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float vlo = (float)(nib(q, j) - 8);
                const float vhi = (float)(nib_hi(q, j) - 8);
#pragma unroll
                for (int c = 0; c < CP; ++c) {
                    s[c] += vlo * xb[(size_t)j * T_ + c];
                    s[c] += vhi * xb[(size_t)(j + 16) * T_ + c];
                }
            }
#pragma unroll
            for (int c = 0; c < CP; ++c) acc[c] += s[c] * d;
        }
    }

    if constexpr (T_ <= 32) {
        /* column c was covered by lanes {c, c+T, ...}: reduce across them */
        constexpr int KSP = 32 / T_;
        const int kp = lane / T_;
#pragma unroll
        for (int s2 = KSP / 2; s2 > 0; s2 >>= 1)
            acc[0] += __shfl_xor_sync(0xffffffff, acc[0], s2 * T_);
        if (kp == 0) Y[(size_t)row * T_ + col0] = acc[0];
    } else {
#pragma unroll
        for (int c = 0; c < CP; ++c)
            Y[(size_t)row * T_ + col0 + c] = acc[c];
    }
}

/* ---------------- host helpers -------------------------------------------- */
static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return (uint32_t)(rng_s >> 32);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_) { \
    fprintf(stderr, "CUDA error %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

typedef void (*launch_fn)(const uint8_t *, const float *, float *, int, int, dim3, dim3, cudaStream_t);

template<int T_> static void launch_v1(const uint8_t *W, const float *X, float *Y,
                                       int M, int K, dim3 g, dim3 b, cudaStream_t s) {
    k_v1<T_><<<g, b, 0, s>>>(W, X, Y, M, K);
}
template<int T_> static void launch_v2(const uint8_t *W, const float *X, float *Y,
                                       int M, int K, dim3 g, dim3 b, cudaStream_t s) {
    k_v2<T_><<<g, b, 0, s>>>(W, X, Y, M, K);
}

static launch_fn pick(int v, int T) {
    if (v == 1) switch (T) {
        case 1: return launch_v1<1>; case 8: return launch_v1<8>;
        case 32: return launch_v1<32>; case 128: return launch_v1<128>; }
    else        switch (T) {
        case 1: return launch_v2<1>; case 8: return launch_v2<8>;
        case 32: return launch_v2<32>; case 128: return launch_v2<128>; }
    return NULL;
}

static const int WARPS_PER_BLOCK = 8;      // blockDim.y for both kernels

static double bench(launch_fn fn, const uint8_t *dW, const float *dX, float *dY,
                    int M, int K, int T, int iters) {
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    for (int w = 0; w < 3; ++w) fn(dW, dX, dY, M, K, grid, block, 0);
    CK(cudaGetLastError());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) fn(dW, dX, dY, M, K, grid, block, 0);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return (double)ms / iters;
}

static void occ_report(const char *name, const void *fn, int T) {
    cudaFuncAttributes a;
    int blocks;
    cudaError_t e1 = cudaFuncGetAttributes(&a, fn);
    cudaError_t e2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, fn, 32 * WARPS_PER_BLOCK, 0);
    if (e1 || e2) { printf("  %-14s (T=%3d): attr failed\n", name, T); return; }
    printf("  %-14s T=%3d: regs=%2d local=%zu smem=%zu occ=%d blocks/SM (%d warps)\n",
           name, T, a.numRegs, a.localSizeBytes, a.sharedSizeBytes,
           blocks, blocks * WARPS_PER_BLOCK);
}

int main(void) {
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("device: %s, %d SMs, L2 %d MB, %.1f GB/s peak\n\n", p.name,
           p.multiProcessorCount, (int)(p.l2CacheSize >> 20),
           p.memoryClockRate * 2.0 * (p.memoryBusWidth / 8) * 1e-6);

    /* ---- weights: max footprint generated once, shapes use prefix rows ---- */
    const int K = 1536;
    const int nb = K / 32;                       // 48 blocks/row
    const size_t rowbytes = (size_t)nb * 18;     // 864 B/row
    uint8_t *hW = (uint8_t *)malloc(((size_t)MAX_ROWS + 1) * rowbytes);
    for (size_t r = 0; r < (size_t)MAX_ROWS; ++r) {
        uint8_t *row = hW + r * rowbytes;
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = row + (size_t)b * 18;
            float d = 0.25f + 0.5f * ((rnd() >> 8) / 16777216.0f);
            uint16_t d16 = f2h(d);
            blk[0] = d16 & 0xFF; blk[1] = d16 >> 8;
            for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t)rnd();
        }
    }
    float *hXm = (float *)malloc((size_t)K * MAX_T * 4);   // master X [K][MAX_T]
    for (size_t i = 0; i < (size_t)K * MAX_T; ++i)
        hXm[i] = ((double)rnd() / 4294967295.0) * 2.0 - 1.0;

    uint8_t *dW; CK(cudaMalloc(&dW, ((size_t)MAX_ROWS + 1) * rowbytes));
    CK(cudaMemcpy(dW, hW, (size_t)MAX_ROWS * rowbytes, cudaMemcpyHostToDevice));

    /* ================= correctness: T=8 vs CPU double reference ============ */
    {
        const int M = 512, T = 8;
        float *hX = (float *)malloc((size_t)K * T * 4);
        for (int k = 0; k < K; ++k)
            for (int t = 0; t < T; ++t) hX[k * T + t] = hXm[k * MAX_T + t];
        double *ref = (double *)malloc((size_t)M * T * sizeof(double));
        memset(ref, 0, (size_t)M * T * sizeof(double));
        for (int m = 0; m < M; ++m) {
            const uint8_t *rw = hW + (size_t)m * rowbytes;
            double *yr = ref + (size_t)m * T;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = rw + (size_t)b * 18;
                double d = h2f((uint16_t)(blk[0] | (blk[1] << 8)));
                const uint8_t *qs = blk + 2;
                for (int j = 0; j < 16; ++j) {
                    double lo = ((int)(qs[j] & 0xF) - 8) * d;
                    double hi = ((int)(qs[j] >> 4) - 8) * d;
                    for (int t = 0; t < T; ++t) {
                        yr[t] += lo * (double)hX[(b * 32 + j) * T + t];
                        yr[t] += hi * (double)hX[(b * 32 + j + 16) * T + t];
                    }
                }
            }
        }

        struct { const char *n; int v; int tt; } cases[] = {
            {"V1 T=8", 1, 8}, {"V2 T=8", 2, 8}, {"V2 T=128", 2, 128},
        };
        printf("correctness (M=%d K=%d):\n", M, K);
        int all_ok = 1;
        for (unsigned c = 0; c < 3; ++c) {
            const int TT = cases[c].tt;
            /* X/Y buffers in [K][TT]/[M][TT] layout, master columns 0..TT-1 */
            float *hXt = (float *)malloc((size_t)K * TT * 4);
            float *hYt = (float *)malloc((size_t)M * TT * 4);
            for (int k = 0; k < K; ++k)
                for (int t = 0; t < TT; ++t) hXt[k * TT + t] = hXm[k * MAX_T + t];
            float *dX, *dY;
            CK(cudaMalloc(&dX, (size_t)K * TT * 4));
            CK(cudaMalloc(&dY, (size_t)M * TT * 4));
            CK(cudaMemcpy(dX, hXt, (size_t)K * TT * 4, cudaMemcpyHostToDevice));

            dim3 block(32, WARPS_PER_BLOCK), grid(M / WARPS_PER_BLOCK);
            pick(cases[c].v, TT)(dW, dX, dY, M, K, grid, block, 0);
            CK(cudaGetLastError());
            CK(cudaMemcpy(hYt, dY, (size_t)M * TT * 4, cudaMemcpyDeviceToHost));

            /* compare vs T=8 reference (cols >= 8 have no reference) */
            double maxref = 1e-30, maxerr = 0;
            for (int m = 0; m < M; ++m)
                for (int t = 0; t < 8; ++t) {
                    double r = ref[(size_t)m * T + t];
                    double e = fabs((double)hYt[(size_t)m * TT + t] - r);
                    if (fabs(r) > maxref) maxref = fabs(r);
                    if (e > maxerr) maxerr = e;
                }
            double rel = maxerr / maxref;
            int ok = rel < 1e-2;
            all_ok &= ok;
            printf("  %-9s max_abs_err=%.3e max|ref|=%.1f rel=%.2e %s\n",
                   cases[c].n, maxerr, maxref, rel, ok ? "PASS" : "FAIL");
            free(hXt); free(hYt);
            CK(cudaFree(dX)); CK(cudaFree(dY));
        }
        printf("  overall: %s\n\n", all_ok ? "PASS" : "FAIL");
        free(ref); free(hX);
    }

    /* ============================ benchmark matrix ========================= */
    struct { int M; const char *name; } shapes[] = {
        {1536, "attn/mlp"}, {4096, "ffn"}, {12288, "big-mlp"}, {262144, "lm-head"},
    };
    const int TS[] = { 1, 8, 32, 128 };

    printf("occupancy (theoretical, %d threads/block):\n", 32 * WARPS_PER_BLOCK);
    occ_report("V1", (const void *)k_v1<1>,   1);
    occ_report("V1", (const void *)k_v1<8>,   8);
    occ_report("V1", (const void *)k_v1<32>,  32);
    occ_report("V1", (const void *)k_v1<128>, 128);
    occ_report("V2", (const void *)k_v2<1>,   1);
    occ_report("V2", (const void *)k_v2<8>,   8);
    occ_report("V2", (const void *)k_v2<32>,  32);
    occ_report("V2", (const void *)k_v2<128>, 128);
    printf("\n");

    printf("%-9s %4s | %8s %9s %8s | %8s %9s %8s | %7s\n",
           "shape", "T", "V1 GB/s", "V1 tok/s", "V1 ms", "V2 GB/s", "V2 tok/s", "V2 ms", "V2/V1");
    float *dX, *dY;
    CK(cudaMalloc(&dX, (size_t)K * MAX_T * 4));
    CK(cudaMalloc(&dY, (size_t)MAX_ROWS * MAX_T * 4));

    for (unsigned si = 0; si < 4; ++si) {
        int M = shapes[si].M;
        size_t wb = (size_t)M * rowbytes;
        for (unsigned ti = 0; ti < 4; ++ti) {
            int T = TS[ti];
            /* build [K][T]-layout X once */
            float *hXt = (float *)malloc((size_t)K * T * 4);
            for (int k = 0; k < K; ++k)
                for (int t = 0; t < T; ++t) hXt[k * T + t] = hXm[k * MAX_T + t];
            CK(cudaMemcpy(dX, hXt, (size_t)K * T * 4, cudaMemcpyHostToDevice));
            free(hXt);

            size_t bytes = wb + (size_t)K * T * 4 + (size_t)M * T * 4;
            double est = bytes / 200e9 * 1e3;
            int iters = (int)fmax(3.0, fmin(100.0, 40.0 / fmax(est, 0.001)));

            double ms1 = bench(pick(1, T), dW, dX, dY, M, K, T, iters);
            double ms2 = bench(pick(2, T), dW, dX, dY, M, K, T, iters);
            double g1 = bytes / (ms1 * 1e-3) / 1e9;
            double g2 = bytes / (ms2 * 1e-3) / 1e9;
            double tok2 = T / (ms2 * 1e-3);
            printf("%-9s %4d | %8.1f %9.0f %8.3f | %8.1f %9.0f %8.3f | %6.2fx%s\n",
                   shapes[si].name, T, g1, T / (ms1 * 1e-3), ms1,
                   g2, tok2, ms2, g2 / g1,
                   g2 > BASE_GB_S ? " >base" : "");
        }
    }

    /* ================= crossover analysis =================================== */
    printf("\ncrossover vs single-token baseline (%.0f GB/s):\n", BASE_GB_S);
    for (unsigned si = 0; si < 4; ++si) {
        int M = shapes[si].M;
        size_t wb = (size_t)M * rowbytes;
        double base_tok_s = BASE_GB_S * 1e9 / wb;             // tokens/s unbatched
        printf("  %-9s unbatched=%.0f tok/s-equiv; ", shapes[si].name, base_tok_s);
        int cross = -1;
        for (unsigned ti = 0; ti < 4; ++ti) {
            int T = TS[ti];
            float *hXt = (float *)malloc((size_t)K * T * 4);
            for (int k = 0; k < K; ++k)
                for (int t = 0; t < T; ++t) hXt[k * T + t] = hXm[k * MAX_T + t];
            CK(cudaMemcpy(dX, hXt, (size_t)K * T * 4, cudaMemcpyHostToDevice));
            free(hXt);
            size_t bytes = wb + (size_t)K * T * 4 + (size_t)M * T * 4;
            double est = bytes / 200e9 * 1e3;
            int iters = (int)fmax(3.0, fmin(50.0, 20.0 / fmax(est, 0.001)));
            double ms2 = bench(pick(2, T), dW, dX, dY, M, K, T, iters);
            double g2 = bytes / (ms2 * 1e-3) / 1e9;
            if (cross < 0 && g2 > BASE_GB_S) { cross = T; break; }
        }
        if (cross > 0) printf("V2 wins from T=%d\n", cross);
        else printf("V2 never beats baseline in tested set\n");
    }

    printf("\nnote: small shapes (<=~4MB weights) fit 4MB L2 -> reported GB/s\n"
           "exceed DRAM; use ncu (dram__bytes) for true streaming rates.\n");

    free(hW); free(hXm);
    CK(cudaFree(dW)); CK(cudaFree(dX)); CK(cudaFree(dY));
    return 0;
}
