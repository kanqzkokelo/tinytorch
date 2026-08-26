// PROTOTYPE (create-only study): speculative-decode VERIFY batched-GEMM cost
// model. Decides draft length m for M10 tt_verify.
//
// Question: tt_verify needs one forward pass over k+1 positions. How much does
// verify width T actually cost PER TOKEN vs plain decode (T=1)? Speculation
// profits iff (1+a) > (1+f(T)) where a = accept rate and
// f(T) = [cost(T)/T] / cost(1) is the MEASURED marginal-cost factor.
// Simulator context (tests/test_specdec_sim.py): best ngram result was 1.88x
// under the FLAT (memory-bound => width-free) assumption. This measures how
// far reality is from FLAT.
//
//   Standalone q4_0 GEMM  Y[M,T] = W[M,K] @ X[K,T]
//   W[M,K] q4_0 row-major (18 B blocks: fp16 d + 16 B nibbles, v=(q-8)*d)
//   X[K,T] / Y[M,T] row-major
//   Kernel: dequant-inline pattern lifted from tests/proto_batched_gemv.cu
//   (V2, T<=32 branch): block tiles rows across warps, T columns spread over
//   lanes with K-split per column (xor-reduce); weight stream amortized over
//   all T columns. Aligned-u32 + __byte_perm block load (the bandwidth fix).
//
// Measured:
//   1. shape (M,K)=(1536,1536), T in {1,2,4,8,16}: us/token -> marginal f(T)
//   2. lm-head (262144,1536) at T in {1,8}: does full-width logits dominate?
//      (verify needs logits at ALL k+1 positions unless we gate to last only)
//   3. profit contours over (a, m) using measured f, both with FULL lm-head
//      and LAST-POSITION-ONLY lm-head variants.
//
// Build:
//   $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 -o proto_verify_batched \
//       tests/proto_verify_batched.cu
// Run:
//   LD_LIBRARY_PATH=$HOME/mmcuda/lib ./proto_verify_batched
//
// Self-contained: no repo headers, kernels/ and src/specdec.* untouched.
// CPU double reference checks T=8 at rel < 1e-2 (quantization tolerance).
// Footprint: W(lm-head) ~= 262144*864 B = 227 MB device + host copy; well
// under 400 MB GPU. cudaMalloc retries x5 on OOM (E2B sandbox may transiently
// report cudaErrorMemoryAllocation while the loader settles).
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_T 16
#define LMH_ROWS 262144

/* ---------------- fp16 bit conversion (host+device, no deps) --------------- */
static inline __host__ __device__ float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (!man) { bits = sign; }
        else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400));
            m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
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
    if (e <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
}

/* ---------------- q4_0 block load (from proto_batched_gemv.cu) ------------- */
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

/* ---------------- GEMM kernel: rows across warps, T cols over lanes --------
 * T <= 32 power of two.
 *  - One warp per output row (blockDim.y warps/block); weight bytes for the
 *    row streamed exactly ONCE regardless of T.
 *  - Columns spread across lanes: column c owned by lanes with lane%T==c;
 *    the 32/T lanes of a column split K, partial sums xor-reduced.
 *  - X chunk ([CH*32 k-values][T]) staged in shared per iteration; every row
 *    warp consumes the same tile (pitch T+4 avoids bank conflicts).
 * NOTE: tests/proto_batched_gemv.cu V2 carries an analogous-but-misindexed
 * tile walk; this kernel re-derives indexing from scratch:
 *   q4_0 block b covers k-values b*32+j (j=0..31); chunk g covers blocks
 *   g..g+CH-1 i.e. k-values g*32 .. g*32+CH*32-1, staged at shared row
 *   (k - g*32). Block (g+kk), nibble j reads shared row kk*32 + (j|j+16). */
template<int T_>
__global__ void k_gemm(const uint8_t *__restrict__ W, const float *__restrict__ X,
                       float *__restrict__ Y, int M, int K) {
    constexpr int CH = 8;                        // blocks per chunk
    constexpr int RK = CH * 32;                  // k-values per chunk
    constexpr int PITCH = T_ + 4;
    constexpr int KSP = 32 / T_;                 // lanes per column (K-split)
    __shared__ float sx[RK * PITCH];             // 20 KB at T=16
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const bool valid = row < M;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int nth = blockDim.x * blockDim.y;
    const int nb = K >> 5;
    const uint32_t *rw =
        (const uint32_t *)(W + (size_t)(valid ? row : 0) * nb * 18);

    const int col0 = threadIdx.x % T_;           // this lane's column
    const int kp   = threadIdx.x / T_;           // K-split group id
    float acc = 0.f;

    for (int g = 0; g < nb; g += CH) {
        const int cn = min(CH, nb - g);
        for (int i = tid; i < cn * 32 * T_; i += nth)
            sx[(i / T_) * PITCH + (i % T_)] =
                X[(size_t)(g * 32 + i / T_) * T_ + (i % T_)];
        __syncthreads();
        if (valid) {
            /* lane kp takes every KSP-th block -> disjoint K coverage per
             * column group; butterfly reduce below recombines */
            for (int kk = kp; kk < cn; kk += KSP) {
                float d; uint32_t q[4];
                load_blk(rw, g + kk, d, q);
                const float *xr = &sx[(kk * 32) * PITCH + col0];
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    s += (float)(nib(q, j) - 8)    * xr[ j      * PITCH];
                    s += (float)(nib_hi(q, j) - 8) * xr[(j + 16) * PITCH];
                }
                acc += s * d;
            }
        }
        __syncthreads();
    }

    if (!valid) return;
#pragma unroll
    for (int off = KSP / 2; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off * T_);
    if (kp == 0) Y[(size_t)row * T_ + col0] = acc;
}

/* ---------------- host helpers -------------------------------------------- */
static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return (uint32_t)(rng_s >> 32);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_) { \
    fprintf(stderr, "CUDA error %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

/* OOM-resilient device malloc: retry x5 (E2B sandbox can transiently refuse
 * while its allocator settles). Sleep grows between tries. */
static int g_oom_retries = 0;
static void *dmalloc_ck(size_t bytes) {
    void *p = NULL;
    for (int i = 0; i < 5; ++i) {
        cudaError_t e = cudaMalloc(&p, bytes);
        if (!e) { g_oom_retries += i; return p; }
        if (e != cudaErrorMemoryAllocation) {
            fprintf(stderr, "CUDA error cudaMalloc: %s\n", cudaGetErrorString(e));
            exit(1);
        }
        fprintf(stderr, "OOM (try %d/5), %.1f MB requested - retrying\n",
                i + 1, bytes / 1048576.0);
        struct timespec ts = {0, (i + 1) * 200 * 1000000L};
        nanosleep(&ts, NULL);
    }
    fprintf(stderr, "FATAL: OOM after 5 retries\n");
    exit(1);
}
static void *hmalloc_ck(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "host malloc failed (%zu bytes)\n", bytes); exit(1); }
    return p;
}

typedef void (*launch_fn)(const uint8_t *, const float *, float *, int, int, dim3, dim3);

template<int T_> static void launch_g(const uint8_t *W, const float *X, float *Y,
                                      int M, int K, dim3 g, dim3 b) {
    k_gemm<T_><<<g, b>>>(W, X, Y, M, K);
}
static launch_fn pick(int T) {
    switch (T) {
        case 1: return launch_g<1>;  case 2: return launch_g<2>;
        case 4: return launch_g<4>;  case 8: return launch_g<8>;
        case 16: return launch_g<16>;
    }
    return NULL;
}

static const int WARPS_PER_BLOCK = 8;

static double bench(launch_fn fn, const uint8_t *dW, const float *dX, float *dY,
                    int M, int K, int T, int iters) {
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    for (int w = 0; w < 3; ++w) fn(dW, dX, dY, M, K, grid, block);
    CK(cudaGetLastError());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) fn(dW, dX, dY, M, K, grid, block);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return (double)ms / iters;
}

/* fill [K][T]-layout X from master [K][MAX_T] */
static void build_X(float *dst, const float *master, int K, int T) {
    for (int k = 0; k < K; ++k)
        for (int t = 0; t < T; ++t)
            dst[(size_t)k * T + t] = master[(size_t)k * MAX_T + t];
}

int main(void) {
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("device: %s, %d SMs, L2 %d MB\n\n", p.name,
           p.multiProcessorCount, (int)(p.l2CacheSize >> 20));

    /* ---- weights: generate max-shape once (lm-head prefix used for 1536) -- */
    const int TS[] = { 1, 2, 4, 8, 16 };
    const int NT = 5;
    const int K = 1536;
    const int nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;               // 864 B/row
    const size_t Wtot = (size_t)LMH_ROWS * rowbytes;       // ~227 MB
    uint8_t *hW = (uint8_t *)hmalloc_ck(Wtot + 64);        // + slack: load_blk
                                                           // reads <=1 word past
    for (size_t r = 0; r < LMH_ROWS; ++r) {
        uint8_t *row = hW + r * rowbytes;
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = row + (size_t)b * 18;
            float d = 0.25f + 0.5f * ((rnd() >> 8) / 16777216.0f);
            uint16_t d16 = f2h(d);
            blk[0] = d16 & 0xFF; blk[1] = d16 >> 8;
            for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t)rnd();
        }
    }
    float *hXm = (float *)hmalloc_ck((size_t)K * MAX_T * 4);
    for (size_t i = 0; i < (size_t)K * MAX_T; ++i)
        hXm[i] = ((double)rnd() / 4294967295.0) * 2.0 - 1.0;

    uint8_t *dW = (uint8_t *)dmalloc_ck(Wtot + 64);
    CK(cudaMemcpy(dW, hW, Wtot, cudaMemcpyHostToDevice));

    /* ================= correctness: T=8 vs CPU double reference ============ */
    {
        const int M = 512, T = 8;
        float *hX = (float *)hmalloc_ck((size_t)K * MAX_T * 4);
        build_X(hX, hXm, K, T);
        double *ref = (double *)hmalloc_ck((size_t)M * T * sizeof(double));
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
        float *dX = NULL;
        float *dY = NULL;
        float *hY = NULL;
        printf("correctness M=%d K=%d vs CPU double ref:\n", M, K);
        int all_ok = 1;
        for (int ci = 0; ci < NT; ++ci) {
            const int T = TS[ci];
            build_X(hX, hXm, K, T);
            if (!dY) { dX = (float *)dmalloc_ck((size_t)K * MAX_T * 4);
                       dY = (float *)dmalloc_ck((size_t)M * MAX_T * 4);
                       hY = (float *)hmalloc_ck((size_t)M * MAX_T * 4); }
            CK(cudaMemcpy(dX, hX, (size_t)K * T * 4, cudaMemcpyHostToDevice));
            dim3 block(32, WARPS_PER_BLOCK), grid(M / WARPS_PER_BLOCK);
            pick(T)(dW, dX, dY, M, K, grid, block);
            CK(cudaGetLastError());
            CK(cudaMemcpy(hY, dY, (size_t)M * T * 4, cudaMemcpyDeviceToHost));

            /* compare columns that exist in both X layouts (T and ref's 8) */
            double maxref = 1e-30, maxerr = 0;
            for (int m = 0; m < M; ++m)
                for (int t = 0; t < (T < 8 ? T : 8); ++t) {
                    double r = ref[(size_t)m * 8 + t];
                    double e = fabs((double)hY[(size_t)m * T + t] - r);
                    if (fabs(r) > maxref) maxref = fabs(r);
                    if (e > maxerr) maxerr = e;
                }
            double rel = maxerr / maxref;
            int ok = rel < 1e-2;
            all_ok &= ok;
            printf("  T=%2d: max_abs_err=%.3e rel=%.2e %s\n",
                   T, maxerr, rel, ok ? "PASS" : "FAIL");
        }
        printf("  overall: %s (oom retries so far: %d)\n\n",
               all_ok ? "PASS" : "FAIL", g_oom_retries);
        free(ref); free(hX); free(hY);
        CK(cudaFree(dX)); CK(cudaFree(dY));
    }

    /* ================= 1. marginal-cost curve, (1536,1536) ================= */
    double us_tok[NT], fwd_us[NT];

    printf("=== shape (M,K)=(1536,1536): marginal cost of verify width ===\n");
    printf("%3s %10s %12s %9s %7s\n", "T", "fwd_us", "us/tok", "GB/s", "f(T)");
    double f[NT];
    float *dX = (float *)dmalloc_ck((size_t)K * MAX_T * 4);
    float *dY = (float *)dmalloc_ck((size_t)LMH_ROWS * MAX_T * 4);
    float *hXt = (float *)hmalloc_ck((size_t)K * MAX_T * 4);

    for (int ti = 0; ti < NT; ++ti) {
        int T = TS[ti];
        build_X(hXt, hXm, K, T);
        CK(cudaMemcpy(dX, hXt, (size_t)K * T * 4, cudaMemcpyHostToDevice));
        size_t bytes = (size_t)1536 * rowbytes + (size_t)K * T * 4 + (size_t)1536 * T * 4;
        double est = bytes / 20e9 * 1e3;                   // ms at ~20 GB/s
        int iters = (int)fmax(10.0, fmin(500.0, 50.0 / fmax(est, 0.001)));
        fwd_us[ti] = bench(pick(T), dW, dX, dY, 1536, K, T, iters) * 1e3;
        us_tok[ti] = fwd_us[ti] / T;
        double gb = bytes / (fwd_us[ti] * 1e-6) / 1e9;
        f[ti] = us_tok[ti] / us_tok[0];
        printf("%3d %10.2f %12.2f %9.1f %6.2fx\n",
               T, fwd_us[ti], us_tok[ti], gb, f[ti]);
    }
    printf("\n");

    /* ================= 2. lm-head (262144,1536), T in {1,8} ================ */
    double lh_fwd[2], lh_f[2];
    const int LH_T[] = { 1, 8 };
    printf("=== shape (M,K)=(262144,1536): lm-head, full-width logits ===\n");
    printf("%3s %10s %12s %9s %7s\n", "T", "fwd_ms", "ms/tok", "GB/s", "f(T)");
    for (int li = 0; li < 2; ++li) {
        int T = LH_T[li];
        build_X(hXt, hXm, K, T);
        CK(cudaMemcpy(dX, hXt, (size_t)K * T * 4, cudaMemcpyHostToDevice));
        size_t bytes = Wtot + (size_t)K * T * 4 + (size_t)LMH_ROWS * T * 4;
        double est = bytes / 200e9 * 1e3;
        int iters = (int)fmax(3.0, fmin(30.0, 40.0 / fmax(est, 0.001)));
        lh_fwd[li] = bench(pick(T), dW, dX, dY, LMH_ROWS, K, T, iters);
        double gb = bytes / (lh_fwd[li] * 1e-3) / 1e9;
        lh_f[li] = (lh_fwd[li] / T) / (lh_fwd[0]);         // per-token factor
        printf("%3d %10.3f %12.3f %9.1f %6.2fx\n",
               T, lh_fwd[li], lh_fwd[li] / T, gb, lh_f[li]);
    }
    printf("\n");

    /* ================= 3. profit contours =================================== */
    /* Model per verify step of draft length m (width T=m+1):
     *   hidden-layer cost scales as f_hid(m+1) per token (measured above),
     *   lm-head cost either FULL width (all m+1 logits needed) or LAST-ONLY
     *   (logits computed once at T=1).
     * multiplier(a,m) = (1 + a*m) / (1 + f_total(m))
     * with f_total(m) = (hid_us(m+1)+lh_us_variant(m+1)) / (hid_us(1)+lh_us_variant(1)).
     * Profit iff multiplier > 1. a = per-drafted-token accept rate. */
    printf("=== profit contours: multiplier(a,m) = (1+a*m)/(1+f_total(m)) ===\n");
    printf("(a = mean accept rate per drafted token; profit iff > 1)\n\n");

    /* interpolate hid per-token us at width m+1 from measurements */
    auto hid_tok_us = [&](double T_) {
        for (int i = 0; i + 1 < NT; ++i)
            if (T_ >= TS[i] && T_ <= TS[i + 1]) {
                double w = (T_ - TS[i]) / (TS[i + 1] - TS[i]);
                return us_tok[i] * (1 - w) + us_tok[i + 1] * w;
            }
        return T_ < TS[0] ? us_tok[0] : us_tok[NT - 1];
    };
    for (int variant = 0; variant < 2; ++variant) {
        const char *vn = variant == 0 ? "FULL lm-head (all m+1 logits)"
                                      : "LAST-ONLY lm-head (1 logit row)";
        printf("--- %s ---\n", vn);
        printf("m | ");
        for (int ai = 0; ai <= 10; ++ai) printf("a=%.1f ", ai * 0.1);
        printf("\n");
        int best_m = -1; double best_mult_at_a = 0;
        const double A_STAR = 0.55;   // rough ngram-sim accept rate regime
        /* reference point off-grid: a=0.55 */
        for (int mi = 1; mi <= 12; ++mi) {
            double T_ = mi + 1;
            double hid = hid_tok_us(T_);
            /* total per-token verify cost vs plain decode per-token cost */
            double base = us_tok[0] + lh_fwd[0] * 1e3;   // decode: hid+lmhead @T=1
            double tot;
            if (variant == 0) {
                double lhf = T_ <= 1 ? lh_fwd[0] * 1e3
                           : (lh_fwd[1] * 1e3 / T_) ;    // assume flat GB/s beyond 8
                tot = hid + lhf;
            } else {
                tot = hid + lh_fwd[0] * 1e3;             // lm-head pinned at T=1
            }
            double ftot = tot / base;
            printf("%d |", mi);
            for (int ai = 0; ai <= 10; ++ai) {
                double a = ai * 0.1;
                double mult = (1.0 + a * mi) / (1.0 + ftot);
                printf("%5.2f%s", mult, mult > 1.0 ? "*" : " ");
            }
            printf("   f_tot=%.2f\n", ftot);
            double mult_star = (1.0 + A_STAR * mi) / (1.0 + ftot);
            if (mult_star > best_mult_at_a) { best_mult_at_a = mult_star; best_m = mi; }
        }
        printf("profit marker '*'. At a=%.2f: best m=%d, mult=%.2f\n\n",
               A_STAR, best_m, best_mult_at_a);
    }

    printf("notes:\n"
           " - lm-head dominates absolute time (~%.0fx hidden layer at T=1);\n"
           "   whether FULL vs LAST-ONLY changes the CONTOUR depends on its own\n"
           "   f(T) slope, printed above.\n"
           " - small-M shape fits L2; treat GB/s there as optimistic.\n"
           " - oom retries consumed: %d\n",
           lh_fwd[0] * 1e3 / fwd_us[0], g_oom_retries);

    free(hW); free(hXm); free(hXt);
    CK(cudaFree(dW)); CK(cudaFree(dX)); CK(cudaFree(dY));
    return 0;
}
