// PROTOTYPE (M9 perf, CREATE-ONLY): device-fused PLE MatFormer layer stage.
//
// Per-layer chain this replaces (host-assisted today, 35x per token):
//   g[256]   = gelu(inp_gate[256x1536] f32 @ x[1536]) * PLE[layer][256]
//   out[1536]= rmsnorm(pl_proj[1536x256] bf16 @ g, gamma[1536])
//
// Variants measured:
//   HOST  : gemv -> sync -> D2H -> cpu(gelu*ple) -> H2D -> gemv -> sync ->
//           D2H -> cpu(rmsnorm)          (current engine shape)
//   V1    : 4 plain kernels back-to-back on device, no host round-trip:
//           k_gemv_f32, k_gelu_ple, k_gemv_bf16, k_rmsnorm
//   V2    : exactly 2 launches:
//           k_ple_stage1 : gemv f32 + gelu + PLE mul  -> d_tmp[256]
//           k_ple_stage2 : gemv bf16, then last-finishing block
//                          (atomic ticket + threadfence) runs rmsnorm
//                          epilogue over all 1536 outputs
//
// Build:
//   $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 -o tests/proto_ple_fused \
//       tests/proto_ple_fused.cu
// Run:
//   LD_LIBRARY_PATH=$HOME/mmcuda/lib ./tests/proto_ple_fused
//
// Footprint: <1 MB device memory. Correctness vs naive CPU double reference,
// PASS if max relative error < 1e-3.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ROWS_W1 256
#define K_DIM     1536
#define N_PLE     35
#define EPS       1e-5f
#define SEED      42

/* ---------------- helpers ---------------- */

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float bf16_to_float(uint16_t h) {
    __nv_bfloat16 b = *reinterpret_cast<__nv_bfloat16 *>(&h);
    return __bfloat162float(b);
}

__device__ __forceinline__ float gelu_f(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

static float rnd01(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)((*s >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
}

static void fill_host(float *p, int n, uint32_t *seed, float scale) {
    for (int i = 0; i < n; i++) p[i] = rnd01(seed) * scale;
}

/* ---------------- V1 kernels ---------------- */

__global__ void k_gemv_f32(const float *__restrict__ W,   /* [256,1536] */
                           const float *__restrict__ x, float *__restrict__ y,
                           int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const float *rw = W + (long)row * K;
    float s = 0.0f;
    for (int j = lane; j < K; j += 32) s += rw[j] * x[j];
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

__global__ void k_gelu_ple(const float *__restrict__ h,
                           const float *__restrict__ ple, /* one 256 row */
                           float *__restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g[i] = gelu_f(h[i]) * ple[i];
}

__global__ void k_gemv_bf16(const uint16_t *__restrict__ W, /* [1536,256] */
                            const float *__restrict__ x, float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const uint16_t *rw = W + (long)row * K;
    float s = 0.0f;
    for (int j = lane; j < K; j += 32) s += bf16_to_float(rw[j]) * x[j];
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

__global__ void k_rmsnorm(const float *__restrict__ y,
                          const float *__restrict__ gamma,
                          float *__restrict__ out, int n) {
    /* single block, 1024 threads, strided */
    extern __shared__ float red[];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) ss += y[i] * y[i];
    red[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / n + EPS);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = y[i] * inv * gamma[i];
}

/* ---------------- V2 kernels ---------------- */

/* Stage 1: grid of 8-warp blocks; each warp owns whole rows (independent,
   no cross-block reduce needed). Grid = ceil(256 / warps_per_block). */
__global__ void k_ple_stage1(const float *__restrict__ W1,
                             const float *__restrict__ x,
                             const float *__restrict__ ple_row,
                             float *__restrict__ g) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    const int row = blockIdx.x * nwarp + warp;
    if (row >= N_ROWS_W1) return;
    const float *rw = W1 + (long)row * K_DIM;
    float s = 0.0f;
    for (int j = lane; j < K_DIM; j += 32) s += rw[j] * x[j];
    s = warp_reduce_sum(s);
    if (lane == 0) g[row] = gelu_f(s) * ple_row[row];
}

/* Stage 2: 48 warps total (grid 12 x block 128), warp per output row.
   Last block to finish (atomic ticket) recomputes sum-of-squares over the
   1536 outputs in gmem and applies the rmsnorm in place. */
__device__ unsigned int g_ticket = 0;

__global__ void k_ple_stage2(const uint16_t *__restrict__ W2,
                             const float *__restrict__ g,
                             const float *__restrict__ gamma,
                             float *__restrict__ out, int M, int K) {
    /* 16 warps per 512-thread block; grid sized so gridDim.x*16 >= M */
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row < M) {
        const uint16_t *rw = W2 + (long)row * K;
        float s = 0.0f;
        for (int j = lane; j < K; j += 32) s += bf16_to_float(rw[j]) * g[j];
        s = warp_reduce_sum(s);
        if (lane == 0) out[row] = s;
    }

    /* grid-wide barrier via ticket: last block normalizes */
    __shared__ bool is_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int prev = atomicInc(&g_ticket, gridDim.x - 1);
        is_last = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (!is_last) return;

    /* normalize: read back out[], apply rmsnorm in place */
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) ss += out[i] * out[i];
    /* small reduce across 128 threads in smem */
    __shared__ float red[512];
    red[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / M + EPS);
    for (int i = tid; i < M; i += blockDim.x)
        out[i] *= inv * gamma[i];
    g_ticket = 0; /* reset for next launch */
}

/* ---------------- CPU references (double) ---------------- */

static void cpu_chain(const float *W1, const uint16_t *W2, const float *ple_row,
                      const float *x, const float *gamma, double *out,
                      double *tmp_g) {
    static double h[N_ROWS_W1], y[K_DIM];
    for (int r = 0; r < N_ROWS_W1; r++) {
        double s = 0.0;
        for (int j = 0; j < K_DIM; j++) s += (double)W1[r * K_DIM + j] * x[j];
        h[r] = 0.5 * s * (1.0 + erf(s * 0.70710678118654752440)); /* exact erf */
    }
    for (int r = 0; r < N_ROWS_W1; r++) tmp_g[r] = h[r] * ple_row[r];
    for (int r = 0; r < K_DIM; r++) {
        double s = 0.0;
        for (int j = 0; j < N_ROWS_W1; j++) {
            uint32_t f32bits = (uint32_t)W2[r * N_ROWS_W1 + j] << 16;
            float fv;
            memcpy(&fv, &f32bits, 4); /* bf16 = top 16 bits of f32 */
            s += (double)fv * tmp_g[j];
        }
        y[r] = s;
    }
    double ss = 0.0;
    for (int r = 0; r < K_DIM; r++) ss += y[r] * y[r];
    const double inv = 1.0 / sqrt(ss / K_DIM + EPS);
    for (int r = 0; r < K_DIM; r++) out[r] = y[r] * inv * gamma[r];
}

/* ---------------- timing harness ---------------- */

static float time_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

#define CHK(stmt)                                                          \
    do {                                                                   \
        cudaError_t e__ = (stmt);                                          \
        if (e__ != cudaSuccess) {                                          \
            printf("CUDA err %s at %s:%d\n", cudaGetErrorString(e__),      \
                   __FILE__, __LINE__);                                    \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

int main() {
    printf("proto_ple_fused: %dx%d f32 W1, %dx%d bf16 W2, PLE %dx%d\n",
           N_ROWS_W1, K_DIM, K_DIM, N_ROWS_W1, N_PLE, N_ROWS_W1);

    /* host data, seed 42 */
    uint32_t seed = SEED;
    static float W1[N_ROWS_W1 * K_DIM], x[K_DIM], gamma_[K_DIM];
    static uint16_t W2[K_DIM * N_ROWS_W1];
    static float ple[N_PLE * N_ROWS_W1];
    static float raw2[K_DIM * N_ROWS_W1];
    fill_host(W1, N_ROWS_W1 * K_DIM, &seed, 0.05f);
    fill_host(raw2, K_DIM * N_ROWS_W1, &seed, 0.05f);
    fill_host(x, K_DIM, &seed, 1.0f);
    fill_host(gamma_, K_DIM, &seed, 1.0f); /* ~N(1): shift by +1 below */
    for (int i = 0; i < K_DIM; i++) gamma_[i] += 1.0f;
    fill_host(ple, N_PLE * N_ROWS_W1, &seed, 1.0f);
    for (int i = 0; i < K_DIM * N_ROWS_W1; i++) {
        __nv_bfloat16 b = __float2bfloat16(raw2[i]);
        memcpy(&W2[i], &b, 2);
    }

    /* device buffers */
    float *d_W1, *d_x, *d_gamma, *d_ple, *d_h, *d_g, *d_y, *d_out_v1, *d_out_v2;
    uint16_t *d_W2;
    cudaMalloc(&d_W1, sizeof(W1));
    cudaMalloc(&d_W2, sizeof(W2));
    cudaMalloc(&d_ple, sizeof(ple));
    cudaMalloc(&d_x, sizeof(x));
    cudaMalloc(&d_gamma, sizeof(gamma_));
    cudaMalloc(&d_h, N_ROWS_W1 * 4);
    cudaMalloc(&d_g, N_ROWS_W1 * 4);
    cudaMalloc(&d_y, K_DIM * 4);
    cudaMalloc(&d_out_v1, K_DIM * 4);
    cudaMalloc(&d_out_v2, K_DIM * 4);
    cudaMemcpy(d_W1, W1, sizeof(W1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, sizeof(W2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ple, ple, sizeof(ple), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma_, sizeof(gamma_), cudaMemcpyHostToDevice);

    const int LAYER = 7; /* arbitrary slice under test */
    const float *ple_row_d = d_ple + (long)LAYER * N_ROWS_W1;
    static float ple_row_h[N_ROWS_W1];
    memcpy(ple_row_h, ple + (long)LAYER * N_ROWS_W1, sizeof(ple_row_h));

    dim3 blk_gemv(32, 8); /* 8 warps -> 32 rows/block */
    dim3 grd_w1(N_ROWS_W1 / 8);
    dim3 grd_w2(K_DIM / 8);
    dim3 blk_el(256);
    dim3 grd_el(N_ROWS_W1 / 256);

    /* ---- correctness ---- */
    static double ref[K_DIM], tmpg[N_ROWS_W1];
    cpu_chain(W1, W2, ple_row_h, x, gamma_, ref, tmpg);

    static float got_v1[K_DIM], got_v2[K_DIM];

    /* V1 run */
    k_gemv_f32<<<grd_w1, blk_gemv>>>(d_W1, d_x, d_h, N_ROWS_W1, K_DIM);
    k_gelu_ple<<<grd_el, blk_el>>>(d_h, ple_row_d, d_g, N_ROWS_W1);
    k_gemv_bf16<<<grd_w2, blk_gemv>>>(d_W2, d_g, d_y, K_DIM, N_ROWS_W1);
    k_rmsnorm<<<1, 1024, 1024 * 4>>>(d_y, d_gamma, d_out_v1, K_DIM);
    cudaDeviceSynchronize();
    cudaMemcpy(got_v1, d_out_v1, sizeof(got_v1), cudaMemcpyDeviceToHost);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    /* V2 run: 32 blocks x 8 warps = 256 rows */
    k_ple_stage1<<<32, 256>>>(d_W1, d_x, ple_row_d, d_g);
    k_ple_stage2<<<96, 512>>>(d_W2, d_g, d_gamma, d_out_v2, K_DIM, N_ROWS_W1);
    cudaDeviceSynchronize();
    cudaMemcpy(got_v2, d_out_v2, sizeof(got_v2), cudaMemcpyDeviceToHost);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    double err1 = 0, err2 = 0, nrm = 0;
    for (int i = 0; i < K_DIM; i++) {
        err1 = fmax(err1, fabs((double)got_v1[i] - ref[i]) /
                              (fabs(ref[i]) > 1e-9 ? fabs(ref[i]) : 1.0));
        err2 = fmax(err2, fabs((double)got_v2[i] - ref[i]) /
                              (fabs(ref[i]) > 1e-9 ? fabs(ref[i]) : 1.0));
        nrm = fmax(nrm, fabs(ref[i]));
    }
    printf("correctness: V1 max_rel_err=%.3e  V2 max_rel_err=%.3e  "
           "(|ref|max=%.3f)\n",
           err1, err2, nrm);
    const bool pass = err1 < 1e-3 && err2 < 1e-3;
    printf("%s\n", pass ? "PASS (<1e-3 rel)" : "FAIL");

    /* ---- latency ---- */
    const int ITERS = 200, WARMUP = 20;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    static float host_tmp[N_ROWS_W1], host_yin[K_DIM], host_out[K_DIM];

    /* HOST baseline: gemv, sync, D2H, cpu gelu*ple, H2D, gemv, sync, D2H,
       cpu rmsnorm (result written to host_out) */
    float t_host = 0;
    for (int it = 0; it < ITERS + WARMUP; it++) {
        if (it == WARMUP) { cudaDeviceSynchronize(); cudaEventRecord(ev0); }
        k_gemv_f32<<<grd_w1, blk_gemv>>>(d_W1, d_x, d_h, N_ROWS_W1, K_DIM);
        cudaDeviceSynchronize();
        cudaMemcpy(host_tmp, d_h, N_ROWS_W1 * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N_ROWS_W1; i++)
            host_tmp[i] =
                0.5f * host_tmp[i] * (1.0f + erff(host_tmp[i] * 0.70710678f)) *
                ple_row_h[i];
        cudaMemcpy(d_g, host_tmp, N_ROWS_W1 * 4, cudaMemcpyHostToDevice);
        k_gemv_bf16<<<grd_w2, blk_gemv>>>(d_W2, d_g, d_y, K_DIM, N_ROWS_W1);
        cudaDeviceSynchronize();
        cudaMemcpy(host_yin, d_y, K_DIM * 4, cudaMemcpyDeviceToHost);
        double ss = 0.0;
        for (int i = 0; i < K_DIM; i++) ss += (double)host_yin[i] * host_yin[i];
        const float inv = 1.0f / sqrtf(ss / K_DIM + EPS);
        for (int i = 0; i < K_DIM; i++)
            host_out[i] = host_yin[i] * inv * gamma_[i];
    }
    (void)host_out[0]; /* silence set-but-not-read in timing loop */
    cudaDeviceSynchronize();
    cudaEventRecord(ev1);
    CHK(cudaDeviceSynchronize());
    t_host = time_ms(ev0, ev1) / ITERS * 1000.0f; /* us */
    CHK(cudaGetLastError());

    /* V1: 4 launches, no sync inside */
    float t_v1 = 0;
    for (int it = 0; it < ITERS + WARMUP; it++) {
        if (it == WARMUP) { cudaDeviceSynchronize(); cudaEventRecord(ev0); }
        k_gemv_f32<<<grd_w1, blk_gemv>>>(d_W1, d_x, d_h, N_ROWS_W1, K_DIM);
        k_gelu_ple<<<grd_el, blk_el>>>(d_h, ple_row_d, d_g, N_ROWS_W1);
        k_gemv_bf16<<<grd_w2, blk_gemv>>>(d_W2, d_g, d_y, K_DIM, N_ROWS_W1);
        k_rmsnorm<<<1, 1024, 1024 * 4>>>(d_y, d_gamma, d_out_v1, K_DIM);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(ev1);
    CHK(cudaDeviceSynchronize());
    t_v1 = time_ms(ev0, ev1) / ITERS * 1000.0f;

    /* V2: 2 launches, no sync inside */
    float t_v2 = 0;
    for (int it = 0; it < ITERS + WARMUP; it++) {
        if (it == WARMUP) { cudaDeviceSynchronize(); cudaEventRecord(ev0); }
        k_ple_stage1<<<32, 256>>>(d_W1, d_x, ple_row_d, d_g);
        k_ple_stage2<<<96, 512>>>(d_W2, d_g, d_gamma, d_out_v2, K_DIM,
                                  N_ROWS_W1);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(ev1);
    CHK(cudaDeviceSynchronize());
    t_v2 = time_ms(ev0, ev1) / ITERS * 1000.0f;

    /* also: V2 back-to-back capture-friendly check (graph capture smoke) */
    cudaGraph_t graph;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    k_ple_stage1<<<32, 256, 0, s>>>(d_W1, d_x, ple_row_d, d_g);
    k_ple_stage2<<<96, 512, 0, s>>>(d_W2, d_g, d_gamma, d_out_v2, K_DIM,
                                    N_ROWS_W1);
    cudaStreamEndCapture(s, &graph);
    printf("graph capture: OK (%d nodes)\n", 2);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);

    /* report */
    const float budget_us = 3500.0f; /* ~3.5 ms/token */
    const int NL = 35;
    printf("\nper-layer latency (us):\n");
    printf("  host-assist : %8.1f\n", t_host);
    printf("  V1 (4 ker)  : %8.1f\n", t_v1);
    printf("  V2 (2 ker)  : %8.1f\n", t_v2);
    for (int c = 0; c < 3; c++) {
        float t = c == 0 ? t_host : c == 1 ? t_v1 : t_v2;
        const char *nm = c == 0 ? "host" : c == 1 ? "V1  " : "V2  ";
        float save = (t_host - t) * NL;
        printf("  %s : 35 layers = %.1f us | saved/token = %.1f us "
               "(%.2f%% of %.0f us budget)\n",
               nm, t * NL, save, save / budget_us * 100.0f, budget_us);
    }

    size_t free_b, tot_b;
    cudaMemGetInfo(&free_b, &tot_b);
    (void)free_b; (void)tot_b;
    printf("device mem used by proto: ~%zu KB\n",
           (sizeof(W1) + sizeof(W2) + sizeof(ple) + 40960) / 1024);
    return pass ? 0 : 1;
}
