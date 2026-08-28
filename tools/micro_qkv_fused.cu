// Task 1 microbench: k_gemv_q4_0_qkv_fused (one launch for Q, K, V).
//
// Q, K, V projections in a transformer all consume the SAME input
// activation X, so each layer currently launches 3 separate
// k_gemv_q4_0 calls. Each launch costs ~3 microseconds of overhead;
// per layer: 3 launches × 3 μs = 9 μs. A fused QKV kernel reduces
// this to a single launch (~3 μs), saving 6 μs/layer × 24 layers =
// 144 μs/token (≈ +20 tok/s at 250 tok/s).
//
// Design: ONE CUDA launch covers the whole QKV. Each block is
// assigned to (Q, K, or V) by blockIdx.z. The shared input X is
// loaded ONCE into shared memory at the start of the block, then
// reused 3 times (once for the per-block projection; the savings
// come from collapsing 3 launches into 1, not from any in-block
// cross-projection reuse of X — that would require a different
// block layout where one block handles all three projections of the
// same rows, which doesn't match the 4-rows-per-warp design
// constraint). Per-warp: 4 output rows of one projection, identical
// to k_gemv_q4_0_v4 already in gemv_q4_cuda.cu.
//
// Per-block shmem budget: K * 4 bytes for X (K=896 -> 3.5 KB;
// trivial vs the 100 KB/SM Ampere limit). 4 warps per block keeps
// occupancy high while leaving room for the per-warp accumulators.
//
// Constraints (caller checks before launch):
//   - K % 32 == 0
//   - nb = K/32 even (same uint32-streaming + __byte_perm contract
//     as the V2/V4 q4_0 kernels in gemv_q4_cuda.cu)
//   - M_q, M_k, M_v are multiples of 4 (caller pads and falls back
//     when not).
//
// This file is a STANDALONE MICROBENCH. It does not modify the
// engine. It defines its own copies of BlockQ4_0, k_gemv_q4_0 (the
// existing 2-rows-per-warp single-projection kernel, used as the
// "3 sequential" baseline), and the new k_gemv_q4_0_qkv_fused.
// Same pattern as tools/micro_batch4.cu / tools/micro_v4.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Block layout (matches kernels/gemv_q4_cuda.cu).
// ---------------------------------------------------------------------------
typedef struct {
    __half d;
    uint8_t qs[16];
} BlockQ4_0;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// ---------------------------------------------------------------------------
// Reference single-projection V2 kernel (2 rows/warp). Used to time
// the 3-sequential baseline.
// ---------------------------------------------------------------------------
__global__ void k_gemv_q4_0_single(const BlockQ4_0 *__restrict__ W,
                                   const float    *__restrict__ x,
                                   float          *__restrict__ y,
                                   int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x;
    const int nb   = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (18 * b + 2) >> 2;
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
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// ---------------------------------------------------------------------------
// Fused QKV kernel. ONE launch produces all of Q, K, V. The grid is
// (max_grid_x, 1, 3): blockIdx.z in {0,1,2} selects (Q,K,V). Each
// block is assigned 4 rows of its projection; the input X is loaded
// into shared memory ONCE at the start and reused for the single
// projection the block is responsible for.
//
// Per block: 4 warps × 32 lanes = 128 threads; one warp per 4 rows.
// (Same warps-per-block as the 3-seq baseline for comparable
// per-block work.)
// ---------------------------------------------------------------------------
__global__ void k_gemv_q4_0_qkv_fused(const BlockQ4_0 *__restrict__ W_q,
                                      const BlockQ4_0 *__restrict__ W_k,
                                      const BlockQ4_0 *__restrict__ W_v,
                                      const float    *__restrict__ X,
                                      float          *__restrict__ Y_q,
                                      float          *__restrict__ Y_k,
                                      float          *__restrict__ Y_v,
                                      int M_q, int M_k, int M_v, int K) {
    extern __shared__ float sx[];   // K floats
    const int lane    = threadIdx.x;
    const int warp    = threadIdx.y;
    const int z       = blockIdx.z;  // 0=Q, 1=K, 2=V

    // Resolve which weight matrix / output buffer / M for this block.
    const BlockQ4_0 *W;
    float *Y;
    int M;
    if (z == 0) { W = W_q; Y = Y_q; M = M_q; }
    else if (z == 1) { W = W_k; Y = Y_k; M = M_k; }
    else             { W = W_v; Y = Y_v; M = M_v; }

    // Per-block warp covers 4 rows of this projection.
    const int row0 = (blockIdx.x * blockDim.y + warp) * 4;
    // Defer the row-out-of-range early-return until after the
    // __syncthreads so the shmem load is collective for all warps.
    {
        const int total = K;
        const int tid   = warp * 32 + lane;
        const int nthr  = blockDim.y * 32;
        for (int i = tid; i < total; i += nthr) {
            sx[i] = X[i];
        }
    }
    __syncthreads();
    if (row0 >= M) return;
    const int row1 = row0 + 1, row2 = row0 + 2, row3 = row0 + 3;

    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)row1 * nb * 18);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)row2 * nb * 18);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)row3 * nb * 18);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const int sh  = (18 * b + 2) & 2;
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
        const float4 *x4 = (const float4 *)(sx + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = __ldg(rw0 + a0 + k);
            const uint32_t lb = __ldg(rw1 + a0 + k);
            const uint32_t lc = __ldg(rw2 + a0 + k);
            const uint32_t ld = __ldg(rw3 + a0 + k);
            const uint32_t va = sh ? __byte_perm(la, __ldg(rw0 + a0 + k + 1), 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, __ldg(rw1 + a0 + k + 1), 0x5432) : lb;
            const uint32_t vc = sh ? __byte_perm(lc, __ldg(rw2 + a0 + k + 1), 0x5432) : lc;
            const uint32_t vd = sh ? __byte_perm(ld, __ldg(rw3 + a0 + k + 1), 0x5432) : ld;
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
            s2 += (float)((int)(vc         & 0xFu) - 8) * dc * xa.x;
            s2 += (float)((int)((vc >>  4) & 0xFu) - 8) * dc * xb.x;
            s2 += (float)((int)((vc >>  8) & 0xFu) - 8) * dc * xa.y;
            s2 += (float)((int)((vc >> 12) & 0xFu) - 8) * dc * xb.y;
            s2 += (float)((int)((vc >> 16) & 0xFu) - 8) * dc * xa.z;
            s2 += (float)((int)((vc >> 20) & 0xFu) - 8) * dc * xb.z;
            s2 += (float)((int)((vc >> 24) & 0xFu) - 8) * dc * xa.w;
            s2 += (float)((int)(vc >> 28) - 8) * dc * xb.w;
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
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    s2 = warp_reduce_sum(s2);
    s3 = warp_reduce_sum(s3);
    if (lane == 0) {
        Y[row0] = s0;
        if (row1 < M) Y[row1] = s1;
        if (row2 < M) Y[row2] = s2;
        if (row3 < M) Y[row3] = s3;
    }
}

// ---------------------------------------------------------------------------
// CPU reference (scalar q4_0 dequant + dot) for correctness check.
// ---------------------------------------------------------------------------
static float cpu_dot(const BlockQ4_0 *Wrow, const float *x, int K) {
    const int nb = K / 32;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const BlockQ4_0 blk = Wrow[b];
        const float d = __half2float(blk.d);
        const float *xb = x + b * 32;
        for (int i = 0; i < 16; i++) {
            const int lo = (int)(blk.qs[i] & 0x0F) - 8;
            const int hi = (int)(blk.qs[i] >> 4) - 8;
            sum += (float)lo * d * xb[i] + (float)hi * d * xb[i + 16];
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Host launchers.
// ---------------------------------------------------------------------------
static void launch_3seq(const BlockQ4_0 *Wq, const BlockQ4_0 *Wk, const BlockQ4_0 *Wv,
                        const float *X,
                        float *Yq, float *Yk, float *Yv,
                        int M_q, int M_k, int M_v, int K,
                        cudaStream_t s) {
    const int warps = 4;
    {
        const int blocks = (M_q + warps * 2 - 1) / (warps * 2);
        dim3 grid(blocks, 1, 1);
        dim3 block(32, warps, 1);
        k_gemv_q4_0_single<<<grid, block, 0, s>>>(Wq, X, Yq, M_q, K);
    }
    {
        const int blocks = (M_k + warps * 2 - 1) / (warps * 2);
        dim3 grid(blocks, 1, 1);
        dim3 block(32, warps, 1);
        k_gemv_q4_0_single<<<grid, block, 0, s>>>(Wk, X, Yk, M_k, K);
    }
    {
        const int blocks = (M_v + warps * 2 - 1) / (warps * 2);
        dim3 grid(blocks, 1, 1);
        dim3 block(32, warps, 1);
        k_gemv_q4_0_single<<<grid, block, 0, s>>>(Wv, X, Yv, M_v, K);
    }
}

static void launch_fused(const BlockQ4_0 *Wq, const BlockQ4_0 *Wk, const BlockQ4_0 *Wv,
                         const float *X,
                         float *Yq, float *Yk, float *Yv,
                         int M_q, int M_k, int M_v, int K,
                         cudaStream_t s) {
    const int warps = 4;
    const int max_M = (M_q > M_k) ? ((M_q > M_v) ? M_q : M_v)
                                  : ((M_k > M_v) ? M_k : M_v);
    const int blocks = (max_M + warps * 4 - 1) / (warps * 4);
    dim3 grid(blocks, 1, 3);
    dim3 block(32, warps, 1);
    const size_t shmem = (size_t)K * sizeof(float);
    k_gemv_q4_0_qkv_fused<<<grid, block, shmem, s>>>(Wq, Wk, Wv, X, Yq, Yk, Yv,
                                                      M_q, M_k, M_v, K);
}

// ---------------------------------------------------------------------------
// Microbench driver.
// ---------------------------------------------------------------------------
struct Shape {
    const char *name;
    int M_q, M_k, M_v, K;
};

static float time_path(int which, const Shape &sh, const BlockQ4_0 *dWq,
                       const BlockQ4_0 *dWk, const BlockQ4_0 *dWv,
                       const float *dX, float *dYq, float *dYk, float *dYv,
                       int iters, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    // Warmup
    for (int i = 0; i < 10; i++) {
        if (which == 0) launch_3seq(dWq, dWk, dWv, dX, dYq, dYk, dYv,
                                    sh.M_q, sh.M_k, sh.M_v, sh.K, s);
        else            launch_fused(dWq, dWk, dWv, dX, dYq, dYk, dYv,
                                    sh.M_q, sh.M_k, sh.M_v, sh.K, s);
    }
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; i++) {
        if (which == 0) launch_3seq(dWq, dWk, dWv, dX, dYq, dYk, dYv,
                                    sh.M_q, sh.M_k, sh.M_v, sh.K, s);
        else            launch_fused(dWq, dWk, dWv, dX, dYq, dYk, dYv,
                                    sh.M_q, sh.M_k, sh.M_v, sh.K, s);
    }
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

int main(int argc, char **argv) {
    int iters = 200;
    if (argc > 1) iters = atoi(argv[1]);

    printf("micro_qkv_fused: %d timed iters per path\n", iters);

    // qwen2.5-0.5b layer shapes: Q head = 14*64 = 896; KV = 2*64 = 128.
    std::vector<Shape> shapes = {
        {"qwen2.5-0.5b layer", 896, 128, 128, 896},
    };

    // RNG seed for reproducible activations/weights.
    srand(42);

    for (const Shape &sh : shapes) {
        printf("\n=== shape: %s  M_q=%d M_k=%d M_v=%d K=%d ===\n",
               sh.name, sh.M_q, sh.M_k, sh.M_v, sh.K);

        const int nb = sh.K / 32;
        // q4_0 row size in elements.
        const size_t row_bytes = (size_t)nb * sizeof(BlockQ4_0);
        // Host weight matrices (synthesised; non-zero but reproducible).
        std::vector<BlockQ4_0> hWq(sh.M_q * nb);
        std::vector<BlockQ4_0> hWk(sh.M_k * nb);
        std::vector<BlockQ4_0> hWv(sh.M_v * nb);
        for (auto &b : hWq) { b.d = __float2half(0.05f); for (int j = 0; j < 16; j++) b.qs[j] = (uint8_t)rand(); }
        for (auto &b : hWk) { b.d = __float2half(0.05f); for (int j = 0; j < 16; j++) b.qs[j] = (uint8_t)rand(); }
        for (auto &b : hWv) { b.d = __float2half(0.05f); for (int j = 0; j < 16; j++) b.qs[j] = (uint8_t)rand(); }
        std::vector<float> hX(sh.K);
        for (auto &v : hX) v = (float)(rand() % 1000) / 1000.0f - 0.5f;

        // Device buffers.
        BlockQ4_0 *dWq, *dWk, *dWv;
        float *dX, *dYq, *dYk, *dYv;
        cudaMalloc(&dWq, hWq.size() * sizeof(BlockQ4_0));
        cudaMalloc(&dWk, hWk.size() * sizeof(BlockQ4_0));
        cudaMalloc(&dWv, hWv.size() * sizeof(BlockQ4_0));
        cudaMalloc(&dX,  hX.size() * sizeof(float));
        cudaMalloc(&dYq, sh.M_q * sizeof(float));
        cudaMalloc(&dYk, sh.M_k * sizeof(float));
        cudaMalloc(&dYv, sh.M_v * sizeof(float));
        cudaMemcpy(dWq, hWq.data(), hWq.size() * sizeof(BlockQ4_0), cudaMemcpyHostToDevice);
        cudaMemcpy(dWk, hWk.data(), hWk.size() * sizeof(BlockQ4_0), cudaMemcpyHostToDevice);
        cudaMemcpy(dWv, hWv.data(), hWv.size() * sizeof(BlockQ4_0), cudaMemcpyHostToDevice);
        cudaMemcpy(dX,  hX.data(),  hX.size()  * sizeof(float),        cudaMemcpyHostToDevice);
        cudaMemset(dYq, 0, sh.M_q * sizeof(float));
        cudaMemset(dYk, 0, sh.M_k * sizeof(float));
        cudaMemset(dYv, 0, sh.M_v * sizeof(float));

        // Time both paths.
        cudaStream_t s;
        cudaStreamCreate(&s);
        float t3 = time_path(0, sh, dWq, dWk, dWv, dX, dYq, dYk, dYv, iters, s);
        float tf = time_path(1, sh, dWq, dWk, dWv, dX, dYq, dYk, dYv, iters, s);
        cudaStreamDestroy(s);

        // Correctness check: take a few rows of each output and compare
        // against the CPU scalar reference. The fused kernel uses the
        // SAME nibble-pairing as k_gemv_q4_0_single (both ports of the
        // 4-rows-per-warp layout from gemv_q4_cuda.cu), so the values
        // should match to within float-rounding error.
        std::vector<float> hYq3(sh.M_q), hYk3(sh.M_k), hYv3(sh.M_v);
        std::vector<float> hYqf(sh.M_q), hYkf(sh.M_k), hYvf(sh.M_v);
        // Re-run each path once with the default stream to grab the result.
        launch_3seq(dWq, dWk, dWv, dX, dYq, dYk, dYv, sh.M_q, sh.M_k, sh.M_v, sh.K, 0);
        cudaMemcpy(hYq3.data(), dYq, sh.M_q * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hYk3.data(), dYk, sh.M_k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hYv3.data(), dYv, sh.M_v * sizeof(float), cudaMemcpyDeviceToHost);
        launch_fused(dWq, dWk, dWv, dX, dYq, dYk, dYv, sh.M_q, sh.M_k, sh.M_v, sh.K, 0);
        cudaMemcpy(hYqf.data(), dYq, sh.M_q * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hYkf.data(), dYk, sh.M_k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hYvf.data(), dYv, sh.M_v * sizeof(float), cudaMemcpyDeviceToHost);

        // Cross-check 3-seq vs fused + CPU.
        double max_err_q = 0, max_err_k = 0, max_err_v = 0;
        double max_3vsf_q = 0, max_3vsf_k = 0, max_3vsf_v = 0;
        for (int m = 0; m < sh.M_q; m++) {
            float ref = cpu_dot(hWq.data() + (long)m * nb, hX.data(), sh.K);
            max_err_q = std::max(max_err_q, (double)std::abs(hYqf[m] - ref));
            max_3vsf_q = std::max(max_3vsf_q, (double)std::abs(hYq3[m] - hYqf[m]));
        }
        for (int m = 0; m < sh.M_k; m++) {
            float ref = cpu_dot(hWk.data() + (long)m * nb, hX.data(), sh.K);
            max_err_k = std::max(max_err_k, (double)std::abs(hYkf[m] - ref));
            max_3vsf_k = std::max(max_3vsf_k, (double)std::abs(hYk3[m] - hYkf[m]));
        }
        for (int m = 0; m < sh.M_v; m++) {
            float ref = cpu_dot(hWv.data() + (long)m * nb, hX.data(), sh.K);
            max_err_v = std::max(max_err_v, (double)std::abs(hYvf[m] - ref));
            max_3vsf_v = std::max(max_3vsf_v, (double)std::abs(hYv3[m] - hYvf[m]));
        }

        printf("time_3seq  = %.3f us/iter\n", t3 * 1e3f);
        printf("time_fused = %.3f us/iter\n", tf * 1e3f);
        printf("savings    = %.3f us/iter (%.1f%%)\n",
               (t3 - tf) * 1e3f, (t3 - tf) / t3 * 100.0);
        printf("max |fused - cpu|  Q=%.2e K=%.2e V=%.2e\n",
               max_err_q, max_err_k, max_err_v);
        printf("max |3seq - fused| Q=%.2e K=%.2e V=%.2e\n",
               max_3vsf_q, max_3vsf_k, max_3vsf_v);

        cudaFree(dWq); cudaFree(dWk); cudaFree(dWv);
        cudaFree(dX);
        cudaFree(dYq); cudaFree(dYk); cudaFree(dYv);
    }

    return 0;
}
