// MMQ v1: WMMA m16n16k16 fp16->fp32 q4_0 GEMV for single-token decode.
//
// Compares against the scalar V2 q4_0 GEMV in kernels/gemv_q4_cuda.cu
// (k_gemv_q4_0 + k_logits_q4_0_v2, one-warp-per-row or two-rows-per-warp).
//
// Design (per Qwen sanity check 2026-08-26):
//   * One warp = one (M=16, K=16) tile. The single-token case is padded
//     to 16 with zero-result rows; we only WRITE y[0] (and y[1..15] if
//     the caller asks for more output rows, e.g. logits head M=151936).
//   * Per K-tile: dequantize 16 rows of q4_0 -> fp16 A (row-major, ldim=16),
//     build fp16 B (row-major, ldim=16) with all 16 N cols = x[k0..k0+15].
//   * mma_sync into fp32 accumulator. After K loop, store 16x16 to shmem
//     and write column 0 (or all 16 cols if requested) to y[].
//   * One CTA = one block of (M_tile=16, K) = one warp doing one
//     16-row M-slice. For M=151936 we launch 151936/16 = 9496 blocks.
//   * For M<16 (the typical layer GEMV with M=896 or M=4864), we process
//     all 16 rows but only the first M contribute to y[]. M is padded
//     to 16 by the launcher.
//
// Why this should beat V2 scalar at M=151936 (LM head):
//   V2 does one row per 2 lanes (32-thread warp covers 2 rows sharing
//   x float4 loads). At M=151936 we launch ~4748 warps; each warp reads
//   ~28 q4_0 blocks of W and reuses x 28 times. The M=151936 case is
//   bandwidth-bound on W and 1.0-1.3 ms.
//   MMQ v1 amortizes dequant over 16 M-rows per warp with one mma per
//   K-tile. For M=151936 the K dim is small (896) so 151936/16 = 9496
//   warps is plenty; weight bandwidth dominates.
//
// Why this might NOT beat V2 at M=896 (Q proj):
//   At M=896, only 56 warps are launched for MMQ v1 vs 448 warps for V2.
//   V2's 8x more warps better saturate the SMs (RTX 3050 has 16 SMs).
//   We pad M to 16 internally so each warp does 16 rows of work that
//   may produce 0/1 useful outputs. The cost-per-useful-row is ~16x the
//   cost-per-row of V2. Expected to be SLOWER at small M.
//
// Built with -gencode arch=compute_86,code=sm_86 (Ampere consumer).
//
// Q4_0 block layout (GGML): fp16 d + 16 bytes qs (32 packed nibbles).
//   dequant: x[j]   = ((qs[j]   & 0xF) - 8) * d   j in [0,15]
//            x[j+16]= ((qs[j] >> 4)    - 8) * d

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

typedef struct {
    __half d;
    uint8_t qs[16];
} BlockQ4_0;

// ---------- V2 reference (copied verbatim from kernels/gemv_q4_cuda.cu) ----
// Used as the speed/accuracy baseline.
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void k_gemv_q4_0_v2(const BlockQ4_0 *W, const float *x,
                              float *y, int M, int K) {
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
            s0 += (float)((int)(va >> 28)        - 8) * da * xb.w;
            s1 += (float)((int)(vb         & 0xFu) - 8) * db * xa.x;
            s1 += (float)((int)((vb >>  4) & 0xFu) - 8) * db * xb.x;
            s1 += (float)((int)((vb >>  8) & 0xFu) - 8) * db * xa.y;
            s1 += (float)((int)((vb >> 12) & 0xFu) - 8) * db * xb.y;
            s1 += (float)((int)((vb >> 16) & 0xFu) - 8) * db * xa.z;
            s1 += (float)((int)((vb >> 20) & 0xFu) - 8) * db * xb.z;
            s1 += (float)((int)((vb >> 24) & 0xFu) - 8) * db * xa.w;
            s1 += (float)((int)(vb >> 28)        - 8) * db * xb.w;
        }
    }
    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row1 < M) y[row1] = s1;
    }
}

// ---------- MMQ v1: WMMA m16n16k16 fp16->fp32 q4_0 GEMV ------------------
// One CTA = WARPS_PER_BLOCK warps. Each warp = one (M=16, K) tile.
// All 16 N cols = same x broadcast (waste 15/16 mma, amortize via M).
// After K loop, take N=0 column of the 16x16 c_frag as 16 outputs.
// WARPS_PER_BLOCK warps share x in shmem (read once, reused) but each
// warp owns its own sW/sB/sC; shmem scales linearly with WARPS_PER_BLOCK.
//
// K-iteration: process 32 K-elements (one q4_0 block per M-row) per
// outer iter, dequant 32 fp16 values into sW (16 rows x 32 cols) using
// ONE read of each q4_0 block. Then do 2 mma_sync (k=0..15, k=16..31)
// into the same c_frag. This eliminates the 2x W-bandwidth waste from
// the 16-K-tile-per-q4_0-block case (each block now read once, used
// twice for 2 mma's instead of once).
#define WARPS_PER_BLOCK 4
__global__ void k_mmq_v1_q4_0(const BlockQ4_0 *W, const float *x,
                              float *y, int M, int K) {
    using namespace nvcuda;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tile_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int row0    = tile_id * 16;
    if (row0 >= M) return;
    const int nb      = K / 32;                  // q4_0 blocks per row
    const int n_outer = K / 32;                  // outer K-tiles (32 elem each)

    // Shmem layout (dynamic):
    //   sx     : K fp16 (shared)
    //   sW     : WARPS_PER_BLOCK * 16*32 fp16   (16 rows x 32 K cols)
    //   sB     : WARPS_PER_BLOCK * 16*16 fp16   (B tile, same as before)
    //   sC     : WARPS_PER_BLOCK * 16*16 fp32
    extern __shared__ __half smem[];
    __half *sx = smem;
    __half *sW_base = smem + K;                  // per-warp offset = warp_id * 512
    __half *sB_base = sW_base + WARPS_PER_BLOCK * 512;
    float  *sC_base = (float *)(sB_base + WARPS_PER_BLOCK * 256);
    __half *sW = sW_base + warp_id * 512;        // 16 rows x 32 K-cols
    __half *sB = sB_base + warp_id * 256;
    float  *sC = sC_base + warp_id * 256;

    if (warp_id == 0)
        for (int i = lane; i < K; i += 32) sx[i] = __float2half(x[i]);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int t = 0; t < n_outer; t++) {
        const int k0_outer = t * 32;             // start of this q4_0 block
        const int blk_idx  = t;                  // q4_0 block index in row

        // Dequant A tile: 16 M-rows x 32 K-elements (1 full q4_0 block per
        // row) -> fp16 row-major 16x32. Each lane handles 512/32 = 16
        // elements. row = i/32, col = i%32.
        for (int i = lane; i < 16 * 32; i += 32) {
            const int row = i / 32;
            const int col = i % 32;              // 0..31 within block
            const int q_byte = col & 15;         // 0..15 qs[] index
            const int nib_shift = (col < 16) ? 0 : 4;
            const BlockQ4_0 *blk = W + (long)(row0 + row) * nb + blk_idx;
            const float d = __half2float(blk->d);
            const int nib = (blk->qs[q_byte] >> nib_shift) & 0xF;
            const float w = ((float)nib - 8.f) * d;
            sW[i] = __float2half(w);
        }
        __syncwarp();

        // Two mma's: k=0..15 and k=16..31. sW ldim = 32.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag1;

        // First mma: A = sW[:, 0..15], B = sx[k0_outer+0..15] broadcast.
        // Build B (only the first 16 of 32 K-elements for this mma).
        for (int i = lane; i < 16 * 16; i += 32) {
            const int k = i / 16;
            sB[i] = sx[k0_outer + k];
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag0;
        // ldim for a_frag0: row-major 16x32 storage, slice first 16 cols.
        // load_matrix_sync needs a pointer to the start of the 16x16 tile.
        wmma::load_matrix_sync(a_frag0, sW, 32);
        wmma::load_matrix_sync(b_frag0, sB, 16);
        wmma::mma_sync(c_frag, a_frag0, b_frag0, c_frag);

        // Second mma: A = sW[:, 16..31], B = sx[k0_outer+16..31] broadcast.
        for (int i = lane; i < 16 * 16; i += 32) {
            const int k = i / 16;
            sB[i] = sx[k0_outer + 16 + k];
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag1;
        wmma::load_matrix_sync(a_frag1, sW + 16, 32);  // pointer to col 16 of sW
        wmma::load_matrix_sync(b_frag1, sB, 16);
        wmma::mma_sync(c_frag, a_frag1, b_frag1, c_frag);
    }

    wmma::store_matrix_sync(sC, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    if (lane < 16) {
        const int r = lane;
        if (row0 + r < M) y[row0 + r] = sC[r * 16 + 0];
    }
}

// ------------------- harness -------------------

static void v2_launch(const BlockQ4_0 *W, const float *x, float *y,
                      int M, int K, cudaStream_t s) {
    // Same launcher shape as tt_gemv_q4_0 in the engine (16 warps per block).
    dim3 block(32, 16, 1);
    dim3 grid((M + 31) / 32, 1, 1);
    k_gemv_q4_0_v2<<<grid, block, 0, s>>>(W, x, y, M, K);
}

static void mmq_launch(const BlockQ4_0 *W, const float *x, float *y,
                       int M, int K, cudaStream_t s) {
    const int M_p = (M + 15) & ~15;
    dim3 block(32 * WARPS_PER_BLOCK, 1, 1);
    dim3 grid((M_p / 16 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);
    // shmem: K*2 (sx) + WARPS_PER_BLOCK * (512*2 + 256*2 + 256*4) bytes
    int shmem = K * 2
              + WARPS_PER_BLOCK * (512 * 2 + 256 * 2 + 256 * 4);
    k_mmq_v1_q4_0<<<grid, block, shmem, s>>>(W, x, y, M, K);
}

static float time_kernel_us(cudaStream_t s, int reps,
                            void (*launch)(const BlockQ4_0 *, const float *,
                                           float *, int, int, cudaStream_t),
                            const BlockQ4_0 *W, const float *x, float *y,
                            int M, int K) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    // warmup
    for (int i = 0; i < 5; i++) launch(W, x, y, M, K, s);
    cudaStreamSynchronize(s);
    cudaEventRecord(a, s);
    for (int i = 0; i < reps; i++) launch(W, x, y, M, K, s);
    cudaEventRecord(b, s); cudaStreamSynchronize(s);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (ms / reps) * 1e3f;                  // us
}

int main(int argc, char **argv) {
    // Default shapes to test (qwen2.5 0.5B sizes, plus small sanity).
    struct Shape { int M, K; const char *name; };
    Shape shapes[] = {
        {896,   896,   "qwen2.5 Q/K/V proj"},
        {4864,  896,   "qwen2.5 FFN up/gate"},
        {896,   4864,  "qwen2.5 FFN down"},
        {151936, 896,  "qwen2.5 LM head (vocab)"},
    };
    int nshapes = sizeof(shapes) / sizeof(shapes[0]);
    if (argc > 1) nshapes = 1;
    int shape_idx = (argc > 1) ? atoi(argv[1]) : -1;

    int REPS = 100;

    for (int si = 0; si < nshapes; si++) {
        int sidx = (shape_idx >= 0) ? shape_idx : si;
        if (sidx < 0 || sidx >= (int)(sizeof(shapes)/sizeof(shapes[0]))) continue;
        Shape sh = shapes[sidx];
        int M = sh.M, K = sh.K;
        int nb = K / 32;
        if ((K & 15) != 0) { fprintf(stderr, "K must be %% 16, got %d\n", K); continue; }

        size_t wbytes = (size_t)M * nb * 18;
        size_t xbytes = (size_t)K * 4;
        size_t ybytes = (size_t)M * 4;

        std::vector<uint8_t> hW(wbytes, 0xab);
        // Deterministic fill: nibbles 0..15 evenly, scale = fp16(0.05)
        // (raw 0x2A66 LE: 0x2A66 in fp16 = 0.05). AVOID random fp16
        // bytes for d because bit patterns like 0x7e/0x7f can produce
        // NaN/Inf which contaminates the correctness comparison.
        for (size_t i = 0; i < wbytes; i += 18) {
            // d = fp16(0.05) = 0x2A66
            hW[i + 0] = 0x66; hW[i + 1] = 0x2A;
            // qs[0..15]: 0..15 cycle (low nibble) | 15..0 cycle (high nibble)
            for (int j = 0; j < 16; j++) {
                hW[i + 2 + j] = (uint8_t)((j & 0xF) | ((15 - j) << 4));
            }
        }

        std::vector<float> hx(K);
        for (int i = 0; i < K; i++) hx[i] = sinf(0.7f * i + 0.3f);
        std::vector<float> hy_v2(M, 0.0f), hy_mmq(M, 0.0f);

        void *dW, *dx, *dy;
        cudaMalloc(&dW, wbytes);
        cudaMalloc(&dx, xbytes);
        cudaMalloc(&dy, ybytes);
        cudaMemcpy(dW, hW.data(), wbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx.data(), xbytes, cudaMemcpyHostToDevice);

        cudaStream_t s; cudaStreamCreate(&s);

        float us_v2  = time_kernel_us(s, REPS, v2_launch,
                                      (const BlockQ4_0 *)dW, (const float *)dx,
                                      (float *)dy, M, K);
        cudaMemcpy(hy_v2.data(), dy, ybytes, cudaMemcpyDeviceToHost);

        float us_mmq = time_kernel_us(s, REPS, mmq_launch,
                                      (const BlockQ4_0 *)dW, (const float *)dx,
                                      (float *)dy, M, K);
        cudaMemcpy(hy_mmq.data(), dy, ybytes, cudaMemcpyDeviceToHost);

        // Correctness: max abs error
        double maxd = 0; double rowmax_v2 = 0;
        for (int r = 0; r < M; r++) {
            double d = fabs((double)hy_v2[r] - hy_mmq[r]);
            if (d > maxd) maxd = d;
            if (fabs(hy_v2[r]) > rowmax_v2) rowmax_v2 = fabs(hy_v2[r]);
        }
        double tol = 1e-2 * (rowmax_v2 > 1e-6 ? rowmax_v2 : 1e-6);
        const char *status = (maxd <= tol) ? "OK" : "MISMATCH";

        // Bandwidth: weights are the dominant data (x is reused M times).
        double bw_v2  = (double)wbytes / (us_v2  * 1e-6) / 1e9;
        double bw_mmq = (double)wbytes / (us_mmq * 1e-6) / 1e9;
        double ratio = us_v2 / us_mmq;           // >1 = MMQ faster

        printf("[%s] M=%-6d K=%-5d  V2 %.1f us  MMQ %.1f us  ratio %.2fx  "
               "BW V2=%.0f MMQ=%.0f GB/s  |err|=%.2e (tol=%.2e) %s\n",
               sh.name, M, K, us_v2, us_mmq, ratio, bw_v2, bw_mmq, maxd, tol, status);

        cudaStreamDestroy(s);
        cudaFree(dW); cudaFree(dx); cudaFree(dy);
    }
    return 0;
}
