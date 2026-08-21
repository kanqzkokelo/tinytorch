// M3: CUDA sgemm ladder — naive -> tiled shared-memory -> fp16 WMMA.
// All kernels compute row-major C[MxN] = A[MxK] @ B[KxN].
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return (int)e_; } while (0)

/* ---------------- naive: one thread per output ---------------- */
__global__ void k_sgemm_naive(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C,
                              int M, int N, int K) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;
    float s = 0.0f;
    for (int k = 0; k < K; k++)
        s += A[(long)i * K + k] * B[(long)k * N + j];
    C[(long)i * N + j] = s;
}

/* ------- tiled shared memory: 64x64 tile, 16x16 threads, 4x4 subtile ---- */
#define TS 64
__global__ void k_sgemm_tiled(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C,
                              int M, int N, int K) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    const int tx = threadIdx.x, ty = threadIdx.y;   /* 16x16 threads */
    const int row = blockIdx.y * TS + ty * 4;
    const int col = blockIdx.x * TS + tx * 4;
    float r[4][4] = {};
    const int lid = ty * 16 + tx;                   /* 0..255 */

    for (int t = 0; t < K; t += TS) {
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            int e = p * 256 + lid;                  /* 0..4095 */
            int lr = e >> 6, lc = e & 63;
            int gr = blockIdx.y * TS + lr;
            int gc = blockIdx.x * TS + lc;
            As[lr][lc] = (gr < M && t + lc < K)
                             ? A[(long)gr * K + t + lc] : 0.0f;
            Bs[lr][lc] = (t + lr < K && gc < N)
                             ? B[(long)(t + lr) * N + gc] : 0.0f;
        }
        __syncthreads();
        #pragma unroll 8
        for (int kk = 0; kk < TS; kk++) {
            float av[4], bv[4];
            #pragma unroll
            for (int p = 0; p < 4; p++) av[p] = As[ty * 4 + p][kk];
            #pragma unroll
            for (int q = 0; q < 4; q++) bv[q] = Bs[kk][tx * 4 + q];
            #pragma unroll
            for (int p = 0; p < 4; p++)
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    r[p][q] += av[p] * bv[q];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int p = 0; p < 4; p++)
        #pragma unroll
        for (int q = 0; q < 4; q++)
            if (row + p < M && col + q < N)
                C[(long)(row + p) * N + col + q] = r[p][q];
}

/* ---------------- fp32 -> fp16 conversion ---------------- */
__global__ void f2h(const float *__restrict__ f, half *__restrict__ h,
                    long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2half(f[i]);
}

/* ---------------- WMMA fp16 in / fp32 accumulate ---------------- */
#define WM 16
__global__ void k_sgemm_wmma(const half *__restrict__ Ah,
                             const half *__restrict__ Bh,
                             float *__restrict__ C,
                             int M, int N, int K) {
    wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    const int warpM = (threadIdx.x / warpSize) % 2;
    const int warpN = (threadIdx.x / warpSize) / 2;
    const int row = (blockIdx.y * 2 + warpM) * WM;
    const int col = (blockIdx.x * 2 + warpN) * WM;

    wmma::fragment<wmma::matrix_a, WM, WM, WM, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, WM, WM, WM, half, wmma::row_major> fb;

    for (int t = 0; t < K; t += WM) {
        wmma::load_matrix_sync(fa, Ah + (long)row * K + t, K);
        wmma::load_matrix_sync(fb, Bh + (long)t * N + col, N);
        wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(C + (long)row * N + col, acc, N,
                            wmma::mem_row_major);
}

/* ---------------- host wrappers ---------------- */
static int run_naive(const float *dA, const float *dB, float *dC,
                     int M, int N, int K) {
    dim3 b(16, 16), g((N + 15) / 16, (M + 15) / 16);
    k_sgemm_naive<<<g, b>>>(dA, dB, dC, M, N, K);
    return (int)cudaGetLastError();
}

static int run_tiled(const float *dA, const float *dB, float *dC,
                     int M, int N, int K) {
    dim3 b(16, 16), g((N + TS - 1) / TS, (M + TS - 1) / TS);
    k_sgemm_tiled<<<g, b>>>(dA, dB, dC, M, N, K);
    return (int)cudaGetLastError();
}

static int run_wmma(const float *dAf, const float *dBf, float *dC,
                    int M, int N, int K) {
    if (M % 16 || N % 16 || K % 16) return -100;
    half *dAh, *dBh;
    CHK(cudaMalloc(&dAh, (size_t)M * K * sizeof(half)));
    CHK(cudaMalloc(&dBh, (size_t)K * N * sizeof(half)));
    const int conv = 256;
    const long nf = (long)M * K, nb = (long)K * N;
    f2h<<<(int)((nf + conv - 1) / conv), conv>>>(dAf, dAh, nf);
    f2h<<<(int)((nb + conv - 1) / conv), conv>>>(dBf, dBh, nb);
    CHK(cudaGetLastError());
    dim3 b(128, 1), g(N / (2 * WM), M / (2 * WM));
    k_sgemm_wmma<<<g, b>>>((half *)dAh, (half *)dBh, dC, M, N, K);
    const int err = (int)cudaGetLastError();
    cudaFree(dAh);
    cudaFree(dBh);
    return err;
}

extern "C" {

int tt_cuda_alloc(float **p, size_t bytes) { return (int)cudaMalloc(p, bytes); }
void tt_cuda_free(float *p) { cudaFree(p); }
int tt_cuda_h2d(float *d, const float *h, size_t bytes) {
    return (int)cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
}
int tt_cuda_d2h(float *h, const float *d, size_t bytes) {
    return (int)cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
}
int tt_cuda_sync(void) { return (int)cudaDeviceSynchronize(); }

int tt_cuda_sgemm_naive(const float *dA, const float *dB, float *dC,
                        int M, int N, int K) {
    return run_naive(dA, dB, dC, M, N, K);
}
int tt_cuda_sgemm_tiled(const float *dA, const float *dB, float *dC,
                        int M, int N, int K) {
    return run_tiled(dA, dB, dC, M, N, K);
}
int tt_cuda_sgemm_wmma(const float *dA, const float *dB, float *dC,
                       int M, int N, int K) {
    return run_wmma(dA, dB, dC, M, N, K);
}

} /* extern "C" */
