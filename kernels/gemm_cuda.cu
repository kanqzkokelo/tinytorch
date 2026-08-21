// M3: CUDA sgemm ladder — naive -> 2D 64x64 8x8-tiled shared-memory -> fp16 WMMA.
// All kernels compute row-major C[MxN] = A[MxK] @ B[KxN].
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return (int)e_; } while (0)

/* ---------------- naive: one thread per output (uncoalesced A loads) ---------------- */
__global__ void k_sgemm_naive(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C,
                              int M, int N, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;
    float s = 0.0f;
    for (int k = 0; k < K; k++)
        s += A[(long)i * K + k] * B[(long)k * N + j];
    C[(long)i * N + j] = s;
}

/* ------- 2D register-tiled 64x64 tile, 8x8 subtile per thread, BK=16 ------- */
#define BM 64
#define BN 64
#define BK 16

__global__ void k_sgemm_tiled_64x64(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C,
                                    int M, int N, int K) {
    // 8x8 threads per block = 64 threads
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 8 + tx; // 0..63

    // Shared memory for 64x16 A tile and 16x64 B tile
    __shared__ float As[BK][BM + 4]; // padded to avoid bank conflicts
    __shared__ float Bs[BK][BN + 4];

    // 8x8 register accumulator tile per thread (64 floats)
    float accum[8][8] = {0.0f};

    // Load coordinates: 64 threads load 64x16 = 1024 floats -> 4 float4s per thread (16 floats per thread)
    // A tile is BK=16 rows, BM=64 cols
    // B tile is BK=16 rows, BN=64 cols

    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    for (int bk = 0; bk < K; bk += BK) {
        // Load A: 64 threads load 16x64 floats (1024 floats)
        // Each thread loads 4 float4s (16 floats)
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int e = p * 64 + tid; // 0..255 (in units of float4) -> 256 float4s
            int a_row = e / 16;   // 0..15 (k-index)
            int a_col = (e % 16) * 4; // 0..60 (m-index)

            int g_a_row = by * BM + a_col;
            int g_a_col = bk + a_row;
            if (g_a_row < M && g_a_col < K) {
                As[a_row][a_col] = A[(long)g_a_row * K + g_a_col];
            } else {
                As[a_row][a_col] = 0.0f;
            }
            if (g_a_row + 1 < M && g_a_col < K) {
                As[a_row][a_col + 1] = A[(long)(g_a_row + 1) * K + g_a_col];
            } else {
                As[a_row][a_col + 1] = 0.0f;
            }
            if (g_a_row + 2 < M && g_a_col < K) {
                As[a_row][a_col + 2] = A[(long)(g_a_row + 2) * K + g_a_col];
            } else {
                As[a_row][a_col + 2] = 0.0f;
            }
            if (g_a_row + 3 < M && g_a_col < K) {
                As[a_row][a_col + 3] = A[(long)(g_a_row + 3) * K + g_a_col];
            } else {
                As[a_row][a_col + 3] = 0.0f;
            }
        }

        // Load B: 64 threads load 16x64 floats (1024 floats)
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int e = p * 64 + tid; // 0..255 (units of 4 floats)
            int b_row = e / 16;   // 0..15 (k-index)
            int b_col = (e % 16) * 4; // 0..60 (n-index)

            int g_b_row = bk + b_row;
            int g_b_col = bx * BN + b_col;

            if (g_b_row < K && g_b_col + 3 < N) {
                float4 tmpB = *reinterpret_cast<const float4*>(&B[(long)g_b_row * N + g_b_col]);
                Bs[b_row][b_col + 0] = tmpB.x;
                Bs[b_row][b_col + 1] = tmpB.y;
                Bs[b_row][b_col + 2] = tmpB.z;
                Bs[b_row][b_col + 3] = tmpB.w;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    Bs[b_row][b_col + j] = (g_b_row < K && g_b_col + j < N)
                                              ? B[(long)g_b_row * N + g_b_col + j] : 0.0f;
                }
            }
        }

        __syncthreads();

        // Inner tile computation: BK=16 k-steps
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[8];
            float regB[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                regA[i] = As[k][ty * 8 + i];
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                regB[j] = Bs[k][tx * 8 + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back 8x8 subtile to C
    const int c_row_base = by * BM + ty * 8;
    const int c_col_base = bx * BN + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int g_c_row = c_row_base + i;
        if (g_c_row < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int g_c_col = c_col_base + j;
                if (g_c_col < N) {
                    C[(long)g_c_row * N + g_c_col] = accum[i][j];
                }
            }
        }
    }
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
    dim3 b(16, 16), g((M + 15) / 16, (N + 15) / 16);
    k_sgemm_naive<<<g, b>>>(dA, dB, dC, M, N, K);
    return (int)cudaGetLastError();
}

static int run_tiled(const float *dA, const float *dB, float *dC,
                     int M, int N, int K) {
    dim3 b(8, 8), g((N + BN - 1) / BN, (M + BM - 1) / BM);
    k_sgemm_tiled_64x64<<<g, b>>>(dA, dB, dC, M, N, K);
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
