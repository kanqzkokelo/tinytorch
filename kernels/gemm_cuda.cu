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

/* ------- 2D register-tiled 128x128 tile, 8x8 subtile per thread, BK=16, Double Buffered ------- */
#define BM 128
#define BN 128
#define BK 16

__global__ void k_sgemm_tiled_128x128_db(const float *__restrict__ A,
                                         const float *__restrict__ B,
                                         float *__restrict__ C,
                                         int M, int N, int K) {
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int tid = threadIdx.y * 16 + threadIdx.x; // 256 threads

    __shared__ alignas(16) float As[2][BK][BM + 4];
    __shared__ alignas(16) float Bs[2][BK][BN + 16];

    float accum[8][8] = {0.0f};
    int write_stage = 0;

    const int a_load_row = tid / 4;       // 0..63
    const int a_load_col = (tid % 4) * 4; // 0,4,8,12

    const int b_load_row = tid / 32;      // 0..7
    const int b_load_col = (tid % 32) * 4;// 0,4,8..124

    // Initial stage 0 load using float4
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int r = a_load_row + i * 64;
        int c = a_load_col;
        int gr = by * BM + r;
        int gc = c;
        float4 vA = (gr < M && gc + 3 < K) ? *reinterpret_cast<const float4*>(&A[(long)gr * K + gc]) : make_float4(0,0,0,0);
        As[0][c + 0][r] = vA.x;
        As[0][c + 1][r] = vA.y;
        As[0][c + 2][r] = vA.z;
        As[0][c + 3][r] = vA.w;
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int r = b_load_row + i * 8;
        int c = b_load_col;
        int gr = r;
        int gc = bx * BN + c;
        float4 vB = (gr < K && gc + 3 < N) ? *reinterpret_cast<const float4*>(&B[(long)gr * N + gc]) : make_float4(0,0,0,0);
        Bs[0][r][c + 0] = vB.x;
        Bs[0][r][c + 1] = vB.y;
        Bs[0][r][c + 2] = vB.z;
        Bs[0][r][c + 3] = vB.w;
    }

    __syncthreads();

    // Main K Loop
    for (int bk = BK; bk < K; bk += BK) {
        int read_stage = write_stage;
        write_stage = 1 - write_stage;

        // Prefetch next tile with float4
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = a_load_row + i * 64;
            int c = a_load_col;
            int gr = by * BM + r;
            int gc = bk + c;
            float4 vA = (gr < M && gc + 3 < K) ? *reinterpret_cast<const float4*>(&A[(long)gr * K + gc]) : make_float4(0,0,0,0);
            As[write_stage][c + 0][r] = vA.x;
            As[write_stage][c + 1][r] = vA.y;
            As[write_stage][c + 2][r] = vA.z;
            As[write_stage][c + 3][r] = vA.w;
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = b_load_row + i * 8;
            int c = b_load_col;
            int gr = bk + r;
            int gc = bx * BN + c;
            float4 vB = (gr < K && gc + 3 < N) ? *reinterpret_cast<const float4*>(&B[(long)gr * N + gc]) : make_float4(0,0,0,0);
            Bs[write_stage][r][c + 0] = vB.x;
            Bs[write_stage][r][c + 1] = vB.y;
            Bs[write_stage][r][c + 2] = vB.z;
            Bs[write_stage][r][c + 3] = vB.w;
        }

        // Compute current stage
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[8];
            float regB[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) regA[i] = As[read_stage][k][threadIdx.y * 8 + i];
            
            #pragma unroll
            for (int j = 0; j < 8; j++) regB[j] = Bs[read_stage][k][threadIdx.x * 8 + j];

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

    // Compute last stage
    #pragma unroll
    for (int k = 0; k < BK; k++) {
        float regA[8];
        float regB[8];

        #pragma unroll
        for (int i = 0; i < 8; i++) regA[i] = As[write_stage][k][threadIdx.y * 8 + i];
        #pragma unroll
        for (int j = 0; j < 8; j++) regB[j] = Bs[write_stage][k][threadIdx.x * 8 + j];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                accum[i][j] += regA[i] * regB[j];
            }
        }
    }

    // Write back to C with float4
    const int c_row_base = by * BM + threadIdx.y * 8;
    const int c_col_base = bx * BN + threadIdx.x * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int gr = c_row_base + i;
        if (gr < M) {
            #pragma unroll
            for (int j = 0; j < 8; j += 4) {
                int gc = c_col_base + j;
                if (gc + 3 < N) {
                    float4 vC = make_float4(accum[i][j], accum[i][j+1], accum[i][j+2], accum[i][j+3]);
                    *reinterpret_cast<float4*>(&C[(long)gr * N + gc]) = vC;
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
    dim3 b(16, 16), g((N + BN - 1) / BN, (M + BM - 1) / BM);
    k_sgemm_tiled_128x128_db<<<g, b>>>(dA, dB, dC, M, N, K);
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
