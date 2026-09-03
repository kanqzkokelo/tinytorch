// M3: minimal cuBLAS sgemm reference wrapper (row-major semantics).
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

static cublasHandle_t g_handle = nullptr;

static void ensure_handle(void) {
    if (!g_handle)
        cublasCreate(&g_handle);
}

/* Computes row-major C[MxN] = A[MxK] @ B[KxN] on device memory.
 * Column-major equivalence used: C_cm(N x M) = B_cm(N x K) @ A_cm(K x M).
 * Returns 0 on success, nonzero cublas status otherwise. */
extern "C" int tt_cublas_ref(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    ensure_handle();
    if (!g_handle) return -1;
    const float alpha = 1.0f, beta = 0.0f;
    return (int)cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha, B, N,
                            A, K,
                            &beta, C, N);
}

/* Prefill NT GEMM: row-major Y[N,M] = X[N,K] @ W^T[M,K].
 * dW is MxK row-major (FP32 shadow), dX is NxK row-major, dY is NxM.
 * Column-major: Y^T[MxN] = W[MxK] @ X^T[KxN], where W_mem (MxK row)
 * reads as col KxM (=W^T), so: Y_col = W_mem^T @ X_mem.
 * Call: OP_T(W,lda=K) OP_N(X,ldb=K), m=M n=N k=K ldc=M.
 * allow_tf32=0 -> CUBLAS_COMPUTE_32F (strict, q/k/v/o);
 * allow_tf32!=0 -> CUBLAS_COMPUTE_32F_FAST_TF32 (gate/up/down only).
 * stream may be NULL (uses handle default). Returns 0 on success. */
extern "C" int tt_cublas_prefill_nt(const float *dW, const float *dX, float *dY,
                                    int M, int K, int N,
                                    void *stream, int allow_tf32) {
    ensure_handle();
    if (!g_handle) return -1;
    if (stream) cublasSetStream(g_handle, (cudaStream_t)stream);
    const float alpha = 1.0f, beta = 0.0f;
    cudaDataType dt = CUDA_R_32F;
    cublasComputeType_t ct = allow_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32
                                        : CUBLAS_COMPUTE_32F;
    cublasStatus_t st = cublasGemmEx(g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        dW, dt, K,
        dX, dt, K,
        &beta,
        dY, dt, M,
        ct, CUBLAS_GEMM_DEFAULT);
    return (int)st;
}
