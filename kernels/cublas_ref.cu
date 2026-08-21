// M3: minimal cuBLAS sgemm reference wrapper (row-major semantics).
#include <cublas_v2.h>
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
