// M9.5: micro-bench the F16 / BF16 GEMV kernels. Times tt_gemv_typed on
// (M, K) shapes that match smollm2-135m-f16 per-layer, and reports
// kernel time (CUDA event) plus a sanity-vs-CPU dot product.
//
// Usage: build/bench_f16_kernels MODEL
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "loader_gguf.h"
#include "dequant_ref.h"

extern "C" int tt_gemv_typed(const void *W, int dtype, const float *x,
                             float *y, int M, int K, cudaStream_t stream);

static float frand(unsigned *s) {
    *s = *s * 1664525u + 1013904223u;
    return ((float)(*s >> 8) / 8388608.0f) - 1.0f;
}

static void bench_one(GGUFTensor *t, int code) {
    int K = (int)t->shape[0];
    int M = (int)t->shape[1];
    printf("  tensor=%-40s M=%-5d K=%-5d\n", t->name, M, K);

    std::vector<float> x(K), yref(M, 0.f), ygpu(M, 0.f);
    unsigned s = 0xC0FFEEu;
    for (int i = 0; i < K; i++) x[i] = frand(&s);

    std::vector<float> wref((size_t)M * K);
    ttq_dequant(t->data, code, (long)M * K, wref.data());
    for (int mi = 0; mi < M; mi++) {
        double acc = 0.0;
        for (int ki = 0; ki < K; ki++) acc += (double)wref[(size_t)mi * K + ki] * x[ki];
        yref[mi] = (float)acc;
    }

    void *dW; cudaMalloc(&dW, t->size_bytes);
    cudaMemcpy(dW, t->data, t->size_bytes, cudaMemcpyHostToDevice);
    float *dx, *dy; cudaMalloc(&dx, K * 4); cudaMalloc(&dy, M * 4);
    cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice);

    int rc = tt_gemv_typed(dW, code, dx, dy, M, K, 0);
    if (rc) { printf("    FAIL kernel rc=%d\n", rc); cudaFree(dW); cudaFree(dx); cudaFree(dy); return; }
    cudaMemcpy(ygpu.data(), dy, M * 4, cudaMemcpyDeviceToHost);
    float maxerr = 0.f;
    for (int i = 0; i < M; i++) {
        float e = fabsf(ygpu[i] - yref[i]);
        if (e > maxerr) maxerr = e;
    }
    float rowmax = 0.f;
    for (int i = 0; i < M; i++) if (fabsf(yref[i]) > rowmax) rowmax = fabsf(yref[i]);
    float atol = 1e-2f * (rowmax > 1e-6f ? rowmax : 1e-6f);
    printf("    sanity: maxerr=%.3e atol=%.3e %s\n", maxerr, atol, maxerr <= atol ? "OK" : "FAIL");

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ts;
    for (int it = 0; it < 50; it++) {
        cudaEventRecord(e0, 0);
        tt_gemv_typed(dW, code, dx, dy, M, K, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        ts.push_back(ms * 1000.f);
    }
    std::sort(ts.begin(), ts.end());
    float med = ts[ts.size() / 2];
    float gflops = 2.f * M * K / (med * 1e-6f) / 1e9f;
    printf("    bench: median=%.2f us  gflops=%.1f  bw=%.1f GB/s\n",
           med, gflops, (float)t->size_bytes / (med * 1e-6f) / 1e9f);

    cudaFree(dW); cudaFree(dx); cudaFree(dy);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    GGUFModel *m = gguf_load(argv[1]);
    if (!m) { fprintf(stderr, "load fail\n"); return 1; }
    int target_dtypes[] = {1 /*F16*/, 30 /*BF16*/, 0 /*F32*/};
    const char *names[] = {"F16", "BF16", "F32"};

    for (int di = 0; di < 3; di++) {
        int code = target_dtypes[di];
        int n = 0;
        for (int i = 0; i < m->tensor_count; i++) {
            GGUFTensor *t = &m->tensors[i];
            if ((int)t->type == code && t->ndim == 2 && t->shape[0] % 4 == 0) {
                if (n == 0) printf("%-4s\n", names[di]);
                bench_one(t, code);
                if (++n >= 5) break;  /* cap at 5 tensors per dtype */
            }
        }
        if (n == 0) printf("%-4s no tensor found (skip)\n", names[di]);
    }
    gguf_free(m);
    return 0;
}
