// F16/F32 GEMV micro-test: real tensor from a llama-family f16 GGUF.
// Usage: build/test_gemv_f16 MODEL [tensor_name]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#include "loader_gguf.h"
#include "dequant_ref.h"

extern "C" int tt_gemv_typed(const void *W, int dtype, const float *x,
                             float *y, int M, int K, cudaStream_t stream);

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "data/testmodels/tinyllama-f16.gguf";
    const char *tname = argc > 2 ? argv[2] : "blk.0.attn_k.weight";
    GGUFModel *m = gguf_load(model);
    if (!m) { fprintf(stderr, "load fail\n"); return 1; }
    GGUFTensor *t = gguf_get_tensor(m, tname);
    if (!t) { fprintf(stderr, "tensor %s missing\n", tname); return 1; }
    long numel = 1;
    for (int d = 0; d < t->ndim; d++) numel *= (long)t->shape[d];
    int K = (int)t->shape[0];      /* ne0 = in features */
    int M = (int)t->shape[1];      /* ne1 = out features */
    printf("tensor %s type=%d shape=[%d,%d] numel=%ld\n", tname, t->type, K, M, numel);

    /* CPU reference */
    std::vector<float> Wref(numel);
    { FILE *f=fopen("/tmp/wref.bin","rb"); if(!f){fprintf(stderr,"need /tmp/wref.bin\n"); return 1;} fread(Wref.data(),4,numel,f); fclose(f); }
    /* layout: ne0 fastest -> row m at m*K */
    /* dequant_tensor layout: ggml order ne0 fastest => row m starts at m*K */
    std::vector<float> x(K), yref(M, 0.f);
    for (int i = 0; i < K; i++) x[i] = sinf(0.7f * i + 0.3f);
    for (int mi = 0; mi < M; mi++) {
        double acc = 0;
        for (int ki = 0; ki < K; ki++) acc += (double)Wref[(size_t)mi * K + ki] * x[ki];
        yref[mi] = (float)acc;
    }

    void *dW; cudaMalloc(&dW, t->size_bytes);
    cudaMemcpy(dW, t->data, t->size_bytes, cudaMemcpyHostToDevice);
    float *dx, *dy; cudaMalloc(&dx, K * 4); cudaMalloc(&dy, M * 4);
    cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice);
    int rc = tt_gemv_typed(dW, (int)t->type, dx, dy, M, K, 0);
    if (rc) { fprintf(stderr, "gemv rc=%d\n", rc); return 1; }
    std::vector<float> ygpu(M);
    cudaMemcpy(ygpu.data(), dy, M * 4, cudaMemcpyDeviceToHost);

    double maxerr = 0; int argbad = -1;
    for (int i = 0; i < M; i++) {
        double e = fabs((double)ygpu[i] - yref[i]);
        if (e > maxerr) { maxerr = e; argbad = i; }
    }
    printf("max abs err=%.6f at row %d (yref=%.4f ygpu=%.4f) rel=%.2e\n",
           maxerr, argbad, yref[argbad], ygpu[argbad], maxerr / fabs(yref[argbad]));
    return maxerr > 1e-2 ? 1 : 0;
}
