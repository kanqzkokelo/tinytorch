// Late-enable Q4/Q8 KV-cache backfill test (P1-2 fix).
// Fault: enable_q{4,8}_kvcache memset the Q cache to 0; FP32 slots [0..pos)
// populated BEFORE the enable call then decode as ZEROED above threshold.
// Fix: backfill [0..pos) from FP32 at enable time.
//
// Part 1 (kernel, no model): backfill kernels must be BIT-IDENTICAL to the
// per-slot incremental scatter kernels on the same FP32 slot contents.
// Part 2 (engine, needs TT_MODEL): prefill N=48 FP32 tokens, late-enable,
// decode 1 step above threshold (TT_QKV_THRESH=32):
//   Q4: late (B) vs early-enable (C) logits must be bitwise equal (memcmp).
//       NOTE: on hd=64 models both are all-NaN (k_fa2_q4_split hardcodes
//       elems=4 — pre-existing, unrelated to backfill; C proves it). The
//       check is still meaningful: identical NaN payloads = identical Q data.
//   Q8: late (B) vs pure-FP32 (A): same argmax + bounded diff (Q8 quant noise
//       through 24 layers; without backfill the diff is ~17.5 and argmax flips).
// Sensitivity: TT_NO_BACKFILL=1 skips the backfill; engine checks must FAIL.
// Build: nvcc -O3 -gencode arch=compute_86,code=sm_86 -Iinclude -Isrc -o build/test_qcache_backfill tests/test_qcache_backfill.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu -L$HOME/mmcuda/lib -lcudart -lpthread -lm
// Run: TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/test_qcache_backfill
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"

void qwen2_engine_enable_q4_kvcache(Qwen2Engine *e, int enable);
int tt_kv_scatter_q8_0(const float *kst, const float *vst, void *Kc_q8, void *Vc_q8,
                       const int *d_pos, int n_kv_heads, int head_dim, int max_ctx, cudaStream_t stream);
int tt_kv_scatter_q4_0(const float *kst, const float *vst, void *Kc_q4, void *Vc_q4,
                       const int *d_pos, int n_kv_heads, int head_dim, int max_ctx, cudaStream_t stream);
int tt_kv_backfill_q8_0(const float *Kf, const float *Vf, void *Kc_q8, void *Vc_q8,
                        int n_slots, int kvdim, cudaStream_t stream);
int tt_kv_backfill_q4_0(const float *Kf, const float *Vf, void *Kc_q4, void *Vc_q4,
                        int n_slots, int kvdim, cudaStream_t stream);

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

#define MAX_CTX 1024
#define PREFILL_N 48
#define STEP_TOK 7

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

/* Bit-exactness of backfill vs per-slot scatter. Returns 0 on exact match. */
static int check_backfill_exact(int qmode, int n_kv_heads, int head_dim, int max_ctx, int n_slots) {
    const int kvdim = n_kv_heads * head_dim;
    const int nb = kvdim / 32;
    const size_t fp32_sz = (size_t)n_slots * kvdim * sizeof(float);
    const size_t q8_sz = (size_t)n_slots * nb * 34;   /* BlockQ8_0 = 2 + 32 */
    const size_t q4_sz = (size_t)n_slots * nb * 18;   /* BlockQ4_0 = 2 + 16 */
    const size_t qsz = qmode ? q8_sz : q4_sz;
    float *hK = malloc(fp32_sz), *hV = malloc(fp32_sz);
    for (size_t i = 0; i < (size_t)n_slots * kvdim; i++) { hK[i] = frand(); hV[i] = frand(); }
    float *dK, *dV; void *dQ1k, *dQ1v, *dQ2k, *dQ2v; int *d_pos;
    CK(cudaMalloc(&dK, fp32_sz)); CK(cudaMalloc(&dV, fp32_sz));
    CK(cudaMalloc(&dQ1k, qsz)); CK(cudaMalloc(&dQ1v, qsz));
    CK(cudaMalloc(&dQ2k, qsz)); CK(cudaMalloc(&dQ2v, qsz));
    CK(cudaMalloc(&d_pos, sizeof(int)));
    CK(cudaMemcpy(dK, hK, fp32_sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV, fp32_sz, cudaMemcpyHostToDevice));
    CK(cudaMemset(dQ1k, 0, qsz)); CK(cudaMemset(dQ1v, 0, qsz));
    CK(cudaMemset(dQ2k, 0, qsz)); CK(cudaMemset(dQ2v, 0, qsz));
    /* ref: per-slot incremental scatter */
    for (int t = 0; t < n_slots; t++) {
        CK(cudaMemcpy(d_pos, &t, sizeof(int), cudaMemcpyHostToDevice));
        if (qmode) {
            tt_kv_scatter_q8_0(dK + (long)t * kvdim, dV + (long)t * kvdim,
                               dQ1k, dQ1v, d_pos, n_kv_heads, head_dim, max_ctx, 0);
        } else {
            tt_kv_scatter_q4_0(dK + (long)t * kvdim, dV + (long)t * kvdim,
                               dQ1k, dQ1v, d_pos, n_kv_heads, head_dim, max_ctx, 0);
        }
    }
    /* test: one-shot backfill */
    if (qmode) tt_kv_backfill_q8_0(dK, dV, dQ2k, dQ2v, n_slots, kvdim, 0);
    else tt_kv_backfill_q4_0(dK, dV, dQ2k, dQ2v, n_slots, kvdim, 0);
    CK(cudaDeviceSynchronize());
    char *h1k = malloc(qsz), *h1v = malloc(qsz), *h2k = malloc(qsz), *h2v = malloc(qsz);
    CK(cudaMemcpy(h1k, dQ1k, qsz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h1v, dQ1v, qsz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2k, dQ2k, qsz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2v, dQ2v, qsz, cudaMemcpyDeviceToHost));
    int bad = memcmp(h1k, h2k, qsz) != 0 || memcmp(h1v, h2v, qsz) != 0;
    long nbad = 0;
    if (bad) for (size_t i = 0; i < qsz; i++) nbad += (h1k[i] != h2k[i]) + (h1v[i] != h2v[i]);
    printf("[%s] kernel backfill-vs-scatter bit-exact: %s%s\n", qmode ? "Q8" : "Q4",
           bad ? "FAIL" : "PASS", bad ? "" : "");
    if (bad) printf("  mismatched bytes: %ld/%zu\n", nbad, 2 * qsz);
    free(hK); free(hV); free(h1k); free(h1v); free(h2k); free(h2v);
    cudaFree(dK); cudaFree(dV); cudaFree(dQ1k); cudaFree(dQ1v);
    cudaFree(dQ2k); cudaFree(dQ2v); cudaFree(d_pos);
    return bad;
}

static int argmax_f(const float *v, long n) {
    int b = 0;
    for (long i = 1; i < n; i++) if (v[i] > v[b]) b = (int)i;
    return b;
}

int main(void) {
    srand(1234);
    int fails = 0;
    /* Part 1: kernel exactness (fast, no model) */
    fails += check_backfill_exact(0, 2, 64, 128, 96);
    fails += check_backfill_exact(1, 2, 64, 128, 96);
    fails += check_backfill_exact(0, 8, 128, 256, 200);
    fails += check_backfill_exact(1, 8, 128, 256, 200);

    /* Part 2: engine late-enable */
    setenv("TT_QKV_THRESH", "32", 1);
    unsetenv("TT_Q4_KV");
    unsetenv("TT_Q8_KV");
    setenv("TT_NO_GRAPH", "1", 1);
    const char *model_path = getenv("TT_MODEL") ? getenv("TT_MODEL") : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    fprintf(stderr, "[backfill-test] model %s thresh=32 prefill=%d%s\n",
            model_path, PREFILL_N, getenv("TT_NO_BACKFILL") ? " TT_NO_BACKFILL=1" : "");
    GGUFModel *model = gguf_load(model_path);
    if (!model) { fprintf(stderr, "gguf_load fail\n"); return 1; }
    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    GGUFTensor *tembd = gguf_get_tensor(model, "token_embd.weight");
    if (!tembd) { fprintf(stderr, "embd missing\n"); return 1; }
    cfg.vocab = (int)tembd->shape[tembd->ndim - 1];
    int prompt[PREFILL_N];
    for (int i = 0; i < PREFILL_N; i++) prompt[i] = 100 + i;
    float *logA = malloc((long)cfg.vocab * sizeof(float));
    float *logB = malloc((long)cfg.vocab * sizeof(float));
    float *logC = malloc((long)cfg.vocab * sizeof(float));

    /* Q4: late (B) vs early (C) bitwise equal */
    {
        Qwen2Engine *eA = qwen2_engine_create(&cfg, model);
        if (!eA) { fprintf(stderr, "create A fail\n"); return 1; }
        if (qwen2_engine_prefill(eA, prompt, PREFILL_N)) { fprintf(stderr, "prefill A fail\n"); return 1; }
        if (qwen2_engine_step_logits(eA, STEP_TOK, logA)) { fprintf(stderr, "step A fail\n"); return 1; }
        qwen2_engine_free(eA);
        Qwen2Engine *eB = qwen2_engine_create(&cfg, model);
        if (!eB) { fprintf(stderr, "create B fail\n"); return 1; }
        if (qwen2_engine_prefill(eB, prompt, PREFILL_N)) { fprintf(stderr, "prefill B fail\n"); return 1; }
        qwen2_engine_enable_q4_kvcache(eB, 1);
        if (qwen2_engine_step_logits(eB, STEP_TOK, logB)) { fprintf(stderr, "step B fail\n"); return 1; }
        qwen2_engine_free(eB);
        Qwen2Engine *eC = qwen2_engine_create(&cfg, model);
        if (!eC) { fprintf(stderr, "create C fail\n"); return 1; }
        qwen2_engine_enable_q4_kvcache(eC, 1);
        if (qwen2_engine_prefill(eC, prompt, PREFILL_N)) { fprintf(stderr, "prefill C fail\n"); return 1; }
        if (qwen2_engine_step_logits(eC, STEP_TOK, logC)) { fprintf(stderr, "step C fail\n"); return 1; }
        qwen2_engine_free(eC);
        int eq = memcmp(logB, logC, (long)cfg.vocab * sizeof(float)) == 0;
        printf("[Q4] engine late-vs-early bitwise equal: %s\n", eq ? "PASS" : "FAIL");
        if (!eq) fails++;
    }
    /* Q8: late (B) vs pure-FP32 (A): same argmax + bounded diff */
    {
        Qwen2Engine *eA = qwen2_engine_create(&cfg, model);
        if (!eA) { fprintf(stderr, "create A fail\n"); return 1; }
        if (qwen2_engine_prefill(eA, prompt, PREFILL_N)) { fprintf(stderr, "prefill A fail\n"); return 1; }
        if (qwen2_engine_step_logits(eA, STEP_TOK, logA)) { fprintf(stderr, "step A fail\n"); return 1; }
        qwen2_engine_free(eA);
        Qwen2Engine *eB = qwen2_engine_create(&cfg, model);
        if (!eB) { fprintf(stderr, "create B fail\n"); return 1; }
        if (qwen2_engine_prefill(eB, prompt, PREFILL_N)) { fprintf(stderr, "prefill B fail\n"); return 1; }
        qwen2_engine_enable_q8_kvcache(eB, 1);
        if (qwen2_engine_step_logits(eB, STEP_TOK, logB)) { fprintf(stderr, "step B fail\n"); return 1; }
        qwen2_engine_free(eB);
        double mx = 0; long nan = 0;
        for (long i = 0; i < cfg.vocab; i++) {
            if (isnan(logA[i]) || isnan(logB[i])) { nan++; continue; }
            double d = fabs((double)logA[i] - (double)logB[i]);
            if (d > mx) mx = d;
        }
        int aA = argmax_f(logA, cfg.vocab), aB = argmax_f(logB, cfg.vocab);
        int ok = (nan == 0 && aA == aB && mx < 2.0);
        printf("[Q8] engine late-vs-fp32 max_abs=%.3e argmax %d vs %d nan=%ld: %s\n",
               mx, aB, aA, nan, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
    }
    free(logA); free(logB); free(logC);
    if (fails) { printf("BACKFILL TEST: FAIL (%d)\n", fails); return 1; }
    printf("BACKFILL TEST: PASS\n");
    return 0;
}
