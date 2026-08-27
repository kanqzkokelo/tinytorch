// Speculative-decode VERIFY correctness test.
//
// Verifies qwen2_engine_verify_speculative() against the per-position
// logits produced by N sequential single-token forwards. Bit-exact
// equality is the goal: the verify path is implemented as N sequential
// advance() + compute_logits_into_d_logits() calls, so it must produce
// the same float values as the golden path that mirrors it.
//
//   Build: make build/test_spec_verify
//   Run:   ./build/test_spec_verify
//
// Loads qwen2.5-0.5b-q4_0 (or whatever TT_MODEL points at), prefills 8
// tokens, then runs the compare in two engine instances (A: golden; B:
// verify). Both engines start from the same prefill state, so per-token
// logits at position i must match exactly.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"

#define MAX_CTX 1024
#define PREFILL_N 8
#define N_VERIFY  4    /* [current, draft0, draft1, draft2] */

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_)); exit(1); } } while(0)

/* Deterministic candidate token ids chosen to be valid within any vocab:
 * small integers that map to the embedding table. We do NOT rely on them
 * being meaningful tokens — we only need the model to produce stable
 * logits at each step. */
static int g_candidates[N_VERIFY] = { 1, 2, 3, 4 };

/* Build a fresh engine + prefill it. Returns the engine. */
static Qwen2Engine *make_prefilled_engine(const TTConfig *cfg, GGUFModel *m,
                                          const int *prompt, int n_prompt) {
    Qwen2Engine *e = qwen2_engine_create(cfg, m);
    if (!e) { fprintf(stderr, "engine create failed\n"); exit(1); }
    if (qwen2_engine_prefill(e, prompt, n_prompt)) {
        fprintf(stderr, "prefill failed\n"); exit(1);
    }
    return e;
}

/* Build a deterministic prompt (8 small ints).  Valid token ids. */
static int build_prompt(int *out) {
    for (int i = 0; i < PREFILL_N; i++) out[i] = 100 + i;
    return PREFILL_N;
}

int main(void) {
    const char *model_path = getenv("TT_MODEL")
        ? getenv("TT_MODEL")
        : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    fprintf(stderr, "[verify-test] loading model: %s\n", model_path);
    GGUFModel *model = gguf_load(model_path);
    if (!model) { fprintf(stderr, "gguf_load failed\n"); return 1; }
    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    if (cfg.dim == 0) { fprintf(stderr, "config failed\n"); return 1; }
    /* vocab is filled by tt_config_from_gguf to 0; engine_create resolves
     * it from the token_embd tensor. Mirror that lookup here for buffer
     * sizing before we know the engine exists. */
    GGUFTensor *tembd = gguf_get_tensor(model, "token_embd.weight");
    if (!tembd) { fprintf(stderr, "token_embd.weight missing\n"); return 1; }
    cfg.vocab = (int)tembd->shape[tembd->ndim - 1];
    fprintf(stderr, "[verify-test] dim=%d vocab=%d layers=%d heads=%d\n",
            cfg.dim, cfg.vocab, cfg.n_layers, cfg.n_heads);

    int prompt[PREFILL_N];
    const int n_prompt = build_prompt(prompt);

    /* === Engine A: golden path. N sequential step_logits calls. === */
    Qwen2Engine *eA = make_prefilled_engine(&cfg, model, prompt, n_prompt);
    const int pos_A_start = qwen2_engine_pos(eA);
    fprintf(stderr, "[verify-test] A: pos_start=%d vocab=%d\n",
            pos_A_start, cfg.vocab);
    const long vocab_f = (long)cfg.vocab;
    float *golden_h = (float *)calloc(N_VERIFY * vocab_f, sizeof(float));
    if (!golden_h) { fprintf(stderr, "OOM golden_h\n"); return 1; }
    for (int i = 0; i < N_VERIFY; i++) {
        const int rc = qwen2_engine_step_logits(eA, g_candidates[i],
                                                golden_h + (long)i * vocab_f);
        if (rc) { fprintf(stderr, "A step_logits rc=%d at i=%d\n", rc, i); return 1; }
    }
    const int pos_A_end = qwen2_engine_pos(eA);
    fprintf(stderr, "[verify-test] A: pos_end=%d (delta=%d)\n",
            pos_A_end, pos_A_end - pos_A_start);

    /* === Engine B: verify path. One call. === */
    Qwen2Engine *eB = make_prefilled_engine(&cfg, model, prompt, n_prompt);
    const int pos_B_start = qwen2_engine_pos(eB);
    fprintf(stderr, "[verify-test] B: pos_start=%d\n", pos_B_start);
    float *d_out = NULL;
    CUDA_OK(cudaMalloc(&d_out, N_VERIFY * vocab_f * sizeof(float)));
    const int vrc = qwen2_engine_verify_speculative(eB, g_candidates, N_VERIFY, d_out);
    if (vrc) { fprintf(stderr, "verify rc=%d\n", vrc); return 1; }
    const int pos_B_end = qwen2_engine_pos(eB);
    fprintf(stderr, "[verify-test] B: pos_end=%d (delta=%d)\n",
            pos_B_end, pos_B_end - pos_B_start);
    float *verify_h = (float *)calloc(N_VERIFY * vocab_f, sizeof(float));
    if (!verify_h) { fprintf(stderr, "OOM verify_h\n"); return 1; }
    CUDA_OK(cudaMemcpy(verify_h, d_out, N_VERIFY * vocab_f * sizeof(float),
                       cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    /* === Compare === */
    int any_fail = 0;
    double worst_abs = 0, worst_rel = 0;
    long worst_idx = -1;
    int worst_pos = -1;
    /* For per-position logit scale, sample 4 positions worth of stats. */
    for (int i = 0; i < N_VERIFY; i++) {
        double max_abs = 0, max_rel = 0;
        long n_nonzero = 0, n_match_bit = 0, n_total = vocab_f;
        double max_g = 0, max_v = 0;
        for (long v = 0; v < vocab_f; v++) {
            const float g = golden_h[i * vocab_f + v];
            const float x = verify_h[i * vocab_f + v];
            const double d = fabs((double)g - (double)x);
            if (d > max_abs) max_abs = d;
            if (fabs(g) > max_g) max_g = fabs(g);
            if (fabs(x) > max_v) max_v = fabs(x);
            if (g != 0.0f) {
                n_nonzero++;
                if (d / (fabs(g) + 1e-30) > max_rel) max_rel = d / (fabs(g) + 1e-30);
            }
            /* bit-exact: float bits identical */
            union { float f; uint32_t u; } ga, xa;
            ga.f = g; xa.f = x;
            if (ga.u == xa.u) n_match_bit++;
        }
        if (max_abs > worst_abs) {
            worst_abs = max_abs;
            worst_idx = -1; /* (would need argmax; skip detail) */
            worst_pos = i;
        }
        if (max_rel > worst_rel) worst_rel = max_rel;
        const int bit_ok = (n_match_bit == n_total);
        fprintf(stderr,
                "[verify-test] pos %d: max_abs=%.3e max_rel=%.3e "
                "bit_exact=%ld/%ld %s  max|g|=%.3f max|v|=%.3f\n",
                i, max_abs, max_rel, n_match_bit, n_total,
                bit_ok ? "PASS" : "FAIL", max_g, max_v);
        if (!bit_ok) any_fail = 1;
    }

    /* === Engine C: pos invariant. verify() advances pos by N. === */
    if (pos_B_end - pos_B_start != N_VERIFY) {
        fprintf(stderr, "[verify-test] FAIL: B pos delta %d != N_VERIFY %d\n",
                pos_B_end - pos_B_start, N_VERIFY);
        any_fail = 1;
    }
    if (pos_A_end - pos_A_start != N_VERIFY) {
        fprintf(stderr, "[verify-test] FAIL: A pos delta %d != N_VERIFY %d\n",
                pos_A_end - pos_A_start, N_VERIFY);
        any_fail = 1;
    }

    free(golden_h);
    free(verify_h);
    qwen2_engine_free(eA);
    qwen2_engine_free(eB);
    gguf_free(model);

    if (any_fail) {
        fprintf(stderr, "[verify-test] overall: FAIL (worst_abs=%.3e worst_rel=%.3e)\n",
                worst_abs, worst_rel);
        return 1;
    }
    fprintf(stderr, "[verify-test] overall: PASS (bit-exact across all %d positions)\n",
            N_VERIFY);
    return 0;
}
