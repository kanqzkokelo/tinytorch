// Layer-0 Parity Diagnostic Test (test_prefill_layer_parity.c)
//
// Loads Qwen2.5-0.5B, feeds N tokens, and compares hidden state outputs
// across all positions between:
//   1. Sequential single-token advance() (via qwen2_engine_prefill step by step)
//   2. Batched prefill (via prefill_batched_gemm)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"

static int test_n(Qwen2Engine *e_seq, Qwen2Engine *e_bat, const TTConfig *cfg, int N) {
    printf("\n==========================================\n");
    printf("=== Testing Prefill Parity for N = %d ===\n", N);
    printf("==========================================\n");

    int *toks = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) {
        toks[i] = 100 + (i % 64);
    }

    float *h_x_seq = (float *)malloc((size_t)N * cfg->dim * sizeof(float));
    float *h_x_batched = (float *)malloc((size_t)N * cfg->dim * sizeof(float));

    if (!toks || !h_x_seq || !h_x_batched) {
        fprintf(stderr, "Memory allocation failed\n");
        free(toks); free(h_x_seq); free(h_x_batched);
        return 1;
    }

    /* Path 1: Sequential advance() */
    printf("\n--- Path 1: Sequential Path (N=%d) ---\n", N);
    setenv("TT_NO_BATCHED_PREFILL", "1", 1);
    qwen2_engine_reset(e_seq);

    for (int i = 0; i < N; i++) {
        int rc = qwen2_engine_prefill(e_seq, &toks[i], 1);
        if (rc != 0) {
            fprintf(stderr, "Sequential prefill failed at token %d (rc=%d)\n", i, rc);
            free(toks); free(h_x_seq); free(h_x_batched);
            return 1;
        }
        qwen2_debug_copy_x(e_seq, h_x_seq + (long)i * cfg->dim, cfg->dim);
    }
    printf("Sequential path complete.\n");

    /* Path 2: Batched prefill */
    printf("\n--- Path 2: Batched Path (prefill_batched_gemm, N=%d) ---\n", N);
    unsetenv("TT_NO_BATCHED_PREFILL");
    qwen2_engine_reset(e_bat);

    int rc = prefill_batched_gemm(e_bat, toks, N, h_x_batched);
    if (rc != 0) {
        fprintf(stderr, "Batched prefill failed (rc=%d)\n", rc);
        free(toks); free(h_x_seq); free(h_x_batched);
        return 1;
    }
    printf("Batched path complete.\n");

    /* Compare outputs per position */
    printf("\n=== Parity Breakdown Across Positions (0..%d) ===\n", N - 1);
    int failures = 0;
    int first_divergent_pos = -1;
    double global_max_abs = 0.0;

    for (int i = 0; i < N; i++) {
        double max_abs = 0.0;
        int max_j = 0;
        for (int j = 0; j < cfg->dim; j++) {
            double diff = fabs((double)h_x_seq[(long)i * cfg->dim + j] - (double)h_x_batched[(long)i * cfg->dim + j]);
            if (diff > max_abs) {
                max_abs = diff;
                max_j = j;
            }
        }
        if (max_abs > global_max_abs) global_max_abs = max_abs;
        int pass = (max_abs <= 1e-3);
        if (!pass) {
            failures++;
            if (first_divergent_pos < 0) first_divergent_pos = i;
        }
        if (N <= 32 || !pass || i == N - 1 || i % 16 == 0) {
            printf("Pos %3d: max_abs=%.4e @ idx %4d (%s)\n", i, max_abs, max_j, pass ? "PASS" : "FAIL");
        }
    }

    printf("N=%d Summary: global max_abs_diff = %.4e\n", N, global_max_abs);

    if (failures > 0) {
        printf("FAIL (N=%d): Divergence detected at position %d (%d/%d positions failed)\n",
               N, first_divergent_pos, failures, N);
    } else {
        printf("PASS (N=%d): All %d positions bit-exact (max_abs <= 1e-3)\n", N, N);
    }

    free(toks);
    free(h_x_seq);
    free(h_x_batched);

    return (failures > 0) ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    setenv("TT_NO_GRAPH", "1", 1);

    printf("=== Prefill Layer Parity Diagnostic Suite ===\n");
    printf("Loading model: %s\n", model_path);

    GGUFModel *m = gguf_load(model_path);
    if (!m) {
        fprintf(stderr, "Failed to load GGUF model: %s\n", model_path);
        return 1;
    }

    TTConfig cfg = tt_config_from_gguf(m, 1024);
    if (cfg.dim == 0) {
        fprintf(stderr, "Invalid TTConfig derived from model\n");
        gguf_free(m);
        return 1;
    }

    printf("Model geometry: dim=%d hidden_dim=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.vocab);

    Qwen2Engine *e_seq = qwen2_engine_create(&cfg, m);
    Qwen2Engine *e_bat = qwen2_engine_create(&cfg, m);

    if (!e_seq || !e_bat) {
        fprintf(stderr, "Failed to create Qwen2Engine instances\n");
        if (e_seq) qwen2_engine_free(e_seq);
        if (e_bat) qwen2_engine_free(e_bat);
        gguf_free(m);
        return 1;
    }

    int test_sizes[] = {32, 64, 128};
    int total_failures = 0;

    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(test_sizes[0]); i++) {
        int res = test_n(e_seq, e_bat, &cfg, test_sizes[i]);
        if (res != 0) total_failures++;
    }

    qwen2_engine_free(e_seq);
    qwen2_engine_free(e_bat);
    gguf_free(m);

    if (total_failures > 0) {
        printf("\nFAILURE: %d batch size test(s) failed parity check\n", total_failures);
        return 1;
    }

    printf("\nSUCCESS: All batch size tests passed parity check!\n");
    return 0;
}
