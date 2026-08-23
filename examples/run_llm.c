#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "loader_gguf.h"
#include "ops_llm.h"

int main(int argc, char **argv) {
    const char *model_path = "/home/mitesh/nnfromscratch/data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    if (argc > 1) model_path = argv[1];

    printf("Loading GGUF model: %s...\n", model_path);
    GGUFModel *model = gguf_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load GGUF model.\n");
        return 1;
    }

    printf("\n=== Model Architecture Summary ===\n");
    printf("  Embedding Dim (dim):   %d\n", model->dim);
    printf("  Hidden Dim (ffn):      %d\n", model->hidden_dim);
    printf("  Number of Layers:      %d\n", model->n_layers);
    printf("  Number of Heads:       %d\n", model->n_heads);
    printf("  Number of KV Heads:    %d\n", model->n_kv_heads);
    printf("  Context Window:        %d\n", model->max_seq_len);

    // Measure forward pass latency of 1 Transformer step
    float *x = (float *)calloc(model->dim, sizeof(float));
    float *out = (float *)calloc(model->dim, sizeof(float));
    float *weight = (float *)malloc(model->dim * sizeof(float));
    for (int i = 0; i < model->dim; i++) {
        x[i] = 1.0f;
        weight[i] = 1.0f;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int STEPS = 100;
    for (int s = 0; s < STEPS; s++) {
        // RMSNorm
        tt_rmsnorm(out, x, weight, model->dim, model->rms_norm_eps);
        // RoPE
        tt_rope(out, s, model->dim / model->n_heads, model->n_heads, 10000.0f);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double tok_per_sec = (double)STEPS / elapsed_sec;

    printf("\n=== tinytorch Transformer Step Performance ===\n");
    printf("  Executed %d steps in %.4f seconds\n", STEPS, elapsed_sec);
    printf("  Step Latency: %.3f ms/token\n", (elapsed_sec / STEPS) * 1000.0);
    printf("  CPU Execution Speed: %.1f tokens/sec\n", tok_per_sec);

    free(x);
    free(out);
    free(weight);
    gguf_free(model);

    return 0;
}
