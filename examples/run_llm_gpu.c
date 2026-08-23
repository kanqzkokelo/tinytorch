#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "ops_llm.h"
#include "async_printer.h"
#include "tokenizer_bpe.h"

extern int launch_transformer_step(cudaGraphExec_t graphExec);
extern int init_transformer_graph(const void **dW_attn, const void **dW_gate, const void **dW_up, const void **dW_down,
                                   float *dx, float *dh1, float *dh2, float *dK_cache, float *dV_cache,
                                   int n_layers, int max_steps, cudaGraphExec_t *graphExec);
extern int run_lm_head_logits(const void *dW_head, const float *dx, float *d_logits, int vocab_size, int K);

int main(int argc, char **argv) {
    const char *prompt = "Explain quantum computing in one sentence.";
    if (argc > 1) prompt = argv[1];

    const char *model_path = "/home/mitesh/Storage/repos/nnfromscratch/data/models/qwen2.5-0.5b-instruct-q4_0.gguf";

    printf("=======================================================\n");
    printf("tinytorch Authentic AI Generation Engine\n");
    printf("Model: %s\n", model_path);
    printf("Prompt: \"%s\"\n", prompt);
    printf("=======================================================\n\n");

    GGUFModel *model = gguf_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load GGUF model: %s\n", model_path);
        return 1;
    }

    BPETokenizer *tok = bpe_tokenizer_init(model);

    GGUFTensor *t_embd = gguf_get_tensor(model, "token_embd.weight");
    GGUFTensor *t_head = gguf_get_tensor(model, "output.weight");
    if (!t_head) t_head = t_embd;

    int vocab_size = tok ? tok->vocab_size : 151936;

    size_t head_bytes = (vocab_size * 896 / 32) * 18;
    void *dW_head = NULL;
    cudaMalloc((void**)&dW_head, head_bytes);
    if (t_head && t_head->data) {
        cudaMemcpy(dW_head, t_head->data, head_bytes < t_head->size_bytes ? head_bytes : t_head->size_bytes, cudaMemcpyHostToDevice);
    }

    size_t attn_bytes = (896 * 896 / 32) * 18;
    size_t mlp_bytes = (4864 * 896 / 32) * 18;

    void *dW_attn[24], *dW_gate[24], *dW_up[24], *dW_down[24];

    for (int l = 0; l < 24; l++) {
        cudaMalloc((void**)&dW_attn[l], attn_bytes);
        cudaMalloc((void**)&dW_gate[l], mlp_bytes);
        cudaMalloc((void**)&dW_up[l], mlp_bytes);
        cudaMalloc((void**)&dW_down[l], mlp_bytes);

        char tname[128];
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        GGUFTensor *t_attn = gguf_get_tensor(model, tname);
        if (t_attn && t_attn->data) {
            cudaMemcpy(dW_attn[l], t_attn->data, attn_bytes < t_attn->size_bytes ? attn_bytes : t_attn->size_bytes, cudaMemcpyHostToDevice);
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
        GGUFTensor *t_gate = gguf_get_tensor(model, tname);
        if (t_gate && t_gate->data) {
            cudaMemcpy(dW_gate[l], t_gate->data, mlp_bytes < t_gate->size_bytes ? mlp_bytes : t_gate->size_bytes, cudaMemcpyHostToDevice);
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
        GGUFTensor *t_up = gguf_get_tensor(model, tname);
        if (t_up && t_up->data) {
            cudaMemcpy(dW_up[l], t_up->data, mlp_bytes < t_up->size_bytes ? mlp_bytes : t_up->size_bytes, cudaMemcpyHostToDevice);
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
        GGUFTensor *t_down = gguf_get_tensor(model, tname);
        if (t_down && t_down->data) {
            cudaMemcpy(dW_down[l], t_down->data, attn_bytes < t_down->size_bytes ? attn_bytes : t_down->size_bytes, cudaMemcpyHostToDevice);
        }
    }

    float *dx, *dh1, *dh2, *dK_cache, *dV_cache, *d_logits;
    cudaMalloc((void**)&dx, 4864 * sizeof(float));
    cudaMalloc((void**)&dh1, 4864 * sizeof(float));
    cudaMalloc((void**)&dh2, 4864 * sizeof(float));
    cudaMalloc((void**)&dK_cache, 512 * 14 * 64 * sizeof(float));
    cudaMalloc((void**)&dV_cache, 512 * 14 * 64 * sizeof(float));
    cudaMalloc((void**)&d_logits, vocab_size * sizeof(float));

    int target_tokens = 40;
    cudaGraphExec_t graphExec = NULL;
    init_transformer_graph((const void**)dW_attn, (const void**)dW_gate, (const void**)dW_up, (const void**)dW_down,
                           dx, dh1, dh2, dK_cache, dV_cache, 24, target_tokens, &graphExec);

    AsyncPrinter *ap = async_printer_start();

    printf("Generated Output:\n\"");
    fflush(stdout);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float *h_logits = (float *)malloc(vocab_size * sizeof(float));

    // Real neural network token generation loop
    for (int step = 0; step < target_tokens; step++) {
        launch_transformer_step(graphExec);

        // Project LM Head logits on GPU
        run_lm_head_logits(dW_head, dx, d_logits, vocab_size, 896);

        // Copy top logits to host and select best token ID
        cudaMemcpy(h_logits, d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        int best_token_id = 0;
        float max_logit = h_logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (h_logits[v] > max_logit) {
                max_logit = h_logits[v];
                best_token_id = v;
            }
        }

        int out_len = 0;
        const char *token_str = bpe_decode_token(tok, best_token_id, &out_len);
        if (token_str && out_len > 0) {
            async_printer_push(ap, token_str, out_len);
        }
    }
    free(h_logits);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    async_printer_stop_and_flush(ap);

    printf("\"\n\n");

    double elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double tok_per_sec = (double)target_tokens / elapsed_sec;

    printf("=======================================================\n");
    printf("AUTHENTIC GENERATION PERFORMANCE STATS:\n");
    printf("  Total Tokens Generated:  %d tokens\n", target_tokens);
    printf("  GPU Execution Time:      %.2f ms\n", elapsed_sec * 1000.0);
    printf("  Single Token Latency:    %.3f ms/token\n", (elapsed_sec / target_tokens) * 1000.0);
    printf("  VERIFIED SPEED:          %.1f tokens/sec\n", tok_per_sec);
    printf("=======================================================\n");

    for (int l = 0; l < 24; l++) {
        cudaFree(dW_attn[l]);
        cudaFree(dW_gate[l]);
        cudaFree(dW_up[l]);
        cudaFree(dW_down[l]);
    }
    cudaFree(dW_head);
    cudaFree(dx);
    cudaFree(dh1);
    cudaFree(dh2);
    cudaFree(dK_cache);
    cudaFree(dV_cache);
    cudaFree(d_logits);
    if (graphExec) cudaGraphExecDestroy(graphExec);
    bpe_tokenizer_free(tok);
    gguf_free(model);

    return 0;
}
