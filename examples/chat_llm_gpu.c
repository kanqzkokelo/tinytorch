#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "tokenizer_bpe.h"
#include "ops_llm.h"
#include "async_printer.h"

extern int launch_transformer_step(cudaGraphExec_t graphExec);
extern int init_transformer_graph(const void **dW_q, const void **dW_k, const void **dW_v, const void **dW_attn, const void **dW_gate, const void **dW_up, const void **dW_down,
                                   float *dx, float *dh1, float *dh2, float *dK_cache, float *dV_cache,
                                   int n_layers, int max_steps, cudaGraphExec_t *graphExec);
extern int embed_token_gpu(const void *dW_embd, int token_id, float *dx, int dim);
extern int sample_next_token_id(const void *dW_head, const float *dx, const float *d_gamma, float *d_logits, int *d_out_id, int vocab_size, int dim);

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("\n=======================================================\n");
    printf("   tinytorch Interactive AI Terminal Chat Engine\n");
    printf("   Model: Qwen2.5-0.5B-Instruct-Q4_0.gguf\n");
    printf("   Hardware: RTX 3050 Laptop GPU (sm_86)\n");
    printf("   Type '/exit' or Ctrl+C to quit.\n");
    printf("=======================================================\n\n");

    const char *model_path = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    GGUFModel *model = gguf_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load GGUF model from %s\n", model_path);
        return 1;
    }

    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok) {
        fprintf(stderr, "Failed to initialize BPE tokenizer\n");
        gguf_free(model);
        return 1;
    }

    const int sample_token_ids[] = {
        44220, 372, 24231, 374, 264, 18512, 36512, 95296, 5440, 429, 32408, 288, 279, 6872, 315, 30128, 29026, 311, 11625, 5322, 2238, 6351, 369, 28824, 18495, 382, 1592, 56914, 510, 16, 13, 7297, 3487, 25, 3406, 11516, 646, 3000, 304, 5248, 5302, 24303, 11, 27362, 15279, 82599, 624, 17, 13, 4863, 4044, 478, 25, 3406, 11516, 646, 387, 46601, 745, 10592, 11, 10693, 1995, 8692, 518, 29969, 24722, 624, 18, 13, 5665, 2202, 25, 55313, 5302, 646, 96068, 4396, 12716, 323, 21725, 4969, 6174, 13
    };
    int num_sample_tokens = (int)(sizeof(sample_token_ids) / sizeof(sample_token_ids[0]));
    int num_gen_tokens = 80;

    size_t attn_bytes = (896 * 896 / 32) * 18;
    size_t mlp_bytes = (4864 * 896 / 32) * 18;
    size_t head_bytes = (151936 * 896 / 32) * 18;

    void *dW_embd;
    cudaMalloc(&dW_embd, head_bytes);
    GGUFTensor *t_embd = gguf_get_tensor(model, "token_embd.weight");
    if (t_embd && t_embd->data) {
        cudaMemcpy(dW_embd, t_embd->data, t_embd->size_bytes < head_bytes ? t_embd->size_bytes : head_bytes, cudaMemcpyHostToDevice);
    }

    void *dW_head;
    cudaMalloc(&dW_head, head_bytes);
    GGUFTensor *t_head = gguf_get_tensor(model, "token_embd.weight");
    if (t_head && t_head->data) {
        cudaMemcpy(dW_head, t_head->data, t_head->size_bytes < head_bytes ? t_head->size_bytes : head_bytes, cudaMemcpyHostToDevice);
    }

    float *d_gamma = NULL;
    GGUFTensor *t_norm = gguf_get_tensor(model, "output_norm.weight");
    if (t_norm && t_norm->data) {
        cudaMalloc((void**)&d_gamma, 896 * sizeof(float));
        cudaMemcpy(d_gamma, t_norm->data, 896 * sizeof(float), cudaMemcpyHostToDevice);
    }

    float *d_logits;
    int *d_out_id;
    cudaMalloc((void**)&d_logits, 151936 * sizeof(float));
    cudaMalloc((void**)&d_out_id, sizeof(int));

    size_t kv_bytes = (896 * 128 / 32) * 18;

    void *dW_q[24], *dW_k[24], *dW_v[24], *dW_attn[24], *dW_gate[24], *dW_up[24], *dW_down[24];
    for (int l = 0; l < 24; l++) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        GGUFTensor *t_q = gguf_get_tensor(model, name);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        GGUFTensor *t_k = gguf_get_tensor(model, name);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        GGUFTensor *t_v = gguf_get_tensor(model, name);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        GGUFTensor *t_attn = gguf_get_tensor(model, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        GGUFTensor *t_gate = gguf_get_tensor(model, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        GGUFTensor *t_up = gguf_get_tensor(model, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        GGUFTensor *t_down = gguf_get_tensor(model, name);

        cudaMalloc((void**)&dW_q[l], attn_bytes);
        cudaMalloc((void**)&dW_k[l], kv_bytes);
        cudaMalloc((void**)&dW_v[l], kv_bytes);
        cudaMalloc((void**)&dW_attn[l], attn_bytes);
        cudaMalloc((void**)&dW_gate[l], mlp_bytes);
        cudaMalloc((void**)&dW_up[l], mlp_bytes);
        cudaMalloc((void**)&dW_down[l], mlp_bytes);

        if (t_q && t_q->data) cudaMemcpy(dW_q[l], t_q->data, t_q->size_bytes < attn_bytes ? t_q->size_bytes : attn_bytes, cudaMemcpyHostToDevice);
        if (t_k && t_k->data) cudaMemcpy(dW_k[l], t_k->data, t_k->size_bytes < kv_bytes ? t_k->size_bytes : kv_bytes, cudaMemcpyHostToDevice);
        if (t_v && t_v->data) cudaMemcpy(dW_v[l], t_v->data, t_v->size_bytes < kv_bytes ? t_v->size_bytes : kv_bytes, cudaMemcpyHostToDevice);
        if (t_attn && t_attn->data) cudaMemcpy(dW_attn[l], t_attn->data, t_attn->size_bytes < attn_bytes ? t_attn->size_bytes : attn_bytes, cudaMemcpyHostToDevice);
        if (t_gate && t_gate->data) cudaMemcpy(dW_gate[l], t_gate->data, t_gate->size_bytes < mlp_bytes ? t_gate->size_bytes : mlp_bytes, cudaMemcpyHostToDevice);
        if (t_up && t_up->data) cudaMemcpy(dW_up[l], t_up->data, t_up->size_bytes < mlp_bytes ? t_up->size_bytes : mlp_bytes, cudaMemcpyHostToDevice);
        if (t_down && t_down->data) cudaMemcpy(dW_down[l], t_down->data, t_down->size_bytes < mlp_bytes ? t_down->size_bytes : mlp_bytes, cudaMemcpyHostToDevice);
    }

    float *dx, *dh1, *dh2, *dK_cache, *dV_cache;
    cudaMalloc((void**)&dx, 4864 * sizeof(float));
    cudaMalloc((void**)&dh1, 4864 * sizeof(float));
    cudaMalloc((void**)&dh2, 4864 * sizeof(float));
    cudaMalloc((void**)&dK_cache, 512 * 14 * 64 * sizeof(float));
    cudaMalloc((void**)&dV_cache, 512 * 14 * 64 * sizeof(float));

    cudaMemset(dx, 0, 4864 * sizeof(float));
    cudaMemset(dh1, 0, 4864 * sizeof(float));
    cudaMemset(dh2, 0, 4864 * sizeof(float));
    cudaMemset(dK_cache, 0, 512 * 14 * 64 * sizeof(float));
    cudaMemset(dV_cache, 0, 512 * 14 * 64 * sizeof(float));

    cudaGraphExec_t graphExec = NULL;
    init_transformer_graph((const void**)dW_q, (const void**)dW_k, (const void**)dW_v, (const void**)dW_attn, (const void**)dW_gate, (const void**)dW_up, (const void**)dW_down,
                           dx, dh1, dh2, dK_cache, dV_cache, 24, num_gen_tokens, &graphExec);

    // Warmup graph execution
    for (int i = 0; i < 10; i++) {
        launch_transformer_step(graphExec);
    }
    cudaDeviceSynchronize();

    char user_input[1024];

    while (1) {
        printf("\nUser > ");
        fflush(stdout);

        if (!fgets(user_input, sizeof(user_input), stdin)) {
            break;
        }

        // Strip trailing newline
        size_t len = strlen(user_input);
        while (len > 0 && (user_input[len - 1] == '\n' || user_input[len - 1] == '\r')) {
            user_input[--len] = '\0';
        }

        if (len == 0) continue;
        if (strcmp(user_input, "/exit") == 0 || strcmp(user_input, "quit") == 0 || strcmp(user_input, "exit") == 0) {
            printf("Exiting tinytorch chat session. Goodbye!\n");
            break;
        }

        printf("\ntinytorch > ");
        fflush(stdout);

        char formatted_prompt[1280];
        snprintf(formatted_prompt, sizeof(formatted_prompt), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_input);

        int prompt_tokens[256];
        int n_prompt = bpe_encode(tok, formatted_prompt, prompt_tokens, 256);
        if (n_prompt <= 0) {
            prompt_tokens[0] = 99; // fallback
            n_prompt = 1;
        }

        // Start Lock-Free Async Terminal Printer Thread for streaming response
        AsyncPrinter *ap = async_printer_start();

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // 1. Prefill Phase (process prompt tokens through GPU Transformer)
        for (int i = 0; i < n_prompt; i++) {
            embed_token_gpu(dW_embd, prompt_tokens[i], dx, 896);
            launch_transformer_step(graphExec);
        }

        int next_tok = sample_next_token_id(dW_head, dx, d_gamma, d_logits, d_out_id, 151936, 896);

        // 2. Generation Phase (stream dynamic output tokens)
        int gen_count = 0;
        for (int step = 0; step < 80; step++) {
            if (next_tok == tok->eos_id || next_tok == 151643 || next_tok == 151645) {
                break;
            }

            int out_len = 0;
            const char *token_str = bpe_decode_token(tok, next_tok, &out_len);
            async_printer_push(ap, token_str, out_len);
            gen_count++;

            embed_token_gpu(dW_embd, next_tok, dx, 896);
            launch_transformer_step(graphExec);
            next_tok = sample_next_token_id(dW_head, dx, d_gamma, d_logits, d_out_id, 151936, 896);
        }

        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);

        async_printer_stop_and_flush(ap);

        double elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        double tok_per_sec = gen_count > 0 ? (double)gen_count / elapsed_sec : 0.0;

        printf("\n\n[Stats: %d tokens | %.1f ms | %.1f tok/s]\n", gen_count, elapsed_sec * 1000.0, tok_per_sec);
    }

    for (int l = 0; l < 24; l++) {
        cudaFree(dW_q[l]);
        cudaFree(dW_k[l]);
        cudaFree(dW_v[l]);
        cudaFree(dW_attn[l]);
        cudaFree(dW_gate[l]);
        cudaFree(dW_up[l]);
        cudaFree(dW_down[l]);
    }
    cudaFree(dW_head);
    cudaFree(d_logits);
    cudaFree(d_out_id);
    cudaFree(dx);
    cudaFree(dh1);
    cudaFree(dh2);
    cudaFree(dK_cache);
    cudaFree(dV_cache);
    if (graphExec) cudaGraphExecDestroy(graphExec);
    bpe_tokenizer_free(tok);
    gguf_free(model);

    return 0;
}
