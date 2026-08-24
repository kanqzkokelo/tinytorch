// Single-shot generation over the M6-correct engine.
// The prompt is tokenized, prefilled into the KV cache, and generation
// continues from it — the prompt actually reaches the model.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#include "async_printer.h"

int main(int argc, char **argv) {
    const char *prompt = argc > 1 ? argv[1]
        : "Explain quantum computing in one sentence.";
    int target_tokens = argc > 2 ? atoi(argv[2]) : 64;
    if (target_tokens < 1) target_tokens = 1;

    const char *model_path =
        "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    const int MAX_CTX = 1024;

    GGUFModel *model = gguf_load(model_path);
    if (!model) return 1;
    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok) return 1;

    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    if (cfg.dim == 0) { fprintf(stderr, "config failed\n"); return 1; }
    printf("[run] dim=%d ffn=%d layers=%d heads=%d kv_heads=%d vocab=%d rope_base=%g\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
           cfg.vocab, cfg.rope_base);

    Qwen2Engine *eng = qwen2_engine_create(&cfg, model);
    if (!eng) { fprintf(stderr, "engine init failed\n"); return 1; }

    char formatted[8192];
    if (!getenv("TT_RAW_PROMPT"))
        snprintf(formatted, sizeof(formatted),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    else
        snprintf(formatted, sizeof(formatted), "%s", prompt);

    int prompt_tokens[512];
    int n_prompt = bpe_encode(tok, formatted, prompt_tokens, 512);
    printf("[run] prompt: %d tokens\n", n_prompt);
    if (n_prompt <= 0) return 1;

    struct timespec t0, t1, tp0, tp1;
    AsyncPrinter *ap = async_printer_start();
    clock_gettime(CLOCK_MONOTONIC, &tp0);

    if (qwen2_engine_prefill(eng, prompt_tokens, n_prompt)) {
        fprintf(stderr, "prefill failed\n"); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);   /* decode-only window starts here */

    int gen_count = 0;
    for (int s = 0; s < target_tokens && qwen2_engine_pos(eng) < MAX_CTX - 1; s++) {
        const int id = qwen2_engine_next(eng);
        if (id < 0 || id == tok->eos_id || id == 151643 || id == 151645) break;
        int out_len = 0;
        const char *txt = bpe_decode_token(tok, id, &out_len);
        async_printer_push(ap, txt, out_len);
        gen_count++;
    }
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    async_printer_stop_and_flush(ap);

    const double dec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    const double tot = (t1.tv_sec - tp0.tv_sec) + (t1.tv_nsec - tp0.tv_nsec) * 1e-9;
    printf("\"\n[gen: %d tokens | decode %.1f tok/s | incl prefill %.1f tok/s | greedy]\n",
           gen_count, gen_count / dec, gen_count / tot);
    printf("STATS tokens=%d prefill=%d decode_us=%.0f\n", gen_count, n_prompt, dec * 1e6);

    qwen2_engine_free(eng);
    bpe_tokenizer_free(tok);
    gguf_free(model);
    return 0;
}
