// Interactive terminal chat over the M6-correct Qwen2 decode engine.
// Real prompt encoding, real prefill (KV cache populated), real generation.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#include "async_printer.h"

int main(void) {
    const char *model_path = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    const int MAX_CTX = 1024;

    printf("\n=======================================================\n");
    printf("   tinytorch chat — M6-correct Qwen2 decode engine\n");
    printf("   Model: %s\n", model_path);
    printf("   Type '/exit' to quit.\n");
    printf("=======================================================\n\n");

    GGUFModel *model = gguf_load(model_path);
    if (!model) return 1;

    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok) { gguf_free(model); return 1; }

    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    if (cfg.dim == 0) {
        fprintf(stderr, "[chat] could not derive config from GGUF metadata\n");
        bpe_tokenizer_free(tok); gguf_free(model); return 1;
    }
    printf("[chat] config: dim=%d ffn=%d layers=%d heads=%d kv_heads=%d head_dim=%d "
           "vocab=%d eps=%g rope_base=%g\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
           cfg.head_dim, cfg.vocab, cfg.rms_eps, cfg.rope_base);

    Qwen2Engine *eng = qwen2_engine_create(&cfg, model);
    if (!eng) { fprintf(stderr, "[chat] engine init failed\n"); return 1; }

    /* sampling defaults for chat (env-tunable); greedy via TT_GREEDY=1 */
    if (!getenv("TT_GREEDY")) {
        const float temp = getenv("TT_TEMP") ? atof(getenv("TT_TEMP")) : 0.8f;
        const float pen  = getenv("TT_REPEAT_PENALTY")
                           ? atof(getenv("TT_REPEAT_PENALTY")) : 1.15f;
        qwen2_engine_set_sampling(eng, temp, 40, pen);
    }

    char user_input[1024];
    while (1) {
        printf("\nUser > ");
        fflush(stdout);
        if (!fgets(user_input, sizeof(user_input), stdin)) break;

        size_t len = strlen(user_input);
        while (len > 0 && (user_input[len - 1] == '\n' || user_input[len - 1] == '\r'))
            user_input[--len] = '\0';
        if (len == 0) continue;
        if (!strcmp(user_input, "/exit") || !strcmp(user_input, "quit")) break;

        char formatted[1400];
        snprintf(formatted, sizeof(formatted),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_input);

        int prompt_tokens[512];
        int n_prompt = bpe_encode(tok, formatted, prompt_tokens, 512);
        if (n_prompt <= 0) { fprintf(stderr, "[chat] tokenization failed\n"); continue; }

        AsyncPrinter *ap = async_printer_start();
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        if (qwen2_engine_prefill(eng, prompt_tokens, n_prompt)) {
            fprintf(stderr, "\n[chat] context full — restart session or shorten input\n");
            async_printer_stop_and_flush(ap);
            continue;
        }

        int gen_count = 0;
        char turn_text[8192];
        size_t tl = 0;
        for (int step = 0; step < 256 && qwen2_engine_pos(eng) < MAX_CTX - 1; step++) {
            const int next_tok = qwen2_engine_next(eng);
            if (next_tok < 0 || next_tok == tok->eos_id ||
                next_tok == 151643 /* <|endoftext|> */ ||
                next_tok == 151645 /* <|im_end|> */)
                break;
            int out_len = 0;
            const char *s = bpe_decode_token(tok, next_tok, &out_len);
            /* stop-string guard: model sometimes spells control tokens as BPE
             * pieces instead of emitting their ids — truncate at the marker */
            if (tl + (size_t)out_len < sizeof(turn_text)) {
                memcpy(turn_text + tl, s, (size_t)out_len);
                tl += (size_t)out_len;
                turn_text[tl] = '\0';
            }
            const char *cut = NULL;
            static const char *markers[] = {"<|im_end|>", "<|endoftext|>",
                                            "<|im_start|>", NULL};
            for (int mi = 0; markers[mi]; mi++)
                if ((cut = strstr(turn_text, markers[mi]))) break;
            if (cut) {
                const size_t keep = (size_t)(cut - turn_text);
                if (keep > tl - (size_t)out_len)          /* marker inside this token */
                    async_printer_push(ap, turn_text + (tl - (size_t)out_len),
                                       (int)(keep - (tl - (size_t)out_len)));
                break;
            }
            async_printer_push(ap, s, out_len);
            gen_count++;
        }
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        async_printer_stop_and_flush(ap);

        const double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        const double tps = gen_count > 0 ? gen_count / sec : 0.0;
        printf("\n\n[%d tokens | %.1f ms | %.1f tok/s | ctx %d/%d]\n",
               gen_count, sec * 1000.0, tps,
               qwen2_engine_pos(eng), MAX_CTX);

        if (qwen2_engine_pos(eng) >= MAX_CTX - 8) {
            fprintf(stderr, "[chat] context nearly full — session should be reset\n");
        }
    }

    qwen2_engine_free(eng);
    bpe_tokenizer_free(tok);
    gguf_free(model);
    return 0;
}
