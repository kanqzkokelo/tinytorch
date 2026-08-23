// Oracle: greedy next-token logits from llama.cpp for given prompt tokens.
// Usage: oracle_logits MODEL "prompt text"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL PROMPT\n", argv[0]); return 1; }
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                       /* CPU oracle: deterministic */
    struct llama_model *model = llama_load_model_from_file(argv[1], mp);
    if (!model) return 1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    struct llama_context *ctx = llama_new_context_with_model(model, cp);

    // tokenize prompt (no BOS for qwen)
    int toks[512];
    int n = llama_tokenize(vocab, argv[2], strlen(argv[2]), toks, 512, false, false);
    fprintf(stderr, "[oracle] %d tokens:", n);
    for (int i = 0; i < n; i++) fprintf(stderr, " %d", toks[i]);
    fprintf(stderr, "\n");

    llama_batch batch = llama_batch_get_one(toks, n);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }

    const float *logits = llama_get_logits_ith(ctx, n - 1);
    const int nv = llama_vocab_n_tokens(vocab);
    // top-8
    int idx[8]; float val[8];
    for (int k = 0; k < 8; k++) { idx[k] = -1; val[k] = -1e30f; }
    for (int i = 0; i < nv; i++) {
        for (int k = 0; k < 8; k++) {
            if (logits[i] > val[k]) {
                for (int m = 7; m > k; m--) { val[m]=val[m-1]; idx[m]=idx[m-1]; }
                val[k] = logits[i]; idx[k] = i;
                break;
            }
        }
    }
    printf("TOP8:");
    for (int k = 0; k < 8; k++) printf(" (%d,%.4f)", idx[k], val[k]);
    printf("\n");
    // full logits dump option
    if (argc > 3 && !strcmp(argv[3], "--dump")) {
        FILE *f = fopen(argv[4], "wb");
        fwrite(logits, sizeof(float), nv, f);
        fclose(f);
        printf("DUMPED %d logits\n", nv);
    }
    return 0;
}
