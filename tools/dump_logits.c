// Dump final-position logits from the tinytorch engine for teacher-forced tokens.
// Usage: dump_logits [--model PATH] <id,id,...> [out.bin]
//   --model PATH   explicit model; falls back to $TT_MODEL, then the M6 default
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"

int qwen2_debug_copy_logits(Qwen2Engine*, float*, int);
int qwen2_debug_copy_xn(Qwen2Engine*, float*, int);

int main(int argc, char **argv) {
    const char *model_path = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    int argi = 1;
    if (argi < argc && strcmp(argv[argi], "--model") == 0) {
        if (argi + 1 >= argc) { fprintf(stderr, "--model requires a path\n"); return 1; }
        model_path = argv[argi + 1];
        argi += 2;
    } else if (getenv("TT_MODEL") && getenv("TT_MODEL")[0]) {
        model_path = getenv("TT_MODEL");
    }
    if (argc - argi < 1) {
        fprintf(stderr, "usage: %s [--model PATH] id,id,... [out.bin]\n", argv[0]);
        return 1;
    }
    GGUFModel *m = gguf_load(model_path);
    if (!m) { fprintf(stderr, "gguf load failed: %s\n", model_path); return 1; }
    printf("model: %s\n", model_path);
    TTConfig cfg = tt_config_from_gguf(m, 1024);
    if (cfg.dim == 0) { fprintf(stderr, "config failed\n"); return 1; }
    Qwen2Engine *e = qwen2_engine_create(&cfg, m);
    if (!e) return 1;

    const char *ids_arg = argv[argi];
    const char *dump_path = (argc - argi > 1) ? argv[argi + 1] : NULL;
    int toks[512], n = 0;
    char *save = NULL, *p = strtok_r((char *)ids_arg, ",", &save);
    while (p && n < 512) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (!end || *end != '\0' || end == p || v < 0 || v > INT32_MAX) {
            fprintf(stderr, "invalid token id: '%s' (expected integer)\n", p);
            qwen2_engine_free(e);
            return 1;
        }
        toks[n++] = (int)v;
        p = strtok_r(NULL, ",", &save);
    }

    // prefill all but last; then run final norm+logits manually via next() path
    // NOTE: next() also advances; we instead replicate its norm+logits stage here
    // by calling prefill on n-1 tokens, embedding the last, and reading logits
    // through next()'s buffers BEFORE advance — simplest correct route:
    // prefill everything, snapshot x, compute norm+logits via a second engine pass.
    if (qwen2_engine_prefill(e, toks, n)) { fprintf(stderr, "prefill failed\n"); return 1; }

    static float lg[262144];
    // next() computes rmsnorm->logits->argmax then advances; logits buffer still valid
    int id = qwen2_engine_next(e);   // consumes one step; logits correspond to LAST prompt token
    (void)id;
    const int nvocab = qwen2_debug_copy_logits(e, lg, 262144); /* accessor caps at true vocab */
    if (nvocab <= 0) { fprintf(stderr, "logits copy failed\n"); return 1; }

    int best = 0; float mv = -1e30f;
    for (int i = 0; i < nvocab; i++) if (lg[i] > mv) { mv = lg[i]; best = i; }
    printf("ARGMAX %d %.4f VOCAB %d\n", best, mv, nvocab);
    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        fwrite(lg, 4, nvocab, f);
        fclose(f);
    }
    qwen2_engine_free(e);
    return 0;
}
