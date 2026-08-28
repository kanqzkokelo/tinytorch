// Time qwen2_engine_prefill at several token counts.
// Usage: bench_prefill [--model PATH] [--tokens T,T,...] [--repeats N]
//   default model: $TT_MODEL or data/models/qwen2.5-0.5b-instruct-q4_0.gguf
//   default tokens: 8,16,32,64,128
//   default repeats: 2
// Emits CSV: n_tokens,prefill_ms,prefill_tok_s
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *model_path = getenv("TT_MODEL") ? getenv("TT_MODEL")
                          : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    int repeats = 2;
    int tokens[16]; int n_tokens_set = 0;
    int argi = 1;
    while (argi < argc) {
        if (!strcmp(argv[argi], "--model") && argi + 1 < argc) {
            model_path = argv[argi + 1]; argi += 2;
        } else if (!strcmp(argv[argi], "--repeats") && argi + 1 < argc) {
            repeats = atoi(argv[argi + 1]); argi += 2;
        } else if (!strcmp(argv[argi], "--tokens") && argi + 1 < argc) {
            char *save = NULL, *p = strtok_r(argv[argi + 1], ",", &save);
            while (p && n_tokens_set < 16) {
                tokens[n_tokens_set++] = atoi(p);
                p = strtok_r(NULL, ",", &save);
            }
            argi += 2;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[argi]); return 1;
        }
    }
    if (n_tokens_set == 0) {
        const int defaults[] = {8, 16, 32, 64, 128};
        for (int i = 0; i < 5; i++) tokens[n_tokens_set++] = defaults[i];
    }
    if (repeats < 1) repeats = 1;

    GGUFModel *m = gguf_load(model_path);
    if (!m) { fprintf(stderr, "gguf load failed: %s\n", model_path); return 1; }
    int max_n = 0;
    for (int i = 0; i < n_tokens_set; i++) if (tokens[i] > max_n) max_n = tokens[i];
    int max_ctx = max_n + 16;
    if (max_ctx < 1024) max_ctx = 1024;
    TTConfig cfg = tt_config_from_gguf(m, max_ctx);
    if (cfg.dim == 0) { fprintf(stderr, "config failed\n"); return 1; }
    Qwen2Engine *e = qwen2_engine_create(&cfg, m);
    if (!e) return 1;
    printf("model: %s\n", model_path);
    printf("n_tokens,prefill_ms,prefill_tok_s\n");

    for (int ti = 0; ti < n_tokens_set; ti++) {
        const int n = tokens[ti];
        if (n > max_ctx) { fprintf(stderr, "skip n=%d > max_ctx=%d\n", n, max_ctx); continue; }
        int *toks = (int *)malloc(sizeof(int) * n);
        for (int i = 0; i < n; i++) toks[i] = 10 + (i % 100);
        double best_ms = 1e9;
        for (int r = 0; r < repeats; r++) {
            qwen2_engine_reset(e);
            const double t0 = now_ms();
            int rc = qwen2_engine_prefill(e, toks, n);
            const double t1 = now_ms();
            if (rc) { fprintf(stderr, "prefill rc=%d n=%d\n", rc, n); break; }
            const double dt = t1 - t0;
            if (dt < best_ms) best_ms = dt;
        }
        printf("%d,%.3f,%.1f\n", n, best_ms, n / (best_ms / 1000.0));
        fflush(stdout);
        free(toks);
    }
    qwen2_engine_free(e);
    return 0;
}
