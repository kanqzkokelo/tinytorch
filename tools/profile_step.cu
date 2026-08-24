// Per-step decode profiler (M6.3b Task 0).
// Loads the engine exactly like tools/dump_logits.c, then:
//   - warms up 3 steps
//   - times K=20 decode steps with cudaEvents around each step call
//   - prints STEP_MS <ms> of the MINIMUM
// With TT_PROFILE=1 the engine runs EAGER (TT_PROFILE forces no-graph inside
// qwen2_engine_create; event records are illegal inside a captured region)
// and per-stage accumulators are reset before the timed loop and printed as
// a median table after it.
//
// Usage: profile_step <id,id,...>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"

#define K_STEPS 20
#define K_WARM 3

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s id,id,..\n", argv[0]); return 1; }
    const char *model_path = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    GGUFModel *m = gguf_load(model_path);
    if (!m) { fprintf(stderr, "gguf load failed: %s\n", model_path); return 1; }
    printf("model: %s\n", model_path);
    TTConfig cfg = tt_config_from_gguf(m, 1024);
    if (cfg.dim == 0) { fprintf(stderr, "config failed\n"); return 1; }
    Qwen2Engine *e = qwen2_engine_create(&cfg, m);
    if (!e) return 1;

    int toks[512], n = 0;
    char *save = NULL, *p = strtok_r(argv[1], ",", &save);
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

    if (qwen2_engine_prefill(e, toks, n)) { fprintf(stderr, "prefill failed\n"); return 1; }

    const int profiling = getenv("TT_PROFILE") ? 1 : 0;

    /* First step via the public API: under graph mode this performs the eager
     * sample + capture and returns s_last (prompt's continuation); under
     * TT_PROFILE-forced eager mode it samples AND advances. */
    int id = qwen2_engine_next(e);
    if (id < 0) { fprintf(stderr, "first step failed: %d\n", id); return 1; }

    /* One timed step: replay hook feeds back the previous sample (graph mode),
     * or the legacy public next() (forced-eager profiling mode). */
    #define ONE_STEP(prev) (profiling ? qwen2_engine_next(e) \
                                      : qwen2_debug_replay_step(e, (prev)))

    cudaStream_t st = (cudaStream_t)qwen2_debug_stream(e);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    /* warm up: 3 replays of the same path being timed */
    for (int k = 0; k < K_WARM; k++) {
        const int r = ONE_STEP(id);
        if (r < 0) { fprintf(stderr, "warmup step %d failed: %d\n", k, r); return 1; }
        id = r;
    }

    if (profiling) qwen2_debug_profile_reset();

    float best = 1e9f;
    for (int k = 0; k < K_STEPS; k++) {
        cudaEventRecord(ev0, st);
        const int r = ONE_STEP(id);
        cudaEventRecord(ev1, st);
        cudaEventSynchronize(ev1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (r < 0) { fprintf(stderr, "timed step %d failed: %d\n", k, r); return 1; }
        id = r;
        if (ms < best) best = ms;
    }

    printf("STEP_MS %.3f\n", best);

    if (profiling) qwen2_debug_profile_report(K_STEPS);

    #undef ONE_STEP
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    qwen2_engine_free(e);
    return 0;
}
