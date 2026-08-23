#ifndef QWEN2_ENGINE_H
#define QWEN2_ENGINE_H

#include "loader_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Model geometry — every field derived from GGUF metadata, never hardcoded. */
typedef struct {
    int dim;          /* embedding / residual stream width            */
    int hidden_dim;   /* FFN intermediate width                       */
    int n_layers;
    int n_heads;      /* query heads                                  */
    int n_kv_heads;   /* key/value heads (GQA when < n_heads)         */
    int head_dim;
    int vocab;
    int max_ctx;      /* KV cache capacity (caller-chosen)            */
    float rms_eps;
    float rope_base;
} TTConfig;

/* Returns cfg with dim==0 if required metadata keys are missing. */
TTConfig tt_config_from_gguf(const GGUFModel *m, int max_ctx);

typedef struct Qwen2Engine Qwen2Engine;

Qwen2Engine *qwen2_engine_create(const TTConfig *cfg, GGUFModel *m);

/* Feed prompt tokens through the layers, populating the KV cache. */
int qwen2_engine_prefill(Qwen2Engine *e, const int *toks, int n);

/* Sample one token (greedy argmax) from the current hidden state,
 * then feed it back through the layers. Returns the token id. */
int qwen2_engine_next(Qwen2Engine *e);

int  qwen2_engine_pos(const Qwen2Engine *e);

/* debug/parity helpers */
int qwen2_debug_copy_x(Qwen2Engine *e, float *host, int n);
int qwen2_debug_copy_kv(Qwen2Engine*, int layer, float*, long);
int qwen2_debug_copy_xn(Qwen2Engine*, float*, int);
int qwen2_debug_copy_logits(Qwen2Engine *e, float *host, int n);
void qwen2_engine_free(Qwen2Engine *e);

#ifdef __cplusplus
}
#endif
#endif /* QWEN2_ENGINE_H */
