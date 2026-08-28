#ifndef QWEN2_ENGINE_H
#define QWEN2_ENGINE_H

#include "loader_gguf.h"
#include "arch_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CUDA_STREAM_T_DEFINED
#define CUDA_STREAM_T_DEFINED
typedef struct CUstream_st *cudaStream_t;
#endif

/* Model geometry + M7 architecture traits — every field derived from
 * GGUF metadata, never hardcoded. The trait block selects kernel variants
 * at fixed variation points inside the forward pass (rope style, activation,
 * q/k norm, logit softcap, SWA window, tied embeddings). */
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
    TTraits tr;       /* per-family behavior (arch_registry defaults) */
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

/* Speculative-decode verify pass: feed N candidate tokens in one batched
 * call, return N sets of logits (one per candidate position).
 *
 *   h_candidate_tokens : host array of N int32 token ids (typically
 *                        [current, draft_0, draft_1, draft_2]).
 *   n_candidate        : count in [1, max_ctx-pos]. 2-4 typical.
 *   out_logits         : pre-allocated DEVICE buffer of size
 *                        n_candidate * vocab floats. Layout is
 *                        [candidate_0 logits | candidate_1 logits | ...].
 *                        Filled row-major contiguous; no D2H copy.
 *
 * Semantics: identical to N sequential qwen2_engine_next() calls would
 * produce for their per-token logits — but no sampling/argmax/D2H sync
 * happens inside. KV cache slots [pos, pos+n) are populated; host pos
 * advances by n. To reject all candidates, the caller must restore pos
 * and overwrite the KV cache (Task 3 / orchestrator responsibility).
 *
 * Returns 0 on success, negative on error. */
int qwen2_engine_verify_speculative(Qwen2Engine *e,
                                    const int *h_candidate_tokens,
                                    int n_candidate,
                                    float *out_logits);

/* Debug/profiling hooks.
 * qwen2_debug_replay_step performs exactly the graph-replay body of
 * qwen2_engine_next: H2D next_tok -> graph launch -> D2H sample -> sync,
 * plus pos/pending bookkeeping. Returns the sampled id, or -1 when the
 * graph path is unavailable (eager unsupported for profiling). Used by
 * tools/profile_step.cu to time one replayed step with cudaEvents. */
int qwen2_debug_replay_step(Qwen2Engine *e, int next_tok);

/* e->stream as void* so callers can record timing events on the same
 * stream the decode work runs on (header stays CUDA-type-free). */
void *qwen2_debug_stream(Qwen2Engine *e);

/* TT_PROFILE per-stage accumulators (see kernels/qwen2_cuda.cu).
 * reset clears all stage samples; report(nsteps) splits each stage's
 * samples into nsteps equal-count groups (kernel counts per step are
 * deterministic), sums each group, and prints the median 'PROFILE'
 * table of per-step ms. */
void qwen2_debug_profile_reset(void);
void qwen2_debug_profile_report(int nsteps);

/* sampling: temp>0 enables Gumbel-max sampling + repeat penalty; default greedy */
void qwen2_engine_set_sampling(Qwen2Engine *e, float temp, int topk, float penalty);

/* debug/parity helpers */
int qwen2_debug_copy_x(Qwen2Engine *e, float *host, int n);
int qwen2_debug_copy_kv(Qwen2Engine*, int layer, float*, long);
int qwen2_debug_copy_xn(Qwen2Engine*, float*, int);
int qwen2_debug_copy_logits(Qwen2Engine *e, float *host, int n);

/* Single-token forward WITHOUT sampling/D2H: embed(tok) -> forward_layers
 * (writes K/V to slot pos, advances pos by 1) -> rmsnorm -> lm_head ->
 * softcap.  Result copied to host_logits (size >= vocab).  Used by the
 * spec-verify test to produce a golden set of per-token logits for
 * bit-exact comparison with qwen2_engine_verify_speculative.  */
int qwen2_engine_step_logits(Qwen2Engine *e, int tok, float *host_logits);

// Batched 2D prefill GEMM: Y [N, M] = X [N, K] * W^T [M, K]
// W is Q4_0 quantized. Evaluates N prompt tokens in parallel for N >= 32.
int tt_gemm_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM, int M, int K, int N, cudaStream_t s);
int prefill_batched_gemm(Qwen2Engine *e, const int *toks, int n, float *h_x_out);

void qwen2_engine_free(Qwen2Engine *e);

#ifdef __cplusplus
}
#endif
#endif /* QWEN2_ENGINE_H */
