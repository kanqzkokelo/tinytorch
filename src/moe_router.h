/* moe_router.h -- M11 MoE routing: gate op + top-k + weight norm + plan.
 *
 * Build (standalone, no deps beyond libc -lm):
 *   gcc -std=c99 -O2 -Wall -Wextra -Isrc -o build/test_moe_router \
 *       src/moe_router.c tests/test_moe_router.c
 *
 * Pure C99, CPU-portable, deterministic (stable tie-breaking by lower
 * expert index). Mirrors oracle llama.cpp src/llama-graph.cpp
 * build_moe_ffn (lines ~1941-2105):
 *   - gate ops: softmax over all experts | sigmoid per expert |
 *     softmax_weight = raw logits, softmax applied AFTER top-k over the
 *     k selected probs only.
 *   - top-k selection, then optional weight normalization where the sum
 *     is clamped to 6.103515625e-5f (= 2^-14, smallest F16 normal;
 *     oracle: ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)).
 *   - w_scale applied after normalization, skipped when 0 or 1 exactly
 *     as the oracle does.
 *
 * NOTE: no Makefile change required for the shared library target:
 * lib target rule already compiles every C source under src/.
 * Deliberately NOT touching kernels/ or include/qwen2_engine.h until
 * engine integration lands.
 */

#ifndef TT_MOE_ROUTER_H
#define TT_MOE_ROUTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TT_MOE_MAX_EXPERTS 128
#define TT_MOE_MAX_K       16

typedef enum {
    TT_MOE_GATE_SOFTMAX = 0,        /* softmax over all n_expert logits   */
    TT_MOE_GATE_SIGMOID = 1,        /* sigmoid per logit                  */
    TT_MOE_GATE_SOFTMAX_WEIGHT = 2  /* raw logits; softmax over top-k only */
} tt_moe_gate_op;

typedef struct {
    int              n_expert;      /* total experts (<= TT_MOE_MAX_EXPERTS) */
    int              top_k;         /* experts per token (<= TT_MOE_MAX_K)   */
    tt_moe_gate_op   gate_op;
    int              weight_norm;   /* 0/1: normalize selected weights to sum 1 */
    float            w_scale;       /* post-norm scale; skipped if 0.0 or 1.0  */
} tt_moe_cfg;

/* Routed output for one token. ids sorted by DESCENDING weight; ties
 * broken deterministically by LOWER expert index first (matches stable
 * argsort semantics used for known-answer tests). */
typedef struct {
    int32_t ids[TT_MOE_MAX_K];
    float   weights[TT_MOE_MAX_K];
} tt_moe_route_out;

/* Full routing step: logits[n_expert] -> gate op -> top-k -> optional
 * sum-clamped norm -> w_scale. Returns number of selected experts (k),
 * or -1 on invalid cfg/logits (non-finite values rejected). */
int tt_moe_route(const float *logits, const tt_moe_cfg *cfg,
                 tt_moe_route_out *out);

/* Expert iteration plan builder: given selected ids (any order, e.g.
 * across multiple tokens/slots), emit (expert, slot) pairs grouped by
 * expert ascending — the loop order the engine executes GEMVs in.
 * slots[i] is the caller's opaque slot/tag carried through unchanged.
 * Returns pair count written to out_pairs (== n_sel), or -1 on NULL /
 * out-of-range id. O(n_sel^2) selection grouping; fine at k*tokens
 * sizes. Caller guarantees out_pairs capacity >= n_sel. */
typedef struct {
    int32_t expert;
    int32_t slot;
} tt_moe_plan_pair;

int tt_moe_build_plan(const int32_t *ids, const int32_t *slots, int n_sel,
                      tt_moe_plan_pair *out_pairs);

/* Weight application helper:
 *   routed_out[d] = sum_i weights[i] * expert_outs[i][d] + shared[d]
 * expert_outs: array of n_sel pointers, each to dim floats (slot-major,
 * i.e. expert_outs[i] pairs with sel_ids order, NOT grouped-plan order).
 * shared_out may be NULL (no shared-expert add). Shared-expert hook: in
 * the engine, shared_out = dense MLP block output computed in parallel
 * with the routed branch (gemma4-style parallel shared-MLP); pass it
 * here so a single kernel does combine+residual add. */
void tt_moe_combine(const float *const *expert_outs, const float *weights,
                    int n_sel, int dim,
                    const float *shared_out /* nullable */,
                    float *routed_out);

#ifdef __cplusplus
}
#endif

#endif /* TT_MOE_ROUTER_H */
