/* moe_router.c -- MoE routing logic. See moe_router.h for contract.
 *
 * Oracle reference (local llama.cpp source, src/llama-graph.cpp
 * build_moe_ffn):
 *   - softmax/sigmoid gate ops: lines ~1979-1990
 *   - SOFTMAX_WEIGHT: weights = raw top-k probs, then softmax over the
 *     k of them AFTER selection: lines ~2069-2075
 *   - weight-norm with sum clamp 6.103515625e-5f: lines ~2077-2094
 *     ("Avoid division by zero, clamp to smallest number representable
 *     by F16": ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY))
 *   - w_scale skipped when exactly 0 or 1: lines ~2095-2099
 */

#include "moe_router.h"

#include <math.h>
#include <string.h>

/* Smallest positive normal value representable by F16 = 2^-14. Matches
 * the literal in oracle build_moe_ffn weight-sum clamp. */
#define TT_MOE_WSUM_CLAMP_MIN 6.103515625e-5f

static int tt_moe_cfg_valid(const tt_moe_cfg *c)
{
    if (!c) return 0;
    if (c->n_expert <= 0 || c->n_expert > TT_MOE_MAX_EXPERTS) return 0;
    if (c->top_k <= 0 || c->top_k > TT_MOE_MAX_K) return 0;
    if (c->top_k > c->n_expert) return 0;
    switch (c->gate_op) {
        case TT_MOE_GATE_SOFTMAX:
        case TT_MOE_GATE_SIGMOID:
        case TT_MOE_GATE_SOFTMAX_WEIGHT:
            break;
        default:
            return 0;
    }
    return 1;
}

int tt_moe_route(const float *logits, const tt_moe_cfg *cfg,
                 tt_moe_route_out *out)
{
    if (!logits || !cfg || !out || !tt_moe_cfg_valid(cfg)) return -1;

    const int n = cfg->n_expert;
    const int k = cfg->top_k;

    for (int i = 0; i < n; i++) {
        if (!isfinite(logits[i])) return -1;
    }

    /* Stage 1: per-expert probs.
     * softmax: full-row max-subtracted exp / sum (oracle ggml_soft_max).
     * sigmoid: elementwise 1/(1+e^-x).
     * softmax_weight: leave RAW logits here (oracle `probs = logits`);
     * renormalization happens over the top-k only, after selection. */
    float probs[TT_MOE_MAX_EXPERTS];
    if (cfg->gate_op == TT_MOE_GATE_SIGMOID) {
        for (int i = 0; i < n; i++)
            probs[i] = 1.0f / (1.0f + expf(-logits[i]));
    } else if (cfg->gate_op == TT_MOE_GATE_SOFTMAX_WEIGHT) {
        for (int i = 0; i < n; i++)
            probs[i] = logits[i];
    } else {
        float mx = logits[0];
        for (int i = 1; i < n; i++)
            if (logits[i] > mx) mx = logits[i];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            probs[i] = expf(logits[i] - mx);
            sum += probs[i];
        }
        if (!(sum > 0.0f)) return -1;
        for (int i = 0; i < n; i++)
            probs[i] /= sum;
    }

    /* Stage 2: top-k selection via running insertion sort over a k-slot
     * list kept sorted by DESCENDING prob; ties -> LOWER expert index
     * first (deterministic stable-argsort semantics). Scanning experts
     * in ascending index order makes ties resolve to the earlier id. */
    int sel[TT_MOE_MAX_K];
    int cnt = 0;
    for (int e = 0; e < n; e++) {
        int better_than_last;
        if (cnt < k) {
            better_than_last = 1;
        } else {
            better_than_last = (probs[e] > probs[sel[k - 1]]) ||
                               (probs[e] == probs[sel[k - 1]] && e < sel[k - 1]);
            if (!better_than_last) continue;
        }
        int lim = (cnt < k) ? cnt : k - 1;
        int pos = lim;
        while (pos > 0 &&
               (probs[e] > probs[sel[pos - 1]] ||
                (probs[e] == probs[sel[pos - 1]] && e < sel[pos - 1])))
            pos--;
        for (int j = lim; j > pos; j--)      /* open slot */
            sel[j] = sel[j - 1];
        sel[pos] = e;
        if (cnt < k) cnt++;
    }
    if (cnt != k) return -1;

    /* Stage 3: gather selected weights. */
    float w[TT_MOE_MAX_K];
    for (int c = 0; c < k; c++)
        w[c] = probs[sel[c]];

    /* Stage 3a: SOFTMAX_WEIGHT — softmax over ONLY the k selected
     * values (oracle applies it post-selection). */
    if (cfg->gate_op == TT_MOE_GATE_SOFTMAX_WEIGHT) {
        float mx = w[0];
        for (int c = 1; c < k; c++)
            if (w[c] > mx) mx = w[c];
        float sum = 0.0f;
        for (int c = 0; c < k; c++) {
            w[c] = expf(w[c] - mx);
            sum += w[c];
        }
        if (!(sum > 0.0f)) return -1;
        for (int c = 0; c < k; c++)
            w[c] /= sum;
    }

    /* Stage 3b: optional normalization, sum clamped to F16-min-normal
     * to avoid div-by-zero (exact oracle behavior). */
    if (cfg->weight_norm) {
        float wsum = 0.0f;
        for (int c = 0; c < k; c++)
            wsum += w[c];
        if (wsum < TT_MOE_WSUM_CLAMP_MIN)
            wsum = TT_MOE_WSUM_CLAMP_MIN;
        for (int c = 0; c < k; c++)
            w[c] /= wsum;
    }

    /* Stage 3c: routed scaling — skip when exactly 0 or 1 (oracle). */
    if (cfg->w_scale != 0.0f && cfg->w_scale != 1.0f) {
        for (int c = 0; c < k; c++)
            w[c] *= cfg->w_scale;
    }

    for (int c = 0; c < k; c++) {
        out->ids[c] = (int32_t)sel[c];
        out->weights[c] = w[c];
    }
    return k;
}

int tt_moe_build_plan(const int32_t *ids, const int32_t *slots, int n_sel,
                      tt_moe_plan_pair *out_pairs)
{
    if (!ids || !out_pairs || n_sel < 0) return -1;
    if (n_sel > TT_MOE_MAX_EXPERTS * TT_MOE_MAX_K) return -1;

    /* Selection sort grouped by expert ascending; slot tag carried
     * through. Stable within an expert group (input order preserved),
     * so multi-token plans stay deterministic. */
    const int m = n_sel;
    unsigned char used[TT_MOE_MAX_EXPERTS * TT_MOE_MAX_K];
    for (int i = 0; i < m; i++) used[i] = 0;

    int out_n = 0;
    for (;;) {
        int best = -1;
        for (int i = 0; i < m; i++) {
            if (used[i]) continue;
            if (best < 0 || ids[i] < ids[best]) best = i;
        }
        if (best < 0) break;
        used[best] = 1;
        out_pairs[out_n].expert = ids[best];
        out_pairs[out_n].slot   = slots ? slots[best] : best;
        out_n++;
    }
    return out_n;
}

void tt_moe_combine(const float *const *expert_outs, const float *weights,
                    int n_sel, int dim,
                    const float *shared_out, float *routed_out)
{
    if (!routed_out || dim <= 0) return;

    if (n_sel <= 0 || !expert_outs || !weights) {
        if (shared_out) memcpy(routed_out, shared_out, (size_t)dim * sizeof(float));
        else memset(routed_out, 0, (size_t)dim * sizeof(float));
        return;
    }

    for (int d = 0; d < dim; d++)
        routed_out[d] = weights[0] * expert_outs[0][d];
    for (int i = 1; i < n_sel; i++) {
        const float w = weights[i];
        const float *o = expert_outs[i];
        for (int d = 0; d < dim; d++)
            routed_out[d] += w * o[d];
    }

    /* Shared-expert hook: parallel dense-FFN branch added after the
     * routed weighted sum (gemma4/qwen35moe pattern; our dense MLP block
     * is the implementation per docs/plans/2026-08-27-moe-notes.md). */
    if (shared_out) {
        for (int d = 0; d < dim; d++)
            routed_out[d] += shared_out[d];
    }
}
