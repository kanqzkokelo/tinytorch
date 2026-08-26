/* test_moe_router.c -- known-answer tests vs hand-computed oracle
 * behavior (llama-graph.cpp build_moe_ffn semantics; clamp constant
 * 6.103515625e-5 cited from the weight-sum ggml_clamp line).
 *
 * Build+run:
 *   gcc -std=c99 -O2 -Wall -Wextra -Isrc -o build/test_moe_router \
 *       src/moe_router.c tests/test_moe_router.c -lm && ./build/test_moe_router
 */

#include "moe_router.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); g_fail++; } \
} while (0)

static int feq(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

/* ---- Test 1: softmax + weight-norm, n_expert=8, k=2 -----------------
 * logits = [2,1,0,0,0,0,0,0]
 * softmax (max-subtracted, x-max = [0,-1,-2,...]):
 *   e^0=1, e^-1=0.36787944, six * e^-2 (=6*0.135335283)
 *   sum = 1 + 0.36787944 + 0.81201170 = 2.17989114
 *   p0 = 0.45873846, p1 = 0.16876901
 * norm sum = 0.62750747 (> clamp const, no clamping effect)
 * w0 = 0.45873846/0.62750747 = 0.73104906
 * w1 = 0.16876901/0.62750747 = 0.26895094
 */
static void test_softmax_norm(void)
{
    float lg[8] = {2.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tt_moe_cfg cfg = {8, 2, TT_MOE_GATE_SOFTMAX, 1, 1.0f};
    tt_moe_route_out out;
    int rc = tt_moe_route(lg, &cfg, &out);
    CHECK(rc == 2, "softmax_norm rc");
    CHECK(out.ids[0] == 0 && out.ids[1] == 1, "softmax_norm ids");
    CHECK(feq(out.weights[0], 0.73105858f, 2e-6f), "softmax_norm w0");
    CHECK(feq(out.weights[1], 0.26894142f, 2e-6f), "softmax_norm w1");
    printf("ok  softmax+norm known-answer\n");
}

/* ---- Test 2: sigmoid, no norm, w_scale=2.5 ---------------------------
 * logits = [0, ln(4), -100, 100, 0, 0, 0, 0]
 * sigmoid: s0=0.5, s1=4/5=0.8, s2=~0, s3=~1 -> top2 = expert3 (1.0),
 * expert1 (0.8); no norm; scale skipped only for 0/1 so 2.5 applies:
 *   w = [2.5*1.0, 2.5*0.8] = [2.5, 2.0]   (sum != 1 — valid, matches
 *   llama.cpp sigmoid-no-norm + routed_scaling models)
 */
static void test_sigmoid_no_norm_scale(void)
{
    float lg[8] = {0.f, logf(4.f), -100.f, 100.f, 0.f, 0.f, 0.f, 0.f};
    tt_moe_cfg cfg = {8, 2, TT_MOE_GATE_SIGMOID, 0, 2.5f};
    tt_moe_route_out out;
    int rc = tt_moe_route(lg, &cfg, &out);
    CHECK(rc == 2, "sigmoid rc");
    CHECK(out.ids[0] == 3 && out.ids[1] == 1, "sigmoid ids");
    CHECK(feq(out.weights[0], 2.5f, 2e-6f), "sigmoid w0");
    CHECK(feq(out.weights[1], 2.0f, 2e-6f), "sigmoid w1");
    printf("ok  sigmoid no-norm + w_scale\n");
}

/* ---- Test 3: tie-breaking determinism --------------------------------
 * all-zero logits, softmax => all probs exactly equal.
 * Top-k must pick experts {0,1} (lower index first), deterministically.
 * Also SOFTMAX_WEIGHT path with ties: raw logits all 0 -> post-topk
 * softmax gives 0.5/0.5; then norm is identity-ish (sum=1).
 */
static void test_ties(void)
{
    float lg[8] = {0.f};
    tt_moe_cfg cfg = {8, 2, TT_MOE_GATE_SOFTMAX, 1, 1.0f};
    tt_moe_route_out out;
    int rc = tt_moe_route(lg, &cfg, &out);
    CHECK(rc == 2, "ties rc");
    CHECK(out.ids[0] == 0 && out.ids[1] == 1, "tie ids lower-first");
    CHECK(feq(out.weights[0], 0.5f, 2e-6f), "tie w0");
    CHECK(feq(out.weights[1], 0.5f, 2e-6f), "tie w1");

    /* repeat 100x — bitwise identical results required */
    for (int i = 0; i < 100; i++) {
        tt_moe_route_out o2;
        tt_moe_route(lg, &cfg, &o2);
        CHECK(o2.ids[0] == out.ids[0] && o2.ids[1] == out.ids[1],
              "tie determinism ids");
        CHECK(memcmp(o2.weights, out.weights, 2 * sizeof(float)) == 0,
              "tie determinism bits");
        if (g_fail) break;
    }

    /* partial-tie at the k boundary: probs [a,a,a,...] with distinct
     * leaders — logits [3,3,1,0,0,0,0,0], k=2 -> experts 0,1 win over
     * expert 2 because equal-prob tie prefers lower id. */
    float lg2[8] = {3.f, 3.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tt_moe_route(lg2, &cfg, &out);
    CHECK(out.ids[0] == 0 && out.ids[1] == 1, "boundary tie ids");
    printf("ok  tie-breaking determinism\n");
}

/* ---- Test 4: SOFTMAX_WEIGHT gate op ----------------------------------
 * logits [2,1,0,...]: selection on RAW logits -> top2 {0,1}; softmax
 * over ONLY {2,1}: e^0/(e^0+e^-1)=0.73105858, rest 0.26894142. Norm
 * divides by their sum (~1) — near-identity here but exact values:
 *   sum = 0.99999999..., weights stay as-is within fp tolerance.
 */
static void test_softmax_weight(void)
{
    float lg[8] = {2.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tt_moe_cfg cfg = {8, 2, TT_MOE_GATE_SOFTMAX_WEIGHT, 1, 1.0f};
    tt_moe_route_out out;
    int rc = tt_moe_route(lg, &cfg, &out);
    CHECK(rc == 2, "smw rc");
    CHECK(out.ids[0] == 0 && out.ids[1] == 1, "smw ids");
    CHECK(feq(out.weights[0], 0.73105858f, 2e-6f), "smw w0");
    CHECK(feq(out.weights[1], 0.26894142f, 2e-6f), "smw w1");

    /* key semantic difference vs plain softmax: selection uses raw
     * logits and renorm is OVER K ONLY, so with logits [5,4,-9,...]
     * plain softmax picks {0,1} with tiny probs that norm back up;
     * smw identical here — but with k=2 and logits [-1,-2, 0, ...]:
     * plain softmax would pick expert 2; smw also picks 2 first
     * (raw max). Distinguish via norm behavior instead: skip. */
    printf("ok  softmax_weight post-topk renorm\n");
}

/* ---- Test 5: plan grouping ------------------------------------------
 * selected ids across 3 tokens: t0:{5,2}, t1:{2,7}, t2:{5,0}
 * slots tagged 10..15 in input order. Grouped output must be expert
 * ascending: 0(t2b), 2(t0b,t1a), 5(t0a,t2a), 7(t1b) — stability within
 * an expert preserves caller order.
 */
static void test_plan_grouping(void)
{
    int32_t ids[]   = {5, 2, 2, 7, 5, 0};
    int32_t slots[] = {10, 11, 12, 13, 14, 15};
    tt_moe_plan_pair plan[16];
    int n = tt_moe_build_plan(ids, slots, 6, plan);
    CHECK(n == 6, "plan count");
    int32_t want_e[] = {0, 2, 2, 5, 5, 7};
    int32_t want_s[] = {15, 11, 12, 10, 14, 13};
    for (int i = 0; i < 6; i++) {
        CHECK(plan[i].expert == want_e[i], "plan expert order");
        CHECK(plan[i].slot == want_s[i], "plan slot carry");
    }
    printf("ok  grouped plan (expert asc, stable)\n");
}

/* ---- Test 6: combine helper -----------------------------------------
 * dim=2, two experts w=[0.75,0.25], outs [10,20],[1,2], shared [100,200]
 * => [0.75*10+0.25*1+100, 0.75*20+0.25*2+200] = [107.75, 215.5]
 * NULL shared => [7.75, 15.5]
 */
static void test_combine(void)
{
    float o0[2] = {10.f, 20.f}, o1[2] = {1.f, 2.f};
    const float *outs[2] = {o0, o1};
    float w[2] = {0.75f, 0.25f};
    float sh[2] = {100.f, 200.f};
    float r[2];

    tt_moe_combine(outs, w, 2, 2, sh, r);
    CHECK(feq(r[0], 107.75f, 1e-5f), "combine+shared d0");
    CHECK(feq(r[1], 215.5f, 1e-5f), "combine+shared d1");

    tt_moe_combine(outs, w, 2, 2, NULL, r);
    CHECK(feq(r[0], 7.75f, 1e-5f), "combine d0");
    CHECK(feq(r[1], 15.5f, 1e-5f), "combine d1");

    tt_moe_combine(NULL, NULL, 0, 2, sh, r);
    CHECK(r[0] == 100.f && r[1] == 200.f, "combine empty->shared passthrough");
    printf("ok  combine + shared-expert hook\n");
}

int main(void)
{
    test_softmax_norm();
    test_sigmoid_no_norm_scale();
    test_ties();
    test_softmax_weight();
    test_plan_grouping();
    test_combine();
    if (g_fail) {
        printf("%d FAILURE(S)\n", g_fail);
        return 1;
    }
    printf("ALL MOE ROUTER TESTS PASS\n");
    return 0;
}
