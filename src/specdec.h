/* specdec.h -- M10 speculative decoding: ngram-simple drafter.
 *
 * Build (standalone, no deps beyond libc):
 *   gcc -std=c99 -O2 -Wall -Wextra -c src/specdec.c          # object
 *   gcc -std=c99 -O2 -Wall -Wextra -Isrc -o build/test_specdec \
 *       src/specdec.c tests/test_specdec.c                   # future unit bin
 *
 * NOTE: no Makefile change required for the shared library target:
 * lib target rule already compiles every C source under src/, so
 * specdec.c is picked up automatically. Deliberately NOT added to any
 * nvcc example/chat target until engine integration lands.
 *
 * Pure C99, no GPU dependency: drafter is fully unit-testable on CPU.
 */

#ifndef TT_SPECDEC_H
#define TT_SPECDEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque ngram drafter. Ring-buffer history of token ids (prompt +
 * generated appended via tt_ngram_feed). Drafting = llama.cpp
 * "ngram-simple": take last `window` tokens of history, find the MOST
 * RECENT earlier occurrence of that window, propose up to `max_draft`
 * tokens that followed it.
 */
typedef struct tt_ngram tt_ngram;

/* window == 0 selects default (12, per roadmap M10 item 2).
 * history_cap == 0 selects default (4096 tokens).
 * Returns NULL on allocation failure or max_draft == 0. */
tt_ngram *tt_ngram_create(uint32_t history_cap, uint32_t window,
                          uint32_t max_draft);
void      tt_ngram_free(tt_ngram *g);

/* Append tokens to history. Oldest tokens fall off once history_cap is
 * reached. Returns 0 on success, -1 on bad args. */
int       tt_ngram_feed(tt_ngram *g, const uint32_t *toks, uint32_t count);

/* Current number of tokens in history. */
uint32_t  tt_ngram_len(const tt_ngram *g);

/* Draft: write up to max_draft proposed token ids into out[] (caller
 * owns storage of >= tt_ngram max_draft entries). Returns count written,
 * 0 when the window is not found or history too short (< window + 1). */
uint32_t  tt_ngram_draft(const tt_ngram *g, uint32_t *out);

/* ------------------------------------------------------------------ */
/* FUTURE ENGINE INTERFACE (sketch only -- implement later, when       */
/* qwen2_cuda.cu frees up). Reference: roadmap M10 build order 1.      */
/*                                                                     */
/*  Verify a draft in ONE batched forward. Greedy accept loop:         */
/*                                                                     */
/*    drafts[k]      : proposed ids from tt_ngram_draft                */
/*    k              : number of drafted tokens                        */
/*    out_accepted   : number of drafts matching argmax logits         */
/*    out_next       : first mismatching target id (bonus token)       */
/*    int tt_verify(qwen2_engine *eng, const uint32_t *drafts,         */
/*                  uint32_t k, uint32_t *out_accepted,                */
/*                  uint32_t *out_next);                               */
/*                                                                      */
/* Requirements on the engine side (NOT implemented yet):               */
/*  1. Batched forward: run k+1 positions (last committed token + k     */
/*     drafts) through the PREFILL path in one launch sequence --       */
/*     decode path is 1-token-per-step and cannot do this. Per-slot     */
/*     logits extraction needed for all k+1 rows.                       */
/*  2. KV rollback to first mismatch: equivalent of llama.cpp           */
/*     llama_memory_seq_rm -- reset d_pos to (pos_of_first_mismatch)    */
/*     and memset/zeros the tail K/V slots for every layer so stale     */
/*     rejected drafts cannot leak into later attention windows.        */
/*     Must handle ALL layers' caches, not just layer 0.                */
/*  3. Cost contract: verify_step_cost(k+1 tokens) must stay near-flat  */
/*     vs 1 token (memory-bound GEMV) or speculation loses to the       */
/*     linear-cost fallback. Graph-capture per shape k+1 (M10 item 4)   */
/*     is what makes this hold; without it launch overhead eats gain.   */
/*  4. Greedy-only initially; rejection sampling for temp>0 is M10      */
/*     item 5 (later).                                                  */
/*                                                                     */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
}
#endif

#endif /* TT_SPECDEC_H */
