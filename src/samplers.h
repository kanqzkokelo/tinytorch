/* samplers.h -- production token-sampling pipeline (C99, no external deps).
 *
 * Drop-in SUPERSET of the sampling currently living in
 * examples/chat_llm_gpu.c + kernels/qwen2_cuda.cu:
 *   TT_TEMP             -> cfg.temp                    (same semantics)
 *   TT_REPEAT_PENALTY   -> cfg.repeat_penalty          (same llama.cpp-style
 *                          sign-aware rule: v>0 ? v/p : v*|p|)
 *   TT_GREEDY=1         -> cfg.greedy = 1              (skips RNG entirely)
 * plus frequency/presence penalties, top-K, top-p (nucleus), min-p, and a
 * speculative-decode-ready sorted-candidate API.
 *
 * Build (standalone translation unit, no Makefile changes required):
 *   cc -std=c99 -O2 -fPIC -shared -o libtt_samplers.so src/samplers.c
 * Optional self-test CLI driver (used by tests/test_samplers.py):
 *   cc -std=c99 -O2 -DSAMPLERS_MAIN -o tt_sampler_cli src/samplers.c
 *
 * Pipeline order (matches llama.cpp sampler-chain conventions):
 *   1. repetition penalty      (sign-aware, last-n window over cfg history)
 *   2. frequency/presence      (optional, last-n window)
 *   3. temperature             (divide; temp <= 0 => greedy fallback)
 *   4. top-K                   (k = 0 disables)
 *   5. top-p / nucleus         (p >= 1.0 disables)
 *   6. min-p                   (p <= 0 disables; relative-to-top-prob filter)
 *   7. softmax -> multinomial draw
 *
 * Determinism: identical (logits, cfg, *rng_state) always yield identical
 * tokens. RNG is xorshift64* seeded from caller-owned uint64_t state; the
 * state is advanced in place so callers can stream draws. Greedy path never
 * touches *rng_state.
 */
#ifndef TT_SAMPLERS_H
#define TT_SAMPLERS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sampler chain configuration. Zero-initialize then call
 * tt_sampler_chain_init() for llama.cpp-like defaults (all filters off,
 * temp = 1.0, greedy = 0). */
typedef struct {
    /* chain-enable flags */
    int    greedy;            /* 1 => pure argmax after penalties; no RNG   */
    int    use_rep_penalty;   /* enable step 1                              */
    int    use_freq_presence; /* enable step 2                              */

    /* step 1: repetition penalty */
    int    penalty_last_n;    /* window over trailing history tokens        */
    float  repeat_penalty;    /* >1 suppresses repeats (chat used 1.15f)    */

    /* step 2: frequency / presence penalties */
    int    freq_last_n;       /* window (may differ from penalty_last_n)    */
    float  freq_penalty;      /* logit -= freq_penalty * count(tok)         */
    float  presence_penalty;  /* logit -= presence_penalty * (count > 0)    */

    /* steps 3-6 */
    float  temp;              /* <= 0 behaves as greedy                     */
    int    top_k;             /* 0 = off                                    */
    float  top_p;             /* >= 1.0f = off                              */
    float  min_p;             /* keep prob >= min_p * p_top; <= 0 = off     */

    /* recent-token history for steps 1-2 (caller-owned; may be NULL when
     * both penalty stages are disabled). Only the trailing
     * max(penalty_last_n, freq_last_n) entries are consulted. */
    const int32_t *history;
    int    n_history;
} tt_sampler_chain;

/* Fill cfg with defaults: greedy=0, all penalty stages off, temp=1.0f,
 * top_k=0, top_p=1.0f, min_p=0.0f, history=NULL. */
void tt_sampler_chain_init(tt_sampler_chain *cfg);

/* Workbuffer requirement for tt_sample / tt_sample_candidates, given vocab
 * size n: number of float elements the caller must provide. Currently
 * 5*n (logits copy + probs + sort pairs + compact id list); query this
 * instead of hard-coding. */
int tt_sampler_workbuf_size(int n);

/* Full pipeline. Returns sampled token id in [0, n).
 *   logits     : raw vocabulary logits (not modified)
 *   n          : vocab size (> 0)
 *   cfg        : chain config
 *   rng_state  : caller-owned xorshift64* state (advanced in place);
 *                untouched on the greedy path
 *   workbuf    : scratch, >= tt_sampler_workbuf_size(n) floats */
int tt_sample(const float *logits, int n, const tt_sampler_chain *cfg,
              uint64_t *rng_state, float *workbuf);

/* Speculative-decoding-ready variant: runs the identical pipeline but
 * instead of drawing one token, writes up to out_cap surviving candidates
 * to out_tokens/out_probs, sorted by descending probability (ties broken
 * by ascending token id) and renormalized to sum to 1. Returns candidate
 * count (may exceed out_cap, in which case only the first out_cap are
 * written). Never touches rng_state. */
int tt_sample_candidates(const float *logits, int n,
                         const tt_sampler_chain *cfg,
                         int32_t *out_tokens, float *out_probs, int out_cap,
                         float *workbuf);

#ifdef __cplusplus
}
#endif
#endif /* TT_SAMPLERS_H */

#ifdef SAMPLERS_MAIN
/* Self-test CLI driver (line protocol on stdin, results on stdout).
 * Commands:
 *   run  <greedy|temp=T,k=K,p=P,m=M,rp=R,rln=N,fp=F,pp=S,fln=N,hist=a,b,c> <seed> <csv logits>
 *   seq  <cfg...> <seed> <ndraws> <csv logits>     -- ndraws space-sep tokens
 *   cand <cfg...> <seed> <csv logits>              -- "tok:prob" sorted list
 * Config keys: T=temp K=top_k P=top_p M=min_p rp=repeat_pen rln=window
 *              fp=freq_pen pp=presence_pen fln=freq_window hist=recent toks
 * Empty cfg or "greedy" selects greedy. */
#endif
