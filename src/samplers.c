/* samplers.c -- production token-sampling pipeline. See samplers.h.
 * Self-contained C99; only libc (<stdlib.h>, <math.h>, <string.h>). */
#include "samplers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- RNG */
/* xorshift64* (Vigna). Deterministic, caller-owned state. */
static uint64_t tt_rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

/* uniform in [0,1): 53 random mantissa bits */
static float tt_rng_float(uint64_t *s) {
    return (float)((tt_rng_next(s) >> 11) * (1.0 / 9007199254740992.0));
}

/* ------------------------------------------------------------ defaults */
void tt_sampler_chain_init(tt_sampler_chain *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->temp  = 1.0f;
    cfg->top_p = 1.0f;
}

int tt_sampler_workbuf_size(int n) { return 5 * n; }

typedef struct { float logit; int idx; } tt_pair_t;

/* descending logit, ties by ascending token id -> deterministic order
 * regardless of qsort implementation */
static int cmp_pair_desc(const void *a, const void *b) {
    const tt_pair_t *pa = (const tt_pair_t *)a;
    const tt_pair_t *pb = (const tt_pair_t *)b;
    if (pa->logit > pb->logit) return -1;
    if (pa->logit < pb->logit) return 1;
    return pa->idx - pb->idx;
}

static int argmax_f(const float *w, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (w[i] > w[best]) best = i;
    return best;
}

/* ------------------------------------------------------- penalty stage */
/* Sign-aware repetition penalty (llama.cpp convention: v>0 ? v/p : v*|p|)
 * plus optional frequency/presence penalties, over the trailing window of
 * cfg->history. Counts array must hold n ints. */
static void apply_penalties(float *w, int n, const tt_sampler_chain *cfg,
                            int *counts) {
    memset(counts, 0, (size_t)n * sizeof(int));
    if (!cfg->history || cfg->n_history <= 0) return;

    int win_rep = cfg->use_rep_penalty ? cfg->penalty_last_n : 0;
    int win_frq = cfg->use_freq_presence ? cfg->freq_last_n : 0;
    int win = win_rep > win_frq ? win_rep : win_frq;
    if (win <= 0) return;
    if (win > cfg->n_history) win = cfg->n_history;

    const int32_t *h = cfg->history + (cfg->n_history - win); /* trailing */
    for (int i = 0; i < win; i++) {
        int32_t t = h[i];
        if (t >= 0 && t < n) counts[t]++;
    }

    for (int t = 0; t < n; t++) {
        int c = counts[t];
        if (!c) continue;
        if (cfg->use_rep_penalty && cfg->repeat_penalty != 1.0f)
            w[t] = w[t] > 0.0f ? w[t] / cfg->repeat_penalty
                               : w[t] * cfg->repeat_penalty;
        if (cfg->use_freq_presence) {
            w[t] -= cfg->freq_penalty * (float)c;
            w[t] -= cfg->presence_penalty; /* presence fires once per seen tok */
        }
    }
}

/* --------------------------------------------------------- chain core */
/* Runs penalties -> temperature -> top-K -> min-p -> top-p -> softmax.
 * Never draws RNG.
 *   workbuf layout: [0,n) logits copy | [n,2n) probs | [2n,4n) sort pairs
 *                   | [4n,5n) compact survivor-id list
 * On success ord/probs hold m surviving candidates sorted by prob desc
 * (ties: ascending token id), probs sum to 1. Returns 0, or -1 on bad n. */
static int run_chain(const float *logits, int n, const tt_sampler_chain *cfg,
                     float *workbuf, int **ord_out, float **probs_out, int *m) {
    if (n <= 0) return -1;
    float   *w     = workbuf;
    float   *probs = workbuf + n;
    tt_pair_t *pr  = (tt_pair_t *)(workbuf + 2 * n);
    int     *ord   = (int *)(workbuf + 4 * n);

    memcpy(w, logits, (size_t)n * sizeof(float));
    apply_penalties(w, n, cfg, (int *)(workbuf + 2 * n)); /* counts alias pr */

    /* greedy short-circuit (explicit flag or temp <= 0): penalties applied,
     * RNG never touched */
    if (cfg->greedy || !(cfg->temp > 0.0f)) {
        ord[0] = argmax_f(w, n);
        probs[0] = 1.0f;
        *ord_out = ord;
        *probs_out = probs;
        *m = 1;
        return 0;
    }

    /* 3. temperature */
    if (cfg->temp != 1.0f)
        for (int i = 0; i < n; i++) w[i] /= cfg->temp;

    /* order all candidates once; every later stage is a truncation */
    for (int i = 0; i < n; i++) { pr[i].logit = w[i]; pr[i].idx = i; }
    qsort(pr, (size_t)n, sizeof(tt_pair_t), cmp_pair_desc);

    int m_local = n;

    /* 4. top-K (k = 0 off) */
    if (cfg->top_k > 0 && cfg->top_k < m_local) m_local = cfg->top_k;

    /* 5. min-p: keep p_i >= min_p * p_top <=> logit >= max + ln(min_p)
     * (monotonic in logit space since temperature already applied) */
    if (cfg->min_p > 0.0f && cfg->min_p < 1.0f && m_local > 1) {
        float thr = pr[0].logit + logf(cfg->min_p);
        int keep = 1;
        while (keep < m_local && pr[keep].logit >= thr) keep++;
        m_local = keep;
    }

    /* 6. top-p / nucleus (p >= 1.0 off): smallest ordered prefix whose
     * softmax mass reaches p */
    if (cfg->top_p < 1.0f && m_local > 1) {
        float fmax = pr[0].logit, sum = 0.0f;
        for (int i = 0; i < m_local; i++) sum += expf(pr[i].logit - fmax);
        float cum = 0.0f;
        int cut = m_local;
        for (int i = 0; i < m_local; i++) {
            cum += expf(pr[i].logit - fmax) / sum;
            if (cum >= cfg->top_p) { cut = i + 1; break; }
        }
        m_local = cut;
    }
    if (m_local < 1) m_local = 1;

    /* 7. softmax over survivors, renormalized to exactly 1 */
    float fmax = pr[0].logit, sum = 0.0f;
    for (int i = 0; i < m_local; i++) {
        probs[i] = expf(pr[i].logit - fmax);
        sum += probs[i];
    }
    for (int i = 0; i < m_local; i++) probs[i] /= sum;
    for (int i = 0; i < m_local; i++) ord[i] = pr[i].idx;

    *ord_out = ord;
    *probs_out = probs;
    *m = m_local;
    return 0;
}

/* ------------------------------------------------------------- public */
int tt_sample(const float *logits, int n, const tt_sampler_chain *cfg,
              uint64_t *rng_state, float *workbuf) {
    int *ord = NULL; float *probs = NULL; int m = 0;
    if (run_chain(logits, n, cfg, workbuf, &ord, &probs, &m) != 0) return -1;
    if (m == 1) return ord[0];           /* greedy/top-k=1: no RNG draw */

    float r = tt_rng_float(rng_state);
    float cum = 0.0f;
    for (int i = 0; i < m - 1; i++) {
        cum += probs[i];
        if (r < cum) return ord[i];
    }
    return ord[m - 1];                   /* numerical-safety tail */
}

int tt_sample_candidates(const float *logits, int n,
                         const tt_sampler_chain *cfg,
                         int32_t *out_tokens, float *out_probs, int out_cap,
                         float *workbuf) {
    int *ord = NULL; float *probs = NULL; int m = 0;
    if (run_chain(logits, n, cfg, workbuf, &ord, &probs, &m) != 0) return -1;
    int out = m < out_cap ? m : out_cap;
    for (int i = 0; i < out; i++) {
        out_tokens[i] = (int32_t)ord[i];
        out_probs[i]  = probs[i];
    }
    return m;                            /* full survivor count */
}

#ifdef SAMPLERS_MAIN
/* ---------------------------------------------------------- test CLI --
 * Line protocol on stdin, one command per line, results on stdout:
 *   run  <cfg> <seed> <csv logits>   -> single token id
 *   seq  <cfg> <seed> <ndraws> <csv> -> ndraws space-separated tokens
 *   cand <cfg> <csv logits>          -> "tok:prob" list sorted desc
 * cfg grammar (comma-sep keys, empty or "greedy" => pure greedy):
 *   T=<temp> K=<top_k> P=<top_p> M=<min_p>
 *   rp=<repeat_pen> rln=<window> fp=<freq_pen> pp=<presence> fln=<window>
 *   hist=<semicolon-sep recent tokens, e.g. hist=0;0;1> */
typedef struct { char key[8]; char val[256]; } kv_t;

static int parse_kv(const char *s, kv_t *kvs, int cap) {
    int nk = 0;
    while (*s && nk < cap) {
        char *eq = strchr(s, '=');
        const char *cm = strchr(s, ',');
        if (!eq || (cm && eq > cm)) { s = cm ? cm + 1 : s + strlen(s); continue; }
        size_t kl = (size_t)(eq - s);
        if (kl >= sizeof(kvs[nk].key)) kl = sizeof(kvs[nk].key) - 1;
        memcpy(kvs[nk].key, s, kl); kvs[nk].key[kl] = 0;
        const char *vend = cm ? cm : s + strlen(s);
        size_t vl = (size_t)(vend - eq - 1);
        if (vl >= sizeof(kvs[nk].val)) vl = sizeof(kvs[nk].val) - 1;
        memcpy(kvs[nk].val, eq + 1, vl); kvs[nk].val[vl] = 0;
        nk++;
        s = cm ? cm + 1 : vend;
    }
    return nk;
}

static int parse_csv_ints(const char *s, int32_t *out, int cap) {
    int c = 0; char *end;
    while (*s && c < cap) {
        long v = strtol(s, &end, 10);
        if (end == s) break;
        out[c++] = (int32_t)v;
        s = (*end == ',') ? end + 1 : end;
    }
    return c;
}

static int parse_logits(const char *s, float *out, int cap) {
    int c = 0; char *end;
    while (*s && c < cap) {
        double v = strtod(s, &end);
        if (end == s) break;
        out[c++] = (float)v;
        s = (*end == ',') ? end + 1 : end;
    }
    return c;
}

#define MAXN 65536

static void build_cfg(const char *cfgstr, tt_sampler_chain *cfg,
                      int32_t *hist, int *nhist) {
    tt_sampler_chain_init(cfg);
    *nhist = 0;
    if (!cfgstr || !cfgstr[0] || strcmp(cfgstr, "greedy") == 0) {
        cfg->greedy = 1;
        return;
    }
    kv_t kvs[32];
    int nk = parse_kv(cfgstr, kvs, 32);
    for (int i = 0; i < nk; i++) {
        const char *k = kvs[i].key, *v = kvs[i].val;
        if      (!strcmp(k, "T"))   cfg->temp = strtof(v, NULL);
        else if (!strcmp(k, "K"))   cfg->top_k = atoi(v);
        else if (!strcmp(k, "P"))   cfg->top_p = strtof(v, NULL);
        else if (!strcmp(k, "M"))   cfg->min_p = strtof(v, NULL);
        else if (!strcmp(k, "rp")) { cfg->repeat_penalty = strtof(v, NULL);
                                     cfg->use_rep_penalty = 1; }
        else if (!strcmp(k, "rln")) cfg->penalty_last_n = atoi(v);
        else if (!strcmp(k, "fp")) { cfg->freq_penalty = strtof(v, NULL);
                                     cfg->use_freq_presence = 1; }
        else if (!strcmp(k, "pp")) { cfg->presence_penalty = strtof(v, NULL);
                                     cfg->use_freq_presence = 1; }
        else if (!strcmp(k, "fln")) cfg->freq_last_n = atoi(v);
        else if (!strcmp(k, "hist")) {
            /* semicolon-separated so commas never collide with kv parsing */
            char tmp[4096];
            snprintf(tmp, sizeof tmp, "%s", v);
            for (char *q = tmp; *q; q++) if (*q == ';') *q = ',';
            *nhist = parse_csv_ints(tmp, hist, MAXN);
        }
    }
    if (cfg->penalty_last_n <= 0) cfg->penalty_last_n = 64;
    if (cfg->freq_last_n <= 0)    cfg->freq_last_n = 64;
    cfg->history = hist;
    cfg->n_history = *nhist;
}

int main(void) {
    static float logits[MAXN], workbuf[5 * MAXN], outp[MAXN];
    static int32_t hist[MAXN], toks[MAXN];
    char line[1 << 20];

    while (fgets(line, sizeof line, stdin)) {
        char cmd[16], cfgstr[4096], rest[512];
        if (sscanf(line, "%15s %4095s %511[^\n]", cmd, cfgstr, rest) < 2) {
            if (line[0] == '\n' || line[0] == 0) continue;
            printf("ERR parse\n"); continue;
        }
        tt_sampler_chain cfg;
        int nhist = 0;
        build_cfg(cfgstr, &cfg, hist, &nhist);

        if (!strcmp(cmd, "run") || !strcmp(cmd, "seq")) {
            /* rest: <seed> [ndraws] <csv logits>; parse manually so a seed
             * followed by '3.0,...' is never misread by %d */
            char *p = rest, *end;
            while (*p == ' ') p++;
            unsigned long long seed = strtoull(p, &end, 10);
            if (end == p) { printf("ERR args\n"); continue; }
            int ndraws = 1;
            p = end;
            if (!strcmp(cmd, "seq")) {
                while (*p == ' ') p++;
                long v = strtol(p, &end, 10);
                if (end == p) { printf("ERR args\n"); continue; }
                ndraws = (int)v;
                p = end;
            }
            int n = parse_logits(p, logits, MAXN);
            if (n <= 0) { printf("ERR logits\n"); continue; }
            uint64_t rs = (uint64_t)seed;
            for (int d = 0; d < ndraws; d++)
                printf("%d ", tt_sample(logits, n, &cfg, &rs, workbuf));
            printf("\n");
        } else if (!strcmp(cmd, "cand")) {
            int n = parse_logits(rest, logits, MAXN);
            if (n <= 0) { printf("ERR logits\n"); continue; }
            int m = tt_sample_candidates(logits, n, &cfg, toks, outp, MAXN,
                                         workbuf);
            for (int i = 0; i < m; i++) printf("%d:%.6f ", toks[i], outp[i]);
            printf("\n");
        } else {
            printf("ERR cmd\n");
        }
        fflush(stdout);
    }
    return 0;
}
#endif /* SAMPLERS_MAIN */
