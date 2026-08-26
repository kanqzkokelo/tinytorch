/* specdec.c -- ngram-simple drafter (M10). See specdec.h. */
#include "specdec.h"

#include <stdlib.h>
#include <string.h>

struct tt_ngram {
    uint32_t *buf;      /* ring buffer of token ids            */
    uint32_t  cap;      /* physical capacity of buf            */
    uint32_t  len;      /* logical tokens stored (<= cap)      */
    uint64_t  total;    /* absolute count fed (for wrap index) */
    uint32_t  window;   /* match window n                      */
    uint32_t  max_draft;
};

/* Token at logical position i lives in slot i % cap. Logical positions
 * are absolute; the oldest surviving token is at total - len. */
static uint32_t ng_at(const tt_ngram *g, uint64_t logical) {
    return g->buf[logical % g->cap];
}

tt_ngram *tt_ngram_create(uint32_t history_cap, uint32_t window,
                          uint32_t max_draft) {
    if (max_draft == 0) return NULL;
    if (window == 0) window = 12;
    if (history_cap == 0) history_cap = 4096;
    if (history_cap <= window + max_draft) return NULL; /* useless config */

    tt_ngram *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->buf = malloc((size_t)history_cap * sizeof(uint32_t));
    if (!g->buf) { free(g); return NULL; }
    g->cap = history_cap;
    g->window = window;
    g->max_draft = max_draft;
    return g;
}

void tt_ngram_free(tt_ngram *g) {
    if (!g) return;
    free(g->buf);
    free(g);
}

int tt_ngram_feed(tt_ngram *g, const uint32_t *toks, uint32_t count) {
    if (!g || (!toks && count > 0)) return -1;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slot = (uint32_t)(g->total % g->cap);
        g->buf[slot] = toks[i];
        g->total++;
        if (g->len < g->cap) g->len++;
    }
    return 0;
}

uint32_t tt_ngram_len(const tt_ngram *g) { return g ? g->len : 0; }

/* First logical index currently resident. */
static uint64_t ng_base(const tt_ngram *g) { return g->total - g->len; }

uint32_t tt_ngram_draft(const tt_ngram *g, uint32_t *out) {
    if (!g || !out || g->len <= g->window) return 0;

    /* needle = last `window` tokens of history:
     * logical [total-window, total). Candidate source occurrence must
     * END strictly before that, i.e. its end e satisfies
     * e <= total - window. Search from most recent backwards. */
    uint64_t base = ng_base(g);
    uint64_t needle_start = g->total - g->window;
    uint64_t cand_end_max = needle_start; /* exclusive upper bound */

    for (uint64_t e = cand_end_max; e - g->window >= base; e--) {
        uint64_t cs = e - g->window;
        uint32_t match = 1;
        for (uint32_t j = 0; j < g->window; j++) {
            if (ng_at(g, cs + j) != ng_at(g, needle_start + j)) {
                match = 0;
                break;
            }
        }
        if (match) {
            /* emit tokens at [e, min(e + max_draft, total)) -- only
             * tokens already in history can be proposed. */
            uint64_t avail = g->total - e;
            uint32_t k = (avail < (uint64_t)g->max_draft)
                       ? (uint32_t)avail : g->max_draft;
            for (uint32_t j = 0; j < k; j++)
                out[j] = ng_at(g, e + j);
            return k;
        }
    }
    return 0;
}
