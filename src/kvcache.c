/* kvcache.c -- implementation. See kvcache.h for contract.
 *
 * Pure C99, no GPU dependency: everything here is sizes/offsets/state,
 * so the whole layer is unit-testable on CPU (tests/test_kvcache.c).
 */

#include "kvcache.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static void *xcalloc(size_t n, size_t sz) {
    return calloc(n ? n : 1, sz);
}

uint64_t tt_fnv1a64(const uint8_t *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)data[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static int cfg_sane(const tt_kvcache_cfg *c) {
    if (!c || c->n_layers <= 0 || c->max_ctx <= 0 || c->head_dim <= 0 ||
        c->dtype_size == 0)
        return 0;
    if (c->kv_width) {
        for (int l = 0; l < c->n_layers; l++)
            if (c->kv_width[l] <= 0) return 0;
        /* widths must divide cleanly into slab slots of head_dim rows */
        long maxw = 0;
        for (int l = 0; l < c->n_layers; l++)
            if ((long)c->kv_width[l] > maxw) maxw = c->kv_width[l];
        if (maxw % c->head_dim != 0) return 0;
    }
    if (c->src_layer) {
        for (int l = 0; l < c->n_layers; l++) {
            const int s = c->src_layer[l];
            if (s >= c->n_layers) return 0;
            if (s == l && s >= 0) return 0;   /* self-share nonsense */
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* create / free                                                       */
/* ------------------------------------------------------------------ */

tt_kvcache *tt_kvcache_create(const tt_kvcache_cfg *cfg) {
    if (!cfg_sane(cfg)) return NULL;
    tt_kvcache *kc = calloc(1, sizeof(*kc));
    if (!kc) return NULL;
    const int L = cfg->n_layers;
    kc->kv_width  = xcalloc((size_t)L, sizeof(int));
    kc->src_layer = xcalloc((size_t)L, sizeof(int));
    kc->valid_len = xcalloc((size_t)L, sizeof(uint32_t));
    if (!kc->kv_width || !kc->src_layer || !kc->valid_len) goto fail;

    for (int l = 0; l < L; l++) {
        /* default width: meta n_kv(=1 row per head_dim chunk)... callers
         * with hetero layers pass kv_width explicitly; NULL means every
         * layer is one head_dim-wide row group scaled by head_dim. */
        kc->kv_width[l]  = cfg->kv_width ? cfg->kv_width[l] : cfg->head_dim;
        kc->src_layer[l] = cfg->src_layer ? cfg->src_layer[l] : -1;
    }
    kc->cfg = *cfg;
    /* point cfg at the PRIVATE copies so downstream users of kc->cfg
     * (e.g. tt_kv_layout_build after deserialize) see real arrays */
    kc->cfg.kv_width  = kc->kv_width;
    kc->cfg.src_layer = kc->src_layer;

    /* engine stride formula (kernels/qwen2_cuda.cu alloc site):
     * cache_per = max_kv * max_ctx * hd_meta, where max_kv counts META
     * heads => element-wise it equals max(kv_width_l). */
    long maxw = 0;
    for (int l = 0; l < L; l++)
        if ((long)kc->kv_width[l] > maxw) maxw = kc->kv_width[l];
    kc->cache_per_elems = maxw * (long)cfg->max_ctx;
    return kc;
fail:
    tt_kvcache_free(kc);
    return NULL;
}

void tt_kvcache_free(tt_kvcache *kc) {
    if (!kc) return;
    free(kc->kv_width);
    free(kc->src_layer);
    free(kc->valid_len);
    free(kc);
}

/* ------------------------------------------------------------------ */
/* append + rollback                                                   */
/* ------------------------------------------------------------------ */

int tt_kv_append(tt_kvcache *kc, uint32_t n) {
    if (!kc || kc->cfg.max_ctx <= 0) return -1;
    const uint32_t cap = (uint32_t)kc->cfg.max_ctx;
    for (int l = 0; l < kc->cfg.n_layers; l++) {
        if (kc->src_layer[l] >= 0) continue;   /* shared layer: no own KV */
        uint64_t v = (uint64_t)kc->valid_len[l] + n;
        if (v > cap) v = cap;
        kc->valid_len[l] = (uint32_t)v;
    }
    /* shared layers mirror their source so SWA/serialize stay coherent */
    for (int l = 0; l < kc->cfg.n_layers; l++) {
        const int s = kc->src_layer[l];
        if (s >= 0 && s < kc->cfg.n_layers)
            kc->valid_len[l] = kc->valid_len[s];
    }
    return 0;
}

uint32_t tt_kv_valid(const tt_kvcache *kc, int layer) {
    if (!kc || layer < 0 || layer >= kc->cfg.n_layers) return 0;
    return kc->valid_len[layer];
}

tt_kv_mark *tt_kv_mark_create(const tt_kvcache *kc) {
    if (!kc) return NULL;
    tt_kv_mark *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->len = xcalloc((size_t)kc->cfg.n_layers, sizeof(uint32_t));
    if (!m->len) { free(m); return NULL; }
    m->n = (uint32_t)kc->cfg.n_layers;
    memcpy(m->len, kc->valid_len, m->n * sizeof(uint32_t));
    return m;
}

void tt_kv_mark_free(tt_kv_mark *m) {
    if (!m) return;
    free(m->len);
    free(m);
}

int tt_kv_restore(const tt_kv_mark *m, tt_kvcache *kc) {
    if (!m || !kc || m->n != (uint32_t)kc->cfg.n_layers) return -1;
    memcpy(kc->valid_len, m->len, m->n * sizeof(uint32_t));
    return 0;
}

int tt_kv_truncate(tt_kvcache *kc, uint32_t new_len, int emit_zero,
                   tt_kv_zero_range *out, uint32_t *n_out) {
    if (!kc) return -1;
    if (emit_zero && (!out || !n_out)) return -1;
    for (int l = 0; l < kc->cfg.n_layers; l++) {
        const uint32_t old = kc->valid_len[l];
        const uint32_t nu  = new_len < old ? new_len : old;
        kc->valid_len[l] = nu;
        if (!emit_zero) continue;
        const long w = kc->kv_width[l];
        /* K range then V range; identical element spans, engine memsets
         * both slabs. Offsets relative to each slab start. */
        out[2 * l].offset_elems = (long)nu * w;
        out[2 * l].n_elems      = (long)(old - nu) * w;
        out[2 * l + 1]          = out[2 * l];
    }
    if (emit_zero) *n_out = 2u * (uint32_t)kc->cfg.n_layers;
    return 0;
}

/* ------------------------------------------------------------------ */
/* SWA accounting + compaction plans                                   */
/* ------------------------------------------------------------------ */

uint32_t tt_kv_swa_dead_from(uint32_t valid_len, uint32_t window) {
    if (window == 0 || valid_len <= window) return 0;
    return valid_len - window;
}

void tt_kv_compact_plan_free(tt_kv_compact_plan *p) {
    if (!p) return;
    free(p->moves);
    p->moves = NULL;
    p->n_moves = 0;
}

int tt_kv_compact_plan_from_keep(const int *keep, uint32_t valid_len,
                                 tt_kv_compact_plan *plan) {
    if (!keep || !plan || valid_len == 0) {
        if (plan && valid_len == 0) {   /* empty-but-valid plan */
            plan->moves = NULL; plan->n_moves = 0;
            plan->compacted_len = 0; plan->evicted = 0;
            return valid_len == 0 ? 0 : -1;
        }
        return -1;
    }
    memset(plan, 0, sizeof(*plan));

    /* generic pack-to-front: walk slots; whenever a live slot's target
     * position differs from its source, emit a contiguous run move. */
    uint32_t dst = 0;
    uint32_t live = 0;
    long run_src = -1, run_dst = -1, run_len = 0;
    tt_kv_move *moves = NULL;
    uint32_t nmoves = 0, cap = 0;

    for (uint32_t t = 0; t < valid_len; t++) {
        if (!keep[t]) continue;
        live++;
        if ((long)t == (long)dst) {           /* already in place */
            if (run_src >= 0) {               /* flush pending run */
                if (nmoves == cap) {
                    cap = cap ? cap * 2 : 8;
                    moves = realloc(moves, cap * sizeof(*moves));
                    if (!moves) return -1;
                }
                moves[nmoves].src_slot = run_src;
                moves[nmoves].dst_slot = run_dst;
                moves[nmoves].count    = run_len;
                nmoves++;
                run_src = run_dst = -1; run_len = 0;
            }
        } else if (run_src >= 0 && run_src + run_len == (long)t &&
                   run_dst + run_len == (long)dst) {
            run_len++;                        /* extend current run */
        } else {
            if (run_src >= 0) {
                if (nmoves == cap) {
                    cap = cap ? cap * 2 : 8;
                    moves = realloc(moves, cap * sizeof(*moves));
                    if (!moves) return -1;
                }
                moves[nmoves].src_slot = run_src;
                moves[nmoves].dst_slot = run_dst;
                moves[nmoves].count    = run_len;
                nmoves++;
            }
            run_src = (long)t; run_dst = (long)dst; run_len = 1;
        }
        dst++;
    }
    if (run_src >= 0) {
        if (nmoves == cap) {
            cap = cap ? cap * 2 : 8;
            moves = realloc(moves, cap * sizeof(*moves));
            if (!moves) return -1;
        }
        moves[nmoves].src_slot = run_src;
        moves[nmoves].dst_slot = run_dst;
        moves[nmoves].count    = run_len;
        nmoves++;
    }
    plan->moves = nmoves ? moves : NULL;
    if (nmoves && !moves) return -1;
    plan->n_moves = nmoves;
    plan->compacted_len = live;
    plan->evicted = valid_len - live;
    return 0;
}

int tt_kv_compact_plan_swa(const tt_kvcache *kc, const uint32_t *windows,
                           tt_kv_compact_plan *out) {
    if (!kc || !out) return -1;
    for (int l = 0; l < kc->cfg.n_layers; l++) {
        const uint32_t v = kc->valid_len[l];
        const uint32_t w = windows ? windows[l] : 0;
        int *keep = xcalloc(v ? v : 1, sizeof(int));
        if (!keep) return -1;
        for (uint32_t t = 0; t < v; t++)
            keep[t] = (t >= tt_kv_swa_dead_from(v, w));
        int rc = tt_kv_compact_plan_from_keep(keep, v, &out[l]);
        free(keep);
        if (rc != 0) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* blob layout                                                         */
/* ------------------------------------------------------------------ */

int tt_kv_layout_build(const tt_kvcache_cfg *cfg, tt_kv_layout *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!cfg_sane(cfg)) return -1;
    out->loc = xcalloc((size_t)cfg->n_layers, sizeof(*out->loc));
    if (!out->loc) return -1;

    long maxw = 0;
    for (int l = 0; l < cfg->n_layers; l++)
        if ((long)(cfg->kv_width ? cfg->kv_width[l] : cfg->head_dim) > maxw)
            maxw = cfg->kv_width ? cfg->kv_width[l] : cfg->head_dim;
    out->cache_per_elems = maxw * (long)cfg->max_ctx;
    out->n_layers = (uint32_t)cfg->n_layers;
    out->blob_bytes = out->cache_per_elems * (long)cfg->n_layers *
                      (long)cfg->dtype_size;
    for (int l = 0; l < cfg->n_layers; l++) {
        out->loc[l].slab_offset_elems =
            (long)l * out->cache_per_elems;
        out->loc[l].slot_stride_elems =
            cfg->kv_width ? cfg->kv_width[l] : cfg->head_dim;
        out->loc[l].src_layer = cfg->src_layer ? cfg->src_layer[l] : -1;
    }
    return 0;
}

void tt_kv_layout_free(tt_kv_layout *lay) {
    if (!lay) return;
    free(lay->loc);
    lay->loc = NULL;
}

/* ------------------------------------------------------------------ */
/* serialize                                                           */
/* ------------------------------------------------------------------ */

#define HDR_BYTES 64u   /* fixed header incl. trailing checksum slot */

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put_i64(uint8_t *p, int64_t v) {
    put_u32(p, (uint32_t)(v & 0xffffffffll));
    put_u32(p + 4, (uint32_t)((uint64_t)v >> 32));
}
static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int64_t get_i64(const uint8_t *p) {
    return (int64_t)((uint64_t)get_u32(p) | ((uint64_t)get_u32(p + 4) << 32));
}

size_t tt_kv_serialize_size(const tt_kvcache *kc) {
    if (!kc) return (size_t)-1;
    /* header + per-layer {kv_width u32, src_layer i32, valid_len u32}
     * + trailing u64 fnv1a64 checksum */
    return HDR_BYTES + 12u * (size_t)kc->cfg.n_layers + 8u;
}

size_t tt_kv_serialize(const tt_kvcache *kc, uint8_t *buf, size_t cap) {
    if (!kc || !buf) return (size_t)-1;
    const size_t need = tt_kv_serialize_size(kc);
    if (need == (size_t)-1 || cap < need) return (size_t)-1;
    const int L = kc->cfg.n_layers;

    memset(buf, 0, need);
    uint8_t *p = buf;
    put_u32(p + 0,  TT_KVCACHE_MAGIC);
    put_u32(p + 4,  TT_KVCACHE_VERSION);
    put_u32(p + 8,  0);                       /* flags */
    put_u32(p + 12, (uint32_t)HDR_BYTES);
    put_u32(p + 16, (uint32_t)L);
    put_u32(p + 20, (uint32_t)kc->cfg.max_ctx);
    put_u32(p + 24, (uint32_t)kc->cfg.head_dim);
    put_u32(p + 28, (uint32_t)kc->cfg.dtype_size);
    put_i64(p + 32, kc->cache_per_elems);
    put_i64(p + 40, kc->cache_per_elems * (int64_t)L *
                        (int64_t)kc->cfg.dtype_size);
    p += HDR_BYTES;
    for (int l = 0; l < L; l++, p += 4)
        put_u32(p, (uint32_t)kc->kv_width[l]);          /* widths > 0 */
    for (int l = 0; l < L; l++, p += 4)
        put_u32(p, (uint32_t)kc->src_layer[l]);         /* two's compl */
    for (int l = 0; l < L; l++, p += 4)  put_u32(p, kc->valid_len[l]);
    const uint64_t cs = tt_fnv1a64(buf, need - 8);
    put_u32(buf + need - 8, (uint32_t)(cs & 0xffffffffu));
    put_u32(buf + need - 4, (uint32_t)(cs >> 32));
    return need;
}

int tt_kv_deserialize(const uint8_t *buf, size_t len,
                      tt_kvcache **out_kc, tt_kv_layout **out_layout) {
    if (out_kc) *out_kc = NULL;
    if (out_layout) *out_layout = NULL;
    if (!buf || !out_kc) return -1;
    if (len < HDR_BYTES + 12 + 8) return -4;   /* min: 1 layer + csum */
    if (get_u32(buf + 0) != TT_KVCACHE_MAGIC ||
        get_u32(buf + 4) != TT_KVCACHE_VERSION)
        return -2;
    const uint32_t L       = get_u32(buf + 16);
    const uint32_t max_ctx = get_u32(buf + 20);
    const size_t need = HDR_BYTES + 12u * (size_t)L + 8u;
    if (len < need) return -4;
    const uint64_t cs_stored =
        (uint64_t)get_u32(buf + need - 8) |
        ((uint64_t)get_u32(buf + need - 4) << 32);
    if (cs_stored != tt_fnv1a64(buf, need - 8)) return -3;

    tt_kvcache_cfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_layers   = (int)L;
    cfg.max_ctx    = (int)max_ctx;
    cfg.head_dim   = (int)get_u32(buf + 24);
    cfg.dtype_size = (size_t)get_u32(buf + 28);

    int *kw = xcalloc(L, sizeof(int)), *sl = xcalloc(L, sizeof(int));
    uint32_t *vl = xcalloc(L, sizeof(uint32_t));
    if (!kw || !sl || !vl) { free(kw); free(sl); free(vl); return -5; }
    const uint8_t *p = buf + HDR_BYTES;
    for (uint32_t l = 0; l < L; l++, p += 4) kw[l] = (int)get_u32(p);
    for (uint32_t l = 0; l < L; l++, p += 4) sl[l] = (int)get_u32(p);
    for (uint32_t l = 0; l < L; l++, p += 4) vl[l] = get_u32(p);

    cfg.kv_width  = kw;
    cfg.src_layer = sl;
    tt_kvcache *kc = tt_kvcache_create(&cfg);
    cfg.kv_width = cfg.src_layer = NULL;
    free(kw); free(sl);
    if (!kc) { free(vl); return -5; }
    memcpy(kc->valid_len, vl, L * sizeof(uint32_t));
    free(vl);

    if (out_layout) {
        tt_kv_layout *lay = calloc(1, sizeof(*lay));
        if (!lay) { tt_kvcache_free(kc); return -5; }
        if (tt_kv_layout_build(&kc->cfg, lay) != 0) {
            free(lay); tt_kvcache_free(kc); return -5;
        }
        /* restore serialized stride/blob so roundtrip is bit-exact even
         * if future layout rules change */
        lay->cache_per_elems = get_i64(buf + 32);
        lay->blob_bytes      = get_i64(buf + 40);
        *out_layout = lay;
    }
    *out_kc = kc;
    return 0;
}

/* ------------------------------------------------------------------ */
/* memory accounting                                                   */
/* ------------------------------------------------------------------ */

void tt_kv_memory(const tt_kvcache_cfg *cfg, tt_kv_mem *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg_sane(cfg)) return;

    long maxw = 0, owned_exact = 0;
    int n_own = 0;
    for (int l = 0; l < cfg->n_layers; l++) {
        const long w = cfg->kv_width ? cfg->kv_width[l] : cfg->head_dim;
        if (w > maxw) maxw = w;
        const int src = cfg->src_layer ? cfg->src_layer[l] : -1;
        if (src < 0) { n_own++; owned_exact += w; }
    }
    const double dt = (double)cfg->dtype_size;
    const double tok = (double)cfg->max_ctx;

    const long b_no_share = (long)((double)cfg->n_layers * 2.0 *
                                   (double)maxw * tok * dt);
    const long b_shared   = (long)((double)n_own * 2.0 *
                                   (double)maxw * tok * dt);
    const long b_ideal    = 2L * owned_exact * (long)tok *
                            (long)cfg->dtype_size;
    out->b_no_share = b_no_share;
    out->b_shared   = b_shared;
    out->b_ideal    = b_ideal;
    const double gib = 1073741824.0;
    out->gb_no_share = (double)b_no_share / gib;
    out->gb_shared   = (double)b_shared / gib;
    out->gb_ideal    = (double)b_ideal / gib;
}
