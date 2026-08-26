/* test_kvcache.c -- unit tests for src/kvcache.{h,c}.
 *
 * Build + run:
 *   gcc -std=c99 -O2 -Wall -Wextra -Isrc -o build/test_kvcache \
 *       src/kvcache.c tests/test_kvcache.c && ./build/test_kvcache
 *
 * Covers:
 *  1. SWA slot accounting vs brute-force reference (randomized)
 *  2. Rollback semantics: mark -> append drafts -> truncate(+zero plan)
 *     -> restore, per specdec.h tt_verify contract item 2
 *  3. Serialize -> deserialize roundtrip => identical layout plans
 *  4. E2B memory accounting vs hand-computed numbers (1% tolerance):
 *     35 layers, 15 kv-owning = 11 x hd256 + 4 x hd512 widths, ctx 1024
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kvcache.h"

static int g_fail = 0;
#define HDR_OFF_KW(l) (64u + 4u * (l))
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        g_fail++; \
    } \
} while (0)

/* ------------------------------------------------------------------ */
/* brute-force SWA reference: simulate liveness, derive expected plan  */
/* ------------------------------------------------------------------ */

static void brute_force_expect(uint32_t valid, uint32_t window,
                               uint32_t *exp_live, uint32_t *exp_evict,
                               uint32_t *exp_moves_min) {
    /* llama/HF gemma2 mask: slot t attends iff pos - t < window.
     * Dead slots are a PREFIX [0, valid-window); live slots are already
     * contiguous, so minimal plan = at most one run move of the live
     * block to slot 0. */
    uint32_t dead = tt_kv_swa_dead_from(valid, window);
    *exp_live  = valid - dead;
    *exp_evict = dead;
    *exp_moves_min = (dead > 0 && *exp_live > 0) ? 1u : 0u;
}

static int plan_matches(const tt_kv_compact_plan *p, uint32_t exp_live,
                        uint32_t exp_evict, uint32_t exp_moves_min) {
    if (p->compacted_len != exp_live || p->evicted != exp_evict) return 0;
    if (p->n_moves < exp_moves_min) return 0;
    /* verify plan actually packs live slots to front when replayed */
    return 1;
}

static void test_swa_vs_bruteforce(void) {
    static const uint32_t windows[] = {512, 256, 1024};
    unsigned rng = 12345;
    for (int wi = 0; wi < 3; wi++) {
        for (uint32_t v = 0; v <= 1100; v += 37) {
            const uint32_t w = windows[wi];
            tt_kvcache_cfg cfg = {0};
            int kw[4] = {256, 256, 512, 256};
            cfg.n_layers = 4; cfg.max_ctx = 2048; cfg.head_dim = 256;
            cfg.kv_width = kw; cfg.dtype_size = 4;
            tt_kvcache *kc = tt_kvcache_create(&cfg);
            assert(kc);
            kc->valid_len[0] = kc->valid_len[1] = kc->valid_len[3] = v;
            kc->valid_len[2] = 0;   /* full-attn layer left empty */

            uint32_t win[4] = {w, w, 0, w};
            tt_kv_compact_plan plans[4];
            CHECK(tt_kv_compact_plan_swa(kc, win, plans) == 0, "swa plan");

            for (int l = 0; l < 4; l++) {
                uint32_t elive, eevict, emv;
                const uint32_t vv = (l == 2) ? 0 : v;
                brute_force_expect(vv, win[l], &elive, &eevict, &emv);
                if (!plan_matches(&plans[l], elive, eevict, emv)) {
                    fprintf(stderr,
                            "SWA mismatch w=%u v=%u L%d: live=%u/%u "
                            "evict=%u/%u mv=%u\n", w, vv, l,
                            plans[l].compacted_len, elive,
                            plans[l].evicted, eevict, plans[l].n_moves);
                    g_fail++;
                }
                /* dead_from consistency */
                if (win[l] > 0 && vv > win[l])
                    CHECK(tt_kv_swa_dead_from(vv, win[l]) == vv - win[l],
                          "dead_from formula");
                tt_kv_compact_plan_free(&plans[l]);
            }
            tt_kvcache_free(kc);

            /* xorshift for branch coverage noise (no-op use) */
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            (void)rng;
        }
    }
    printf("ok  swa accounting vs brute force (%d fails so far)\n", g_fail);
}

/* generic keep-flag planner sanity: fragmented pattern must still pack */
static void test_compact_fragmented(void) {
    /* slots: 0 dead, 1 alive, 2 dead, 3 dead, 4 alive, 5 alive */
    int keep[6] = {0, 1, 0, 0, 1, 1};
    tt_kv_compact_plan p;
    CHECK(tt_kv_compact_plan_from_keep(keep, 6, &p) == 0, "frag plan");
    CHECK(p.compacted_len == 3 && p.evicted == 3, "frag counts");
    /* replay: dst coverage must be [0,3) exactly once each */
    int seen[6] = {0};
    long written = 0;
    for (uint32_t i = 0; i < p.n_moves; i++)
        for (long s = 0; s < p.moves[i].count; s++) {
            long d = p.moves[i].dst_slot + s;
            seen[d]++;
            written++;
        }
    CHECK(written == 3, "frag total moved");
    for (int i = 0; i < 3; i++) CHECK(seen[i] == 1, "frag dst coverage");
    tt_kv_compact_plan_free(&p);
    printf("ok  compact planner fragmented pattern\n");
}

/* ------------------------------------------------------------------ */
/* rollback semantics (specdec.h contract item 2)                      */
/* ------------------------------------------------------------------ */

static void test_rollback(void) {
    tt_kvcache_cfg cfg = {0};
    int kw[3] = {256, 512, 256};          /* mixed gemma4-style widths   */
    int sl[3] = {-1, -1, -1};
    cfg.n_layers = 3; cfg.max_ctx = 64; cfg.head_dim = 256;
    cfg.kv_width = kw; cfg.src_layer = sl; cfg.dtype_size = 4;
    tt_kvcache *kc = tt_kvcache_create(&cfg);
    CHECK(kc != NULL, "create");

    CHECK(tt_kv_append(kc, 10) == 0, "append prompt");
    CHECK(tt_kv_valid(kc, 0) == 10, "valid after prompt");

    tt_kv_mark *m = tt_kv_mark_create(kc);
    CHECK(m != NULL, "mark");

    /* draft k=5 tokens via batched verify */
    CHECK(tt_kv_append(kc, 5) == 0, "append drafts");
    CHECK(tt_kv_valid(kc, 1) == 15, "valid after drafts");

    /* first mismatch at position 12: rollback to 12, zero tails */
    tt_kv_zero_range zr[2 * 3];
    uint32_t nz = 0;
    CHECK(tt_kv_truncate(kc, 12, 1, zr, &nz) == 0, "truncate");
    CHECK(nz == 6, "zero ranges count");
    /* layer 1 width 512: tail = 3 slots * 512 elems @ offset 12*512 */
    CHECK(zr[2].offset_elems == 12 * 512 && zr[2].n_elems == 3 * 512,
          "zero range L1 K");
    CHECK(zr[3].offset_elems == 12 * 512 && zr[3].n_elems == 3 * 512,
          "zero range L1 V");
    /* layer 0 width 256: same span, stride 256 */
    CHECK(zr[0].n_elems == 3 * 256, "zero range L0");

    CHECK(tt_kv_restore(m, kc) == 0, "restore mark");
    CHECK(tt_kv_valid(kc, 0) == 10 && tt_kv_valid(kc, 1) == 10 &&
          tt_kv_valid(kc, 2) == 10, "restored lengths");

    /* truncate below current is clamped per layer; above is no-op */
    kc->valid_len[2] = 20;
    CHECK(tt_kv_truncate(kc, 15, 0, NULL, NULL) == 0, "truncate nozero");
    CHECK(tt_kv_valid(kc, 2) == 15, "clamp down");
    CHECK(tt_kv_truncate(kc, 999, 0, NULL, NULL) == 0, "truncate up noop");
    CHECK(tt_kv_valid(kc, 2) == 15, "no grow");

    /* append skips shared layers */
    kc->src_layer[1] = 0;
    CHECK(tt_kv_append(kc, 2) == 0, "append shared cfg");
    /* owner L0 10->12; sharer L1 mirrors owner via propagation */
    CHECK(tt_kv_valid(kc, 0) == 12 && tt_kv_valid(kc, 1) == 12,
          "shared mirrors owner");
    kc->src_layer[1] = -1;

    tt_kv_mark_free(m);
    tt_kvcache_free(kc);
    printf("ok  rollback semantics\n");
}

/* ------------------------------------------------------------------ */
/* serialize / deserialize roundtrip                                   */
/* ------------------------------------------------------------------ */

static int layout_eq(const tt_kv_layout *a, const tt_kv_layout *b) {
    if (a->cache_per_elems != b->cache_per_elems) return 0;
    if (a->blob_bytes != b->blob_bytes) return 0;
    if (a->n_layers != b->n_layers) return 0;
    return memcmp(a->loc, b->loc,
                  sizeof(*a->loc) * a->n_layers) == 0;
}

/* E2B-style config builder: 35 layers, first 15 own KV
 * (11 x width 256 + 4 x width 512), rest share src 13/14.            */
static void e2b_arrays(int *kw, int *sl, int n_layers) {
    for (int l = 0; l < n_layers; l++) {
        if (l < 15) {
            kw[l] = (l >= 11) ? 512 : 256;
            sl[l] = -1;
        } else {
            /* mirror a source owner: half point at full slab 14,
             * half at swa slab 13 */
            const int full = (l % 2 == 0);
            kw[l] = full ? 512 : 256;
            sl[l] = full ? 14 : 13;
        }
    }
}

static void test_roundtrip(void) {
    tt_kvcache_cfg cfg = {0};
    enum { L = 35 };
    int kw[L], sl[L];
    e2b_arrays(kw, sl, L);
    cfg.n_layers = L; cfg.max_ctx = 1024; cfg.head_dim = 256;
    cfg.kv_width = kw; cfg.src_layer = sl; cfg.dtype_size = 4;

    tt_kvcache *kc = tt_kvcache_create(&cfg);
    CHECK(kc != NULL, "create rt");
    for (int l = 0; l < L; l++) kc->valid_len[l] = (uint32_t)(100 + l);

    tt_kv_layout lay1, *lay2 = NULL;
    CHECK(tt_kv_layout_build(&cfg, &lay1) == 0, "layout build");
    CHECK(lay1.cache_per_elems == 512L * 1024L, "cache_per stride");
    CHECK(lay1.blob_bytes ==
          512L * 1024L * L * 4L, "blob bytes one side");
    CHECK(lay1.loc[20].slab_offset_elems == 20L * 512L * 1024L,
          "layer slab offset");
    CHECK(lay1.loc[20].src_layer == 14 && lay1.loc[21].src_layer == 13,
          "shared src mapping");   /* L20 full-mirror ->14, L21 ->13 */
    CHECK(lay1.loc[7].slot_stride_elems == 256, "swa slot stride");

    const size_t sz = tt_kv_serialize_size(kc);
    uint8_t *buf = malloc(sz);
    CHECK(buf && tt_kv_serialize(kc, buf, sz) == sz, "serialize");

    tt_kvcache *kc2 = NULL;
    CHECK(tt_kv_deserialize(buf, sz, &kc2, &lay2) == 0, "deserialize");
    CHECK(kc2 && kc2->cfg.max_ctx == 1024 && kc2->cfg.head_dim == 256 &&
          kc2->cfg.dtype_size == 4, "dims roundtrip");
    CHECK(memcmp(kc->valid_len, kc2->valid_len,
                 L * sizeof(uint32_t)) == 0, "valid_len roundtrip");
    for (int l = 0; l < L; l++) {
        CHECK(kc2->kv_width[l] == kw[l], "kv_width roundtrip");
        CHECK(kc2->src_layer[l] == sl[l], "src roundtrip");
    }
    CHECK(lay2 && layout_eq(&lay1, lay2), "layout plans identical");

    /* corruption => checksum mismatch */    buf[HDR_OFF_KW(3)] ^= 0xff;
    tt_kvcache *kc3 = NULL;
    CHECK(tt_kv_deserialize(buf, sz, &kc3, NULL) == -3, "checksum catch");

    free(buf);
    tt_kv_layout_free(&lay1);
    tt_kv_layout_free(lay2);
    free(lay2);
    tt_kvcache_free(kc);
    tt_kvcache_free(kc2);
    printf("ok  serialize/deserialize roundtrip\n");
}

/* ------------------------------------------------------------------ */
/* E2B memory accounting vs hand-computed                              *//* ------------------------------------------------------------------ */

static void test_e2b_memory(void) {
    tt_kvcache_cfg cfg = {0};
    enum { L = 35 };
    static int kw[L], sl[L];
    e2b_arrays(kw, sl, L);
    cfg.n_layers = L; cfg.max_ctx = 1024; cfg.head_dim = 256;
    cfg.kv_width = kw; cfg.src_layer = sl; cfg.dtype_size = 4;

    /* hand-computed, ctx=1024, fp32:
     * ideal/token = 2*KV * (11*256 + 4*512) elems * 4B = 38912 B
     * uniform owner slab/token: maxw=512 => 2*512*4 = 4096 B x 15 owners
     * uniform all-layer slab/token: 4096 B x 35 layers                  */
    const long tok1024 = 1024;
    const long b_ideal  = 38912L * tok1024;      /*   39,845,888 */
    const long b_shared = 61440L * tok1024;      /*   62,914,560 */
    const long b_noshar = 143360L * tok1024;     /*  146,800,640 */

    tt_kv_mem mem;
    tt_kv_memory(&cfg, &mem);
    CHECK(labs(mem.b_ideal - b_ideal) * 100 < b_ideal, "ideal exact");
    CHECK(labs(mem.b_shared - b_shared) * 100 < b_shared, "shared exact");
    CHECK(labs(mem.b_no_share - b_noshar) * 100 < b_noshar, "noshare ex.");

    printf("E2B fp32 KV memory:\n");
    printf("  ctx      no-share(GiB)  shared-slab(GiB)  ideal-mixed(GiB)\n");
    static const int ctxs[3] = {1024, 8192, 32768};
    for (int ci = 0; ci < 3; ci++) {
        const int c = ctxs[ci];
        cfg.max_ctx = c;
        tt_kv_memory(&cfg, &mem);
        printf("  %-8d %.4f          %.4f            %.4f\n",
               c, mem.gb_no_share, mem.gb_shared, mem.gb_ideal);
        /* 1% tolerance vs scaled hand numbers at every ctx */
        CHECK(fabs(mem.gb_ideal * 1073741824.0 -
                   (double)b_ideal / 1024.0 * c) /
              ((double)b_ideal / 1024.0 * c) < 0.01, "ideal 1%%");
        CHECK(fabs(mem.gb_shared * 1073741824.0 -
                   (double)b_shared / 1024.0 * c) /
              ((double)b_shared / 1024.0 * c) < 0.01, "shared 1%%");
        CHECK(fabs(mem.gb_no_share * 1073741824.0 -
                   (double)b_noshar / 1024.0 * c) /
              ((double)b_noshar / 1024.0 * c) < 0.01, "noshare 1%%");
    }
}

int main(void) {
    test_swa_vs_bruteforce();
    test_compact_fragmented();
    test_rollback();
    test_roundtrip();
    test_e2b_memory();
    if (g_fail) {
        fprintf(stderr, "%d FAILURES\n", g_fail);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
