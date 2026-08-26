/* kvcache.h -- M10/M11 KV cache management layer (pure C99, no GPU deps).
 *
 * Build (standalone, no deps beyond libc):
 *   gcc -std=c99 -O2 -Wall -Wextra -c src/kvcache.c             # object
 *   gcc -std=c99 -O2 -Wall -Wextra -Isrc -o build/test_kvcache \
 *       src/kvcache.c tests/test_kvcache.c                      # unit bin
 *
 * NOTE: no Makefile change required for the shared library target:
 * lib rule already compiles every C source under src/, so kvcache.c is
 * picked up automatically. Deliberately NOT added to any nvcc example/
 * chat target until engine integration lands (same pattern as specdec.c).
 *
 * DESIGN: device-pointer agnostic. This layer tracks SIZES/OFFSETS only,
 * never raw pointers -- every plan it emits (compaction moves, zero
 * ranges, blob layout) is expressed in element offsets relative to a
 * per-layer slab, so the engine can execute it against host OR device
 * buffers unchanged.
 *
 * Slab layout mirrors kernels/qwen2_cuda.cu exactly:
 *   cache_per = max_kv * max_ctx * head_dim          (elements, per slab)
 *   layer l K-slab offset = l * cache_per            (same for V slabs)
 *   slot t, kv head h, dim i  ->  slab[(t*kv_width_l + h)*head_dim + i]
 * Shared (gemma4 KV-reuse) layers carry src_layer >= 0 and read the
 * SOURCE layer's slab (llama-model.cpp:2502 semantics; engine side:
 * qwen2_cuda.cu forward_layers() Kl_f/V_f remap).
 */

#ifndef TT_KVCACHE_H
#define TT_KVCACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

typedef struct tt_kvcache_cfg {
    int n_layers;
    int max_ctx;
    int head_dim;              /* meta head_dim (gemma4: 256)            */
    const int *kv_width;       /* [n_layers] per-layer kvdim_l =
                                  pl_kv[l]*pl_hd[l]. NULL => uniform
                                  n_kv_heads*head_dim == head_dim rows of
                                  width head_dim... callers with hetero
                                  layers MUST pass this.                  */
    const int *src_layer;      /* [n_layers] KV source layer, -1 = owns.
                                  NULL => every layer owns its KV.        */
    size_t dtype_size;         /* bytes/elem (engine caches are f32: 4)  */
} tt_kvcache_cfg;

typedef struct tt_kvcache {
    tt_kvcache_cfg cfg;        /* private deep copies of arrays          */
    int *kv_width;
    int *src_layer;
    uint32_t *valid_len;       /* per-layer logical token count          */
    long cache_per_elems;      /* max_kv_width * max_ctx ... NOTE: engine
                                  stride is max_kv*max_ctx*head_dim where
                                  max_kv counts META heads; we store the
                                  equivalent element count directly.      */
} tt_kvcache;

/* Returns NULL on allocation failure or bad args (n_layers/max_ctx <= 0,
 * dtype_size == 0, kv_width entry <= 0). */
tt_kvcache *tt_kvcache_create(const tt_kvcache_cfg *cfg);
void        tt_kvcache_free(tt_kvcache *kc);

/* ------------------------------------------------------------------ */
/* Append + rollback (spec-decode; specdec.h contract item 2)          */
/* ------------------------------------------------------------------ */

/* Advance every layer's valid length by n (clamped to max_ctx).
 * Returns 0, -1 on bad args. */
int       tt_kv_append(tt_kvcache *kc, uint32_t n);

/* Current valid tokens for a layer. */
uint32_t  tt_kv_valid(const tt_kvcache *kc, int layer);

/* Opaque rollback mark: snapshot of all per-layer valid lengths. */
typedef struct { uint32_t *len; uint32_t n; } tt_kv_mark;

tt_kv_mark *tt_kv_mark_create(const tt_kvcache *kc);
void        tt_kv_mark_free(tt_kv_mark *m);
int         tt_kv_restore(const tt_kv_mark *m, tt_kvcache *kc);

/* Truncate ALL layers to new_len (per layer: min(valid_len, new_len)).
 * If emit_zero != 0, also fill out[] (capacity 2*n_layers... see below)
 * with element ranges the ENGINE must memset to clear stale rejected-
 * draft tails: index 2*l -> K slab range, 2*l+1 -> V slab range, both as
 * element offsets relative to that layer's slab start. Layers with
 * nothing to clear get {0,0}. *n_out receives 2*n_layers.
 * Offsets use the layer's OWN kv_width stride. Returns 0, -1 bad args. */
typedef struct { long offset_elems; long n_elems; } tt_kv_zero_range;
int tt_kv_truncate(tt_kvcache *kc, uint32_t new_len, int emit_zero,
                   tt_kv_zero_range *out, uint32_t *n_out);

/* ------------------------------------------------------------------ */
/* SWA slot accounting + compaction PLANS (no copies executed here)    */
/* ------------------------------------------------------------------ */

typedef struct { long src_slot; long dst_slot; long count; } tt_kv_move;

typedef struct {
    tt_kv_move *moves;         /* engine executes: memmove(dst, src,
                                  count * kv_width * dtype) per slab     */
    uint32_t n_moves;
    uint32_t compacted_len;    /* valid length after compaction          */
    uint32_t evicted;          /* tokens dropped (were older than window)*/
} tt_kv_compact_plan;

void tt_kv_compact_plan_free(tt_kv_compact_plan *p);

/* Build compaction plan from per-slot keep flags. Slot t stays iff
 * keep[t] != 0. Live slots are packed to the front preserving order;
 * plan lists contiguous-run memmoves. Caller owns keep[max_ctx].
 * Returns 0, -1 on bad args. */
int tt_kv_compact_plan_from_keep(const int *keep, uint32_t valid_len,
                                 tt_kv_compact_plan *plan);

/* Convenience: SWA policy per layer. windows[l] > 0 => keep only the
 * newest `windows[l]` slots ([valid-w, valid)); 0 => keep everything
 * (full attention). Plans are per-layer; caller iterates layers and
 * frees each plan. windows == NULL means all-full (no moves). */
int tt_kv_compact_plan_swa(const tt_kvcache *kc, const uint32_t *windows,
                           tt_kv_compact_plan *out /*[n_layers]*/);

/* Which slots are logically DEAD for a sliding layer right now: first
 * dead slot index = valid_len > window ? valid_len - window : 0.
 * Slots [0, dead_from) are older-than-window garbage. */
uint32_t tt_kv_swa_dead_from(uint32_t valid_len, uint32_t window);

/* ------------------------------------------------------------------ */
/* Blob layout plan (session save/restore)                             */
/* ------------------------------------------------------------------ */

typedef struct {
    long slab_offset_elems;    /* l * cache_per_elems                    */
    long slot_stride_elems;    /* kv_width[l]                            */
    int  src_layer;            /* -1 = owns KV; else read source slab    */
} tt_kv_layer_loc;

typedef struct {
    long cache_per_elems;      /* uniform slab stride (engine formula)   */
    long blob_bytes;           /* bytes for ONE K or V blob, all layers  */
    uint32_t n_layers;
    tt_kv_layer_loc *loc;      /* [n_layers]                             */
} tt_kv_layout;

/* Build layout from cfg (independent of runtime valid_len). Returns 0,
 * -1 on bad args/alloc failure. Free with tt_kv_layout_free. */
int  tt_kv_layout_build(const tt_kvcache_cfg *cfg, tt_kv_layout *out);
void tt_kv_layout_free(tt_kv_layout *lay);

/* ------------------------------------------------------------------ */
/* Serialize / deserialize (chat session resume)                       */
/* ------------------------------------------------------------------ */

#define TT_KVCACHE_MAGIC   0x54544b56u   /* "TTKV" */
#define TT_KVCACHE_VERSION 1u

/* Wire format (little-endian, fixed-width LE fields):
 *   u32 magic, u32 version, u32 flags(=0), u32 header_bytes,
 *   u32 n_layers, u32 max_ctx, u32 head_dim, u32 dtype_size,
 *   i64 cache_per_elems, i64 blob_bytes,
 *   u32[n_layers] kv_width, i32[n_layers] src_layer,
 *   u32[n_layers] valid_len,
 *   u64 fnv1a64 checksum over ALL preceding bytes.
 * The KV BLOB ITSELF is not serialized here -- engine dumps/restores the
 * two raw blobs using the layout plan (K blob then V blob, same layout).
 */
size_t tt_kv_serialize_size(const tt_kvcache *kc);
/* Returns bytes written, or (size_t)-1 on bad args / cap too small. */
size_t tt_kv_serialize(const tt_kvcache *kc, uint8_t *buf, size_t cap);

/* Validates magic/version/checksum/dims. On success returns 0 and
 * *out_kc (freshly allocated) plus optional *out_layout (may be NULL).
 * Returns -1 bad args, -2 bad magic/version, -3 checksum mismatch,
 * -4 truncated, -5 alloc failure. */
int tt_kv_deserialize(const uint8_t *buf, size_t len,
                      tt_kvcache **out_kc, tt_kv_layout **out_layout);

/* FNV-1a 64 (exposed for tests + future blob-side checksums). */
uint64_t tt_fnv1a64(const uint8_t *data, size_t len);

/* ------------------------------------------------------------------ */
/* Memory accounting                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    double gb_no_share;   /* current-engine shape: uniform slab x ALL
                             n_layers (K+V), stride max_kv*ctx*hd       */
    double gb_shared;     /* uniform slab x KV-OWNING layers only       */
    double gb_ideal;      /* exact per-layer widths, owners only        */
    long   b_no_share;
    long   b_shared;
    long   b_ideal;
} tt_kv_mem;

/* Exact byte math, mirrors gate-script estimates:
 *   no_share = n_layers            * 2 * (max_kv*hd) * max_ctx * dt
 *   shared   = n_own               * 2 * (max_kv*hd) * max_ctx * dt
 *   ideal    = sum over owners     * 2 * kv_width_l  * max_ctx * dt      */
void tt_kv_memory(const tt_kvcache_cfg *cfg, tt_kv_mem *out);

#ifdef __cplusplus
}
#endif

#endif /* TT_KVCACHE_H */
