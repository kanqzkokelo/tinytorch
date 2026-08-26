#include "loader_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define GGUF_MAGIC 0x46554747 // "GGUF"

/* Sanity cap: no real model has anywhere near this many tensors; anything
 * larger in a header is corrupt/hostile input aiming at a malloc bomb. */
#define MAX_TENSORS 10000000ULL

/* Bounds-checked cursor over the mmap. Every read validates against `end`
 * and latches `err`; callers must check c.err before using results. This
 * guarantees a truncated/corrupt file fails cleanly instead of SIGBUS. */
typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int err;
} GCursor;

static size_t cur_left(const GCursor *c) { return (size_t)(c->end - c->p); }

static int take_bytes(GCursor *c, void *out, size_t n) {
    if (c->err || cur_left(c) < n) { c->err = 1; return 0; }
    memcpy(out, c->p, n);
    c->p += n;
    return 1;
}

static uint32_t read_u32(GCursor *c) { uint32_t v = 0; take_bytes(c, &v, sizeof(v)); return v; }
static uint64_t read_u64(GCursor *c) { uint64_t v = 0; take_bytes(c, &v, sizeof(v)); return v; }
static float read_f32(GCursor *c) { float v = 0; take_bytes(c, &v, sizeof(v)); return v; }

/* Bounded string read; always NUL-terminates within max_len. */
static int read_string(GCursor *c, char *buf, size_t max_len) {
    uint64_t len = read_u64(c);
    if (c->err || len > cur_left(c)) {
        c->err = 1;
        if (max_len) buf[0] = '\0';
        return 0;
    }
    size_t copy_len = len < max_len - 1 ? (size_t)len : max_len - 1;
    memcpy(buf, c->p, copy_len);
    buf[copy_len] = '\0';
    c->p += len;
    return 1;
}

/* Fixed payload size per GGUF metadata value type; 0 = variable/unknown. */
static size_t kv_fixed_size(uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;   // u8/i8/bool
        case 2: case 3: return 2;           // u16/i16
        case 4: case 5: case 6: return 4;   // u32/i32/f32
        case 10: case 11: case 12: return 8; // u64/i64/f64
        default: return 0;
    }
}

static int skip_kv_value(GCursor *c, uint32_t type) {
    if (c->err) return 0;
    size_t fixed = kv_fixed_size(type);
    if (fixed) {
        if (cur_left(c) < fixed) { c->err = 1; return 0; }
        c->p += fixed;
        return 1;
    }
    if (type == 8) { // STRING
        uint64_t len = read_u64(c);
        if (c->err || len > cur_left(c)) { c->err = 1; return 0; }
        c->p += len;
        return 1;
    }
    if (type == 9) { // ARRAY — GGUF forbids nested arrays; reject (stack-overflow guard)
        uint32_t item_type = read_u32(c);
        uint64_t array_len = read_u64(c);
        if (c->err || item_type == 9 || item_type > 12) { c->err = 1; return 0; }
        size_t item_sz = kv_fixed_size(item_type);
        if (item_sz) {
            /* overflow-safe: array_len elements must fit in remaining bytes */
            if (array_len > cur_left(c) / item_sz) { c->err = 1; return 0; }
            c->p += array_len * item_sz;
            return 1;
        }
        /* string array: each element is u64 len + bytes; bounded by err latch */
        for (uint64_t i = 0; i < array_len; i++) {
            uint64_t len = read_u64(c);
            if (c->err || len > cur_left(c)) { c->err = 1; return 0; }
            c->p += len;
        }
        return !c->err;
    }
    c->err = 1; // unknown value type
    return 0;
}

GGUFModel *gguf_load(const char *filepath) {
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[GGUF] Failed to open file: %s\n", filepath);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        fprintf(stderr, "[GGUF] fstat failed: %s\n", filepath);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    if (file_size < 8) {
        close(fd);
        fprintf(stderr, "[GGUF] File too small to be GGUF (%zu bytes): %s\n", file_size, filepath);
        return NULL;
    }

    void *mmap_addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (mmap_addr == MAP_FAILED) {
        fprintf(stderr, "[GGUF] mmap failed for %s\n", filepath);
        return NULL;
    }

    GCursor c = { (const uint8_t *)mmap_addr, (const uint8_t *)mmap_addr + file_size, 0 };
    GGUFModel *model = NULL;

    /* Header: validate magic BEFORE consuming counts so even a tiny truncated
     * file gets "invalid magic" instead of a fault. */
    uint32_t magic = read_u32(&c);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "[GGUF] Invalid magic header: 0x%08x\n", magic);
        munmap(mmap_addr, file_size);
        return NULL;
    }
    uint32_t version = read_u32(&c);
    uint64_t tensor_count = read_u64(&c);
    uint64_t kv_count = read_u64(&c);
    if (c.err) {
        fprintf(stderr, "[GGUF] Truncated header: %s\n", filepath);
        munmap(mmap_addr, file_size);
        return NULL;
    }

    /* Malloc-bomb guards (audit finding 10): u64 counts must be sane relative
     * to actual file size before any allocation. */
    if (tensor_count == 0 || tensor_count > MAX_TENSORS) {
        fprintf(stderr, "[GGUF] Unrealistic tensor count %llu (cap %llu): %s\n",
                (unsigned long long)tensor_count, (unsigned long long)MAX_TENSORS, filepath);
        munmap(mmap_addr, file_size);
        return NULL;
    }
    if (kv_count > file_size / 8) {
        fprintf(stderr, "[GGUF] Unrealistic metadata kv count %llu for %zu-byte file: %s\n",
                (unsigned long long)kv_count, file_size, filepath);
        munmap(mmap_addr, file_size);
        return NULL;
    }

    model = (GGUFModel *)calloc(1, sizeof(GGUFModel));
    if (!model) {
        fprintf(stderr, "[GGUF] Out of memory (model struct)\n");
        munmap(mmap_addr, file_size);
        return NULL;
    }
    model->mmap_addr = mmap_addr;
    model->mmap_size = file_size;
    model->tensor_count = (int)tensor_count; /* safe: capped above */

    // Parse KV metadata. M7 task 3: keys are matched by SUFFIX after the
    // first dot, so <arch>.embedding_length works for every family
    // (llama./qwen2./qwen3./gemma2. ...) instead of a hardcoded pair.
    // Suffixes are unique enough across namespaces (tokenizer.* never uses
    // them). general.architecture is stored verbatim for the trait registry.
    char key[128];
    for (uint64_t i = 0; i < kv_count && !c.err; i++) {
        read_string(&c, key, sizeof(key));
        uint32_t value_type = read_u32(&c);
        if (c.err) break;
        const char *dot = strchr(key, '.');
        const char *sfx = dot ? dot + 1 : key;
        if (getenv("TT_KV_DEBUG")) fprintf(stderr, "[kv] %s vt=%u\n", key, value_type);

        /* general.architecture */
        if (value_type == 8 && strcmp(key, "general.architecture") == 0) {
            read_string(&c, model->architecture, sizeof(model->architecture));
            continue;
        }

        /* Numeric model-dimension keys. Some families (gemma4) store these as
         * u32 ARRAYS with header item_type(u32)+count(u64); element[0] is the
         * value (uniform across layers for E-series models). Scalars read i32. */
        static const char *dim_keys[] = {"embedding_length", "feed_forward_length",
            "block_count", "attention.head_count", "attention.head_count_kv",
            "attention.key_length", "attention.value_length",
            "rope.dimension_count", "rope.dimension_count_swa"};
        int dk = -1;
        for (int ki = 0; ki < 9; ki++)
            if (strcmp(sfx, dim_keys[ki]) == 0) { dk = ki; break; }
        if (dk >= 0 && (value_type == 9 || value_type == 4)) {
            int32_t v = 0;
            if (value_type == 9) {
                const uint32_t itype = read_u32(&c);
                const uint64_t alen = read_u64(&c);
                if (c.err) break;
                const uint64_t esz = (itype <= 6 || itype == 7) ? (itype <= 1 ? 1 : itype <= 6 ? 4 : 1) : 8;
                if (alen > cur_left(&c) / (esz ? esz : 1)) { c.err = 1; break; }
                if (alen > 0 && esz >= 4) memcpy(&v, c.p, 4);
                else if (alen > 0 && esz == 1) v = c.p[0];
                c.p += alen * esz;
            } else {
                v = (int32_t)read_u32(&c);
            }
            if (c.err) break;
            switch (dk) {
                case 0: model->dim = v; break;
                case 1: model->hidden_dim = v; break;
                case 2: model->n_layers = v; break;
                case 3: model->n_heads = v; break;
                case 4: model->n_kv_heads = v; break;
                case 5: model->head_dim = v; break;
                default: break;   /* value_length / rope dims unused yet */
            }
            continue;
        }

        if (strcmp(sfx, "attention.layer_norm_rms_epsilon") == 0 && value_type == 6) {
            model->rms_norm_eps = read_f32(&c);
            continue;
        }
        if ((strcmp(sfx, "rope.freq_base") == 0 || strcmp(key, "llama.rope_freq_base") == 0)
            && value_type == 6) {
            model->rope_freq_base = read_f32(&c);
            continue;
        }
        if (strcmp(sfx, "attention.sliding_window") == 0 && value_type == 4) {
            model->sliding_window = (int32_t)read_u32(&c);
            continue;
        }
        if (strcmp(sfx, "attention.shared_kv_layers") == 0 && value_type == 4) {
            model->shared_kv_layers = (int32_t)read_u32(&c);
            continue;
        }
        if (strcmp(sfx, "final_logit_softcapping") == 0 && value_type == 6) {
            model->final_logit_softcapping = read_f32(&c);
            continue;
        }
        if (strcmp(sfx, "embedding_length_per_layer_input") == 0 && value_type == 4) {
            model->per_layer_embd_dim = (int32_t)read_u32(&c);
            continue;
        }
        if (strcmp(sfx, "context_length") == 0 && value_type == 4) {
            model->max_seq_len = (int32_t)read_u32(&c);
            continue;
        }

        skip_kv_value(&c, value_type);
    }
    if (c.err) {
        fprintf(stderr, "[GGUF] Truncated or corrupt metadata section: %s\n", filepath);
        goto fail;
    }

    if (model->rms_norm_eps == 0.0f) model->rms_norm_eps = 1e-6f;
    if (model->rope_freq_base == 0.0f) model->rope_freq_base = 10000.0f;
    if (model->n_kv_heads == 0) model->n_kv_heads = model->n_heads;
    if (model->max_seq_len == 0) model->max_seq_len = 2048;

    // Parse Tensor headers
    model->tensors = (GGUFTensor *)calloc((size_t)model->tensor_count, sizeof(GGUFTensor));
    if (!model->tensors) {
        fprintf(stderr, "[GGUF] Out of memory for %d tensors\n", model->tensor_count);
        goto fail;
    }
    for (int i = 0; i < model->tensor_count; i++) {
        GGUFTensor *t = &model->tensors[i];
        read_string(&c, t->name, sizeof(t->name));
        uint32_t ndim = read_u32(&c);
        if (c.err) break;
        if (ndim < 1 || ndim > 4) { // shape[4] guard (audit finding 4)
            fprintf(stderr, "[GGUF] Tensor '%s': unsupported ndim %u (max 4)\n", t->name, ndim);
            goto fail;
        }
        t->ndim = (int)ndim;
        for (int d = 0; d < t->ndim; d++) {
            t->shape[d] = (int64_t)read_u64(&c);
        }
        t->type = (GGUFType)read_u32(&c);
        t->offset = read_u64(&c);
        if (c.err) break;
        t->data = NULL; // will be resolved after alignment

        // Calculate size in bytes (with overflow-checked product)
        int64_t numel = 1;
        for (int d = 0; d < t->ndim; d++) {
            if (t->shape[d] <= 0 || numel > INT64_MAX / t->shape[d]) {
                fprintf(stderr, "[GGUF] Tensor '%s': invalid/overflowing shape\n", t->name);
                goto fail;
            }
            numel *= t->shape[d];
        }

        /* Block sizes per ggml-common.h (verified vs gguf-py GGML_QUANT_SIZES):
         *   q4_0: fp16 d + 16 nibble bytes            = 18 B / 32 values
         *   q4_1: fp16 d, m + 16 nibble bytes          = 20 B / 32
         *   q5_0: fp16 d + 4 high bits + 16 nibbles     = 22 B / 32
         *   q5_1: fp16 d, m + 4 high bits + 16 nibbles   = 24 B / 32
         *   q8_0: fp16 d + 32 int8                      = 34 B / 32
         * K-quants are super-blocks of 256 values (8 sub-blocks of 32),
         * requiring n_per_row to be a multiple of 256:
         *   q4_K/q4_K_S: fp16 d,dmin + scales[12] + qs[128] = 144 B / 256
         *   q5_K/q5_K_S: ... + qh[32]                        = 176 B / 256
         *   q6_K: ql[128] + qh[64] + int8 scales[16] + fp16 d = 210 B / 256
         */
        if (t->type == GGUF_TYPE_F32) t->size_bytes = (size_t)numel * 4;
        else if (t->type == GGUF_TYPE_F16 || t->type == GGUF_TYPE_BF16)
            t->size_bytes = (size_t)numel * 2;
        else if (t->type == GGUF_TYPE_Q4_0) t->size_bytes = (size_t)(numel / 32) * 18;
        else if (t->type == GGUF_TYPE_Q4_1) t->size_bytes = (size_t)(numel / 32) * 20;
        else if (t->type == GGUF_TYPE_Q5_0) t->size_bytes = (size_t)(numel / 32) * 22;
        else if (t->type == GGUF_TYPE_Q5_1) t->size_bytes = (size_t)(numel / 32) * 24;
        else if (t->type == GGUF_TYPE_Q8_0) t->size_bytes = (size_t)(numel / 32) * 34; /* fp16 d + 32 i8 */
        else if (t->type == GGUF_TYPE_Q4_K || t->type == GGUF_TYPE_Q5_K || t->type == GGUF_TYPE_Q6_K) {
            if (numel % 256 != 0)
                fprintf(stderr, "[GGUF] WARN: %s K-quant numel %lld not multiple of 256\n",
                        t->name, (long long)numel);
            long long blocks = numel / 256;
            if (t->type == GGUF_TYPE_Q4_K)      t->size_bytes = (size_t)(blocks * 144);
            else if (t->type == GGUF_TYPE_Q5_K) t->size_bytes = (size_t)(blocks * 176);
            else                                 t->size_bytes = (size_t)(blocks * 210);
        }
        else { // unknown dtype: hard failure, never a bogus numel fallback (audit finding 8)
            fprintf(stderr, "[GGUF] Unsupported dtype %d for tensor '%s'\n", (int)t->type, t->name);
            goto fail;
        }
    }
    if (c.err) {
        fprintf(stderr, "[GGUF] Truncated tensor table: %s\n", filepath);
        goto fail;
    }

    // Align p to 32 bytes for binary payload base
    uintptr_t current_pos = (uintptr_t)c.p;
    uintptr_t base_pos = (uintptr_t)mmap_addr;
    uintptr_t offset_from_base = current_pos - base_pos;
    uintptr_t aligned_offset = (offset_from_base + 31) & ~31;
    if (aligned_offset > file_size) {
        fprintf(stderr, "[GGUF] Tensor data region beyond EOF: %s\n", filepath);
        goto fail;
    }
    const uint8_t *binary_base = (const uint8_t *)mmap_addr + aligned_offset;

    // Resolve binary data pointers using GGUF tensor offsets, validating each
    // tensor's [offset, offset+size_bytes) window against EOF (audit finding 3).
    uint64_t avail = (uint64_t)(file_size - aligned_offset);
    for (int i = 0; i < model->tensor_count; i++) {
        GGUFTensor *t = &model->tensors[i];
        if (t->offset > avail || t->size_bytes > avail - t->offset) {
            fprintf(stderr,
                    "[GGUF] Tensor '%s' data out of bounds: offset=%llu size=%zu avail=%llu\n",
                    t->name, (unsigned long long)t->offset, t->size_bytes,
                    (unsigned long long)avail);
            goto fail;
        }
        t->data = (void *)(binary_base + t->offset);
    }

    printf("[GGUF] Loaded model: arch=%s dim=%d, hidden=%d, layers=%d, heads=%d, kv_heads=%d, tensors=%d (gguf v%u)\n",
           model->architecture[0] ? model->architecture : "?",
           model->dim, model->hidden_dim, model->n_layers, model->n_heads, model->n_kv_heads,
           model->tensor_count, version);

    return model;

fail:
    if (model) {
        free(model->tensors);
        free(model);
    }
    munmap(mmap_addr, file_size);
    return NULL;
}

GGUFTensor *gguf_get_tensor(GGUFModel *model, const char *name) {
    if (!model) return NULL;
    for (int i = 0; i < model->tensor_count; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) return &model->tensors[i];
    }
    return NULL;
}

void gguf_free(GGUFModel *model) {
    if (!model) return;
    if (model->mmap_addr) munmap(model->mmap_addr, model->mmap_size);
    if (model->tensors) free(model->tensors);
    free(model);
}
