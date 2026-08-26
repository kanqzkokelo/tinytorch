#include "loader_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define GGUF_MAGIC 0x46554747 // "GGUF"

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} GGUFHeader;

static uint32_t read_u32(const uint8_t **p) {
    uint32_t val;
    memcpy(&val, *p, sizeof(val));
    *p += sizeof(val);
    return val;
}

static uint64_t read_u64(const uint8_t **p) {
    uint64_t val;
    memcpy(&val, *p, sizeof(val));
    *p += sizeof(val);
    return val;
}

static void read_string(const uint8_t **p, char *buf, size_t max_len) {
    uint64_t len = read_u64(p);
    size_t copy_len = len < max_len - 1 ? len : max_len - 1;
    memcpy(buf, *p, copy_len);
    buf[copy_len] = '\0';
    *p += len;
}

static void skip_kv_value(const uint8_t **p, uint32_t type) {
    switch (type) {
        case 0: *p += 1; break; // UINT8
        case 1: *p += 1; break; // INT8
        case 2: *p += 2; break; // UINT16
        case 3: *p += 2; break; // INT16
        case 4: *p += 4; break; // UINT32
        case 5: *p += 4; break; // INT32
        case 6: *p += 4; break; // FLOAT32
        case 7: *p += 1; break; // BOOL
        case 8: { // STRING
            uint64_t len = read_u64(p);
            *p += len;
            break;
        }
        case 9: { // ARRAY
            uint32_t item_type = read_u32(p);
            uint64_t array_len = read_u64(p);
            for (uint64_t i = 0; i < array_len; i++) {
                skip_kv_value(p, item_type);
            }
            break;
        }
        case 10: *p += 8; break; // UINT64
        case 11: *p += 8; break; // INT64
        case 12: *p += 8; break; // FLOAT64
        default: break;
    }
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
        return NULL;
    }

    size_t file_size = st.st_size;
    void *mmap_addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (mmap_addr == MAP_FAILED) {
        fprintf(stderr, "[GGUF] mmap failed for %s\n", filepath);
        return NULL;
    }

    const uint8_t *p = (const uint8_t *)mmap_addr;
    GGUFHeader header;
    header.magic = read_u32(&p);
    header.version = read_u32(&p);
    header.tensor_count = read_u64(&p);
    header.metadata_kv_count = read_u64(&p);

    if (header.magic != GGUF_MAGIC) {
        fprintf(stderr, "[GGUF] Invalid magic header: 0x%08x\n", header.magic);
        munmap(mmap_addr, file_size);
        return NULL;
    }

    GGUFModel *model = (GGUFModel *)calloc(1, sizeof(GGUFModel));
    model->mmap_addr = mmap_addr;
    model->mmap_size = file_size;
    model->tensor_count = (int)header.tensor_count;

    // Parse KV metadata. M7 task 3: keys are matched by SUFFIX after the
    // first dot, so <arch>.embedding_length works for every family
    // (llama./qwen2./qwen3./gemma2. ...) instead of a hardcoded pair.
    // Suffixes are unique enough across namespaces (tokenizer.* never uses
    // them). general.architecture is stored verbatim for the trait registry.
    char key[128];
    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        read_string(&p, key, sizeof(key));
        uint32_t value_type = read_u32(&p);
        const char *dot = strchr(key, '.');
        const char *sfx = dot ? dot + 1 : key;
        if (getenv("TT_KV_DEBUG")) fprintf(stderr, "[kv] %s vt=%u\n", key, value_type);

        /* general.architecture */
        if (value_type == 8 && strcmp(key, "general.architecture") == 0) {
            read_string(&p, model->architecture, sizeof(model->architecture));
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
                const uint32_t itype = read_u32(&p);
                const uint64_t alen = read_u64(&p);
                const uint64_t esz = (itype <= 6 || itype == 7) ? (itype <= 1 ? 1 : itype <= 6 ? 4 : 1) : 8;
                if (alen > 0 && esz >= 4) memcpy(&v, p, 4);
                else if (alen > 0 && esz == 1) v = p[0];
                p += alen * esz;
            } else {
                memcpy(&v, p, 4);
                p += 4;
            }
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
            model->rms_norm_eps = *(const float *)p;
            skip_kv_value(&p, value_type);
            continue;
        }
        if ((strcmp(sfx, "rope.freq_base") == 0 || strcmp(key, "llama.rope_freq_base") == 0)
            && value_type == 6) {
            model->rope_freq_base = *(const float *)p;
            skip_kv_value(&p, value_type);
            continue;
        }
        if (strcmp(sfx, "attention.sliding_window") == 0 && value_type == 4) {
            model->sliding_window = *(const int32_t *)p;
            skip_kv_value(&p, value_type);
            continue;
        }
        if (strcmp(sfx, "final_logit_softcapping") == 0 && value_type == 6) {
            model->final_logit_softcapping = *(const float *)p;
            skip_kv_value(&p, value_type);
            continue;
        }
        if (strcmp(sfx, "embedding_length_per_layer_input") == 0 && value_type == 4) {
            model->per_layer_embd_dim = *(const int32_t *)p;
            skip_kv_value(&p, value_type);
            continue;
        }
        if (strcmp(sfx, "context_length") == 0 && value_type == 4) {
            model->max_seq_len = *(const int32_t *)p;
            skip_kv_value(&p, value_type);
            continue;
        }

        skip_kv_value(&p, value_type);
    }

    if (model->rms_norm_eps == 0.0f) model->rms_norm_eps = 1e-6f;
    if (model->rope_freq_base == 0.0f) model->rope_freq_base = 10000.0f;
    if (model->n_kv_heads == 0) model->n_kv_heads = model->n_heads;
    if (model->max_seq_len == 0) model->max_seq_len = 2048;

    // Parse Tensor headers
    model->tensors = (GGUFTensor *)calloc(model->tensor_count, sizeof(GGUFTensor));
    for (int i = 0; i < model->tensor_count; i++) {
        GGUFTensor *t = &model->tensors[i];
        read_string(&p, t->name, sizeof(t->name));
        t->ndim = read_u32(&p);
        for (int d = 0; d < t->ndim; d++) {
            t->shape[d] = (int64_t)read_u64(&p);
        }
        t->type = (GGUFType)read_u32(&p);
        t->offset = read_u64(&p);
        t->data = NULL; // will be resolved after alignment

        // Calculate size in bytes
        int64_t numel = 1;
        for (int d = 0; d < t->ndim; d++) numel *= t->shape[d];

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
        if (t->type == GGUF_TYPE_F32) t->size_bytes = numel * 4;
        else if (t->type == GGUF_TYPE_F16 || t->type == GGUF_TYPE_BF16)
            t->size_bytes = numel * 2;
        else if (t->type == GGUF_TYPE_Q4_0) t->size_bytes = (numel / 32) * 18;
        else if (t->type == GGUF_TYPE_Q4_1) t->size_bytes = (numel / 32) * 20;
        else if (t->type == GGUF_TYPE_Q5_0) t->size_bytes = (numel / 32) * 22;
        else if (t->type == GGUF_TYPE_Q5_1) t->size_bytes = (numel / 32) * 24;
        else if (t->type == GGUF_TYPE_Q8_0) t->size_bytes = (numel / 32) * 34; /* fp16 d + 32 i8 */
        else if (t->type == GGUF_TYPE_Q4_K || t->type == GGUF_TYPE_Q5_K || t->type == GGUF_TYPE_Q6_K) {
            if (numel % 256 != 0)
                fprintf(stderr, "[GGUF] WARN: %s K-quant numel %lld not multiple of 256\n",
                        t->name, (long long)numel);
            long long blocks = numel / 256;
            if (t->type == GGUF_TYPE_Q4_K)      t->size_bytes = (size_t)(blocks * 144);
            else if (t->type == GGUF_TYPE_Q5_K) t->size_bytes = (size_t)(blocks * 176);
            else                                 t->size_bytes = (size_t)(blocks * 210);
        }
        else t->size_bytes = numel;
    }

    // Align p to 32 bytes for binary payload base
    uintptr_t current_pos = (uintptr_t)p;
    uintptr_t base_pos = (uintptr_t)mmap_addr;
    uintptr_t offset_from_base = current_pos - base_pos;
    uintptr_t aligned_offset = (offset_from_base + 31) & ~31;
    const uint8_t *binary_base = (const uint8_t *)mmap_addr + aligned_offset;

    // Resolve binary data pointers using GGUF tensor offsets
    for (int i = 0; i < model->tensor_count; i++) {
        GGUFTensor *t = &model->tensors[i];
        t->data = (void *)(binary_base + t->offset);
    }

    printf("[GGUF] Loaded model: arch=%s dim=%d, hidden=%d, layers=%d, heads=%d, kv_heads=%d, tensors=%d\n",
           model->architecture[0] ? model->architecture : "?",
           model->dim, model->hidden_dim, model->n_layers, model->n_heads, model->n_kv_heads, model->tensor_count);

    return model;
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
