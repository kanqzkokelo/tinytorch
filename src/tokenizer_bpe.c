#include "tokenizer_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GGUF_MAGIC 0x46554747 // "GGUF"

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

BPETokenizer *bpe_tokenizer_init(const GGUFModel *model) {
    if (!model || !model->mmap_addr) {
        fprintf(stderr, "[BPETokenizer] Error: Invalid GGUFModel\n");
        return NULL;
    }

    const uint8_t *p = (const uint8_t *)model->mmap_addr;
    uint32_t magic = read_u32(&p);
    (void)read_u32(&p); // version
    (void)read_u64(&p); // tensor_count
    uint64_t metadata_kv_count = read_u64(&p);

    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "[BPETokenizer] Error: Invalid magic header: 0x%08x\n", magic);
        return NULL;
    }

    BPETokenizer *tok = (BPETokenizer *)calloc(1, sizeof(BPETokenizer));
    tok->bos_id = -1;
    tok->eos_id = -1;

    char key[128];
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        read_string(&p, key, sizeof(key));
        uint32_t value_type = read_u32(&p);

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && value_type == 9) {
            uint32_t item_type = read_u32(&p);
            (void)item_type;
            uint64_t array_len = read_u64(&p);

            tok->vocab_size = (int)array_len;
            tok->tokens = (char **)calloc(tok->vocab_size, sizeof(char *));
            tok->token_lens = (int *)calloc(tok->vocab_size, sizeof(int));

            for (uint64_t t = 0; t < array_len; t++) {
                uint64_t slen = read_u64(&p);
                const uint8_t *src_bytes = p;
                p += slen;

                char *out_buf = (char *)malloc(slen + 1);
                int out_len = 0;

                for (uint64_t j = 0; j < slen; j++) {
                    uint8_t c = src_bytes[j];
                    if (c >= 32 && c <= 126) {
                        out_buf[out_len++] = (char)c;
                    } else if (c == '\n' || c == '\t' || c == '\r') {
                        out_buf[out_len++] = (char)c;
                    }
                }
                if (out_len == 0 && slen > 0) {
                    out_buf[out_len++] = ' ';
                }
                out_buf[out_len] = '\0';
                tok->tokens[t] = out_buf;
                tok->token_lens[t] = out_len;
            }
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && value_type == 9) {
            uint32_t item_type = read_u32(&p);
            (void)item_type;
            uint64_t array_len = read_u64(&p);
            tok->scores = (float *)calloc(array_len, sizeof(float));
            for (uint64_t t = 0; t < array_len; t++) {
                tok->scores[t] = *(const float *)p;
                p += 4;
            }
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            tok->bos_id = *(const int32_t *)p;
            skip_kv_value(&p, value_type);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            tok->eos_id = *(const int32_t *)p;
            skip_kv_value(&p, value_type);
        } else {
            skip_kv_value(&p, value_type);
        }
    }

    if (!tok->scores && tok->vocab_size > 0) {
        tok->scores = (float *)calloc(tok->vocab_size, sizeof(float));
    }

    printf("[BPETokenizer] Loaded tokenizer: vocab_size=%d, bos_id=%d, eos_id=%d\n",
           tok->vocab_size, tok->bos_id, tok->eos_id);

    return tok;
}

const char *bpe_decode_token(const BPETokenizer *tok, int token_id, int *out_len) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size || !tok->tokens[token_id]) {
        if (out_len) *out_len = 0;
        return "";
    }
    if (out_len) *out_len = tok->token_lens[token_id];
    return tok->tokens[token_id];
}

int bpe_encode(const BPETokenizer *tok, const char *text, int *out_tokens, int max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens <= 0) return 0;
    int pos = 0;
    int len = (int)strlen(text);
    int n_out = 0;

    while (pos < len && n_out < max_tokens) {
        int best_id = -1;
        int best_len = 0;
        float best_score = -1e9f;

        for (int i = 0; i < tok->vocab_size; i++) {
            int tlen = tok->token_lens[i];
            if (tlen <= 0 || tlen > len - pos) continue;

            if (strncmp(text + pos, tok->tokens[i], tlen) == 0) {
                float score = tok->scores ? tok->scores[i] : (float)tlen;
                if (tlen > best_len || (tlen == best_len && score > best_score)) {
                    best_id = i;
                    best_len = tlen;
                    best_score = score;
                }
            }
        }

        if (best_id >= 0 && best_len > 0) {
            out_tokens[n_out++] = best_id;
            pos += best_len;
        } else {
            // Fallback byte mapping
            pos++;
        }
    }
    return n_out;
}

void bpe_tokenizer_free(BPETokenizer *tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->tokens[i]) free(tok->tokens[i]);
        }
        free(tok->tokens);
    }
    if (tok->token_lens) free(tok->token_lens);
    if (tok->scores) free(tok->scores);
    free(tok);
}
