// Byte-level BPE tokenizer (M6 correctness rewrite).
//
// Fixes vs M5 version:
//  - Token strings stored VERBATIM from GGUF (the old ASCII filter destroyed
//    multibyte UTF-8 tokens, corrupting both decode output and encode matching).
//  - Encode is real BPE: lowest-rank adjacent-pair merges using
//    tokenizer.ggml.merges, not greedy longest-substring scan.
//  - Unknown bytes fall back to <0xNN> tokens instead of being dropped.
//
// Known limitation vs llama.cpp (documented in PLAN_M6): pre-tokenization uses
// a simplified boundary rule, not Qwen's full GPT-4-style regex. Gate T2
// (100% ID parity) may fail on exotic inputs until that regex is ported.
#include "tokenizer_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#define GGUF_MAGIC 0x46554747

/* ---------------- GGUF stream readers (same as loader) ---------------- */

static uint32_t read_u32(const uint8_t **p) {
    uint32_t v; memcpy(&v, *p, 4); *p += 4; return v;
}
static uint64_t read_u64(const uint8_t **p) {
    uint64_t v; memcpy(&v, *p, 8); *p += 8; return v;
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
        case 0: case 1: case 7: *p += 1; break;
        case 2: case 3: *p += 2; break;
        case 4: case 5: case 6: *p += 4; break;
        case 10: case 11: case 12: *p += 8; break;
        case 8: { uint64_t l = read_u64(p); *p += l; break; }
        case 9: {
            uint32_t it = read_u32(p);
            uint64_t al = read_u64(p);
            for (uint64_t i = 0; i < al; i++) skip_kv_value(p, it);
            break;
        }
        default: break;
    }
}

/* ---------------- open-addressing string hash map ---------------- */

typedef struct { char *key; int val; int used; } HashEnt;

typedef struct {
    HashEnt *ents;
    size_t cap;          /* power of two */
} HashMap;

static uint64_t fnv1a(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void hm_init(HashMap *m, size_t expected) {
    size_t cap = 16;
    while (cap < expected * 2) cap <<= 1;
    m->ents = (HashEnt *)calloc(cap, sizeof(HashEnt));
    m->cap = cap;
}
static void hm_free(HashMap *m) {
    if (!m->ents) return;
    for (size_t i = 0; i < m->cap; i++) free(m->ents[i].key);
    free(m->ents);
    m->ents = NULL;
}
/* returns pointer to slot (existing or empty), NULL if table full */
static HashEnt *hm_slot(HashMap *m, const char *key, int klen) {
    size_t i = fnv1a(key, klen) & (m->cap - 1);
    for (size_t probe = 0; probe < m->cap; probe++) {
        HashEnt *e = &m->ents[i];
        if (!e->used) return e;
        if (strlen(e->key) == (size_t)klen && memcmp(e->key, key, klen) == 0) return e;
        i = (i + 1) & (m->cap - 1);
    }
    return NULL;
}
static void hm_put(HashMap *m, const char *key, int klen, int val) {
    HashEnt *e = hm_slot(m, key, klen);
    if (!e) return;
    if (!e->used) {
        e->key = (char *)malloc((size_t)klen + 1);
        memcpy(e->key, key, (size_t)klen);
        e->key[klen] = '\0';
        e->used = 1;
    }
    e->val = val;
}
static int hm_get(HashMap *m, const char *key, int klen) {
    HashEnt *e = hm_slot(m, key, klen);
    return (e && e->used) ? e->val : -1;
}

#define MAX_SYM_BYTES 64   /* single BPE word chunk bound */

/* ---------------- tokenizer object ----------------
 * Reuses BPETokenizer layout from the header and adds side tables via
 * internal struct extension (header fields stay ABI-stable). */
/* GPT-2 byte-to-unicode mapping used by byte-level BPE vocabs:
 * printable ASCII/Latin-1 ranges map to themselves; the remaining bytes map
 * to codepoints starting at U+0100 (e.g. 0x20 space -> U+0120 'Ġ').
 * GGUF token strings for Qwen-style models live in THIS domain, so encode
 * must map text into it and decode must map back out. */
typedef struct {
    BPETokenizer base;
    HashMap tok2id;      /* token bytes -> id */
    HashMap pair_rank;   /* merged-bytes -> merge rank */
    int sp_mode;         /* SentencePiece vocab ('llama'/'gemma4' ggml models) */
    int pre_add_bos;     /* tokenizer.ggml.pre implies BOS (llama3-style BPE) */
    int add_bos;         /* resolved add-BOS convention */
    int kv_add_bos_seen; /* explicit add_bos KV present (overrides defaults) */
    int max_piece;       /* longest vocab piece in bytes */
    unsigned short byte2cp[256];
    signed int cp2byte[32768];       /* -1 if unused */
    char *dec_buf; size_t dec_cap;   /* decode scratch */
} Tok;

static void build_byte_unicode_tables(Tok *t) {
    int added[256] = {0};
    int idx = 0, sp = 0;                 /* idx: slot; sp: specials so far */
    unsigned short bs[256], cs[256];
    for (int b = 33; b <= 126; b++)  { bs[idx]=b; cs[idx]=b; added[b]=1; idx++; }
    for (int b = 161; b <= 172; b++) { bs[idx]=b; cs[idx]=b; added[b]=1; idx++; }
    for (int b = 174; b <= 255; b++) { bs[idx]=b; cs[idx]=b; added[b]=1; idx++; }
    for (int b = 0; b < 256; b++)
        if (!added[b]) { bs[idx]=b; cs[idx]=(unsigned short)(256+sp); sp++; idx++; }
    /* n == 256 now; cs holds final codepoints in bs order */
    memset(t->cp2byte, 0xff, sizeof(t->cp2byte));   /* -1 */
    for (int i = 0; i < 256; i++) {
        t->byte2cp[bs[i]] = cs[i];
        t->cp2byte[cs[i]] = bs[i];
    }
}

/* encode one codepoint as UTF-8; returns length */
static int utf8_enc(unsigned int cp, char *out) {
    if (cp < 0x80) { out[0]=(char)cp; return 1; }
    if (cp < 0x800) { out[0]=(char)(0xC0|(cp>>6)); out[1]=(char)(0x80|(cp&63)); return 2; }
    if (cp < 0x10000) { out[0]=(char)(0xE0|(cp>>12)); out[1]=(char)(0x80|((cp>>6)&63)); out[2]=(char)(0x80|(cp&63)); return 3; }
    out[0]=(char)(0xF0|(cp>>18)); out[1]=(char)(0x80|((cp>>12)&63));
    out[2]=(char)(0x80|((cp>>6)&63)); out[3]=(char)(0x80|(cp&63)); return 4;
}
/* decode one UTF-8 codepoint; returns length, -1 invalid */
static int utf8_dec(const char *s, int len, unsigned int *cp) {
    unsigned char c0 = (unsigned char)s[0];
    if (c0 < 0x80) { *cp=c0; return 1; }
    int n; unsigned int v;
    if ((c0 & 0xE0)==0xC0) { n=2; v=c0&0x1F; }
    else if ((c0 & 0xF0)==0xE0) { n=3; v=c0&0x0F; }
    else if ((c0 & 0xF8)==0xF0) { n=4; v=c0&0x07; }
    else return -1;
    if (n > len) return -1;
    for (int i=1;i<n;i++){
        unsigned char ci=(unsigned char)s[i];
        if ((ci&0xC0)!=0x80) return -1;
        v=(v<<6)|(ci&0x3F);
    }
    *cp=v; return n;
}

BPETokenizer *bpe_tokenizer_init(const GGUFModel *model) {
    if (!model || !model->mmap_addr) {
        fprintf(stderr, "[BPE] invalid GGUFModel\n");
        return NULL;
    }

    const uint8_t *p = (const uint8_t *)model->mmap_addr;
    if (read_u32(&p) != GGUF_MAGIC) { fprintf(stderr, "[BPE] bad magic\n"); return NULL; }
    (void)read_u32(&p);
    (void)read_u64(&p);
    const uint64_t kv_count = read_u64(&p);

    Tok *t = (Tok *)calloc(1, sizeof(Tok));
    t->base.bos_id = -1;
    t->base.eos_id = -1;
    build_byte_unicode_tables(t);
    /* defaults mirror llama-vocab.cpp: SPM prepends BOS, plain BPE does not;
     * refined after the KV scan by tokenizer.ggml.pre / add_bos_token */
    t->add_bos = 1;
    t->kv_add_bos_seen = 0;
    t->dec_cap = 4096;
    t->dec_buf = (char *)malloc(t->dec_cap);

    char key[128];
    uint64_t n_merges = 0;
    for (uint64_t i = 0; i < kv_count; i++) {
        read_string(&p, key, sizeof(key));
        const uint32_t vtype = read_u32(&p);

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            (void)read_u32(&p);                       /* item type (STRING) */
            const uint64_t n = read_u64(&p);
            t->base.vocab_size = (int)n;
            t->base.tokens = (char **)calloc(n, sizeof(char *));
            t->base.token_lens = (int *)calloc(n, sizeof(int));
            hm_init(&t->tok2id, (size_t)n);

            for (uint64_t id = 0; id < n; id++) {
                const uint64_t slen = read_u64(&p);   /* helper advances p */
                char *s = (char *)malloc((size_t)slen + 1);
                memcpy(s, p, (size_t)slen);            /* RAW BYTES, no filtering */
                s[slen] = '\0';
                p += slen;
                t->base.tokens[id] = s;
                t->base.token_lens[id] = (int)slen;
                if ((int)slen > t->max_piece) t->max_piece = (int)slen;
                hm_put(&t->tok2id, s, (int)slen, (int)id);
            }
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == 9) {
            (void)read_u32(&p);
            const uint64_t n = read_u64(&p);
            hm_init(&t->pair_rank, (size_t)n);
            n_merges = n;
            for (uint64_t r = 0; r < n; r++) {
                const uint64_t slen = read_u64(&p);   /* helper advances p */
                /* entry format "left right" — rank key is the concatenated pair */
                const char *sp = memchr(p, ' ', (size_t)slen);
                const int ll = sp ? (int)(sp - (const char *)p) : -1;
                const int rl = sp ? (int)(slen - ll - 1) : -1;
                if (ll > 0 && rl > 0 && ll + rl < MAX_SYM_BYTES) {
                    char cat[MAX_SYM_BYTES];
                    memcpy(cat, p, (size_t)ll);
                    memcpy(cat + ll, sp + 1, (size_t)rl);
                    hm_put(&t->pair_rank, cat, ll + rl, (int)r);
                }
                p += slen;
            }
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && vtype == 8) {
            const uint64_t slen = read_u64(&p);
            char tm[32];
            const size_t cp = slen < sizeof(tm)-1 ? slen : sizeof(tm)-1;
            memcpy(tm, p, cp); tm[cp] = '\0'; p += slen;
            t->sp_mode = (strcmp(tm, "gpt2") != 0);   /* llama/gemma4/gemma => SP */
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && vtype == 8) {
            const uint64_t slen = read_u64(&p);
            char pre[32];
            const size_t cp2 = slen < sizeof(pre)-1 ? slen : sizeof(pre)-1;
            memcpy(pre, p, cp2); pre[cp2] = '\0'; p += slen;
            /* mirror llama-vocab.cpp: these BPE families prepend BOS by default */
            t->pre_add_bos =
                strcmp(pre, "llama3") == 0 || strcmp(pre, "llama-v3") == 0 ||
                strcmp(pre, "llama-bpe") == 0 || strcmp(pre, "falcon3") == 0 ||
                strcmp(pre, "falcon-h1") == 0 || strcmp(pre, "pixtral") == 0 ||
                strcmp(pre, "midm-2.0") == 0 || strcmp(pre, "lfm2") == 0 ||
                strcmp(pre, "jina-v5-nano") == 0 || strcmp(pre, "tekken") == 0 ||
                strcmp(pre, "chameleon") == 0;
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && vtype == 9) {
            (void)read_u32(&p);
            const uint64_t n = read_u64(&p);
            t->base.scores = (float *)calloc(n, sizeof(float));
            for (uint64_t j = 0; j < n; j++) { t->base.scores[j] = *(const float *)p; p += 4; }
        } else if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0 && vtype == 7) {
            t->add_bos = *(const int8_t *)p ? 1 : 0;
            t->kv_add_bos_seen = 1;
            p += 1;
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            t->base.bos_id = *(const int32_t *)p;
            skip_kv_value(&p, vtype);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            t->base.eos_id = *(const int32_t *)p;
            skip_kv_value(&p, vtype);
        } else {
            skip_kv_value(&p, vtype);
        }
    }

    if (!t->base.tokens || (!t->sp_mode && !t->pair_rank.ents)) {
        fprintf(stderr, "[BPE] missing tokens or merges in GGUF\n");
        bpe_tokenizer_free(&t->base);
        return NULL;
    }
    if (!t->kv_add_bos_seen)
        t->add_bos = t->sp_mode ? 1 : t->pre_add_bos;
    printf("[BPE] loaded: vocab=%d merges=%llu bos=%d eos=%d add_bos=%d\n",
           t->base.vocab_size, (unsigned long long)n_merges,
           t->base.bos_id, t->base.eos_id, t->add_bos);
    return &t->base;
}

const char *bpe_decode_token(const BPETokenizer *tok_, int token_id, int *out_len) {
    Tok *t = (Tok *)tok_;
    const char *raw = "";
    int raw_len = 0;
    if (tok_ && token_id >= 0 && token_id < tok_->vocab_size && tok_->tokens[token_id]) {
        raw = tok_->tokens[token_id];
        raw_len = tok_->token_lens[token_id];
    } else { if (out_len) *out_len = 0; return ""; }

    /* ensure capacity (worst case: every mapped char is 1 byte) */
    if (t->dec_cap < (size_t)raw_len + 1) {
        while (t->dec_cap < (size_t)raw_len + 1) t->dec_cap *= 2;
        t->dec_buf = (char *)realloc(t->dec_buf, t->dec_cap);
    }

    int o = 0, i = 0;
    if (t->sp_mode) {
        /* SentencePiece pieces are raw text: '▁'(U+2581) => space,
         * '<0xNN>' byte-fallback pieces => the single byte, everything
         * else passes through verbatim. */
        while (i < raw_len) {
            if (raw[i] == '<' && i + 5 < raw_len && raw[i+1] == '0' && raw[i+2] == 'x'
                && raw[i+5] == '>') {
                unsigned int bval;
                if (sscanf(raw + i + 3, "%2x", &bval) == 1) {
                    t->dec_buf[o++] = (char)bval;
                    i += 6;
                    continue;
                }
            }
            /* '▁' = U+2581 = 0xE2 0x96 0x81 */
            if ((unsigned char)raw[i] == 0xE2 && i + 2 < raw_len
                && (unsigned char)raw[i+1] == 0x96
                && (unsigned char)raw[i+2] == 0x81) {
                t->dec_buf[o++] = ' ';
                i += 3;
                continue;
            }
            t->dec_buf[o++] = raw[i++];
        }
        t->dec_buf[o] = '\0';
        if (out_len) *out_len = o;
        return t->dec_buf;
    }

    /* BPE (gpt2) domain: map chars back to real bytes */
    while (i < raw_len) {
        unsigned int cp;
        const int n = utf8_dec(raw + i, raw_len - i, &cp);
        if (n < 0) { t->dec_buf[o++] = raw[i++]; continue; }
        i += n;
        const signed int b = (cp < 32768) ? t->cp2byte[cp] : -1;
        if (b >= 0) t->dec_buf[o++] = (char)b;
        else { memcpy(t->dec_buf + o, raw + i - n, (size_t)n); o += n; }
    }
    t->dec_buf[o] = '\0';
    if (out_len) *out_len = o;
    return t->dec_buf;
}

/* ---------------- encode ---------------- */

/* character-class helpers (byte-level; UTF-8 multibyte sequences are treated
 * as opaque non-letter/non-digit symbols and handled via merges/fallback) */
static int is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static int is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }

#define MAX_SYM_BYTES 96   /* single BPE word chunk bound; longer input splits */

/* Simplified pre-tokenization boundaries (see top-of-file limitation note).
 * Mirrors the spirit of the Qwen/cl100k patterns:
 *   newline runs | space-led words | letter runs | digit runs (<=3) | punct runs */
static int chunk_len(const char *s, int len) {
    unsigned char c = (unsigned char)s[0];
    int i = 1;

    if (c == '\r' || c == '\n') {
        while (i < len && ((unsigned char)s[i] == '\r' || (unsigned char)s[i] == '\n')) i++;
        return i;
    }
    if (c == ' ') {
        while (i < len && s[i] == ' ') i++;
        /* a single leading space attaches to the following word/run */
        if (i < len && is_ascii_letter((unsigned char)s[i])) {
            i++;
            while (i < len && is_ascii_letter((unsigned char)s[i]) &&
                   i - 1 < MAX_SYM_BYTES) i++;
        } else if (i < len && is_ascii_digit((unsigned char)s[i])) {
            for (int d = 0; d < 3 && i < len && is_ascii_digit((unsigned char)s[i]); d++) i++;
        }
        return i;
    }
    if (is_ascii_letter(c)) {
        while (i < len && is_ascii_letter((unsigned char)s[i]) && i < MAX_SYM_BYTES) i++;
        return i;
    }
    if (is_ascii_digit(c)) {
        int d = 0;
        while (d < 3 && i < len && is_ascii_digit((unsigned char)s[i])) { i++; d++; }
        return i;
    }
    /* punctuation / symbol run (UTF-8 lead bytes treated as opaque) */
    while (i < len && s[i] != ' ' && s[i] != '\r' && s[i] != '\n' &&
           !is_ascii_letter((unsigned char)s[i]) && !is_ascii_digit((unsigned char)s[i]) &&
           i < MAX_SYM_BYTES) i++;
    return i;
}

int bpe_encode(const BPETokenizer *tok_, const char *text, int *out_tokens, int max_tokens) {
    Tok *t = (Tok *)tok_;
    if (!tok_ || !t->base.tokens || !text || !out_tokens || max_tokens <= 0) return 0;
    const int len = (int)strlen(text);
    int n_out = 0;
    int pos = 0;

    /* BPE-mode families (llama3 etc.) declare add_bos in GGUF metadata;
     * honor it exactly like the SP path so prompts match family convention. */
    if (!t->sp_mode && t->add_bos && t->base.bos_id >= 0 && max_tokens > 0)
        out_tokens[n_out++] = t->base.bos_id;

    if (t->sp_mode) {
        /* SentencePiece (llama/gemma families): greedy longest-piece match.
         * APPROXIMATION of true unigram Viterbi — fine for interactive chat,
         * may split words differently than llama-tokenize on rare inputs.
         * Spaces become '▁'(U+2581) first, matching how pieces are stored. */
        if (t->max_piece <= 0) return 0;
        char *mapped = (char *)malloc((size_t)len * 3 + 4);
        if (!mapped) return 0;
        int mlen = 0;
        /* add_dummy_prefix convention: virtual '▁' at text start */
        mapped[mlen++] = (char)0xE2;
        mapped[mlen++] = (char)0x96;
        mapped[mlen++] = (char)0x81;
        for (int i = 0; i < len; i++) {
            if (text[i] == ' ') {
                mapped[mlen++] = (char)0xE2;
                mapped[mlen++] = (char)0x96;
                mapped[mlen++] = (char)0x81;
            } else {
                mapped[mlen++] = text[i];
            }
        }
        while (pos < mlen && n_out < max_tokens) {
            int rem = mlen - pos;
            int L = rem < t->max_piece ? rem : t->max_piece;
            int best = -1;
            for (; L >= 1; L--) {
                best = hm_get(&t->tok2id, mapped + pos, L);
                if (best >= 0) break;
            }
            if (best < 0) {
                /* unknown byte -> <0xNN> fallback */
                char fb[8];
                const int fl = snprintf(fb, sizeof(fb), "<0x%02X>",
                                        (unsigned char)mapped[pos]);
                best = hm_get(&t->tok2id, fb, fl);
                L = 1;
            }
            if (best < 0) { pos++; continue; }   /* no fallback token: skip byte */
            /* SP convention: BOS (<s>) leads every sequence when defined */
            if (n_out == 0 && t->base.bos_id >= 0)
                out_tokens[n_out++] = t->base.bos_id;
            out_tokens[n_out++] = best;
            pos += L;
        }
        free(mapped);
        return n_out;
    }

    while (pos < len && n_out < max_tokens) {
        /* Special/control tokens have the literal form "<|name|>" and live in
         * the vocab table but NOT in the merge ranks, so the normal chunk+BPE
         * path can never produce them. Match them verbatim first, else chat
         * templates degrade into BPE pieces ([<][|][im][_][start][|][>]) and
         * the model sees a garbled prompt. */
        if (text[pos] == '<' && pos + 2 < len && text[pos + 1] == '|') {
            const char *close = NULL;
            for (const char *q = text + pos + 2; q + 1 < text + len; q++)
                if (q[0] == '|' && q[1] == '>') { close = q; break; }
            if (close) {
                const int slen = (int)(close - (text + pos)) + 2;
                if (slen < 128) {
                    char buf[128];
                    memcpy(buf, text + pos, (size_t)slen);
                    const int id = hm_get(&t->tok2id, buf, slen);
                    if (id >= 0) {
                        out_tokens[n_out++] = id;
                        pos += slen;
                        continue;
                    }
                }
            }
        }
        int clen = chunk_len(text + pos, len - pos);
        if (clen <= 0) clen = 1;
        /* A punct/symbol run can swallow the start of a following special
         * token ("hi!" -> chunk "!<|"). Cut the chunk right before "<|"
         * so the next iteration's verbatim special-token match can fire. */
        for (int k = 1; k < clen; k++) {
            if (text[pos + k] == '<' && pos + k + 1 < len &&
                text[pos + k + 1] == '|') { clen = k; break; }
        }
        const char *chunk = text + pos;
        pos += clen;

        /* map raw chunk bytes into the BPE unicode domain (space -> 'Ġ' etc.) */
        char mapped[MAX_SYM_BYTES * 4 + 4];
        int mlen = 0;
        for (int i = 0; i < clen; i++)
            mlen += utf8_enc(t->byte2cp[(unsigned char)chunk[i]], mapped + mlen);

        /* seed one symbol per mapped CHARACTER (not per byte): multi-byte
         * UTF-8 encodings of mapped codepoints act as atomic units */
        char syms[MAX_SYM_BYTES][MAX_SYM_BYTES];
        int slen[MAX_SYM_BYTES], nsym = 0;
        for (int i = 0; i < mlen && nsym < MAX_SYM_BYTES; ) {
            unsigned int cp;
            int n = utf8_dec(mapped + i, mlen - i, &cp);
            if (n < 0) n = 1;
            if (nsym >= MAX_SYM_BYTES) break;
            memcpy(syms[nsym], mapped + i, (size_t)n);
            slen[nsym] = n;
            nsym++;
            i += n;
        }

        /* repeatedly merge the adjacent pair with lowest merge rank */
        for (;;) {
            int best_rank = INT_MAX, best_j = -1;
            for (int j = 0; j + 1 < nsym; j++) {
                char cat[2 * MAX_SYM_BYTES];
                memcpy(cat, syms[j], (size_t)slen[j]);
                memcpy(cat + slen[j], syms[j + 1], (size_t)slen[j + 1]);
                const int r = hm_get(&t->pair_rank, cat, slen[j] + slen[j + 1]);
                if (r >= 0 && r < best_rank) { best_rank = r; best_j = j; }
            }
            if (best_j < 0) break;
            if (slen[best_j] + slen[best_j + 1] >= MAX_SYM_BYTES) break;
            memcpy(syms[best_j] + slen[best_j], syms[best_j + 1], (size_t)slen[best_j + 1]);
            slen[best_j] += slen[best_j + 1];
            memmove(syms[best_j + 1], syms[best_j + 2],
                    (size_t)(nsym - best_j - 2) * MAX_SYM_BYTES);
            memmove(slen + best_j + 1, slen + best_j + 2,
                    sizeof(int) * (size_t)(nsym - best_j - 2));
            nsym--;
        }

        /* map symbols to ids; unknown strings fall back to <0xNN> byte tokens */
        for (int j = 0; j < nsym; j++) {
            int id = hm_get(&t->tok2id, syms[j], slen[j]);
            if (id >= 0) {
                out_tokens[n_out++] = id;
            } else {
                for (int b = 0; b < slen[j]; b++) {
                    char fb[8];
                    const int fl = snprintf(fb, sizeof(fb), "<0x%02X>",
                                            (unsigned char)syms[j][b]);
                    id = hm_get(&t->tok2id, fb, fl);
                    if (id < 0 && n_out < max_tokens) continue;  /* no fallback token */
                    if (n_out >= max_tokens) goto done;
                    out_tokens[n_out++] = id;
                }
            }
            if (n_out >= max_tokens) goto done;
        }
    }
done:
    return n_out;
}

void bpe_tokenizer_free(BPETokenizer *tok_) {
    Tok *t = (Tok *)tok_;
    if (!tok_) return;
    if (tok_->tokens) {
        for (int i = 0; i < tok_->vocab_size; i++) free(tok_->tokens[i]);
        free(tok_->tokens);
    }
    free(tok_->token_lens);
    free(tok_->scores);
    hm_free(&t->pair_rank);
    hm_free(&t->tok2id);
    free(t);
}
