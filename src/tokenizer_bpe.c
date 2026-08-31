// Byte-level BPE tokenizer (M6 correctness rewrite).
//
// Fixes vs M5 version:
//  - Token strings stored VERBATIM from GGUF (the old ASCII filter destroyed
//    multibyte UTF-8 tokens, corrupting both decode output and encode matching).
//  - Encode is real BPE: lowest-rank adjacent-pair merges using
//    tokenizer.ggml.merges, not greedy longest-substring scan.
//  - Unknown bytes fall back to <0xNN> tokens instead of being dropped.
//
// Pre-tokenization (M6.1 correctness rewrite): full hand-rolled port of the
// llama.cpp unicode.cpp regex splitters over decoded codepoints:
//   pre="qwen2"  -> QWEN2 splitter  ((?i:contractions)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?punct+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)
//   pre="llama-bpe" etc -> LLAMA3 splitter (same shape, \p{N}{1,3} digit groups)
//                          + ignore_merges (whole-piece vocab hit skips BPE)
//   pre="smollm" -> two-pass: isolate every \p{N} codepoint, then GPT2 splitter
//   default      -> GPT2 splitter ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)
//   generic ggml fallback (no pre or "default") -> ggml-unicode.cpp verbatim:
//          [\p{L}\p{N}]+  (maximal letter/number run) + byte fallback
//          i.e. each \p{L} / \p{N} cluster is one piece, every other codepoint
//          (including whitespace/punct/emoji) is its own piece; unknown bytes
//          later fall back to <0xNN>. This mirrors ggml/src/ggml-unicode.cpp
//          regex splitting used for the generic byte-level BPE path.
// \p{L}/\p{N}/\s classification comes from the oracle's own unicode_ranges_flags
// table (src/tokenizer_uni_table.inc, generated from oracle/llama.cpp), so the
// classes match bit-for-bit including the UNDEFINED-bit punct-matchable rule.
// The \s+(?!\S) subtlety is replicated exactly: an interior whitespace run
// emits all but its last char; the trailing space attaches to the next piece.
//
// SP mode (ggml.model=="llama"/"ugm", e.g. gemma SPM vocabs): previously used
// greedy longest-piece matching; now implements the llama.cpp unigram Viterbi
// (llama-vocab.cpp:llm_tokenizer_ugm_session::tokenize and common/tokenize.cpp)
// scored by tokenizer.ggml.scores (min_score-10 penalty for <unk>), with exact
// backtrack and consecutive-<unk> merging. Covers SentencePiece / UGM models
// where greedy is known to diverge (e.g. "hello world" piece scores).
#include "tokenizer_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <math.h>
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

/* ---------------- BPE pre-tokenizer families (mirror llama-vocab.cpp) ------- */
enum {
    PRE_GPT2 = 0,   /* 's|'t|... GPT-2 pattern (gpt-2, phi-2, unknown pre) */
    PRE_LLAMA3,     /* llama-bpe / falcon3 / pixtral ... + ignore_merges */
    PRE_QWEN2,      /* qwen2: like llama3 but single-digit \p{N} pieces */
    PRE_SMOLLM,     /* smollm: \p{N} isolation pass, then GPT2 splitter */
    PRE_GGML,       /* ggml-unicode.cpp fallback: [\p{L}\p{N}]+ + byte fallback */
};

/* ---------------- tokenizer object ----------------
 * Reuses BPETokenizer layout from the header and adds side tables via
 * internal struct extension (header fields stay ABI-stable). */
/* GPT-2 byte-to-unicode mapping used by byte-level BPE vocabs:
 * printable ASCII/Latin-1 ranges map to themselves; the remaining bytes map
 * to codepoints starting at U+0100 (e.g. 0x20 space -> U+0120 'Ġ').
 * GGUF token strings for Qwen-style models live in THIS domain, so encode
 * must map text into it and decode must map back out. */
/* Special-token table entry: verbatim "<...>" / "<|...|>" / "<|...>" / "<...|>"
 * span that the encode path must emit as a SINGLE token id instead of
 * splitting through BPE. The whole-table mirror of llama-vocab.cpp's
 * `special_tokens` map (a parallel control-token string→id store lifted out
 * of the regular vocab so the BPE / SP paths know to short-circuit it). */
typedef struct {
    const char *str;   /* points into base.tokens[id]; not owned           */
    int id;
    int n;             /* strlen(str) cached for fast compares              */
} tt_special;

typedef struct {
    BPETokenizer base;
    HashMap tok2id;      /* token bytes -> id */
    HashMap pair_rank;   /* merged-bytes -> merge rank */
    int sp_mode;         /* SentencePiece vocab ('llama'/'gemma4' ggml models) */
    int pre_type;        /* BPE pre-tokenizer family (see PRE_* below) */
    int ignore_merges;   /* llama3-style: whole-piece vocab hit skips BPE */
    int pre_add_bos;     /* tokenizer.ggml.pre implies BOS (llama3-style BPE) */
    int add_bos;         /* resolved add-BOS convention */
    int kv_add_bos_seen; /* explicit add_bos KV present (overrides defaults) */
    int max_piece;       /* longest vocab piece in bytes */
    /* Longest-match special-token table: populated at init time by scanning
     * the vocab for entries that look like control tokens (contain '<' and
     * '>', no SP/byte-encoding artifacts), sorted by descending length so the
     * encode scan picks the longest span first. Owned strings live in
     * base.tokens[i]; we only mirror pointers. */
    tt_special *specials;
    int n_specials;
    int specials_cap;
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

/* ---------------- special-token table (verbatim "<...>" spans) -----------
 * Build a flat list of vocab entries that look like control tokens. The
 * encode path scans input text for the longest matching entry starting at
 * each position and emits its id verbatim (skipping BPE / SP splitting).
 *
 * Heuristic: a vocab string is a "control token" if it
 *   - contains '<' and '>' (covers <bos>, <eos>, <start_of_turn>, <|im_start|>,
 *     <|turn|>, <turn|>, <|tool|>, <tool|>, <|think|>, <|"|>, ...),
 *   - is short (<= 64 bytes — long control spans are not a thing in any
 *     current model; cap keeps the per-call longest-match scan O(64) worst
 *     case),
 *   - contains NO 0xE2 0x96 0x81 ('▁' SP marker), NO <0xNN> byte-fallback
 *     marker, and NO >64 ASCII chars (so we don't accidentally sweep
 *     user-text strings like "<3 hearts" into the table).
 *
 * "<0xNN>" byte-fallback strings DO match '<...>' and contain '<' and '>'.
 * They are valid vocab entries the encode path emits for unknown bytes, so
 * they must stay reachable as a fallback — but they should NOT be treated as
 * user-emittable special tokens (the BPE/SP path already produces them at
 * the right time). Filter them by the leading "<0x" / "<0X" prefix.
 *
 * Longest-match-first ordering is required: e.g. "<|tool_response|>" must
 * be preferred over "<|tool|>" when both prefixes match. We sort the
 * finished table descending by length, then resolve ties by insertion
 * order (stable, lower id wins) so behavior is deterministic. */
static int looks_like_control_token(const char *s, int n) {
    if (n <= 0 || n > 64) return 0;
    int has_lt = 0, has_gt = 0;
    for (int i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '<') has_lt = 1;
        else if (c == '>') has_gt = 1;
        /* reject SP '▁' (0xE2 0x96 0x81) and BPE 'Ġ' (0xC4 0xA0) */
        if (c == 0xE2 || c == 0xC4) return 0;
    }
    if (!has_lt || !has_gt) return 0;
    /* reject "<0xNN>" / "<0xNNN>" byte-fallback markers (3+ hex digits) */
    if (n >= 6 && s[0] == '<' && s[1] == '0' &&
        (s[2] == 'x' || s[2] == 'X') && s[n - 1] == '>') {
        for (int i = 3; i < n - 1; i++) {
            const char ch = s[i];
            if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F')))
                return 1;
        }
        return 0;
    }
    return 1;
}

static void specials_push(Tok *t, const char *s, int n, int id) {
    if (t->n_specials == t->specials_cap) {
        t->specials_cap = t->specials_cap ? t->specials_cap * 2 : 16;
        t->specials = (tt_special *)realloc(t->specials,
                                            sizeof(tt_special) * (size_t)t->specials_cap);
    }
    t->specials[t->n_specials].str = s;
    t->specials[t->n_specials].id = id;
    t->specials[t->n_specials].n = n;
    t->n_specials++;
}

static int specials_cmp_len(const void *a, const void *b) {
    const tt_special *x = (const tt_special *)a, *y = (const tt_special *)b;
    if (y->n != x->n) return y->n - x->n;   /* descending by length */
    return x->id - y->id;                    /* stable: lower id first */
}

static void specials_build(Tok *t) {
    for (int i = 0; i < t->base.vocab_size; i++) {
        const char *s = t->base.tokens[i];
        const int n = t->base.token_lens[i];
        if (s && looks_like_control_token(s, n))
            specials_push(t, s, n, i);
    }
    /* longest-first so the encode scan can short-circuit on the first match */
    qsort(t->specials, (size_t)t->n_specials, sizeof(tt_special), specials_cmp_len);
}

/* Longest-match lookup at text[pos..pos+maxlen). Returns the special id
 * whose string prefix-matches the input, or -1. Uses linear scan over the
 * length-sorted table — for <1000 entries the inner loop stays well under
 * 1us. If multiple specials share a length (rare; e.g. "<bos>" vs "<eos>"),
 * memcmp breaks the tie deterministically. */
static int specials_match(const Tok *t, const char *text, int len, int pos) {
    int best_id = -1, best_n = -1;
    for (int i = 0; i < t->n_specials; i++) {
        const tt_special *s = &t->specials[i];
        if (s->n > len - pos) continue;
        if (s->n <= best_n) continue;          /* can't beat current best */
        if (memcmp(text + pos, s->str, (size_t)s->n) == 0) {
            best_id = s->id;
            best_n = s->n;
        }
    }
    return best_id;
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
    t->pre_type = PRE_GPT2;
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
            /* pick the exact regex-splitter family (see llama-vocab.cpp switch) */
            if (strcmp(pre, "qwen2") == 0 || strcmp(pre, "deepseek-r1-qwen") == 0 ||
                strcmp(pre, "kormo") == 0 || strcmp(pre, "f2llmv2") == 0 ||
                strcmp(pre, "megrez") == 0)
                t->pre_type = PRE_QWEN2;
            else if (t->pre_add_bos && strcmp(pre, "tekken") != 0 &&
                     strcmp(pre, "chameleon") != 0)
                t->pre_type = PRE_LLAMA3;   /* tekken/chameleon use own patterns: approximated as GPT2 */
            else if (strcmp(pre, "smollm") == 0)
                t->pre_type = PRE_SMOLLM;
            else if (strcmp(pre, "default") == 0 || strcmp(pre, "none") == 0 ||
                     strcmp(pre, "gpt-2") == 0 || strcmp(pre, "grok") == 0)
                t->pre_type = PRE_GGML; /* ggml-unicode.cpp: [\p{L}\p{N}]+ + byte fallback */
            else
                t->pre_type = PRE_GPT2;
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
    t->ignore_merges = (t->pre_type == PRE_LLAMA3);
    if (!t->kv_add_bos_seen)
        t->add_bos = t->sp_mode ? 1 : t->pre_add_bos;
    specials_build(t);
    printf("[BPE] loaded: vocab=%d merges=%llu bos=%d eos=%d add_bos=%d specials=%d\n",
           t->base.vocab_size, (unsigned long long)n_merges,
           t->base.bos_id, t->base.eos_id, t->add_bos, t->n_specials);
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

/* ---------------- unicode codepoint classification ----------------
 * Exact copy of llama.cpp's unicode_ranges_flags semantics (categories +
 * whitespace bit), generated from oracle/llama.cpp/src/unicode-data.cpp.
 * Bits: 0x01 UNDEFINED, 0x02 NUMBER (\p{N}), 0x04 LETTER (\p{L}),
 * 0x08 SEPARATOR(\p{Z}), 0x10 \p{M}, 0x20 \p{P}, 0x40 \p{S}, 0x80 \p{C},
 * 0x100 WHITESPACE (\s). as_uint()!=0 <=> "defined"; UNDEFINED-only
 * codepoints still match [^\s\p{L}\p{N}] in the reference splitters. */
typedef struct { unsigned int start, end; unsigned short flags; } UniRange;
static const UniRange k_uni_ranges[] = {
#include "tokenizer_uni_table.inc"
};
#define UFF_NUMBER 0x002
#define UFF_LETTER 0x004
#define UFF_WS     0x100

static unsigned short uni_flags(unsigned int cp) {
    int lo = 0, hi = (int)(sizeof(k_uni_ranges) / sizeof(k_uni_ranges[0])) - 1;
    if (cp > k_uni_ranges[hi].end) return 0x0001;   /* UNDEFINED */
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const UniRange *r = &k_uni_ranges[mid];
        if (cp < r->start) hi = mid - 1;
        else if (cp > r->end) lo = mid + 1;
        else return r->flags;
    }
    return 0x0001;
}

/* ---------------- regex-equivalent pre-tokenizers ----------------
 * Hand-rolled ports of llama.cpp unicode.cpp custom splitters. Each scans a
 * half-open codepoint range [lo,hi) and records piece END offsets (cpt units).
 * Out-of-range lookups return cpt=OOR_CP / flags=0 exactly like the oracle's
 * _get_cpt/_get_flags guards. */
#define OOR_CP 0xFFFFFFFFu

typedef struct {
    const uint32_t *cp;
    int lo, hi;
    int *ends; int ne, max_ends;
    int prev;   /* last emitted end; invariant: prev == pos at loop top */
} Splitter;

static uint32_t sp_cpt(const Splitter *S, int pos) {
    return (pos >= S->lo && pos < S->hi) ? S->cp[pos] : OOR_CP;
}
static unsigned short sp_flags(const Splitter *S, int pos) {
    return (pos >= S->lo && pos < S->hi) ? uni_flags(S->cp[pos]) : 0;
}
static void sp_add(Splitter *S, int end) {
    if (end > S->prev && S->ne < S->max_ends) S->ends[S->ne++] = end;
    S->prev = end;
}
static uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* GPT2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * ci_contractions=0 for plain gpt2 pre; oracle's custom impl is shared. */
static void split_gpt2(Splitter *S, int ci_contractions) {
    for (int pos = S->lo; pos < S->hi; ) {
        const uint32_t cpt = sp_cpt(S, pos);

        /* contractions (lowercase-only unless case-insensitive variant) */
        if (cpt == '\'' && pos + 1 < S->hi) {
            uint32_t nx = sp_cpt(S, pos + 1);
            if (ci_contractions) nx = ascii_lower(nx);
            if (nx == 's' || nx == 't' || nx == 'm' || nx == 'd') { pos += 2; sp_add(S, pos); continue; }
            if (pos + 2 < S->hi) {
                uint32_t n2 = sp_cpt(S, pos + 2);
                if (ci_contractions) n2 = ascii_lower(n2);
                if ((nx == 'r' && n2 == 'e') || (nx == 'v' && n2 == 'e') ||
                    (nx == 'l' && n2 == 'l')) { pos += 3; sp_add(S, pos); continue; }
            }
        }

        /* flags of the optional-prefix position: cpt==' ' looks one ahead */
        const unsigned short f2 = (cpt == ' ') ? sp_flags(S, pos + 1) : sp_flags(S, pos);
        /*  ?\p{L}+ */
        if (f2 & UFF_LETTER) {
            pos += (cpt == ' ');
            while (sp_flags(S, pos) & UFF_LETTER) pos++;
            sp_add(S, pos); continue;
        }
        /*  ?\p{N}+ */
        if (f2 & UFF_NUMBER) {
            pos += (cpt == ' ');
            while (sp_flags(S, pos) & UFF_NUMBER) pos++;
            sp_add(S, pos); continue;
        }
        /*  ?[^\s\p{L}\p{N}]+  (f2!=0 keeps UNDEFINED-bit codepoints matchable,
         * matching the oracle's flags2.as_uint() guard — emoji etc.) */
        if (!(f2 & (UFF_WS | UFF_LETTER | UFF_NUMBER)) && f2) {
            pos += (cpt == ' ');
            unsigned short g;
            while ((g = sp_flags(S, pos)) && !(g & (UFF_WS | UFF_LETTER | UFF_NUMBER))) pos++;
            sp_add(S, pos); continue;
        }

        /* whitespace run */
        int nw = 0;
        while (sp_flags(S, pos + nw) & UFF_WS) nw++;
        /* \s+(?!\S): interior run keeps its last space for the next piece */
        if (nw > 1 && sp_cpt(S, pos + nw) != OOR_CP) { pos += nw - 1; sp_add(S, pos); continue; }
        /* \s+ */
        if (nw > 0) { pos += nw; sp_add(S, pos); continue; }

        /* no matches: single codepoint piece */
        pos++; sp_add(S, pos);
    }
}

/* LLAMA3/QWEN2:
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3 or 1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ */
static void split_llama3(Splitter *S, int digit_group) {
    for (int pos = S->lo; pos < S->hi; ) {
        const uint32_t cpt = sp_cpt(S, pos);
        const unsigned short f = sp_flags(S, pos);

        /* (?i:contractions) */
        if (cpt == '\'' && pos + 1 < S->hi) {
            const uint32_t nx = ascii_lower(sp_cpt(S, pos + 1));
            if (nx == 's' || nx == 't' || nx == 'm' || nx == 'd') { pos += 2; sp_add(S, pos); continue; }
            if (pos + 2 < S->hi) {
                const uint32_t n2 = ascii_lower(sp_cpt(S, pos + 2));
                if ((nx == 'r' && n2 == 'e') || (nx == 'v' && n2 == 'e') ||
                    (nx == 'l' && n2 == 'l')) { pos += 3; sp_add(S, pos); continue; }
            }
        }

        /* [^\r\n\p{L}\p{N}]?\p{L}+: any single non-cr/lf/number prefix */
        if (!(cpt == '\r' || cpt == '\n' || (f & UFF_NUMBER))) {
            if ((f & UFF_LETTER) || (sp_flags(S, pos + 1) & UFF_LETTER)) {
                pos++;
                while (sp_flags(S, pos) & UFF_LETTER) pos++;
                sp_add(S, pos); continue;
            }
        }

        /* \p{N}{1,3} (llama3) or \p{N} (qwen2) */
        if (f & UFF_NUMBER) {
            if (digit_group <= 1) { pos++; sp_add(S, pos); continue; }
            int ini = pos;
            while (sp_flags(S, pos) & UFF_NUMBER) {
                if (++pos - ini >= 3) { sp_add(S, pos); ini = pos; }
            }
            sp_add(S, pos); continue;
        }

        /* <space>?[^\s\p{L}\p{N}]+[\r\n]* */
        const unsigned short f2 = (cpt == ' ') ? sp_flags(S, pos + 1) : f;
        if (!(f2 & (UFF_WS | UFF_LETTER | UFF_NUMBER)) && f) {
            pos += (cpt == ' ');
            unsigned short g;
            while ((g = sp_flags(S, pos)) && !(g & (UFF_WS | UFF_LETTER | UFF_NUMBER))) pos++;
            while (sp_cpt(S, pos) == '\r' || sp_cpt(S, pos) == '\n') pos++;
            sp_add(S, pos); continue;
        }

        /* whitespace run; remember end of last CR/LF inside it */
        int nw = 0, last_rn = 0;
        while (sp_flags(S, pos + nw) & UFF_WS) {
            const uint32_t c2 = sp_cpt(S, pos + nw);
            if (c2 == '\r' || c2 == '\n') last_rn = pos + nw + 1;
            nw++;
        }
        /* \s*[\r\n]+ */
        if (last_rn > 0) { pos = last_rn; sp_add(S, pos); continue; }
        /* \s+(?!\S) */
        if (nw > 1 && sp_cpt(S, pos + nw) != OOR_CP) { pos += nw - 1; sp_add(S, pos); continue; }
        /* \s+ */
        if (nw > 0) { pos += nw; sp_add(S, pos); continue; }
        pos++; sp_add(S, pos);
    }
}
/* ggml-unicode.cpp fallback: [\p{L}\p{N}]+ + byte fallback (verbatim).
 * Each maximal [\p{L}\p{N}]+ run is one piece; every other codepoint
 * (space, punct, emoji, etc.) is isolated as its own piece. Unknown bytes
 * are handled later by <0xNN> fallback during symbol->id mapping.
 * This matches ggml/src/ggml-unicode.cpp's simplest pre-tokenizer used
 * when no family-specific regex applies (generic BPE / default pre). */
static void split_ggml(Splitter *S) {
    for (int pos = S->lo; pos < S->hi; ) {
        const unsigned short f = sp_flags(S, pos);
        if ((f & (UFF_LETTER | UFF_NUMBER))) {
            int end = pos + 1;
            while (end < S->hi && (sp_flags(S, end) & (UFF_LETTER | UFF_NUMBER))) end++;
            pos = end;
            sp_add(S, pos);
            continue;
        }
        /* single-codepoint fallback: whitespace, punct, symbol, control, or
         * UNDEFINED (emoji etc. still has some flag bit, but we isolate it) */
        pos++;
        sp_add(S, pos);
    }
}

/* dispatch per family; returns number of piece-end offsets written */
static int pretok_split(const Tok *t, const uint32_t *cp, int n, int *ends, int max_ends) {
    Splitter S = { cp, 0, n, ends, 0, max_ends, 0 };
    switch (t->pre_type) {
        case PRE_QWEN2:  split_llama3(&S, 1); return S.ne;
        case PRE_LLAMA3: split_llama3(&S, 3); return S.ne;
        case PRE_SMOLLM: {
            /* pass 1: std::regex fallback on \p{N} isolates every digit into
             * its own piece (alternating digit / non-digit runs) ... */
            int p = 0;
            while (p < n) {
                int q = p;
                if (uni_flags(cp[p]) & UFF_NUMBER) q++;                     /* matched digit */
                else while (q < n && !(uni_flags(cp[q]) & UFF_NUMBER)) q++; /* unmatched gap  */
                /* pass 2: GPT2 splitter re-splits within each piece */
                Splitter T = { cp, p, q, ends, S.ne, max_ends, p };
                split_gpt2(&T, 0);
                S.ne = T.ne;
                p = q;
            }
            return S.ne;
        }
        case PRE_GGML:   split_ggml(&S); return S.ne;
        default:         split_gpt2(&S, 0); return S.ne;
    }
}

/* ---------------- BPE encode helpers ---------------- */

/* Encode one non-special segment: decode codepoints, split with the family
 * pre-tokenizer, then run rank-based BPE merges inside each piece.
 * Workspace buffers are caller-provided and sized to the full text. */
static int encode_bpe_segment(Tok *t, const char *s, int slen,
                              uint32_t *cps, int *bofs, int *ends,
                              char *mapped, int *beg, int *sln,
                              int *out, int cap) {
    /* decode UTF-8 -> codepoints (+ byte offsets); invalid sequences become
     * U+FFFD, matching oracle unicode_cpts_from_utf8 */
    int nc = 0;
    for (int i = 0; i < slen; ) {
        unsigned int cp;
        int n = utf8_dec(s + i, slen - i, &cp);
        if (n < 0) { n = 1; cp = 0xFFFD; }
        cps[nc] = cp; bofs[nc] = i; nc++; i += n;
    }
    bofs[nc] = slen;
    if (nc == 0) return 0;

    const int npieces = pretok_split(t, cps, nc, ends, nc);

    int n_out = 0;
    int p0 = 0;
    for (int k = 0; k <= npieces; k++) {
        const int p1 = (k < npieces) ? ends[k] : nc;
        if (p1 <= p0) { p0 = p1; continue; }

        /* map raw piece bytes into the BPE unicode domain (space -> 'Ġ') */
        const int bstart = bofs[p0], bendv = bofs[p1];
        int mlen = 0;
        for (int b = bstart; b < bendv; b++)
            mlen += utf8_enc(t->byte2cp[(unsigned char)s[b]], mapped + mlen);

        /* seed one symbol per mapped CHARACTER; symbols are contiguous slices
         * of `mapped`, so adjacent-pair concatenation is itself contiguous */
        int nsym = 0;
        for (int i = 0; i < mlen; ) {
            unsigned int cp;
            int n = utf8_dec(mapped + i, mlen - i, &cp);
            if (n < 0) n = 1;
            beg[nsym] = i; sln[nsym] = n; nsym++;
            i += n;
        }

        /* llama3-style ignore_merges: whole-piece vocab hit skips BPE */
        if (t->ignore_merges && nsym > 1) {
            const int wid = hm_get(&t->tok2id, mapped, mlen);
            if (wid >= 0) {
                if (n_out >= cap) return n_out;
                out[n_out++] = wid;
                p0 = p1;
                continue;
            }
        }

        /* repeatedly merge the adjacent pair with lowest merge rank */
        for (;;) {
            int best_rank = INT_MAX, best_j = -1;
            for (int j = 0; j + 1 < nsym; j++) {
                const int r = hm_get(&t->pair_rank, mapped + beg[j], sln[j] + sln[j + 1]);
                if (r >= 0 && r < best_rank) { best_rank = r; best_j = j; }
            }
            if (best_j < 0) break;
            sln[best_j] += sln[best_j + 1];
            memmove(beg + best_j + 1, beg + best_j + 2, sizeof(int) * (size_t)(nsym - best_j - 2));
            memmove(sln + best_j + 1, sln + best_j + 2, sizeof(int) * (size_t)(nsym - best_j - 2));
            nsym--;
        }

        /* map symbols to ids; unknown strings fall back to <0xNN> byte tokens */
        for (int j = 0; j < nsym; j++) {
            const int id = hm_get(&t->tok2id, mapped + beg[j], sln[j]);
            if (id >= 0) {
                if (n_out >= cap) return n_out;
                out[n_out++] = id;
                continue;
            }
            int i = beg[j];
            const int iend = beg[j] + sln[j];
            while (i < iend) {
                unsigned int cp;
                int n = utf8_dec(mapped + i, iend - i, &cp);
                if (n < 0) { n = 1; cp = (unsigned char)mapped[i]; }
                const signed int rb = (cp < 32768) ? t->cp2byte[cp] : -1;
                char fb[8];
                const int fl = snprintf(fb, sizeof(fb), "<0x%02X>",
                                        rb >= 0 ? (unsigned)rb : (unsigned char)mapped[i]);
                const int fid = hm_get(&t->tok2id, fb, fl);
                if (fid >= 0) {
                    if (n_out >= cap) return n_out;
                    out[n_out++] = fid;
                }
                i += n;
            }
        }
        p0 = p1;
    }
    return n_out;
}

/* SP unigram Viterbi (llama.cpp: llm_tokenizer_ugm_session::tokenize).
 * Ports the SentencePiece optimized Viterbi verbatim:
 *  - normalized text is `mapped` (space -> U+2581 '▁', optional dummy prefix
 *    '▁' at sequence start, mirroring llama.cpp tokenize's add_dummy_prefix);
 *  - DP over byte offsets with double score_sum (log probs in ggml.scores);
 *    user-defined tokens score 0 (override), normal tokens use their score;
 *    unknown penalty = min_score - 10.0 as in llama-vocab.cpp;
 *  - per-UTF8-codepoint step, traversing the vocab trie via hash lookups for
 *    every substring up to max_piece (exact-score equivalent to trie walk);
 *  - if no vocab piece covers a single codepoint, fall back to byte token
 *    <0xNN> if present, else <unk> with penalty;
 *  - backtrack from n, merging consecutive <unk> (identical to oracle's
 *    reverse-pushing with is_prev_unknown guard).
 *  Byte fallback: single bytes that have no vocab cover emit <0xNN> when
 *  present, otherwise the <unk> penalty path guarantees exactly one token per
 *  undecodable byte, preventing greedy divergence on e.g. "hello world". */
static int utf8_len_from_byte(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; /* invalid -> 1 */
}
static int encode_sp_segment(Tok *t, const char *s, int slen, int with_dummy_prefix,
                             int *out, int cap) {
    if (t->max_piece <= 0 || cap <= 0) return 0;
    if (slen < 0) slen = 0;
    char *mapped = (char *)malloc((size_t)slen * 3 + 4);
    if (!mapped) return 0;
    int mlen = 0;
    if (with_dummy_prefix) {
        mapped[mlen++] = (char)0xE2;
        mapped[mlen++] = (char)0x96;
        mapped[mlen++] = (char)0x81;
    }
    for (int i = 0; i < slen; i++) {
        if (s[i] == ' ') {
            mapped[mlen++] = (char)0xE2;
            mapped[mlen++] = (char)0x96;
            mapped[mlen++] = (char)0x81;
        } else {
            mapped[mlen++] = s[i];
        }
    }
    if (mlen == 0) { free(mapped); return 0; }
    int n = mlen;
    /* score prep */
    double min_score = 1e30, max_score = -1e30;
    int have_scores = (t->base.scores != NULL);
    if (have_scores) {
        for (int i = 0; i < t->base.vocab_size; i++) {
            double sc = (double)t->base.scores[i];
            if (sc < min_score) min_score = sc;
            if (sc > max_score) max_score = sc;
        }
    }
    if (min_score > 1e29) { min_score = -10.0; max_score = 0.0; have_scores = 0; }
    double unk_score = min_score - 10.0;
    /* find <unk> id: vocab often has it at 0; search by string else default 0 */
    int unk_id = -1;
    int tmp = hm_get(&t->tok2id, "<unk>", 5);
    if (tmp >= 0) unk_id = tmp;
    else if (t->base.vocab_size > 0) {
        /* fallback: first token that looks like unk, else 0 */
        for (int i = 0; i < t->base.vocab_size; i++) {
            if (t->base.tokens[i] && strcmp(t->base.tokens[i], "<unk>") == 0) { unk_id = i; break; }
        }
        if (unk_id < 0) unk_id = 0;
    } else unk_id = 0;
    /* DP buffers: n+1 entries */
    double *best_score = (double *)malloc((size_t)(n + 1) * sizeof(double));
    int *best_prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *best_id = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!best_score || !best_prev || !best_id) {
        free(best_score); free(best_prev); free(best_id); free(mapped); return 0;
    }
    for (int i = 0; i <= n; i++) { best_score[i] = -1e100; best_prev[i] = -1; best_id[i] = -1; }
    best_score[0] = 0.0;
    /* Viterbi forward */
    for (int i = 0; i < n; ) {
        if (best_score[i] < -1e90) { /* unreachable, advance by one codepoint */
            int l = utf8_len_from_byte((unsigned char)mapped[i]);
            if (i + l > n) l = n - i;
            i += l;
            continue;
        }
        int n_units = utf8_len_from_byte((unsigned char)mapped[i]);
        if (i + n_units > n) n_units = n - i;
        /* guard: if invalid continuation bytes, n_units may overshoot; clamp to 1 */
        if (n_units < 1) n_units = 1;
        if (n_units > 4) n_units = 1;
        double cur = best_score[i];
        int single_found = 0;
        int max_l = t->max_piece;
        if (max_l > n - i) max_l = n - i;
        for (int l = 1; l <= max_l; l++) {
            int j = i + l;
            int id = hm_get(&t->tok2id, mapped + i, l);
            if (id < 0) continue;
            if (l == n_units) single_found = 1;
            double tok_score = 0.0;
            if (have_scores && id >= 0 && id < t->base.vocab_size) tok_score = (double)t->base.scores[id];
            /* user_defined tokens would score 0 in oracle; we lack token_type,
             * but they are rare and typically already have score 0 in GGUF */
            double cand = cur + tok_score;
            if (cand > best_score[j]) {
                best_score[j] = cand;
                best_prev[j] = i;
                best_id[j] = id;
            }
        }
        if (!single_found) {
            int j = i + n_units;
            /* try byte fallback <0xNN> before generic unk */
            char fb[8];
            int fl = snprintf(fb, sizeof(fb), "<0x%02X>", (unsigned char)mapped[i]);
            int bid = hm_get(&t->tok2id, fb, fl);
            if (bid >= 0) {
                double tok_score = 0.0;
                if (have_scores && bid >= 0 && bid < t->base.vocab_size) tok_score = (double)t->base.scores[bid];
                double cand = cur + tok_score;
                if (cand > best_score[j]) {
                    best_score[j] = cand;
                    best_prev[j] = i;
                    best_id[j] = bid;
                }
            } else {
                double cand = cur + unk_score;
                if (cand > best_score[j]) {
                    best_score[j] = cand;
                    best_prev[j] = i;
                    best_id[j] = unk_id;
                }
            }
        }
        /* advance to next codepoint */
        i += n_units;
    }
    /* if end unreachable (should not happen due to unk fallback), fallback to greedy byte walk */
    if (best_score[n] < -1e90) {
        free(best_score); free(best_prev); free(best_id); free(mapped);
        return 0;
    }
    /* backtrack, merging consecutive unk */
    int cur = n;
    int tmp_cap = n * 2 + 4;
    int *rev = (int *)malloc((size_t)tmp_cap * sizeof(int));
    int rev_n = 0;
    int is_prev_unk = 0;
    while (cur > 0) {
        int prev = best_prev[cur];
        int id = best_id[cur];
        if (prev < 0 || id < 0) break; /* safety */
        int is_unk = (id == unk_id);
        if (!(is_prev_unk && is_unk)) {
            if (rev_n < tmp_cap) rev[rev_n++] = id;
        }
        is_prev_unk = is_unk;
        cur = prev;
    }
    /* reverse into out */
    int n_out = 0;
    for (int i = rev_n - 1; i >= 0 && n_out < cap; i--) out[n_out++] = rev[i];
    free(rev);
    free(best_score); free(best_prev); free(best_id); free(mapped);
    return n_out;
}

int bpe_encode(const BPETokenizer *tok_, const char *text, int *out_tokens, int max_tokens) {
    Tok *t = (Tok *)tok_;
    if (!tok_ || !t->base.tokens || !text || !out_tokens || max_tokens <= 0) return 0;
    const int len = (int)strlen(text);
    int n_out = 0;
    int pos = 0;

    /* BPE-mode families (llama3 etc.) declare add_bos in GGUF metadata;
     * honor it exactly like the SP path so prompts match family convention.
     * The SP path auto-prepends BOS inside its segment loop, so we only
     * do the explicit prepend for BPE mode here. */
    if (!t->sp_mode && t->add_bos && t->base.bos_id >= 0 && max_tokens > 0)
        out_tokens[n_out++] = t->base.bos_id;

    /* Workspace sized to the full text (BPE path only; freed at exit). */
    uint32_t *cps = NULL;
    int *bofs = NULL, *ends = NULL;
    char *mapped = NULL;
    int *beg = NULL, *sln = NULL;
    if (!t->sp_mode) {
        cps    = (uint32_t *)malloc(((size_t)len + 1) * sizeof(uint32_t));
        bofs   = (int *)malloc(((size_t)len + 1) * sizeof(int));
        ends   = (int *)malloc(((size_t)len + 1) * sizeof(int));
        mapped = (char *)malloc((size_t)len * 4 + 16);
        beg    = (int *)malloc(((size_t)len * 4 + 4) * sizeof(int));
        sln    = (int *)malloc(((size_t)len * 4 + 4) * sizeof(int));
        if (!cps || !bofs || !ends || !mapped || !beg || !sln) {
            free(cps); free(bofs); free(ends); free(mapped); free(beg); free(sln);
            return n_out;
        }
    }

    while (pos < len && n_out < max_tokens) {
        /* Longest-match verbatim special-token span (e.g. <|turn|>, <|im_start|>,
         * <start_of_turn>, <bos>). Built at init from the vocab so the same
         * id the model was trained on is emitted as a single token instead of
         * being split into '<', '|', 'turn', '|', '>' byte-fallback pieces. */
        const int sp_id = specials_match(t, text, len, pos);
        if (sp_id >= 0) {
            /* resolve the matched specials entry's string length (cheap
             * linear scan over a small table) */
            int sl = 0;
            for (int i = 0; i < t->n_specials; i++)
                if (t->specials[i].id == sp_id) { sl = t->specials[i].n; break; }
            /* SP convention: emit a leading BOS only if the first token
             * isn't already a control token. The first-segment SP loop
             * below used to do this; we hoist it out so a literal "<|turn|>"
             * at position 0 doesn't double-BOS. */
            if (n_out == 0 && t->sp_mode && t->base.bos_id >= 0
                && sp_id != t->base.bos_id)
                out_tokens[n_out++] = t->base.bos_id;
            out_tokens[n_out++] = sp_id;
            pos += sl;
            continue;
        }

        /* Find the next '<...' special-token start so we can process the
         * non-special span in one chunk. If none, run to end of input. */
        int seg_end = len;
        for (int q = pos; q < len; q++) {
            if (text[q] == '<' && specials_match(t, text, len, q) >= 0) {
                seg_end = q; break;
            }
        }

        if (t->sp_mode) {
            /* SP path: greedy longest-piece on the raw span with '▁' mapping.
             * Only the first span gets add_dummy_prefix '▁' (mirrors
             * llama.cpp behavior: the virtual prefix is at sequence start).
             * The first-span BOS is also emitted here for the same reason. */
            const int is_first = (pos == 0);
            if (is_first && n_out == 0 && t->base.bos_id >= 0)
                out_tokens[n_out++] = t->base.bos_id;
            n_out += encode_sp_segment(t, text + pos, seg_end - pos, is_first,
                                       out_tokens + n_out, max_tokens - n_out);
        } else {
            n_out += encode_bpe_segment(t, text + pos, seg_end - pos,
                                        cps, bofs, ends, mapped, beg, sln,
                                        out_tokens + n_out, max_tokens - n_out);
        }
        pos = seg_end;
    }

    free(cps); free(bofs); free(ends); free(mapped); free(beg); free(sln);
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
    free(t->specials);
    hm_free(&t->pair_rank);
    hm_free(&t->tok2id);
    free(t);
}
