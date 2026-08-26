/* chat_template.c -- hand-written per-family chat prompt formatters.
 *
 * Transcribed from official HF chat_template Jinja (see chat_template.h
 * header for sources). Deterministic: no locale, no time calls, no
 * allocation; single output buffer with snprintf return semantics.
 *
 * Per-family shapes (add_generation_prompt = 1):
 *
 * qwen2/qwen3 (ChatML):
 *   [<|im_start|>system\n{sys}<|im_end|>\n]
 *   <|im_start|>user\n{u}<|im_end|>\n
 *   <|im_start|>assistant\n{a}<|im_end|>\n ...
 *   <|im_start|>assistant\n            <- generation prompt
 *   qwen3 extras: <think>..</think> stripped from assistant history
 *   unless opts.keep_think; opts.add_empty_think appends an empty think
 *   block after the generation prompt (Qwen3 enable_thinking=false).
 *
 * gemma/gemma2/gemma4:
 *   <bos>
 *   <start_of_turn>user\n[{sys}\n\n]{u}<end_of_turn>\n     <- system folded
 *   <start_of_turn>model\n{a}<end_of_turn>\n                  into 1st user
 *   <start_of_turn>model\n                                    turn (HF does
 *   this too: system role unsupported, prepended as first_user_prefix).
 *   Content is |trim'd per template.
 *
 * llama3 ("llama" arch):
 *   [<|begin_of_text|>]
 *   [<|start_header_id|>system<|end_header_id|>\n\n{dates}{sys}<|eot_id|>]
 *   <|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>
 *   ...
 *   <|start_header_id|>assistant<|end_header_id|>\n\n
 *   DEVIATION vs Llama-3.1/3.2 HF template: the "Cutting Knowledge Date"
 *   / "Today Date:" preamble lines are emitted only when opts.date_string
 *   is set (template default "26 Jul 2024"); NULL omits them entirely for
 *   determinism (Llama-3.0 has no date lines). Content |trim'd.
 */

#include "chat_template.h"

#include <stdio.h>
#include <string.h>

/* ---- tiny append buffer with snprintf-style length accounting ---------- */

typedef struct {
    char  *out;
    size_t cap;    /* usable bytes incl. NUL */
    size_t len;    /* bytes written so far (may exceed cap -> truncated) */
    int    truncated;
} abuf;

static void ab_putn(abuf *b, const char *s, size_t n) {
    if (!s || n == 0) return;
    if (b->len + 1 < b->cap) {           /* room for at least 1 byte + NUL */
        size_t room = b->cap - b->len - 1;
        size_t w = n < room ? n : room;
        memcpy(b->out + b->len, s, w);
        if (w < n) b->truncated = 1;
    } else if (b->len < b->cap) {
        b->truncated = 1;
    }
    b->len += n;
}

static void ab_puts(abuf *b, const char *s) {
    if (s) ab_putn(b, s, strlen(s));
}

/* ---- helpers ------------------------------------------------------------ */

static const char *safe(const char *s) { return s ? s : ""; }

static size_t trim_span(const char *s, size_t n, const char **begin) {
    size_t a = 0, z = n;
    while (a < z && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' ||
                     s[a] == '\n')) a++;
    while (z > a && (s[z - 1] == ' ' || s[z - 1] == '\t' ||
                     s[z - 1] == '\r' || s[z - 1] == '\n')) z--;
    *begin = s + a;
    return z - a;
}

/* strip "<think>...</think>" blocks (and an unterminated trailing
 * "<think>...") from content, in place up to one level of nesting-free
 * scanning; returns a malloc-free view via begin/len out-params into buf when
 * stripping occurred (buf must live as long as the view), else points
 * at the original string. */
#define THINK_OPEN  "<think>"
#define THINK_CLOSE "</think>"

static size_t strip_think(const char *s, char *buf, size_t bufsz,
                          const char **begin) {
    size_t n = strlen(s);
    const char *p = strstr(s, THINK_OPEN);
    if (!p) { *begin = s; return n; }

    size_t head = (size_t)(p - s);
    if (head >= bufsz) head = bufsz ? bufsz - 1 : 0;
    memcpy(buf, s, head);
    size_t w = head;

    while (p) {
        p += strlen(THINK_OPEN);                       /* skip open tag */
        const char *close = strstr(p, THINK_CLOSE);
        if (!close) break;                             /* unterminated: drop rest */
        p = close + strlen(THINK_CLOSE);               /* resume after block */
        const char *next = strstr(p, THINK_OPEN);
        size_t chunk = next ? (size_t)(next - p) : strlen(p);
        if (w + chunk >= bufsz) chunk = bufsz > w + 1 ? bufsz - w - 1 : 0;
        memcpy(buf + w, p, chunk);
        w += chunk;
        p = next;
    }
    buf[w] = '\0';
    *begin = buf;
    return w;
}

/* ---- family mapping ------------------------------------------------------ */

int tt_chat_family_from_arch(const char *arch) {
    if (!arch) return -1;
    if (!strcmp(arch, "qwen2"))  return TT_CHAT_QWEN2;
    if (!strcmp(arch, "qwen3"))  return TT_CHAT_QWEN3;
    if (!strcmp(arch, "gemma") || !strcmp(arch, "gemma2")) return TT_CHAT_GEMMA;
    if (!strcmp(arch, "gemma4")) return TT_CHAT_GEMMA4;
    if (!strcmp(arch, "llama"))  return TT_CHAT_LLAMA3;
    return -1;
}

const char *tt_chat_stop_string(tt_chat_family fam) {
    switch (fam) {
    case TT_CHAT_QWEN2:
    case TT_CHAT_QWEN3:   return "<|im_end|>";
    case TT_CHAT_GEMMA:
    case TT_CHAT_GEMMA4:  return "<end_of_turn>";
    case TT_CHAT_LLAMA3:  return "<|eot_id|>";
    default:              return NULL;
    }
}

tt_chat_opts tt_chat_opts_default(void) {
    tt_chat_opts o;
    o.add_generation_prompt = 1;
    o.keep_think      = 0;
    o.add_empty_think = 0;
    o.add_bos_text    = 1;
    o.date_string     = NULL;
    return o;
}

/* ---- per-family renderers ------------------------------------------------ */

static void fmt_qwen(abuf *b, const tt_msg *msgs, int n,
                     const tt_chat_opts *o, int is_qwen3) {
    char think_buf[TT_CHAT_MAX_CONTENT];
    for (int i = 0; i < n; i++) {
        const char *role = safe(msgs[i].role);
        /* system only as first message; later system msgs render like user */
        if (!strcmp(role, "system") && i != 0) role = "user";

        const char *content = safe(msgs[i].content);
        const char *view = content;
        if (is_qwen3 && !o->keep_think && !strcmp(role, "assistant"))
            strip_think(content, think_buf, sizeof(think_buf), &view);

        ab_puts(b, "<|im_start|>");
        ab_puts(b, role);
        ab_puts(b, "\n");
        ab_puts(b, view);
        ab_puts(b, "<|im_end|>\n");
    }
    if (o->add_generation_prompt) {
        ab_puts(b, "<|im_start|>assistant\n");
        if (is_qwen3 && o->add_empty_think)
            ab_puts(b, "<think>\n\n</think>\n\n");
    }
}

static void fmt_gemma(abuf *b, const tt_msg *msgs, int n,
                      const tt_chat_opts *o) {
    /* gemma's SP-mode tokenizer auto-prepends bos_id=2 in bpe_encode;
     * emitting the literal "<bos>" string here would be mis-tokenized
     * (BPE splits "<bos>" into '<', 'bos', '>'), garbling the prompt prefix.
     * Always skip for gemma regardless of opts->add_bos_text. */
    (void)o;

    /* HF gemma template folds a leading system message into the first
     * user turn's prefix ({system}\n\n{user}); roles must alternate. */
    char sys_prefix[TT_CHAT_MAX_CONTENT];
    sys_prefix[0] = '\0';
    int start = 0;
    if (n > 0 && !strcmp(safe(msgs[0].role), "system")) {
        snprintf(sys_prefix, sizeof(sys_prefix), "%s\n\n", safe(msgs[0].content));
        start = 1;
    }

    for (int i = start; i < n; i++) {
        const char *role = strcmp(safe(msgs[i].role), "assistant") == 0
                               ? "model" : safe(msgs[i].role);
        ab_puts(b, "<start_of_turn>");
        ab_puts(b, role);
        ab_puts(b, "\n");
        if (i == start && i > 0) ab_puts(b, sys_prefix); /* fold system */
        const char *s = safe(msgs[i].content);
        const char *t;
        size_t tl = trim_span(s, strlen(s), &t);
        ab_putn(b, t, tl);
        ab_puts(b, "<end_of_turn>\n");
    }
    if (o->add_generation_prompt)
        ab_puts(b, "<start_of_turn>model\n");
}

static void fmt_llama3(abuf *b, const tt_msg *msgs, int n,
                       const tt_chat_opts *o) {
    if (o->add_bos_text) ab_puts(b, "<|begin_of_text|>");

    /* System block: emitted when a system message exists or a date string
     * is configured (the HF 3.1/3.2 template always emits the header; the
     * 3.0 template skips the block entirely without a system message). */
    int has_sys = n > 0 && !strcmp(safe(msgs[0].role), "system");
    if (has_sys || o->date_string) {
        ab_puts(b, "<|start_header_id|>system<|end_header_id|>\n\n");
        if (o->date_string && o->date_string[0]) {
            ab_puts(b, "Cutting Knowledge Date: December 2023\n");
            ab_puts(b, "Today Date: ");
            ab_puts(b, o->date_string);
            ab_puts(b, "\n\n");
        }
        if (has_sys) {
            const char *s = safe(msgs[0].content);
            const char *t;
            size_t tl = trim_span(s, strlen(s), &t);   /* |trim per template */
            ab_putn(b, t, tl);
        }
        ab_puts(b, "<|eot_id|>");
    }

    for (int i = has_sys ? 1 : 0; i < n; i++) {
        ab_puts(b, "<|start_header_id|>");
        ab_puts(b, safe(msgs[i].role));
        ab_puts(b, "<|end_header_id|>\n\n");
        const char *s = safe(msgs[i].content);
        const char *t;
        size_t tl = trim_span(s, strlen(s), &t);       /* |trim per template */
        ab_putn(b, t, tl);
        ab_puts(b, "<|eot_id|>");
    }
    if (o->add_generation_prompt)
        ab_puts(b, "<|start_header_id|>assistant<|end_header_id|>\n\n");
}

/* ---- public API ---------------------------------------------------------- */

int tt_chat_format_ex(tt_chat_family fam, const tt_msg *msgs, int n,
                      const tt_chat_opts *opts, char *out, size_t cap) {
    if (!out || cap == 0 || (n > 0 && !msgs)) return -2;
    if (fam < TT_CHAT_QWEN2 || fam > TT_CHAT_LLAMA3) return -1;
    tt_chat_opts dflt;
    if (!opts) { dflt = tt_chat_opts_default(); opts = &dflt; }

    abuf b = { out, cap, 0, 0 };
    switch (fam) {
    case TT_CHAT_QWEN2: fmt_qwen(&b, msgs, n, opts, 0); break;
    case TT_CHAT_QWEN3: fmt_qwen(&b, msgs, n, opts, 1); break;
    case TT_CHAT_GEMMA:
    case TT_CHAT_GEMMA4: fmt_gemma(&b, msgs, n, opts); break;
    case TT_CHAT_LLAMA3: fmt_llama3(&b, msgs, n, opts); break;
    default: return -1;
    }
    out[b.len < cap ? b.len : cap - 1] = '\0';
    return (int)b.len;
}

int tt_chat_format(tt_chat_family fam, const tt_msg *msgs, int n,
                   char *out, size_t cap) {
    tt_chat_opts o = tt_chat_opts_default();
    return tt_chat_format_ex(fam, msgs, n, &o, out, cap);
}

/* ---- history helper ------------------------------------------------------- */

void tt_chat_history_init(tt_chat_history *h) {
    if (h) { h->n = 0; h->role[0][0] = '\0'; h->content[0][0] = '\0'; }
}

int tt_chat_history_push(tt_chat_history *h,
                         const char *role, const char *content) {
    if (!h || h->n >= TT_CHAT_MAX_MSGS) return -1;
    int i = h->n++;
    snprintf(h->role[i], sizeof(h->role[i]), "%s", role ? role : "");
    snprintf(h->content[i], sizeof(h->content[i]), "%s",
             content ? content : "");
    return 0;
}

int tt_chat_history_format(const tt_chat_history *h, tt_chat_family fam,
                           const tt_chat_opts *opts, char *out, size_t cap) {
    if (!h) return -2;
    tt_msg m[TT_CHAT_MAX_MSGS];
    for (int i = 0; i < h->n; i++) {
        m[i].role = h->role[i];
        m[i].content = h->content[i];
    }
    return tt_chat_format_ex(fam, m, h->n, opts, out, cap);
}
