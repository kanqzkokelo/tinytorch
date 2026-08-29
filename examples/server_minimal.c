// examples/server_minimal.c -- M12 P1 minimal HTTP server (single-request, N=1).
//
// Plain C, no external deps beyond libc + pthread + libcuda (linked via the
// qwen2_cuda kernel objects, same as examples/chat_llm_gpu.c). No JSON library:
// the request body is parsed by a tiny hand-rolled extractor and the response
// is hand-serialized. The shape is small ({"messages":[...]} in, one
// {"choices":[...]} out) and we want zero new build-system dependencies.
//
// ENGINE THREAD-SAFETY ASSUMPTION
// -------------------------------
// Qwen2Engine is single-stream today. There is exactly ONE in-flight generation
// at a time, and ALL engine API calls (qwen2_engine_prefill, qwen2_engine_next,
// qwen2_debug_replay_step, qwen2_debug_copy_logits) are serialized through the
// single g_engine_mutex below. This is the P1 / "N=1 slot" constraint from
// docs/plans/2026-08-27-server-design.md §0 + §2.2 step 1. P0 (slot_id
// threading through the engine) and P2 (continuous batching with N>1) are
// future milestones; this server is intentionally not thread-safe beyond the
// one-engine-at-a-time rule.
//
// WHAT THIS WRAPS (no engine code is reimplemented here)
// -----------------------------------------------------
//   qwen2_engine_create / qwen2_engine_free        - engine lifetime
//   qwen2_engine_prefill                            - prompt ingestion
//   qwen2_engine_next                               - first (eager) scoring step
//   qwen2_debug_replay_step                         - per-token graph replay
//   qwen2_debug_copy_logits                         - per-step logits for sampling
//   qwen2_engine_pos                                - ctx-bounds check
//   bpe_encode                                      - prompt tokenization
//   bpe_decode_token                                - per-token detokenization
//   tt_chat_format_ex + tt_chat_family_from_arch    - chat template
//   tt_sample + tt_sampler_chain                    - sampling (greedy default)
//
// Endpoints
// ---------
//   POST /v1/chat/completions    -- primary OAI-compatible entry
//   GET  /v1/models              -- reports the loaded model id
//   GET  /healthz                -- "ok" once model is loaded
//
// Streaming (SSE) is intentionally NOT implemented in P1. Single JSON response
// per request; token-by-token streaming is a P2+ add.
//
// Listen address: env TT_LISTEN (default 127.0.0.1:8080).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <cuda_runtime.h>

#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#include "chat_template.h"
#include "samplers.h"

/* --------------------------- configuration ---------------------------- */

#define MAX_REQ_BYTES     (1 << 20)   /* 1 MiB hard cap on a request body   */
#define MAX_MSGS          64          /* mirrors TT_CHAT_MAX_MSGS            */
#define MAX_CONTENT       4096        /* mirrors TT_CHAT_MAX_CONTENT         */
#define MAX_TOK_PROMPT    4096        /* hard cap on prompt token count      */
#define MAX_GEN_TOKENS    2048        /* hard cap server-side on max_tokens  */
#define REQ_BUF_GROW      (1 << 14)   /* per-conn read buffer growth chunk   */

#define ENV_LISTEN        "TT_LISTEN"
#define ENV_MODEL         "TT_MODEL"
#define ENV_MAX_CTX       "TT_MAX_CTX"
#define ENV_MAX_TOKENS    "TT_MAX_TOKENS"
#define ENV_TEMP          "TT_TEMP"
#define ENV_GREEDY        "TT_GREEDY"

static const char *DEFAULT_LISTEN  = "127.0.0.1:8080";
static const char *DEFAULT_MODEL   = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
static const int   DEFAULT_MAX_CTX = 1024;
static const int   DEFAULT_MAX_GEN = 512;

static const char *SERVER_NAME = "tinytorch-server-minimal/0.1 (M12-P1)";

/* --------------------------- shared state ----------------------------- */

typedef struct {
    GGUFModel    *model;
    BPETokenizer *tok;
    Qwen2Engine  *eng;
    TTConfig      cfg;
    tt_chat_family fam;
    int           vocab;
    int           max_ctx;
    int           listening;     /* 1 between bind() and shutdown           */
    pthread_mutex_t mu;          /* serializes all engine + tokenizer use  */
} ServerState;

static ServerState g_srv = {
    .model = NULL, .tok = NULL, .eng = NULL,
    .vocab = 0, .max_ctx = 0, .listening = 0,
    .mu = PTHREAD_MUTEX_INITIALIZER
};

/* shutdown flag: set by SIGINT/SIGTERM, drained in accept() loop. */
static volatile sig_atomic_t g_shutdown = 0;
static int g_listen_fd = -1;

/* ------------------------------ utils --------------------------------- */

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[server] fatal: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void log_info(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[server] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *env_or(const char *k, const char *dflt) {
    const char *v = getenv(k);
    return (v && v[0]) ? v : dflt;
}
static int env_int(const char *k, int dflt) {
    const char *v = getenv(k);
    return (v && v[0]) ? atoi(v) : dflt;
}
static float env_float(const char *k, float dflt) {
    const char *v = getenv(k);
    return (v && v[0]) ? (float)atof(v) : dflt;
}

/* ------------------------- HTTP framing ------------------------------- */

/* Read until we have a full request: header block (terminated by "\r\n\r\n")
 * + Content-Length body bytes. Returns total bytes in *out_len, or -1 on
 * protocol/IO error. Caller frees *out_buf. */
static int read_http_request(int fd, char **out_buf) {
    size_t cap = REQ_BUF_GROW, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return -1;

    /* growable read until \r\n\r\n is observed */
    int header_end = -1;
    while (header_end < 0) {
        if (len + 1 >= cap) {
            size_t ncap = cap * 2;
            if (ncap > MAX_REQ_BYTES + 4096) { free(buf); return -1; }
            char *n = (char *)realloc(buf, ncap);
            if (!n) { free(buf); return -1; }
            buf = n; cap = ncap;
        }
        ssize_t r = recv(fd, buf + len, cap - 1 - len, 0);
        if (r == 0) { free(buf); return -1; }                /* peer closed */
        if (r < 0) { if (errno == EINTR) continue; free(buf); return -1; }
        len += (size_t)r;
        buf[len] = '\0';
        char *e = strstr(buf, "\r\n\r\n");
        if (e) header_end = (int)(e - buf);
    }
    /* parse Content-Length (0 if missing) */
    long content_length = 0;
    {
        char *cl = strcasestr(buf, "Content-Length:");
        if (cl && cl < buf + header_end) {
            cl += strlen("Content-Length:");
            while (*cl == ' ' || *cl == '\t') cl++;
            content_length = strtol(cl, NULL, 10);
        }
    }
    if (content_length < 0 || content_length > MAX_REQ_BYTES) {
        free(buf); return -1;
    }
    size_t header_len = (size_t)header_end + 4;
    size_t need_total = header_len + (size_t)content_length;
    if (need_total > MAX_REQ_BYTES) { free(buf); return -1; }
    while (len < need_total) {
        if (len + 1 >= cap) {
            size_t ncap = cap * 2;
            if (ncap > MAX_REQ_BYTES + 4096) { free(buf); return -1; }
            char *n = (char *)realloc(buf, ncap);
            if (!n) { free(buf); return -1; }
            buf = n; cap = ncap;
        }
        ssize_t r = recv(fd, buf + len, cap - 1 - len, 0);
        if (r == 0) { free(buf); return -1; }
        if (r < 0) { if (errno == EINTR) continue; free(buf); return -1; }
        len += (size_t)r;
        buf[len] = '\0';
    }
    buf[need_total] = '\0';
    *out_buf = buf;
    return (int)need_total;
}

/* Minimal HTTP response writer. */
static void write_all(int fd, const char *buf, size_t n) {
    while (n) {
        ssize_t w = send(fd, buf, n, MSG_NOSIGNAL);
        if (w <= 0) { if (w < 0 && errno == EINTR) continue; return; }
        buf += w; n -= (size_t)w;
    }
}
static void writef(int fd, const char *fmt, ...) {
    char tmp[1024];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n >= sizeof(tmp)) n = sizeof(tmp) - 1;
    write_all(fd, tmp, (size_t)n);
}
static void write_status(int fd, int code, const char *reason,
                         const char *ctype, const char *body, size_t body_len) {
    writef(fd, "HTTP/1.1 %d %s\r\n", code, reason);
    writef(fd, "Content-Type: %s\r\n", ctype);
    writef(fd, "Content-Length: %zu\r\n", body_len);
    writef(fd, "Server: %s\r\n", SERVER_NAME);
    writef(fd, "Access-Control-Allow-Origin: *\r\n");
    writef(fd, "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
    writef(fd, "Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    writef(fd, "Connection: close\r\n\r\n");
    if (body && body_len) write_all(fd, body, body_len);
}

/* ----------------------------- JSON ----------------------------------- *
 * Tiny hand-rolled helpers. We do NOT support nested arrays-of-arrays or
 * escapes other than \" \\ \/ \n \r \t \b \f \uXXXX. The request we accept
 * only has {"messages":[{"role":..,"content":..}], "max_tokens":N,
 * "model":"..."} so the supported surface is small. */

/* Skip leading whitespace inside a JSON string (modifies *p). */
static void json_skip_ws(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
}

/* Read a JSON string starting at *p (which points at the opening quote).
 * On success, stores a malloc'd NUL-terminated copy in *out (caller frees),
 * advances *p past the closing quote, returns 0. Returns -1 on parse error
 * or unterminated string. Supports the same escape subset as the writer. */
static int json_parse_string(const char **p, char **out) {
    if (**p != '"') return -1;
    (*p)++;
    /* first pass: measure */
    const char *s = *p;
    size_t raw_len = 0;
    while (*s && *s != '"') {
        if (*s == '\\' && s[1]) { s += 2; raw_len += 2; }
        else { s++; raw_len++; }
    }
    if (*s != '"') return -1;
    char *buf = (char *)malloc(raw_len + 1);
    if (!buf) return -1;
    size_t i = 0;
    while (**p && **p != '"') {
        if (**p == '\\' && (*p)[1]) {
            buf[i++] = *(*p)++; buf[i++] = **p;
        } else {
            buf[i++] = **p;
        }
        (*p)++;
    }
    buf[i] = '\0';
    if (**p != '"') { free(buf); return -1; }
    (*p)++;            /* consume closing quote */
    *out = buf;
    return 0;
}

/* Find the value of a top-level string field by key. Returns malloc'd copy
 * or NULL. Scans only top-level object (one nesting level). */
static char *json_top_string(const char *body, const char *key) {
    const char *p = body;
    json_skip_ws(&p);
    if (*p != '{') return NULL;
    p++;
    while (1) {
        json_skip_ws(&p);
        if (*p == '}') return NULL;
        if (*p != '"') return NULL;
        char *k = NULL;
        if (json_parse_string(&p, &k) < 0) return NULL;
        json_skip_ws(&p);
        int match = k && strcmp(k, key) == 0;
        free(k);
        if (*p != ':') return NULL;
        p++;
        json_skip_ws(&p);
        if (match && *p == '"') {
            char *v = NULL;
            if (json_parse_string(&p, &v) < 0) return NULL;
            return v;
        }
        /* skip past the value */
        int depth = 0;
        if (*p == '{' || *p == '[') { depth = 1; p++; }
        else if (*p == '"') { char *tmp; if (json_parse_string(&p, &tmp)<0) return NULL; free(tmp); }
        else {
            while (*p && *p != ',' && *p != '}') p++;
        }
        while (depth) {
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        }
        json_skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return NULL;
    }
}

/* Find a top-level integer field. Returns dflt if missing. */
static int json_top_int(const char *body, const char *key, int dflt) {
    const char *p = body;
    json_skip_ws(&p);
    if (*p != '{') return dflt;
    p++;
    while (1) {
        json_skip_ws(&p);
        if (*p == '}') return dflt;
        if (*p != '"') return dflt;
        char *k = NULL;
        if (json_parse_string(&p, &k) < 0) return dflt;
        json_skip_ws(&p);
        int match = k && strcmp(k, key) == 0;
        free(k);
        if (*p != ':') return dflt;
        p++;
        json_skip_ws(&p);
        if (match) {
            int neg = 0; long v = 0;
            if (*p == '-') { neg = 1; p++; }
            if (*p < '0' || *p > '9') return dflt;
            while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
            return neg ? (int)-v : (int)v;
        }
        int depth = 0;
        if (*p == '{' || *p == '[') { depth = 1; p++; }
        else if (*p == '"') { char *tmp; if (json_parse_string(&p, &tmp)<0) return dflt; free(tmp); }
        else {
            while (*p && *p != ',' && *p != '}') p++;
        }
        while (depth) {
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        }
        json_skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return dflt;
    }
}

/* Find a top-level float field. Returns dflt if missing. */
static float json_top_float(const char *body, const char *key, float dflt) {
    const char *p = body;
    json_skip_ws(&p);
    if (*p != '{') return dflt;
    p++;
    while (1) {
        json_skip_ws(&p);
        if (*p == '}' || *p == '\0') return dflt;
        if (*p != '"') return dflt;
        char *k = NULL;
        if (json_parse_string(&p, &k) < 0) return dflt;
        json_skip_ws(&p);
        int match = k && strcmp(k, key) == 0;
        free(k);
        if (*p != ':') return dflt;
        p++;
        json_skip_ws(&p);
        if (match) {
            char *end = NULL;
            float v = strtof(p, &end);
            if (end != p) return v;
            return dflt;
        }
        int depth = 0;
        if (*p == '{' || *p == '[') { depth = 1; p++; }
        else if (*p == '"') { char *tmp; if (json_parse_string(&p, &tmp) < 0) return dflt; free(tmp); }
        else {
            while (*p && *p != ',' && *p != '}') p++;
        }
        while (depth) {
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        }
        json_skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return dflt;
    }
}

/* Find a top-level boolean field. Returns dflt if missing. */
static int json_top_bool(const char *body, const char *key, int dflt) {
    const char *p = body;
    json_skip_ws(&p);
    if (*p != '{') return dflt;
    p++;
    while (1) {
        json_skip_ws(&p);
        if (*p == '}' || *p == '\0') return dflt;
        if (*p != '"') return dflt;
        char *k = NULL;
        if (json_parse_string(&p, &k) < 0) return dflt;
        json_skip_ws(&p);
        int match = k && strcmp(k, key) == 0;
        free(k);
        if (*p != ':') return dflt;
        p++;
        json_skip_ws(&p);
        if (match) {
            if (strncmp(p, "true", 4) == 0) return 1;
            if (strncmp(p, "false", 5) == 0) return 0;
            return dflt;
        }
        int depth = 0;
        if (*p == '{' || *p == '[') { depth = 1; p++; }
        else if (*p == '"') { char *tmp; if (json_parse_string(&p, &tmp) < 0) return dflt; free(tmp); }
        else {
            while (*p && *p != ',' && *p != '}') p++;
        }
        while (depth) {
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        }
        json_skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return dflt;
    }
}

/* Extract messages[]: returns malloc'd tt_msg array, sets *n_out.
 * Each message content is truncated to MAX_CONTENT-1 bytes. */
static int json_extract_messages(const char *body, tt_msg *out_msgs) {
    /* find "messages" key in the top-level object */
    const char *p = body;
    json_skip_ws(&p);
    if (*p != '{') return -1;
    p++;
    int n = 0;
    while (1) {
        json_skip_ws(&p);
        if (*p == '}') return n;
        if (*p != '"') return -1;
        char *k = NULL;
        if (json_parse_string(&p, &k) < 0) return -1;
        int is_msg = k && strcmp(k, "messages") == 0;
        free(k);
        if (*p != ':') return -1;
        p++;
        json_skip_ws(&p);
        if (is_msg) {
            if (*p != '[') return -1;
            p++;
            while (1) {
                json_skip_ws(&p);
                if (*p == ']') return n;
                if (*p != '{') return -1;
                p++;
                out_msgs[n].role = NULL;
                out_msgs[n].content = NULL;
                while (1) {
                    json_skip_ws(&p);
                    if (*p == '}') break;
                    if (*p != '"') return -1;
                    char *mk = NULL;
                    if (json_parse_string(&p, &mk) < 0) return -1;
                    json_skip_ws(&p);
                    if (*p != ':') { free(mk); return -1; }
                    p++;
                    json_skip_ws(&p);
                    if (strcmp(mk, "role") == 0) {
                        free((void *)out_msgs[n].role);
                        json_parse_string(&p, (char **)&out_msgs[n].role);
                    } else if (strcmp(mk, "content") == 0) {
                        free((void *)out_msgs[n].content);
                        char *c = NULL;
                        json_parse_string(&p, &c);
                        if (c) {
                            /* truncate to MAX_CONTENT-1 to match the
                             * tt_chat_history storage cap. */
                            size_t cl = strlen(c);
                            if (cl >= MAX_CONTENT) c[MAX_CONTENT - 1] = '\0';
                            out_msgs[n].content = c;
                        }
                    } else {
                        /* skip the value */
                        if (*p == '"') { char *tmp; json_parse_string(&p, &tmp); free(tmp); }
                        else { while (*p && *p != ',' && *p != '}') p++; }
                    }
                    free(mk);
                    json_skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    if (*p == '}') break;
                }
                p++; /* closing } */
                if (n >= MAX_MSGS) return n;
                n++;
                json_skip_ws(&p);
                if (*p == ',') { p++; continue; }
                if (*p == ']') return n;
            }
        }
        /* skip past the value */
        int depth = 0;
        if (*p == '{' || *p == '[') { depth = 1; p++; }
        else if (*p == '"') { char *tmp; if (json_parse_string(&p, &tmp)<0) return -1; free(tmp); }
        else {
            while (*p && *p != ',' && *p != '}') p++;
        }
        while (depth) {
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        }
        json_skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return n;
    }
}

/* JSON string escape: write a quoted+escaped copy of s into a heap buffer
 * with growable capacity. Returns the buffer (caller frees) and sets *len. */
typedef struct { char *p; size_t len, cap; } SB;
static void sb_init(SB *s) { s->cap = 256; s->p = (char *)malloc(s->cap); s->len = 0; if (s->p) s->p[0] = '\0'; }
static void sb_grow(SB *s, size_t need) {
    if (s->len + need + 1 < s->cap) return;
    size_t ncap = s->cap;
    while (ncap < s->len + need + 1) ncap *= 2;
    char *n = (char *)realloc(s->p, ncap);
    if (!n) return;
    s->p = n; s->cap = ncap;
}
static void sb_putc(SB *s, char c) { sb_grow(s, 1); s->p[s->len++] = c; s->p[s->len] = '\0'; }
static void sb_puts(SB *s, const char *t) { size_t n = strlen(t); sb_grow(s, n); memcpy(s->p + s->len, t, n); s->len += n; s->p[s->len] = '\0'; }
static void sb_putd(SB *s, long v) { char t[32]; int n = snprintf(t, sizeof(t), "%ld", v); sb_grow(s, (size_t)n); memcpy(s->p + s->len, t, (size_t)n); s->len += (size_t)n; s->p[s->len] = '\0'; }
static void sb_json_string(SB *s, const char *t) {
    sb_putc(s, '"');
    if (t) for (const unsigned char *p = (const unsigned char *)t; *p; p++) {
        switch (*p) {
            case '"':  sb_puts(s, "\\\""); break;
            case '\\': sb_puts(s, "\\\\"); break;
            case '\n': sb_puts(s, "\\n"); break;
            case '\r': sb_puts(s, "\\r"); break;
            case '\t': sb_puts(s, "\\t"); break;
            case '\b': sb_puts(s, "\\b"); break;
            case '\f': sb_puts(s, "\\f"); break;
            default:
                if (*p < 0x20) {
                    char esc[8]; snprintf(esc, sizeof(esc), "\\u%04x", *p);
                    sb_puts(s, esc);
                } else {
                    sb_putc(s, (char)*p);
                }
        }
    }
    sb_putc(s, '"');
}

static void sb_json_string_len(SB *s, const char *t, size_t len) {
    sb_putc(s, '"');
    if (t) {
        for (size_t i = 0; i < len; i++) {
            unsigned char c = (unsigned char)t[i];
            switch (c) {
                case '"':  sb_puts(s, "\\\""); break;
                case '\\': sb_puts(s, "\\\\"); break;
                case '\n': sb_puts(s, "\\n"); break;
                case '\r': sb_puts(s, "\\r"); break;
                case '\t': sb_puts(s, "\\t"); break;
                case '\b': sb_puts(s, "\\b"); break;
                case '\f': sb_puts(s, "\\f"); break;
                default:
                    if (c < 0x20) {
                        char esc[8]; snprintf(esc, sizeof(esc), "\\u%04x", c);
                        sb_puts(s, esc);
                    } else if (c >= 0x80) {
                        char esc[8]; snprintf(esc, sizeof(esc), "\\u00%02x", c);
                        sb_puts(s, esc);
                    } else {
                        sb_putc(s, (char)c);
                    }
            }
        }
    }
    sb_putc(s, '"');
}

/* --------------------------- stop strings ----------------------------- */

#define MAX_STOP_STRINGS 16
#define MAX_STOP_LEN     63
static char g_stop[MAX_STOP_STRINGS][MAX_STOP_LEN + 1];
static int  g_nstop = 0;

static void add_stop(const char *s, size_t n) {
    if (g_nstop >= MAX_STOP_STRINGS || n == 0 || n > MAX_STOP_LEN) return;
    memcpy(g_stop[g_nstop], s, n);
    g_stop[g_nstop][n] = '\0';
    g_nstop++;
}

/* return offset of earliest stop-string match in buf, or -1 */
static int find_stop(const char *buf, size_t n) {
    int best = -1;
    for (int i = 0; i < g_nstop; i++) {
        const char *hit = strstr(buf, g_stop[i]);
        if (hit) {
            int off = (int)(hit - buf);
            if (best < 0 || off < best) best = off;
        }
    }
    (void)n;
    return best;
}

/* -------------------------- generation core --------------------------- */

typedef struct {
    int   n_generated;
    int   finish_reason;  /* 0=stop, 1=length, 2=eos, 3=ctx_full, 4=error */
    char  text[8192];
    size_t text_len;
    double latency_sec;
} GenResult;

/* Run one chat completion. MUST be called with g_srv.mu held. */
static int run_chat(tt_msg *msgs, int n_msgs, int max_tokens,
                    float temp, float top_p, float rep_pen,
                    int stream_fd, GenResult *out) {
    memset(out, 0, sizeof(*out));
    out->text[0] = '\0';
    if (n_msgs <= 0) { out->finish_reason = 4; return -1; }

    /* format the conversation via the engine's chat_template helper */
    char formatted[16384];
    tt_chat_opts opts = tt_chat_opts_default();
    int need = tt_chat_format_ex(g_srv.fam, msgs, n_msgs, &opts,
                                 formatted, sizeof(formatted));
    if (need < 0) {
        log_info("chat_template format failed: %d", need);
        out->finish_reason = 4; return -1;
    }
    if ((size_t)need >= sizeof(formatted)) {
        log_info("formatted prompt truncated (%d bytes)", need);
        out->finish_reason = 4; return -1;
    }

    /* tokenize */
    int prompt_toks[MAX_TOK_PROMPT];
    int n_prompt = bpe_encode(g_srv.tok, formatted, prompt_toks, MAX_TOK_PROMPT);
    if (n_prompt <= 0) {
        log_info("tokenization failed (n=%d)", n_prompt);
        out->finish_reason = 4; return -1;
    }
    if (n_prompt >= g_srv.max_ctx - 1) {
        log_info("prompt too long: %d tokens (max_ctx=%d)",
                 n_prompt, g_srv.max_ctx);
        out->finish_reason = 1; return -1;
    }

    /* prefill */
    if (qwen2_engine_prefill(g_srv.eng, prompt_toks, n_prompt) != 0) {
        log_info("qwen2_engine_prefill failed");
        out->finish_reason = 4; return -1;
    }

    /* seed the sampler */
    tt_sampler_chain sc;
    tt_sampler_chain_init(&sc);
    const int greedy = (temp <= 0.0f) || (getenv(ENV_GREEDY) && getenv(ENV_GREEDY)[0] != '\0');
    sc.greedy = greedy;
    sc.temp   = temp > 0.0f ? temp : env_float(ENV_TEMP, 0.8f);
    sc.repeat_penalty = rep_pen > 0.0f ? rep_pen : 1.15f;
    sc.use_rep_penalty = 1;
    sc.penalty_last_n = 64;
    sc.freq_last_n = 64;
    if (top_p > 0.0f && top_p < 1.0f) {
        sc.top_p = top_p;
    }
    int32_t hist[256];
    int n_hist = 0;
    /* seed history with the prompt tail */
    int start = n_prompt > 64 ? n_prompt - 64 : 0;
    for (int i = start; i < n_prompt && n_hist < 256; i++) hist[n_hist++] = prompt_toks[i];
    sc.history = hist; sc.n_history = n_hist;

    if (stream_fd >= 0) {
        writef(stream_fd, "HTTP/1.1 200 OK\r\n"
                          "Content-Type: text/event-stream\r\n"
                          "Cache-Control: no-cache\r\n"
                          "Access-Control-Allow-Origin: *\r\n"
                          "Connection: close\r\n"
                          "Server: %s\r\n\r\n", SERVER_NAME);
    }
    /* logits buffer for host-side sampling (graph path only) */
    float *logits = (float *)malloc(sizeof(float) * (size_t)g_srv.vocab);
    float *wb     = (float *)malloc(sizeof(float) * (size_t)tt_sampler_workbuf_size(g_srv.vocab));
    if (!logits || !wb) {
        free(logits); free(wb);
        out->finish_reason = 4; return -1;
    }
    uint64_t rng = 1;

    double t0 = now_sec();

    /* first engine_next() scores the prompt (eager greedy) and primes the
     * graph path; its sampled token is discarded because we drive every
     * subsequent step via qwen2_debug_replay_step (matches chat_llm_gpu.c). */
    if (qwen2_engine_next(g_srv.eng) < 0) {
        free(logits); free(wb);
        out->finish_reason = 4; return -1;
    }

    int cap = (int)sizeof(out->text) - 1;
    int gen = 0;
    while (gen < max_tokens && qwen2_engine_pos(g_srv.eng) < g_srv.max_ctx - 1) {
        int tok;
        if (qwen2_debug_copy_logits(g_srv.eng, logits, g_srv.vocab) < 0) break;
        tok = tt_sample(logits, g_srv.vocab, &sc, &rng, wb);
        if (tok < 0) break;
        if (tok == g_srv.tok->eos_id || tok == 151643 /*<|endoftext|>*/
            || tok == 151645 /*<|im_end|>*/) {
            /* feed the stop token so the transcript closes (parity with
             * chat_llm_gpu; also covers the eager path's advance) */
            if (qwen2_engine_pos(g_srv.eng) < g_srv.max_ctx - 1)
                qwen2_debug_replay_step(g_srv.eng, tok);
            out->finish_reason = 0;   /* "stop" */
            break;
        }
        int olen = 0;
        const char *s = bpe_decode_token(g_srv.tok, tok, &olen);
        if (olen > 0 && out->text_len + (size_t)olen < (size_t)cap) {
            memcpy(out->text + out->text_len, s, (size_t)olen);
            out->text_len += (size_t)olen;
            out->text[out->text_len] = '\0';
            if (stream_fd >= 0) {
                SB chunk; sb_init(&chunk);
                sb_puts(&chunk, "data: {\"id\":\"chatcmpl-tt-1\",\"object\":\"chat.completion.chunk\",\"created\":");
                sb_putd(&chunk, (long)time(NULL));
                sb_puts(&chunk, ",\"model\":");
                sb_json_string(&chunk, g_srv.model->architecture[0] ? g_srv.model->architecture : "model");
                sb_puts(&chunk, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
                sb_json_string_len(&chunk, s, (size_t)olen);
                sb_puts(&chunk, "},\"finish_reason\":null}]}\n\n");
                write_all(stream_fd, chunk.p, chunk.len);
                free(chunk.p);
            }
        }
        /* stop-string scan on the WHOLE accumulated text so markers that
         * span token boundaries are caught (mirrors chat_llm_gpu). */
        int cut = find_stop(out->text, out->text_len);
        if (cut >= 0) {
            out->text[cut] = '\0';
            out->text_len = (size_t)cut;
            out->finish_reason = 0;   /* "stop" (custom string) */
            /* still feed the marker so the transcript closes */
            if (qwen2_engine_pos(g_srv.eng) < g_srv.max_ctx - 1)
                qwen2_debug_replay_step(g_srv.eng, tok);
            break;
        }
        if (qwen2_engine_pos(g_srv.eng) >= g_srv.max_ctx - 1) {
            out->finish_reason = 3;   /* ctx_full */
            break;
        }
        if (qwen2_debug_replay_step(g_srv.eng, tok) < 0) {
            out->finish_reason = 4;
            break;
        }
        gen++;
        out->n_generated = gen;
        /* maintain sampler history */
        if (n_hist < 256) hist[n_hist++] = tok;
        else { memmove(hist, hist + 1, sizeof(int32_t) * 255); hist[255] = tok; }
        sc.history = hist; sc.n_history = n_hist;
    }
    if (gen >= max_tokens && out->finish_reason == 0) out->finish_reason = 1; /* "length" */
    if (out->finish_reason == 0 && gen > 0 && qwen2_engine_pos(g_srv.eng) >= g_srv.max_ctx - 1) {
        /* loop exited only because ctx was full */
        out->finish_reason = 3;
    }
    if (out->finish_reason == 0) {
        /* if we never hit a stop reason but exited cleanly, classify as stop */
        if (gen >= max_tokens) out->finish_reason = 1;
    }

    if (stream_fd >= 0) {
        const char *fin = (out->finish_reason == 0 || out->finish_reason == 2) ? "stop" :
                          (out->finish_reason == 1 || out->finish_reason == 3) ? "length" : "error";
        SB chunk; sb_init(&chunk);
        sb_puts(&chunk, "data: {\"id\":\"chatcmpl-tt-1\",\"object\":\"chat.completion.chunk\",\"created\":");
        sb_putd(&chunk, (long)time(NULL));
        sb_puts(&chunk, ",\"model\":");
        sb_json_string(&chunk, g_srv.model->architecture[0] ? g_srv.model->architecture : "model");
        sb_puts(&chunk, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
        sb_json_string(&chunk, fin);
        sb_puts(&chunk, "}]}\n\ndata: [DONE]\n\n");
        write_all(stream_fd, chunk.p, chunk.len);
        free(chunk.p);
    }
    cudaDeviceSynchronize();
    out->latency_sec = now_sec() - t0;

    free(logits); free(wb);
    return 0;
}

/* ----------------------- per-connection worker ------------------------- */

typedef struct {
    int fd;
    char *raw;       /* malloc'd full HTTP request (header + body) */
    int   raw_len;
} ConnJob;

static void respond_health(int fd) {
    const char *body = "{\"status\":\"ok\"}\n";
    write_status(fd, 200, "OK", "application/json", body, strlen(body));
}
static void respond_models(int fd) {
    SB s; sb_init(&s);
    sb_puts(&s, "{\"object\":\"list\",\"data\":[");
    sb_puts(&s, "{\"id\":");
    sb_json_string(&s, g_srv.model->architecture[0]
                        ? g_srv.model->architecture
                        : "model");
    sb_puts(&s, ",\"object\":\"model\"}");
    sb_puts(&s, "]}\n");
    write_status(fd, 200, "OK", "application/json", s.p, s.len);
    free(s.p);
}
static void respond_404(int fd) {
    const char *body = "{\"error\":\"not found\"}\n";
    write_status(fd, 404, "Not Found", "application/json", body, strlen(body));
}
static void respond_400(int fd, const char *msg) {
    SB s; sb_init(&s);
    sb_puts(&s, "{\"error\":");
    sb_json_string(&s, msg);
    sb_puts(&s, "}\n");
    write_status(fd, 400, "Bad Request", "application/json", s.p, s.len);
    free(s.p);
}
static void respond_500(int fd, const char *msg) {
    SB s; sb_init(&s);
    sb_puts(&s, "{\"error\":");
    sb_json_string(&s, msg);
    sb_puts(&s, "}\n");
    write_status(fd, 500, "Internal Server Error", "application/json", s.p, s.len);
    free(s.p);
}

static void respond_chat(int fd, const char *body) {
    /* parse + dispatch under the engine mutex (engine is single-stream) */
    tt_msg msgs[MAX_MSGS];
    memset(msgs, 0, sizeof(msgs));
    int n_msgs = json_extract_messages(body, msgs);
    if (n_msgs <= 0) {
        for (int i = 0; i < n_msgs; i++) {
            free((void *)msgs[i].role);
            free((void *)msgs[i].content);
        }
        respond_400(fd, "missing or empty messages[]");
        return;
    }
    int max_tokens = json_top_int(body, "max_tokens", 0);
    if (max_tokens <= 0) max_tokens = env_int(ENV_MAX_TOKENS, DEFAULT_MAX_GEN);
    if (max_tokens > MAX_GEN_TOKENS) max_tokens = MAX_GEN_TOKENS;
    char *model_id = json_top_string(body, "model");    /* advisory */
    free(model_id);

    int stream = json_top_bool(body, "stream", 0);
    float temp = json_top_float(body, "temperature", env_float(ENV_TEMP, 0.8f));
    float top_p = json_top_float(body, "top_p", 1.0f);
    float rep_pen = json_top_float(body, "repetition_penalty", 1.15f);

    GenResult r;
    double t_lock = now_sec();
    pthread_mutex_lock(&g_srv.mu);
    double t_locked = now_sec();
    int rc = run_chat(msgs, n_msgs, max_tokens, temp, top_p, rep_pen, stream ? fd : -1, &r);
    pthread_mutex_unlock(&g_srv.mu);
    double t_done = now_sec();
    log_info("req: msgs=%d stream=%d max_tokens=%d gen=%d finish=%d lock_wait=%.1fms engine=%.1fms total=%.1fms",
             n_msgs, stream, max_tokens, r.n_generated, r.finish_reason,
             (t_locked - t_lock) * 1000.0,
             r.latency_sec * 1000.0,
             (t_done - t_lock) * 1000.0);
    for (int i = 0; i < n_msgs; i++) {
        free((void *)msgs[i].role);
        free((void *)msgs[i].content);
    }
    if (rc < 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "generation failed (finish=%d)", r.finish_reason);
        if (!stream) respond_500(fd, msg);
        return;
    }
    if (stream) return;

    /* response body */
    SB s; sb_init(&s);
    sb_puts(&s, "{\"id\":\"chatcmpl-tt-1\",\"object\":\"chat.completion\",");
    sb_puts(&s, "\"created\":");
    sb_putd(&s, (long)time(NULL));
    sb_puts(&s, ",\"model\":");
    sb_json_string(&s, g_srv.model->architecture[0] ? g_srv.model->architecture : "model");
    sb_puts(&s, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    sb_json_string(&s, r.text);
    sb_puts(&s, "},\"finish_reason\":");
    switch (r.finish_reason) {
        case 0: sb_puts(&s, "\"stop\""); break;
        case 1: sb_puts(&s, "\"length\""); break;
        case 2: sb_puts(&s, "\"stop\""); break;     /* eos -> stop */
        case 3: sb_puts(&s, "\"length\""); break;   /* ctx_full -> length */
        case 4: sb_puts(&s, "\"error\""); break;
        default: sb_puts(&s, "\"stop\""); break;
    }
    sb_puts(&s, "}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":");
    sb_putd(&s, r.n_generated);
    sb_puts(&s, ",\"total_tokens\":");
    sb_putd(&s, r.n_generated);
    sb_puts(&s, "}}\n");
    write_status(fd, 200, "OK", "application/json", s.p, s.len);
    free(s.p);
}

static void *conn_thread(void *arg) {
    ConnJob *j = (ConnJob *)arg;
    int fd = j->fd;
    /* Disable Nagle for low-latency replies. */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    char *buf = NULL;
    int total = read_http_request(fd, &buf);
    if (total < 0) { close(fd); free(buf); free(j); return NULL; }

    /* parse request line: METHOD SP PATH SP HTTP/1.x */
    char method[16] = {0}, path[256] = {0};
    if (sscanf(buf, "%15s %255s", method, path) != 2) {
        respond_400(fd, "malformed request line");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/healthz") == 0) {
        respond_health(fd);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        respond_models(fd);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
        char *body = strstr(buf, "\r\n\r\n");
        if (!body) respond_400(fd, "missing body");
        else respond_chat(fd, body + 4);
    } else {
        respond_404(fd);
    }
    free(buf);
    close(fd);
    free(j);
    return NULL;
}

/* --------------------- listen socket helpers -------------------------- */

static int parse_hostport(const char *s, struct sockaddr_in *out) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%s", s);
    char *colon = strrchr(buf, ':');
    if (!colon) return -1;
    *colon = '\0';
    const char *host = buf;
    const char *port = colon + 1;
    int p = atoi(port);
    if (p <= 0 || p > 65535) return -1;
    memset(out, 0, sizeof(*out));
    out->sin_family = AF_INET;
    out->sin_port = htons((uint16_t)p);
    if (host[0] == '\0' || !strcmp(host, "0.0.0.0")) {
        out->sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, host, &out->sin_addr) != 1) {
        return -1;
    }
    return 0;
}

static void on_signal(int sig) {
    (void)sig;
    g_shutdown = 1;
    if (g_listen_fd >= 0) {
        shutdown(g_listen_fd, SHUT_RDWR);
        close(g_listen_fd);
        g_listen_fd = -1;
    }
}

/* ------------------------------- main --------------------------------- */

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    const char *model_path = env_or(ENV_MODEL, DEFAULT_MODEL);
    const char *listen_str = env_or(ENV_LISTEN, DEFAULT_LISTEN);
    int max_ctx = env_int(ENV_MAX_CTX, DEFAULT_MAX_CTX);
    if (max_ctx < 64) max_ctx = 64;

    log_info("M12 P1 minimal server starting; model=%s listen=%s max_ctx=%d",
             model_path, listen_str, max_ctx);

    g_srv.model = gguf_load(model_path);
    if (!g_srv.model) die("failed to load model: %s", model_path);
    g_srv.fam = tt_chat_family_from_arch(g_srv.model->architecture);
    if (g_srv.fam < 0) g_srv.fam = TT_CHAT_QWEN2;
    g_srv.tok = bpe_tokenizer_init(g_srv.model);
    if (!g_srv.tok) die("failed to init tokenizer");
    g_srv.vocab = g_srv.tok->vocab_size;
    g_srv.cfg = tt_config_from_gguf(g_srv.model, max_ctx);
    if (g_srv.cfg.dim == 0) die("could not derive engine config from GGUF");
    g_srv.eng = qwen2_engine_create(&g_srv.cfg, g_srv.model);
    if (!g_srv.eng) die("qwen2_engine_create failed");
    g_srv.max_ctx = max_ctx;
    g_srv.listening = 1;
    log_info("engine ready: arch=%s vocab=%d dim=%d layers=%d ctx=%d",
             g_srv.model->architecture[0] ? g_srv.model->architecture : "?",
             g_srv.vocab, g_srv.cfg.dim, g_srv.cfg.n_layers, max_ctx);

    /* stop strings: family default + legacy guards (mirrors chat_llm_gpu.c) */
    const char *fs = tt_chat_stop_string(g_srv.fam);
    if (fs) add_stop(fs, strlen(fs));
    add_stop("<|endoftext|>", 13);
    add_stop("<|im_start|>", 12);

    /* bind + listen */
    struct sockaddr_in addr;
    if (parse_hostport(listen_str, &addr) != 0) die("bad TT_LISTEN: %s", listen_str);
    g_listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen_fd < 0) die("socket: %s", strerror(errno));
    int yes = 1;
    setsockopt(g_listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    if (bind(g_listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        die("bind %s: %s", listen_str, strerror(errno));
    if (listen(g_listen_fd, 32) < 0)
        die("listen: %s", strerror(errno));
    log_info("listening on %s", listen_str);

    /* accept loop */
    while (!g_shutdown) {
        struct sockaddr_in cli; socklen_t cl = sizeof(cli);
        int cfd = accept(g_listen_fd, (struct sockaddr *)&cli, &cl);
        if (cfd < 0) {
            if (errno == EINTR || g_shutdown) continue;
            log_info("accept: %s", strerror(errno));
            continue;
        }
        ConnJob *j = (ConnJob *)malloc(sizeof(*j));
        if (!j) { close(cfd); continue; }
        j->fd = cfd; j->raw = NULL; j->raw_len = 0;
        pthread_t tid;
        if (pthread_create(&tid, NULL, conn_thread, j) != 0) {
            log_info("pthread_create: %s", strerror(errno));
            close(cfd); free(j);
            continue;
        }
        pthread_detach(tid);
    }

    log_info("shutting down");
    g_srv.listening = 0;
    if (g_listen_fd >= 0) { close(g_listen_fd); g_listen_fd = -1; }
    qwen2_engine_free(g_srv.eng);
    bpe_tokenizer_free(g_srv.tok);
    gguf_free(g_srv.model);
    return 0;
}
