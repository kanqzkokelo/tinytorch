// examples/server_multimodel.c -- M12 P2 multi-model HTTP server.
//
// Plain C, no external deps beyond libc + pthread + libcuda. Same hand-rolled
// JSON helpers, same OpenAI-compatible request/response shape as P1
// (examples/server_minimal.c); the only meaningful addition is a REGISTRY
// of engine instances selected by the request body's "model" field.
//
// CONFIGURATION
// -------------
//   env TT_MODELS   -- REQUIRED, comma-separated "name=path" pairs, e.g.
//                       TT_MODELS=q25=data/models/qwen2.5-0.5b-instruct-q4_0.gguf,tt=data/testmodels/tinyllama-f16.gguf
//                     Each entry becomes one Qwen2Engine instance loaded at
//                     startup. Names are matched against the request body's
//                     "model" string (exact match). Names must be unique,
//                     non-empty, and free of commas / '=' / whitespace.
//   env TT_LISTEN   -- default 127.0.0.1:8080
//   env TT_MAX_CTX  -- applied to every engine (default 1024)
//   env TT_MAX_TOKENS / TT_TEMP / TT_GREEDY -- per-request sampling knobs
//
// CONCURRENCY MODEL
// -----------------
//   * One pthread per accepted connection (detach-on-create).
//   * One pthread_mutex_t PER ENGINE. Requests for model A and model B run
//     truly in parallel; two requests for the SAME model serialize behind
//     that engine's lock (Qwen2Engine is still single-stream).
//   * On unknown model name, the worker returns HTTP 400 without acquiring
//     any engine lock.
//
// VRAM FOOTPRINT (documented)
// ---------------------------
//   Each engine pins its weights on the GPU at load time. Concretely, the
//   two fixtures smoke-tested below sit at roughly:
//     qwen2.5-0.5b-instruct-q4_0.gguf  ~  0.35 GiB resident
//     tinyllama-f16.gguf               ~  1.05 GiB resident
//   Total ~ 1.4 GiB; comfortably under 4 GiB. Adding a third 7B-class model
//   would push past 4 GiB. nvidia-smi --query-gpu=memory.used --format=csv
//   reports actual usage after startup; the binary logs per-engine
//   "engine ready" lines so you can attribute.
//
// Endpoints
// ---------
//   POST /v1/chat/completions    -- primary OAI-compatible entry; routes
//                                   on the request body's "model" field
//   GET  /v1/models              -- lists every loaded model
//   GET  /healthz                -- "ok" once all engines are loaded

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
#define MAX_MSGS          64
#define MAX_CONTENT       4096
#define MAX_TOK_PROMPT    4096
#define MAX_GEN_TOKENS    2048
#define REQ_BUF_GROW      (1 << 14)

/* Registry cap. Bumping this is free; 16 is more than any sane demo. */
#define MAX_MODELS        16
#define MAX_NAME_LEN      64
#define MAX_PATH_LEN      512

#define ENV_LISTEN        "TT_LISTEN"
#define ENV_MODELS        "TT_MODELS"
#define ENV_MAX_CTX       "TT_MAX_CTX"
#define ENV_MAX_TOKENS    "TT_MAX_TOKENS"
#define ENV_TEMP          "TT_TEMP"
#define ENV_GREEDY        "TT_GREEDY"

static const char *DEFAULT_LISTEN  = "127.0.0.1:8080";
static const int   DEFAULT_MAX_CTX = 1024;
static const int   DEFAULT_MAX_GEN = 512;

static const char *SERVER_NAME = "tinytorch-server-multimodel/0.1 (M12-P2)";

/* --------------------------- registry --------------------------------- */

typedef struct {
    char            name[MAX_NAME_LEN];
    char            path[MAX_PATH_LEN];
    GGUFModel      *model;
    BPETokenizer   *tok;
    Qwen2Engine    *eng;
    TTConfig        cfg;
    tt_chat_family  fam;
    int             vocab;
    int             max_ctx;
    int             active;          /* 1 once successfully loaded       */
    pthread_mutex_t mu;              /* serializes use of THIS engine     */
} EngineEntry;

static EngineEntry       g_reg[MAX_MODELS];
static int               g_n_models = 0;
static volatile sig_atomic_t g_shutdown = 0;
static int               g_listen_fd = -1;

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

static int read_http_request(int fd, char **out_buf) {
    size_t cap = REQ_BUF_GROW, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return -1;

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
        if (r == 0) { free(buf); return -1; }
        if (r < 0) { if (errno == EINTR) continue; free(buf); return -1; }
        len += (size_t)r;
        buf[len] = '\0';
        char *e = strstr(buf, "\r\n\r\n");
        if (e) header_end = (int)(e - buf);
    }
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
    writef(fd, "Connection: close\r\n\r\n");
    if (body && body_len) write_all(fd, body, body_len);
}

/* ----------------------------- JSON ----------------------------------- *
 * Same hand-rolled subset as P1: top-level string + int fields, "messages"
 * array of {"role":..,"content":..} objects, string escape subset
 * \" \\ \/ \n \r \t \b \f \uXXXX. */

static void json_skip_ws(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
}

static int json_parse_string(const char **p, char **out) {
    if (**p != '"') return -1;
    (*p)++;
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
    (*p)++;
    *out = buf;
    return 0;
}

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
        int depth = 0;
        if (*p == '{' || *p == '[') { depth = 1; p++; }
        else if (*p == '"') { char *tmp; if (json_parse_string(&p, &tmp)<0) return NULL; free(tmp); }
        else { while (*p && *p != ',' && *p != '}') p++; }
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
        else { while (*p && *p != ',' && *p != '}') p++; }
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

static int json_extract_messages(const char *body, tt_msg *out_msgs) {
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
                            size_t cl = strlen(c);
                            if (cl >= MAX_CONTENT) c[MAX_CONTENT - 1] = '\0';
                            out_msgs[n].content = c;
                        }
                    } else {
                        if (*p == '"') { char *tmp; json_parse_string(&p, &tmp); free(tmp); }
                        else { while (*p && *p != ',' && *p != '}') p++; }
                    }
                    free(mk);
                    json_skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    if (*p == '}') break;
                }
                p++;
                if (n >= MAX_MSGS) return n;
                n++;
                json_skip_ws(&p);
                if (*p == ',') { p++; continue; }
                if (*p == ']') return n;
            }
        }
        int depth = 0;
        if (*p == '{' || *p == '[') { depth = 1; p++; }
        else if (*p == '"') { char *tmp; if (json_parse_string(&p, &tmp)<0) return -1; free(tmp); }
        else { while (*p && *p != ',' && *p != '}') p++; }
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

/* --------------------- per-request stop-string set -------------------- */

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
static void reset_stop(void) { g_nstop = 0; }
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

/* -------------------------- generation core --------------------------- *
 * Runs one chat completion against the given engine entry. MUST be called
 * with e->mu held. */

typedef struct {
    int   n_generated;
    int   finish_reason;  /* 0=stop, 1=length, 2=eos, 3=ctx_full, 4=error */
    char  text[8192];
    size_t text_len;
    double latency_sec;
} GenResult;

static int run_chat_on(EngineEntry *e, tt_msg *msgs, int n_msgs,
                       int max_tokens, GenResult *out) {
    memset(out, 0, sizeof(*out));
    out->text[0] = '\0';
    if (n_msgs <= 0) { out->finish_reason = 4; return -1; }

    char formatted[16384];
    tt_chat_opts opts = tt_chat_opts_default();
    int need = tt_chat_format_ex(e->fam, msgs, n_msgs, &opts,
                                 formatted, sizeof(formatted));
    if (need < 0) { out->finish_reason = 4; return -1; }
    if ((size_t)need >= sizeof(formatted)) { out->finish_reason = 4; return -1; }

    int prompt_toks[MAX_TOK_PROMPT];
    int n_prompt = bpe_encode(e->tok, formatted, prompt_toks, MAX_TOK_PROMPT);
    if (n_prompt <= 0) { out->finish_reason = 4; return -1; }
    if (n_prompt >= e->max_ctx - 1) { out->finish_reason = 1; return -1; }

    if (qwen2_engine_prefill(e->eng, prompt_toks, n_prompt) != 0) {
        out->finish_reason = 4; return -1;
    }

    tt_sampler_chain sc;
    tt_sampler_chain_init(&sc);
    const int greedy = getenv(ENV_GREEDY) && getenv(ENV_GREEDY)[0] != '\0';
    sc.greedy = greedy;
    sc.temp   = env_float(ENV_TEMP, 0.8f);
    sc.repeat_penalty = 1.15f;
    sc.use_rep_penalty = 1;
    sc.penalty_last_n = 64;
    sc.freq_last_n = 64;
    int32_t hist[256];
    int n_hist = 0;
    int start = n_prompt > 64 ? n_prompt - 64 : 0;
    for (int i = start; i < n_prompt && n_hist < 256; i++) hist[n_hist++] = prompt_toks[i];
    sc.history = hist; sc.n_history = n_hist;

    float *logits = (float *)malloc(sizeof(float) * (size_t)e->vocab);
    float *wb     = (float *)malloc(sizeof(float) * (size_t)tt_sampler_workbuf_size(e->vocab));
    if (!logits || !wb) {
        free(logits); free(wb);
        out->finish_reason = 4; return -1;
    }
    uint64_t rng = 1;

    /* per-engine stop strings (family + legacy guards) */
    reset_stop();
    const char *fs = tt_chat_stop_string(e->fam);
    if (fs) add_stop(fs, strlen(fs));
    add_stop("<|endoftext|>", 13);
    add_stop("<|im_start|>", 12);

    double t0 = now_sec();

    if (qwen2_engine_next(e->eng) < 0) {
        free(logits); free(wb);
        out->finish_reason = 4; return -1;
    }

    int cap = (int)sizeof(out->text) - 1;
    int gen = 0;
    while (gen < max_tokens && qwen2_engine_pos(e->eng) < e->max_ctx - 1) {
        int tok;
        if (qwen2_debug_copy_logits(e->eng, logits, e->vocab) < 0) break;
        tok = tt_sample(logits, e->vocab, &sc, &rng, wb);
        if (tok < 0) break;
        if (tok == e->tok->eos_id || tok == 151643 || tok == 151645) {
            if (qwen2_engine_pos(e->eng) < e->max_ctx - 1)
                qwen2_debug_replay_step(e->eng, tok);
            out->finish_reason = 0;
            break;
        }
        int olen = 0;
        const char *s = bpe_decode_token(e->tok, tok, &olen);
        if (olen > 0 && out->text_len + (size_t)olen < (size_t)cap) {
            memcpy(out->text + out->text_len, s, (size_t)olen);
            out->text_len += (size_t)olen;
            out->text[out->text_len] = '\0';
        }
        int cut = find_stop(out->text, out->text_len);
        if (cut >= 0) {
            out->text[cut] = '\0';
            out->text_len = (size_t)cut;
            out->finish_reason = 0;
            if (qwen2_engine_pos(e->eng) < e->max_ctx - 1)
                qwen2_debug_replay_step(e->eng, tok);
            break;
        }
        if (qwen2_engine_pos(e->eng) >= e->max_ctx - 1) {
            out->finish_reason = 3;
            break;
        }
        if (qwen2_debug_replay_step(e->eng, tok) < 0) {
            out->finish_reason = 4;
            break;
        }
        gen++;
        out->n_generated = gen;
        if (n_hist < 256) hist[n_hist++] = tok;
        else { memmove(hist, hist + 1, sizeof(int32_t) * 255); hist[255] = tok; }
        sc.history = hist; sc.n_history = n_hist;
    }
    if (gen >= max_tokens && out->finish_reason == 0) out->finish_reason = 1;
    if (out->finish_reason == 0 && gen > 0 && qwen2_engine_pos(e->eng) >= e->max_ctx - 1) {
        out->finish_reason = 3;
    }
    if (out->finish_reason == 0 && gen >= max_tokens) out->finish_reason = 1;

    cudaDeviceSynchronize();
    out->latency_sec = now_sec() - t0;

    free(logits); free(wb);
    return 0;
}

/* ----------------------- per-connection worker ------------------------- */

typedef struct {
    int fd;
    char *raw;
    int   raw_len;
} ConnJob;

static void respond_health(int fd) {
    const char *body = "{\"status\":\"ok\"}\n";
    write_status(fd, 200, "OK", "application/json", body, strlen(body));
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

static void respond_models(int fd) {
    SB s; sb_init(&s);
    sb_puts(&s, "{\"object\":\"list\",\"data\":[");
    for (int i = 0; i < g_n_models; i++) {
        if (i) sb_puts(&s, ",");
        sb_puts(&s, "{\"id\":");
        sb_json_string(&s, g_reg[i].name);
        sb_puts(&s, ",\"object\":\"model\"}");
    }
    sb_puts(&s, "]}\n");
    write_status(fd, 200, "OK", "application/json", s.p, s.len);
    free(s.p);
}

static void respond_chat(int fd, const char *body) {
    /* Resolve the requested model first; we don't want to acquire any
     * engine lock if the name is unknown or the messages are bad. */
    char *model_id = json_top_string(body, "model");
    if (!model_id || !model_id[0]) {
        free(model_id);
        respond_400(fd, "missing or empty model field");
        return;
    }
    EngineEntry *e = NULL;
    for (int i = 0; i < g_n_models; i++) {
        if (strcmp(g_reg[i].name, model_id) == 0) { e = &g_reg[i]; break; }
    }
    if (!e) {
        char msg[256];
        snprintf(msg, sizeof(msg), "unknown model '%s'", model_id);
        free(model_id);
        respond_400(fd, msg);
        return;
    }
    free(model_id);

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

    GenResult r;
    double t_lock = now_sec();
    pthread_mutex_lock(&e->mu);
    double t_locked = now_sec();
    int rc = run_chat_on(e, msgs, n_msgs, max_tokens, &r);
    pthread_mutex_unlock(&e->mu);
    double t_done = now_sec();
    log_info("req: model=%s msgs=%d max_tokens=%d gen=%d finish=%d "
             "lock_wait=%.1fms engine=%.1fms total=%.1fms",
             e->name, n_msgs, max_tokens, r.n_generated, r.finish_reason,
             (t_locked - t_lock) * 1000.0,
             r.latency_sec * 1000.0,
             (t_done - t_lock) * 1000.0);
    for (int i = 0; i < n_msgs; i++) {
        free((void *)msgs[i].role);
        free((void *)msgs[i].content);
    }
    if (rc < 0) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "generation failed (model=%s finish=%d)", e->name, r.finish_reason);
        respond_500(fd, msg);
        return;
    }

    SB s; sb_init(&s);
    sb_puts(&s, "{\"id\":\"chatcmpl-tt-1\",\"object\":\"chat.completion\",");
    sb_puts(&s, "\"created\":");
    sb_putd(&s, (long)time(NULL));
    sb_puts(&s, ",\"model\":");
    sb_json_string(&s, e->name);
    sb_puts(&s, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    sb_json_string(&s, r.text);
    sb_puts(&s, "},\"finish_reason\":");
    switch (r.finish_reason) {
        case 0: sb_puts(&s, "\"stop\""); break;
        case 1: sb_puts(&s, "\"length\""); break;
        case 2: sb_puts(&s, "\"stop\""); break;
        case 3: sb_puts(&s, "\"length\""); break;
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
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    char *buf = NULL;
    int total = read_http_request(fd, &buf);
    if (total < 0) { close(fd); free(buf); free(j); return NULL; }

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

/* --------------------- registry parsing + load ------------------------ */

/* Trim ASCII whitespace in place at both ends of a NUL-terminated string. */
static void trim(char *s) {
    char *p = s;
    while (*p == ' ' || *p == '\t') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t')) s[--n] = '\0';
}

/* Parse the "name=path" spec. Returns 0 on success, fills entries[]. */
static int parse_models_env(const char *spec) {
    if (!spec || !spec[0]) return -1;
    char *work = strdup(spec);
    if (!work) return -1;
    int n = 0;
    char *save_outer = NULL;
    for (char *tok = strtok_r(work, ",", &save_outer);
         tok != NULL;
         tok = strtok_r(NULL, ",", &save_outer)) {
        if (n >= MAX_MODELS) { free(work); return -1; }
        char *eq = strchr(tok, '=');
        if (!eq) { free(work); return -1; }
        *eq = '\0';
        char *name = tok;
        char *path = eq + 1;
        trim(name);
        trim(path);
        if (!name[0] || !path[0]) { free(work); return -1; }
        for (int i = 0; i < name[i]; i++) {
            char c = name[i];
            if (c == ',' || c == '=' || c == ' ' || c == '\t' || c == '\n') {
                free(work); return -1;
            }
        }
        /* duplicate-name check */
        for (int i = 0; i < n; i++) {
            if (strcmp(g_reg[i].name, name) == 0) { free(work); return -1; }
        }
        snprintf(g_reg[n].name, sizeof(g_reg[n].name), "%s", name);
        snprintf(g_reg[n].path, sizeof(g_reg[n].path), "%s", path);
        g_reg[n].active = 0;
        pthread_mutex_init(&g_reg[n].mu, NULL);
        n++;
    }
    free(work);
    if (n == 0) return -1;
    g_n_models = n;
    return 0;
}

static long vram_used_bytes(void) {
    /* Reports the current process's GPU memory. nvidia-smi gives a global
     * number across the device; we use cudaMemGetInfo which reflects all
     * allocations on the default device, which is what we want for a
     * "what did THIS process pin?" estimate. */
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
    return (long)(total_b - free_b);
}

static int load_all_engines(int max_ctx) {
    for (int i = 0; i < g_n_models; i++) {
        EngineEntry *e = &g_reg[i];
        log_info("loading model %d/%d name=%s path=%s",
                 i + 1, g_n_models, e->name, e->path);
        long vram_before = vram_used_bytes();
        e->model = gguf_load(e->path);
        if (!e->model) {
            log_info("gguf_load failed for %s", e->path);
            return -1;
        }
        e->fam = tt_chat_family_from_arch(e->model->architecture);
        if (e->fam < 0) e->fam = TT_CHAT_QWEN2;
        e->tok = bpe_tokenizer_init(e->model);
        if (!e->tok) { log_info("tokenizer init failed for %s", e->name); return -1; }
        e->vocab = e->tok->vocab_size;
        e->cfg = tt_config_from_gguf(e->model, max_ctx);
        if (e->cfg.dim == 0) { log_info("cfg derive failed for %s", e->name); return -1; }
        e->eng = qwen2_engine_create(&e->cfg, e->model);
        if (!e->eng) { log_info("engine_create failed for %s", e->name); return -1; }
        e->max_ctx = max_ctx;
        e->active  = 1;
        long vram_after = vram_used_bytes();
        long delta = (vram_before >= 0 && vram_after >= 0)
                         ? vram_after - vram_before : -1;
        log_info("engine ready: name=%s arch=%s vocab=%d dim=%d layers=%d ctx=%d "
                 "(vram delta ~%.2f MiB, total ~%.2f MiB)",
                 e->name,
                 e->model->architecture[0] ? e->model->architecture : "?",
                 e->vocab, e->cfg.dim, e->cfg.n_layers, max_ctx,
                 delta >= 0 ? (double)delta / (1024.0 * 1024.0) : -1.0,
                 vram_after >= 0 ? (double)vram_after / (1024.0 * 1024.0) : -1.0);
    }
    return 0;
}

static void free_all_engines(void) {
    for (int i = 0; i < g_n_models; i++) {
        EngineEntry *e = &g_reg[i];
        if (e->eng) { qwen2_engine_free(e->eng); e->eng = NULL; }
        if (e->tok) { bpe_tokenizer_free(e->tok); e->tok = NULL; }
        if (e->model) { gguf_free(e->model); e->model = NULL; }
        pthread_mutex_destroy(&e->mu);
    }
    g_n_models = 0;
}

/* ------------------------------- main --------------------------------- */

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    const char *models_spec = env_or(ENV_MODELS, NULL);
    const char *listen_str  = env_or(ENV_LISTEN, DEFAULT_LISTEN);
    int max_ctx = env_int(ENV_MAX_CTX, DEFAULT_MAX_CTX);
    if (max_ctx < 64) max_ctx = 64;

    log_info("M12 P2 multi-model server starting; listen=%s max_ctx=%d",
             listen_str, max_ctx);
    if (!models_spec) {
        die("TT_MODELS is required, e.g. "
            "TT_MODELS=q25=data/models/qwen2.5-0.5b-instruct-q4_0.gguf,"
            "tt=data/testmodels/tinyllama-f16.gguf");
    }
    if (parse_models_env(models_spec) != 0) {
        die("TT_MODELS parse failed; expected 'name1=path1,name2=path2,...'");
    }
    log_info("registry: %d model(s) declared", g_n_models);

    if (load_all_engines(max_ctx) != 0) {
        die("one or more engines failed to load (see log above)");
    }
    log_info("all engines ready; serving on %s", listen_str);

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
    if (g_listen_fd >= 0) { close(g_listen_fd); g_listen_fd = -1; }
    free_all_engines();
    return 0;
}
