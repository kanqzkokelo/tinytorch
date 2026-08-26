// Interactive terminal chat over the M6-correct Qwen2 decode engine.
// Real prompt encoding, real prefill (KV cache populated), real generation.
//
// M-latest integration:
//   - prompt formatting via src/chat_template.h (family auto-detected from
//     GGUF general.architecture, multi-turn accumulation via tt_chat_history)
//   - sampling via src/samplers.h tt_sampler_chain (engine runs greedy; the
//     chain samples on the host from the engine's logits each step)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#include "async_printer.h"
#include "chat_template.h"
#include "samplers.h"

/* ---- configurable stop strings ----
 * Default marker comes from the chat family (tt_chat_stop_string); legacy
 * guards and any TT_STOP_STRINGS entries (';'-separated) are ADDED ON TOP,
 * so user overrides stay additive. */
#define MAX_STOP_STRINGS 40
#define MAX_STOP_LEN     63
static char g_stop[MAX_STOP_STRINGS][MAX_STOP_LEN + 1];
static int  g_nstop = 0;

static void chat_add_stop(const char *s, size_t n) {
    if (g_nstop >= MAX_STOP_STRINGS || n == 0 || n > MAX_STOP_LEN) return;
    memcpy(g_stop[g_nstop], s, n);
    g_stop[g_nstop][n] = '\0';
    g_nstop++;
}

static void chat_init_stop_strings(tt_chat_family fam) {
    const char *fs = tt_chat_stop_string(fam);
    if (fs) chat_add_stop(fs, strlen(fs));
    /* legacy hardcoded guards stay active regardless of family/env */
    chat_add_stop("<|endoftext|>", 13);
    chat_add_stop("<|im_start|>", 12);
    const char *src = getenv("TT_STOP_STRINGS");
    while (src && *src) {
        const char *semi = strchr(src, ';');
        size_t n = semi ? (size_t)(semi - src) : strlen(src);
        chat_add_stop(src, n);
        if (!semi) break;
        src = semi + 1;
    }
}

/* earliest occurrence of any stop string in NUL-terminated buf; NULL if none */
static const char *chat_find_stop(const char *buf) {
    const char *best = NULL;
    for (int i = 0; i < g_nstop; i++) {
        const char *hit = strstr(buf, g_stop[i]);
        if (hit && (!best || hit < best)) best = hit;
    }
    return best;
}

/* length of longest suffix of buf[0..n) that is a PROPER PREFIX of some stop
 * string — these bytes must be withheld until we know the marker's fate */
static size_t chat_holdback_len(const char *buf, size_t n) {
    size_t hold = 0;
    for (int i = 0; i < g_nstop; i++) {
        size_t slen = strlen(g_stop[i]);
        size_t maxl = slen - 1 < n ? slen - 1 : n;
        for (size_t l = maxl; l > hold; l--)
            if (memcmp(buf + n - l, g_stop[i], l) == 0) { hold = l; break; }
    }
    return hold;
}

/* ---- sampler-chain env knobs ---- */
static float env_float(const char *k, float dflt) {
    const char *v = getenv(k);
    return (v && v[0]) ? (float)atof(v) : dflt;
}
static int env_int(const char *k, int dflt) {
    const char *v = getenv(k);
    return (v && v[0]) ? atoi(v) : dflt;
}

/* recent-token window for repetition/frequency/presence penalties */
#define PENALTY_WINDOW 64
static int32_t g_recent[PENALTY_WINDOW];
static int     g_nrecent = 0;
static void recent_push(int32_t tok) {
    if (PENALTY_WINDOW > 1 && g_nrecent == PENALTY_WINDOW)
        memmove(g_recent, g_recent + 1, sizeof(int32_t) * (PENALTY_WINDOW - 1));
    if (g_nrecent < PENALTY_WINDOW) g_nrecent++;
    g_recent[g_nrecent - 1] = tok;
}

/* ---- multi-turn formatting state ---- */
static tt_chat_history g_hist;
static char   g_prev_fmt[16384];          /* last fully-formatted prompt     */
static size_t g_prev_len = 0;             /* bytes already fed through KV    */

static const char *env_or_empty(const char *k) {
    const char *v = getenv(k);
    return v ? v : "";
}

int main(void) {
    const char *model_path =
        (getenv("TT_MODEL") && getenv("TT_MODEL")[0])
            ? getenv("TT_MODEL")
            : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    const int MAX_CTX = 1024;

    printf("\n=======================================================\n");
    printf("   tinytorch chat — M6-correct Qwen2 decode engine\n");
    printf("   Model: %s\n", model_path);
    printf("   Type '/exit' to quit.\n");
    printf("=======================================================\n\n");

    GGUFModel *model = gguf_load(model_path);
    if (!model) return 1;

    /* chat template family straight from GGUF general.architecture */
    tt_chat_family fam = tt_chat_family_from_arch(model->architecture);
    if (fam < 0) {
        fprintf(stderr, "[chat] unknown arch '%s' — falling back to ChatML\n",
                model->architecture[0] ? model->architecture : "?");
        fam = TT_CHAT_QWEN2;
    }
    chat_init_stop_strings(fam);

    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok) { gguf_free(model); return 1; }

    /* cfg.vocab can be 0 when GGUF metadata omits <arch>.vocab_size; the
     * tokenizer's row count always matches the embedding matrix the engine
     * actually uses (engine re-derives it from token_embd shape itself) */
    const int VOCAB = tok->vocab_size;

    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    if (cfg.dim == 0) {
        fprintf(stderr, "[chat] could not derive config from GGUF metadata\n");
        bpe_tokenizer_free(tok); gguf_free(model); return 1;
    }
    printf("[chat] config: dim=%d ffn=%d layers=%d heads=%d kv_heads=%d head_dim=%d "
           "vocab=%d eps=%g rope_base=%g arch=%s\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
           cfg.head_dim, cfg.vocab, cfg.rms_eps, cfg.rope_base,
           model->architecture[0] ? model->architecture : "?");

    Qwen2Engine *eng = qwen2_engine_create(&cfg, model);
    if (!eng) { fprintf(stderr, "[chat] engine init failed\n"); return 1; }

    /* sampler chain from env knobs (supersedes ad-hoc temp/penalty code) */
    tt_sampler_chain sc;
    tt_sampler_chain_init(&sc);
    const int greedy = env_or_empty("TT_GREEDY")[0] != '\0';
    sc.greedy           = greedy;
    sc.temp             = env_float("TT_TEMP", 0.8f);
    sc.repeat_penalty   = env_float("TT_REPEAT_PENALTY", 1.15f);
    sc.top_k            = env_int("TT_TOP_K", 0);
    sc.top_p            = env_float("TT_TOP_P", 1.0f);
    sc.min_p            = env_float("TT_MIN_P", 0.0f);
    sc.freq_penalty     = env_float("TT_FREQ_PENALTY", 0.0f);
    sc.presence_penalty = env_float("TT_PRESENCE_PENALTY", 0.0f);
    sc.penalty_last_n   = PENALTY_WINDOW;
    sc.freq_last_n      = PENALTY_WINDOW;
    sc.use_rep_penalty  = sc.repeat_penalty != 1.0f;
    sc.use_freq_presence = sc.freq_penalty != 0.0f || sc.presence_penalty != 0.0f;

    /* xorshift64* state: TT_SEED for reproducibility, time-based otherwise */
    uint64_t rng_state;
    const char *se = getenv("TT_SEED");
    if (se && se[0]) rng_state = strtoull(se, NULL, 10);
    else rng_state = ((uint64_t)time(NULL) << 17) ^ (uint64_t)clock() ^
                     0x9E3779B97F4A7C15ull;
    if (!rng_state) rng_state = 1;
    if (!greedy)
        fprintf(stderr, "[chat] sampler: temp=%.2f top_k=%d top_p=%.2f min_p=%.2f "
                "rep=%.2f freq=%.2f pres=%.2f seed=%llu%s\n",
                sc.temp, sc.top_k, sc.top_p, sc.min_p, sc.repeat_penalty,
                sc.freq_penalty, sc.presence_penalty,
                (unsigned long long)(se && se[0] ? strtoull(se, NULL, 10) : 0),
                se && se[0] ? "" : " (time-based)");

    /* Engine default is GREEDY. On the graph path all stochastic sampling
     * happens host-side via tt_sampler_chain. The eager fallback
     * (TT_NO_GRAPH=1 / TT_PROFILE) has no injection point for host sampling,
     * so it keeps the engine's legacy GPU Gumbel-max path (temp + repeat
     * penalty only; top-p/min-p/freq/presence need the graph path). */
    const int no_graph = (getenv("TT_NO_GRAPH") || getenv("TT_PROFILE")) ? 1 : 0;
    if (no_graph && !greedy) {
        qwen2_engine_set_sampling(eng, sc.temp,
                                  sc.top_k > 0 ? sc.top_k : 40,
                                  sc.repeat_penalty);
        fprintf(stderr, "[chat] eager fallback: engine-side sampling "
                "(temp/repeat-penalty only)\n");
    }

    float *logits = malloc(sizeof(float) * (size_t)VOCAB);
    float *wb     = malloc(sizeof(float) * (size_t)tt_sampler_workbuf_size(VOCAB));
    if (!logits || !wb) { fprintf(stderr, "[chat] OOM (sampler buffers)\n"); return 1; }

    const int max_tokens = getenv("TT_MAX_TOKENS") && atoi(getenv("TT_MAX_TOKENS")) > 0
                               ? atoi(getenv("TT_MAX_TOKENS")) : 512;

    static const char *SYSTEM_PROMPT =
        "You are a helpful assistant. Respond in English by default unless "
        "the user writes in another language. Give complete, detailed answers.";
    tt_chat_history_init(&g_hist);
    tt_chat_history_push(&g_hist, "system", SYSTEM_PROMPT);
    const tt_chat_opts opts = tt_chat_opts_default();
    /* SP-mode gemma tokenizers auto-prepend bos_id in bpe_encode; emitting
     * the literal "<bos>" in fmt_gemma would be mis-tokenized as several
     * letter pieces ('<','bos','>',...) garbling the prompt prefix and
     * causing the model to stop after one short token. Suppress the
     * template's <bos> so the SP path inserts the single real BOS id. */
    if (fam == TT_CHAT_GEMMA || fam == TT_CHAT_GEMMA4) {
        tt_chat_opts *writable = (tt_chat_opts *)&opts;
        writable->add_bos_text = 0;
    }

    char user_input[1024];
    while (1) {
        printf("\nUser > ");
        fflush(stdout);
        if (!fgets(user_input, sizeof(user_input), stdin)) break;

        size_t len = strlen(user_input);
        while (len > 0 && (user_input[len - 1] == '\n' || user_input[len - 1] == '\r'))
            user_input[--len] = '\0';
        if (len == 0) continue;
        if (!strcmp(user_input, "/exit") || !strcmp(user_input, "quit")) break;

        /* accumulate + render the full conversation, then prefill ONLY the
         * byte-suffix that extends the previously fed prompt (KV cache keeps
         * prior turns; the split always lands on a '<|im_start|>' boundary) */
        tt_chat_history_push(&g_hist, "user", user_input);
        char formatted[sizeof(g_prev_fmt)];
        const int need = tt_chat_history_format(&g_hist, fam, &opts,
                                                formatted, sizeof(formatted));
        if (need < 0) { fprintf(stderr, "[chat] format failed (%d)\n", need); continue; }
        if ((size_t)need >= sizeof(formatted)) {
            fprintf(stderr, "[chat] formatted prompt truncated (%d bytes) — "
                    "shorten the conversation\n", need);
            continue;
        }
        if (g_prev_len > (size_t)need ||
            memcmp(formatted, g_prev_fmt, g_prev_len) != 0) {
            fprintf(stderr, "[chat] history desync — restart session\n");
            continue;
        }
        const char *suffix = formatted + g_prev_len;

        int prompt_tokens[512];
        int n_prompt = bpe_encode(tok, suffix, prompt_tokens, 512);
        if (n_prompt <= 0) { fprintf(stderr, "[chat] tokenization failed\n"); continue; }
        if (getenv("TT_DUMP_PROMPT")) {
            fprintf(stderr, "\n[TT_DUMP_PROMPT] prompt bytes (%zu):\n----\n%s\n----\n",
                    strlen(suffix), suffix);
            fprintf(stderr, "[TT_DUMP_PROMPT] first %d token ids:", n_prompt);
            for (int i = 0; i < n_prompt && i < 32; i++)
                fprintf(stderr, " %d", prompt_tokens[i]);
            fprintf(stderr, "\n");
        }

        AsyncPrinter *ap = async_printer_start();
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        if (qwen2_engine_prefill(eng, prompt_tokens, n_prompt)) {
            fprintf(stderr, "\n[chat] context full — restart session or shorten input\n");
            async_printer_stop_and_flush(ap);
            continue;
        }
        memcpy(g_prev_fmt, formatted, (size_t)need + 1);
        g_prev_len = (size_t)need;

        /* seed the penalty window with the prompt tail */
        for (int i = n_prompt > PENALTY_WINDOW ? n_prompt - PENALTY_WINDOW : 0;
             i < n_prompt; i++)
            recent_push((int32_t)prompt_tokens[i]);

        /* Materialize logits for the last prompt position WITHOUT advancing
         * (graph path only): first engine_next() call eager-samples greedily
         * and stashes the result as pending; we discard its choice and drive
         * every step ourselves via replay_step(). */
        if (!no_graph && qwen2_engine_next(eng) < 0) {
            fprintf(stderr, "\n[chat] failed to score prompt\n");
            async_printer_stop_and_flush(ap);
            continue;
        }

        int gen_count = 0;
        char turn_text[8192];
        size_t tl = 0;
        size_t emitted = 0;   /* bytes already streamed to the printer */
        int stop_cut = 0;
        while (gen_count < max_tokens && qwen2_engine_pos(eng) < MAX_CTX - 1) {
            int next_tok;
            if (no_graph) {
                /* eager engine samples AND feeds the token itself */
                next_tok = qwen2_engine_next(eng);
            } else {
                if (qwen2_debug_copy_logits(eng, logits, VOCAB) < 0) break;
                next_tok = tt_sample(logits, VOCAB, &sc, &rng_state, wb);
            }
            if (getenv("TT_DUMP_FIRST_TOK") && gen_count == 0) {
                fprintf(stderr, "[TT_DUMP_FIRST_TOK] next_tok=%d (eos=%d, eot=151645, endoftext=151643, gemma_end_of_turn=106)\n",
                        next_tok, tok->eos_id);
            }
            if (next_tok < 0 || next_tok == tok->eos_id ||
                next_tok == 151643 /* <|endoftext|> */ ||
                next_tok == 151645 /* <|im_end|> */) {
                /* close the assistant turn in the KV transcript: eager path
                 * already advanced; graph path feeds the id explicitly */
                if (!no_graph && qwen2_engine_pos(eng) < MAX_CTX - 1)
                    qwen2_debug_replay_step(eng, next_tok);
                break;
            }
            int out_len = 0;
            const char *s = bpe_decode_token(tok, next_tok, &out_len);
            /* stop-string guard: model sometimes spells control tokens as BPE
             * pieces instead of emitting their ids — truncate at the marker */
            if (tl + (size_t)out_len < sizeof(turn_text)) {
                memcpy(turn_text + tl, s, (size_t)out_len);
                tl += (size_t)out_len;
                turn_text[tl] = '\0';
            }
            /* stop-string check across the WHOLE buffer so markers spanning
             * token boundaries ("<end" + "_of_turn>") are caught too; any
             * suffix that is a prefix of a marker is withheld from streaming */
            const char *cut = chat_find_stop(turn_text);
            if (cut) {
                const size_t k = (size_t)(cut - turn_text);
                if (k > emitted)
                    async_printer_push(ap, turn_text + emitted,
                                       (int)(k - emitted));
                stop_cut = 1;
            } else {
                const size_t hold = chat_holdback_len(turn_text, tl);
                const size_t safe = tl - hold;
                if (safe > emitted) {
                    async_printer_push(ap, turn_text + emitted,
                                       (int)(safe - emitted));
                    emitted = safe;
                }
                gen_count++;
                recent_push((int32_t)next_tok);
            }
            /* feed OUR sampled token through the engine (computes the next
             * step's logits). The stopper token is fed TOO: the eager path
             * has already advanced it internally, and feeding it on the graph
             * path keeps both KV transcripts identical (gate cross-checks). */
            if (qwen2_engine_pos(eng) >= MAX_CTX - 1) break;
            if (!no_graph && qwen2_debug_replay_step(eng, next_tok) < 0) break;
            if (stop_cut) break;
        }
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        async_printer_stop_and_flush(ap);

        /* Eager parity probe: the graph path leaves a pending token that the
         * next prefill flushes (one greedy step past the last fed token); an
         * explicit step here gives the eager path the same trailing token so
         * both modes see byte-identical context next turn. */
        if (no_graph && qwen2_engine_pos(eng) < MAX_CTX - 1)
            qwen2_engine_next(eng);

        /* record what the model actually said so future prompts include it;
         * the formatted delta is NOT re-prefilled — those bytes are already
         * in the KV cache as the tokens generated above (the cursor below
         * just skips them). */
        tt_chat_history_push(&g_hist, "assistant", turn_text);
        {
            /* Snapshot WITHOUT the trailing generation prompt: the prompt
             * header moves to the end of the string on the next render, so
             * the fed-prefix cursor must stop right after the assistant
             * turn's <|im_end|> for the prefix check to stay valid. */
            tt_chat_opts snap = opts;
            snap.add_generation_prompt = 0;
            char refmt[sizeof(g_prev_fmt)];
            const int rneed = tt_chat_history_format(&g_hist, fam, &snap,
                                                     refmt, sizeof(refmt));
            if (rneed >= 0 && (size_t)rneed < sizeof(refmt)) {
                memcpy(g_prev_fmt, refmt, (size_t)rneed + 1);
                g_prev_len = (size_t)rneed;
            }
        }

        const double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        const double tps = gen_count > 0 ? gen_count / sec : 0.0;
        printf("\n\n[%d tokens | %.1f ms | %.1f tok/s | ctx %d/%d]\n",
               gen_count, sec * 1000.0, tps,
               qwen2_engine_pos(eng), MAX_CTX);

        if (qwen2_engine_pos(eng) >= MAX_CTX - 8) {
            fprintf(stderr, "[chat] context nearly full — session should be reset\n");
        }
    }

    free(logits); free(wb);
    qwen2_engine_free(eng);
    bpe_tokenizer_free(tok);
    gguf_free(model);
    return 0;
}
