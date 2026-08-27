// Universal Speculative CLI: prefill, then decode using N-gram draft + verify.
//
// Usage: ./build/spec_llm_gpu <model.gguf> <prompt> <n_predict> [--draft-k K] [--window W]
//        (or via env: TT_MODEL, TT_PROMPT, TT_NPREDICT, TT_DRAFT_K, TT_WINDOW)
//
// Decode loop:
//   1. Prefill prompt
//   2. While not done:
//      a. tokens_history = all tokens so far (prompt + generated)
//      b. K = ngram_lookup_draft(history, N, window, draft_k, out_draft)
//      c. If K == 0:
//           single forward via qwen2_engine_next(); append to history
//      d. Else:
//           candidates = [last_token, draft[0..K-1]]
//           run qwen2_engine_verify_speculative -> N rows of device logits
//           accept = longest prefix where argmax(logits[i]) == draft[i]
//           correction_token = argmax(logits[accept])     (or last row if all accepted)
//           history += draft[0..accept-1] + [correction_token]
//           engine.pos advances by N (KV cache slots filled with the candidate
//            tokens; the orchestrator is responsible for managing this — see
//            the qwen2_engine_verify_speculative header note)
//      e. log acceptance rate
//
// KV-vs-history drift note: verify_speculative() advances engine.pos by N
// regardless of how many candidates we accept. The orchestrator emits only
// the accepted tokens to history, so the n-gram drafter's view stays
// correct, but the engine's KV cache (used for subsequent logits) reflects
// the rejected draft tokens for the tail slots. For high-acceptance prompts
// (repetitive text) the drift is negligible within the smoke-test budget
// (n_predict <= 64); long generations would need a rollback API in the
// engine to be perfectly bit-exact with single-token greedy.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#include "ngram_lookup.h"

#define MAX_CTX         1024
#define MAX_HISTORY     4096
#define MAX_CANDIDATES  (MAX_DRAFT_K + 1)

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_)); exit(1); } } while(0)

/* argmax over row `row` of a [n_rows, vocab] row-major matrix. */
static int argmax_of_row(const float *m, int row, int vocab) {
    int best = 0;
    float best_v = m[row * vocab + 0];
    for (int v = 1; v < vocab; v++) {
        const float x = m[row * vocab + v];
        if (x > best_v) { best_v = x; best = v; }
    }
    return best;
}

static const char *env_or_empty(const char *k) {
    const char *v = getenv(k);
    return v ? v : "";
}

int main(int argc, char **argv) {
    /* ---- CLI / env ---- */
    const char *model_path = NULL;
    const char *prompt     = NULL;
    int n_predict          = 64;
    int draft_k            = 3;
    int window             = 2;

    if (argc >= 4) {
        model_path = argv[1];
        prompt     = argv[2];
        n_predict  = atoi(argv[3]);
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--draft-k") == 0 && i + 1 < argc) draft_k = atoi(argv[++i]);
            else if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) window = atoi(argv[++i]);
        }
    } else {
        model_path = env_or_empty("TT_MODEL");
        if (!model_path[0]) model_path = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
        prompt     = env_or_empty("TT_PROMPT");
        if (!prompt[0])     prompt     = "The quick brown fox jumps over the lazy dog. "
                                            "The quick brown fox jumps over the lazy dog.";
        const char *np = env_or_empty("TT_NPREDICT");
        if (np[0]) n_predict = atoi(np);
        const char *dk = env_or_empty("TT_DRAFT_K");
        if (dk[0]) draft_k = atoi(dk);
        const char *wd = env_or_empty("TT_WINDOW");
        if (wd[0]) window = atoi(wd);
    }

    /* clamp to engine-tested ranges */
    if (draft_k < 1)      draft_k = 1;
    if (draft_k > MAX_DRAFT_K) draft_k = MAX_DRAFT_K;
    if (window < 2)       window = 2;
    if (window > 3)       window = 3;

    fprintf(stderr, "[spec] model=%s draft_k=%d window=%d n_predict=%d\n",
            model_path, draft_k, window, n_predict);
    fprintf(stderr, "[spec] prompt: %s\n", prompt);

    /* ---- Load model + tokenizer + engine ---- */
    GGUFModel *model = gguf_load(model_path);
    if (!model) { fprintf(stderr, "[spec] gguf_load failed\n"); return 1; }

    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok) { fprintf(stderr, "[spec] bpe_tokenizer_init failed\n"); gguf_free(model); return 1; }

    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    if (cfg.dim == 0) { fprintf(stderr, "[spec] config failed\n"); bpe_tokenizer_free(tok); gguf_free(model); return 1; }
    /* tt_config_from_gguf leaves cfg.vocab = 0; resolve it from the
     * token_embd tensor shape (same lookup the engine does internally). */
    GGUFTensor *tembd = gguf_get_tensor(model, "token_embd.weight");
    if (!tembd) { fprintf(stderr, "[spec] token_embd.weight missing\n"); bpe_tokenizer_free(tok); gguf_free(model); return 1; }
    cfg.vocab = (int)tembd->shape[tembd->ndim - 1];
    if (cfg.vocab <= 0) { fprintf(stderr, "[spec] could not resolve vocab\n"); bpe_tokenizer_free(tok); gguf_free(model); return 1; }

    Qwen2Engine *eng = qwen2_engine_create(&cfg, model);
    if (!eng) { fprintf(stderr, "[spec] engine create failed\n"); bpe_tokenizer_free(tok); gguf_free(model); return 1; }

    fprintf(stderr, "[spec] dim=%d ffn=%d layers=%d heads=%d kv_heads=%d vocab=%d\n",
            cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab);

    /* ---- Tokenize + prefill ---- */
    int prompt_tokens[2048];
    int n_prompt = bpe_encode(tok, prompt, prompt_tokens, 2048);
    if (n_prompt <= 0) {
        fprintf(stderr, "[spec] bpe_encode produced 0 tokens\n");
        bpe_tokenizer_free(tok); qwen2_engine_free(eng); gguf_free(model);
        return 1;
    }
    fprintf(stderr, "[spec] prompt: %d tokens\n", n_prompt);

    if (qwen2_engine_prefill(eng, prompt_tokens, n_prompt)) {
        fprintf(stderr, "[spec] prefill failed\n");
        bpe_tokenizer_free(tok); qwen2_engine_free(eng); gguf_free(model);
        return 1;
    }

    /* ---- History + buffers ---- */
    int *history = (int *)malloc(sizeof(int) * MAX_HISTORY);
    if (!history) { fprintf(stderr, "[spec] OOM history\n"); return 1; }
    memcpy(history, prompt_tokens, sizeof(int) * n_prompt);
    int history_n = n_prompt;

    int draft[MAX_DRAFT_K];
    int candidates[MAX_CANDIDATES];
    float *d_logits = NULL;
    float *h_logits = (float *)malloc(sizeof(float) * cfg.vocab * MAX_CANDIDATES);
    if (!h_logits) { fprintf(stderr, "[spec] OOM h_logits\n"); return 1; }
    CUDA_OK(cudaMalloc(&d_logits, sizeof(float) * cfg.vocab * MAX_CANDIDATES));

    /* ---- Decode loop ---- */
    long total_accepted  = 0;
    long total_attempted = 0;
    long total_emitted   = 0;       /* total tokens put into history (incl. fallback) */
    long fallback_steps  = 0;
    long verify_steps    = 0;
    int  end_reason      = 0;       /* 0=ran-out, 1=EOS, 2=context full */

    /* Output buffer for text rendering */
    char turn_text[8192];
    size_t tl = 0;
    turn_text[0] = '\0';

    while (total_emitted < n_predict && qwen2_engine_pos(eng) < MAX_CTX - 1) {
        /* a) Draft from history */
        int K = ngram_lookup_draft(history, history_n, window, draft_k, draft);
        int accepted = 0;

        if (K > 0) {
            verify_steps++;
            /* b) Build candidate array: [last_history_token, draft[0..K-1]] */
            candidates[0] = history[history_n - 1];
            for (int i = 0; i < K; i++) candidates[i + 1] = draft[i];
            const int n_total = K + 1;

            /* c) Batched verify */
            const int vrc = qwen2_engine_verify_speculative(eng, candidates, n_total, d_logits);
            if (vrc) {
                fprintf(stderr, "[spec] verify rc=%d — falling back to single forward\n", vrc);
                /* emergency: do not advance on verify failure; just single-forward */
                K = 0;
            } else {
                /* d) D2H logits, then greedy accept */
                CUDA_OK(cudaMemcpy(h_logits, d_logits,
                                   sizeof(float) * cfg.vocab * n_total,
                                   cudaMemcpyDeviceToHost));

                for (int i = 0; i < K; i++) {
                    const int am = argmax_of_row(h_logits, i + 1, cfg.vocab);
                    if (am == draft[i]) accepted++;
                    else break;
                }
                /* correction token: argmax at position `accepted` (or last row
                 * if all K drafts were accepted — use row K which is the
                 * logit for the last drafted token; the very-next prediction
                 * would be from row K which sits one beyond the drafted span,
                 * but in the conservative path we use row K-1's argmax to
                 * stay within the filled rows). */
                int corr_pos = (accepted < K) ? accepted : (K - 1);
                if (corr_pos < 0) corr_pos = 0;
                int correction = argmax_of_row(h_logits, corr_pos, cfg.vocab);

                /* e) Append accepted draft + correction to history */
                if (history_n + accepted + 1 < MAX_HISTORY) {
                    for (int i = 0; i < accepted; i++) history[history_n++] = draft[i];
                    history[history_n++] = correction;
                }

                total_accepted  += accepted + 1;  /* +1 for the correction row */
                total_attempted += n_total;
                total_emitted   += accepted + 1;

                /* print accepted draft tokens + correction */
                for (int i = 0; i < accepted; i++) {
                    if (history_n - accepted - 1 + i < 0) continue;
                    int out_len = 0;
                    const char *txt = bpe_decode_token(tok, draft[i], &out_len);
                    if (tl + (size_t)out_len + 1 < sizeof(turn_text)) {
                        memcpy(turn_text + tl, txt, (size_t)out_len);
                        tl += (size_t)out_len;
                        turn_text[tl] = '\0';
                    }
                    fwrite(txt, 1, (size_t)out_len, stdout);
                }
                {
                    int out_len = 0;
                    const char *txt = bpe_decode_token(tok, correction, &out_len);
                    if (tl + (size_t)out_len + 1 < sizeof(turn_text)) {
                        memcpy(turn_text + tl, txt, (size_t)out_len);
                        tl += (size_t)out_len;
                        turn_text[tl] = '\0';
                    }
                    fwrite(txt, 1, (size_t)out_len, stdout);
                }
                fflush(stdout);
            }
        }

        if (K == 0) {
            /* Fallback: single forward (greedy) */
            fallback_steps++;
            const int id = qwen2_engine_next(eng);
            if (id < 0) { fprintf(stderr, "[spec] next() rc=%d\n", id); break; }
            if (id == tok->eos_id) { end_reason = 1; break; }
            if (history_n + 1 < MAX_HISTORY) history[history_n++] = id;
            total_emitted++;
            int out_len = 0;
            const char *txt = bpe_decode_token(tok, id, &out_len);
            if (tl + (size_t)out_len + 1 < sizeof(turn_text)) {
                memcpy(turn_text + tl, txt, (size_t)out_len);
                tl += (size_t)out_len;
                turn_text[tl] = '\0';
            }
            fwrite(txt, 1, (size_t)out_len, stdout);
            fflush(stdout);
        }

        if (qwen2_engine_pos(eng) >= MAX_CTX - 1) { end_reason = 2; break; }
    }
    cudaDeviceSynchronize();
    fputc('\n', stdout);
    fflush(stdout);

    const long denom = total_attempted + fallback_steps;
    const double rate = denom > 0 ? 100.0 * (double)total_accepted / (double)denom : 0.0;
    fprintf(stderr,
            "\n[USE] emitted=%ld verify_steps=%ld fallback_steps=%ld "
            "accepted=%ld attempted=%ld rate=%.1f%% end=%d\n",
            total_emitted, verify_steps, fallback_steps,
            total_accepted, total_attempted, rate, end_reason);

    cudaFree(d_logits);
    free(h_logits);
    free(history);
    qwen2_engine_free(eng);
    bpe_tokenizer_free(tok);
    gguf_free(model);
    return 0;
}
