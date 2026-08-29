// Single-shot generation over the M6-correct engine.
// The prompt is tokenized, prefilled into the KV cache, and generation
// continues from it — the prompt actually reaches the model.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#include "async_printer.h"

typedef struct {
    int id;
    float val;
} TokenScore;

static int compare_token_scores(const void *a, const void *b) {
    float diff = ((const TokenScore *)b)->val - ((const TokenScore *)a)->val;
    return (diff > 0.0f) - (diff < 0.0f);
}

int sample_token_advanced(const float *logits, int vocab_size,
                          const int *history, int history_len,
                          float temp, int top_k, float top_p, float rep_penalty) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    TokenScore *scores = (TokenScore *)malloc(sizeof(TokenScore) * vocab_size);
    if (!scores) return 0;
    for (int i = 0; i < vocab_size; i++) {
        scores[i].id = i;
        scores[i].val = logits[i];
    }

    if (rep_penalty > 1.0f && history && history_len > 0) {
        for (int i = 0; i < history_len; i++) {
            int tok = history[i];
            if (tok >= 0 && tok < vocab_size) {
                if (scores[tok].val > 0.0f) scores[tok].val /= rep_penalty;
                else scores[tok].val *= rep_penalty;
            }
        }
    }

    for (int i = 0; i < vocab_size; i++) scores[i].val /= temp;

    qsort(scores, vocab_size, sizeof(TokenScore), compare_token_scores);

    int cutoff = vocab_size;
    if (top_k > 0 && top_k < cutoff) cutoff = top_k;

    float max_val = scores[0].val;
    float sum_exp = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        scores[i].val = expf(scores[i].val - max_val);
        sum_exp += scores[i].val;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cum_prob = 0.0f;
        int p_cutoff = cutoff;
        for (int i = 0; i < cutoff; i++) {
            cum_prob += scores[i].val / sum_exp;
            if (cum_prob >= top_p) {
                p_cutoff = i + 1;
                break;
            }
        }
        cutoff = p_cutoff;
    }

    sum_exp = 0.0f;
    for (int i = 0; i < cutoff; i++) sum_exp += scores[i].val;

    float r = ((float)rand() / (float)RAND_MAX) * sum_exp;
    float acc = 0.0f;
    int sampled_id = scores[0].id;
    for (int i = 0; i < cutoff; i++) {
        acc += scores[i].val;
        if (r <= acc) {
            sampled_id = scores[i].id;
            break;
        }
    }

    free(scores);
    return sampled_id;
}

int main(int argc, char **argv) {
    const char *prompt = "Explain quantum computing in one sentence.";
    int target_tokens = 64;
    float temp = 0.0f;
    float top_p = 0.9f;
    int top_k = 0;
    float rep_penalty = 1.1f;

    int pos_arg = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--temp") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                temp = (float)atof(argv[++i]);
            } else {
                temp = 0.7f;
            }
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            rep_penalty = (float)atof(argv[++i]);
        } else if (argv[i][0] != '-') {
            if (pos_arg == 0) prompt = argv[i];
            else if (pos_arg == 1) target_tokens = atoi(argv[i]);
            pos_arg++;
        }
    }
    if (target_tokens < 1) target_tokens = 1;

    const char *se = getenv("TT_SEED");
    if (se && se[0]) {
        srand((unsigned int)strtoul(se, NULL, 10));
    } else {
        srand((unsigned int)time(NULL));
    }

    const char *model_path = getenv("TT_MODEL") ? getenv("TT_MODEL")
        : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    const int MAX_CTX = getenv("TT_MAX_CTX") ? atoi(getenv("TT_MAX_CTX")) : 1024;

    GGUFModel *model = gguf_load(model_path);
    if (!model) return 1;
    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok)
        /* M7 task 3 smoke path: SentencePiece models (gemma/llama-vocab)
         * decode in Task 4. Engine plumbing still gets exercised with
         * placeholder ids. */
        fprintf(stderr, "[run] WARN: BPE tokenizer unavailable for %s "
                "(arch=%s); feeding placeholder ids, text output invalid\n",
                model_path, model->architecture[0] ? model->architecture : "?");

    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    if (cfg.dim == 0) { fprintf(stderr, "config failed\n"); return 1; }

    int vocab_size = tok ? tok->vocab_size : 0;
    if (vocab_size <= 0) {
        GGUFTensor *tembd = gguf_get_tensor(model, "token_embd.weight");
        if (tembd) vocab_size = (int)tembd->shape[tembd->ndim - 1];
    }
    cfg.vocab = vocab_size;

    printf("[run] dim=%d ffn=%d layers=%d heads=%d kv_heads=%d vocab=%d rope_base=%g\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
           cfg.vocab, cfg.rope_base);

    Qwen2Engine *eng = qwen2_engine_create(&cfg, model);
    if (!eng) { fprintf(stderr, "engine init failed\n"); return 1; }

    char formatted[262144];
    if (!getenv("TT_RAW_PROMPT"))
        snprintf(formatted, sizeof(formatted),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    else
        snprintf(formatted, sizeof(formatted), "%s", prompt);

    int prompt_tokens[16384];
    int n_prompt;
    if (tok) {
        n_prompt = bpe_encode(tok, formatted, prompt_tokens, 16384);
    } else {
        /* placeholder prefill: valid ids within any vocab */
        const int smoke_ids[5] = {1, 2, 3, 4, 5};
        memcpy(prompt_tokens, smoke_ids, sizeof(smoke_ids));
        n_prompt = 5;
    }
    printf("[run] prompt: %d tokens\n", n_prompt);
    if (n_prompt <= 0) return 1;

    struct timespec t0, t1, tp0, tp1;
    AsyncPrinter *ap = async_printer_start();
    clock_gettime(CLOCK_MONOTONIC, &tp0);

    if (qwen2_engine_prefill(eng, prompt_tokens, n_prompt)) {
        fprintf(stderr, "prefill failed\n"); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);   /* decode-only window starts here */

    int history[MAX_CTX];
    int history_len = 0;
    for (int i = 0; i < n_prompt && history_len < MAX_CTX; i++) {
        history[history_len++] = prompt_tokens[i];
    }

    float *logits = (float *)malloc(sizeof(float) * vocab_size);
    if (!logits) { fprintf(stderr, "OOM logits\n"); return 1; }

    if (temp > 0.0f) {
        if (qwen2_engine_next(eng) < 0) {
            fprintf(stderr, "prompt scoring failed\n");
            free(logits);
            return 1;
        }
    }

    int gen_count = 0;
    char turn_text[8192];
    size_t tl = 0;
    for (int s = 0; s < target_tokens && qwen2_engine_pos(eng) < MAX_CTX - 1; s++) {
        int id;
        if (temp <= 0.0f) {
            id = qwen2_engine_next(eng);
        } else {
            if (qwen2_debug_copy_logits(eng, logits, vocab_size) < 0) break;
            id = sample_token_advanced(logits, vocab_size, history, history_len,
                                       temp, top_k, top_p, rep_penalty);
        }

        if (id < 0 || (tok && (id == tok->eos_id || id == 151645))) break;
        if (!tok) { gen_count++; continue; }   /* no decoder yet: count only */
        int out_len = 0;
        const char *txt = bpe_decode_token(tok, id, &out_len);
        if (tl + (size_t)out_len < sizeof(turn_text)) {
            memcpy(turn_text + tl, txt, (size_t)out_len);
            tl += (size_t)out_len;
            turn_text[tl] = '\0';
        }
        if (id == tok->eos_id) {
            break;
        }
        async_printer_push(ap, txt, out_len);
        gen_count++;

        if (history_len < MAX_CTX) {
            history[history_len++] = id;
        }

        if (temp > 0.0f && qwen2_engine_pos(eng) < MAX_CTX - 1) {
            qwen2_debug_replay_step(eng, id);
        }
    }
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    async_printer_stop_and_flush(ap);

    const double dec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    const double tot = (t1.tv_sec - tp0.tv_sec) + (t1.tv_nsec - tp0.tv_nsec) * 1e-9;
    const double prefill_sec = (t0.tv_sec - tp0.tv_sec) + (t0.tv_nsec - tp0.tv_nsec) * 1e-9;
    printf("\"\n[gen: %d tokens | decode %.1f tok/s | prefill %.1f tok/s | incl prefill %.1f tok/s | %s]\n",
           gen_count, gen_count / dec, n_prompt / prefill_sec, gen_count / tot, temp > 0.0f ? "sampled" : "greedy");
    printf("STATS tokens=%d prefill=%d decode_us=%.0f prefill_us=%.0f prefill_tok_s=%.1f\n", gen_count, n_prompt, dec * 1e6, prefill_sec * 1e6, n_prompt / prefill_sec);

    free(logits);
    qwen2_engine_free(eng);
    if (tok) bpe_tokenizer_free(tok);
    gguf_free(model);
    return 0;
}
