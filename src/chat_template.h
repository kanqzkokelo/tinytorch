/* chat_template.h -- deterministic, hand-written chat prompt formatters.
 *
 * Covers the model families this repo ships (see src/arch_registry.c):
 *   qwen2, qwen3   -> ChatML  (<|im_start|>/<|im_end|>)
 *   gemma, gemma2  -> Gemma   (<start_of_turn>/<end_of_turn>)
 *   gemma4         -> Gemma   (same turn format as gemma/gemma2)
 *   llama          -> Llama 3 (<|start_header_id|>/<|eot_id|>)
 *
 * Templates were transcribed by hand (no Jinja) from the official HF
 * tokenizer_config.json chat_template fields:
 *   - google/gemma-3-4b-it        (mirror: unsloth/gemma-3-4b-it)
 *   - Qwen/Qwen2.5-Instruct       (ChatML, verified vs Qwen3-2507 variant)
 *   - Qwen/Qwen3-Instruct         (+ thinking-channel handling)
 *   - meta-llama/Llama-3.2-1B-Instruct (mirror: unsloth/Llama-3.2-1B-Instruct)
 * Deliberate deviations are marked DEVIATION below and in chat_template.c.
 *
 * Build: picked up automatically by the top-level Makefile's wildcard
 *   over src .c files (libtinytorch.so / pybind). Standalone:
 *   cc -std=c11 -O2 -Iinclude -shared -fPIC -o libchat_template.so \
 *      src/chat_template.c      (header-only consumer: just #include it)
 *
 * Migration note for examples/chat_llm_gpu.c: replace the hardcoded
 * ChatML snprintf block (~line 128) with tt_chat_history_push +
 * tt_chat_format(fam_from_arch(model->architecture), ...), and derive
 * stop strings from tt_chat_stop_string().
 */

#ifndef CHAT_TEMPLATE_H
#define CHAT_TEMPLATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- families ---------------------------------------------------------- */

typedef enum {
    TT_CHAT_QWEN2 = 0,   /* ChatML                                  */
    TT_CHAT_QWEN3,       /* ChatML + <think> channel handling       */
    TT_CHAT_GEMMA,       /* gemma / gemma2                          */
    TT_CHAT_GEMMA4,      /* identical turn format to TT_CHAT_GEMMA  */
    TT_CHAT_LLAMA3,      /* llama arch (Meta-Llama-3 / 3.x instruct)*/
} tt_chat_family;

/* Map a GGUF general.architecture string ("qwen2", "qwen3", "gemma",
 * "gemma2", "gemma4", "llama") to a family. Returns -1 if unknown. */
int tt_chat_family_from_arch(const char *arch);

/* The string that terminates an assistant generation for this family:
 * "<|im_end|>" (qwen), "<end_of_turn>" (gemma), "<|eot_id|>" (llama3).
 * Feed these to the stop-string machinery in examples/chat_llm_gpu.c. */
const char *tt_chat_stop_string(tt_chat_family fam);

/* ---- messages ---------------------------------------------------------- */

typedef struct {
    const char *role;     /* "system" | "user" | "assistant" */
    const char *content;  /* NUL-terminated UTF-8; NULL treated as "" */
} tt_msg;

typedef struct {
    int add_generation_prompt; /* append trailing assistant header (default 1) */
    int keep_think;            /* qwen3: keep <think>..</think> in assistant
                                * history instead of stripping (default 0)   */
    int add_empty_think;       /* qwen3 thinking-disabled mode: append
                                * "<think>\n\n</think>\n\n" after the final
                                * generation prompt (default 0)              */
    int add_bos_text;          /* gemma: leading "<bos>"; llama3: leading
                                * "<|begin_of_text|>" (default 1). Disable if
                                * the tokenizer/engine injects BOS as an id. */
    const char *date_string;   /* llama3 only: "Today Date:" line content.
                                * NULL -> no Cutting-Knowledge/Today preamble
                                * (matches Llama-3.0; set e.g. "26 Jul 2024"
                                * for byte-exact 3.1/3.2 output).            */
} tt_chat_opts;

/* Default options: generation prompt on, everything else off. */
tt_chat_opts tt_chat_opts_default(void);

/* Format a conversation. snprintf return semantics: returns the TOTAL
 * length the formatted prompt would need (excluding NUL). If that is
 * >= cap, output was truncated; call again with a larger buffer.
 * Negative on error: -1 unknown family, -2 bad arguments (NULL out,
 * cap == 0, n > 0 with NULL msgs). */
int tt_chat_format_ex(tt_chat_family fam, const tt_msg *msgs, int n,
                      const tt_chat_opts *opts, char *out, size_t cap);

/* Convenience wrapper: tt_chat_format_ex with tt_chat_opts_default(). */
int tt_chat_format(tt_chat_family fam, const tt_msg *msgs, int n,
                   char *out, size_t cap);

/* ---- multi-turn accumulation helper ------------------------------------ */

#define TT_CHAT_MAX_MSGS     64
#define TT_CHAT_MAX_CONTENT  4096

typedef struct {
    int   n;
    char  role[TT_CHAT_MAX_MSGS][16];
    char  content[TT_CHAT_MAX_MSGS][TT_CHAT_MAX_CONTENT]; /* truncated */
} tt_chat_history;

void tt_chat_history_init(tt_chat_history *h);

/* Append one message. Returns 0 on success, -1 if full (or h NULL).
 * role/content longer than storage is silently truncated. */
int tt_chat_history_push(tt_chat_history *h,
                         const char *role, const char *content);

/* Format the accumulated history (generation prompt appended).
 * Same return semantics as tt_chat_format_ex. */
int tt_chat_history_format(const tt_chat_history *h, tt_chat_family fam,
                           const tt_chat_opts *opts, char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* CHAT_TEMPLATE_H */
