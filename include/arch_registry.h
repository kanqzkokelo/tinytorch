#ifndef ARCH_REGISTRY_H
#define ARCH_REGISTRY_H

/* M7 task 3: architecture trait registry.
 * Maps the GGUF `general.architecture` string to per-family forward-pass
 * defaults. Magic values are cited against oracle/llama.cpp source in
 * src/arch_registry.c. Zeroed TTraits = NEOX rope + SiLU + everything off,
 * which is exactly today's Qwen2 behavior. */

#ifdef __cplusplus
extern "C" {
#endif

#include "loader_gguf.h"

typedef enum { ROPE_NEOX = 0, ROPE_GPTJ = 1 } RopeStyle;
typedef enum { ACT_SILU = 0, ACT_GELU = 1 } Activation;

typedef struct {
    RopeStyle rope;          /* half-split vs interleaved-pair rotary      */
    Activation act;          /* SwiGLU vs GeGLU (gemma) epilogue           */
    float softcap_value;     /* final-logits tanh softcap, 0 = disabled    */
    int   swa_size;          /* sliding-window attention size, 0 = full    */
    int   tied_embeddings;   /* lm_head aliases token_embd                 */
    int   qk_norm_rms;       /* rmsnorm on q/k heads pre-rope (qwen3)      */
    float qk_norm_eps;
    float norm_offset;       /* extra added to every rmsnorm gamma         */
    int   embed_sqrt;        /* gemma: scale embeddings by sqrt(dim)       */
                             /* NOTE gemma's (1+w) is baked into GGUF weights
                              * at conversion (oracle convert_hf_to_gguf.py:4730),
                              * so this stays 0 for converted models.        */
} TTraits;

struct GGUFModel_unused;

/* Defaults for a known architecture string ("qwen2", "llama", "qwen3",
 * "gemma", "gemma2"). Returns NULL for unknown arch. */
const TTraits *tt_traits_lookup(const char *arch);

/* Resolve traits for a loaded model: registry defaults + GGUF metadata
 * overrides (sliding_window / final_logit_softcapping when present).
 * Returns 0 and fills *out on success, -1 if m->architecture is unknown. */
int tt_traits_resolve(const GGUFModel *m, TTraits *out);

/* Comma-separated list of supported architecture strings (for error text). */
const char *tt_traits_supported(void);

#ifdef __cplusplus
}
#endif
#endif /* ARCH_REGISTRY_H */
