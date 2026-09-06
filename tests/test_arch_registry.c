/* Tier-1 registry resolve test: tt_traits_resolve() per new key.
 * Build: gcc -std=c11 -O2 -Wall -Wextra -Iinclude -o build/test_arch_registry \
 *          tests/test_arch_registry.c src/arch_registry.c src/loader_gguf.c -lm
 * Run: ./build/test_arch_registry (exit 0 = all pass).
 */
#include <stdio.h>
#include <string.h>

#include "arch_registry.h"
#include "loader_gguf.h"

static int fails = 0;
static int checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
} while (0)

static int resolve_as(const char *arch, TTraits *out) {
    GGUFModel m;
    memset(&m, 0, sizeof(m));
    if (arch) {
        strncpy(m.architecture, arch, sizeof(m.architecture) - 1);
    }
    return tt_traits_resolve(&m, out);
}

int main(void) {
    TTraits t;

    /* internlm2: GPTJ + SILU, plain untied */
    CHECK(resolve_as("internlm2", &t) == 0, "internlm2 resolve rc");
    CHECK(t.rope == ROPE_GPTJ && t.act == ACT_SILU, "internlm2 rope/act");
    CHECK(t.tied_embeddings == 0 && t.qk_norm_rms == 0, "internlm2 tied/qknorm");
    CHECK(t.softcap_value == 0.0f && t.swa_size == 0, "internlm2 softcap/swa");

    /* xverse: GPTJ + SILU, plain untied */
    CHECK(resolve_as("xverse", &t) == 0, "xverse resolve rc");
    CHECK(t.rope == ROPE_GPTJ && t.act == ACT_SILU, "xverse rope/act");
    CHECK(t.tied_embeddings == 0 && t.qk_norm_rms == 0, "xverse tied/qknorm");

    /* exaone: NEOX + SILU + tied (llama.cpp output TENSOR_DUPLICATED) */
    CHECK(resolve_as("exaone", &t) == 0, "exaone resolve rc");
    CHECK(t.rope == ROPE_NEOX && t.act == ACT_SILU, "exaone rope/act");
    CHECK(t.tied_embeddings == 1 && t.qk_norm_rms == 0, "exaone tied/qknorm");

    /* ernie4_5 (underscore key): NORM=GPTJ + SILU + tied */
    CHECK(resolve_as("ernie4_5", &t) == 0, "ernie4_5 resolve rc");
    CHECK(t.rope == ROPE_GPTJ && t.act == ACT_SILU, "ernie4_5 rope/act");
    CHECK(t.tied_embeddings == 1 && t.qk_norm_rms == 0, "ernie4_5 tied/qknorm");

    /* default path untouched: existing families resolve as before */
    CHECK(resolve_as("qwen2", &t) == 0, "qwen2 rc");
    CHECK(t.rope == ROPE_NEOX && t.act == ACT_SILU && t.tied_embeddings == 0, "qwen2 traits");
    CHECK(resolve_as("llama", &t) == 0, "llama rc");
    CHECK(t.rope == ROPE_GPTJ && t.act == ACT_SILU, "llama traits");
    CHECK(resolve_as("qwen3", &t) == 0, "qwen3 rc");
    CHECK(t.qk_norm_rms == 1, "qwen3 qknorm");
    CHECK(resolve_as("gemma", &t) == 0, "gemma rc");
    CHECK(t.act == ACT_GELU && t.tied_embeddings == 1, "gemma traits");

    /* metadata overrides still apply on new keys */
    {
        GGUFModel m;
        memset(&m, 0, sizeof(m));
        strncpy(m.architecture, "internlm2", sizeof(m.architecture) - 1);
        m.sliding_window = 4096;
        m.final_logit_softcapping = 30.0f;
        CHECK(tt_traits_resolve(&m, &t) == 0, "override rc");
        CHECK(t.swa_size == 4096, "override swa");
        CHECK(t.softcap_value == 30.0f, "override softcap");
        CHECK(t.rope == ROPE_GPTJ, "override keeps rope");
    }

    /* Tier-2 / MoE keys must stay UNKNOWN (no silent wrong-logits entry) */
    const char *skipped[] = {
        "nemotron", "qwen", "olmo2", "apertus", "exaone4",
        "arctic", "dots1", "bailingmoe", "bailingmoe2", "deci",
        "grovemoe", "ernie4_5-moe", "exaone-moe", "seed_oss",
        "nope-not-an-arch", NULL
    };
    for (int i = 0; skipped[i]; i++) {
        CHECK(resolve_as(skipped[i], &t) == -1, "skip %s still unknown", skipped[i]);
    }
    /* NULL / empty arch also unknown */
    CHECK(tt_traits_resolve(NULL, &t) == -1, "null model unknown");

    printf("%s: %d checks, %d failures\n",
           fails ? "RESULT FAIL" : "RESULT PASS", checks, fails);
    return fails ? 1 : 0;
}
