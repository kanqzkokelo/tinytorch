/* M7 task 3: architecture trait registry.
 *
 * Every non-obvious default below was verified against the vendored oracle
 * tree (oracle/llama.cpp). Citations use file:line of that checkout:
 *
 *   llama-model.cpp:3835-3896  llama_model_rope_type():
 *     - LLM_ARCH_LLAMA (covers mistral/tinyllama/smollm GGUFs, which all
 *       carry general.architecture == "llama") -> LLAMA_ROPE_TYPE_NORM,
 *       described there as "a normal RoPE, operating on pairs of consecutive
 *       head values" = interleaved pairs = our ROPE_GPTJ.
 *     - LLM_ARCH_QWEN2 and LLM_ARCH_GEMMA/GEMMA2 -> LLAMA_ROPE_TYPE_NEOX
 *       ("pairs of head values are offset by n_rot/2") = our ROPE_NEOX.
 *   llama.cpp:3411-3415  build_qwen2(): ffn with LLM_FFN_SILU, LLM_FFN_PAR.
 *   llama.cpp:1560-1564  build_llama(): ffn with LLM_FFN_SILU, LLM_FFN_PAR.
 *   llama.cpp:4976-4978  build_gemma2(): ffn with LLM_FFN_GELU, LLM_FFN_PAR
 *                        (GeGLU shape: gate projection with GELU activation).
 *   llama.cpp:5000-5003  build_gemma2(): final logit softcap implemented as
 *                        scale(1/c) -> tanh -> scale(c), i.e. tanh(x/c)*c.
 *   llama-model.cpp:857  gemma2 reads f_final_logit_softcapping from the
 *                        gguf key (gemma2.final_logit_softcapping); HF config
 *                        ships 30.0 for gemma-2-2b/9b/27b -> our fallback.
 *   llama-model.cpp:853  gemma2 hparams.n_swa = 4096 default ("default value
 *                        of gemma 2"), overridden by the gguf sliding_window
 *                        key when present.
 *   llama-model.cpp:2420-2424  gemma/gemma2 output weight is TENSOR_DUPLICATED
 *                        from token_embd when absent -> tied_embeddings = 1.
 *   convert_hf_to_gguf.py:4730  Gemma conversion adds +1 to every norm weight
 *                        ("implement layernorm1p w/o changing anything on the
 *                        GGML engine side") -> engine-side norm_offset stays 0;
 *                        plain rmsnorm-multiply reproduces gemma exactly.
 *
 * Qwen3 note: this oracle checkout predates qwen3 (no build_qwen3 /
 * LLM_ARCH_QWEN3 in llama-model.cpp). Qwen3 keeps qwen2's half-split NEOX
 * rope per HF modeling_qwen3.py (apply_rotary_pos_emb on half-split views)
 * and later llama.cpp; its distinguishing trait here is RMSNorm over each
 * q/k head pre-rope (q_norm/k_norm weights in the GGUF), eps from
 * attention.key_epsilon (HF default 1e-6).
 */
#include <stdio.h>
#include <string.h>

#include "arch_registry.h"
#include "loader_gguf.h"
typedef struct {
    const char *arch;
    TTraits tr;
} ArchEntry;

static const ArchEntry kArchTable[] = {
    /* qwen2: NEOX rope + SiLU + optional qkv biases (per-tensor detection). */
    { "qwen2", { ROPE_NEOX, ACT_SILU, 0.0f, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0 } },
    /* "llama" covers mistral / tinyllama / smollm conversions. Interleaved
     * rope (LLAMA_ROPE_TYPE_NORM), no biases, untied head. */
    { "llama",    { ROPE_GPTJ, ACT_SILU, 0.0f, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0 } },
    { "llama3",   { ROPE_GPTJ, ACT_SILU, 0.0f, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0 } },
    { "llama2",   { ROPE_GPTJ, ACT_SILU, 0.0f, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0 } },
    { "mistral",  { ROPE_GPTJ, ACT_SILU, 0.0f, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0 } },
    { "smollm",   { ROPE_GPTJ, ACT_SILU, 0.0f, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0 } },
    { "smollm2",  { ROPE_GPTJ, ACT_SILU, 0.0f, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0 } },
    { "qwen3", { ROPE_NEOX, ACT_SILU, 0.0f, 0, 0, 1, 1e-6f, 0.0f, 0, 0, 0 } },
    /* gemma: NEOX rope, GeGLU(gelu), tied embeddings, no softcap/swa. */
    { "gemma", { ROPE_NEOX, ACT_GELU, 0.0f, 0, 1, 0, 0.0f, 0.0f, 1, 0, 0 } },
    /* gemma2: + SWA 4096 default + final-logits softcap 30.0 fallback. */
    { "gemma2", { ROPE_NEOX, ACT_GELU, 30.0f, 4096, 1, 0, 0.0f, 0.0f, 1, 0, 0 } },
    /* gemma4 (E2B/E4B): QK-norms on every layer, attention scale 1.0 (no
     * 1/sqrt hd), plain RMSNorm on V, per-layer token embeddings (MatFormer),
     * per-layer output scales, learned rope_freqs on global layers. */
    { "gemma4", { ROPE_NEOX, ACT_GELU, 30.0f, 0, 1, 1, 1e-6f, 0.0f, 1, 1, 1, 1 } },
};

const TTraits *tt_traits_lookup(const char *arch) {
    if (!arch || !arch[0]) return NULL;
    for (size_t i = 0; i < sizeof(kArchTable) / sizeof(kArchTable[0]); i++)
        if (strcmp(kArchTable[i].arch, arch) == 0) return &kArchTable[i].tr;
    return NULL;
}

int tt_traits_resolve(const GGUFModel *m, TTraits *out) {
    const TTraits *base = tt_traits_lookup(m ? m->architecture : NULL);
    if (!base) return -1;
    *out = *base;
    /* GGUF metadata overrides where the keys exist (loader stores them). */
    if (m->sliding_window > 0) out->swa_size = m->sliding_window;
    if (m->final_logit_softcapping > 0.0f) out->softcap_value = m->final_logit_softcapping;
    return 0;
}

const char *tt_traits_supported(void) {
    return "qwen2, llama (incl. llama3.x), qwen3, gemma, gemma2, gemma4";
}

/* ---------------------------------------------------------------------
 * M6 family-broadening findings (llama-3.2 / phi-2 probes, gate_m7_arch):
 *
 * llama-3.2-1B (arch "llama"): PASSES 7/7 unmodified.
 *  - GGUF conversion permutes q/k so HF half-split NEOX rope becomes
 *    interleaved GPTJ -> existing { ROPE_GPTJ, ACT_SILU } entry correct.
 *  - rope.freq_base=500000 read via loader suffix match; bartowski Q4_0
 *    carries NO llama3 rope-scaling keys (Llama 3.2 has none in its HF
 *    config; only 3.1+ 8B does). If a future GGUF ships
 *    <arch>.rope.scaling.llama3.* (type + freq_factors vector), the loader
 *    must parse them and the rope path must apply per-band freq factors --
 *    currently ignored: exact at short ctx, drifts long-context.
 *
 * phi-2 (arch "phi2"): OUT OF SCOPE for the trait registry alone. All four
 * gaps require kernels/qwen2_cuda.cu work (owned elsewhere). Verified
 * against TheBloke/phi-2-GGUF Q4_0 metadata + tensor inventory:
 *   1. LayerNorm, not RMSNorm: every norm carries .bias (beta) and
 *      subtracts the mean (keys *.attn_norm.{weight,bias},
 *      output_norm.{weight,bias}; eps under attention.layer_norm_EPSILON
 *      -- loader only parses layer_norm_RMS_epsilon today).
 *   2. Fused blk.N.attn_qkv.weight (+ .bias): kernel loads separate
 *      attn_q/attn_k/attn_v; needs a fused-QKV split path.
 *   3. Partial rotary: rope.dimension_count=32 vs head_dim=80. Both rope
 *      kernels rotate the full head dim; need a rot-dim parameter
 *      (GPTJ-style interleaved over first 32 channels only).
 *   4. Biases on attn_output, ffn_up, ffn_down and output (lm_head);
 *      FFN is plain GELU up/down with NO gate tensor, while the kernel
 *      expects a gate+up pair.
 * Loader change needed for phi2 is only an epsilon-key alias once kernels
 * gain LayerNorm support. Do NOT add a "phi2" registry entry until then:
 * TTraits cannot express any of the four gaps, so an entry would silently
 * produce wrong logits instead of the current clear unsupported-arch error.
 * --------------------------------------------------------------------- */
