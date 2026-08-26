# Gemma-3n (and Gemma-4 26B-A4B) Support — Implementation Plan

> **Prereqs green:** M8 gemma-4 forward is parity-true (m84 6/7 PASS),
> `src/moe_router.{h,c}` golden-verified (top-k, gate ops, weight-norm,
> plan, combine). M8 trait machinery already covers rope/GeGLU/QK-norm
> /attn-scale-1/v-plain-norm/per-layer-embd. This plan reuses all of that
> and adds what gemma-3n uniquely needs: **altup + laurel + gaussian-topk
> activation sparsity**. MoE experts (for 26B-A4B) come "for free" from
> the MoE notes path.
>
> Sources cited by file:line from `oracle/llama.cpp` (read-only).

## 0. Verification of starting assumptions

`oracle/llama.cpp/src/llama-model.h:126` defines `LLM_TYPE_26B_A4B // Gemma4`.
`oracle/llama.cpp/src/llama-model.h:151-152` defines `LLM_TYPE_E2B`, `LLM_TYPE_E4B`
(both gemma-3n in gemma3n.cpp:17-18, and gemma-3 family in gemma4.cpp:24-25).
**Naming collision:** E2B/E4B are used by *both* gemma-3n and gemma-4 — the
type disambiguates by which build's `switch(hparams.n_layer())` matched.
Our engine keys off `general.architecture` string, not the type, so the
collision is harmless: `gemma3n` → gemma-3n, `gemma4` → gemma-4.

## 1. gemma-3n spec (from llama.cpp oracle)

Read: `oracle/llama.cpp/src/models/gemma3n.cpp` (468 lines, full read),
`oracle/llama.cpp/src/models/models.h:821-848`, `llama-hparams.h:234-237`.

### Variants
- `gemma-3n-E2B-it` — n_layer=30, type=`LLM_TYPE_E2B` (gemma3n.cpp:17)
- `gemma-3n-E4B-it` — n_layer=35, type=`LLM_TYPE_E4B` (gemma3n.cpp:18)

### Shared architecture
- **Attention**: per-head RMSNorm on Q and K (gemma3n.cpp:148-152), plain
  RMSNorm on V with `f_norm_rms_eps` and NO gamma (`v_plain_norm` trait,
  already in our gemma4 entry). `f_attention_scale = 1.0` (gemma3n.cpp:8,
  no 1/sqrt(hd)).
- **RoPE**: NEOX; `freq_base_train_swa` (separate SWA base) read from
  `LLM_KV_ROPE_FREQ_BASE_SWA`; per-layer base via `get_rope_freq_base` —
  same dual-base scheme as gemma-4.
- **Sliding window**: `set_swa_pattern(5)` (gemma3n.cpp:5) — pattern
  repeats every 5 layers. `n_layer_kv_from_start=20` (gemma3n.cpp:7) —
  last 10 (E2B) or 15 (E4B) layers reuse earlier KV (shared_kv_layers).
  Our KV-cache already supports per-layer window + shared-KV (M10 +
  gemma4 work). **Trait hook needed:** SWA pattern of 5 (not the
  single-window size of gemma2's 4096).
- **Activation**: GeGLU with GELU (gemma3n.cpp:217, `ggml_gelu(gate_proj)
  * up_proj`) — same as gemma-4. **gating has activation sparsity**:
  `gaussian_topk(gate_proj)` for `il < n_layer_sparsity (=10)`
  (gemma3n.cpp:212, models.h:833). The 10-layer cap is hardcoded in
  models.h:833 — same magic for all variants; no metadata key.
- **MatFormer per-layer embeds**: same PLE pipeline as gemma-4
  (gemma3n.cpp:319-372), but the per-layer dim is `n_embd_altup` (default
  256, hparams.h:237) NOT the `n_embd_per_layer` that gemma-4 uses.
  Our loader already reads `embedding_length_per_layer_input` (loader
  .c:253) but that maps to `n_embd_per_layer`. We need a *separate*
  field for `n_embd_altup`. **Two distinct "per-layer" dims now exist.**
- **Final logits softcap**: `tanh`-form softcap at the END
  (gemma3n.cpp:308-311), value from `LLM_KV_FINAL_LOGIT_SOFTCAPPING`
  (already parsed, loader_gguf.c:249).
- **Output tied to `token_embd`** when `output` is NULL
  (gemma3n.cpp:25-27, TENSOR_DUPLICATED) — `tied_embeddings=1`.

### AltUp (the gemma-3n-only block)
The single biggest new block. From gemma3n.cpp:88-94 and models.h:830-848:
- `n_altup` (default 4) parallel hidden-state streams; `i_altup_act=0`
  is the "active" stream that the FFN/attn paths actually transform.
- Before layer loop: project 1 active stream into (n_altup-1) "added"
  streams via `altup_proj` [n_embd, n_embd, n_altup-1] (line 31 tensor,
  gemma3n.cpp:111-119) using magnitude-preserving rescale
  (`calc_magnitude` x2 then `mul/div` to match target magnitude, line
  117). Concatenate → cur shape [n_embd, n_tokens, n_altup].
- **Per layer**:
  1. `altup_predict(cur)` (gemma3n.cpp:414-437): compute router
     modalities from active stream (`altup_router [n_embd,n_altup]` +
     `altup_router_norm [n_embd]`, tanh, scale 1/n_embd), then matmul
     `altup_predict_coef [n_altup, n_altup*n_altup]` → reshape → predict
     new altup states. Add residual.
  2. Take active slice, run standard attn + LAUREL low-rank branch
     (`laurel_l [n_embd,laurel_rank=64]`, `laurel_r [laurel_rank,n_embd]`,
     `laurel_post_norm`; gemma3n.cpp:375-385). Combine: attn_out
     post-norm + active residual, then `(cur + laurel) / sqrt(2)`
     (gemma3n.cpp:179).
  3. GeGLU FFN with gaussian-topk for first 10 layers.
  4. `altup_correct(predictions, attn+ffn_out)` (gemma3n.cpp:439-467):
     residual innovation = (activated - active_prediction), modulated by
     `altup_correct_coef [n_altup, n_altup]` (+1 bias) per altup, then
     add to predictions.
  5. `per_layer_inp_gate [n_embd, n_embd_altup]` → GELU → ×
     per-layer-tok-embd slice → `per_layer_proj [n_embd_altup,n_embd]`
     → `per_layer_post_norm` (norm = top-level `per_layer_proj_norm` —
     same DUPLICATED trick as gemma-4). Add this to altup slices 1..N-1
     only (gemma3n.cpp:242-260).
  6. `layer_output_scale` (gemma4-style, scalar [1]) — gemma-3n does NOT
     load this in the tensor list, but its graph reads the same scalar
     the same way (verify during bring-up: gemma3n.cpp loads no
     `LLM_TENSOR_LAYER_OUT_SCALE` so we may need to treat as 1.0 default
     or check a tensor that exists).

### Post-loop
- `altup_unembd_proj [n_embd, n_embd, n_altup-1]` (gemma3n.cpp:32 tensor,
  applied at lines 283-301) merges the n_altup streams back to 1 by the
  inverse magnitude-preserving projection + mean.
- `output_norm` + `output` (tied) + softcap.

### 26B-A4B variant
This is a **gemma-4** type (`LLM_TYPE_26B_A4B`, gemma4.cpp:23), NOT gemma-3n.
It is a different family from gemma-3n and the engine must NOT conflate.
Spec from gemma4.cpp:23 + MoE notes:
- n_layer=30, dim=2048, head_dim=256, hidden=6144 (dense shared expert),
  n_expert=128, n_expert_used=4 (verify at bring-up by reading the GGUF
  metadata), MoE merged `ffn_gate_up_exps` 3D `[n_embd, 2*n_ff_exp,
  n_expert]` (gemma4.cpp:99) plus `ffn_down_exps [n_ff_exp,n_embd,
  n_expert]`, plus per-expert `ffn_gate_inp_s` (router input scale,
  gemma4.cpp:101).
- Pre-scaled router logits (`ffn_gate_inp_s` per layer,
  LLM_KV_EXPERT_WEIGHTS_SCALE).
- Uses our `tt_moe_route` directly (softmax/sigmoid already supported;
  may need `sigmoid` + weight_norm + w_scale path). The gemma4
  `TT_MOE_GATE_SOFTMAX_WEIGHT` (raw logits, softmax over top-k only)
  variant from moe-router.h:39 maps to gemma4's pre-scaled logits
  pattern.

## 2. Already supported by engine (reusable as-is)

| gemma-3n feature                       | Engine support                       | Source              |
|----------------------------------------|--------------------------------------|---------------------|
| NEOX rope                              | yes (gemma4 entry)                   | arch_registry.c:96  |
| GeGLU w/ GELU                          | yes (gemma4 `ACT_GELU`)              | arch_registry.c:96  |
| final-logits softcap (tanh form)       | yes                                  | arch_registry.c:96  |
| per-head QK RMSNorm                    | yes (`qk_norm_rms`, `qk_norm_eps`)   | arch_registry.c:96  |
| attn scale = 1.0                       | yes (`attn_scale_one`)               | arch_registry.c:96  |
| V plain RMSNorm                        | yes (`v_plain_norm`)                 | arch_registry.c:96  |
| per-layer token embeds (MatFormer)     | yes (gemma4 PLE pipeline)            | m84 plan            |
| per-layer output scale                 | yes (gemma4)                         | m84 plan            |
| sandwich (post_attention/post_ffw)norm | yes (gemma4)                         | m84 plan            |
| tied embeddings                        | yes                                  | arch_registry.c:96  |
| SWA + shared-KV (per-layer window)     | yes (M10 + gemma4)                   | kvcache.c           |
| MoE router (softmax/sigmoid/topk)      | yes, golden-verified                 | moe_router.c        |
| per-expert GEMV                        | yes (reuses `tt_gemv_typed`)         | gemv_typed.cu:478   |
| MoE combine (weighted sum + shared)    | yes (`tt_moe_combine`)               | moe_router.c:181    |
| MoE plan (group by expert)             | yes (`tt_moe_build_plan`)            | moe_router.c:152    |
| BF16 dtype for PLE/embd                | yes (`GGUF_TYPE_BF16=30`)            | loader_gguf.h:14    |

## 3. MISSING — delta the engine must add

### 3.1 Loader (`src/loader_gguf.c`, +~50 lines, `include/loader_gguf.h` +6 fields)
- New `GGUFModel` fields: `n_altup` (u32), `n_embd_altup` (u32),
  `i_altup_act` (u32), `laurel_rank` (u32), `n_layer_sparsity` (u32,
  default 10), `f_sparsity_std_mul` (f32, default 1.6448533535),
  `n_expert` (u32), `n_expert_used` (u32), `n_ff_exp` (u32, distinct
  from `hidden_dim` which is the shared expert dim).
- New KV parsing in the `for(kv)` loop: `embedding_length_altup` (or
  piggyback on existing `embedding_length_per_layer_input` and
  reinterpret for gemma3n — see §6 risk #1), `altup_num_inputs`,
  `altup_active_idx`, `laurel_rank`, `expert_count`, `expert_used_count`,
  `expert_feed_forward_length`, `attention.sliding_window_pattern` (u32
  pattern=5, not just the size).
- Tensor-name catalog needs to accept altup/laurel/per-layer names (the
  loader is currently a flat-name lookup; we use `gguf_get_tensor` so
  the *names* are already known — no code change there, but the forward
  path must look them up).

### 3.2 Architecture registry (`src/arch_registry.c`, +~12 lines, `include/arch_registry.h` +3 fields)
- New entry `{ "gemma3n", { ROPE_NEOX, ACT_GELU, 30.0f, 0, 1, 1, 1e-6f,
  0.0f, 1, 1, 1, 1, /*new fields*/ } }`.
- `TTraits` extends with: `int swa_pattern` (0=off, N=layer%N==0 is
  full otherwise SWA), `int has_altup`, `int has_per_layer_embd` (gemma-4
  already gates on `per_layer_embd_dim>0`; gemma-3n needs the same hook
  but with `n_embd_altup`).
- Note: gemma-4 26B-A4B already passes through the existing `gemma4`
  entry; the MoE activation is selected by the presence of
  `blk.0.ffn_gate_inp.weight` at forward-construction time (see §3.4).
- `tt_traits_supported()` string update.

### 3.3 Forward path (`kernels/qwen2_cuda.cu` + `src/ops_llm.c`, +~600 lines)
This is the bulk of the work. gemma-3n is NOT a small delta on gemma-4
because altup is a fundamentally different inner loop.
- New struct `tt_engine_gemma3n_t` (or extend `tt_engine_t` with a
  conditional branch — prefer struct variant to keep hot path clean).
  Holds per-layer device pointers for the altup/laurel/per-layer
  tensors (≈14 new `TTensor` slots/layer, not all the same shape).
- New device buffer pool: `d_altup_state[n_embd*n_altup*MAX_TOKENS]`
  scratch for the altup stream stacking/unstacking. Sizes: altup_in/out
  are [n_embd, n_altup] for one token, so scratch is tiny (<4KB/token)
  but multiplied by n_layers active at once.
- New kernels in `kernels/qwen2_cuda.cu` (or new `kernels/gemma3n.cu`):
  - `k_altup_magnitude(...)` — sum-of-squares along embed dim.
  - `k_altup_project` — matmul `altup_proj` against stacked
    `inp_repeated` (already covered by `tt_gemv_typed` per stream, but
    we can batch the n_altup-1 streams into a single call).
  - `k_altup_predict`, `k_altup_correct` — both reduce to a few
    `tt_gemv_typed` + `k_add` per token. Reuse the existing
    `k_rmsnorm` and `k_add`.
  - `k_laurel` — `tt_gemv_typed(W_L) → tt_gemv_typed(W_R) →
    k_rmsnorm → k_add`. Reuses existing kernels, just a function.
  - `k_gaussian_topk` — needs mean/var/relu-sub. Mean and var exist in
    the codebase; if not, add a `k_mean_var` reduction. ReLU+sub is
    trivial.
  - `k_softcap_logits` — already exists for gemma-2/4.
- The new code does NOT need new GEMV kernels — `tt_gemv_typed` is the
  per-expert slab GEMV path (MoE notes + gemv_typed.cu:478). The only
  new kernels are reductions and elementwise.

### 3.4 MoE activation hook (no new code path; selection)
- In the engine constructor, after the gemma-4 path is taken, check
  `gguf_get_tensor(model, "blk.0.ffn_gate_inp.weight")` (NOT_REQUIRED,
  exactly like gemma4.cpp:99). If non-NULL, set a flag
  `engine->cfg.moe_active = 1` and populate per-layer expert slab
  device pointers (`pl_moe[i]` for router + 3 expert slabs, mirroring
  `qwen2_cuda.cu` MoE plan from MoE notes).
- Per-layer expert FFN forward: existing dense-FFN forward path
  produces the shared-expert output; the MoE branch is added via
  `tt_moe_route → tt_moe_build_plan → per-expert tt_gemv_typed × 3 →
  tt_moe_combine` (all the pieces exist, ~40 lines of glue in the
  per-layer forward).

### 3.5 No new top-level metadata key required for activation
The altup/gaussian-topk magic numbers live in `models.h:833-834` as
hardcoded defaults (n_layer_sparsity=10, f_sparsity_std_mul=1.644...).
The engine should encode these as compile-time constants (no
runtime-config decision needed). Documented in the trait comment so
they get updated when gemma-3n evolves.

## 4. Test plan

### 4.1 Model files
- E2B: `data/models/gemma-3n-E4B-it-UD-IQ3_XXS.gguf.part` already
  partially present. Complete the download (or pull from
  unsloth/gemma-3n-E4B-it-GGUF / bartowski google/gemma-3n-E4B-it-GGUF
  for f16/Q4_0). IQ3_XXS is fine for parity but messy for golden —
  use Q4_0 or Q5_K_M when available.
- 26B-A4B: must come from a gemma-4 build (NOT gemma-3n). Try
  unsloth/gemma-4-26B-A4B-it-GGUF. ~16 GB Q4_0 — exceeds 4 GB VRAM
  budget; needs hybrid offload (M11 already designed).
- Optional: re-quantize f16 → Q4_0 via the existing `llama-quantize`
  workflow in `scripts/quantize.py` (M7 task 1) for golden-by-f16.

### 4.2 Parity bar
- **Numeric parity vs llama.cpp oracle**: same 8-prompt parity set as
  `tests/gate_m84_gemma4.py`. Pass criterion: max abs logit diff
  ≤1e-3 at Q4_0 / ≤5e-4 at F16 across all 8 prompts. The dense E2B
  must pass 6/7 like gemma-4 E2B (m84) before MoE bits turn on.
- **Determinism**: same prompt → same logits across runs (already
  enforced by the deterministic topk in moe_router.c:90-110).
- **Tests to add** (`tests/`):
  - `test_gemma3n_ple.py` — golden-trace PLE pipeline against
    `ref_gemma3n_numpy.py` reference (mirror of `ref_gemma4_numpy.py`).
  - `test_gemma3n_altup.py` — altup predict/correct against numpy
    reference for n_altup=4, n_altup_act=0, with a few random seeds.
  - `test_gemma3n_laur.py` — laurel low-rank branch.
  - `test_gemma3n_sparsity.py` — gaussian-topk activation sparsity
    does not change argmax of the FFN gate on small inputs.
  - `test_gemma3n_moe_e2b.py` — 26B-A4B variant: verify router
    weights + expert slab pointers + that shared expert (dense
    path) produces the gemma-4 dense output.
  - `gate_m85_gemma3n.py` — end-to-end parity gate (mirror of
    `gate_m84_gemma4.py`).

### 4.3 Regression
- `tests/gate_m84_gemma4.py` must remain PASS unchanged. The gemma-4
  forward path is shared (trait-driven); the gemma-3n path goes
  through the new `tt_engine_gemma3n_*` variant.
- `tests/test_moe_router.c` — untouched, must remain green.
- `tests/test_kvcache.c` — untouched, must remain green.

## 5. Implementation phases

### Phase 1 — Trait + loader + non-MoE forward (1-2 days)
- Loader parses new gemma-3n metadata keys; new `TTraits` fields.
- Forward path: `tt_engine_gemma3n_t` with altup + laurel + gaussian
  topk + per-layer-embd + softcap. **No MoE experts yet** (the dense
  gemma-3n path — but gemma-3n has *no* pure dense variant; this phase
  sets up the trait and verifies the path with a mock layer count of
  experts=0 to keep the engine green before MoE lands).
- Tests: `test_gemma3n_ple.py`, `test_gemma3n_altup.py`,
  `test_gemma3n_laur.py`, `test_gemma3n_sparsity.py` all green against
  numpy refs. End-to-end on E2B IQ3_XXS is allowed to be loose (no
  parity yet) since the engine hasn't been wired to the new
  per-layer routing.
- **Acceptance**: gemma-4 m84 still PASS; new gemma-3n trait resolves
  correctly; engine produces *some* output for E2B (correctness
  verified by deterministic shape/dtype, not parity).

### Phase 2 — MoE experts via tt_gemv_typed (2-3 days)
- Wire `engine->cfg.moe_active` detection in the gemma-4 forward
  constructor.
- Per-expert GEMV loop using `tt_moe_route → tt_moe_build_plan →
  tt_gemv_typed × 3 per (token, expert) → tt_moe_combine`. The dense
  GeGLU path produces the shared-expert output that `tt_moe_combine`
  folds in.
- Tests: `test_gemma3n_moe_e2b.py`, `gate_m85_gemma3n.py` parity
  against llama.cpp for E2B 26B-A4B at Q4_0.
- **Acceptance**: 6/7 parity on the parity set; expert-routing
  determinism; per-expert GEMV on stream in a single kernel
  launch per layer.

### Phase 3 — 26B-A4B and other variants (1-2 days)
- 26B-A4B is the bigger variant of gemma-4 (E2B gemma-4 is the small
  one). Same MoE path as Phase 2; only the per-tensor shapes change.
- Add `gemma-3n-E4B-it` (35 layers) parity. Per-layer dim and head
  count differ; gated by hparams.
- Wire hybrid offload (M11) for any variant that doesn't fit in 4 GB.
- **Acceptance**: 7/7 parity for E2B; 6/7 for E4B; 26B-A4B runs
  end-to-end via offload path with hybrid-correctness gate (per
  `2026-08-27-hybrid-offload-findings.md`).

## 6. Risks (ranked)

1. **Altup is the gemma-3n-only block — no parity-true template.**
   Unlike gemma-4 which inherits 60% from gemma-2, gemma-3n's
   altup_predict / altup_correct / laurel / gaussian-topk are net-new
   on our engine. A subtle bug in `altup_correct` (the
   `+1.0` bias on all_coefs at gemma3n.cpp:454) shifts every layer's
   residual by 1.0 and silently breaks parity. Mitigate with the
   per-block numpy golden (`ref_gemma3n_numpy.py`) BEFORE the
   end-to-end gate. M84 took ~2 days for a smaller delta; budget
   3-4 days for altup alone.

2. **MoE memory bandwidth on the 4 GB card.** Even gemma-3n-E2B
   unquantized is ~7-8 GB. With Q4_0 expert slabs the per-token
   expert-GEMV touches `(k/n_expert)·n_ff·(embd+embd)·sizeof(Q4_0)`
   ≈ small, but the *weights themselves* must reside somewhere. Full
   experts on GPU is impossible for 26B-A4B at any quant; hybrid
   offload is mandatory. Even E2B-with-MoE (a 26B-A4B-style config)
   needs careful pinning. The Phase 3 acceptance *requires* the M11
   hybrid offload to be solid.

3. **Per-expert cache granularity.** Today the KV cache wraps
   per-token, not per-expert-token. The MoE branch is *independent*
   of KV (router + expert slabs), so this is NOT a correctness
   risk — but a perf risk. Each per-expert GEMV is one kernel
   launch per (token, expert) in the plan; at n_expert_used=4 we
   add 4×3=12 extra small kernel launches per layer per token
   decode. Mitigate by batching the same expert across multiple
   decode tokens into a single launch (group-by-expert grouping is
   already in `tt_moe_build_plan`).

### Lower-severity risks
- **gemma-3 / gemma-3n naming overlap**: E2B/E4B types are shared
  between gemma-3n and gemma-4 in the oracle. Engine keys off
  `general.architecture` (`gemma3n` vs `gemma4`), so disambiguation
  is clean — but anyone reading the code needs the comment. Add to
  `tt_traits_supported()` docstring.
- **Output layer tie vs literal**: gemma-3n uses TENSOR_DUPLICATED
  for output when `output==NULL`. Our existing gemma-4 path already
  handles this; gemma-3n should inherit. Verify at bring-up.
- **gaussian-topk vs ReLU difference**: gemma-3n applies
  `relu(x - (mean + std_mul*std))` to `gate_proj` (gemma3n.cpp:387-394)
  but the standard GeGLU is `gelu(gate) * up` — the sparsity mask
  is a *post-GELU* zeroing, NOT a pre-GELU activation change. The
  math is correct in the oracle; needs careful test in
  `test_gemma3n_sparsity.py` to ensure we apply it AFTER `gelu`,
  not BEFORE.
- **SWA pattern of 5 (gemma-3n) vs single-window 4096 (gemma-2)**
  vs per-layer array (gemma-4). TTraits needs a new
  `swa_pattern` field (or the gemma-3n code reaches into the
  KV-cache SWA config directly). Pick the latter — keeps
  arch_registry.c clean.

## 7. Specific code change list

| File                                                | Change                                  | Lines (delta) |
|-----------------------------------------------------|-----------------------------------------|---------------|
| `include/arch_registry.h`                           | new TTraits fields: `swa_pattern`, `has_altup` | +2 / -0      |
| `src/arch_registry.c`                               | new `gemma3n` entry + supported-list update | +12 / -0     |
| `include/loader_gguf.h`                             | new GGUFModel fields: n_altup, n_embd_altup, i_altup_act, laurel_rank, n_layer_sparsity, f_sparsity_std_mul, n_expert, n_expert_used, n_ff_exp | +9 / -0 |
| `src/loader_gguf.c`                                 | new KV parsing for the above (6 new sfx cases) | +50 / -0     |
| `src/moe_router.h` + `.c`                           | no change (already complete)            | 0             |
| `kernels/qwen2_cuda.cu` (or new `kernels/gemma3n.cu`) | altup/laurel/gaussian-topk kernels + per-layer forward variant | +600 / -0   |
| `src/ops_llm.c`                                     | `tt_gaussian_topk` CPU reference for parity | +20 / -0     |
| `src/loader_gguf.c`                                 | tensor-name alias table for gemma-3n (altup/laurel) — only if direct name-lookup needs it | +0 / -0 (already name-flat) |
| `tests/test_gemma3n_ple.py`                         | new — PLE pipeline golden              | +60           |
| `tests/test_gemma3n_altup.py`                       | new — altup predict/correct golden     | +120          |
| `tests/test_gemma3n_laur.py`                        | new — laurel golden                    | +40           |
| `tests/test_gemma3n_sparsity.py`                    | new — gaussian-topk golden             | +50           |
| `tests/test_gemma3n_moe_e2b.py`                     | new — MoE wiring smoke                 | +80           |
| `tests/gate_m85_gemma3n.py`                         | new — E2E parity gate                  | +100          |
| `tests/ref_gemma3n_numpy.py`                        | new — numpy reference for goldens      | +200          |
| `scripts/quantize.py`                               | no change (already supports f16→Q4_0) | 0             |
| `data/models/.gitignore` (or similar)               | add `gemma-3n-*.gguf` if not present   | +1            |
| `docs/plans/2026-08-27-gemma3n-support.md`          | this document                          | new           |

**Total engine LoC delta:** ≈ 870 lines (≈ 600 kernel, ≈ 70 trait/loader, ≈ 200 tests).
**Per-phase LoC:** P1 ≈ 280 (trait+loader+altup+ple+laurel+sparsity kernels
+ 4 unit tests + 1 numpy ref), P2 ≈ 420 (MoE wiring + per-expert GEMV
loop + moe + gate tests), P3 ≈ 170 (E4B + 26B-A4B shape variant + gate
extension).

## 8. Out of scope (explicit)

- **Multimodal encoder path** (gemma-3n supports image+audio input).
  The oracle's `build_inp_per_layer` has a `if (ubatch.token)` branch
  (gemma3n.cpp:319) that handles the non-token (encoded embedding)
  case. Text-only is sufficient for our parity work; image encoder
  integration is a separate project.
- **gemma-3 (non-n)**: handled by `LLM_ARCH_GEMMA3` oracle build which
  we have not pulled. Out of scope; add later if needed.
- **Custom kernel for altup-coef matmul**: the 3D permute +
  `mul_mat` is handled by `tt_gemv_typed` against
  `altup_predict_coef` (shape `[n_altup, n_altup*n_altup]` per layer)
  via a 2D view. No new kernel needed.

---

## Summary for caller

- **Doc path:** `docs/plans/2026-08-27-gemma3n-support.md`
- **Phase times:** P1 1-2d, P2 2-3d, P3 1-2d (total 4-7d)
- **Top 3 risks:**
  1. Altup is net-new on our engine (no gemma-4 template); subtle
     `+1.0` bias in `altup_correct` could silently break parity.
  2. MoE memory bandwidth on 4 GB card — even small gemma-3n-with-MoE
     needs hybrid offload (mandatory for 26B-A4B).
  3. Per-expert GEMV is per-(token, expert) launch granularity;
     cache layer wraps per-token not per-expert-token (perf risk,
     not correctness).
