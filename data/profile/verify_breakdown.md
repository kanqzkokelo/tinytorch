# Verification Pass Breakdown (Pre-Batched-Kernel)

**Model:** qwen2.5-0.5b-instruct-q4_0
**GPU:** RTX 3050 sm_86
**Date:** 2026-08-27
**Branch:** m6-correctness
**Source:** `tools/profile_step` with `TT_PROFILE=1`, 5-run median of per-stage accumulators
**Commit baseline:** 0c497f4 (Universal Speculative orchestrator / verify / N-gram)

## How `qwen2_engine_verify_speculative` is currently implemented

Read at `kernels/qwen2_cuda.cu:1994`. Per-candidate loop is `for (i = 0; i < n_candidate; i++)`:

1. `advance(e, h_candidate_tokens[i])` — runs `embed_token()` + `forward_layers()` for 24 layers (Qwen2.5-0.5B has 24 layers) + `e->pos++` + sync H2D copy of `d_pos`.
2. `compute_logits_into_d_logits(e)` — final RMSNorm + LM-head GEMV (V4 dispatch for q4_0) + optional tanh softcap. No sampling / argmax / D2H.
3. `cudaMemcpyAsync(d_logits → out_logits + i*vocab)` — D2D, async.
4. `cudaStreamSynchronize` at the end of the loop.

Per-layer body (`forward_layers`, ~line 1350-1557) is the standard eager pipeline:
`k_rmsnorm` (pre-attn) → `gemv_q4` (QKV) → `k_kv_scatter` → `flash_decode` → `gemv_q4` (O proj) →
`k_rmsnorm` (pre-MLP) → `gemv_q4` (gate+up+down MLP). All 24 layers run **sequentially**, batch_size=1.

There is **no batching of any kind** in the current verify path. The 4× cost is exact.

## Per-stage cost — single token (median over 5 runs)

| Stage            | Single ms | % of decode | What it does                            |
|------------------|-----------|-------------|-----------------------------------------|
| embed            | 0.006     | 0.13%       | gather + RMSNorm on token embedding     |
| qkv-gemv         | 0.620     | 13.32%      | Q/K/V projection per layer × 24 layers  |
| o+mlp-gemv       | 1.830     | 39.30%      | O projection + gate/up/down MLP × 24    |
| flash (attn)     | 0.420     |  9.02%      | flash_decode per layer × 24             |
| rmsnorm          | 0.260     |  5.58%      | pre-attn + pre-MLP RMSNorm × 24         |
| kv-scatter       | 0.090     |  1.93%      | write K/V to cache slot                 |
| logits-gemv      | 1.360     | 29.21%      | LM head (V4 q4_0) — once, not per layer |
| argmax           | 0.070     |  1.50%      | not actually called by verify           |
| **TOTAL**        | **4.656** | **100%**    |                                         |

GEMV total (qkv + o+mlp + logits) = **3.81 ms (81.8 %)**.
This is the part that a true batched-4 kernel replaces with ~1.05× the single-token cost.

## Per-stage cost — 4-token verify (extrapolation)

`verify_speculative` calls `advance()` × 4 + `compute_logits_into_d_logits` × 4.
Stage profile above runs per-layer / per-decode. Verify stages map 1-to-1 to 4 single-token
decodes, except `argmax` (verify never argmaxes; orchestrator does it on the accepted token
on the host). So 4× verify is the sum of 4× (embed + qkv-gemv + o+mlp-gemv + flash + rmsnorm + kv-scatter + logits-gemv) = 4 × 4.516 ms ≈ **18.06 ms** (using the 4-token-subtotal of 4.516 ms; the single-token 4.656 includes argmax which verify skips — recompute below).

Single-token verify (no argmax) = 0.006 + 0.620 + 1.830 + 0.420 + 0.260 + 0.090 + 1.360 = **4.586 ms**.
Sequential 4-token verify = **18.34 ms**.

| Stage        | Single ms | 4× sequential ms | % of 4× verify | Can batch? | Batched cost (ms) |
|--------------|-----------|------------------|----------------|------------|-------------------|
| embed        | 0.006     | 0.024            | 0.13%          | NO         | 0.024             |
| qkv-gemv     | 0.620     | 2.480            | 13.5%          | YES        | 0.651  (=1.05×)   |
| o+mlp-gemv   | 1.830     | 7.320            | 39.9%          | YES        | 1.922  (=1.05×)   |
| flash        | 0.420     | 1.680            |  9.2%          | PARTIAL    | 0.84 – 1.26       |
| rmsnorm      | 0.260     | 1.040            |  5.7%          | YES        | 0.273  (=1.05×)   |
| kv-scatter   | 0.090     | 0.360            |  2.0%          | NO         | 0.360             |
| logits-gemv  | 1.360     | 5.440            | 29.7%          | YES (V4)   | 1.428  (=1.05×)   |
| **TOTAL**    | **4.586** | **18.344**       | **100%**       |            |                   |

`flash` partial: a "grouped Q@Kᵀ attention" kernel can process 4 candidates in 2-3 single-pass
times (read KV once, run 4 dot-products). Conservatively 2.5× → **1.05 ms**.

## Upper-bound speedup estimate

### Scenario A — batch GEMVs only (RMSNorm + flash still sequential)

New verify = 4·(embed + kv-scatter) + 1.05·(qkv + omlp + logits + rmsnorm) + 4·flash
            = 0.024 + 0.360 + 1.05·(0.620 + 1.830 + 1.360 + 0.260) + 1.680
            = 0.384 + 4.271 + 1.680
            = **6.34 ms**
Speedup vs sequential = 18.34 / 6.34 = **2.89×** over the current verify path.

### Scenario B — batch GEMVs + RMSNorm + partial flash batching

New verify = 4·(embed + kv-scatter) + 1.05·(qkv + omlp + logits + rmsnorm) + 2.5·flash
            = 0.384 + 4.271 + 1.050
            = **5.71 ms**
Speedup vs sequential = 18.34 / 5.71 = **3.21×** over the current verify path.

### Scenario C — batch everything except kv-scatter (theoretical ceiling)

New verify = 4·(embed + kv-scatter) + 1.05·(qkv + omlp + logits + rmsnorm) + 1.05·flash
            = 0.384 + 4.271 + 0.441
            = **5.10 ms**
Speedup vs sequential = 18.34 / 5.10 = **3.60×** over the current verify path.

## Realistic end-to-end gain (factoring in acceptance rate)

A single decode step currently costs 4.656 ms (single-token baseline).
With speculative K=4 + verify:

- Drafter (N-gram lookup, GPU): ~negligible vs verify (N-gram is a few µs on the device).
- Verify (sequential, current): 18.34 ms.
- **Total per speculative step: ~18.34 ms** for up to 5 emitted tokens (4 candidates + 1 bonus).

If we hit Scenario A (2.89× verify speedup):
- Verify = 6.34 ms. Total = ~6.34 ms / ≤5 tokens → ~1.27 ms/token vs 4.66 ms/token baseline.
- **End-to-end speedup ≈ 3.67× (if every candidate accepted).**
- With realistic 50% acceptance (2 bonus tokens per round, ~3 useful tokens):
  effective time per useful token ≈ 6.34 / 3 = 2.11 ms → **~2.2× end-to-end.**

If we hit Scenario B (3.21× verify):
- Verify = 5.71 ms. 100% accept → 5.71 / 5 = 1.14 ms/token → **~4.1× end-to-end.**
- 50% accept → 5.71 / 3 = 1.90 ms/token → **~2.45× end-to-end.**

## What this means for the implementation plan

1. **GEMVs are 81.8% of verify time** — far above the >50% target. A batched-4 kernel
   for QKV + O+MLP + LM-head alone (Scenario A) is worth doing.
2. **flash attention is only 9% of decode** — single-token context is small, so the
   attention cost does not blow up with K=4. Grouped/batched flash would add ≤0.6 ms
   to the speedup envelope (Scenario A vs B). Worth bundling if cheap, but not the
   blocker. **If attention later becomes 20-30% (longer context), grouped attention
   becomes the next bottleneck.**
3. **RMSNorm (5.6%)** is small but not negligible; a 1×N RMSNorm kernel is essentially
   free to add alongside the GEMV batcher (shared launch, same grid pattern). Include it.
4. **kv-scatter (1.9%)** is per-position writes — cannot be amortized. Floor.
5. **argmax (1.5%)** is already not part of verify; the orchestrator handles the accepted
   token via the regular sample_eager path. No change needed.

## Surprises / unexpected findings

- **GEMV share is 81.8%, not the assumed ~50%** — the LM-head alone (V4 q4_0) is 29% of
  every decode step. This is the biggest single cost in the model. Batching it gives the
  largest single chunk of the speedup. (Note: V4 dispatch already includes
  the "conditional M≥128 → V4" check from commit 24c705b; for M=4 it falls back to V2,
  so the 1.36 ms logits-gemv here is actually a 4-wide V2-style GEMV, not the V4 fast
  path. The V4 batched kernel for M=4 would still benefit from shared-weight loading.)
- **flash attention is 9%, not 30-50%** — at a fresh-context, short-prompt decode
  (n_kv ≈ a few hundred tokens), the attention compute is small. If a benchmark uses
  long contexts (>4k tokens) this will shift. Recommend re-profiling on long-context
  test before claiming the 2.9× end-to-end — it could go higher or lower depending on
  prompt type.
- **kv-scatter is a tiny 0.09 ms** — much cheaper than expected. The "scattered write"
  cost is hidden by HBM bandwidth, not by launch overhead.

## Bottom line

Implementing a batched-4 verify kernel that fuses **QKV, O+MLP, RMSNorm, LM-head** is
expected to deliver **~2.5-3× verify speedup** (Scenario A) and **~2.2-2.5× end-to-end
token throughput** at 50% acceptance. That meets the task's goal.

Grouped/batched flash attention is a follow-up (Scenario B → C) and is not required
to hit the main milestone.

## Commit

`profile: verify-pass stage breakdown + upper-bound speedup estimate` (Task 0).
