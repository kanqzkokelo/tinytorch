# KV-cache quantization design

**Date**: 2026-08-27
**Repo**: `~/Storage/repos/nnfromscratch`
**Status**: design / pre-implementation research
**Audience**: M11/M12 engine work; required to land E4B-on-4GB future path

## 0. Motivation

E2B cache footprints today (fp32 storage, ctx 1k / 8k / 32k):

| shape | 1k | 8k | 32k |
|---|---:|---:|---:|
| no-share | 0.137 GiB | 1.094 GiB | 4.375 GiB |
| shared-slab (current) | 0.059 | 0.469 | 1.875 |
| ideal-mixed (ceiling) | 0.037 | 0.297 | 1.188 |

The shared-slab 1.875 GiB at 32k already costs meaningful VRAM on a 4 GB
card; the 1.188 GiB ideal-mixed is the **theoretical lower bound** when
slabs are sized to the actual per-layer kv_width, and that bound is
**unchanged by quantization** (it is a layout question, not a bit-width
question). The quant design space therefore addresses the **factor**
between ideal-mixed and the engine today:

- ideal-mixed is the **fp32 ideal**; the *quantized* ideal is
  `ideal-mixed × (new_dtype_bytes / 4)`. Going fp32 → fp16 cuts it in
  half; → q8_0 to ~¼; → q4_0 to ~⅙.
- The **gap-to-shared-slab** is layout work, not quant work, and is
  tracked separately (M9/M10 ideal-mixed layout tasks).

This doc covers only the **quant** axis. It reads current engine state
(kvcache.h/.c + `kernels/qwen2_cuda.cu`) read-only and produces an
implementation plan that plugs quant in at the existing write/read
boundaries (`k_kv_scatter`, `k_flash_gqa`).

## 1. Quant types: K vs V

### What the literature / llama.cpp says

Authoritative measurements:

- llama.cpp **PR #6183** (LLaMA-v2-7B Q4_K_S, ctx 4096) — K cache
  quality vs size:

  | K dtype | size @ 4k ctx | wikitext PPL |
  |---|---:|---:|
  | fp16 | 1024 MB | 5.8671 |
  | q8_0 | 544 MB | 5.8681 (**+0.002%**) |
  | q5_1 | 384 MB | 5.8802 |
  | q5_0 | 352 MB | 5.8920 |
  | q4_1 | 320 MB | 5.9233 |
  | q4_0 | 288 MB | 5.9790 (**+1.9%**) |

- **v-code01/kvbits** (asymmetric study, multiple models): K needs
  ≥ 8 bits; V is lossless at 4 bits. **Kq8 / Vq4** asymmetric gives a
  59 % KV reduction at **< 1 % PPL**, while symmetric 4-bit is
  catastrophic.
- **TurboQuant (TQ1_0 / TQ2_0)** in llama.cpp uses a Hadamard rotation
  pre-quant to make 1–2-bit V cache viable.

### Why the rotation path is dangerous (do **not** lead with it)

Three concrete llama.cpp failures, all in the rotation pipeline:

- **#25382** (DeepSeek-V4): `--cache-type-k q8_0` + the model's
  Hadamard `attn_rot_k` produced "confident gibberish" on every
  backend. f16 cache worked. Root cause: sparse attention path
  ignored the rotated layout. Closed by PR #25202 — but only after
  burning user time.
- **#27109** (qwen3-5 hybrid): `q4_1`/`q4_0` K cache collapsed
  prefill from ~700 t/s to **~34 t/s** on RTX 3090. MMQ guard
  passed; the regression is in the K-write scatter path. Open at
  the time of writing.
- **TQ1_0/TQ2_0 history** (issue #24485 and follow-ups): the
  rotation types were reverted/broken multiple times. cf6270c
  notes "pre-rotate-queries never executed because Q ne[0]=256
  (GQA concatenated heads) vs rotation matrix ne[0]=128" — a
  dim-mismatch class of bug. The fix in atomic-llama-cpp-turboquant
  measured PPL 6.19 at 10.7 tok/s vs the q8_0-only 77.7 tok/s path
  — a **7× slowdown** for a 1.2 % PPL delta. Not worth it for us.

### Recommendation

| tensor | dtype | granularity | reason |
|---|---|---|---|
| **K** | **q8_0** (symmetric, 32-elem blocks, fp16 scale) | **per-head** (1 scale per kv-head per slot) | K feeds `QK^T/√d`; outliers in K dominate the attention logit. q8_0 is measured **+0.002 %** PPL on LLaMA-2-7B and is what llama.cpp's `dequantize_view` defaults to. Per-head (1 scale per kv-head per slot) matches llama.cpp's K layout: `K` is `[n_kv_heads, ctx, head_dim]`, so the natural block is the `head_dim`-wide row per slot. |
| **V** | **q4_0** (symmetric, 32-elem blocks, fp16 scale) | **per-token** (1 scale per token across the full kv-head × head_dim row) | V is consumed as `P @ V`; v-code01/kvbits measured V as lossless at 4 bits across multiple architectures. Per-token scales exploit the fact that V magnitudes vary more token-to-token than channel-to-channel within a slot — this is the granularity llama.cpp's V-cache dequant uses. |

K and V get **different dtypes and different granularities**. This is
the asymmetric Kq8 / Vq4 design.

### Skip rotation in v1

Rotation (Hadamard / TurboQuant) is a future option for pushing V
below q4_0. Defer it. The tar-pit is real: the only llama.cpp users
running TQ2_0 today are research forks, and they report 7×
slowdowns vs the q8_0 path. If q4_0 V is not enough, the next move
is **q4_K V with per-head scales** (iq4_nl is the llama.cpp "4-bit
non-linear" type that hit the same memory as q4_0 with q5-class
quality in PR #6183's table) — still no rotation.

## 2. Per-head vs per-token vs per-channel

- **K = per-head per slot**: K's natural storage shape is
  `[n_kv_heads, ctx, head_dim]`. One fp16 scale per row (= one per
  kv-head per slot). For Qwen2.5-0.5B that's 4 fp16 scales per token
  per layer. Trivial overhead: at ctx 32k, 4 × 32k = 128k fp16
  scales per layer = 256 KiB per layer of K metadata. Negligible.
- **V = per-token across the full kv row**: V's natural storage
  shape is `[ctx, n_kv_heads, head_dim]`. One fp16 scale per token
  per layer. Same overhead: 1 × 32k = 64k fp16 scales per layer =
  128 KiB per layer. V metadata is 50 % of K metadata because V has
  one scale per token, K has one per head per token and n_kv_heads
  can be > 1 (GQA).
- **Per-channel** (1 scale per element) is the iq-level granularity
  in llama.cpp and not what we want — too many scales to read
  efficiently inside the flash kernel.

## 3. Where to plug in

The engine today (after the post-fix in
`kernels/qwen2_cuda.cu:1144` referenced in `RESUME.md`) runs:

```
forward_layers(l):
  ...
  k_gemv_typed(K)   -> d_k_stage
  bias + qk_norm + RoPE on d_k_stage
  k_gemv_typed(V)   -> d_v_stage
  bias + (gemma4 V-RMSNorm on d_v_stage)
  k_kv_scatter(d_k_stage, d_v_stage, Kl_f, Vl_f, d_pos, KV_l, HDl, max_ctx)
  k_flash_gqa(d_q, Kl_f, Vl_f, d_att, d_pos, H_l, KV_l, HDl, ...)
```

### Hook points

| site | file:line | change |
|---|---|---|
| Engine storage alloc | `kernels/qwen2_cuda.cu:915-923` (`cudaMalloc(d_kc, …)`, `cudaMemset`) | Allocate enough for `cache_per * n_layers * k_bytes_per_elem` where `k_bytes_per_elem = 1 + 32/4 = 9` for q8_0 (1 fp16 scale + 32 int8 per 32-elem block packed into 32 bytes = 1 byte/elem avg) and 4/4 + 1/32 = 1.125 for q4_0. Add a per-layer scale buffer `d_k_scales [n_layers, ctx, n_kv_heads] fp16` and `d_v_scales [n_layers, ctx] fp16` allocated together. |
| Scatter write | `kernels/qwen2_cuda.cu:1239` (`k_kv_scatter<<<…>>>`) | Replace with `k_kv_scatter_q` that takes the typed slot, the scale-pointer, and a per-tensor dtype tag. Internally, per-block quant on 32 elements → write nibbles + 1 fp16 scale. For V: one block = the whole kv row (`KV_l * HDl` elems), per-token scale. For K: one block = one head's `head_dim` elements, per-head per-slot scale. **Stage in shared memory** (head_dim ≤ 256 fits in shmem) and write packed. |
| Flash read | `kernels/qwen2_cuda.cu:1249` (`k_flash_gqa<<<…>>>`) | Replace with `k_flash_gqa_q` that takes typed K/V, scale pointers, dtype tags. The K read path inside the loop (`kp = Kc + …; score += qreg[i] * kp[i]`) becomes `score += qreg[i] * dequant(kp_block, scale_k)` where `dequant` is a per-lane 32-elem block dequant. V path same. The per-lane dequant adds ~1 cycle/elem of FMAs; in steady state this is **memory-bound** on cache reads, not compute-bound. |
| Per-tensor config | `kernels/qwen2_cuda.cu` (engine struct) | Add `TTConfig.kq_type` and `TTConfig.vq_type` (enum: `TT_Q_F32`, `TT_Q_F16`, `TT_Q_Q8_0`, `TT_Q_Q4_0`). Default to `TT_Q_F32` (today's behavior) so nothing regresses. |
| Shared (KV-reuse) layer path | `kernels/qwen2_cuda.cu` `forward_layers` (`Kl_f = d_kc + pl_src[l] * cache_layer` block, ~line 1115) | **No change needed.** A shared layer's `Kl_f`/`Vl_f` already point into the source layer's slab; if the source layer is q8_0 K + q4_0 V, the shared layer reads the same packed buffer via the same dequant. The dtype tag is per-slab, owned by the source. **This is the main reason we do not want per-tensor dtype tags per layer** — keep the source slab's dtype the only thing that matters, and the shared layer inherits it for free. |
| KVC layout | `src/kvcache.h` + `src/kvcache.c` | The existing `tt_kvcache_cfg.dtype_size` becomes a misnomer. Add `kq_bytes_per_elem` and `vq_bytes_per_elem` (replacing the single `dtype_size`); `tt_kv_memory()` reuses the per-slab formula and multiplies by the relevant byte factor. Blob layout (offsets, strides) is **unchanged** because it is expressed in *elements*, and element-counting still works on the packed representation. |
| SWA compaction | `src/kvcache.c` `tt_kv_compact_plan_from_keep` and `tt_kv_zero_range` | The existing memmove/zero plans operate in element counts. With packed storage the "element" is 1 byte at the wire, but the engine treats it as a packed-block row. Two choices: (a) add `bytes_per_elem` to the move plan and let the engine `cudaMemcpy2D` the packed slabs; (b) keep an fp32 staging copy for compact/zero operations and re-quant afterward. **(a) is correct and free** — `cudaMemcpy2DAsync` handles packed layouts natively. |
| Serialize | `src/kvcache.c` `tt_kv_serialize` | Header grows two new fields: `kq_bytes_per_elem`, `vq_bytes_per_elem`. Wire format bump to `TT_KVCACHE_VERSION = 2`; old v1 blobs fail with `-2` (bad version) and callers fall back to fp32 replay. Scales are part of the K/V blob — no separate save. |

## 4. CPU / hybrid path

If a model runs CPU-only (offload), KV lives in host fp32 already.
Quant is a **memory-only** optimization, not a compute optimization
in our CPU backend (cpu_backend.c does dequant-on-the-fly for
weights; KV-cache hits no GEMV). The benefit on CPU is therefore:

- Smaller RAM footprint — meaningful for hybrid offload where the
  KV must fit alongside weights on a 16/32 GB host.
- No compute speedup (the K read is a `score += q*K` inner loop, not
  a GEMV).

**Decision: defer.** Add a `cpu_kq_type`/`cpu_vq_type` field in
`TTConfig` that mirrors the GPU path, but the implementation is
**fp16 K + fp16 V only on the CPU side** in the first pass. Pushing
to q8_0 / q4_0 on CPU requires AVX2 dequant kernels (open work item
E2 in TODO.md) and a separate parity sweep. Quant-K for GPU is the
target; CPU follows once the GPU path is stable.

## 5. Quality loss estimation

| K dtype | V dtype | PPL delta vs fp32 (llama.cpp LLaMA-2-7B) | source |
|---|---|---:|---|
| f16 | f16 | 0.000% | PR #6183 |
| q8_0 | f16 | +0.002% | PR #6183 |
| q8_0 | q4_0 | < +1.0% | v-code01/kvbits (asymmetric) |
| q4_0 | f16 | +1.9% | PR #6183 (K is the bottleneck) |
| q4_0 | q4_0 | catastrophic | v-code01/kvbits (symmetric 4-bit) |

### Our proposed v1 = Kq8_0 / Vq4_0

Expected PPL delta: **< 1 %** vs the fp32 ideal-mixed shape,
dominated by V quantization error (Kq8 is essentially free per the
table above).

### Parity test plan (because we have no llama.cpp-equivalent for
our specific models)

The repo's gate scripts (`tests/gate_ple_golden.py`,
`tests/gate_m6_parity.py`) measure per-token logit diff against a
golden numpy reference. They do not measure PPL directly. Add:

1. **wikitext-2 PPL sweep** at the existing M5/M6 fixture set
   (`tests/fixtures/models.json` — `qwen2.5-0.5b-instruct-q4_0` is
   the only small fixture; add a wikitext-2 slice under
   `data/wikitext2/`). Measure PPL at
   `fp32 / fp16-KV / q8_K-q8_V / q8_K-q4_V` (the four meaningful
   points). Pass criteria: PPL delta < 1.0 % at the chosen v1
   dtype. **Owner: M11.** Cost: ½ day.
2. **logit-diff gate** in the existing test harness, comparing
   `q8_K-q4_V` against the `fp32` golden at fixed prompt
   `(reasoning, code, math)` — pass if `mean abs diff < 5e-3` and
   `top-1 token agreement > 99.0 %` over 256 generated tokens.
   This is the regression net for ongoing changes.

The Qwen3-0.6B-q8_0 fixture is small enough that the PPL sweep is
fast on CPU even outside the GPU gate. gemma4 (when it lands) needs
a separate sweep because the q/k/v RMSNorm trait changes K
statistics; defer to when gemma4 parity is on the matrix.

## 6. Implementation plan

### Phase A — fp16 K and V (1–2 days)

The "trivial first win." No quant math; just change the storage
type and the dequant-on-read to a `float16 → float32` cast in
`k_flash_gqa`.

- Add `TTConfig.kq_type = TT_Q_F16`, `TTConfig.vq_type = TT_Q_F16`.
- Allocate `d_kc` and `d_vc` at `2 bytes/elem`.
- Add `__half` dequant path in `k_flash_gqa` (one cvt per
  element; fully vectorized; trivial).
- `k_kv_scatter_q`: cast fp32 → fp16 on write, one thread per
  element, no scale buffer.
- **Result: cache halves.** 0.137 → 0.069 / 0.469 → 0.234 /
  1.875 → 0.937 GiB.
- Risk: 0 (cast is exact up to fp16 precision; matches the q8_0
  PPL row's fp16 baseline).

**Phase A is the codepath validator.** If `k_flash_gqa_q` is wrong
in fp16 mode, it will be wrong in q8_0 mode too. Land A first,
regress parity at the existing golden, then enable B.

### Phase B — q8_0 K, q4_0 V (1 day after A)

- Add `TT_Q_Q8_0` and `TT_Q_Q4_0` to the dtype enum.
- Add per-block quant/dequant in a new shared
  `kernels/kvquant.cuh` (header-only block dequant/quant,
  templated on type):
  - `quantize_q8_0_block(const float* x32, uint8_t* qs_out, __half* d_out)`
  - `quantize_q4_0_block(...)` (subtract 8, pack nibbles, fp16 scale)
  - `dequant_q8_0_block(...)` and `dequant_q4_0_block(...)` — used
    inside `k_flash_gqa_q`'s inner loop. Per-block dequant fits in
    8–10 instructions; the inner loop stays memory-bound.
- `k_kv_scatter_q`: per-block quant on write. For K, one block per
  kv-head per slot (size = `head_dim`); for V, one block per token
  across the full kv row (size = `KV_l * HDl`).
- `k_flash_gqa_q`: dequant-on-load. q8_0 K dequant is one
  `__byte_perm`-style unpack + one `fmul`; q4_0 V dequant is two
  nibble unpacks + bias add + one `fmul`. Both fit the
  `head_dim / 32` lanes cleanly.
- Update `tt_kv_memory` to take `kq_bytes_per_elem` and
  `vq_bytes_per_elem`.
- Update serialize: wire format v2.

**Result: 32k cache drops from 1.875 → ~0.55 GiB on the K side
(q8_0 is ~25 % of fp32) and ~0.30 GiB on the V side (q4_0 is
~12.5 %). With current shared-slab, the new number is
`0.469 × (K_to_q8 + V_to_q4) / 2 = 0.469 × 0.1875 ≈ 0.088 GiB` at
8k ctx, and `1.875 × 0.1875 ≈ 0.352 GiB` at 32k ctx. **Fits E4B on
4 GB by a wide margin.** Numbers in §7.

### Phase C — deferred (only if needed)

Pushes only if Phase B's PPL sweep fails the `< 1 %` bar (unlikely
per §5) or if a future E4B-on-2GB class path appears:

- **C1**: q4_K V (iq4_nl — non-linear 4-bit with lookup) instead of
  q4_0 V. Same memory as q4_0, q5-class quality. Adds lookup
  table to dequant; minor.
- **C2**: q4_0 K (skip — measured catastrophic for some models).
- **C3**: Hadamard rotation for V (TQ1_0/TQ2_0) — only if the
  tar-pit clears in llama.cpp. Do **not** lead with this.

## 7. Memory savings (E2B, fp32 → phases)

E2B with current shared-slab baseline: 0.059 / 0.469 / 1.875 GiB.

Conversion factors vs fp32 storage:

- Phase A (fp16 K + fp16 V): **0.5×** on each tensor.
- Phase B (q8_0 K + q4_0 V): q8_0 = `1 (fp16 scale) + 32 (int8 packed)` over 32 elems ≈ 1.031 bytes/elem → **0.258×**; q4_0 = `1 + 16` over 32 → 0.531 bytes/elem → **0.133×**. Weighted average per slab (K and V are equal size): `(0.258 + 0.133) / 2 = 0.195`.

| phase | 1k ctx | 8k ctx | 32k ctx | vs shared-slab | vs ideal-mixed fp32 |
|---|---:|---:|---:|---:|---:|
| shared-slab (today, fp32) | 0.059 | 0.469 | 1.875 | 1.000× | 1.59× |
| **A** fp16 K + fp16 V | 0.030 | 0.234 | 0.937 | **0.500×** | 0.79× |
| **B** q8_0 K + q4_0 V | 0.012 | 0.092 | 0.366 | **0.195×** | 0.31× |

**Interpretation**: Phase B takes the **E4B-on-4GB 32k cache** from
1.875 GiB (today, 47 % of the budget) to **0.366 GiB** (9 % of
budget) — leaving 3.6 GiB for weights, headroom, and activations.

The ideal-mixed fp32 ceiling (0.297 GiB at 8k ctx / 1.188 GiB at
32k) is no longer the relevant target once we quantize: with Phase
B, we are already **0.31× under it**. The new meaningful target
is "stay above PPL -1 %" and Phase B is well inside that bar per
§5.

## 8. Risks ranked

| # | risk | likelihood | impact | mitigation |
|---|---|---|---|---|
| 1 | **Quality regression** (PPL > 1 % at q8_K/q4_V) | low | high | Phase A lands first as a zero-math reference; Phase B's PPL sweep is the explicit gate; q4_0 K is off the menu (v-code01/kvbits: catastrophic). |
| 2 | **Flash attention dequant-in-loop cost** | medium | medium | Inner loop is memory-bound; per-block dequant is ~8 instructions, well under DRAM latency. ncu-validate at Phase B; if it shows up in profile, hoist dequant to a separate pre-pass per layer (one tile per warp) and reuse via shared memory. |
| 3 | **KV-shared layer dtype inheritance** | low | medium | Shared layer reads source layer's slab via existing `pl_src` remap; no change. Verified in §3. Risk: if a future model has heterogeneous K/V dtypes *per source*, the design must gain a per-slab dtype tag. Out of scope today. |
| 4 | **cudaGraph capture invalidation** | medium | medium | `k_flash_gqa` runs **inside** the captured decode-step graph (M6.3 graph replay; `kernels/qwen2_cuda.cu:958` forces eager when `TT_PROFILE` is set, but the captured path is the default). The new `k_flash_gqa_q` must remain capture-safe: no host syncs, no `cudaMalloc`, no events. Quantize-on-write also lives inside the captured region. Mitigation: capture-time validator (run a probe replay and check the graph didn't change). |
| 5 | **SWA compaction / zero range on packed slabs** | medium | low | Memset-to-zero of a packed K slab still works byte-wise. The existing `tt_kv_zero_range` in element-counts needs a `bytes_per_elem` companion in the plan; trivial change in `src/kvcache.c`. `cudaMemcpy2DAsync` for compaction is byte-exact. |
| 6 | **Wire-format backward compat** | low | low | Bump to v2; v1 readers fall back to fp32 (the engine never persisted quantized K today, so nothing to migrate). |
| 7 | **CPU-path divergence** | low | low | Defer; CPU offload path keeps fp32 KV in v1 (or fp16 once Phase A's codepath is stable enough to share). |
| 8 | **Gemma4 V-RMSNorm + q4_0 interaction** | low | medium | The gemma4 `k_qk_norm_rms` call on V reduces dynamic range before quant, which should *help* (smaller scale, tighter int distribution). Untested; measure PPL on gemma4 fixture when it lands. |

## 9. Hook-point summary (one-line list)

- `include/qwen2_engine.h` — add `kq_type`, `vq_type` to `TTConfig`.
- `kernels/qwen2_cuda.cu:579-608` — engine struct: add `d_k_scales`, `d_v_scales` ptrs.
- `kernels/qwen2_cuda.cu:915-923` — alloc `d_kc`/`d_vc` at typed byte size + scale buffers.
- `kernels/qwen2_cuda.cu:1239` — replace `k_kv_scatter` with `k_kv_scatter_q`.
- `kernels/qwen2_cuda.cu:1249` — replace `k_flash_gqa` with `k_flash_gqa_q`.
- `kernels/qwen2_cuda.cu:1115` — shared-layer `Kl_f`/`Vl_f` remap: no change (inherits source dtype).
- `kernels/kvquant.cuh` (new) — `quantize_q8_0_block`, `quantize_q4_0_block`, dequant counterparts.
- `include/kvcache.h` + `src/kvcache.c` — replace `dtype_size` with `kq_bytes_per_elem`/`vq_bytes_per_elem`; bump `TT_KVCACHE_VERSION` to 2; update `tt_kv_memory`, `tt_kv_zero_range`, serialize.
- `tests/gate_ple_golden.py` — add wikitext-2 PPL sweep at 4 dtype combinations.

## 10. Open questions

- **Per-tensor dtype tag in CUDA** — the simplest implementation has
  one K-dtype and one V-dtype for the whole engine (matching
  llama.cpp's `--cache-type-k` / `--cache-type-v` switches). If
  gemma4 needs mixed per-layer (e.g. SWA layers q8_0 V, full-attn
  layers q4_0 V) the design extends: `pl_src`-style per-layer dtype
  arrays, no conceptual change. **Default: single K dtype + single
  V dtype engine-wide.**
- **Backward decode of old sessions** — session resume via
  `tt_kv_deserialize` would need a v1→v2 bridge. The cleanest
  answer is: keep the engine state machine on v1 blobs, only
  re-quantize when a session *is opened* on the new engine. No
  in-place migration. Confirm with chat-session work in M11.
- **PPL sweep fixture** — we need a wikitext-2 slice. The repo
  has no LLM eval data today; sourcing it is a small task on its
  own. Add to M11 prep.

## 11. References

- llama.cpp **PR #6183** — K-cache PPL table (fp16 / q8_0 / q5_1 / q5_0 / q4_1 / q4_0) on LLaMA-v2-7B. The canonical "K quality" reference.
- llama.cpp **PR #2969** — original q8_0 KV cache PoC.
- llama.cpp **PR #7412** — CUDA quantized KV cache demo.
- llama.cpp **PR #21038** — Hadamard rotation for KV cache (TurboQuant-style).
- llama.cpp **PR #22631** / issue **#21352** — Fast Walsh-Hadamard transform (rotation O(N log N) vs O(N²)).
- llama.cpp **issue #25382** — DeepSeek-V4 q8_0 K + Hadamard = garbage (rotation ignored by sparse attention).
- llama.cpp **issue #27109** — q4_0/q4_1 K cache collapses prefill 20× on qwen3-5 hybrid.
- llama.cpp **PR #25202** — fix quantized KV for DSV4 (concat alignment to 512 required).
- **v-code01/kvbits** — asymmetric K/V bit allocation study. Kq8/Vq4 = 59 % reduction at < 1 % PPL.
- **localbench.substack** (Gemma 4 / Qwen 3.6 q8_0 and q4_0 KL-divergence study) — cross-model validation.
- **back2matching/kvcache-bench** — tok/s and VRAM table for f16/q8_0/q4_0 at ctx 4k/16k.
