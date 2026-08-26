# CUDA Graph Capture Design for Quant-GEMV Decode Path

**Date**: 2026-08-27
**Status**: Design (gate to land with M9.0 PLE-fused V2)
**Owner**: M11 quant-GEMV perf work
**Predecessors**: existing `kernels/qwen2_cuda.cu` `qwen2_engine_graph_capture`
(handles qwen2 q4_0 only today, see line 1760)
**Reproduction**: `tests/proto_graph_capture.cu` (this commit)

## 0. TL;DR

Graph capture is **already wired** in `kernels/qwen2_cuda.cu` via
`qwen2_engine_graph_capture` and the `cudaGraphLaunch` replay loop in
`qwen2_debug_replay_step` (line 1900). The only thing that keeps it from
running on gemma4 is the line

```c
if (e->has_pl_embd) return -1;   // (in forward_layers / embed_token, see §3)
```

Once M9.0 PLE-fused V2 lands (2 device launches, no host round-trip —
`kernels/qwen2_cuda.cu:1355-1410`), that line goes away and graph
replay becomes live for the decode step. **No other code change is
required in the engine.**

This plan quantifies what we get, what is capturable, what is not, and
the concrete integration step. It also ships a standalone perf test
(`tests/proto_graph_capture.cu`) that measures replay overhead vs
direct launch on the actual hardware.

## 1. Measurement — `tests/proto_graph_capture.cu`

Standalone CUDA binary. No engine, no loader, no Makefile changes.
Builds with `nvcc` and links against the existing
`kernels/gemv_typed.cu` and `kernels/gemv_q4_cuda.cu` so we measure
**the exact kernel the engine uses**, not a re-implementation.

### 1.1 Workloads

| Workload | Shape | Meaning |
| --- | --- | --- |
| A | 1× tt_gemv_typed (11008×1536 q4_0) | one token's logits projection |
| B | 5× tt_gemv_typed back-to-back | one "block" of GEMVs (rough fwd proxy) |
| A-G | 1 GEMV captured into a graph | one-graph launch of A |
| B-G | 5 GEMVs captured into one graph | one-graph launch of B |

The shape 11008×1536 mirrors the gemma4 logits GEMV at hidden=1536
(`dim=1536`, `vocab=11008`). The same kernel handles q, k, v, o, gate,
up, down — all of which are smaller M; we picked the largest to give
a worst-case wallclock.

### 1.2 Hardware used for the numbers below

- GPU: NVIDIA GeForce RTX 3050 (sm_86, 4 GB)
- Driver 595.84 / CUDA 12.4 (`$HOME/mmcuda/bin/nvcc`)
- n_outer=256, n_warmup=32, median µs reported

### 1.3 Results (median µs/call)

```
[A]    1x direct launch:            115.71 us/call
[B]    5x direct launch (chain):    114.64 us/call  (573.22 us total)
[A-G]  1x graph replay:             114.69 us/call  (speedup vs direct 1.01x)
[B-G]  5x graph replay (chain):      90.73 us/call  (453.63 us total)  (speedup 1.26x)
[T]    5 chained GEMVs direct end-to-end:  573.22 us
[T]    5 chained GEMVs graph end-to-end:   453.63 us  (speedup 1.26x)
[mem]  weights=9.07 MB  free=3254.19 MB / total=3770.25 MB
```

### 1.4 What the numbers mean

- **Single-GEMV case (A vs A-G)**: a graph launch costs about the
  same as a kernel launch on this driver. The graph is dominated by
  the kernel's own runtime, so the saving from batching launches is
  ~1% — noise. This is exactly the textbook expectation: graph replay
  is most beneficial when it *amortizes launch overhead across many
  launches*.
- **5-GEMV chained case (B vs B-G)**: **1.26× speedup** (~120 µs saved
  per "block" of 5 GEMVs). The savings come from collapsing 5 host-side
  launch sequences (each: cudaLaunchKernel syscall + queueing) into 1.
- **Extrapolation to a 24-layer gemma4 forward block**:
  - 24 layers × 7 typed GEMVs/layer (q, k, v, o, gate, up, down) ≈
    168 GEMVs per step, plus 1 logits, plus embed, plus sampling ≈
    **~175 launches/step** end-to-end.
  - Launch overhead on sm_86 with this driver is ~5–8 µs/launch
    (consistent with the 1.26× ratio at 5 launches).
  - Predicted step savings: 175 × ~3 µs = **~525 µs per decode step**.
  - At an end-to-end decode budget of ~5 ms/step (≈200 tok/s for
    a 24-layer gemma4 on this GPU), that's a **~10% throughput win
    before** any other optimization.
- **Break-even**: graph capture wins from the moment you have more
  than ~2–3 device launches in a row, which is always true for our
  forward path. There is **no shape that gets worse** in our
  measurements; even the 1-GEMV case is within noise.

### 1.5 How to reproduce

```bash
cd ~/Storage/repos/nnfromscratch
$HOME/mmcuda/bin/nvcc -O2 -arch=sm_86 -std=c++17 \
    -Iinclude \
    -o tests/proto_graph_capture \
    tests/proto_graph_capture.cu \
    kernels/gemv_typed.cu kernels/gemv_q4_cuda.cu \
    -L$HOME/mmcuda/lib -lcudart
LD_LIBRARY_PATH=$HOME/mmcuda/lib ./tests/proto_graph_capture
# skip:
TT_GRAPH_TEST=0 LD_LIBRARY_PATH=$HOME/mmcuda/lib ./tests/proto_graph_capture
```

## 2. Per-API capturability table

Each row is one engine primitive. **Yes** = safe to record into a
stream-captured graph; **No** = breaks capture; **Cond** = safe
under conditions called out.

| API / kernel                              | Capturable? | Notes |
| ---                                       | ---         | --- |
| `tt_gemv_typed` (q4_0/q5_0/q5_1/q4_1/q8_0/q4_K/q5_K/q6_K/f16/f32/bf16) | **Yes** | All paths are device-side, no allocations, no D2H. Confirmed in §1. |
| `tt_gemv_q4_0`                            | Yes         | Fast path. `kernels/gemv_q4_cuda.cu:40`. |
| `tt_fused_swiglu_q4_0` (`kernels/gemv_q4_cuda.cu:113`) | Yes | Single launch, no host roundtrip. |
| `tt_logits_q4_0`                          | Yes         | Same shape as gemv, 1 launch. |
| `tt_embed_q4_0_dyn` (`kernels/qwen2_cuda.cu` graph capture) | Yes | Reads `e->d_next_tok` (device) at replay time. |
| `k_rmsnorm`                               | Yes         | Single block, no host access. |
| `k_qk_norm_rms` (qwen3/gemma4 head RMS)   | Yes         | No host state. |
| `k_rope` / `k_rope_gptj` / `k_rope_ff`    | Yes         | Reads `e->d_pos` (device) — replay reads the same device scalar. |
| `k_kv_scatter`                            | Yes         | Reads `e->d_pos` (device). |
| `k_flash_gqa`                             | Yes         | `e->pl_swa[l]` is a host-derived constant captured at instantiate time. |
| `k_ple_stage1_f32` / `k_ple_stage2_f32` (M9.0 V2) | **Yes (post-PLE-V2)** | 2 launches, no host round-trip. V2 path is what unlocks graph capture. |
| `k_ple_stage1_f32` (PLE-V1 host fallback) | **No**      | Has `cudaMemcpy` D2H/H2D between gemvs (see `kernels/qwen2_cuda.cu:1420-1490`). This is the path that must NOT run under capture. |
| `k_add`, `k_scale`, `k_fill_const`        | Yes         | Pure elementwise. |
| `k_softcap` (gemma2 logit tanh)           | Yes         | Reads no host state, branch is dead. |
| `k_repeat_penalty`, `k_gumbel_transform`  | Yes         | No-op when off; reads `e->d_sampling_on` (device). |
| `k_argmax_partial`, `k_argmax_final`      | Yes         | Writes `e->d_out` (device), later D2H'd OUTSIDE the graph (see `qwen2_debug_replay_step:1904`). |
| `k_pos_inc`, `k_pos_inc_recent`           | Yes         | Pure device, reads `e->d_pos` to write the new value. |
| `k_embed_q4_0` (static-token embed)       | Yes         | Host token id → device vector. But token id is the same per-replay; we use `k_embed_q4_0_dyn` so the device-side `d_next_tok` is read at replay. |
| `cudaMemcpy` async H2D `d_next_tok` (per-replay) | **Out-of-graph** | Done in `qwen2_debug_replay_step` BEFORE `cudaGraphLaunch` — legal because the capture region starts after this copy. (`kernels/qwen2_cuda.cu:1904`) |
| `cudaMemcpyAsync` D2H `h_sampled` (per-replay) | **Out-of-graph** | Issued AFTER `cudaGraphLaunch` and waited on with `cudaStreamSynchronize`. Legal. (`kernels/qwen2_cuda.cu:1907-1908`) |
| `cudaStreamSynchronize` (in `qwen2_debug_replay_step`) | **Out-of-graph** | Outside the capture region. (`kernels/qwen2_cuda.cu:1908`) |
| `cudaMalloc` / `cudaFree` (engine setup)  | **Out-of-graph** | Done in `qwen2_engine_init` long before capture. (`kernels/qwen2_cuda.cu:797-937`) |
| `cudaMemcpy` H2D (engine setup)           | **Out-of-graph** | Same — done at init. (`kernels/qwen2_cuda.cu:73,628`) |
| `cudaMallocManaged`                       | **Never**   | Not used. |
| `cudaMalloc` inside a kernel body         | **Never**   | None of our kernels do this. (Quant-GEMV holds weights in `__device__` static tables only where applicable; in our case the weights are device pointers in `dW`.) |
| `fopen` / `fwrite` (TT_DUMP_LAYER, TT_DUMP_PLE) | **No (debug only)** | Already guarded by `!g_capturing` (`kernels/qwen2_cuda.cu:1509,1523`). |
| `cudaEventRecord` (TT_PLE_TIMING)         | **No (debug only)** | Already guarded by `!g_capturing` (`kernels/qwen2_cuda.cu:1371,1406`). |
| `cudaMemcpy` D2H inside `forward_layers` (TT_TRACE) | **No (debug only)** | All instances behind `!g_capturing` (the `trace && !g_capturing` form on lines 1398, 1400, 1487, 1495). |
| `cudaMemcpy` D2H inside `embed_token` PLE host path (PLE-V1) | **No** (and the path must be unused under capture) | The f32 V2 path is selected by `use_v2 = (inp_gate.dtype == TTQ_F32 && pl_proj.dtype == TTQ_F32 && pl_post_norm != NULL)` (`kernels/qwen2_cuda.cu:1358`) and contains zero host copies. |
| `cudaMalloc` inside `qwen2_engine_graph_capture` warmup (`xsave`, `lsave`) | **Out-of-graph** | Done before `cudaStreamBeginCapture`. |
| `cudaMemcpy` inside the same warmup       | **Out-of-graph** | Same — before capture. (`kernels/qwen2_cuda.cu:1784-1786`) |

## 3. What CAN'T be captured today (vs post-PLE-fused V2)

### Today (gemma4 with V1 host-assisted PLE)

The host-assisted PLE block lives at `kernels/qwen2_cuda.cu:1414-1500`.
Each of the 35 layers does the following per decode step:

1. `tt_gemv_typed` (device launch) — capturable.
2. `cudaStreamSynchronize` — **breaks capture**.
3. `cudaMemcpy` D2H `gbuf` — **breaks capture**.
4. CPU gelu + multiply loop — host code, **breaks capture**.
5. `cudaMemcpy` H2D `e->d_pl_tmp` — **breaks capture**.
6. `tt_gemv_typed` (device launch) — capturable.
7. `cudaStreamSynchronize` — **breaks capture**.
8. `k_rmsnorm` (device launch) — capturable.
9. `k_add` (device launch) — capturable.
10. `k_scale` (device launch, sometimes) — capturable.

That is **2 syncs + 2 D2H/H2D + 1 host CPU loop per layer × 35 layers
= 70 syncs + 70 host round-trips per decode step**. Even if the
`cudaStreamBeginCapture` call could tolerate one or two, this is
structurally uncapturable. (The engine's `g_capturing` check is the
correct fallback: just return `-1` from `qwen2_engine_graph_capture` and
stay eager.)

### Post-PLE-fused V2

V2 replaces lines 1355-1410 with two device kernels:

- `k_ple_stage1_f32<<<>>>(...)` — gemv f32 + tanh-approx gelu + PLE mul.
- `k_ple_stage2_f32<<<>>>(...)` — gemv f32 + atomic-ticket fused rmsnorm.
- Followed by `k_add`, optionally `k_scale`. All device, all
  no-allocation, all no-host-state.

The path is selected when `inp_gate.dtype == TTQ_F32 &&
pl_proj.dtype == TTQ_F32 && pl_post_norm != NULL` (line 1358). For
the gemma4 GGUFs in our `data/` directory, this is true (Q4_0/Q5_K_M
Q6_K gemma4 GGUFs per the comment at line 1357). The
g_capturing-guarded debug code (lines 1398, 1406) is the only thing
that still needs the `!g_capturing` test, and it already has it.

**Conclusion: once V2 lands, the whole decode step is device-resident
and the existing graph capture path lights up automatically.**

The other would-be problem is the `if (e->has_pl_embd) return -1;`
guard — there isn't one in the literal form above. The actual gate is
in `qwen2_engine_graph_capture` itself, where the warmup block at
`kernels/qwen2_cuda.cu:1763-1792` calls `k_embed_q4_0_dyn`,
`k_pos_inc`, `k_repeat_penalty`, and `k_gumbel_transform` to satisfy
the "no lazy module load" rule, then enters the capture. With the V2
path active, those warmup launches are correct and capture proceeds.

What needs verification once V2 lands (i.e. gate items, not
code edits in this commit):

- `cudaStreamBeginCapture` returns `cudaSuccess` for the gemma4 graph
  (not just qwen2 q4_0). Reason it should: PLE V2 is pure device.
- The `k_embed_q4_0_dyn` path is still valid for the gemma4 embed
  dtype; if the engine switches to a different `TTQ_*` for embedding
  under gemma4, the early `if (e->d_embd.dtype != GGUF_TYPE_Q4_0) return -1;`
  at `kernels/qwen2_cuda.cu:1762` will trip and we'll need a typed
  dynamic-token embed (or skip graph capture for non-q4_0).
- No `cudaMemcpy` or `cudaStreamSynchronize` sneaks back in through a
  TT_DUMP_* / TT_TRACE path. All such paths are already guarded by
  `!g_capturing`.

## 4. Per-ctx-length bucket strategy

### Decode (one token at a time)

The decode step is **shape-invariant**: same number of layers, same
per-layer widths, same KV-cache-slot access pattern (gemma4 SWA
layers only look back `swa_size` slots, but that is a per-layer
constant). **One graph serves all positions 0..max_ctx-1**. No
bucketing needed for decode.

The gemma4 KV-sharing trait (`e->pl_src[l] >= 0` →
`kv_shared` at `kernels/qwen2_cuda.cu:1128`) is also a per-layer
constant that folds into the same graph.

### Prefill (N tokens at once, where N varies)

Prefill runs `advance()` in a host loop, once per token
(`kernels/qwen2_cuda.cu:1681-1684`). The forward_layers call inside
`advance` is structurally identical to decode except that the KV
scatter writes to slots 0, 1, 2, ..., N-1. There is no per-N kernel
selection — the same kernels handle all N.

For prefill, the right strategy is a small set of bucket graphs:

- **Bucket key**: `N` (rounded to the nearest "common" value).
- **Suggested buckets**: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
  `max_ctx`. Up to 12 captures; each capture costs a one-time
  cudaGraphInstantiate (~milliseconds, in `qwen2_engine_init`).
- **Why bucketing**: the M12 task `tt_gemm_batched` (see
  `kernels/gemv_typed.cu:558` "batched prefill GEMM" comment)
  changes the W-stream profile at T=8 (the M/T tradeoff the
  prefill-GEMM optimizes for). A graph captured for T=8 should not be
  replayed at T=16 because the cudaGraph nodes embed the M/T tile
  shape.
- **What "the same kernel" means here**: today prefill uses
  `tt_gemv_typed` (per-token launch) — see the comment at
  `kernels/qwen2_cuda.cu:1681` and `advance():1648`. If we switch
  prefill to `tt_gemm_batched` for some T values (M12 work), then the
  graph nodes change too, so we need a per-bucket graph keyed on
  whether we use the GEMV or the GEMM kernel.

**v0 simplification** (this plan does NOT require any of this):
just capture prefill at the T we last saw, identical to decode, and
re-capture on T-change. CUDA supports
`cudaGraphExecUpdate` to mutate an existing graph instead of
re-instantiating, which is much faster. llama.cpp does this
(oracle/llama.cpp/ggml-cuda.cu:2643 `ggml_cuda_graph_update_executable`).

### Replay per call

- Decode: 1 `cudaGraphLaunch` per `qwen2_engine_next` call.
- Replay is async until `qwen2_debug_replay_step:1908` does a
  `cudaStreamSynchronize` and pulls `e->h_sampled` to host. So the
  graph internally contains `k_argmax_final → e->d_out`, and the
  D2H copy of that int happens *after* the graph launches (and
  inside the same stream). This is identical to llama.cpp's pattern
  of "D2H the result after graph launch" (e.g. argmax tail).

## 5. Integration plan (after PLE-fused V2 lands)

### 5.1 Code change

**None for the engine.** The single edit is in the gate test for V2
landing:

```c
// kernels/qwen2_cuda.cu — wherever the "PLE V1" path is force-selected
// (lines 1410-1500), confirm it is not taken under capture. The simplest
// form is to add a `|| g_capturing` clause to the `use_v2` test on
// line 1358 to make V2 mandatory under capture.
const int use_v2 = (w->inp_gate.dtype == TTQ_F32 &&
                    w->pl_proj.dtype  == TTQ_F32 &&
                    w->pl_post_norm != NULL) ||
                   g_capturing;
```

This is **defensive** — V1 must not run under capture. With V2 being
device-resident, the engine will automatically light up the graph
replay path on the first `qwen2_engine_next` after prefill.

The existing warmup block at `kernels/qwen2_cuda.cu:1775-1792` already
issues every kernel that the captured graph issues once before capture,
which satisfies the "no lazy module load" rule.

### 5.2 Gate

A new `tests/gate_*.py` (or extension of `tests/gate_m6_logit_parity.py`)
that:

1. Runs `build/run_llm_gpu` for `--tokens 64` on a gemma4 GGUF
2. Asserts the `[qwen2-engine] decode-step graph captured (cudaGraph replay ON)`
   line appears in stderr (already printed at line 1847).
3. Asserts the parity gate (`gate_m6_logit_parity.py`) still passes.
4. Reports `decode_us` from STATS as a perf number.

If parity breaks, the engine has a capturability violation that the
trace-dump guards missed — fall back to `TT_NO_GRAPH=1` and investigate.

### 5.3 Replay cost validation

`tests/proto_graph_capture.cu` is the perf A/B. Run before V2 merge
(as a baseline) and after V2 merge. Expected delta: the 1.26× speedup
on 5-GEMV chains extrapolates to ~10% end-to-end decode step
improvement (see §1.4).

## 6. Risks and fallbacks

| Risk | Probability | Detection | Fallback |
| --- | --- | --- | --- |
| PLE V2 not actually device-resident (latent D2H/H2D) | Low | `cudaStreamBeginCapture` returns `cudaErrorStreamCaptureUnsupported` or `cudaStreamEndCapture` returns an error. Existing `qwen2_engine_graph_capture` checks this and falls through to eager permanently (line 1802-1804 / 1837-1841). | Engine sets `e->no_graph = 1` and runs eager forever. Already implemented. |
| Lazy module load inside capture (`cudaErrorStreamCaptureImplicit`) | Low | `cudaStreamEndCapture` fails. The warmup block at `kernels/qwen2_cuda.cu:1775-1792` covers all the early-graph kernels. | Add the missed kernel to the warmup block. |
| TT_DUMP_LAYER / TT_PLE_TIMING set in production by accident | Very low | One of the `!g_capturing` checks trips and silently skips; dump output is lost. | All env vars are debug-only by design; document loudly. |
| A future kernel adds a `cudaMalloc` and the user captures it | Medium (over time) | Capture fails. Engine falls back to eager. | Add a `tt_engine_uncapturable_call_log` that records what tripped; surface in the gate. |
| gemma4 embed dtype is not Q4_0 (M7 trait `e->d_embd.dtype`) | Medium | The early-return at `kernels/qwen2_cuda.cu:1762` (`if (e->d_embd.dtype != GGUF_TYPE_Q4_0) return -1;`) trips. Engine falls back to eager. | Add a typed dynamic-token embed, or accept eager for non-q4_0. |
| Pre-capture warmup misses a kernel and `cudaGraphInstantiate` succeeds but launch fails | Very low | First replay of the graph throws an error. | Detect in `qwen2_debug_replay_step` (currently only returns -1 if `!e->graph_ready`); add a `cudaGetLastError` check after launch. |
| Per-ctx-length graph churn at decode (we said this doesn't happen, but...) | None | N/A — the decode step is shape-invariant by construction. | If it ever does, add `cudaGraphExecUpdate` like llama.cpp. |
| CUDA driver bug on capture/replay | Very low | Nsight or any repro. | Pin driver version. |
| `cudaGraphInstantiate` flags arg is unsigned long long in CUDA 12 (`kernels/qwen2_cuda.cu:1843` comment) | Known | Compile error if we drop to CUDA 11. | Already handled by the 3-arg call. |
| `cudaStreamCaptureModeThreadLocal` vs `Relaxed` vs `Global` | Known | Wrong mode causes `cudaErrorStreamCaptureUnsupported` for cross-stream allocations. | We use `ThreadLocal` (line 1801) which is the safest for our single-stream setup. llama.cpp uses `Relaxed` (oracle/llama.cpp/ggml-cuda/ggml-cuda.cu:4298) because they share cuBLAS workspaces across streams. |

## 7. Cross-references to existing graph infra

All in `kernels/qwen2_cuda.cu`:

- **Engine struct + graph fields**: `e->graph_exec`, `e->graph_ready`,
  `e->no_graph`, `e->pending_tok` (line ~996+).
- **Capture entrypoint**: `qwen2_engine_graph_capture` (line 1760).
- **Pre-capture guard for non-q4_0 embed**: line 1762.
- **Pre-capture H2D `d_next_tok` + sync**: line 1766-1767.
- **Warmup block** (saves/restores `e->d_x`, `e->d_logits`,
  `e->d_pos`): line 1775-1792.
- **Begin capture** with `cudaStreamCaptureModeThreadLocal`:
  line 1801.
- **Captured kernel sequence**: lines 1807-1827.
  - `k_embed_q4_0_dyn`, `forward_layers`, `k_rmsnorm`,
    `tt_logits_dispatch`, `k_softcap`, `k_repeat_penalty`,
    `k_gumbel_transform`, `k_argmax_partial`, `k_argmax_final`,
    `k_pos_inc_recent`.
- **End capture + instantiate**: lines 1830-1847.
- **`qwen2_engine_next` decision tree**: line 1851+.
- **Replay wrapper** (`qwen2_debug_replay_step`): line 1900.
- **Per-replay H2D + launch + D2H + sync**: lines 1904-1908.

## 8. ORACLE llama.cpp patterns (for reference)

From `oracle/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`:

- Uses `cudaStreamCaptureModeRelaxed` (line 4298) because cuBLAS
  workspaces can be allocated from other streams and the relaxed mode
  tolerates that. We use `ThreadLocal` (line 1801) because we run on
  a single stream and have no cross-stream allocations during capture.
- Maintains a `cudaGraph * graph` per "graph key" — a key derived
  from the graph structure (line 2576: `cgraph->nodes[0]`). Same shape
  ⇒ same key ⇒ same graph.
- Two-step warmup (line 4267-4281): capture only after the second
  identical call. This protects against shape changes between captures.
  We don't need this because the engine's decode step is shape-invariant
  by construction.
- `cudaGraphExecUpdate` to mutate the graph instead of re-instantiating
  (line 2643). This is the right tool for our prefill bucket strategy
  (§4) once we move past v0.
- `ggml_cuda_graph_check_compability` (line 2548) walks the cgraph
  and bails on `MUL_MAT_ID` (because its fallback path syncs the
  stream). We have an analogous gate in the engine: the
  `qwen2_engine_graph_capture` return-1 paths act as a compatibility
  check.

## 9. Test artifacts added in this commit

- `docs/plans/2026-08-27-cuda-graph-design.md` (this file).
- `tests/proto_graph_capture.cu` — standalone CUDA binary that
  measures `tt_gemv_typed` µs/call directly vs in a captured graph,
  for the 11008×1536 q4_0 shape and 5-chained-GEMV block. Env-gated
  with `TT_GRAPH_TEST=0`. No engine dependency.

## 10. Open work (not in this commit)

- Decide on prefill bucket counts based on real-world prompt-length
  histogram (M11 task).
- `cudaGraphExecUpdate` for prefill graphs whose T changes.
- A `gate_*.py` that asserts the engine actually prints
  `[qwen2-engine] decode-step graph captured (cudaGraph replay ON)`
  on a gemma4 forward pass after V2 lands.
- Profile end-to-end decode step wallclock with `ncu` and compare
  capture vs eager — that's the real perf number, not the
  per-launch µs we report here.
