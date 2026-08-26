# M11 — Vulkan backend: architectural design

Date: 2026-08-27
Status: DESIGN (pre-code, per M11 roadmap gate)
Scope: hardware-breadth backend for the from-scratch engine. No code in this doc.
Follows: docs/plans/2026-08-26-m9-m11-roadmap.md §M11 ("Big design doc required before code").

## 0. Current architecture (what must be abstracted)

Inventory as of m6-correctness:

| Layer | Files | Vulkan-relevant coupling |
|---|---|---|
| Typed GEMV kernels | `kernels/gemv_q4_cuda.cu`, `kernels/gemv_typed.cu` | q4_0..q8_0/q4_K/q5_K/q6_K/BF16/F16/F32; warp-shuffle reductions; `__ldg`; bf16 intrinsics |
| Attention/elementwise | `kernels/qwen2_cuda.cu` | `k_flash_gqa` single-warp-per-head, `k_rmsnorm`, `k_rope{,_ff,_gptj}`, `k_qk_norm_rms`, `k_swiglu` fused, `k_kv_scatter`, two-stage argmax |
| Graph replay | `qwen2_engine_graph_capture()` → `cudaGraphExec_t` | whole decode step captured once; token id + pos read from device memory so one graph serves all tokens; `TT_NO_GRAPH` escape hatch |
| Host-assisted PLE | embed_token precompute before captured region | host-side, portable already |
| Dispatch | `src/arch_registry.c`, trait-driven forward | the natural seam for a backend dispatch table |
| Loader / CPU ref | `src/loader_gguf.c`, `src/dequant_ref.{c,h}` | already portable |

Key property to preserve: **single stream + persistent device activations**, decode is launch-bound (~50–100 launches/step) which is exactly why graph capture exists. Any Vulkan design that re-pays launch overhead per token loses the M6.3 win.

Goal: AMD/Intel/iGPUs/anything Vulkan 1.2+. Later Metal/ROCm behind same abstraction. Non-goal (M11): beating CUDA perf on NVIDIA — CUDA path stays primary.

## 1. Abstraction shape decision

### Options

**A. Thin runtime shim** — C ABI over ~15 functions mirroring what qwen2_cuda.cu actually uses:
buffer alloc/free/copy, kernel-launch-by-handle, barrier/fence, timeline wait, command-buffer record/replay, push-constants write. Kernels stay per-backend source files (`.cu` today, `.comp`+SPIR-V loader for Vulkan), selected by the existing dispatch table in `arch_registry`. Engine logic (`ops_llm.c`, trait-driven forward) untouched.

**B. Full op-abstraction layer** — `backend_ops.gemv_q4_K(...)` virtual table; every tensor op becomes an indirect call through a generic op interface; kernels become opaque behind it.

### Recommendation: A (thin shim), strongly

Maintenance-cost analysis for a solo maintainer:

- **Op-abstraction cost is quadratic in ops × backends.** We have ~30 kernels × (CUDA, Vulkan) = the interface itself becomes a third artifact that must be kept consistent on every kernel change. Every new quant (q3_K? iq?) touches interface + both impls + trait plumbing. Thin shim: new quant = one `.cu` + one `.comp`, zero interface churn.
- **The engine already IS an op layer.** Trait-driven forward in `arch_registry` + `ops_llm.c` already decide *which* kernel class runs. Option B duplicates that decision at a lower level. Two dispatch layers = classic double-dispatch bug farm.
- **Shim surface is provably small**: grep of `*.cu` shows the runtime API usage is `cudaMalloc/Free/Memcpy/Async`, `cudaLaunchKernel`, `cudaStreamSynchronize`, `cudaGraphInstantiate/Launch`, event timing. That maps ~1:1 onto Vulkan primitives. An abstraction wider than actual usage is speculative weight.
- **Escape-hatch rule**: any place the shim can't express something cleanly (e.g. cooperative groups in flash attention tuning), the backend keeps a private extension function reachable via one `void* backend_ctx` carried through the shim. This prevents the "abstraction gap" pressure that otherwise forces option B later.
- **Metal/ROCm payoff**: shim vtable ≈ Metal's command buffer/encoder model almost literally (MTLBuffer, MTLComputeCommandEncoder, MTLCommandBuffer). ROCm/HIP is nearly CUDA-shaped. Shim generalizes to all three with <10 extra entry points.

Concretely: `include/device_shim.h` (~150 LOC, C, no deps), `src/backend_cuda.c` wraps current calls (zero behavior change, verifies seam correctness by construction), `src/backend_vulkan.c` + `kernels/vulkan/*.comp`.

### Shim sketch (names only)

```
tt_dev_malloc / tt_dev_free / tt_dev_h2d / tt_dev_d2h / tt_dev_dtod
tt_stream_create / tt_stream_sync / tt_stream_wait_timeline
tt_kernel_get(name, spirv_or_cubin) -> tt_kernel*
tt_launch(tt_kernel*, grid[3], block[3], smem, stream, params*, param_sizes*, nparams, pushes*)
tt_cmd_begin / tt_cmd_launch / tt_cmd_end / tt_cmd_submit / tt_cmd_replay   // graph equivalent
tt_event_ms / tt_fence
tt_backend_probe() -> caps { subgroup_size, fp16, bf16, max_wg, smem, uma }
```

Params passed as array-of-pointers (CUDA-style). Vulkan side packs them into descriptor sets internally (see §3); CUDA side passes through to `cudaLaunchKernel` unchanged. This asymmetry is deliberate and cheap.

## 2. Compute-shader porting notes per kernel class

All shaders GLSL → SPIR-V via `glslc`/`glslang`, compiled **at build time** into a single embedded blob (llama.cpp's `vulkan-shaders-gen` pattern). No runtime GLSL dependency.

### 2.1 Typed GEMV (`gemv_typed.cu` family)

Current CUDA shape: one warp per output row, threads stride over K, `__shfl_xor_sync` tree reduce, dequant inline.

Vulkan mapping:

- **Workgroup = 128 or 256 invocations, 1D.** Do NOT hardcode warp=32. Query `subgroupSize` (via caps probe; NVIDIA 32, AMD RDNA wave32/64, Intel typically 32, older GCN 64).
- **Warp reduce → subgroup reduce**: `subgroupAdd`/`subgroupShuffleXor` requires `VK_KHR_shader_subgroup_basic/arithmetic/shuffle` — core-ish since Vulkan 1.1 but *feature-gated*: check `shaderSubgroupArithmetic` at probe time.
- **Subgroup-size variance strategy** (this is the crux): compile **two SPIR-V variants** per GEMV kernel — (a) `USE_SUBGROUP_ADD=1` pinned via `layout(constant_id) requiredsubgroupsize` to probed size, (b) fallback LDS tree-reduce across workgroup with `barrier()`s (works regardless of subgroup semantics). llama.cpp ships exactly this duality (`mul_mat_vec_q{4,6}_k.comp` `_subgroup` variants). Community data point: pinned-subgroupSize `subgroupAdd` GEMV was perf-neutral vs LDS fallback on RDNA4 — so don't burn weeks chasing variant (a); ship (b) first, (a) as opt-in flag.
- **Quant layout access**: q4_0 blocks are 18 bytes — misaligned for scalar loads. Use `uint32_t` loads + bitfield extract like the CUDA version does; keep byte layouts identical to GGUF (already true). Buffer device address or plain SSBO reads both fine; prefer SSBO arrays of `uint` for driver-compat.
- **Row-per-workgroup vs row-per-subgroup**: start row-per-workgroup (matches current warp logic scaled up), NUM_ROWS specialization constant (llama.cpp pattern: `NUM_ROWS=1..8` variants picked by hd size) for small hidden dims where launch count dominates.
- **Spec constants not template bloat**: hidden dim, K-stride, NUM_ROWS as `constant_id`s; one SPIR-V per config cached in pipeline cache keyed by (kernel, spec-consts).

### 2.2 Flash attention (`k_flash_gqa`, single-warp-per-head)

Current: one warp owns one head, streams KV, online softmax with shuffle reductions.

- Head-per-**workgroup-of-one-subgroup** maps directly when wg=subgroup size. On AMD wave64 this doubles arithmetic width per head — usually good for decode-length KV.
- **Shared-memory limits**: mobile/iGPU (Mali, older Intel) may report `maxComputeSharedMemorySize` as low as 16–32 KB (desktop typically ≥48–64 KB, some 96–112 KB). Current kernel's per-warp scratch is tiny (one head's q vector + running m/l), so fine — but if we later tile Q-blocks, add smem-size probe and fall back to smaller tile spec constant.
- Reductions: online-max + exp-sum are subgroup reductions again — reuse §2.1's dual-variant machinery.
- Softcap (`k_softcap`) and RMS qk-norm fold into the attention shader as spec-const flags rather than separate dispatches (fewer launches; matches fused philosophy).

### 2.3 rmsnorm / rope / swiglu / elementwise

Trivial. One workgroup per token-row (or flat invocation mapping), LDS partial-sum reduction for variance, no subgroup dependence. `rope_ff`/`rope_gptj` become spec-const style switch in one shader. Fused ffn (`k_fused_swiglu_q4_0`) ports as two pipelines back-to-back inside one command buffer — cross-kernel dependency handled by buffer barrier only between them (see §3).

### 2.4 "Graph capture" equivalent

This is THE architectural difference (roadmap open question). Verdict:

- There is no true capture API in core Vulkan. `VK_EXT_device_generated_commands` / NVIDIA execution-graph extensions exist but are vendor-skewed (NVIDIA-only for compute graphs in practice). Conditional rendering is graphics-stage only — useless here. **Do not chase capture semantics.**
- Instead: **explicitly-recorded command buffers per ctx-length bucket**, which is architecturally *cleaner* than capture because our engine knows its shapes statically:
  - Decode bucket: ctx lengths rounded up to buckets (e.g. 512-step granularity). Per bucket, record one primary cmd buffer containing every decode-step dispatch + barriers, at engine init / first-seen bucket. Token id & pos already come from device memory (the M6.3 dynamic-embedding trick) → **identical replay-any-token property survives**. Descriptor sets bound at record time; weights descriptors stable across steps.
  - Prefill: recorded fresh per prompt length (rare, amortized).
  - Re-recording on bucket change: ~ms-scale, once per 512 tokens — negligible vs per-token launch savings.
- Replay win mechanism identical to CUDA graphs: one `vkQueueSubmit` per token instead of N dispatches. The zolotukhin.ai RDNA4 analysis (2026-07) confirms reused-command-buffer submission recovers the CUDA-graph-class decode overhead on Vulkan.
- Keep `TT_NO_GRAPH`-equivalent env (`TT_VK_EAGER=1`) for profiling, mirroring current code.

### 2.5 Pipeline caching

- One `VkPipelineCache` created at device init, serialized to disk (`~/.cache/tt/vk_pipelines.bin`, hashed by driver+device ID) — cuts first-decode latency after cold start, matters because startup is a roadmap metric.
- Pipeline-per-(kernel × spec-config) stored in a hash map keyed by u64; lookup hot path is one hash + pointer compare. llama.cpp's `vk_pipeline` struct (name, shader module, pipeline layout, push-constant size, wg denoms, align) is the right shape — steal it nearly verbatim.

## 3. Memory-model differences

### Push constants vs UBOs

- Kernel launch params in CUDA (grid dims aside) → Vulkan split:
  - **Push constants** (≤128 B guaranteed, often 256): scalar args — hidden_dim, pos, n_heads, eps, scale, flags. Fastest path, no descriptor churn. Our kernels' scalar footprints fit easily.
  - **Storage buffers (SSBO)**: all pointers. One descriptor set per kernel binding layout.
- UBOs: skip entirely — SSBO covers read-only data fine and avoids a second descriptor layout family.

### Descriptor set strategy

Per pipeline: one `VkDescriptorSetLayout` fixed at shader compile (bindings = ordered params). Allocation via a few large `VkDescriptorPool`s with free-set recycling (llama.cpp `descriptor_set_mode`: per-descriptor vs per-set pooling — steal their simplification: bind-once-per-cmd-buffer since decode replays fixed sets).

For decode cmd buffers: descriptor sets written once at record time (weights never move; activations have persistent addresses thanks to the persistent-activations design — a lucky architectural match). Dynamic offsets avoided: allocate activations at aligned addresses instead, keeps sets static.

### Barrier discipline vs CUDA implicit ordering

- CUDA: same-stream launches serialize implicitly. **Vulkan: NOTHING serializes without explicit `vkCmdPipelineBarrier` or semaphore.**
- Rule set for recorded decode cmd buffer: between consecutive dispatches sharing a buffer, insert `vkCmdPipelineBarrier` with COMPUTE→COMPUTE, `MEMORY_WRITE→SHADER_READ|MEMORY_READ`. But coalesce: only insert where there's an actual RAW/WAR hazard — most adjacent kernels in the transformer step touch disjoint buffers (rmsnorm out → gemv in is a hazard; rope on q/k parallel to nothing else is not). Build a tiny hazard list per kernel pair during recording, not a blanket barrier-after-everything (blanket costs 5–15% on iGPUs).
- `vkCmdFillBuffer` for KV-tail memset (spec-decode rollback helper from M10 plan) replaces memset kernels.

### Timeline semaphores vs streams

- Single compute queue mirrors single-CUDA-stream exactly: submissions to one queue execute in order, so intra-engine ordering needs only barriers, no semaphores.
- Host sync points (`cudaStreamSynchronize` on sampling readback) → `vkQueueWaitIdle` initially (fine for sync-per-token), upgrade to **timeline semaphores** (`VkSemaphoreType.TIMELINE`, core in 1.2) when we want D2H readback overlapped with next-token dispatches: signal timeline value N at end of step, host `vkWaitSemaphores` on N, next submit waits N-1. Also the correct primitive for the eventual async-transfer queue (KV prefetch) — design the shim around timeline values now, use binary-semaphore-free flow always. Avoid binary semaphores entirely (reset-management nightmare per Khronos guidance).
- Transfer queue: optional second queue family for H2D weight load at startup; probe supports GRAPHICS+COMPUTE union. Low priority.

## 4. What breaks (honest list)

1. **Single-capture replay does not translate literally.** No cudaGraphCapture analog. Answered in §2.4: per-bucket explicit recording preserves the win; VK_EXT_device_generated_commands/execution graphs rejected (vendor skew, complexity). Residual risk: buckets multiply memory for recorded buffers — trivial (KBs each).
2. **BF16 variance.** `VK_KHR_shader_bfloat16` landed Mar 2025 (Vulkan 1.4.311), RADV/Intel Mesa 25.2 support, NVIDIA beta drivers — but NOT on older drivers, mobile, or Windows-pre-Mesa-25. Consequence: BF16 GEMV falls back to fp32-upconvert-on-load (bf16→f32 in shader, math in f32) — same trick as reading bf16 as u16 and shifting. Costs bandwidth, still correct. Gate native-bf16 pipeline behind probe feature flag. FP16: `VK_KHR_shader_float16_int8` widely supported (promoted 1.2) but `shaderFloat16` feature still opt-in per device — probe it; fallback f32 math for F16 weights likewise.
3. **Subgroup size 32 vs 64.** Handled by dual-variant compilation (§2.1) + requiredsubgroupsize pinning. Deeper impact: any kernel assuming 32-lane shuffle patterns (current warp-per-head flash attention) breaks semantically on wave64 unless written against `gl_SubgroupSize`. Rule for ALL ports: never assume 32; loop `for (i = gl_SubgroupInvocationID; i < N; i += gl_SubgroupSize)` and use sized-generic reductions. This is a rewrite discipline, not a flag.
4. **Intrinsics loss**: `__ldg` → just const-qualified SSBO reads (compiler infers); `__expf` → `exp2(x*log2e)` with mediump hints where precision allows; fast-math flags differ per compiler (`glslc` conservative by default — measure, may need `-ffaast` equivalents per shader). Argmax two-stage kernel: fine, pure integer work.
5. **Driver-quality spread** (biggest operational risk): Mesa RADV/ANV excellent and fast-moving; NVIDIA proprietary solid; **Windows AMD/Intel and Android (Mali/Adreno) laggy or buggy** with edge features. Mitigation: feature matrix in probe, CI on RADV + ANV + NVIDIA-Linux minimum; treat anything else best-effort. Known RADV gotcha historically: pipeline-cache deserialization bugs across driver versions → include driver version in cache key.
6. **Cooperative-groups-style tuning absent**: no cluster/distributed smem; fine — none of our kernels need it. WMMA/tensor cores: `VK_KHR_cooperative_matrix` exists but vendor patchwork; explicitly OUT of scope for M11 (GEMV decode doesn't want tensor cores anyway; prefill GEMM later).

## 5. Effort estimate, phases, risks

Assumptions: solo maintainer, session-economics constraints (inline work, committed gates). Estimates in focused working days, calendar-realistic ±40%.

### Phase 0 — spike (BUILD FIRST): smallest E2E q4_0-only Qwen2 decode on one AMD iGPU
Target hw: whatever AMD iGPU is at hand (RADV driver — debuggable, open-source, representative of worst-case subgroup handling). Scope cut to bone:
- Shim header + CUDA pass-through backend (proves seam, ~1 day, zero risk).
- Vulkan instance/device/queue/pipeline-cache scaffolding + q4_0 GEMV shader ONLY (LDS-reduce variant) + rmsnorm + argmax + embedding copy. Single-layer smoke test, then full qwen2 decode, eager mode, no graph, no flash attention (use naive attention path if needed — actually keep k_flash_gqa port, it's small).
- Gate: greedy decode matches CUDA logits within tolerance on fixed prompt (reuse existing correctness harness); tok/s reported but NOT gated.
Estimate: **5–8 days**. Kill criteria: if RADV iGPU can't hit ≥30% of CUDA tok/s on the same q4_0 model, revisit before investing further (unlikely; iGPU memory bandwidth explains most gap).

Deliverable value: proves shim shape, descriptor strategy, barrier discipline, subgroup fallback path — every risky unknown touched at least once.

### Phase 1 — full typed-GEMV parity + command-buffer replay (8–12 days)
- All quants q4_0→q6_K + q8_0 + BF16-fallback; embed/logits variants.
- Per-ctx-bucket decode cmd buffers replacing per-launch submits; `TT_VK_EAGER`.
- Subgroup-add GEMV variant as opt-in.
- Gate: bit-match suite green on RADV + NVIDIA-Vulkan (NVIDIA cross-check catches AMD-only assumptions).

### Phase 2 — full kernel parity + robustness (10–15 days)
- Flash attention full (GQA, softcap, qk-norm fusion), rope variants, fused swiglu FFN, kv_scatter, PLE path verification under Vulkan.
- Pipeline disk cache; fp16/bf16 feature gating matrix; Intel ANV bring-up.
- Gate: full verify.sh suite on RADV; benchmark table vs CUDA published in docs.

### Phase 3 — polish + breadth (ongoing, 5–10 days then tail)
- Windows AMD/Intel sanity, iGPU smem-limit fallbacks, transfer-queue overlap, Metal shim feasibility note (should be near-zero new design).

Total to "full parity on Linux AMD+Intel+NVIDIA": **~4–7 weeks part-time-solo**. Spike alone: week one.

### Risks ranked

1. **Driver fragmentation eating schedule** (likelihood high, impact medium): every "works on RADV" surprise repeats on ANV/Windows. Mitigation: lowest-common-denominator feature floor (no bfloat16, no cooperative matrix, no DGC in required path), CI matrix early.
2. **Perf cliff vs CUDA on iGPU memory-bound reality** (medium, medium): decode GEMV is bandwidth-bound; iGPUs have less bandwidth — expectations management: goal is *runs well*, not *matches dGPU*. Mitigation: phase-0 kill criterion.
3. **Subgroup-porting bugs silently corrupting outputs** (medium, high): wave64 shuffle bugs produce wrong-but-plausible tokens. Mitigation: bit-match harness against dequant_ref/CUDA oracle per kernel BEFORE integration; fuzz over ctx lengths.
4. **Descriptor/barrier bugs = intermittent hangs** (high likelihood during dev, low post-fix): validation layers mandatory (`VK_LAYER_KHRONOS_validation` in debug builds, CI runs validation-enabled smoke test).
5. **Solo-maintenance drag from second kernel dialect** (structural): GLSL-vs-CUDA drift. Mitigation: shared quant-layout headers via code-gen include (same struct defs textually included in .comp and .cu), doc'd port checklist per kernel class.

## 6. Prior art — what to steal, specifically

- **llama.cpp ggml-vulkan** (ggml/src/ggml-vulkan/ggml-vulkan.cpp + vulkan-shaders/*.comp):
  - `vk_pipeline` struct: name/module/layout/push-const-size/wg-denoms/align — adopt shape verbatim.
  - `vulkan-shaders-gen` build-time GLSL→SPIR-V→embedded-blob toolchain — replicate (we add CMake/Make target emitting a C array).
  - Dual-variant GEMV (`mul_mat_vec_q4_K.comp` + `_subgroup` flavor, USE_SUBGROUP_ADD) — the exact answer to subgroup variance.
  - Spec-constant specialization (NUM_ROWS etc.) + pipeline cache keyed per config.
  - Their matmul-shader column-major trick and split-K reduce for prefill GEMM later.
- **Khronos Vulkan subgroup tutorial + docs.vulkan.org ML-inference tutorial series** (vendor-optimizations chapter): workgroup sizing rules of thumb per vendor; subgroup broadcast/shuffle/reduce taxonomy; LDS sizing guidance.
- **TVM/MLC Relax** (tvm.apache.org/docs/arch/relax_vm.html, relax call_tir design): validates the thin-runtime choice from the opposite direction — MLC's own lesson is that a *small* device-API surface (their `Device`/`NDArray` + TIR packed funcs) survives new backends, while fat op-interfaces ossify. Steal their "kernels as opaque registered handles, graph executor holds no backend knowledge" separation — it's what our shim + arch_registry already approximates.
- **wgpu ecosystem cautionary tale**: candle #344 and the WebGPU dispatch-overhead study (arXiv 2604.02344) quantify per-dispatch validation/overhead killing bs=1 LLM decode — reinforces (a) going native Vulkan not wgpu, (b) command-buffer batching being non-negotiable. Burn's wgpu backend shows the abstraction-can-be-thin point too, but its perf notes confirm raw Vulkan is the ceiling.
- **zolotukhin.ai RDNA4 command-buffer-reuse post (2026-07)**: direct evidence reused cmd-buffer submit ≈ CUDA-graph win on Vulkan; cites the 14% llama.cpp CUDA-graph uplift as the target magnitude.
- **nvpro-samples/vk_timeline_semaphore**: reference flow for timeline-semaphore async compute/transfer — copy their signal/wait discipline for the future overlap path.
- **vulkanforge sprint14b commit (maeddesg)**: honest datapoint that pinned-size subgroupAdd GEMV was neutral on RDNA4 — deprioritizes subgroup variant work; LDS fallback is not a consolation prize.

## 7. Open questions (park until spike data)

1. Exact bucket granularity for decode cmd buffers (512 vs dynamic per-seen-length map) — measure re-record cost first.
2. Whether flash attention wants head-per-subgroup or head-per-workgroup on wave64 (spike measures both, one afternoon).
3. Prefill GEMM: keep cuBLAS-ref-style naive tiled shader vs defer prefill to CPU backend on Vulkan targets (iGPU prefill is often fine on CPU given memory-bound anyway).
4. Cooperative matrix adoption trigger: revisit when/if prefill throughput becomes a Vulkan-path product requirement.
