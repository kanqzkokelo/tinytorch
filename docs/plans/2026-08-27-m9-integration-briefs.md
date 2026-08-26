# M9 integration briefs — pre-staged for when kernels/ frees

> **STATUS: Not yet run.** These are pre-staged agent prompts. When T3 lands
> and `kernels/qwen2_cuda.cu` is clean, fire them in order. Each owns only
> `kernels/qwen2_cuda.cu` and the corresponding `tests/proto_*.cu`; the
> existing protos have already proven the perf wins.

## Sequence (fire in order; each independent after kernels/ is free)

1. **C1 — Wire `tt_gemm_batched_q4_0` into prefill path** (`kernels/gemv_typed.cu` already has the function)
2. **C2 — Wire PLE-fused V2** (`tests/proto_ple_fused.cu` has the kernel + epilogue ticket pattern)
3. **C3 — Wire Split-K flash** (`tests/proto_flash_splitk.cu` has the kernel + dispatch rule)
4. **C5 — Re-enable CUDA graph capture for gemma4** (already guarded by `if (e->has_pl_embd) return -1;` at line 1589, just remove once host PLE is gone)

## C1 brief

```
You are the M9.1 batched-prefill wire-in agent. Repo: ~/Storage/repos/nnfromscratch.
You own `kernels/qwen2_cuda.cu` (and ONLY that file). READ:
- kernels/gemv_typed.cu (the tt_gemm_batched_q4_0 function — already there)
- tests/proto_batched_gemv.cu (the perf numbers — f(8)=0.11x)
- tools/dump_logits.c (how prefill is invoked today)
- the function that handles prefill in kernels/qwen2_cuda.cu (search for "prefill"
  to find it; it's a few hundred lines; DO NOT modify it — you only add a
  dispatch point).

## Step-by-step

1. Add a single dispatch point at the prefill entry: when the number of input
   tokens ≥ 8 (or some smaller threshold you measure), use
   `tt_gemm_batched_q4_0(W, dtype, X_kmajor, Y, M, K, T, stream)` for
   attention Q/K/V/O and the MLP projections; otherwise fall through to the
   existing per-token tt_gemv_typed loop.
2. The X layout is [K][T] (column-major per token). For each weight matrix,
   X is the same input. For per-token we currently reshape to [K] and call
   T times; batched wants [K][T] in one call. So you need to arrange X into
   a single device buffer ONCE for the prefill call (it stays in the prefill
   path's scratch), and reuse for all M output rows.
3. Build + verify: ./scripts/ci_local.sh (must stay green), ./scripts/verify.sh m61 (must stay 7/7).
4. Time: prefill on qwen2.5 with 32 tokens before vs after. If f(8)=0.11x
   observation holds, pp tok/s goes 100 → ~900 tok/s on small models. If
   your implementation gives f(T) different, report honestly.
5. Commit: "M9: wire batched prefill (tt_gemm_batched_q4_0 in prefill path, n_tokens>=8)"
```

## C2 brief

```
You are the M9.0 PLE-fused V2 wire-in agent. Repo: ~/Storage/repos/nnfromscratch.
You own `kernels/qwen2_cuda.cu` ONLY.

## Step-by-step

1. Read tests/proto_ple_fused.cu (V2 design: 2 launches — k_ple_stage1
   gemv+gelu+ple_mul, k_ple_stage2 gemv_bf16+rmsnorm fused with atomic-ticket
   epilogue). Copy the kernel source INTO kernels/qwen2_cuda.cu (or
   declare extern and link from a new tests/ TU that becomes part of the
   engine build — your call).
2. Find the current MatFormer block in forward_layers (search "MatFormer"
   or "ple_finish_host" or "inp_gate" — it has host D2H/H2D per layer).
3. Replace the host path with V2: pure-device chain.
4. CRITICAL: this RE-ENABLES CUDA graph capture for gemma4. Once done,
   the `if (e->has_pl_embd) return -1;` guard in qwen2_engine_graph_capture
   (line ~1589) can be removed. Do that as part of the same commit.
5. Verify: ./scripts/verify.sh m61 (must stay green), ./scripts/verify.sh ple
   (3/3), ./scripts/verify.sh m84 if A1/T3 landed by now (should be same
   or better).
6. Measure: per-layer latency in forward_layers (add a cudaEvent if not
   already there) before and after. Expect ~22% per-layer latency reduction.
7. Commit: "M9: PLE-fused V2 in forward MatFormer (22% per-layer saving,
   graph-capturable for gemma4)"
```

## C3 brief

```
You are the M9.0 split-K flash wire-in agent. Repo: ~/Storage/repos/nnfromscratch.
You own `kernels/qwen2_cuda.cu` ONLY.

## Step-by-step

1. Read tests/proto_flash_splitk.cu (the split-K kernel + workspace + combine).
2. Find k_flash_gqa in kernels/qwen2_cuda.cu. It currently runs per-warp-per-head
   with full ctx loop.
3. Add a dispatch: when ctx > 128, call the new split-K path; otherwise
   keep the existing per-warp path.
4. Dispatch rule: S = clamp(ctx/256, 2, 16). Workspace ~1MB per model —
   allocate at qwen2_engine_create after model load.
5. Verify: ./scripts/verify.sh m61 (must stay green), ./scripts/verify.sh m84
   (if A1 landed; should be same or better — m84 doesn't exercise long ctx
   but should not regress).
6. Measure: at ctx=4096, time before vs after; expect ~10x at ctx≥1024.
7. Commit: "M9: split-K flash attention (12x at ctx>=1024, dispatch when ctx>128)"
```

## C5 brief (trivial — covered by C2)

When C2 lands, the `if (e->has_pl_embd) return -1;` guard becomes obsolete
since the host PLE round-trips are gone. Remove the guard in
`qwen2_engine_graph_capture` (line ~1589). Re-run m61 + m84 to confirm graphs
work for gemma4. Commit: "M9: re-enable CUDA graph capture for gemma4
(post PLE-fused V2 integration)"

## Rollback plan

If any of C1/C2/C3 regresses m61 (parity 7/7 lost), the agent must
investigate FIRST, not paper over. Each wire-in is a small
change; the worst case is "revert the dispatch branch" which is one
`if (n_tokens < threshold)` block.
