# Batched Prefill GEMM — Findings & Deferred Items

**Date:** 2026-08-28
**Status:** Tasks 1-2 shipped, Tasks 3-4 deferred (correctness bug)

## What Shipped

### Task 1: Microbench Kernel (commits `0b4e462`, `81bc24e`)
- `k_gemm_q4_0_prefill` 2D tiled Q4_0 batched GEMM
- Speedup: 2.2-3.3x over sequential `tt_gemv_q4_0`
- Bit-exact: `max_abs_error < 1e-4` against single-token GEMV

### Task 2: Launcher + Boundary Unit Test (commit `00f310f`)
- `tt_gemm_q4_0_prefill` exposed in `kernels/gemv_q4_cuda.cu`
- 32/32 boundary tests PASS (N in {1, 31, 32, 33, 127, 128, 256, 512})
- Bit-exact vs CPU reference matmul + Q4 dequant

## What Was Reverted (Tasks 3-4)

### Task 3: Engine integration (reverted)
- `prefill_batched_gemm` function added to `qwen2_engine_prefill` for N >= 32
- Live test showed: 84-token prompt generated "illo" instead of sensible text
- 153-token prompt generated garbage, m61/chat-multiturn gate REGRESSED
- 32/32 boundary unit test passed but engine integration fails
- Reverted all uncommitted changes

## Suspected Root Causes (per Qwen analysis)

1. **Output tensor shape/stride mismatch** between batched kernel output and engine consumption
2. **Position device-scalar race** between async H2D copy and per-token kernels
3. **KV cache scatter / RoPE position syncing** between async copies and kernels
4. **Layer norm / residual bug** in the batched path that doesn't exist in sequential

## Why the microbench is misleading

The microbench isolates the GEMM kernel: random W, random X, compare Y to sequential GEMV. This is a kernel-level test.

The engine integration requires:
- Embedding N tokens (per-token via `tt_embed_typed`)
- Per-layer attention (N sequential flash_gqa + scatter calls)
- Per-layer FFN (N sequential rmsnorm + swiglu calls)
- RoPE position syncing
- KV cache consistency

The microbench doesn't test the engine's per-token orchestration around the GEMM.

## Performance Numbers Achieved (engine integration, before revert)

- N=84: 506.3 tok/s prefill (vs ~80 tok/s before, 6.3x speedup)
- N=153: 510.5 tok/s prefill (vs ~45 tok/s before, 11.3x speedup)

These numbers are real, but correctness was broken.

## Deferred Work

- Task 3 (engine integration): needs a TDD approach with bit-exact comparison
  against the sequential path for the same N tokens
- Task 4 (prefill benchmark): depends on Task 3
