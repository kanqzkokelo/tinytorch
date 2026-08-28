# GEMV V5 Optimization Findings — Rejection Report

**Date:** 2026-08-28
**Status:** REJECTED (V5 is 20x slower than V4)

## Benchmark Comparison (RTX 3050 sm_86)

| Shape ($M, K$) | V2 Baseline (ms) | V4 4-rows/warp (ms) | V5 Double-Buffered (ms) | V5 vs V4 Speedup | Max Abs Error |
|---|---:|---:|---:|---:|---:|
| 896, 896 | 0.008 | 0.006 | 0.092 | **0.07x** | 17.0 |
| 4864, 896 | 0.035 | 0.024 | 0.448 | **0.05x** | 56.7 |
| 896, 4864 | 0.034 | 0.024 | 0.485 | **0.05x** | 39.5 |
| 4864, 4864 | 0.126 | 0.086 | 1.860 | **0.05x** | 39.5 |
| 151936, 896 (LM Head) | 0.714 | 0.444 | 10.487 | **0.04x** | 56.7 |

## Why V5 Failed

1. **Zero Weight Reuse for $N=1$**: Single-token decode reads each Q4_0 weight once. Staging weights into shared memory adds `__syncthreads()` barrier stalls without offering any memory bandwidth reduction.
2. **Register Pressure & Unrolling**: Attempting double-buffered `__ldg` loops caused register spills and destroyed warp scheduling efficiency.
3. **Existing V4 is Optimal**: Commit `82567a6` (V4: 4-rows-per-warp) already hits maximum VRAM bandwidth utilization on RTX 3050 (0.444ms for 151,936 LM head rows = 145 GB/s, near the hardware limit of 176 GB/s).

## Decision

REJECT GEMV V5. Keep V4 as the active decode launcher.
