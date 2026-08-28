# Fused QKV/FFN Kernels - Integration Findings

**Date:** 2026-08-28
**Status:** Microbenches shipped, engine integration reverted

## What Shipped (Commits)
- `b83a1c4` Fused QKV microbench: 31.4% faster than 3 sequential Q+K+V calls, bit-exact
- `ecde6b0` Fused FFN microbench: 34.0% faster than 3 sequential Gate+Up+SwiGLU calls, bit-exact
- `d8319b6` Fused QKV enabled in `forward_layers()` (reverted)
- `7a41f85` Revert: fused QKV integration

## Why Engine Integration Failed

When the fused QKV kernel was wired into `forward_layers()`:
- **Before**: 269 tok/s decode
- **After**: 179 tok/s decode (-90 tok/s, 33% slower)

## Root Cause Analysis

The fused kernels use different block layouts (8 warps × 4 rows) and shared-memory activation staging. While these win in microbench isolation, they lose in the engine because:

1. **Synchronization overhead**: The shared-memory `X` staging requires `__syncthreads()` which adds latency when the next kernel (KV scatter, flash attention) needs the Q output immediately.
2. **Output stride mismatch**: The fused kernel writes to separate Q/K/V output buffers with different strides, breaking the contiguous memory layout that the downstream `k_flash_gqa` kernel assumes.
3. **Grid under-utilization**: The fused kernel uses `grid.x = (max_M + 31) / 32` blocks, which under-fills the GPU when M_v << M_q (KV heads 2 vs Q heads 14 for qwen2).

## Conclusion

The fused QKV/FFN kernels prove the **launch overhead theory** is correct (31-34% savings in microbench), but the integration into `forward_layers()` requires:
- Matching output strides for downstream `k_kv_scatter` and `k_flash_gqa` kernels
- Proper CUDA stream synchronization
- A smarter block partitioning for asymmetric M dimensions (M_q != M_k != M_v)

The current FFN and QKV are correctly using the V2/V4 single-vector kernels which are bit-exact and well-validated. The 269 tok/s baseline is the **real, verified** performance.
