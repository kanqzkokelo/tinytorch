# V4-q8_0 and V4-F16: Why We Didn't Ship Them

**Date**: 2026-08-27
**Context**: After V4-q4_0 shipped successfully (commit 82567a6, 1.7× on LM head, 1.5× on FFN), we attempted to port the same V4 pattern to q8_0 and F16 dtypes. The microbench data showed V4 wins for q4_0 do NOT generalize.

## V4-q8_0 results (RTX 3050 sm_86)

| Shape | M | K | V2 μs | V4-q8_0 μs | ratio |
|-------|---|---|-------|------------|-------|
| Q/K/V proj | 1024 | 1024 | 9.6 | 8.4 | **1.14× wins** |
| FFN up/gate | 2048 | 1024 | 23 | 37 | 0.62× |
| FFN up/gate | 3072 | 1024 | 31 | 55 | 0.56× |
| FFN up/gate | 1024 | 3072 | 28 | 30 | 0.93× |
| LM head | 151936 | 1024 | 1016 | 1840 | 0.55× |

V4-q8_0 ONLY wins for tiny M=1024. For M≥2048 (which is most FFN shapes) it loses by 1.5-2×. The LM head is catastrophic at 0.55×.

**Why:** q8_0 has 34 bytes per 32-element block. V4's 4-rows-per-warp approach reads 4×34=136 bytes per block-step. At M=151936, this becomes the dominant cost. V2's 2-rows-per-warp reads 2×34=68 bytes per block-step, half the bandwidth.

**Lesson:** V4's amortization (sharing x across more rows) helps when K is small relative to M AND the per-row weight load is cheap. q8_0 doubles the per-row weight load, breaking the amortization.

## V4-F16 results (RTX 3050 sm_86)

V4-F16 was a near-mirror of V4-q4_0. Microbench showed small wins at LM head shape but neutral/negative elsewhere. The end-to-end bench regressed smollm2 from 0.76× to 0.73×.

**Why:** F16 weights are 2× the bytes of q4_0. V4's 4-rows-per-warp reads 4×K×2 bytes per warp vs V2's 2×K×2. The amortization benefit of sharing x is overwhelmed by the doubled weight load.

**Lesson:** V4's win is dtype-specific. q4_0 is the sweet spot (small per-row weight load). F16 needs a different approach (likely WMMA with proper coalesced dequant — see MMQ v1 below).

## MMQ v1 results (RTX 3050 sm_86)

| Shape | M | K | V2 μs | MMQ μs | ratio |
|-------|---|---|-------|--------|-------|
| Q/K/V proj | 896 | 896 | 8.4 | 47.4 | 0.18× |
| FFN up/gate | 4864 | 896 | 34.9 | 95.2 | 0.37× |
| FFN down | 896 | 4864 | 34.5 | 376.4 | 0.09× |
| LM head | 151936 | 896 | 905.8 | 2077.9 | 0.44× |

**Verdict:** MMQ v1 (WMMA m16n16k16 + q4_0 dequant) is 0.09-0.44× of V2 on every shape. Loses by 2-10× across the board.

**Why:** Tensor cores' mma_sync has 15/16 N-wasted for single-token decode. The dequant is uncoalesced (each lane reads 1 byte from a different q4_0 block, 32 separate transactions per K-tile). The mma is essentially free; the dequant is the cost. Bandwidth is 2-3× lower than V2.

**Lesson:** WMMA is the wrong tool for single-token decode on consumer Ampere. The compute is already free; what we need is bandwidth. Tensor cores help when the arithmetic intensity is high (batched decode, large M) — not the single-token case.

## What to try next (if we continue)

1. **V4 for K-quants** (Q4_K, Q5_K, Q6_K) — they have a different per-row structure (super-blocks of 256, with per-block scale/min). V4 may win for them because the per-block overhead amortizes differently.
2. **Coalesced-dequant MMQ** (per Qwen's suggestion) — 32 lanes cooperatively dequant 16 q4_0 blocks per wave. Could 2-3× MMQ bandwidth.
3. **Async-copy + double-buffered V2** — overlap weight loads with compute. May push V2 from 60% to 80% of peak bandwidth.
4. **Accept single-token decode ceiling and pivot to batched decode** — the architectural answer that Qwen flagged earlier.

## Commits
- V4-q4_0: 82567a6 (shipped)
- V4-q8_0 / V4-F16 / MMQ v1: NOT shipped, microbench data preserved
