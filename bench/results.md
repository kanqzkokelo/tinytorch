# CPU matmul benchmark results

- Date: 2026-08-23
- CPU: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz (12 logical cores)
- GPU clocks (context): 210 MHz, 405 MHz
- Protocol: warmup 5, median of 20 runs, fp32
- NumPy backend: OpenBLAS (pinned to 1 thread for np-1t row)
- OMP kernel threads: 6

| N | naive | AVX2 1T | AVX2 OMP | NumPy 1T | NumPy MT | AVX2/naive | AVX2/NumPy1T |
|---|-------|---------|----------|----------|----------|------------|--------------|
| 256 | 7.2 | 67.6 | 98.7 | 94.6 | nan | 9.4x | 0.72x |
| 512 | 6.8 | 63.2 | 159.9 | 81.6 | nan | 9.3x | 0.77x |
| 1024 | 5.7 | 64.3 | 238.9 | 84.1 | nan | 11.2x | 0.76x |

GFLOPS. Gate (AVX2 1T >= NumPy 1T @ 1024^3): FAIL

## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 88.7 | 2689.2 | 2043.0 | 2837.4 | 30.3x | 94.8% | 72.0% |
| 1024 | 115.2 | 4536.8 | 1514.7 | 4527.5 | 39.4x | 100.2% | 33.5% |
| 2048 | 116.1 | 4093.0 | 2769.4 | 4682.0 | 35.2x | 87.4% | 59.2% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 88.8 | 2692.9 | 2036.3 | 2960.0 | 30.3x | 91.0% | 68.8% |
| 1024 | 116.1 | 4560.6 | 1555.3 | 4529.5 | 39.3x | 100.7% | 34.3% |
| 2048 | 116.4 | 4085.8 | 2809.0 | 4188.1 | 35.1x | 97.6% | 67.1% |


## LLM decode ladder (M6.3) — RTX 3050 laptop, Qwen2.5-0.5B-Instruct q4_0/q8_0-head
| Date | Config | Decode tok/s | Parity |
|------|--------|--------------|--------|
| 2026-08-23 | eager (post stream-unify) | 58.0 | 7/7 top1 |
| 2026-08-23 | + cudaGraph replay | 59.1 | 7/7 top1 |
| 2026-08-23 | + lm-head q8_0 vectorize (float4 x, uint32 W) | **75.6** (min 75.0 / max 75.9, 5x128) | 7/7 top1 |
| — | lm-head V1 (rows/block 16→32) | reverted: slower (1.45 vs 1.31 ms kernel) | — |
| — | lm-head W-vectorize alone | reverted: +1.5%, under bar | — |
| — | fused add+rmsnorm | reverted (0% post-graphs) | — |
| — | rmsnorm multi-block | skipped (measured <1% share) | — |
| ref | llama.cpp tg (short ctx, this box) | ~58 | n/a |

Final: **75.6 tok/s decode** (median of 5x128 tokens), 1.30x llama.cpp reference.

### Task 6 note (rmsnorm multi-block): SKIPPED by measure-first analysis
49 rmsnorm launches/token x 1.5-3us in-graph execution = 0.6-1.1% of the 13.3ms step,
below the 2% action bar. Multi-block rewrite would touch the reduction for <0.5%
realistic gain. Revisit only if a future profile shows rmsnorm >2%.


### Accepted shortfall note (2026-08-23)
Plan target was >=100 tok/s; shipped 75.6 tok/s = 1.30x the llama.cpp reference on
this box but below plan goal. Remaining bottleneck is per-kernel bandwidth (~110 of
176 GB/s) across ~400 small GEMV launches; next levers are cp.async staging or
persistent-block schemes (documented in tools review). Parity maintained at every step.

## M6.3b profile baseline
Date: 2026-08-23 — NVIDIA GeForce RTX 3050 Laptop GPU (sm_86), Qwen2.5-0.5B-Instruct q4_0/q8_0-head, ctx<=6 tokens.

Harness: `tools/profile_step 785,6722,315,9625,374` — K=20 replayed steps, cudaEvents around each `qwen2_debug_replay_step`, MIN reported.
Per-stage table: `TT_PROFILE=1` forces EAGER mode (event records are illegal inside a captured region); per-stage cudaEvents inside forward_layers + sampling stage; table is median per-step ms over the same K=20 loop.

```
STEP_MS 12.072            (graph replay, min of 20)
```

```
PROFILE mode=eager        (STEP_MS 13.623 eager — launch overhead visible)
PROFILE embed           0.006
PROFILE qkv-gemv        1.471
PROFILE o+mlp-gemv      8.754
PROFILE flash           0.378
PROFILE rmsnorm         0.267
PROFILE kv-scatter      0.080
PROFILE logits-gemv     1.285
PROFILE argmax          0.568
PROFILE TOTAL(med)     12.809   (vs 13.62 eager STEP_MS; remainder = launch gaps + D2H)
```

Reading: o+mlp GEMVs = 68% of the step (the 253 MB q4_0 layer-weight read — Task 1 target). logits-GEMV 1.29 ms confirms Task 2 relevance (< 1.4 ms bar, marginal). flash 0.38 ms at pos<=6 = 2.9% — healthy. rmsnorm+scatter+embed ≈ 3% — confirmed skip. ncu: NOT available in $HOME/mmcuda/bin (`ls | grep -i ncu` empty, `which ncu` empty) — event timings carry the plan.

## M6.3b TASK 1.5 — argmax fix (kept)
Date: 2026-08-23. Baseline argmax 0.568 ms for a 600 KB logits read = launch-bound
(256 partial blocks x 128 threads + single-thread final over 256 partials).

Change: float4 vectorized partial kernel (64 blocks x 256 threads), final reduce now
parallel (1 block x 64 threads, tree, same value-then-index tie-break). Deterministic
tie-break preserved -> parity gate unaffected.

Result: argmax 0.568 -> 0.071 ms (-0.50 ms/step). Bench: 78 -> 80.0 tok/s median
(--runs 3 --tokens 64). verify.sh m61 7/7. KEPT.

## M6.3b TASK 1 — q4_0 GEMV variant ladder
| Variant | Change | step ms | tok/s | Verdict |
|---|---|---|---|---|
| baseline | scalar byte loads, blockDim.y=16 | 12.81 | 78 | — |
| (1.5) | argmax vectorized + parallel final | — | 80.0 | KEPT |
| V1 | uint32 W words + __byte_perm odd/even merge + float4 x (k_gemv_q4_0 AND k_fused_swiglu_q4_0) | 5.42 | 213.4 | KEPT |

V1 detail: port of lm-head winner pattern with q4_0 layout twist — row stride
nb*18 B (nb=28 -> 504 B, nb=152 -> 2736 B, both 4B-aligned); block qs starts at
18*b+2, aligned iff b ODD; nb even => last block odd-b => merges stay in-row.
o+mlp-gemv 8.81 -> 1.86 ms (~21 -> ~100 GB/s). qkv-gemv 1.48 -> 0.61 ms
(same kernel serves Q/K/V projs). Parity 7/7.
