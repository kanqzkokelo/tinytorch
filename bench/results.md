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


## LLM decode ladder (M6.3)

| Date | GPU | Config | Median decode tok/s | Parity |
|------|-----|--------|---------------------|--------|
| 2026-08-23 | RTX 3050 laptop | eager (post stream-unify) | 58.0 (96 tok, 5 runs, min 57.9 / max 58.1) | 7/7 top1 |
| 2026-08-23 | RTX 3050 laptop | cudaGraph replay of decode step | 59.1 (57 tok, 5 runs, min 58.8 / max 59.4) | 7/7 top1 |
| 2026-08-23 | RTX 3050 laptop | fused add+rmsnorm @ attn boundary — REVERTED | 59.2 / 59.0 (128 tok, 5 runs each) — no gain over graph-replay baseline; launch overhead already gone post-cudaGraph | n/a |
| 2026-08-23 | RTX 3050 laptop | lm-head tuning: q8_0 GEMV uint32+float4 vectorization (kept) | 75.1 (128 tok, 5 runs, min 74.7 / max 75.4); kernel 5.05→1.31 ms via standalone event microbench (28.7→110 GB/s eff). V1 rows/block 32 REVERTED (55.2, −6%); V4 two rows/warp REVERTED (75.8, +0.8% < 2% bar); V3 dp4a SKIPPED (x is fp32, on-the-fly int8 quant not viable) | 7/7 top1 |

### Task 6 note (rmsnorm multi-block): SKIPPED by measure-first analysis
49 rmsnorm launches/token x 1.5-3us in-graph execution = 0.6-1.1% of the 13.3ms step,
below the 2% action bar. Multi-block rewrite would touch the reduction for <0.5%
realistic gain. Revisit only if a future profile shows rmsnorm >2%.

