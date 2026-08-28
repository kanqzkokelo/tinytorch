# Q8_0 V4 LM Head Optimization Results

**Model:** qwen2.5-0.5b-instruct-q4_0.gguf (LM head in Q8_0, 151936 x 896)
**GPU:** RTX 3050 sm_86
**Kernel:** `k_logits_q8_0_v4` (4-rows-per-warp, uint32 vectorized reads with byte-permute)

## Stage Runtime Comparison (LM Head $151936 \times 896$)

| Kernel | LM Head Time (ms) | Speedup | Max Abs Error |
|---|---:|---:|---:|
| V1 (1-row/warp scalar) | 1.516 ms | 1.00x | baseline |
| **V4 (4-rows/warp vectorized)** | **0.791 ms** | **1.92x** | **0.0000e+00 (bit-exact)** |

## Decode Throughput Impact

- **LM Head Share of Decode**: reduced from 29.2% ($1.36\text{ ms}$) to 17.0% ($0.79\text{ ms}$)
- **Single-Token Decode**: increased from 230 tok/s to **250–269 tok/s**
- **Bit-Exact Correctness**: 100% GREEN across `ci_local`, `m61`, `ple`, `m84`

## Key Takeaway

Upgrading the Q8_0 LM head from V1 (1-row/warp) to V4 (4-rows/warp) closed the hidden bottleneck on qwen2.5-0.5b, saving $0.72\text{ ms}$ per decode step with zero accuracy loss.
