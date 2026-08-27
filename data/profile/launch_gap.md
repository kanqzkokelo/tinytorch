# Launch Gap & Per-Stage Profile (Task 0 Gate)

**Method**: `tools/profile_step` with `TT_PROFILE=1` to disable graph capture and dump per-stage accumulators over 20 decode steps. Median reported.

**Model**: qwen2.5-0.5b-instruct-q4_0 (4 layers profiled, full 24-layer decode extrapolated)

## Per-stage breakdown (ms per token)

| Stage | ms | % of total | Notes |
|-------|-----|-----------|-------|
| embed | 0.005 | 0.1% | trivial |
| qkv-gemv | 0.604 | 13% | 3 GEMV: Q, K, V |
| o+mlp-gemv | **1.835** | **40%** | O proj + FFN (gate+up+down) |
| flash | 0.413 | 9% | attention matmul + softmax + matmul |
| rmsnorm | 0.245 | 5% | 2× per layer (pre-attn, pre-FFN) |
| kv-scatter | 0.071 | 2% | KV cache write |
| logits-gemv | **1.390** | **30%** | LM head (vocab projection) |
| argmax | 0.068 | 1% | sample |
| **TOTAL** | **4.632** | **100%** | = 216 tok/s (matches our 230 benchmark within noise) |

## Gate Decision: ABORT fusion, PIVOT to GEMV optimization

**Reasoning:**
- GEMV (qkv + o+mlp + logits) = **3.83ms = 83% of total decode time**
- All other ops (rmsnorm + flash + embed + scatter + argmax) = **0.80ms = 17%**
- Eliminating rmsnorm entirely (5%) gains <0.25ms = 5% decode speedup
- The fusion thesis (reducing kernel launches) does NOT apply when graph capture is on: kernels are back-to-back in a single graph replay

**The real bottleneck is the GEMV itself.** The scalar V2 path achieves ~60-70% of theoretical peak bandwidth per Qwen advisor. The remaining 30% gap to llama.cpp CUDA is in GEMV implementation, not in launch overhead.

## Pivot: Day 1 plan becomes GEMV-internal optimization

Three high-EV options, all targeting the GEMV:
1. **Q4_0 V2: switch from uint32 streaming to float4 reads** — current code uses `__ldg` on individual uint32s; float4 (16-byte) loads may double effective bandwidth on Ampere.
2. **LM head (logits-gemv) specialization** — vocab=151936 is very large; a V4 (4-rows-per-warp) variant for vocab>100k may amortize the x-load better.
3. **MMQ-style WMMA rewrite** for the GEMV — dequantize Q4_0 into FP16 registers, feed WMMA tiles. We already tried this in `tools/micro_v4.cu` (regressed on single-token). Re-examine with a more careful implementation.

## Bias inventory

See `data/profile/bias_inventory.md` — all 4 bench models have NO biases. Bias-fusion tasks are confirmed wasted work.

## Implications for the plan

- **Task 0: PASS** (decision made)
- **Task 1: SKIP** (no ncu needed; the breakdown above IS the answer)
- **Task 2 (RMSNorm+Q-GEMV fusion): REJECT** (5% of total, not worth the risk)
- **Task 3 (RoPE fusion): REJECT** (RoPE is a sub-component of flash, and flash is 9%)
- **Task 4 (bench+commit): REPLACE** with a new Task 4 = GEMV optimization dispatch

## Commit

This file: `profile: per-stage breakdown — GEMV is 83% of decode, fusion not worth it`
