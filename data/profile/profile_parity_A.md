# Profile Parity A — 2026-08-29 Manual Honest

## Engine TG64 breakdown (TT_PROFILE=1 eager, median 20 steps, RTX 3050 sm_86)
Prompt: 2,2202 (2 tokens) -> decode step measured

```
STEP_MS 4.406 (graph replay)
PROFILE TOTAL(med) 3.725 ms eager breakdown:
- embed 0.005 ms 0.1%
- qkv-gemv 0.551 ms 14.8%  (896x896 and 896x896 per layer)
- o+mlp-gemv 1.601 ms 43.0% (896x4864 + 4864x896 per layer)
- flash 0.406 ms 10.9%
- rmsnorm 0.237 ms 6.4%
- kv-scatter 0.069 ms 1.8%
- logits-gemv 0.790 ms 21.2% (151936x896)
TOTAL 3.725 ms => 268 tok/s eager, 227 tok/s graph (4.406 ms)
```

Comparison: llama-bench tg64 330 tok/s => 3.02 ms/token. Delta = 0.70-1.38 ms/token to close.

GEMV is 79% of time. Hidden GEMV micro honest:
- M=896 K=896 V4 0.008 ms 55 GB/s
- M=4864 K=896 V4 0.025 ms 99.7 GB/s
- LM head M=151936 K=896 V4 0.456 ms 168 GB/s

=> hidden FFN is 99 vs 168 gap. Fix hidden GEMV coalescing/block tuning to reach 130+ GB/s should save ~0.4 ms/token (1.601+0.551 -> ~1.2+0.4).

FA is only 10% at ctx~2, but at ctx128 it dominates (180 tok/s vs 272). Paged FA honest 0.207 ms/layer *24=4.99 ms would be worse than total graph at long ctx — explains 0.55x.

## Recommendation
Priority: fix hidden GEMV 896/4864 to 130+ GB/s first (saves 0.4-0.6 ms), then address long-ctx FA (maybe bypass paged at short ctx or tune S).
