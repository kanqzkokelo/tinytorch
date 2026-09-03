# Prefill Roofline Baseline (2026-09-03, pp759)

- FP32: 1382 / 1420 / 1419 tok/s (median ~1419)
- WMMA FP16 (`TT_USE_WMMA_PRE=1`): 2274 / 2274 / 2266 (median ~2274, +60% over FP32)
- Stages (TT_PROFILE=1): qkv 27.8ms, o+mlp 337.0ms, flash 164.0ms, norm 3.1ms, TOTAL 531.9ms
- GEMM share: 68.6% (flash 30.8% — attention O(n²) grows at 759 tok vs 434)
- Target: INT8 WMMA ≥40% o+mlp cut → prefill ≥3800; then fusion + tiles → 8-11k
