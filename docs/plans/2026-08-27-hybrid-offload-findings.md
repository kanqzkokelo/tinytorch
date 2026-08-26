# Hybrid CPU+GPU offload — research findings (2026-08-26)

Source agent study of local oracle llama.cpp + web measurements.

## Placement granularity (llama.cpp evidence)
- Whole-layer `-ngl N`: contiguous top block, all-or-nothing (llama-model.cpp i_gpu_start).
- Tensor-class `-ot` regex → buft override. `--cpu-moe` = ffn_*_exps → CPU buffer
  (common.h:1113); composed per-block via `-ncmoe N`.
- Expert-granularity: unused by llama.cpp (experts packed whole-tensor).
- MoE class-level wins because experts compute IN PLACE on CPU: zero PCIe weight
  traffic. Measured: Qwen3.5-35B-A3B @ RTX 4070 --cpu-moe = 2.8× naive;
  late-layer-only -ot = 75.9 vs 5.3 t/s at 96K ctx (14×). Caveat: regresses ~10%
  on Vulkan/AMD (#24846).

## Dense model split: PCIe math (E-series dim=1536)
- Activation crossing = 1536×2B = 3 KB/boundary. Contiguous split = 2
  crossings/token → transfer <1µs; sync latency 20–50µs/event dominates.
- Interleaved per-layer alternation ≈ 2ms/token → ~500 t/s ceiling (sloppy).
- Binding constraint = memory bandwidth both sides, not PCIe.

## E4B-on-RTX3050-4GB verdict (dense, 5.15GB weights)
- GPU-resident share ~2.5–2.8GB @ 168GB/s → ~12–18 t/s realistic.
- CPU side ~2.5GB @ DDR4-3200 45GB/s, threaded q4_0 GEMV 55–70% eff → ~10–13 t/s.
- Sequential total ≈ **5–9 t/s** (pure-CPU floor ~5 t/s). Hybrid buys +30–70%,
  not magic. Full-GPU would be 15–20 t/s but doesn't fit.

## Engine design when implemented
```
placement table: {TOK_EMBD→CPU(mmap), ATTN_QKVO→GPU,
                  FFN blk<N→GPU, blk>=N→CPU, FINAL_NORM→CPU, LM_HEAD→GPU}
shuttle: h_fp16[1536] pinned buffer, cudaMemcpyAsync + sync at boundaries only
KV: allocated per-device beside its attention op — never migrates
```
Correctness rules: KV residency follows placement; residual-stream owner explicit
per boundary; reuse existing PLE host path as CPU-execution plumbing.
mmap CPU-resident tensors: zero RAM pressure cold; page faults amortize instantly
in decode (every weight touched every token).

## Recommendation
Implement placement table + single-boundary split first; measure before optimizing
shuttles. Priority BELOW E2B parity + M9 perf; revisit if E4B demand persists.
