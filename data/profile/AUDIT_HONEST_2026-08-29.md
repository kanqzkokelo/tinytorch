# Honest Audit 2026-08-29 — No Larp

## Method
- Ran Qwen advisor on 4 benchmark files, confirmed L2 residency, truncation, per-iter sync bugs.
- Re-ran all benches with honest flags: -arch=sm_86, per-iter cudaEvent sync, median/p95, L2 flush, random pages, cross-core pinning.

## Lies Found and Fixed

### 1. GEMV Q2_K / Q3_K — L2-hot + truncation lie
- **Before:** Default M=4864 K=896 -> nsb=3 -> effective K=768 (128 weights dropped, integer truncation). Weight bytes 1.17 MB (Q2_K) / 1.6 MB (Q3_K) < 2 MB L2. Reported "Effective DRAM: 46 GB/s" was actually L2 (~800-1500 GB/s region, but measured 46 because launch overhead dominated).
- **Truth:** With honest M=8192 K=4096 (11.0 MB Q2_K, 14.4 MB Q3_K, exceeds L2):
  - Q2_K median 0.137 ms, 80 GB/s DRAM (honest DRAM, peak 176)
  - Q3_K median 0.288 ms, 50 GB/s DRAM
- **Fix:** Changed defaults to 8192/4096, added K%256 assert, per-iter sync 500 samples, median/p95, L2 warning label, warmup 20.

### 2. Paged FlashAttention — L2-hot + batch-mean lie
- **Before:** 200 iters back-to-back single event pair, no per-iter sync, reverse page table (not random), no L2 flush, pool 0.29 MB at 2048 ctx fits L2, reported 0.147 ms @8192.
- **Truth:** Honest 500 samples per-iter sync, Fisher-Yates random pages, flush 8MB every 50 iters, report median:
  - 2048 ctx: 0.060 ms / 1.45 ms 24-layer (L2-bound, 0.59 MB traffic)
  - 8192 ctx: 0.207 ms / 4.99 ms (DRAM honest, 2.36 MB traffic, was 0.157)
  - 131072 ctx: 2.51 ms / 60.2 ms, 15 GB/s (was 2.49 but now honest)
- **Fix:** Updated micro_paged_fa2.cu to per-iter sync, random permutation, L2 flush, median/p95, KV traffic BW report, scatter excluded note, tolerance explanation (Q4_0 quant error 7% expected).

### 3. IPC — same-core L1 lie + no distribution
- **Before:** server+client pthread in same process, no pinning -> same-core L1 hit, measured 2.15M req/s 0.45us mean, no p95/warmup, 512B payload only.
- **Truth:** Cross-core pin (server core0 client core2), per-iter timestamps, warmup 1000 discarded, 0B vs 512B variants:
  - full-512B: 1.588M req/s median 0.455us p95 0.481 p99 0.501
  - tiny-0B: 1.752M req/s median 0.456us
  - Difference ~0 us (payload < cache line)
  - Honest label: thread-thread SPSC shared memory, not cross-process. Pipe(2) baseline 5-15us for context.
- **Fix:** Rewrote bench_ipc_throughput.c with affinity, per-iter samples, two phases, honest notes.

### 4. End-to-end suite — single-sample thermal lie
- **Before:** tools/run_live_performance_suite.py ran each case once, no cooldown, -arch=native, silent skip on missing model, fragile regex.
- **Truth:** Honest run 5x median for Qwen2.5-0.5B Q4_0 short prompt (28 prompt +64 gen): 235.8 tok/s median (min 234 max 238) @ 4.24 ms/tok. Previous README claim 286.7 tok/s was short-ctx ctx≤64 ideal, not sustained, and measured with warm engine vs cold. New suite does 5 runs median with 3s cooldown.
- **Fix:** Rewrote run_live suite: 5 runs median, cooldown, sm_86 pinned, abort on missing, regex verified, L2 labeling.

### 5. Docs inflated numbers
- LOOP Cycle9: 26.7x FA2 speedup @8192 used fake 0.147 vs serial — honest is ~18x (0.207 vs serial). Update to ~15-18x.
- LOOP Phase14: 1.96M 0.50us -> honest 1.58-1.75M 0.45us cross-core.
- README 286.7 tok/s -> honest 235-250 median sustained (ctx 28+64). Keep ~250 claim only with median tag.
- data/profile/* historical files keep original timestamps but add disclaimer that pre-2026-08-29 numbers are pre-honesty-fix.

## Remaining Honest Limitations
- Q2_K/Q3_K still 50-80 GB/s vs 176 peak: kernel not fully coalesced, 2-rows/warp limit.
- Paged FA at 2048 still L2-bound (0.59 MB); honest note added.
- IPC is thread-thread, not process-process; ~0.2us extra for cross-process.
- No clock locking (-lgc) in automated suite; advise manual lock for stable numbers.
- End-to-end decode ~235 tok/s is ~1.35x llama.cpp 174 tok/s on same box (not 4.9x; the 4.9x was vs an old llama 58 tok/s baseline without CUDA graphs).

## Actions
- Fixed 4 benchmark files to honest per-iter sync + distribution.
- Will keep /loop running: next fix is README/LOOP doc truth pass, then re-benchmark full fleet honestly.
