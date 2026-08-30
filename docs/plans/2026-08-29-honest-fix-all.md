# Honest Fix All — Plan 2026-08-29

## Goal
Remove every fake/inflated benchmark number in repo, make every perf claim reproducible with single command, per-iter sync, median/p95, L2 vs DRAM labeled, cross-core where relevant.

## Lies Remaining (from grep 2026-08-29)

### P1 — README header
- `README.md:6` "outperforms llama.cpp ~4x" — measured 235 vs 174 = 1.35x on same CUDA box. Fix to "1.35x vs current llama.cpp CUDA (was 4.9x vs old pre-graph 58)" with honest tag.
- `README.md:17` claims 286.7 — honest median is 235. Fix.

### P2 — LOOP.md remaining cycles
- Cycle 9: already fixed to 0.207, but Phase9 box still says "26.7x" — now 15-18x (fix)
- Phase13: "0.026ms 2.5x vs Q4_0" — that was L2-hot 1.17MB. Honest Q2_K 0.137 ms vs Q4_0 needs re-measure Q4_0 at same M/K (8192/4096). Fix after re-measure.
- Cycle11: "15.29 seconds" clean build — honest is 3.82s. Fix.
- Phase 5 prefill 1288 tok/s @512 — verify with honest suite or mark as pre-honest.
- Cycle4/5 numbers need median tag.

### P3 — Historical docs (data/profile, bench)
- All `data/profile/*.md` pre-2026-08-29 are pre-honest. Add banner: "Pre-honest — numbers are single-sample/batch-mean, L2-hot, see AUDIT_HONEST_2026-08-29.md"
- `bench/results.md` final 75.6 tok/s is ancient ladder, keep but add banner.
- `bench/RESULTS_S0_vs_S4` keep.

### P4 — Microbenches still batch-mean
- `micro_v4.cu` (LM head 151936x896) 76 MB honest DRAM but still batch-mean 200 iters. Fix to per-iter 500 median like others.
- `micro_fa2_decode_q4.cu` and `micro_fa2_q8.cu` (FA decode) similar batch-mean. Fix.
- `micro_wmma_prefill_gemm.cu` batch-mean.
- `micro_gemv_peak.cu`, `tools/micro_batch4.cu` not headline but add L2 warning.
- For each: defaults must not truncate, must exceed L2 if claiming DRAM, per-iter sync, median/p95, L2 label.

### P5 — Engine truth vs llama.cpp oracle
- Need honest 5-run median for Qwen2.5-0.5B, Qwen3-0.6B, LLaMA-3.2-1B, SmolLM2 vs llama.cpp same model, same prompt, same -n, same host. Use bench/bench_llm.py with --runs 5 or custom loop, report median.
- Previous claim "4.9x" came from llama 58 tok/s baseline without graphs. New baseline with llama.cpp CUDA graphs is ~170-180 tok/s. Honest delta is ~1.3-1.4x, not 4.9x.

## Execution Order

1. Fix micro_v4, fa2_decode, fa2_q8, wmma to honest per-iter (parallel edits)
2. Re-measure: build time, GEMV Q4 vs Q2 honest speedup, end-to-end median fleet
3. Patch README, LOOP, add disclaimer banners to historical docs
4. Verify ci_local GREEN, verify.sh m61/m84/ple, test_engine_golden
5. Commit `fix(bench): honest remaining microbenches and doc truth`

## Acceptance

- Every `tok/s`, `GB/s`, `req/s` in README/LOOP has suffix "(median, 5-run, p50, per-iter sync, sm_86, cross-core where IPC)"
- No `Effective DRAM` when bytes < L2 — label L2
- No truncated K (K%256 assert)
- `AUDIT_HONEST` is single source of truth, linked from README
- `scripts/ci_local.sh` and `tools/run_live_performance_suite.py --help` both pass
