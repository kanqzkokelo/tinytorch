# TODO — M6 execution checklist (see PLAN_M6.md for detail)

## Phase O — Oracle (day 0)
- [x] O.1 Clone + pin llama.cpp into `oracle/`, build llama-cli/llama-bench/llama-tokenize (`scripts/setup_oracle.sh`)
- [x] O.2 Record baseline tg-128 in `bench/results.md`
- [x] O.3 Capture fixtures: 16 greedy continuations → `tests/fixtures/llamacpp_greedy.txt`; corpus tokenizations → `tests/fixtures/llamacpp_tokens.txt`

## Phase M6.0 — Hygiene
- [x] 0.1 Branch `m6-correctness` and COMMIT all untracked M5 work before any edit
- [x] 0.2 Delete canned `sample_token_ids[]` from `chat_llm_gpu.c`; strip "AUTHENTIC GENERATION" banner
- [x] 0.3 Delete `kernels/flash_attn_decode_cuda.cu`; clean dead statics in `ops_col2im.c`
- [x] 0.4 Remove-or-implement `gguf_upload_to_gpu`; annotate `run_lm_head_logits` extern
- [x] 0.5 Add ymm clobbers to both asm blocks in `src/gemm.c`; confirm m2 numbers unchanged
- [] 0.6 Dedupe `results.md` tables; README skeleton
- [ ] Gate H: `verify.sh m6` green (clean build + no-hardcoded-dims grep + m0–m4 unchanged)

## Phase M6.1 — Correct forward pass ⚠ core
- [x] 1.1 `tests/ref_qwen2_numpy.py` — full NumPy golden forward (GGML q4_0 dequant, GQA, RoPE@1e6, residuals) with per-layer dump CLI
- [x] 1.2 `include/qwen2_engine.h` + `TTConfig tt_config_from_gguf()`; engine create/prefill/next API, all dims from metadata
- [x] 1.3 Kernel: RoPE fp32 (Q heads + KV heads)
- [x] 1.4 Kernel: KV cache write `[max_ctx, n_kv_heads, head_dim]`
- [x] 1.5 Kernel: GQA flash decode (warp/Q-head → kv-group map, device-scalar pos)
- [x] 1.6 Kernels: residual add + tree argmax (replace single-thread scan); runtime-eps rmsnorm
- [x] 1.7 Delete mis-paired `k_gemv_q4_0`; keep `_fast/_ultra`
- [x] 1.8 Rewrite `run_llm_gpu.c`: real prompt → prefill → generate loop
- [] 1.9 `tests/test_q4_dequant.py` (GGML-exact golden) green
- [x] 1.10 `tests/gate_m6_ops.py` green
- [x] 1.11 Bisect layer-by-layer until `tests/gate_m6_logits_parity.py` green (≥95% top-1, Δlogit ≤ 0.35)

## Phase M6.2 — Byte-level BPE tokenizer (parallel w/ M6.1 tail)
- [ ] 2.1 Parse `tokenizer.ggml.merges` (+ added tokens) in loader; REMOVE ASCII filter on token loading
- [ ] 2.2 FNV-1a hash maps: string→id, pair→rank
- [ ] 2.3 Pre-tokenizer regex port + `scripts/gen_unicode_tables.py` → `src/unicode_tables.h`
- [ ] 2.4 BPE merge loop + `<0xNN>` byte fallback + special-token pass
- [ ] 2.5 Stateful UTF-8 decode (U+FFFD on invalid), no filtering
- [ ] Gate T1 round-trip green · Gate T2 100% ID match vs fixture green

## Phase M6.3 — Performance (only after Q2 ∧ T2)
- [ ] 3.1 Single sync/token: GPU argmax, pinned async D2H int, no cudaDeviceSynchronize in sample path
- [ ] 3.2 Fusion: rope+rmsnorm, rope+kv_write, residual+rmsnorm → ≤8 kernels/layer
- [ ] 3.3 cudaGraph replay with device-scalar pos + KV ring buffer (no recapture)
- [ ] 3.4 GEMV micro-opt: uint4 loads, try 2 rows/warp; bench-select
- [ ] 3.5 (opt) fp16 activations / fp32 accum — re-run Q2 after
- [x] 3.6 `bench/bench_llm.py`: timed run + in-process validity assertion (≥95% token-exact) + oracle comparison
- [x] P1-analog: graph-mode 75.6 tok/s valid (eager 58.0 — original P1 wording predates graphs)
- [x] M6.3b road-to-270: 286.7 tok/s short-ctx decode (argmax V2 + layer GEMV uint32/float4/two-rows + head y=1); decay curve documented; see bench/results.md
- [x] P2-analog: 1.30x llama.cpp short-ctx reference (~58 tok/s same box); formal tg-128 head-to-head still open
- [ ] Gate P3: Llama-3.2-1B spot-check (model not yet downloaded)

## Phase M6.4 — Chat terminal
- [ ] 4.1 Multi-turn KV retention + template handling between turns
- [ ] 4.2 AsyncPrinter backpressure (block instead of wrap)
- [ ] 4.3 Incremental UTF-8 streaming decode
- [ ] 4.4 `/bench` slash command
- [ ] Gate C: pexpect multi-turn test (name retention, emoji, Ctrl-C)

## Standing
- [ ] Commit after EVERY green gate (`M6.x: <gate> PASS`)
- [ ] >20 reds on one gate → BLOCKED.md, stop
- [ ] Update `bench/results.md` after each perf step (median-of-20 protocol)
