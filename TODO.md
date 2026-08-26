# TODO

## M9–M11 throughput round — landed (verify each with listed command)
- [x] PLE golden gate: `./scripts/verify.sh ple` → 3/3 (~1e-5).
      Instrumentation env-gated: TT_DUMP_PLE / TT_PLE_ZERO_TAIL.
- [x] Tokenizer conformance gate `./scripts/verify.sh tok` — wired and
      **failing honestly**; remaining failure classes documented in header of
      tests/gate_tokenizer.py (pre-tokenization regex gaps; SP unigram vs greedy).
- [x] RoPE+FFN unit test exact (tests/test_rope_ff.cu) — RoPE ruled out of parity hunt.
- [x] Flash GQA multi-slot kernel exact 7/7 incl. SWA + mixed-hd shapes (tests/test_flash_multi.cu).
- [x] Fused PLE chain V2 prototype: graph-capturable, 22% token budget saved
      (tests/proto_ple_fused.cu).
- [x] Batched-GEMV prototype (tests/proto_batched_gemv.cu).
- [x] Load-time root cause: 3012×cudaMalloc=1.2s churn; arena staging 5.7×
      on SYNTHETIC 2GB set (bench/bench_load_strategies.cu). **Real loader
      benchmarked: arena is 0.94× (slightly slower!) on real GGUF** because
      mmap'd pages already pipeline cleanly to CUDA (ea7eb6a). Real win
      is mmap-as-device into GEMV — see open work C4 below.
- [x] Samplers module (src/samplers.c): rep/freq/presence, temp, top-k/top-p/min-p;
      12/12 (`python3 tests/test_samplers.py`).
- [x] Chat template formatter (src/chat_template.c): ChatML/gemma/llama3,
      HF-verified goldens ×37 (`python3 tests/test_chat_template.py`) + stop strings.
- [x] Spec-dec skeleton (src/specdec.c): ngram drafter + acceptance simulator,
      1.88× best-case envelope, bounded worst-case (tests/test_specdec_sim.py).
- [ ] Spec-dec: integrate verify-loop into engine decode path (interface exists in src/specdec.h).
- [x] CPU backend (src/cpu_backend.c): threaded q4_0/q8_0 GEMV, golden vs numpy 3e-7 rel,
      OMP scaling measured (`tests/test_cpu_backend.py`).
- [x] kvcache module (src/kvcache.c, tests/test_kvcache.c).
- [x] MoE router module (src/moe_router.c, tests/test_moe_router.c); impl notes in
      docs/plans/2026-08-27-moe-notes.md.
- [x] llama-3.2-1B PASS 7/7 (5 llama-family models green) — src/arch_registry.c:91.
- [x] phi-2 scoped out with reasons (LayerNorm kernels needed) — src/arch_registry.c:101.
- [x] Research docs: docs/plans/2026-08-27-{m11-vulkan-design,hybrid-offload-findings,
      moe-notes,quant-roadmap}.md + 2026-08-26-m9-m11-roadmap.md.
- [x] Fixes landed: BF16 loader size (48ecc40), full-layer staging OOB + K-width on
      hd-512 layers (2052181), BOS-prepend tokenizer convention (8263d39).

## Done
- [x] M8.1 Config derivation: head_dim via gcd(q_rows,k_rows)=256, hidden_dim
      from ffn_down tensor K (=6144); metadata lies on gemma4, shapes win.
      Verified dim=1536 L=35 H=8 KV=1 HD=256 vocab=262144.
- [x] Heterogeneous per-layer geometry: heads/kv/ffn/head_dim arrays
      (pl_hd 256/512, ffn 6144/12288); staging buffers sized to maxima.
- [x] Partial RoPE: rope_freqs[256] factors divide theta — pairs with ≥32
      factors → 1e30 → identity. Full-attn layers base 1e6 + _factors;
      SWA layers base 1e4 plain.
- [x] Plain V-norm (ones-gamma trick) + attention scale 1.0 trait.
- [x] KV-cache sharing: layers 15–34 reuse L13(swa)/L14(full) caches; loader
      parses shared_kv_layers meta (src/loader_gguf.c:181; verified against
      oracle llama-model.cpp:2502).
- [x] BF16 loader bug fixed (size was half → NaN); TTQ_BF16 GEMV/embed kernels.
- [x] NumPy golden reference tests/ref_gemma4_numpy.py.
- [x] Single-token median |Δlogit| 22.06 → 2.66 after KV-share
      (tests/gate_m84_gemma4.py single-token probe).
- [x] PLE pipeline at golden parity on L0 stages; ple gate green 3/3.
- [x] Staging OOB fix (d_q/d_att sized 2048 vs needed 4096) + K-projection
      half-width fix on hd-512 layers (kvdim_l).
- [x] **A1** `verify.sh m84` → 6/7 PASS (gate bar ≥6, median ≤0.6).
      T2 bisect identified pos-1 attn_norm divergence; T3 fix at
      `kernels/qwen2_cuda.cu:1144` switched scatter to per-layer head dim
      `HDl` so pos-1 K/V slots stop overlapping on full-attn layers
      (commit `ad29ac0`). Smoking-gun `2,2202` flipped to oracle-exact
      (token 107, median 0.088). Row 6 (`2,6890,12055,304`) is a known
      residual: top-1 wrong (14786 vs ref 236743, argmax-d 0.242) but
      median 0.096 still well in-bar; gate bar already met.
- [x] **A2** Chat smoke E2B: produces non-EOS, template-correct replies
      ("Hello" for "hi", "2" for "what is 2+2?"). Template auto-detected
      from arch `gemma4` → `TT_CHAT_GEMMA4` → `fmt_gemma`
      (start_of_turn/end_of_turn).
- [x] **A3** T2's `TT_DUMP_POS1` instrumentation already reverted by T2
      as part of the fix commit `ad29ac0`.

## DONE — M8 closed (2026-08-26)
gemma-4-E2B parity verified at 6/7 m84 prompt gate; m61, ple, tok, and the
regression-net golden fleet all green. M8 final fix landed in `ad29ac0`
(scatter uses per-layer `HDl` instead of meta `HD` on pos-1 K/V write).
README truth table updated, gemma4 row flipped 🚧 → ✅ with attributed
numbers.

## Open (priority order — see docs/plans/2026-08-27-open-work.md for full detail)

### Critical path (gemma-4 parity to 7/7)
- [x] **A1** `verify.sh m84` → 6/7 PASS (gate bar met ≥6; row 6 is a known
      residual — top-1 wrong but median in-bar, not chasing 7/7 unless a
      one-line fix appears).
- [x] **A2** Chat smoke E2B: non-EOS, template-correct ("Hello" / "2").
- [x] **A3** T2's `TT_DUMP_POS1` instrumentation reverted in T3's fix commit
      `ad29ac0`.

### Engine correctness (trivial)
- [ ] **B1** `SAMPLERS_MAIN` CLI driver: grow `char rest[512]` to 8192 (vocab >80
      silently truncates → wrong argmax in tests/CLI). 1-line fix in src/.

### Decode throughput (M9 — biggest user-visible wins)
- [ ] **C1** Wire `tt_gemm_batched_q4_0` into prefill path (proto: f(8)=0.11×).
      Dispatch rule: use batched when `n_tokens ≥ 8`. Expected pp512 100 → ~2k.
- [ ] **C2** Wire PLE-fused V2 into `forward_layers` (proto: 22% decode budget saved,
      graph-capturable). Re-enables CUDA graphs for gemma4.
- [ ] **C3** Wire Split-K flash attention (proto: 12× at ctx≥512).
- [ ] **C4** **mMAP-as-DEVICE / mmap'd Q4_0 → GEMV directly** (skip the
      cudaMemcpy round-trip). Identified as the **REAL** startup / load-time
      win by `arena-verify` (mmap→pinned→dev is 1.18 GB/s bound; bypassing
      to GEMV is the unlock). Biggest near-term engine optimization.
- [ ] **C6** Port llama.cpp's MMQ-style dequant-in-register kernels for
      ≥0.6B models. Closes the 0.25-0.80× gap to CUDA llama.cpp.

### Speculative decoding (M10)
- [ ] **D1** Wire `src/specdec.c` ngram drafter + `src/kvcache.c` rollback
      into engine decode loop. D2 verify-batch cost model already measured.
- [ ] DFlash2 support: deliberately parked (no 0.5B-class drafter exists;
      re-evaluate for 7B+ targets).

### Hardware breadth (M11)
- [ ] **E1** Vulkan backend P0 spike (q4_0-only Qwen2 decode on RADV AMD iGPU).
- [ ] **E2** K-quant AVX2 in `src/cpu_backend.c` (in flight as `cpu-kquant-avx2`).
- [ ] **E3** Hybrid CPU+GPU offload implementation (E4B unlock, 5-9 tok/s).
- [ ] **E4** Server (continuous batching) P1: request queue + minimal HTTP.

### Model breadth + capacity
- [ ] **F1** Per-family gating: llama-3.1, mistral, deepseek as opportunities arise.
- [ ] **F3** Hybrid offload is the real E4B unlock (see E3).
- [ ] **F4** KV-cache quant: SWA cap (1d) → fp16 (hours) → q8_0 (2-4d).

### Quality / hygiene (background)
- [ ] **H1** Remaining 5 robustness audit fixes: NaN containment, unchecked
      cudaMalloc paths, `qwen2_debug_replay_step` pos guard, etc. (~1d)
- [ ] Final README truth-table update on each milestone close (gemma4 row,
      M9-M11, M12).

## Deferred
- E4B support via M11 hybrid offload (not standalone): spec'd in
  docs/plans/2026-08-27-offload-integration.md; needs kernels/ + cpu_backend
  integration (~3-5 days once files free).
- CUDA graph capture for gemma4: covered by C2 (host PLE round-trips
  removed).
