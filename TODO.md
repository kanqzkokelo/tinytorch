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
- [x] Load-time root cause + arena fix: 3012×cudaMalloc=1.2s churn; arena+pinned
      staging = 5.7× startup (bench/bench_load_strategies.cu). Production wiring open:
- [ ] Wire single-arena loader path into src/loader_gguf.c production load.
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

## Open (in order)
- [ ] Two-token divergence bisection (~9 median) — agent running; do not
      duplicate work, check its result first.
- [ ] M8.4 Parity gate green: ./scripts/verify.sh m84
      (tests/gate_m84_gemma4.py) — blocked on two-token fix.
- [ ] Chat smoke E2B: TT_MODEL=<gemma-E2B gguf> ./chat (thinking-channel
      markers in stop strings).
- [ ] tg-128 benchmark vs llama.cpp; target ≥25–30 tok/s (Windows baseline 17).
- [ ] Final README grid update once m84 passes.

## Deferred
- E4B support: 5.15GB > 4GB VRAM — see BLOCKED.md.
- CUDA graph capture for gemma4 path (PLE host stage).
