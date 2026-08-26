# RESUME — fresh-session handoff

## Step 0: gate status awareness
Run `./scripts/verify.sh` targets before touching anything:
- `m61` — main parity fleet, green (7/7; chat-multiturn PASS).
- `ple` — PLE golden 3/3, green (~1e-5).
- `tok` — tokenizer conformance, **red by design right now**: documented gaps in
  header of tests/gate_tokenizer.py (simplified pre-tokenization regex, SP greedy
  vs unigram Viterbi). BOS handling fixed (8263d39) and is NOT a failure class.
- `m84` — gemma4 parity, NOT passing. This is the active work front.

## Active: gemma4-E2B parity hunt
State when last session ended:
- Single-token median |Δlogit| = 2.66 (tests/gate_m84_gemma4.py single-token probe).
- Two-token median ~9 → bisection was IN FLIGHT via agent.
  **PLACEHOLDER — block-reconcile agent outcome: [UNRESOLVED at handoff].**
  First action tomorrow: collect that result. Do not duplicate the bisection.
- Already ruled out: RoPE (tests/test_rope_ff.cu exact), flash GQA multi-slot
  kernel (tests/test_flash_multi.cu 7/7 exact), PLE L0 stages (golden).
- Fixed already: staging OOB + K-width (2052181), KV-share (6962cf5),
  BF16 loader size (48ecc40), BOS-prepend (8263d39).

## Exact next steps
1. Collect two-token bisection agent result (see placeholder above); fix root cause.
2. `./scripts/verify.sh m84` until ≥6/7 green.
3. `TT_MODEL=data/models/gemma-4-E2B-it-Q4_0.gguf ./chat` smoke.
4. tg-128 bench vs llama.cpp oracle; target ≥25–30 tok/s (Windows baseline 17).
5. README grid: flip gemma4 row only after m84 green.

## Open engineering items (beyond parity)
- Wire arena+pinned loader (5.7× startup proven in bench/bench_load_strategies.cu)
  into production src/loader_gguf.c load path.
- Spec-dec verify-loop integration into engine decode path (src/specdec.h interface).
- tok gate: close pre-tokenization regex gap or scope it out explicitly.

## Key file:line pointers
- shared_kv_layers meta parse: src/loader_gguf.c:181
- pl_src mapping + kv_shared cache skip: kernels/qwen2_cuda.cu forward_layers
  ~line 1025 (`const int kv_shared = e->has_pl_embd && e->pl_src[l] >= 0;`)
- Golden NumPy reference: tests/ref_gemma4_numpy.py
- Parity gate: tests/gate_m84_gemma4.py (verify.sh m84)
- Plan spec: docs/plans/2026-08-24-m8-gemma4-port.md
- Research: docs/plans/2026-08-27-*.{md} (vulkan-design, hybrid-offload-findings,
  moe-notes, quant-roadmap), docs/plans/2026-08-26-m9-m11-roadmap.md

## Key facts
- gemma4-E2B: dim=1536 L=35 H=8 KV=1 HD=256 vocab=262144 ffn=6144/12288(mixed)
- KV sharing: layers 15–34 reuse L13(swa)/L14(full) caches
- Model grid truth table lives in README.md; keep it honest — gemma4 stays
  🚧 IN PROGRESS until m84 passes.

## Blocked (do not attempt)
- E4B on 4GB VRAM GPU: 5.15GB weights — see BLOCKED.md; offload verdict
  5–9 t/s in docs/plans/2026-08-27-hybrid-offload-findings.md.
