# RESUME — fresh-session handoff

## Step 0: gate status awareness
Run `./scripts/verify.sh` targets before touching anything:
- `m61` — main parity fleet, green (7/7; chat-multiturn PASS).
- `m84` — gemma4 parity, **6/7 PASS** (gate bar ≥6 met; row 6 is a known
  residual with median in-bar).
- `ple` — PLE golden 3/3, green (~1e-5).
- `tok` — tokenizer conformance, **red by design right now**: documented gaps
  in header of `tests/gate_tokenizer.py` (simplified pre-tokenization regex,
  SP greedy vs unigram Viterbi). BOS handling fixed (8263d39) and is NOT a
  failure class.

## M8 closeout (2026-08-26)
gemma-4-E2B parity at 6/7 on the m84 prompt gate. T2's bisect pinned
pos-1 attn_norm divergence to the K/V scatter inside the per-layer attention
write path; the actual K/V tensors were being written with the meta-level
head dim stride, which overlapped positions on full-attn (hd=512) layers
where per-layer head dim differs from the meta value. T3's fix at
`kernels/qwen2_cuda.cu:1144` switched the scatter to use the per-layer
head dim `HDl` (which can be 256 or 512 on gemma-4's heterogeneous layers)
instead of meta `HD`, stopping the slot overlap. Result: the smoking-gun
prompt `2,2202` flipped to oracle-exact (token 107, median 0.088, was the
prompt T2 originally identified as a cos=0.19 divergence). T2's
`TT_DUMP_POS1` instrumentation was reverted in the same fix commit
(`ad29ac0`). Chat smoke (`TT_MODEL=data/models/gemma-4-E2B-it-Q4_0.gguf
./build/chat_llm_gpu`) produces non-EOS, template-correct replies — "Hello"
for "hi" and "2" for "what is 2+2?" — with the gemma template auto-detected
from the `gemma4` arch (`TT_CHAT_GEMMA4` → `fmt_gemma`).

### m84 gate table (post-fix, 6/7)

| # | prompt | top-1 | argmax-d | median \|Δ\| | result |
|---|---|---|---|---|---|
| 0 | `2,2202` | ✓ | 0.001 | 0.088 | PASS |
| 1 | `2,9302,1110` | ✓ | 0.066 | 0.126 | PASS |
| 2 | `2,5103,7841` | ✓ | 0.081 | 0.092 | PASS |
| 3 | `2,1110,2202,9302` | ✓ | 0.060 | 0.094 | PASS |
| 4 | `2,15003,402` | ✓ | 0.051 | 0.120 | PASS |
| 5 | `2,6890,12055,304` | ✗ (14786 vs 236743) | 0.242 | 0.096 | **FAIL** (known residual) |
| 6 | `2,3305,9980` | ✓ | 0.069 | 0.068 | PASS |

Row 6 is the only residual: top-1 wrong (argmax-d 0.242) but median 0.096
still well under the 0.6 bar. Not chasing 7/7 — gate bar met; any further
work is a separate hunt and should not be conflated with the M8 fix.

## Active front (M9 — decode throughput)
- [ ] **C1** Wire `tt_gemm_batched_q4_0` into prefill (proto: f(8)=0.11×).
- [ ] **C2** Wire PLE-fused V2 into `forward_layers` (proto: 22% decode
      budget saved, graph-capturable). Also re-enables CUDA graphs for
      gemma4.
- [ ] **C3** Wire Split-K flash attention (proto: 12× at ctx≥512).
- [ ] **C4** **mMAP-as-DEVICE / mmap'd Q4_0 → GEMV directly** (skip the
      cudaMemcpy round-trip). Real startup / load-time win.
- [ ] **C6** Port llama.cpp's MMQ-style dequant-in-register kernels for
      ≥0.6B models.
- [ ] **D1** Spec-dec ngram drafter + KV rollback into engine decode loop.
- [ ] **B1** `SAMPLERS_MAIN` CLI driver: grow `char rest[512]` to 8192 (1-line).

## Open engineering items (beyond M8/M9)
- Wire arena+pinned loader (5.7× startup proven in bench/bench_load_strategies.cu)
  into production src/loader_gguf.c load path.
- tok gate: close pre-tokenization regex gap or scope it out explicitly.
- E4B on 4GB VRAM GPU: see BLOCKED.md; offload verdict 5–9 t/s in
  docs/plans/2026-08-27-hybrid-offload-findings.md.

## Key file:line pointers
- M8 final fix site: `kernels/qwen2_cuda.cu:1144` (scatter uses `HDl` per-layer
  head dim, not meta `HD`).
- shared_kv_layers meta parse: `src/loader_gguf.c:181`
- pl_src mapping + kv_shared cache skip: `kernels/qwen2_cuda.cu forward_layers`
  ~line 1025 (`const int kv_shared = e->has_pl_embd && e->pl_src[l] >= 0;`)
- Golden NumPy reference: `tests/ref_gemma4_numpy.py`
- Parity gate: `tests/gate_m84_gemma4.py` (`./scripts/verify.sh m84`)
- Plan spec: `docs/plans/2026-08-24-m8-gemma4-port.md` and
  `docs/plans/2026-08-26-m84-7of7-parity.md`
- Research: `docs/plans/2026-08-27-*.{md}` (vulkan-design,
  hybrid-offload-findings, moe-notes, quant-roadmap),
  `docs/plans/2026-08-26-m9-m11-roadmap.md`

## Key facts
- gemma-4-E2B: dim=1536 L=35 H=8 KV=1 HD=256 vocab=262144 ffn=6144/12288(mixed)
- KV sharing: layers 15–34 reuse L13(swa)/L14(full) caches
- Model grid truth table lives in `README.md`; gemma-4 row now ✅ 6/7 m84 gate
  (median ≤0.6).

## Blocked (do not attempt)
- E4B on 4GB VRAM GPU: 5.15GB weights — see BLOCKED.md; offload verdict
  5–9 t/s in docs/plans/2026-08-27-hybrid-offload-findings.md.
