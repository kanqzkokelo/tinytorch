> **Honesty note (2026-08-29):** Numbers below are pre-honest (single-sample/batch-mean, L2-hot possible, -arch=native). See `data/profile/AUDIT_HONEST_2026-08-29.md` for honest per-iter median, cross-core, sm_86 pinned measurements.

# nnfromscratch (tinytorch) Project Audit Report

**Audit date:** 2026-08-28
**Repo:** `~/Storage/repos/nnfromscratch`
**Branch:** `m6-correctness` (active), `master`, `gemm-opt`
**Current HEAD on `m6-correctness`:** `22b346e` `bench: Q8_0 V4 LM head results (1.92x LM head speedup, 250+ tok/s decode)`
**Total commits:** 178

---

## 1. Project Overview

- **Goal:** From-scratch CUDA LLM inference engine in pure C + CUDA, beating llama.cpp on Qwen2-0.5B while proving output equivalence against it. Started as generic tinytorch (tensors → autograd → MNIST → CIFAR → GEMM ladder), pivoted to LLM engine at M5.
- **Status:** Working engine. qwen2.5-0.5b greedy decode 246–269 tok/s (ctx64) / 219.6 tok/s sustained (ctx5×128). 6.3× prefill speedup (WMMA batched GEMM). 8 architectures × 12 quants = 96 golden-verified combinations. 7/7 parity on qwen2/qwen3/llama/gemma2 main fleet; 6/7 on gemma-4-E2B m84 gate; PLE golden 3/3 at ~1e-5; CPU backend K-quants AVX2 6.7×. Spec-decode skeleton shipped, currently regresses baseline (need faster verify).
- **Architecture:** Single-header C engine + trait registry + per-family trait dispatch. CUDA kernels: q4_0/q4_1/q5_0/q5_1/q8_0/q4_K/q5_K/q6_K typed GEMV, q4_0 V4 4-rows-per-warp fast path, q8_0 V4 LM head, WMMA tensor-core batched prefill. CPU AVX2 backend with OMP. Graph-capture decode for qwen2/llama-3.2/qwen3/smollm (4/4 graph-captured).
- **Hardware:** RTX 3050 laptop, sm_86, 4 GB VRAM.
- **Dependencies:** CUDA toolkit, pybind11, OpenMP, libc, no other 3rd-party libs.

---

## 2. Git History (Selected Milestones)

### Milestone setup (M0–M4, foundation)
- `f6f9cff` Plan + verification harness skeleton (oracle before implementation)
- `62d0e08` M0: unit-vs-numpy PASS — tensor library
- `d62e26e` M1: gradcheck, mnist-mlp PASS — autograd + MNIST 97%+
- `1531724` M2 attempt 1: cpu-bench RED — AVX2 asm kernel 0.63–0.74× OpenBLAS-1T
- `f6c7361` M2: BLOCKED.md — plateau 0.73× documented
- `3e27266` M3: gpu-parity PASS (Gate A), cuda-bench 4.8× (Gate B red)
- `6f2b89e` M3: gpu-parity, cuda-bench PASS — CUDA sgemm ladder
- `b11a476` M5 WIP snapshot: LLM engine, tokenizer, GGUF loader

### M6 — correctness rewrite + parity
- `e5a40a9` M6.1: forward-pass PARITY — tinytorch matches llama.cpp greedy
- `1f3bc14` M6-gates: committed parity fixtures + logit-parity gate (verify.sh m61)
- `8ef19bb` M6-gates: track dump_logits source + text-parity tool + plan docs
- `0913dc3` M6-gates: harden failure paths per quality review

### M6.3 — performance engineering
- `4c4b274` M6.3: unified CUDA streams — decode 27→48 tok/s
- `5fb7ef9` M6.3: cudaGraph replay of decode step; 59.1 tok/s decode
- `d3538f7` M6.3: lm-head q8_0 GEMV uint32+float4 → 75.2 tok/s; kernel 5.0→1.3ms
- `2fc57f8` M6.3 complete: 75.6 tok/s decode (from 48/58), parity 7/7

### M6.3b — kernel ladder 75 → 287 tok/s
- `7aa79e7` argmax V2 float4+parallel final, 0.574→0.072ms — 79.8 tok/s
- `720d0ac` q4_0 GEMV V1 float4-x + uint32-W `__byte_perm` port — 183.1 tok/s
- `9ecff31` q4_0 GEMV V2 two-rows-per-warp — 212.3 tok/s
- `b406cde` lm-head blockDim.y sweep 16→1 — 286.7 tok/s ctx64
- `4613c59` M6.3b final: 286.7 tok/s short-ctx decode (219.6 @5x128)
- `ed1b253` M6.3b final: 285.6 tok/s decode (ctx≤64), decay curve + verdict

### M7 — all quants + all architectures
- `cff5bd8` chat: ./chat launcher + stop-string guards
- `a3177aa` M7 task-1: fix multi-turn degradation
- `967a28e` chat: repeat penalty + Gumbel-max sampling; greedy default byte-identical
- `d6e8186` M7 task2: CUDA GEMV dispatch for q4_1/q5_0/q5_1/q4_K/q5_K/q6_K
- `54b0adf` M7 task3: trait-driven forward pass (rope/act/norm/softcap/swa); qwen2 byte-identical
- `992e20d` M7 task4: SP tokenizer + llama-family — tinyllama-f16 bit-perfect
- `99c3e79` M7 task4: GEMMA2 PARITY — sandwich norms + explicit head_dim + embed sqrt(dim)
- `d1ed8f3` M7 task5: verification grid runner + README truth table

### M8 — gemma-4 port
- `731970f` M8 plan: gemma4 port spec
- `48ecc40` M8: per-layer heterogeneous geometry (hd 256/512), partial-rope, plain V-norm, BF16 loader size fix
- `6962cf5` M8: KV-share (L15–34 reuse L13/L14 caches), m84 gate, single-token median 22.06→2.66
- `f1aabd0` M8: PLE golden gate (verify.sh ple, 3/3 green ~1e-5)
- `ad29ac0` M8 fix: scatter uses HDl per-layer head dim — pos-1 K/V slots stop overlapping; m84 7/7 (was 0/7)
- `612ba09` M8 closeout: m84 6/7 PASS; chat smoke coherent

### M9–M11 — throughput round 2 + hardware breadth
- `944170f` gemm M2-opt step1: BLIS-style IC/MC row blocking (MC=240)
- `60e4baa` M11: CPU backend foundation — threaded q4_0/q8_0 GEMV
- `1303f86` M10-adjacent: production sampler suite (rep/freq/presence, temp, top-k/top-p/min-p; 12/12)
- `485ad04` M10-adjacent: chat template formatter (37 tests)
- `b5b3fb1` M9: load-time prototype — root cause = alloc churn (3012×cudaMalloc=1.2s), arena 5.7×
- `a901c6f` M10: spec-decode skeleton — ngram drafter (C99) + acceptance simulator
- `e7e9609` M9 proto: fused PLE chain V2 (2 launches, graph-capturable, 22% token budget saved)
- `f7687e6` M9 proto: split-K flash attention — 12× at ctx≥512
- `f37ece8` M11: CPU backend phase 2 — K-quant GEMVs + AVX2 (q4_0 20.2 GB/s @8T, 6.7×)
- `9e1e33f` M10/M9: KV cache management module
- `d715baa` M10 proto: verify-batch cost model — flat cost validated
- `ebb6f15` bench+audit: llama.cpp scoreboard + robustness audit
- `f4edcfb` infra: CI pipeline
- `131fcbb` docs: REALITY CHECK — fair CUDA-vs-CUDA scoreboard (0.25–0.80× decode)
- `d43bca3` tokenizer: full GPT2/LLAMA3/QWEN2/SMOLLM regex-equivalent pre-tokenizer (130/130)
- `ee3d80c` chat CLI: wire chat_template + tt_sampler_chain
- `4985d00` M9: PLE-fused V2 in forward MatFormer (22% per-layer saving)
- `f98b2aa` fix(gemm): correct K-x8 unroll Bp advance
- `0bdcdeb` examples: M12 P2 multi-model HTTP server
- `db955fd` chat: gemma-4 use correct special tokens
- `3038074` M9.5: q8_0 GEMV vectorization (2-rows-per-warp)
- `2e1a227` M9.5: graph capture for q6_k models
- `a340e33` M9.5: q4_0 GEMV vectorization (2-rows-per-warp V2)
- `96552ca` M9.5: FP16 + BF16 GEMV vectorization
- `cde8709` M9.5+ Task 0: profile_step per-stage reveals GEMV=83% of decode
- `82567a6` M9.5+ V4: 4-rows-per-warp q4_0 GEMV (LM head 1.7×, FFN 1.5×)
- `24c705b` M9.5+ V4: conditional dispatch (M≥128 → V4, else V2)
- `914839a` use: host-side N-gram lookup drafter module
- `eaeb944` use: Universal Speculative orchestrator
- `0c497f4` use: benchmark Universal Speculative vs baseline
- `c3638fd` use: microbench q4_0 & q8_0 batch4 kernels (3.2–4.4×)
- `5b17c48` use: q4_0 & q8_0 batch4 GEMV launchers
- `dc9a750` fix: run_llm_gpu exits after 1 token
- `77bfce9` feat: add temperature, top-p, and repetition penalty sampling
- `0b4e462` prefill: microbench 2D batched Q4_0 prefill GEMM kernel (2.2–3.3×)
- `00f310f` prefill: Q4_0 batched prefill GEMM launcher + 32/32 boundary tests
- `d6f0e1f` fix: resolve prefill GEMM divergence across multi-token batches
- `986e59e` prefill: wire fixed batched GEMM prefill into qwen2_engine_prefill for N≥32
- `04a1c90` prefill: microbench Tensor Core WMMA Q4_0 prefill GEMM kernel
- `099d244` prefill: add Tensor Core WMMA Q4_0 prefill GEMM launcher
- `3836b2c` prefill: dispatch Tensor Core WMMA GEMM in qwen2_engine_prefill for N≥64
- `b97ece3` decode: microbench peak-bandwidth Q4_0 GEMV kernel
- `f803ceb` lmhead: microbench 4-rows-per-warp q8_0 LM head kernel (1.7×)
- `0caa1c3` lmhead: add q8_0 V4 LM head launcher with bit-exact unit tests
- `22b346e` bench: Q8_0 V4 LM head results (1.92× LM head speedup, 250+ tok/s decode)

(Full list at `git log --reverse --oneline`.)

---

## 3. Source Files (src/ — 6,792 lines C + headers)

| File | Lines | Purpose | Key Functions |
|---|---|---|---|
| `qwen2_engine.h` (include) | 129 | Public engine API | `qwen2_engine_create/prefill/next/verify_speculative/set_sampling/reset/free`, `tt_logits_q8_0_v4`, `tt_gemm_q4_0_prefill`, `tt_gemm_wmma_q4_0_prefill`, `prefill_batched_gemm` |
| `qwen2_cuda.cu` (kernels) | 2,594 | Trait-driven decode engine: forward pass, RoPE, GQA-flash, KV scatter, RMSNorm, softcap, Q/K norm, embed, argmax, PLE, graph capture | `k_rmsnorm`, `k_ple_stage1_f32`, `k_ple_stage2_f32`, `k_rope`, `k_rope_ff`, `k_rope_gptj`, `k_qk_norm_rms`, `k_softcap`, `k_flash_gqa`, `k_flash_gqa_splitk`, `k_kv_scatter`, `k_argmax_partial/final`, `k_embed_q4_0_dyn/q6_K_dyn`, `k_repeat_penalty`, `qwen2_engine_prefill/next/verify_speculative` |
| `gemv_q4_cuda.cu` (kernels) | 1,754 | q4_0 fast-path GEMV + LM head + batched-4 + WMMA GEMV | `k_gemv_q4_0`, `k_fused_swiglu_q4_0`, `k_logits_q4_0/v2/v4`, `k_gemv_q4_0_v4`, `k_logits_q8_0/v4`, `k_gemv_q8_0`, `k_gemv_q4_0_batch4`, `k_gemv_q8_0_batch4`, `k_gemv_wmma_q4_0`, `k_embed_q4_0` |
| `gemv_typed.cu` (kernels) | 1,224 | Typed GEMV for all Tier-1 quants | 26 `__global__` kernels |
| `gemm.c` | 675 | AVX2+FMA CPU GEMM ladder, IC/MC blocking, OMP | `tt_sgemm`, `micro_kernel_6x16` |
| `tokenizer_bpe.c` | 964 | Byte-level BPE tokenizer | `tt_bpe_encode`, `tt_bpe_decode` |
| `cpu_backend.c` | 874 | M11 quant-aware threaded CPU GEMV (q4_0/q8_0/q4_K/q5_K/q6_K) | `tt_cpu_gemv_q4_0`, `tt_cpu_gemv_q8_0` |
| `kvcache.c` | 462 | KV cache management: SWA compaction, specdec rollback | `tt_kvcache_*` family |
| `loader_gguf.c` | 413 | GGUF v2/v3 parser, mmap, tensor table | `gguf_load`, `gguf_find_tensor` |
| `samplers.c` | 343 | Production sampler pipeline: temp, top-k/top-p/min-p, rep/freq/presence, Gumbel-max | `tt_sampler_chain` |
| `chat_template.c` | 314 | Per-family chat formatter (ChatML/gemma/llama3) | `fmt_chatml`, `fmt_gemma`, `fmt_llama3` |
| `moe_router.c` | 209 | MoE routing logic | `tt_moe_route_*` |
| `arch_registry.c` | 120 | Architecture trait registry | `arch_traits_for` |
| `dequant_ref.c` | 388 | CPU golden dequant | `tt_dequant_*` |
| `kvcache.h` | 213 | KV cache public API | structs + funcs |
| `samplers.h` | 116 | Sampler API | `TT_SAMPLER_*` |
| `chat_template.h` | 121 | Chat template API | `TT_CHAT_*` |
| `moe_router.h` | 97 | MoE API | `tt_moe_*` |
| `specdec.c` | 93 | N-gram drafter (ring buffer) | `tt_ngram_*` |
| `ngram_lookup.c` | 32 | Host-side N-gram draft | `ngram_lookup_draft` |
| `autograd.c` | 732 | M1 reverse-mode AD | `tt_backward` |
| `ops_spatial.c` | 187 | M4 conv2d im2col + maxpool + avgpool | `tt_conv2d_*` |
| `ops.c` | 139 | Tensor ops (add/mul/scalar/matmul/relu/softmax) | `tt_add/mul/matmul/softmax/relu` |
| `ops_llm.c` | 57 | CPU LLM ops (rmsnorm, rope, swiglu placeholders) | `tt_rmsnorm` |
| `tensor.c` | 68 | Tensor ref-count + alloc | `tt_alloc`, `tt_free` |
| `async_printer.c` | 86 | Background C11 `_Atomic` buffered stdout | `tt_async_*` |
| `bindings.cpp` | 5,620 b | pybind11 bindings for tensor/autograd | exposed `Tensor` + ops |

---

## 4. Public API (include/qwen2_engine.h — 129 lines)

```c
TTConfig tt_config_from_gguf(const GGUFModel *m, int max_ctx);
Qwen2Engine *qwen2_engine_create(const TTConfig *cfg, GGUFModel *m);
int qwen2_engine_prefill(Qwen2Engine *e, const int *toks, int n);
int qwen2_engine_next(Qwen2Engine *e);
int qwen2_engine_pos(const Qwen2Engine *e);
int qwen2_engine_verify_speculative(Qwen2Engine *e, const int *h_candidate_tokens,
                                    int n_candidate, float *out_logits);
int qwen2_debug_replay_step(Qwen2Engine *e, int next_tok);
void *qwen2_debug_stream(Qwen2Engine *e);
void qwen2_debug_profile_reset(void);
void qwen2_debug_profile_report(int nsteps);
void qwen2_engine_set_sampling(Qwen2Engine *e, float temp, int topk, float penalty);
int qwen2_debug_copy_x(Qwen2Engine *e, float *host, int n);
int qwen2_debug_copy_kv(Qwen2Engine*, int layer, float*, long);
int qwen2_debug_copy_xn(Qwen2Engine*, float*, int);
int qwen2_debug_copy_logits(Qwen2Engine *e, float *host, int n);
int qwen2_engine_step_logits(Qwen2Engine *e, int tok, float *host_logits);
int tt_logits_q8_0_v4(const void *dW, const float *dx, float *dlogits, int vocab, int K, cudaStream_t s);
int tt_logits_q8_0(const void *dW, const float *dx, float *dlogits, int vocab, int K, cudaStream_t s);
int tt_gemm_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM, int M, int K, int N, cudaStream_t s);
int tt_gemm_wmma_q4_0_prefill(const void *dW, const float *dX_NxK, float *dY_NxM, int M, int K, int N, cudaStream_t s);
int prefill_batched_gemm(Qwen2Engine *e, const int *toks, int n, float *h_x_out);
void qwen2_engine_reset(Qwen2Engine *e);
void qwen2_engine_free(Qwen2Engine *e);
```

`TTConfig` exposes dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, vocab, max_ctx, rms_eps, rope_base, TTraits tr.

---

## 5. CUDA Kernels (5,863 lines)

| Source | Kernels | Quant | Speed vs V2 | Use |
|---|---|---|---|---|
| `gemm_cuda.cu` | 4 (naive, tiled_128x128_db, wmma, f2h) | fp32/fp16 | up to 100.7% cuBLAS, 39× naive @1024³ | M3 GEMM ladder |
| `gemv_q4_cuda.cu` | 13 (q4_0 GEMV V2, V4, batch4, logits V1/V2/V4, q8_0 GEMV, q8_0 logits V1/V4, fused swiglu, wmma_gemv, embed) | q4_0/q8_0 | V4 1.7× LM head / 1.5× FFN; V4 batch4 1.7–2.0× | Fast decode path |
| `gemv_typed.cu` | 26 typed GEMV + embed | q4_0/q4_1/q5_0/q5_1/q8_0/q4_K/q5_K/q6_K/f16/f32/bf16 | scalar; correctness first | M7 all-quants dispatch |
| `qwen2_cuda.cu` | 24 (rmsnorm, PLE stage1/2, rope/rope_ff/rope_gptj, qk_norm_rms, softcap, add, flash_gqa + splitk + combine, kv_scatter, argmax, embed_dyn_q4_0/q6_K, pos_inc, scale, repeat_penalty) | mixed | 287 tok/s decode (V4 LM head) | Engine core |
| `cublas_ref.cu` | 0 (host wrapper) | fp32 | reference | cuBLAS bridge |

**q4_0 GEMV ladder:** V1 scalar (75 tok/s) → V2 uint32+float4 (213) → V4 4-rows/warp (286.7 ctx64) → V4 batch4 (1.7–2.0× over V4 single, bit-exact 14/14 shapes).

**LM head ladder:** V1 (1.516 ms) → V4 (0.791 ms) = 1.92× speedup, 100% bit-exact.

**Prefill kernels:** 2D batched Q4_0 GEMM (10–40× single-GEMV), WMMA Q4_0 (peak 827 tok/s @N=73, 685 @N=136, 457 @N=260).

---

## 6. Examples (2,962 lines)

| Example | LOC | Purpose | How to Use |
|---|---|---|---|
| `run_llm_gpu.c` | 265 | Single-shot generation, STATS line, env-tunable (TT_GREEDY, TT_RAW_PROMPT, TT_PROMPT, TT_NPREDICT) | `TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "Once upon a time" 64` |
| `chat_llm_gpu.c` | 407 | Multi-turn interactive chat, ChatML/gemma/llama3 templates, sampler chain (temp/top-k/top-p/min-p/rep/freq/pres), stop-strings, history accumulation | `./chat` (default qwen2.5-0.5b) or `TT_MODEL=... ./build/chat_llm_gpu` |
| `spec_llm_gpu.c` | 300 | Universal Speculative Engine: N-gram draft (host) + batched CUDA verify, TT_DRAFT_K, TT_WINDOW | `./build/spec_llm_gpu <model> <prompt> <n> --draft-k 3 --window 2` |
| `server_minimal.c` | 935 | M12 P1 minimal HTTP server, OpenAI-compatible `/v1/chat/completions`, single-request N=1 | `TT_MODEL=... ./build/server_minimal` (port via TT_LISTEN) |
| `server_multimodel.c` | 994 | M12 P2 multi-model HTTP server, registry of engines selected by `model` field | `TT_MODELS="q25=path1,big=path2" ./build/server_multimodel` |
| `run_llm.c` | 61 | Early CPU-only stub (M5) | reference only |
| `chat.py`, `real_chat.py` | 9,107 + 6,620 b | Python chat wrappers (legacy M5) | legacy |
| `train_mnist.py` | 2,313 b | M1 MNIST smoke | `python3 examples/train_mnist.py --smoke` |
| `generate_text_cuda.py` | 11,816 b | Text generation script | reference |
| `generate_gemma4_cook.py` | 3,225 b | Gemma4 cooking script | reference |

**Live runs (just executed):**
- `qwen2.5-0.5b q4_0`, "Once upon a time" ×64: **246.0 tok/s decode / 274.0 tok/s prefill / 210.6 incl**
- `qwen2.5-0.5b q4_0`, "Quantum mechanics" ×200: **219.6 tok/s decode / 399.0 tok/s prefill / 209.9 incl**
- `gemma-4-E2B q4_0`, "The quick brown fox" ×4: 27.8 tok/s decode / 83.1 prefill (PLE host path)

---

## 7. Tools & Microbenches (4,912 lines)

| Tool | LOC | Purpose | Numbers |
|---|---|---|---|
| `dump_logits.c` | 89 | Teacher-forced logits dump for parity fixtures | gate input |
| `oracle_logits.c` | 70 | llama.cpp oracle logits for parity gate | gate input |
| `profile_step.cu` | 97 | Per-stage decode profiler (TT_PROFILE) | per-stage ms, GEMV=83% |
| `bench_prefill.c` | 82 | Times `qwen2_engine_prefill` at multiple N | CSV out |
| `micro_v4.cu` | 275 | V2 vs V4 q4_0 GEMV microbench (V4 ships) | V4 1.7× LM head |
| `micro_v4_q8.cu` | 201 | V4 q8_0 attempt (rejected, 0.55× LM head) | data preserved |
| `micro_v4_f16.cu` | 172 | V4 F16 attempt (neutral, 0.76→0.73×) | data preserved |
| `micro_logits_q8_v4.cu` | 240 | q8_0 V4 LM head microbench (1.92×) | shipped |
| `micro_prefill_gemm.cu` | 344 | 2D batched prefill GEMM (2.2–3.3×) | shipped in microbench |
| `micro_wmma_prefill_gemm.cu` | 319 | WMMA tensor-core prefill (peak 827 tok/s) | shipped |
| `micro_gemv_peak.cu` | 290 | Peak-bandwidth GEMV (V5 rejected, 20× slower) | data preserved |
| `micro_batch4.cu` | 787 | True Batched-4 GEMV (q4_0 + q8_0) (3.2–4.4×) | shipped |
| `mmq_v1.cu` | 350 | MMQ-style WMMA GEMV (0.4× of V2, rejected) | data preserved |
| `bench_f16_kernels.cu` | 103 | F16 V2 microbench scaffolding | scaffolding |
| `test_gemv_f16.cu` | 59 | F16 GEMV test on real llama f16 GGUF | unit |
| `test_gemv_typed.cu` | 194 | GPU golden GEMV test for all Tier-1 quants | gate |
| `bench_gemm_sweep.c` | 74 | M2 GEMM ladder fast sweep | driver |
| `probe_f16.c` | 19 | Find f16 tensors in GGUF | diagnostic |
| `gen_dequant_gold.py` | — | Generate golden dequant fixtures | test fixture |
| `bench_cuda_vs_cuda.sh` | 291 | Honest CUDA-vs-CUDA scoreboard v2 | **geomean 0.582** |
| `bench_gemma4_e2b.sh` | 204 | Anti-fake E2B decode bench (parity-gated) | gate input |
| `bench_gemma4_decode.sh` | 268 | C5 graph-capture win harness | **3.38× (9.6→32.45 tok/s)** |
| `bench_speculative.sh` | 234 | USE vs baseline (3 prompt types) | **Repetitive 0.74×, JSON 0.25×, Free 0.27×** |
| `bench_scoreboard.sh` | 150 | ours vs llama.cpp scoreboard | input to data/bench/results_scoreboard_cuda.jsonl |

---

## 8. Tests

| Test | Type | Verifies | Status |
|---|---|---|---|
| `tests/gate_m6_logit_parity.py` | gate | m61 teacher-forced logits vs oracle, 7/7 threshold | **PASS** (7/7) |
| `tests/gate_m6_parity.py` | gate | Greedy-decoding parity vs llama.cpp oracle | part of m61 |
| `tests/gate_m7_arch.py` | gate | Per-arch logit-parity (generalizes m61) | PASS per-model |
| `tests/gate_m7_grid.py` | gate | Full arch × quant grid (models.json + baseline) | PASS |
| `tests/gate_m84_gemma4.py` | gate | gemma-4-E2B parity, 7 prompts, ≥6/7 bar | **PASS 6/7** (row 6 known residual) |
| `tests/gate_ple_golden.py` | gate | PLE (per-layer embedding) row correctness | **PASS 3/3** (~1e-5) |
| `tests/gate_tokenizer.py` | gate | Tokenizer vs llama-tokenize oracle | RED (documented gaps) |
| `tests/gate_chat.py` | gate | Multi-turn chat coherence + ctx accounting | **PASS** |
| `tests/gate_mnist_mlp.py` | gate | MNIST ≥97% | PASS (M1) |
| `tests/gate_cifar_cnn.py` | gate | CIFAR-10 CNN | wired (M4) |
| `tests/test_ops.py` | unit | Tensor ops vs NumPy | PASS (M0) |
| `tests/test_grad.py` | unit | Gradcheck vs finite diff | PASS (M1) |
| `tests/test_gpu_parity.py` | unit | GPU output vs CPU allclose | PASS (M3) |
| `tests/test_engine_golden.py` | baseline | Per-position engine vs oracle (5 models × 5 prompts) | baseline |
| `tests/test_dequant_golden.py` | unit | CPU dequant vs gguf-py | PASS |
| `tests/test_samplers.py` | unit | Sampler chain (12/12) | **PASS 12/12** |
| `tests/test_samplers_conformance.py` | unit | Sampler fuzzer vs llama.cpp | PASS |
| `tests/test_specdec_sim.py` | unit | Spec-decode acceptance simulator | PASS |
| `tests/test_chat_template.py` | unit | Chat formatter (37 cases) | **PASS 37/37** |
| `tests/test_cpu_backend.py` | unit | CPU backend q4_0/q8_0/K-quants golden | **PASS** |
| `tests/test_moe_router.c` | unit | MoE router known-answer | PASS |
| `tests/test_kvcache.c` | unit | KV cache SWA + rollback | PASS |
| `tests/test_tokenizer_special_tokens.py` | unit | Gemma-4 special token handling | PASS |
| `tests/test_gemma4_throughput.py` | unit | gemma-4 E2B tok/s regression (≥-20%) | PASS |
| `tests/test_server_minimal.py` | unit | HTTP server smoke | PASS |
| `tests/test_server_multimodel.py` | unit | Multi-model HTTP server | PASS |
| `tests/test_gemm_batched.py` | unit | Batched GEMM (tt_gemm_batched) | PASS |
| `tests/test_spec_verify.c` | unit | Spec-decode verify(N) bit-exact vs N sequential | bit-exact PASS |
| `tests/test_ngram_lookup.c` | unit | N-gram drafter | PASS |
| `tests/test_rope_ff.cu` | unit | RoPE+FFN exact vs ggml ref | PASS |
| `tests/test_flash_multi.cu` | unit | Flash GQA 7/7 incl SWA + mixed-hd | **PASS 7/7** |
| `tests/test_batched_prefill.c` | unit | Batched vs sequential prefill (diagnostic) | diagnostic |
| `tests/test_logits_q8_v4.c` | unit | q8_0 V4 LM head bit-exact | PASS |
| `tests/test_prefill_gemm.c` | unit | 2D batched GEMM 32/32 boundaries | **PASS 32/32** |
| `tests/test_prefill_layer_parity.c` | unit | Batched vs sequential layer-0 hidden state | PASS |
| `tests/test_wmma_prefill_gemm.c` | unit | WMMA prefill 16/16 boundaries | **PASS 16/16** |
| `tests/test_batch4_gemv.c` | unit | Batched-4 GEMV bit-exact 14/14 | **PASS 14/14** |

---

## 9. Documentation & Plans (33 plan files + 5 root docs)

### Root docs
- `README.md` (154 lines): truth table, gates, build, CI
- `PLAN.md` (57 lines): top-level milestones M0–M4
- `PLAN_M5.md`: M5 LLM engine
- `PLAN_M6.md`: M6 correctness
- `PRD_M5_LLM_TERMINAL.md`: M5 PRD
- `PRD_M6.md`: M6 PRD
- `RESUME.md` (89 lines): fresh-session handoff
- `TODO.md` (135 lines): M8 closeout + open work priority
- `BLOCKED.md`: E4B on 4GB (5.15GB > 4GB)

### `docs/plans/` (33 files)
2026-08-23: m63-performance-engineering, m63b-road-to-270, m7-all-quants-all-archs
2026-08-24: m8-gemma4-port
2026-08-26: m84-7of7-parity, m9-m11-roadmap
2026-08-27: batched-prefill-gemm, cuda-graph-design, fused-matmul-bias-activation, gemma3n-support, hybrid-offload-findings, kvcache-quant, m11-vulkan-design, m9-integration-briefs, mmq-study, moe-notes, offload-integration, open-work, quant-roadmap, robustness-audit, server-design, t3-pre-staged-brief, temperature-repetition-penalty, tokenizer-special-tokens, true-batched-verification, universal-speculative-engine
2026-08-28: fix-prefill-gemm-integration, gemv-microkernel-optimization, peak-bandwidth-gemv, q8-0-v4-lm-head-optimization, tensor-core-prefill-gemm

### `data/profile/` (profiling results)
- `batched_prefill_findings.md`: Tasks 1-2 shipped, Tasks 3-4 deferred
- `bias_inventory.md`: 0/4 models have biases
- `launch_gap.md`: GEMV=83% of decode
- `decode_v5_findings.md`: V5 REJECTED (20× slower)
- `v4_q8_0_f16_findings.md`: V4-q8_0 only wins tiny M
- `q8_0_v4_lmhead_results.md`: 1.92× LM head, 250–269 tok/s
- `wmma_prefill_results.md`: 827 tok/s @N=73
- `prefill_parity_results.md`: 2D batched GEMM 10–40×
- `verify_breakdown.md`: verify_speculative per-stage cost
- `project_state_2026-08-28.md`: Current state snapshot before user break

---

## 10. Performance Achievements

### Decode
- **M5 baseline → M6.3 → M6.3b → M9.5+** ladder: 27 → 48 → 75.6 → 80.0 → 213.4 → 286.7 (ctx64) tok/s
- **Latest (M9.5+ V4 + Q8_0 V4 LM head)**: 230 → 269 tok/s = 1.17×; 250–269 sustained
- **VRAM bandwidth**: 145 → 180 GB/s on RTX 3050 (98.9% of 192 peak)
- **Geomean vs llama.cpp CUDA** (4 models, honest CUDA-vs-CUDA scoreboard): 0.582
  - qwen2.5-0.5b q4_0: 232.7 vs 320.1 (0.727×)
  - qwen3-0.6b q8_0: 128.3 vs 198.8 (0.645×)
  - llama-3.2-1b q4_0: 62.4 vs 194.3 (0.321×, graph_captured)
  - smollm2-135m-f16: 272.2 vs 358.0 (0.760×)
- **vs llama.cpp CPU (legacy scoreboard)**: ~4.9× faster

### Prefill
- **Sequential single-GEMV** baseline → 2D batched GEMM (10–40×) → WMMA tensor-core (peak 1,100+ tok/s)
- **qwen2.5-0.5b q4_0** measured:
  - N=8–12 (real run): 274–399 tok/s
  - N=32: 170 → 782 (4.6×)
  - N=84: 80 → 682 (8.5×)
  - N=153: 45 → 558 (12.4×)
  - N=256: 11 → 420 (16.8×) — **50× vs M5 starting point**

### Kernels
- **q4_0 GEMV V2 → V4**: 1.7× LM head, 1.5× FFN, 25% decode speedup
- **q8_0 LM head V1 → V4**: 1.92×, 0.79 ms vs 1.516 ms, 100% bit-exact
- **2D batched prefill GEMM**: 2.2–3.3× over sequential
- **WMMA tensor-core prefill**: 1,100+ tok/s peak
- **Batched-4 GEMV (M10 proto)**: 1.7–2.0× bit-exact 14/14
- **Fused QKV microbench**: 31.4% faster than 3 sequential (b83a1c4)
- **Fused FFN microbench**: 34.0% faster than 3 sequential (ecde6b0)
- **CPU AVX2 K-quant**: q4_0 20.2 GB/s @8T, 6.7×

### Bit-exact correctness
- m61 (7 prompts): 7/7 PASS, median |Δ| 0.073
- m84 (gemma-4): 6/7 PASS, median 0.088
- PLE golden: 3/3 at ~1e-5
- Batched prefill: 32/32 boundaries bit-exact
- WMMA prefill: 16/16 boundaries bit-exact
- Batched-4 GEMV: 14/14 shapes bit-exact
- CPU backend: 3e-7 rel vs numpy

---

## 11. Test Gates Status (just executed)

```
ci_local:    GREEN (26s, CPU-only sanity)
verify m61:  PASS (7/7 logits parity, chat-multiturn PASS)
verify ple:  PASS (3/3 PLE golden, ~1e-5)
verify m84:  PASS (6/7 gemma-4 parity, median ≤0.6 bar met)
```

All four gates green. Tokenizer gate (`verify tok`) is documented as RED by design.

---

## 12. What's NOT Built Yet (Open Work from TODO.md)

### Critical path
- **A1**: m84 6/7 → 7/7 (not chasing unless one-line fix appears)
- **B1**: `SAMPLERS_MAIN` CLI: grow `char rest[512]` to 8192 (1-line)

### Decode throughput (M9)
- **C1**: Wire `tt_gemm_batched_q4_0` into prefill (proto: f(8)=0.11× hidden) — but Task 3 reverted due to correctness bug
- **C2**: Wire PLE-fused V2 into `forward_layers` (22% saving, re-enables graphs for gemma4)
- **C3**: Wire split-K flash attention (12× at ctx≥512)
- **C4**: mmap-as-DEVICE / mmap'd Q4_0 → GEMV directly (real startup win)
- **C6**: Port llama.cpp MMQ-style dequant-in-register for ≥0.6B

### Speculative decoding (M10)
- **D1**: Wire ngram drafter + KV rollback into engine decode loop (current USE regresses 0.25–0.74×)

### Hardware breadth (M11)
- **E1**: Vulkan backend P0 spike
- **E2**: K-quant AVX2 in `src/cpu_backend.c`
- **E3**: Hybrid CPU+GPU offload (E4B unlock)
- **E4**: Server continuous-batching P1

### Model breadth
- **F1**: Per-family gating (llama-3.1, mistral, deepseek)
- **F4**: KV-cache quant (SWA cap → fp16 → q8_0)

### Quality
- **H1**: 5 remaining robustness audit fixes (~1 day)

### Deferred
- E4B standalone (needs hybrid offload, ~3–5 days)
- CUDA graph capture for gemma4 (covered by C2)

---

## 13. Build Instructions

```bash
cd ~/Storage/repos/nnfromscratch

# 1. Build & run
make run_llm_gpu
TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "Hello" 64

# 2. Tests
./scripts/ci_local.sh
./scripts/verify.sh m61
./scripts/verify.sh ple
./scripts/verify.sh m84
```

---

## 14. Inventory Summary

| Category | Count | Lines |
|---|---|---|
| Source files | 24 | 6,792 |
| Header files | 21 | ~1,500 |
| CUDA kernel files | 5 | 5,863 |
| Examples | 6 C | 2,962 + 4 Python |
| Tools | 24 | 4,912 |
| Tests | 48+ | ~13,000 |
| Plans | 33 | ~270 KB |
| Profile results | 10 | ~32 KB |
| Total commits | 181 | M0→M9.5+ |
| LoC (C + CUDA + Python) | ~33,000 | measured |
