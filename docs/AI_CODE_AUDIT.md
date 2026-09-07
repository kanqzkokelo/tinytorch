# AI Code Audit — `nnfromscratch`

**Scope:** Forensic attribution of LLM-generated vs human-authored code, with code-quality faults in the AI-produced portions.
**Method:** Static review of source, commit history, doc provenance, duplication fingerprints, and dated file bursts. No runtime/sanitizer runs.
**Date:** 2026-09-04
**Revision:** v2 — faults re-verified against source. v1 contained two fabricated faults (F2, F5) and two miscounts; those are removed/corrected here. Every fault below was read directly from source at the cited line.

---

## TL;DR

A **human-architected project**. The human owns the engine, the CUDA kernels, the loader, the debug archaeology, and the commit history. An LLM (subagents + a `gpt-5.6-thinking` distillation daemon) produced a large share of the **peripheral** code: module scaffolds, tests, micro-benches, docs, and the audit itself. The two voices are separable by style, timeline, and duplication pattern.

The AI code is **clean and mostly correct** (gates pass: m61 7/7, PLE 3/3, ci_local GREEN). The verified faults are concentrated in **missing error handling** and **duplicated helpers** — the standard LLM "happy path only" pattern. Two faults I initially reported (tokenizer error handling, a weight leak) do **not** exist and are retracted.

---

## 1. Attribution method

Three signals:

1. **Voice/style.** Human code: bug post-mortems, PTX-level detail, terse one-line comments, security-minded notes, commit messages recording failures. LLM code: multi-paragraph "what/why" headers, enumerated "Verifies: 1. 2. 3." lists, "Build (standalone…)" recipes, "Self-contained C99", "Transcribed from official HF", repeated "Deterministic".
2. **Timeline.** Human work clusters in milestone windows (M1–M6, Sep 2–4 roofline chase). LLM work clusters in bursts (Aug 26 15:49–16:34 six-module burst; Aug 27–29 micro-bench + agent-study docs).
3. **Duplication.** A human writing incrementally shares helpers; an LLM asked to "write a self-contained microbench" regenerates `fp16_to_fp32` / `warp_sum` / block structs in every file.

Legend: **A** = human, **B** = LLM, **A+B** = mixed.

> **Caveat:** voice-based attribution is probabilistic, not proof. The stylistic contrast is real but less sharp than v1 of this doc implied. Several "B" files (e.g. `samplers.c`) have terse headers that read more human than florid. Treat per-file labels as best-effort.

---

## 2. Timeline of AI usage

| Date | Window | What happened | Voice |
|---|---|---|---|
| Aug 21–23 | M1–M6 | Engine, kernels, loader, autograd, gemm born. Bug-driven commits. | **A** |
| Aug 24–25 | M7–M8 | `arch_registry.c` (deep llama.cpp cites), gemma4 port. | **A+B** |
| **Aug 26 15:49–16:34** | **M9–M11** | **Six modules in 45 min:** `cpu_backend`, `chat_template`, `specdec`, `samplers`, `kvcache`, `moe_router`. | **B (burst)** |
| Aug 27–29 | — | 18 `tools/micro_*.cu` + agent-study docs. | **B** |
| Sep 2–4 | cycles 22–28 | Roofline chase, FA2, cuBLAS, honest reverts. | **A** |

The **Aug 26 15:49–16:34 burst** is the clearest tell: six production modules in 45 minutes is a prompt fan-out, not incremental human work.

---

## 3. AI usage inventory (smoking guns)

These are **direct admissions or artifacts**, not style inferences:

| Artifact | Evidence |
|---|---|
| `docs/two_week_sprint_manual.md` (443 L, untracked) | **An LLM refusal saved to disk.** Opens: *"I can't truthfully provide a 'complete, compilable, drop-in' 14-day production manual from the information given."* |
| `scripts/distill_master_encyclopedia.py` (untracked) | Daemon shelling out to `askgpt -m gpt-5.6-thinking -r max` to "extract exhaustive, production-grade C/CUDA implementation manuals"; 25 s cooldown for "SecretGPT rate limit reset." Hardcoded `/home/mitesh/...` paths. |
| `docs/plans/2026-08-27-hybrid-offload-findings.md:3` | *"Source agent study of local oracle llama.cpp + web measurements."* |
| `docs/plans/2026-08-27-moe-notes.md:3` | *"Agent study of local oracle llama.cpp source (file:line cites inline)."* |
| `docs/plans/2026-08-27-t3-pre-staged-brief.md` | References a "subagent that fires the moment T2 lands" + a subagent transcript path. |
| `docs/plans/2026-09-03-three-hour-loop.md:27` | *"Get result via `get_subagent_result(24188bc2)`."* |
| `LOOP.md` | *"Reviewed by Minimax subagent (`q4kv_reviewer` - APPROVED with 97% confidence)"*; `q3k_reviewer`, `q2k_reviewer`. |
| `AUDIT.md` (8066 L, untracked) | LLM-generated static pattern dump; own "Limitations" admits static-only. |
| `docs/future_proofing_roadmap.md` (2005 L, untracked) | Blueprint prose. |

### File-by-file attribution (best-effort)

| File / dir | Voice | Note |
|---|---|---|
| `kernels/qwen2_cuda.cu` (6310 L) | **A** | Bug post-mortems, PTX detail |
| `kernels/gemv_q4_cuda.cu` (2014 L) | **A** | Pair-nibble fix narrative |
| `kernels/gemv_typed.cu` (1604 L) | **A** | Warp-shuffle reduction detail |
| `src/loader_gguf.c` (417 L) | **A** | Security comment: "corrupt/hostile input aiming at a malloc bomb" |
| `src/autograd.c` (732 L) | **A** | Terse |
| `src/gemm.c` (675 L) | **A** | Terse |
| `src/ngram_lookup.c` | **A** | 32 lines, bare |
| `src/arch_registry.c` | **A+B** | Deep llama.cpp cites (human-verified), LLM prose |
| `src/tokenizer_bpe.c` (1116 L) | **A+B** | LLM transcription of llama.cpp semantics, human-tested |
| `src/dequant_ref.c` | **A+B** | "only the math semantics are reproduced" |
| `src/samplers.c/.h` | **B** | Aug-26 burst; "Self-contained C99" |
| `src/chat_template.c/.h` | **B** | "Transcribed from official HF chat_template Jinja" |
| `src/specdec.c/.h` | **B** | Contract-header style |
| `src/kvcache.c/.h` | **B** | "See kvcache.h for contract" |
| `src/moe_router.c/.h` | **B** | "Oracle reference (local llama.cpp…)" |
| `src/cpu_backend.c/.h` | **B** | Standalone-driver CLI |
| `src/tinytorch_ipc.c/.h` | **B** | "> 50,000 req/s" framing |
| `examples/server_minimal.c` | **A+B** | Human spec, LLM polish |
| `examples/chat_llm_gpu.c`, `run_llm_gpu.c` | **A** | Stop-string guards, terse |
| `tools/micro_*.cu` (18) | **B** | Identical "Verifies:" headers; duplicated helpers |
| `tests/test_*.py` (newer) | **B** | "Golden-string tests… transcribed from official HF" |
| `docs/plans/*.md` | **B** | "Source agent study…" |

---

## 4. Verified code-quality faults

Severity: **P0** = correctness/data-loss, **P1** = robustness, **P2** = hygiene.
**Every line below was read from source.** v1's F2 and F5 are retracted (see §6).

### 4.1 Missing error handling

| # | Fault | Location | Severity | Verified |
|---|---|---|---|---|
| F1 | **66 of 67 `cudaMemcpy` calls unchecked** — no `cudaError_t` / `CUDA_CHECK` / `cudaGetLastError` in the following 3 lines. A failed transfer silently yields garbage logits. | `kernels/qwen2_cuda.cu` | **P1** | Counted programmatically: 67 total, 66 lack an adjacent check. |
| F3 | **`ttq_dequant` has no NULL check on `data` or `out`.** K-quants check `numel % QK_K` (return -5) but the pointer args are dereferenced unconditionally. | `src/dequant_ref.c:336` | **P1** | Read the switch; 0 NULL guards in file. |
| F4 | **Loader leaves `t->data = NULL`** with a "will be resolved after alignment" / "engine falls back to skip" comment — the resolve site is not guarded here. | `src/loader_gguf.c:296,378` | **P2** | Read both sites; relies on downstream NULL-fallback. |

### 4.2 Duplication (LLM fingerprint)

| # | Fault | Location | Severity | Verified |
|---|---|---|---|---|
| F6 | **`fp16_to_fp32` duplicated identically in 3 files** (`dequant_ref.c`, `micro_gemv_q2_K.cu`, `micro_gemv_q3_K.cu`). `cpu_backend.c` has a namespaced variant `cb_fp16_to_fp32`, not a blind copy. | those 3 files | **P2** | Diffed the bodies — identical bit-manipulation. |
| F7 | **`warp_sum`/`warp_reduce` re-implemented in 15 `tools/micro_*.cu`** files. | `tools/micro_*.cu` | **P2** | `grep -rln` = 15 files. |
| F8 | **`BlockQ2_K`/`BlockQ3_K` structs duplicated** between the two K-quant microbenches. | `micro_gemv_q2_K.cu`, `micro_gemv_q3_K.cu` | **P2** | Both define the struct + `static_assert`. |

### 4.3 Process / provenance

| # | Fault | Location | Severity | Verified |
|---|---|---|---|---|
| F9 | **LLM refusal saved as a deliverable.** | `docs/two_week_sprint_manual.md` | **P2** | First line is the refusal. |
| F10 | **Non-portable distillation daemon in-tree** — hardcoded home paths + private `askgpt` binary. | `scripts/distill_master_encyclopedia.py` | **P2** | Read the `cmd` array. |
| F11 | **8066-line static-only audit checked in** — pattern dump, no runtime verification. | `AUDIT.md` | **P2** | Untracked; self-declared static-only. |

---

## 5. Production risk summary

- **Highest verified risk (P1):** F1 — 66 unchecked `cudaMemcpy`. Under GPU memory pressure or a bad pointer, transfers fail silently.
- **P1:** F3 (dequant no NULL/bounds on pointers), F4 (loader NULL-offset reliance on downstream fallback).
- **P2:** F6–F11 — duplication and provenance hygiene.

None block the current gates (which pass). They bite under: GPU transfer failures, NULL/misaligned dequant inputs, or reuse of the microbench helpers.

---

## 6. Retractions (v1 → v2)

Honesty ledger — what v1 got wrong:

| v1 claim | Reality | Status |
|---|---|---|
| "F2: `bpe_tokenizer_init` ignores `c.err`" | No `c.err` exists. File uses `fprintf(stderr)+return NULL` — **correct** error handling. | **Retracted (fabricated)** |
| "F5: weight leak, `cudaFree` commented out" | `cudaFree` present at `qwen2_cuda.cu:3781–3799`; weights **are** freed. | **Retracted (fabricated)** |
| "F1: 47 unchecked cudaMemcpy" | Actually **66/67**. | **Corrected (undercount)** |
| "F6: fp16_to_fp32 in 4 files" | 3 identical + 1 namespaced variant. | **Corrected (miscount)** |

Cause: v1 pattern-matched "LLM = missing error handling" and asserted specific faults without reading each file. That is the same unverified-assertion failure mode this audit warns about. v2 reads every cited line.

---

## 7. Verdict

**Human = architect + kernel engineer.** Load-bearing code — FA2 tensor-core flash, pair-nibble GEMV fix, HD=64 OOB hunts, CUDA-graph capture, GGUF loader, debug archaeology — is human.

**LLM = documentation writer, test generator, micro-bench factory, doc-distiller.** Competent scaffolding with the usual gaps: unchecked CUDA calls, missing pointer validation, duplicated helpers, one saved refusal, one non-portable daemon.

**The AI code is good scaffolding, not yet good production code** until F1–F4 are closed.

---

## 8. Recommended actions

1. **Close P1:** add a `TT_CHECK_CUDA(cudaMemcpy(...))` macro and apply to the 66 unchecked calls; NULL-check `data`/`out` in `ttq_dequant`; guard the loader resolve site.
2. **De-duplicate (P2):** consolidate `fp16_to_fp32`, `warp_sum`, and K-quant block structs into shared headers.
3. **Quarantine provenance artifacts:** move `two_week_sprint_manual.md` (refusal), `distill_master_encyclopedia.py` (private-path daemon), and `AUDIT.md` (unverified) out of the shipping tree or mark non-deliverable.
4. **Tag provenance:** add `CODE_PROVENANCE.md` marking each file A / B / A+B.
