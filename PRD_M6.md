# PRD: Milestone 6 — A *Correct* LLM Engine, Then an Honestly Fast One

**Status:** Supersedes `PRD_M5_LLM_TERMINAL.md` and `PLAN_M5.md`.
M5's throughput claims (277.7 tok/s, ">130 tok/s beats llama.cpp") are voided by the
M6 design review: the captured CUDA graph does not compute Qwen2.5 (no residuals,
no per-layer RMSNorm, no RoPE, KV cache never written, GQA violated), the tokenizer
destroys byte-level vocab entries, and `run_llm_gpu.c` never embeds the prompt.
No performance number counts until Gate Q passes.

---

## 0. Non-negotiable rules

1. Truth = exit code of `./scripts/verify.sh m6x`. Never self-assessment.
2. **Anti-fake clause:** every benchmark invocation runs a validity assertion
   (greedy-output agreement vs llama.cpp) *in the same process* before printing
   tokens/sec. If validity fails, the bench FAILS regardless of speed.
3. Commit after every green gate (`M6.x: <gate> PASS`). >20 red commits on one
   gate → stop, write `BLOCKED.md`, exit.
4. Dependencies: CUDA toolkit, pybind11, **plus exactly one new oracle**:
   a pinned llama.cpp build (for reference logits/tokenization/baseline timing).
   Nothing else.
5. All model dims come from GGUF metadata. Hardcoding 896/4864/14/24 anywhere in
   `src/` or `kernels/` is a review-rejectable offense.

## Hardware & physics budget (targets are derived, not vibes)

RTX 3050 laptop, sm_86, 14 SMs, ~176–192 GB/s VRAM. Qwen2.5-0.5B-Instruct q4_0:
dim=896, ffn=4864, 24 layers, 14 Q heads / **2 KV heads (GQA)**, head_dim=64,
vocab=151936, tied embeddings. Per decode step the engine must read ≥ once:

| stream | bytes |
|---|---|
| layer weights (q4_0) | ~253 MB |
| embedding/lm-head (tied, q4_0) | ~76 MB |
| KV cache @512 ctx (fp32, 2 kv heads) | ~14 MB |
| **total ≈ 340 MB** | → **ceiling ≈ 520–550 tok/s** |

Realistic well-engineered target is 45–65% of ceiling. llama.cpp tg-128 on this
GPU is expected at ~60–90 tok/s. Beating it is plausible because we pay zero
abstraction cost; it is not guaranteed. Gates below reflect that.

---

## M6.0 — Hygiene & honesty (½ day)

- [ ] Commit all untracked M5 work to `m6-correctness` branch before anything else.
- [ ] Delete or quarantine fake/demo artifacts: canned `sample_token_ids[]` in
      `examples/chat_llm_gpu.c`, "AUTHENTIC GENERATION" banner, dead
      `flash_attn_decode_cuda.cu` (duplicate symbol), unused statics in
      `ops_col2im.c`, undefined decls (`gguf_upload_to_gpu`, `run_lm_head_logits`
      — define or remove).
- [ ] Dedupe `bench/results.md`; add README skeleton linking PLAN/PRD/results.
- [ ] Fix inline-asm clobber list in `src/gemm.c` (declare ymm0–ymm14).
- **Gate H (hygiene):** repo greps clean of hardcoded shape constants in engine
  paths; `make lib pybind cuda cublas` all build from clean checkout; verify.sh
  m0–m4 unchanged results.

## M6.1 — Correct transformer forward (the core milestone)

Rewrite the engine path (`kernels/llm_engine_cuda.cu`, new `kernels/qwen2_cuda.cu`)
to compute real Qwen2.5 per token t with KV cache:

1. Embedding lookup (dequant q4_0 row → fp32).
2. Per layer: `h ← x·rmsnorm(x, attn_norm)` then:
   Q,K,V = q4_0 GEMV against x_norm; **K,V written into cache slot t**
   (layout `[seq, n_kv_heads, 64]`).
   **RoPE** applied to Q and K in-place, fp32 angles, freq base from GGUF
   (`llama.rope.freq_base` / qwen2 key, default parse → 1e6 for Qwen2.5).
   Attention with **GQA**: each of 14 Q heads maps to KV head `q/7`;
   online-softmax flash decode reads cache rows [0..t] for its KV group only.
3. `x += attn_out·W_o` (residual). `h ← rmsnorm(x, ffn_norm)`.
   SwiGLU MLP fused kernel (gate/up already fused). `x += down(h)` (residual).
4. Final `rmsnorm(x, output_norm)` → logits GEMV (tied embd) → **argmax kernel
   parallelized** (block-tree reduction, no single-thread scan).

Implementation notes:
- No cudaGraph capture until M6.3; first make the eager path correct.
- Keep existing warp-per-row q4_0 GEMV (`_fast` pairing: nibble j low → x[j],
  j high → x[j+16]); delete the mis-paired `k_gemv_q4_0` or fix its indexing.
- Rebuild golden q4_0 dequant tests against true GGML layout
  (`tests/test_q4_dequant.py`: random blocks, NumPy GGML-style dequant,
  rtol 1e-3) — this invalidates any prior parity pass through the old kernel.

**Gate Q1 (op parity):** rmsnorm / rope(pos 0..511) / swiglu / gqa-flash-decode /
kv-write match a NumPy golden (`tests/gate_m6_ops.py`), rtol 1e-3.
**Gate Q2 (end-to-end logits parity, the truth gate):** teacher-forced greedy:
feed 64 prompt tokens through our engine and pinned llama.cpp; compare final-position
logits top-1 and top-5 sets across 16 diverse prompts.
Pass = ≥95% prompts agree on top-1 AND max |Δlogit| ≤ 0.35 on agreeing prompts.
Script: `tests/gate_m6_logits_parity.py` (uses llama.cpp via `llama-cli`/cffi harness).

## M6.2 — Real byte-level BPE tokenizer (2 days)

Replace `tokenizer_bpe.c` matching logic; keep GGUF loading but store raw bytes:

- Store tokens verbatim (no ASCII filtering). Vocab entry = byte string.
- Encode = true byte-level BPE: split on regex pre-tokenization pattern (port the
  Qwen2.5 GPT-4-style regex), rank-driven merges using `tokenizer.ggml.merges`
  array from GGUF (new parser case), byte-fallback via `<0xNN>` tokens.
- Decode = concatenation + UTF-8 validation (emit replacement char on bad seq),
  special-token awareness (`<|im_start|>` etc. from `tokenizer.ggml.tokens` +
  added-tokens KV).
- Chat template applied host-side (keep current `<|im_start|>` formatting).

**Gate T1:** encode(decode(ids)) == ids round-trip on ASCII + emoji + CJK corpora.
**Gate T2 (exactness):** token IDs identical to llama.cpp tokenizer on a 200-line
mixed-language corpus incl. the chat-template wrapper: 100% match required
(`tests/gate_tokenizer.py`, compares against `llama-tokenize` output).

## M6.3 — Performance engineering (only after Q2+T2 green)

Ordered by expected ROI on a bandwidth-bound workload:

1. **One sync per token.** Kill both `cudaDeviceSynchronize()` in sampling;
   argmax stays on GPU, D2H copy of one int via pinned memory + stream sync.
2. **Fuse per-layer elementwise chains**: rmsnorm+rope, rope+kv-cache-write,
   residual-add+rmsnorm. Target ≤8 kernels per layer.
3. **cudaGraph replay with ring-buffer KV cache**: capture once at max_ctx;
   per-step position passed via device-side pointer update (graph exec update,
   not recapture). Flash-decode loop bound becomes a device scalar.
4. **Vectorize GEMV loads**: uint4 loads of qs[16], keep warp shuffle reduction;
   try 2 rows per warp (better occupancy at dim=896).
5. Optional: fp16 x-vector with fp32 accumulate in GEMV (halves activation
   traffic; validate under Gate Q2 tolerance again).

Benchmarks follow house protocol (median of 20, warmup 5, clocks noted) plus:
- Baseline: pinned llama.cpp commit, `-ngl 99 -p <same prompts>`, report its
  tg-128 alongside ours in the same table row region.
- **Validity assertion runs inside every timed process** (rule 0.2): greedy
  128-token continuation of a fixed prompt set must match llama.cpp greedy
  output ≥95% token-exact; otherwise print FAIL and exit nonzero.

**Gate P1 (parity-preserving speed):** eager path ≥ 60 tok/s end-to-end on
Qwen2.5-0.5B q4_0 while Gate-Q2 script still passes post-change.
**Gate P2 (head-to-head):** graph path ≥ 1.0× llama.cpp tg-128 median on
identical prompts/model/hardware. Stretch (non-gating): ≥ 1.3×, i.e. ~100+ tok/s.
**Gate P3 (scaling sanity):** Llama-3.2-1B q4_0 generates correctly (Q2-style
spot check, 4 prompts) at ≥ 40 tok/s — proves metadata-driven dims, not
0.5B-specific hacks.

## M6.4 — Terminal chat that deserves the name (1 day)

- Interactive REPL: proper prefill over encoded prompt tokens (KV populated
  during prefill — decode attends to real context), streaming via AsyncPrinter
  (add producer backpressure: block when queue full instead of wrapping).
- UTF-8-safe incremental decode (stateful partial-sequence buffering).
- `/bench` command runs the P2 harness live.
- **Gate C:** scripted pexpect session — prompt echo, multi-turn context
  retention ("my name is X … what is my name?"), clean Ctrl-C, no mojibake.

---

## Verification wiring

```
scripts/verify.sh m6    → m6.0 hygiene checks
scripts/verify.sh m61   → test_q4_dequant.py, gate_m6_ops.py, gate_m6_logits_parity.py
scripts/verify.sh m62   → gate_tokenizer.py
scripts/verify.sh m63   → bench_llm.py --gate   (includes validity assertion)
scripts/verify.sh m64   → tests/gate_chat.py
```

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Logits parity never reaches 95% | Bisect layer-by-layer: dump per-layer hidden states vs Python gguf reference (numpy + dequant), find first diverging op. This debug path is built *first* in M6.1. |
| Graph replay incompatible with dynamic seq_len | Fallback: persistent-kernel mega-kernel or plain stream launches; P2 gate does not require graphs, only the tok/s. |
| Beating llama.cpp proves impossible at equal correctness | Ship P1 + publish honest ladder table; the result "within X% of llama.cpp, from scratch, with parity proofs" still meets the spirit of the original plan. Update BLOCKED.md per rule 3. |
| 4GB VRAM pressure at 1B model + long ctx | Cap ctx at 1024 for 1B; KV in fp16 if needed (re-run Q2 tolerance check). |

## Human involvement (~5 min/day)

`git log --oneline && ./scripts/verify.sh m6 && tail bench/results.md`.
Decision points requiring human sign-off:
1. After M6.1 Gate Q2 first attempt (parity tolerance choice 0.35 logits / 95%).
2. After P2: accept stretch-chase vs freeze and move to docs/README polish.

## Definition of done

A stranger can run `make -B all && ./scripts/verify.sh m6`, watch every gate go
green, type a question into `build/chat_llm_gpu`, get a coherent streamed answer
that provably comes from Qwen2.5, and see a tokens/sec number ≥ the llama.cpp
number printed next to it in the same table — with every line of engine code
derived from GGUF metadata rather than constants.
