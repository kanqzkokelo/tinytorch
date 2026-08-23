# PLAN_M6 — Implementation Plan for PRD_M6 (Correct LLM Engine → Honestly Fast)

Companion to `PRD_M6.md`. Order is strict: no phase starts before the previous
phase's gates are green (exception: M6.2 tokenizer can proceed in parallel with
late M6.1 since it shares no code path).

```
M6.0 ──→ M6.1 ──→ M6.3 ──→ M6.4
            ↘ M6.2 ↗   (T2 must be green before P-gates run)
Oracle setup (llama.cpp pin) runs first, everything depends on it.
Est. total: 12–18 autonomous-loop days.
```

---

## Phase O — Oracle setup (day 0, prerequisite for everything)

1. Build pinned llama.cpp into `oracle/llama.cpp` (record commit hash in
   `bench/results.md`). Targets needed: `llama-cli`, `llama-bench`,
   `llama-tokenize`.
2. `scripts/setup_oracle.sh`: clones, checks out pin, builds with CUDA,
   verifies `llama-bench -m data/models/qwen2.5-0.5b-instruct-q4_0.gguf`
   runs; writes baseline tg-128 number into `bench/results.md`.
3. Capture reference artifacts used by later gates:
   - Greedy continuations for 16 fixed prompts (`tests/fixtures/prompts.txt`)
     → `tests/fixtures/llamacpp_greedy.txt`.
   - Tokenizations of the T2 corpus → `tests/fixtures/llamacpp_tokens.txt`.

**Exit:** oracle baseline table row exists; fixtures committed.

---

## Phase M6.0 — Hygiene (½ day)

Files: `.cu` deletions, `examples/chat_llm_gpu.c`, `ops_col2im.c`,
`include/loader_gguf.h`, `src/gemm.c`, `bench/results.md`, `README.md`.

| # | Task | Detail |
|---|------|--------|
| 0.1 | Branch + commit M5 WIP | `git checkout -b m6-correctness && git add src/loader_gguf.c src/tokenizer_bpe.c src/ops_llm.c src/async_printer.c kernels/gemv_q4_cuda.cu kernels/flash_attn_decode_cuda.cu kernels/llm_engine_cuda.cu examples/*.c *.py PRD_M5* PRD_M6.md` — commit BEFORE any edit |
| 0.2 | Purge demo fossils | Delete `sample_token_ids[]`/`num_sample_tokens` from `chat_llm_gpu.c`; remove "AUTHENTIC GENERATION" banner wording from `run_llm_gpu.c` |
| 0.3 | Dead code | Delete `kernels/flash_attn_decode_cuda.cu` (unlinked duplicate symbol); delete unused statics in `src/ops_col2im.c` or fold the live col2im into `autograd.c` and drop the file |
| 0.4 | Undefined symbols | Remove `gguf_upload_to_gpu` decl from `loader_gguf.h` OR implement it (prefer remove); `run_lm_head_logits` gets implemented properly in M6.1 — mark extern with comment until then |
| 0.5 | Asm clobbers | Add `"%"ymm0.."%ymm14"` (and `"cc"` not needed for AVX) to both asm blocks in `src/gemm.c`; re-run `verify.sh m2` — expect identical numbers |
| 0.6 | Docs | Dedupe CUDA ladder tables in `results.md`; README skeleton: what/oracle/results/how-to-run |

**Gate H** (`scripts/verify.sh m6`): clean-build all targets; grep gate
`scripts/check_no_hardcoded_dims.sh` — engine paths (`kernels/*qwen*, llm_engine*,
gemv_q4*`) must contain no literal 896/4864/151936/24-layer constants;
m0–m4 gates unchanged.

---

## Phase M6.1 — Correct forward pass (days 1–7) ⚠ core milestone

### Step 1: NumPy golden reference FIRST (debug bisection path)
- **File:** `tests/ref_qwen2_numpy.py`
  - Loads GGUF via `loader_gguf.c` ctypes (reuse `real_chat.py` plumbing).
  - Implements full Qwen2.5 forward in NumPy with correct GGML q4_0 dequant
    (nibble j low → x[j], j high → x[j+16]), GQA (14Q/2KV), RoPE @1e6,
    RMSNorm eps from metadata, residuals.
  - CLI: `--dump-layer N --tokens i,j,k` writes intermediate hidden states
    to `.npy` for diffing against device dumps.

### Step 2: Metadata-driven engine skeleton
- **New:** `include/qwen2_engine.h`, `kernels/qwen2_cuda.cu`
  ```c
  typedef struct { int dim, ffn, n_layers, n_heads, n_kv_heads, head_dim,
                   vocab; float rms_eps, rope_base; } TTConfig;
  TTConfig tt_config_from_gguf(GGUFModel *m);      // fails if keys missing
  void *tt_engine_create(const TTConfig *cfg, GGUFModel *m); // allocates weights+KV on device
  int   tt_engine_prefill(void *eng, const int *toks, int n);   // fills KV
  int   tt_engine_next(void *eng);                              // returns argmax id
  ```
- All grid dims derived from cfg at engine-create time. No constants.

### Step 3: Kernels (eager launches only — no graphs this phase)
Modify/new in `kernels/qwen2_cuda.cu`:
1. `k_rope_f32` — in-place RoPE on Q rows `[n_heads,64]` and K rows
   `[n_kv_heads,64]`, angle from `pos * base^(-2i/64)`, fp32 sinf/cosf.
2. `k_kv_write` — append K,V projections into cache layout
   `[max_ctx, n_kv_heads, 64]` at slot `pos`.
3. `k_flash_gqa_decode` — one warp per Q head; maps head h → kv_group h/(n_heads/n_kv_heads);
   streams cache rows [0..pos]; online softmax (port existing math); reads
   `pos` from a device scalar so graphs can reuse it later.
4. `k_add_residual` — `x += y`.
5. `k_argmax_tree` — two-stage block reduction replacing single-thread scan.
Keep: `_fast`/`_ultra` q4_0 GEMV (pairing already correct). Delete mis-paired
`k_gemv_q4_0`. Generalize `k_rmsnorm_gamma` eps → runtime arg.

### Step 4: Rewrite driver
- `examples/run_llm_gpu.c`: encode prompt (temporarily llama.cpp-side fixture
  IDs if M6.2 isn't done yet), `tt_engine_prefill`, loop `tt_engine_next`,
  print via AsyncPrinter. Prompt actually reaches the model now.

### Step 5: Bisect until parity
Debug protocol when Q2 fails: dump per-layer hidden states from device after
layer L (`cudaMemcpy` + save), diff against `ref_qwen2_numpy.py --dump-layer L`;
fix first diverging op; repeat. Budget: this is where most days go.

**Gates:**
- `tests/test_q4_dequant.py` — dequant vs GGML-exact NumPy, rtol 1e-3 (replaces
  old parity that inherited the wrong pairing).
- `tests/gate_m6_ops.py` — rope/kv-write/gqa-flash/rmsnorm/residual goldens.
- `tests/gate_m6_logits_parity.py` — teacher-forced 64-token prompts ×16:
  ≥95% top-1 agreement, max|Δlogit| ≤ 0.35 on agreements.

**Definition of done:** engine generates coherent English on arbitrary prompts;
greedy output matches `tests/fixtures/llamacpp_greedy.txt` line-for-line on
≥95% of prompts.

---

## Phase M6.2 — Byte-level BPE tokenizer (days 4–6, parallelizable)

Files: rewrite `src/tokenizer_bpe.c` internals (~400 lines new), header unchanged.

1. **Loader additions** (`loader_gguf.c`): parse `tokenizer.ggml.merges`
   (STRING array, each entry "left right"), `tokenizer.ggml.added_tokens` if
   present; store raw bytes — REMOVE the ASCII filter in token loading.
2. **Data structures**: open-addressing hash map (FNV-1a) byte-string→token-id;
   pair map `(id_left,id_right)→rank` encoded as u64 key.
3. **Encode**: regex pre-tokenization (port Qwen2 pattern
   `'(?i:'sd|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|^\s+`
   — hand-rolled matcher over UTF-8, ~150 lines, unicode category tables
   generated into `src/unicode_tables.h` by `scripts/gen_unicode_tables.py`);
   per chunk: start from single bytes, repeatedly merge lowest-rank adjacent
   pair (heap or linear scan — chunks are short); unknown byte → `<0xNN>`.
4. **Decode**: concat raw bytes; stateful UTF-8 validator emits U+FFFD on
   invalid sequences; never filters.
5. Special tokens: exact-string match pass before BPE (chat template tags).

**Gates:** T1 round-trip incl. emoji/CJK/control chars; T2 = 100% ID match vs
`tests/fixtures/llamacpp_tokens.txt` on 200-line corpus + templated prompts.
Encode speed target: <50 ms for a 200-token prompt (hash maps make this easy;
the old vocab-scan took seconds).

---

## Phase M6.3 — Performance (days 7–12) — only after Q2 AND T2

Order = ROI order. After EACH step re-run Gate Q2 (parity is load-bearing):

1. **Single sync/token**: remove both `cudaDeviceSynchronize()` from sampling;
   argmax result D2H via pinned buffer + `cudaMemcpyAsync` +
   `cudaStreamSynchronize` on a side stream. (~biggest latency win at bs=1)
2. **Fusion**: rmsnorm+rope → one kernel; rope+kv_write → one; residual+
   next-rmsnorm → one. Target ≤8 kernels/layer (from ~12).
3. **Graph replay**: capture whole step once at max_ctx; flash-decode already
   reads device-scalar `pos`; KV ring-buffer wrap handled by modular index in
   k_kv_write. No recapture per token.
4. **GEMV micro-opt**: uint4 loads of `qs[16]` (the `_ultra` body minus its
   misleading comments); try 2 rows/warp variant; pick winner by bench.
5. *(Optional)* fp16 activations w/ fp32 accum in GEMV — halves activation
   traffic; MUST re-pass Q2 tolerance.

**Harness:** `bench/bench_llm.py --gate` — runs engine binary end-to-end
(prefill prompt + 128 greedy tokens, wall clock), runs validity assertion
(≥95% token-exact vs llama.cpp greedy fixture) in-process, prints tok/s ONLY
if valid, compares to oracle tg-128 from Phase O, exits nonzero on any fail.

**Gates:** P1 eager ≥60 tok/s valid · P2 graph ≥1.0× llama.cpp median ·
P3 Llama-3.2-1B spot-check coherent ≥40 tok/s.

---

## Phase M6.4 — Real chat terminal (day 13–14)

1. Prefill path populates KV for full prompt (already true via M6.1 design);
   multi-turn = keep KV, append turn tokens, template between turns.
2. `AsyncPrinter` backpressure: producer blocks (semaphore) when queue full
   instead of wrapping.
3. Stateful incremental UTF-8 decode before pushing partial tokens.
4. `/bench` slash command invoking the M6.3 harness live.

**Gate C:** `tests/gate_chat.py` (pexpect): asks name, asks name back,
checks answer contains it; emoji prompt survives; Ctrl-C exits cleanly.

---

## Checkpoints requiring human sign-off

| When | Decision |
|---|---|
| First Q2 attempt fails | Accept tolerance (0.35 logits / 95%) vs push to 99%? |
| P2 result known | Chase 1.3× stretch vs freeze + README/docs polish |
| Any BLOCKED.md written | Recalibrate gate vs extend effort |

## Standing rules recap
Exit-code truth via verify.sh · commit per green gate · >20 reds → BLOCKED.md ·
validity assertion inside every timed run · zero hardcoded dims.
