# Robustness Audit — 2026-08-27

Report-only. Static reading + targeted grep; nothing executed on GPU. No source edited.
Scope: loader, engine create, tokenizer, runtime invariants, tools/CLI. Target user: solo-dev
inference engine people actually run on arbitrary downloaded GGUFs.

Severity classes: **crash** (SIGSEGV/SIGBUS/abort), **silent-wrong** (bad output, no error),
**ugly** (leak, confusing message, bad exit code).

---

## 1. Loader — `src/loader_gguf.c`

The loader mmaps the file and then walks it with a raw cursor `p` that is **never compared
against `mmap_addr + file_size`**. Every finding below is a facet of that.

### L1. Zero bounds checking on all header/KV/tensor reads — crash
`src/loader_gguf.c:88-107` (gguf_load), `read_u32/read_u64/read_string` (24-45).
A truncated or corrupt file makes `p` walk past the mapping → SIGBUS. Worst case is the
*header itself*: lines 92-95 read 24 bytes before the magic check at line 97, so even a
1-byte file crashes instead of printing "invalid magic". Also `fstat`-size 0 mmap is fine,
but any size < 24 crashes.
**Fix sketch:** compute `const uint8_t *end = mmap_addr + file_size`; thread it through
`read_u32/read_u64/read_string/skip_kv_value` (add `end` param, return failure flag); check
before every read; validate magic/version *before* reading counts. ~60 lines, mechanical.
Effort: **0.5 day**.

### L2. `t->ndim` unbounded write into `shape[4]` — crash (memory corruption)
`src/loader_gguf.c:215-218`. `ndim = read_u32()` then `for (d < ndim) t->shape[d] = ...`
with `int64_t shape[4]` (`include/loader_gguf.h:36`). A crafted/corrupt tensor header with
ndim > 4 smashes the rest of GGUFTensor and adjacent heap entries.
**Fix:** reject `ndim < 1 || ndim > 4`. One line + error print. Effort: **5 min**.

### L3. Tensor-count malloc bomb / int truncation — crash
`src/loader_gguf.c:109` casts u64 `tensor_count` to `int` (truncation: 2^32+10 becomes 10);
line 211 `calloc(tensor_count, sizeof(GGUFTensor))` (~176 B each) with no sanity cap → NULL
deref on calloc failure, or multi-GB alloc attempt from a 2 MB file. Same class:
`metadata_kv_count` loop (line 124) iterates u64 times.
**Fix:** after parsing header, require
`tensor_count > 0 && tensor_count <= file_size / 64 && metadata_kv_count <= file_size / 8`;
check calloc result; keep count as u64 internally. Effort: **30 min**.

### L4. Unvalidated tensor offsets → wild pointers — crash / silent-wrong
`src/loader_gguf.c:263-269`. `binary_base` is aligned but `t->offset` is never checked;
`t->data = binary_base + offset` can point anywhere (past EOF → SIGBUS on first memcpy;
into the KV region → silently wrong weights). Likewise computed `size_bytes` is never
checked against EOF.
**Fix:** per tensor, verify `aligned_offset + t->offset + t->size_bytes <= file_size`,
else fail load. Effort: **20 min**.

### L5. Unknown dtype falls through to `size_bytes = numel` — silent-wrong
`src/loader_gguf.c:256`. Any dtype not in the if-chain (e.g. Q5_K variant mislabeled, BF16
handled, but Q6_K-variants/IQ quants are not) gets 1 byte/elem — wrong sizes, wrong data
sliced out, engine later sees garbage or "missing weight". Only K-quants get the numel%256
warning.
**Fix:** explicit `else { fprintf(stderr, "unsupported dtype %d for %s\n", ...); fail }`.
Effort: **10 min**.

### L6. Unbounded recursion in `skip_kv_value` — crash (stack overflow)
`src/loader_gguf.c:59-83`: type 9 (ARRAY) recurses with `item_type` read from the file; a
crafted file sets item_type=9 forever → infinite recursion. GGUF forbids nested arrays but
corrupt input doesn't care.
**Fix:** reject nested array (`if (type == 9) item_type must be <= 12 && != 9`). Effort:
**5 min**.

### L7. Missing required keys — handled correctly (credit)
`tt_config_from_gguf` (`kernels/qwen2_cuda.cu:540`) returns zeroed cfg on dim/layers/heads ≤ 0;
`dump_logits` prints "config failed", exit 1. Engine create re-checks divisibility and
MAX_LAYERS/MAX_DIM (line 594). Clean failure chain.

### Non-NUL-terminated strings — handled correctly (credit)
All string reads go through bounded `read_string` (always NUL-terminates) or explicit
`slen`+manual NUL (tokenizer). Token strings stored verbatim with length table. No finding.

## 2. Engine create — `kernels/qwen2_cuda.cu`

### E1. Activation/cache cudaMalloc results unchecked — crash (delayed)
`kernels/qwen2_cuda.cu:700-702, 752-753, 815-818, 831-...` — `d_x/d_xn/d_q/d_att/d_h/d_g/
d_u/d_logits/d_kc/d_vc/d_k_stage/...` all unchecked. On OOM you get NULL device pointers;
first kernel launch/memset fails obscurely mid-run ("invalid argument" at some CHK_STAGE,
only visible with TT_DEBUG) instead of a clean create-time abort.
**Fix:** wrap in `CUDA_CHECK(must(...))` macro that ABORT_CREATEs. Note weight uploads ARE
checked inside `upload_w`/`upload_f32`, and layer-missing-weight check (line ~689) frees via
`qwen2_engine_free(e)` — good. Effort: **1 hour**.

### E2. Early-return leak before first tracked alloc — ugly (minor)
`kernels/qwen2_cuda.cu:627`: `if (!tembd) { fail(...); return NULL; }` skips
`qwen2_engine_free(e)` — leaks the calloc'd struct + created stream. Every other early
return frees correctly (ABORT_CREATE macro is used consistently — credit). Weight device
buffers leaked by design on free (documented, line ~908 comment) — acceptable for
process-lifetime engines, but means mid-create OOM also leaks all uploaded weights; fix as
part of E1's uniform path if desired. Effort: **5 min**.

### E3. `static float lg[262144]` vs vocab — silent-wrong (tool only)
`tools/dump_logits.c:61`. Buffer is safe (accessor caps copy at true vocab, line ~1810),
but any model with vocab > 262,144 yields argmax over a *prefix* of the vocab — silently
wrong dump, no warning. Gemma4-large vocabularies (256k+) are within one model family of
the cliff.
**Fix:** `float *lg = malloc(e->cfg.vocab * 4)` after create; pass `e->cfg.vocab`. Effort:
**15 min**.

### E4. MAX_LAYERS=128 vs gemma4 26B-A4B — OK today, cliff documented
`kernels/qwen2_cuda.cu:451` MAX_LAYERS=128; pl_* arrays `[MAX_LAYERS]` (lines 508-510);
guard at line 594 rejects n_layers>128 *cleanly*. gemma4 26B-A4B = 35 layers
(docs/plans/2026-08-27-quant-roadmap.md:5), E2B = 35 — headroom 3.6x. llama.cpp uses 512.
Only risk: future 128+ layer dense models get an ugly-but-safe "dims exceed engine limits".
**Verdict:** no bug; consider bumping to 256 when MoE lands. Effort: **0** (or 10 min bump).

### E5. Graph-capture warmup cudaMalloc unchecked — ugly
`kernels/qwen2_cuda.cu` graph warmup block: `cudaMalloc(&xsave,...)` unchecked; freed
correctly on both paths though. Fold into E1's macro sweep. Effort: **included in E1**.

### NaN poisoning — see R2.

## 3. Tokenizer — `src/tokenizer_bpe.c`

### T1. `MAX_SYM_BYTES` redefined 64 → 96 — ugly (warning), latent mismatch
`src/tokenizer_bpe.c:108` (`#define MAX_SYM_BYTES 64`) re-defined at line 363 as 96.
Line 234-235 build merge-rank keys into `char cat[64]` under the 64 definition; encode-side
lookups use 96-based buffers. Today keys ≥64 bytes can't be *stored* but chunk logic caps
chunks at 96 bytes → merged symbols up to 95 bytes can never hit a rank → those merges
silently never apply. Practically invisible (no real merge pair is that long), but the
redefinition warning is real and the two constants are semantically different things
(merge-pair bound vs chunk bound) sharing one name.
**Fix:** rename second to `MAX_CHUNK_BYTES`; single header define. Effort: **20 min**.

### T2. Long-input buffer bounds — mostly correct (credit), one soft spot
`chunk_len` caps runs (letter/punct/space-word) at MAX_SYM_BYTES=96 (lines 382-400);
`mapped[388]` fits worst-case 2 B/byte UTF-8 re-encode of a 96-byte chunk (192 B); `syms`,
merge `cat[192]`, and the `slen[j]+slen[j+1] >= MAX_SYM_BYTES` break all stay in bounds.
SP path `malloc(len*3+4)` covers '▁' expansion exactly. Decode grows `dec_buf` correctly and
output ≤ raw_len. **No overflow found.** Soft spot: `bpe_encode` returns 0 on malloc failure
of `mapped` (indistinguishable from empty text) — ugly only. Effort: **0** (optional: log).

### T3. Invalid UTF-8 — handled correctly (credit)
Decode: `utf8_dec` validates continuation bytes, falls back to byte passthrough (line ~330).
Encode operates on raw bytes via byte→unicode table, so invalid sequences tokenize as byte
fallback `<0xNN>` rather than crashing.

### T4. Vocab-size truncation `(int)n` — crash (same family as L3)
`src/tokenizer_bpe.c` tokens KV handler: `vocab_size = (int)n` then `calloc(n, ...)`;
huge n → NULL deref writing tokens[id]. Guard `n == 0 || n > 10_000_000` + calloc checks.
Covered by L1's end-bounds threading too. Effort: **15 min** (after L1).

## 4. Runtime

### R1. d_pos / max_ctx invariant — held, guards correct (credit) + one bypass
`prefill` returns -2 on `pos+n > max_ctx` (line ~1490); `next()` returns -2 at
`pos >= max_ctx-1`; `k_flash_gqa` documents "callers enforce pos < max_ctx"; scatter wraps
via `% max_ctx`. The one unguarded entry: `qwen2_debug_replay_step` checks token validity
but NOT `pos >= max_ctx` before launching the graph — a caller looping on it directly
(tools/profile_step.cu path) can push pos past max_ctx, where flash would read slots
≥ max_ctx out of the slab (into next layer's cache — silent-wrong) before anything traps.
**Fix:** add `if (!e || e->pos >= e->cfg.max_ctx - 1) return -2;` to replay_step. Effort:
**5 min**. Ring-wraparound itself (slot reuse after window) is only exercised by SWA which
attends [t0..pos], never wrapped slots while pos < max_ctx — invariant sound.

### R2. NaN poisoning — no containment; cheap mitigation worth having
No finite-check anywhere in the hot path; TT_TRACE does manual nan counting (debug only).
One inf/NaN layer output propagates through residuals → all logits NaN → argmax comparisons
all false → deterministically emits token 0 forever, streaming gibberish/BOS with no
diagnostic. For a solo-dev engine run on untested quant conversions this is the most likely
real-world "silent wrong" users will hit.
**Mitigation worth having (ranked by value/effort):**
1. In `sample_eager` (eager path) D2H the scalar already synced — extend `k_argmax_partial`
   to also OR-reduce `!isfinite(logits[i])` into a status int; host prints
   "[engine] non-finite logits at pos=N" once. Capture-path: same reduction node inside
   graph, status read alongside h_sampled. ~30 lines total.
2. Optional TT_ABORT_NAN env for parity debugging.
Not worth: per-layer containment/clamp (changes numerics, masks converter bugs).
Effort: **half day** including graph-path test.

## 5. Tools / CLI

### C1. `dump_logits` fwrite to unchecked FILE* — crash
`tools/dump_logits.c:72-74`: `fopen(dump_path,"wb")` result unchecked → unwritable/bad dir →
NULL deref segfault after minutes of GPU work.
**Fix:** `if (!f) { perror(dump_path); qwen2_engine_free(e); return 1; }` + check fwrite.
Effort: **10 min**.

### C2. Silent token-list truncation at 512 — silent-wrong
`tools/dump_logits.c:40-48`: `toks[512]`, loop stops at 512; extra ids dropped without a
word — teacher-forced dumps compare against a prompt the tool silently shortened.
**Fix:** warn+exit(1) when a further id exists after n==512. Effort: **10 min**.

### C3. Exit codes / messages — largely consistent (credit)
dump_logits: 0 success / 1 on every failure branch, usage line present, `--model` missing-
arg handled, engine freed on the token-parse error path. `chat` shell wrapper pre-checks
model file existence with clear message + exit 1. Minor inconsistency: prefill-failure path
(line ~55) returns 1 without freeing engine (process-exit cleanup makes it cosmetic).
Effort: **5 min** to tidy.

---

## Top-10 ranked fixes

| # | Finding | Sev | Where | Effort |
|---|---------|-----|-------|--------|
| 1 | L1 loader end-bounds checks (covers L3/L4/T4 partially) | crash | loader_gguf.c | 0.5 d |
| 2 | R2 non-finite logits detection + one-line diagnostic | silent-wrong | qwen2_cuda.cu | 0.5 d |
| 3 | L4 tensor offset+size vs EOF validation | crash | loader_gguf.c:263 | 20 min |
| 4 | L2 ndim>4 shape[] overflow guard | crash | loader_gguf.c:215 | 5 min |
| 5 | E1 checked cudaMalloc for activations/caches | crash | qwen2_cuda.cu | 1 h |
| 6 | R1 replay_step pos<max_ctx guard | silent-wrong | qwen2_cuda.cu | 5 min |
| 7 | C1 fopen NULL check in dump_logits | crash | dump_logits.c:72 | 10 min |
| 8 | L5 unknown-dtype hard failure (not numel fallback) | silent-wrong | loader_gguf.c:256 | 10 min |
| 9 | C2 warn on >512 token truncation | silent-wrong | dump_logits.c:42 | 10 min |
| 10 | L3/L6 tensor-count caps + nested-array recursion guard | crash | loader_gguf.c | 35 min |

Honorable mention: T1 MAX_SYM_BYTES rename (kills warning, prevents future confusion) — 20 min.
E3 dynamic lg buffer — 15 min, do before any >262k-vocab model lands.

## Already handled correctly (credit)

- Missing required config keys → clean chained failures (L7 above).
- All GGUF strings NUL-terminated via bounded reads; token strings length-tabled.
- Tokenizer long-input/UTF-8 paths fully bounds-checked; invalid UTF-8 degrades to byte fallback.
- MAX_LAYERS guard exists and errors cleanly; 35-layer gemma4 has 3.6x headroom.
- ABORT_CREATE used consistently; only one early return misses free (E2, cosmetic).
- prefill/next context-overflow guards; SWA attention window math; sync-copy discipline
  around d_pos (comments show the async-copy race was found and fixed properly).
- CLI exit codes consistent (0/1), usage text present, chat wrapper validates model path.

## Totals

Top-10: **~1.5 dev-days**. Full list incl. honorable mentions: **~2 days**.
Highest leverage single change: #1 (loader bounds) — eliminates 4 crash classes for
arbitrary downloaded files.
