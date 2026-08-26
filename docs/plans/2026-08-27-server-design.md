# Server design: continuous-batching multi-request LLM inference

Date: 2026-08-27
Status: DESIGN (pre-code)
Author: post-M11 hardware-breadth survey, server M12 work
Depends on: M9 arena loader (M9.3), M10 spec-decode skeleton, M11 backend shim
Refs: `oracle/llama.cpp/tools/server/{server,server-context,server-task,server-queue}.{cpp,h}`,
`src/kvcache.h`, `src/specdec.h`, `include/qwen2_engine.h`,
`docs/plans/2026-08-27-{m11-vulkan-design,hybrid-offload-findings,m9-m11-roadmap}.md`

Scope: from-scratch C/CUDA/C++ engine → OpenAI-compatible HTTP server with
continuous batching (vLLM/llama.cpp "iteration-level scheduling"). No code in
this doc. Read-only analysis of local sources.

## 0. Current shape (one-shot decode loop)

`Qwen2Engine` today is single-stream, one-pos-at-a-time:

- KV cache: device buffers `d_kc/d_vc` laid out `[layer][kv_head][max_ctx][head_dim]`
  (kernels/qwen2_cuda.cu:223 comment, `:286-291` scatter). Indexing:
  `Kc[layer][pos*kvdim + i]`. One logical sequence.
- Position: `d_pos` (single int on device, kernels read via `*d_pos`, M6.3
  capture-friendly).
- Forward entry points (qwen2_engine.h):
  - `qwen2_engine_prefill(e, toks, n)` — loops `n` prefill tokens through
    `forward_layers`; only one path touches K/V write (`k_kv_scatter` per
    token).
  - `qwen2_engine_next(e)` — graph-replay 1-token decode step
    (cudaGraphExec_t captured in M6.3, see `qwen2_cuda.cu:1777`).
  - `qwen2_debug_replay_step` — bypasses sampler for profile.
- `tt_kv_*` layer (kvcache.h) is **already multi-slot-capable**:
  `cache_per = max_kv*max_ctx*head_dim` per slab, compact-plan + mark/restore
  primitives, no raw pointers. Built for M10 spec-decode rollback; the
  slot-dimension primitive is free.

Verdict: the engine API is the bottleneck, not the KV layer. Adding a
`slot_id` parameter to every forward path is the minimum-evolution shape.

## 1. Request lifecycle (OpenAI-style POST /v1/chat/completions)

```
HTTP parse → auth/ratelimit
  → RequestQueue (bounded MPSC, backpressure on 503)
  → Scheduler.assign(req)         // pick free slot or reject
  → ServerSlot.ingest(req)         // tokenize prompt via BPE; populate slot.prompt
  → server_loop.step()             // mixed-batch decode (see §2)
  → stream tokens via Server-Sent-Events or accumulate
  → EOS / max_tokens / stop string → slot.release() → req done
```

Per-request state (`ServerSlot` mirrors llama.cpp server-context.cpp:194):
- `id` (slot index), `state` enum (IDLE, PROCESSING_PROMPT, GENERATING, DONE, ERROR).
- `tokens` (prompt + generated), `pos_next` (== engine.pos for this slot).
- `n_predict_max`, `sampling_params` (temp/top-k/repeat-penalty/logit_bias/stops).
- `generated_text` (UTF-8 accumulator), `n_sent_text` (SSE byte cursor).
- Speculative state (M10): `spec_draft[]`, `spec_ckpt` (KVCACHE rollback mark).
- `t_last_used` (LRU eviction tiebreak), `callback_on_release`.
- Sampler instance: per-slot (mirroring llama.cpp `common_sampler` per slot).

Shared (one per server process):
- Model + arena-loaded weights (M9.3, see §4).
- Device `d_kc/d_vc` (single contiguous allocation, sliced by `slot_id`).
- `d_x/d_xn/d_q/...` activation scratch (single set — see §2.4).
- Tokenizer, chat template, `tt_kvcache` metadata struct.
- A `cudaGraphExec_t` per (ctx-bucket, slot-count) combination (see §2.5).

Per-batch-step (rebuilt every iteration, not retained):
- The set of slots in flight this step, packed into one `forward_layers`
  invocation (or N invocations for heterogeneous prompt shapes).
- A small `(slot_id, pos, sampled_tok)` tuple table for post-step routing.

## 2. Continuous batching data structure

### 2.1 Slot is the unit

Each `ServerSlot` owns one slice of the KV pool. `slot_id ∈ [0, N)`, the
state machine is the per-slot state. Pre-fill and decode are not separate
phases of the server loop — both can coexist in one batch.

llama.cpp's `update_slots()` (server-context.cpp:2677) walks slots, appends
one (or speculatively k+1) token per active slot into a `server_batch`, then
chunks it into `n_batch`-sized `llama_decode` calls. The chunking is
unavoidable in llama.cpp because `llama_batch` has a fixed cap; we have
the same ceiling (`n_ubatch`) so copy the pattern, not redesign it.

### 2.2 Minimal-evolution path from here

Today the engine knows nothing about slots. Three incremental steps, each
shippable as a "single-stream server with N=1 slot" first, then N>1:

**Step 1 — slot param threading (no perf change for N=1).**
Add `slot_id` to every internal call (`forward_layers`, `k_kv_scatter`,
`k_flash_gqa`, sampler). The KV layout doesn't move (§3); `slot_id` indexes
the outermost dimension. Replace the device-mirror `d_pos` with a
device-side array `d_pos[N]` (per-slot) and have kernels read
`d_pos[slot_id]`. M6.3 graph capture now needs to be parameterized by
`slot_id` ⇒ either (a) one graph per slot (cache-friendly, replay-any-token
property intact because pos+slot_id both come from device memory), or
(b) one "batched" graph that runs all N slots — only viable once every
kernel accepts a `slot_mask`. Start with (a); N graphs at decode-time cost
~ms each at init, replay overhead is identical.

**Step 2 — request queue + scheduler.** Decouple HTTP intake from
inference. `server_queue.h` is a port-friendly subset of llama.cpp's
`server_queue` (MPSC ring, task types: COMPLETION, NEXT_RESPONSE, METRICS,
SLOT_RELEASE, CANCEL). One consumer thread drives the inference loop;
worker threads (or libuv/io_uring) drive HTTP.

**Step 3 — batched prefill.** Today's `qwen2_engine_prefill` is one slot
running `n` prefill tokens. For N>1 with mixed prefill+decode, replace with
a per-step batch builder: collect prompt chunks from all slots currently
in PROCESSING_PROMPT, run a single batched prefill forward (T≥8 → batched
GEMM per the roadmap; the existing T>=8 dispatch rule in ops_llm.c covers
it), then sample and dispatch the first generated token of each. Decode
slots that are mid-generation run the 1-token graph in the same step.
Cost asymmetry: prefill token `i` of slot A is ~100× decode token of slot B
at the M9 perf gap; one step may run two kernel shapes back-to-back, but
the prefill amortization dwarfs the cost (see §5 of hybrid-offload-findings).

### 2.3 What's in one "step"

```
build_batch():
  for slot in slots:
    if slot.state == PROCESSING_PROMPT and slot.prompt has tokens:
      collect remaining prefill tokens into prefill_batch[slot.id]
    elif slot.state == GENERATING:
      collect 1 token (sampled last step) into decode_batch[slot.id]

  if prefill_batch not empty: run batched_prefill(...) ; sample first toks
  if decode_batch  not empty: run graph_replay_step(...) per slot_id
  for slot in slots: post_step_sample_and_emit(slot)
```

### 2.4 Activation scratch

`d_x/d_xn/d_q/...` are sized for **one token** today. Two options:

- **A. Reuse single scratch, serialize tokens.** Easiest. Each prefill
  token is one forward pass writing the same buffers. Loses batched-GEMM
  prefill speedup (M9 REV item 1).
- **B. Allocate `[N_slots][token_capacity]` scratch, run batched.** Required
  for the M9 100× prefill win. Cost: N× current scratch (B is in MB even
  for large N, e.g. dim=2048 fp32 → 8KB per token × 4096 ctx = 32MB per
  slot — too big if N×full-ctx). Compromise: prefill scratch sized to
  `max_prompt_chunk` (typical 256–1024), decode scratch N×1.

For M12 phase 0, ship A; migrate to B when batched prefill lands.

### 2.5 Graph capture shapes

CUDA graphs are shape-frozen. Decoding needs one graph per
ctx-bucket (M11 design §2.4 already plans this for Vulkan; applies to CUDA
too). For N slots with identical ctx-length bucket, we **can** share one
graph by making `slot_id` a captured dynamic value (read from device
memory before replay, the M6.3 trick generalized). Per-slot graphs are
simpler; share later. Max graphs = `n_ctx_buckets × N_slots`; with
512-token buckets and 8K ctx = 16 × N. Init cost negligible.

Spec-decode (M10) adds k+1 position graphs; same per-slot×bucket pattern.

## 3. Per-request KV cache isolation

### 3.1 Current vs target layout

Current: `Kc[layer][kv_head][slot_in_ctx][head_dim]`
— `slot_in_ctx` is the position index (`*d_pos % max_ctx` per kernel
comment qwen2_cuda.cu:290), one logical sequence.

Target: `Kc[req_id][layer][kv_head][slot_in_ctx][head_dim]`
— `req_id` is the server-slot id, dimensions are independent.

Total memory scales: `N_reqs × current`. Per-request ctx remains `max_ctx`.

### 3.2 Minimal reshape

**The K layout already has the right shape — it just needs a wrapper.**

The `tt_kvcache` layer in `kvcache.h` was designed for this. It treats
the slab as a flat element array indexed by per-layer `kv_width` stride.
The natural extension is `tt_kvcache_pool`:

```c
typedef struct {
    tt_kvcache_cfg cfg;           // single-request template
    int n_reqs;                    // = N_slots
    long slab_elems_per_req;       // 2 * n_layers * max_kv * max_ctx * head_dim
    long total_elems;              // n_reqs * slab_elems_per_req
    uint32_t *valid_len;           // [n_reqs * n_layers] (was per-layer)
    tt_kv_zero_range *zero_tail;   // [n_reqs * n_layers]
} tt_kvcache_pool;
```

Device-side: `d_kc, d_vc` are **one allocation** of `n_reqs × cache_per`
elements, and the kernel index becomes
`Kc[(req_id * n_layers + l) * cache_per + pos * kvdim + i]`. The scatter
kernel (`k_kv_scatter` at qwen2_cuda.cu:286) needs a `req_id` added to the
base offset, two lines of code.

### 3.3 What does NOT need to change

- `tt_kv_mark`, `tt_kv_restore`, `tt_kv_truncate`, `tt_kv_compact_plan_*` —
  all work per-request by passing the appropriate `valid_len` slice. The
  M10 spec-decode rollback (`tt_kv_truncate` with `emit_zero`) composes
  unchanged: rollback one slot's K/V tail without touching other slots.
- The `kvcache.h` "device-pointer agnostic, sizes/offsets only" design
  pays off here: the engine issues the same memsets and memmoves, just
  with `req_id`-scaled base offsets.

### 3.4 Attention reads

`k_flash_gqa` (single-warp-per-head) reads `Kc[l, h, 0..pos, :]`. The `pos`
cap is per-slot (`d_pos[slot_id]`), the start is 0, so the existing
"attend to 0..*d_pos" semantics transfer 1:1. **No change to flash
attention beyond the base-offset addition.** This is the single biggest
reason the reshape is "minimal".

## 4. Shared-tensor model reuse + arena loader boot

The M9.3 arena loader holds the entire model in `mmap`'d pages with
quantized weights in-place. For a server:

1. **One arena per process.** M9.3 lifetime = process lifetime. `d_*` device
   buffers are the same `cudaMalloc`-then-`cudaMemcpy` we do today, just
   for N=1 (one engine, one model) rather than per-request. The arena is
   shared implicitly by being process-global.
2. **Boot sequence:**
   ```
   server_main()
     → load GGUF via arena loader (M9.3) into mmap
     → tt_config_from_gguf()                       // existing
     → qwen2_engine_create()                       // uploads d_embd/d_out_w/d_*
     → qwen2_engine_set_sampling(0, 1, 1.0)        // server default: greedy
     → load chat template (chat_template.c)
     → start HTTP listener (POST /v1/chat/completions, /v1/completions, /health, /v1/models)
     → server_loop()                                 // blocks
   ```
3. **No reload on slot assignment.** A slot borrows the engine's KV
   partition. Only the KV and per-slot sampler state are per-slot; weights
   are pinned on device for the process lifetime. Memory cost: exactly
   `model_size` once, not N×.
4. **Multi-model future:** the M9.3 arena supports multiple `GGUFModel*`
   in one process via per-model weight regions; defer to M12 phase 3+.

## 5. Scheduling policy

Defaults for a 4 GB VRAM card serving small models (≤1B q4_0, ctx 4K):

| Policy | Choice | Rationale |
|---|---|---|
| Slot count | `N = 4` default, env `TT_N_SLOTS` overrides | q4_0 0.5B model ≈ 0.4 GB weights; KV @ 4K ctx GQA-4 = 4×80MB = 320 MB. 4 slots ≈ 1.6 GB VRAM headroom, safe on 4 GB. |
| Prefill priority | Interleave (no special priority) | A 4-slot server with 1 long + 3 short prompts pre-empts the long only if you let it; not worth the policy complexity. |
| Admission | FCFS, reject on full | Per-slot `n_predict_max` caps avoid one user hogging; FCFS is the universal default. |
| Eviction | None for v0; LRU on OOM | LRU "context-shift" only when `prompt.n_tokens() + 1 >= n_ctx`; 4 GB leaves margin, OOM rare. |
| Spec-decode | Per-slot opt-in (M10 ngram-simple) | ngram helps code/summarize, neutral on chat; gating avoids overhead for users it doesn't help. |
| Sampling | Per-slot independent samplers | Cheap (CPU, called once per step per slot); lets each user pick temp/top-k. |
| Backpressure | Bounded request queue (depth 64), 503 on full | Standard, no client fairness story in v0. |

Future: token-bucket (priority + rate) when prompt-cache hits become a
thing; that's a v1.1 problem.

## 6. API surface

llama.cpp server (`tools/server/server.cpp`) exposes the OpenAI-compatible
subset plus extras: `/completion`, `/v1/chat/completions`, `/v1/embeddings`,
`/health`, `/metrics`, `/v1/models`, `/props`, `/tokenize`, `/detokenize`,
OAI `/rerank`, plus OAI-style SSE streaming.

For a from-scratch engine M12 phase 0, the minimum-useful surface is
**4 endpoints**, all OAI-compatible JSON, all SSE-streaming:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Primary entry; messages[] → streamed tokens (SSE `data: {...}` chunks, `[DONE]` terminator) |
| `POST /v1/completions` | Legacy single-prompt form; optional, ~20 LOC to add |
| `GET /v1/models` | Returns `{data:[{id:"<gguf-name>"}]}` for client compatibility |
| `GET /health` | `200 OK` after model loaded, `503` during init/shutdown |

Stream format follows OAI: each event has `choices[0].delta.content` for
chat, `choices[0].text` for completions. Stop reasons: `length`,
`stop` (custom stop string hit), `eos` (model EOS token).

**Why not gRPC:** gRPC adds protobuf codegen, schema versioning, and
zero marginal value for the use case (low fanout, low QPS, large payloads
on the streaming path). REST + SSE is what every LLM client already
speaks. Defer gRPC until inter-service traffic shows up.

**Why not llama.cpp's `/props` or `/tokenize`:** the model is fixed at
boot, so props are static; tokenize/detokenize are CLI-facing, server users
already have tokenizers. Both are 1-day adds later.

## 7. Phases

Each phase = one coherent user-visible feature. Effort in focused
working days, calendar-realistic for solo maintainer (mirrors M11 Vulkan
estimates for consistency).

### Phase 0 — `qwen2_engine` slot_id (1–2 days)

Thread `slot_id` through the engine API. `n_slots=1` behavior bit-identical
to current. Land the `tt_kvcache_pool` wrapper. Gate: existing
`./scripts/verify.sh` suite green; tg-128 delta <1% (single-slot).

User-visible: **none** (internal). Unblocks everything else.

### Phase 1 — request queue + minimal HTTP server (3–5 days)

- Bounded MPSC queue.
- HTTP listener on libuv (use the libuv we already link in
  `examples/llm-cli` if any, else a 200-LOC raw-socket fallback).
- Single thread drives the inference loop; one or more HTTP workers.
- `POST /v1/chat/completions` streaming SSE, `GET /health`, `GET /v1/models`.
- N=1 slot (the queue may have many requests, but only one in flight at a
  time). Per-slot sampler, chat template apply, UTF-8 streaming with
  partial-byte tracking (`n_sent_text`).

Gate: `curl POST /v1/chat/completions` returns streamed OAI response;
bit-identical to `chat.py` for same prompt/temp. No regression on
`./scripts/verify.sh`.

User-visible: **a server**. Local dev can swap `chat.py` for `curl`.

### Phase 2 — continuous batching (5–8 days)

- N slots (env-tunable, default 4). Slot state machine mirrors
  llama.cpp (IDLE → PROCESSING_PROMPT → GENERATING → DONE/ERROR).
- Per-slot graph capture (one cudaGraphExec_t per slot, ctx-bucketed).
- Mixed prefill+decode step (§2.3).
- Per-slot KV pool (§3). Spec-decode integration deferred to M10's
  per-slot verify loop (sketch in specdec.h already accommodates).

Gate: `tt-server-bench` (new in this phase): 8 concurrent clients with
heterogeneous prompts, measure aggregate tok/s vs single-stream baseline.
Target: aggregate ≥ 60% of `N × single_stream_tok/s` (memory-bound
ceiling, not 100%, due to prefill/decode shape mixing). Bit-identical
outputs to single-stream at temp=0.

User-visible: **multi-client server**. Realistic shared use.

### Phase 3 — speculative decoding integration (4–6 days, depends on M10)

- Per-slot ngram-simple drafter (M10 already designed).
- Verify loop: batched forward over (last_committed, k drafts) per slot,
  per-slot accept, per-slot KV rollback via `tt_kv_truncate` + memset tail.
- Greedy only (rejection sampling for temp>0 is M10 item 5, later).
- Opt-in via request param `speculative: true` (default off per
  ngram-neutral-on-chat).

Gate: code-edit prompts (HumanEval-style single-line completion) at
2× single-stream tok/s; chat at parity with non-spec baseline.

User-visible: **faster code/summarize workloads**, no API change.

### Phase 4 — robustness + ops surface (3–5 days)

- LRU context-shift on per-slot ctx overflow (mirror
  `slot.prompt.n_tokens() + 1 >= slot.n_ctx` branch in server-context.cpp:2797).
- Bounded memory: refuse new requests with `n_predict` that would OOM.
- `/metrics` (Prometheus text format): in-flight slots, kv_used_bytes,
  prefill_tok_s, decode_tok_s, accept_rate (phase 3+).
- Cancellation: `slot.release()` on client disconnect; `slot.prompt_clear`
  calls `mem.seq_rm` for that slot's `id`.
- Graceful shutdown: drain queue, finish in-flight, free engine.

Gate: 1000-request soak with random prompts, no leaks (KV buf stable),
no hangs, p99 stream-first-byte latency <500 ms on a 0.5B q4_0 model.

User-visible: **production-ready**. Operable, observable, won't OOM-crash.

**Total: 16–26 focused days (~4–7 weeks part-time). Phase 0+1 alone is
~1 week; that's the smallest "I can curl it" milestone.**

## 8. Risks ranked

1. **Graph capture explosion (high likelihood, medium impact).** N slots ×
   ctx-buckets × (k+1) spec-shapes = combinatorial blowup. Mitigation:
   lazy capture on first-seen shape, LRU cap on captured graphs (e.g.
   64 total), use `cudaGraphInstantiateFlagAutoFreeOnLaunch` for
   transient ones. Measure init time + RSS delta in phase 2 gate.

2. **Per-slot scratch bloat (high likelihood, high impact at high N).**
   §2.4: scratch sized to N × full ctx is infeasible for ctx≥4K. Mitigation:
   per-slot scratch sized to (1 decode + max_prompt_chunk prefill) tokens;
   share the Q/K/V staging buffers across slots (read-only during a single
   step, so safe to alias). Phase 2 explicit memory budget assertion.

3. **Mixed-shape step overhead (medium likelihood, medium impact).**
   A batch with 1 prefill@2048 + 4 decode@1 may split into two kernel
   sequences (or worse, two `forward_layers` calls). Mitigation: defer the
   prefill slot to the next step if it would more than double the step
   time; vLLM/llama.cpp do the same ("prefill chunking"). Add a
   per-slot prefill-bucket counter; first-seen bucket = capture graph
   for it.

4. **GGUF prefill latency variance for large prompts (medium likelihood,
   high impact on UX).** Even with batched prefill, a 4K-token prompt
   takes 200–500 ms on a 0.5B model; clients may not see first token for
   half a second. Mitigation: stream the first token of each prefill slot
   as soon as it's sampled, don't wait for the whole prompt to land
   (matches llama.cpp's "chunked prefill" if we need it). Phase 1 baseline
   is honest about TTFT, phase 4 surfaces it in `/metrics`.

5. **KV eviction under memory pressure (low likelihood at 4 GB / N=4,
   high impact when it hits).** A misbehaving client streaming forever
   would OOM the KV pool. Mitigation: `n_predict_max` enforced strictly;
   if pool >90% used, oldest generating slot is LRU-evicted (return
   `stop_reason: "slot_evicted"` to client). Phase 4 deliverable.

6. **Isolation between concurrent requests (low likelihood, very high
   impact).** Sampler state, KV writes, draft ring buffer — all must be
   per-slot, not shared. Class of bug: forgetting to reset a per-slot
   sampler on slot reuse, leading to "weird continuation tokens" from a
   previous tenant. Mitigation: per-slot struct with explicit
   constructor/reset, no global sampler, unit tests that run 2 slots
   with disjoint prompts and assert no cross-contamination. Phase 2
   gate includes a "tenant isolation" fuzz (10 random prompt pairs,
   assert each slot's KV after N tokens matches a single-stream reference).

7. **Spec-decode per-slot rollback bugs (medium likelihood in phase 3,
   very high impact: silent corruption).** `tt_kv_truncate` with
   `emit_zero=1` + the kernel `vkCmdFillBuffer` analog must zero exactly
   the rejected-draft tail for that slot, not bleed into other slots.
   Mitigation: bitmap-based zero ranges verified per-slot; phase 3 gate
   includes bit-identical-vs-greedy check for all-draft-rejected,
   half-accepted, all-accepted cases. Same risk as M10 standalone, but
   compounded by N>1.

## 9. Open questions

- **Cross-slot prefill chunking**: do we chunk one long prompt across
  multiple decode steps, or always finish prefill in one step? llama.cpp
  has `n_ubatch`; we have the same ceiling. Default: finish in one step
  up to `n_ubatch`, spill to multiple steps beyond.
- **Paged KV vs contiguous-per-slot**: vLLM uses paged attention to
  eliminate fragmentation under heterogeneous ctx lengths. Worth it for
  us? Probably not at N=4–8 with small models; defer to v1.1.
- **Multi-model serving**: phase 4+ candidate, requires per-model
  engine instances. Arena loader already supports this; the question is
  shared-vs-dedicated samplers and KV pools.
- **HTTP framework**: raw libevhtp-like? cpp-httplib (header-only, MIT,
  ~3 KLOC, no deps)? Pick at phase 1 start; we have ~1 day budget for
  HTTP plumbing total.
