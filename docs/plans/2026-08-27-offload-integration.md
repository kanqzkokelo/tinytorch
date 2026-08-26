# Hybrid CPU+GPU offload — integration plan (2026-08-27)

Status: SPEC (no code written). Builds on `2026-08-27-hybrid-offload-findings.md`
(placement design, PCIe math, E4B verdict) and `2026-08-27-moe-notes.md`.
Scope: dense-model single-boundary split first (`--layer-split N`), tensor-class
override second (`--cpu-moe` equivalent lands with MoE engine work, not here).

Goal: run E4B-class models that exceed VRAM by keeping a contiguous tail of
decoder layers on CPU-mmap, executing them via `tt_cpu_gemv` + host helpers,
with one pinned activation shuttle per GPU↔CPU boundary.

---

## 1. Placement table

### Struct (new: `include/offload.h`, `src/offload.c`)

```c
typedef enum { PL_GPU, PL_CPU_MMAP } tt_placement;

typedef struct {
    char     pattern[128];    /* tensor-class regex, llama.cpp -ot syntax subset */
    tt_placement place;
} tt_override;

typedef struct {
    int          n_layers;
    int          split_layer;   /* -1 = none; layers >= N are CPU */
    tt_placement tok_embd;      /* default GPU (embed is one row-read/token) */
    tt_placement lm_head;       /* default GPU (vocab GEMV stays on device) */
    tt_placement final_norm;    /* follows lm_head side */
    tt_placement attn[MAX_LAYERS];
    tt_placement ffn[MAX_LAYERS];   /* enables future --cpu-moe granularity */
    tt_override  overrides[16]; /* applied last, first-match wins */
    int          n_overrides;
} tt_offload_plan;

/* Resolve plan from CLI: --layer-split N sets split_layer + fills arrays;
 * --ot 'pat=CPU' appends overrides. Returns 0/-1. */
int  tt_offload_plan_build(const TTConfig *cfg, int layer_split,
                           const char *ot_spec, tt_offload_plan *out);
/* Classify a tensor name: "blk.12.ffn_up.weight" -> PL_CPU_MMAP etc.
 * Regex subset: prefix match on "blk.N." with N=* wildcard, suffix class
 * (attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down|token_embd|
 * output|...). Full PCRE unnecessary for v1. */
tt_placement tt_offload_classify(const tt_offload_plan *p,
                                 const char *tensor_name, int layer);
```

### Loader-phase assignment logic

Hook point A — `kernels/qwen2_cuda.cu` **engine create / weight upload**:
`qwen2_engine_create()` at **:588**, per-layer `upload_w()` calls at
**:652–658** (q/k/v/o/gate/up/down), embedding at **:631**, output head +
norm at **:635/:648**.

Change: wrap every `upload_w` in placement check.

```c
/* pseudo-diff, qwen2_cuda.cu :652 */
- snprintf(name,...,"blk.%d.attn_q.weight", l); upload_w(m, name, &w->q);
+ snprintf(name,...,"blk.%d.attn_q.weight", l);
+ if (tt_offload_classify(&e->offload, name, l) == PL_GPU)
+     upload_w(m, name, &w->q);              /* cudaMalloc+memcpy as today */
+ else
+     keep_host(m, name, &w->h_q);           /* NEW: no cudaMalloc; retain the
+                                               mmap'd GGUF host pointer as-is.
+                                               Zero-copy: decode touches every
+                                               weight every token, so page
+                                               faults amortize (findings doc).
+                                               dtype recorded like TTensor. */
```

New engine field: `LayerW` gains host twins `h_q..h_down` (`TTensor`-shaped:
ptr = GGUF host pointer, dtype = gguf type) plus per-layer flag
`int cpu_resident`. Plan built once before the upload loop:
`tt_offload_plan_build(cfg, cfg->cpu_layer_split, getenv("TT_OT"), &e->offload)`.

Presets:
- `--layer-split N` (env `TT_LAYER_SPLIT`): layers `[N, n_layers)` fully CPU
  (attn+ffn), everything else GPU. This is the v1 path and the only measured one.
- `--cpu-moe` equivalent: NOT in this spec — needs MoE engine (`pl_moe`
  arrays, moe-notes). Recorded as follow-up; struct already carries `ffn[]`
  so the override composes later without ABI churn.
- `-ot` regex passthrough: v1 supports class-suffix patterns only (see
  classify contract above).

### Presets summary

| Flag | Meaning | Maps to |
|---|---|---|
| `--layer-split N` | contiguous tail on CPU | `split_layer=N`; attn/ffn[l>=N]=CPU |
| `--ot 'ffn_*=CPU'` | class override | `overrides[]`, first match wins |

---

## 2. Forward-loop changes (`forward_layers`, qwen2_cuda.cu)

Hook point B — **per-layer loop head**: `forward_layers()` starts at **:1008**;
KV remap `cache_layer`/`Kl_f`/`Vf_l` at **:1011–1037**; stage-1 rmsnorm at
**:1049–1052**. The CPU-resident branch replaces stages 1–8 (rmsnorm → MLP
residual add), i.e. lines **~1051–1234**, and rejoins at the PLE block
(:1259+) which stays untouched.

### Pinned-buffer lifecycle

One pair of pinned staging buffers sized `dim * sizeof(float)` (+ one
`max_ffn` scratch for gate/up intermediates), allocated in engine create next
to the existing `cudaHostAlloc(&e->h_sampled, ...)` at **:866**:

```c
cudaHostAlloc(&e->h_x_in,  dim * sizeof(float), cudaHostAllocDefault);
cudaHostAlloc(&e->h_x_out, dim * sizeof(float), cudaHostAllocDefault);
cudaHostAlloc(&e->h_ffn,   max_ffn * sizeof(float), cudaHostAllocDefault); /* gate/up scratch */
freed in qwen2_engine_destroy() beside other cudaFree/cudaHostFree calls (:878+)
```

Lifecycle rule: allocate once at engine create (never per-token); write only
inside the CPU branch; synchronize the engine stream BEFORE reading
`h_x_in` and AFTER writing `h_x_out` (see pseudo-code). No async H2D/D2H of
the shuttle — the transfer is 3KB; the sync event, not bandwidth, dominates
(findings doc: <1µs PCIe vs 20–50µs sync), so a blocking
`cudaMemcpy` + explicit `cudaStreamSynchronize` pair per boundary is correct
and simplest. Two boundaries per token for contiguous split.

### Branch pseudo-code

```c
/* pseudo-diff, inside for(l...) after kv_shared remap (:1037), before :1049 */
+ const int cpu_l = w->cpu_resident;
+ if (cpu_l && !e->has_pl_embd) {   /* gemma4 PLE layers: GPU-only in v1 */
+     /* ---- D2H boundary: residual stream owner handoff GPU -> CPU ---- */
+     cudaStreamSynchronize(e->stream);            /* drain prior kernels */
+     cudaMemcpy(e->h_x_in, e->d_x, c->dim*4, cudaMemcpyDeviceToHost);
+
+     /* attention half (host): xn = rmsnorm(x)*attn_norm */
+     h_rmsnorm(e->h_x_in, w->h_attn_norm_host /*upload_f32 twin*/, e->h_xn,
+               c->dim, c->rms_eps, c->tr.norm_offset);
+     /* q/k/v/o projections reuse cpu_backend verbatim: same quant block
+        layout as device (byte-identical per cpu_backend.h header). */
+     tt_cpu_gemv(w->h_q.ptr,     w->h_q.dtype,     e->h_xn, e->h_q_out,
+                 attn_qout, c->dim, e->n_threads);          /* +bias via h_add */
+     tt_cpu_gemv(w->h_k.ptr,     w->h_k.dtype,     e->h_xn, e->h_k_stage,
+                 kvdim_l,   c->dim, e->n_threads);
+     tt_cpu_gemv(w->h_v.ptr,     w->h_v.dtype,     e->h_xn, e->h_v_stage,
+                 kvdim_l,   c->dim, e->n_threads);
+     rope_host(...);            /* mirror k_rope/k_rope_gptj math exactly */
+     attention_host(...);       /* see KV residency, section 4 */
+     tt_cpu_gemv(w->h_o.ptr,    w->h_o.dtype,    e->h_att, e->h_xn,
+                 c->dim, attn_qout, e->n_threads);
+     h_add(e->h_x_in, e->h_xn, c->dim);           /* x += o@Wo^T */
+
+     /* FFN half (host) */
+     h_rmsnorm(e->h_x_in, ..., e->h_xn, ...);
+     tt_cpu_gemv(w->h_gate.ptr, ..., e->h_g, FF_l, c->dim, n_threads);
+     tt_cpu_gemv(w->h_up.ptr,   ..., e->h_u, FF_l, c->dim, n_threads);
+     h_swiglu_gelu(e->h_g, e->h_u, e->h_h, FF_l, act_gelu); /* silu|gelu trait */
+     tt_cpu_gemv(w->h_down.ptr, ..., e->h_h, e->h_xn, c->dim, FF_l, n_threads);
+     h_add(e->h_x_in, e->h_xn, c->dim);
+
+     /* ---- H2D boundary: owner handoff CPU -> GPU ---- */
+     cudaMemcpy(e->d_x, e->h_x_in, c->dim*4, cudaMemcpyHostToDevice);
+     continue;   /* skip GPU stage bodies 1-8; PLE block still skipped (v1) */
+ }
```

NOT ported: no new CUDA-side anything; no reuse of device kernels. The task
brief's "reuse cpu_backend GEMV" is honored — attention/rope/swiglu/rmsnorm
are NEW ~100-line host helpers (section 3), because cpu_backend covers GEMV
only.

Attention-on-CPU note: scores over slots [t0..pos] against K/V slabs. For v1
the CPU layer's K/V slabs live in HOST memory (malloc'd mirrors of the
device slab layout, section 4) — attention reads them directly, zero PCIe KV
traffic. Scatter writes land in host slabs; flash kernel never sees them.

---

## 3. Host-side op helpers (new: `src/host_ops.h`/`.c`, ~100 LOC total)

| Helper | Signature sketch | Mirrors |
|---|---|---|
| `h_rmsnorm(x, g, y, dim, eps, woff)` | fp32, two-pass mean-square | `k_rmsnorm` :96 incl. gemma `woff` branch |
| `h_swiglu_gelu(g, u, h, n, act)` | silu or tanh-gelu select | `k_swiglu_apply` :437 (act trait identical formula) |
| `h_add(dst, src, n)` | elementwise | `k_add` |
| `rope_host(q, heads, hd, pos, base, style)` | NEOX half-split + GPT-J pair variants | `k_rope` :126 / `k_rope_gptj` |
| `attention_host(q, K, V, out, pos, t0, H, KV, HDl, scale, swa)` | softmax(QK^T/sqrt)V, causal, SWA window | `k_flash_gqa` :237 masking rules |

Parity contract: each helper must be bit-comparable to its device twin on
same input within f32 tolerance (test plan §6 uses rms-diff < 1e-5 gates, the
existing eng_rms trace tooling at :1181+ doubles as the comparator).

Norm-offset/gamma handling: `h_rmsnorm` takes `woff` explicitly — do NOT
duplicate the `(1+w)` baking logic; converted gemma GGUFs have it baked
(k_rmsnorm comment :96-99), raw HF conversions pass `tr.norm_offset`. Single
source of truth = `c->tr.norm_offset` passed straight through, same as the
five existing `k_rmsnorm` call sites (:1051, :1174, :1189, :1222, :1329).

---

## 4. KV residency rules (per findings doc: "KV allocated per-device beside
its attention op — never migrates")

- Layer CPU-resident ⇒ its K/V slabs are HOST buffers (`malloc`, layout
  identical: `Kl = h_kc + l*cache_layer`, stride formula from :1011–1018
  reused verbatim so `kvcache.c` plans stay valid — that layer tracks sizes/
  offsets only, deliberately pointer-agnostic per kvcache.h DESIGN note).
- Layer GPU-resident ⇒ slabs stay in `e->d_kc/e->d_vc` as today.
- No cross-device KV access ever: attention for layer l runs on whatever
  device holds layer l's slabs. Boundary crossings carry only the residual
  stream (3KB), never KV.
- Rollback/session: `tt_kv_restore` / `tt_kv_truncate` operate on lengths +
  offset plans, so rejected-tail zeroing must be executed twice — memset on
  device ranges for GPU layers, memset on host ranges for CPU layers — driven
  by the SAME `tt_kv_zero_range[]` output (specdec.h contract item 2 holds
  unchanged; consumers just dispatch per-layer by residency flag).
- SWA compaction: `tt_kv_move` memmoves execute on the owning device's slab.
- Session save/restore blob: one file, sections ordered GPU-layers then
  CPU-layers; layout table already records per-layer locs (`tt_kv_layout`).

---

## 5. Prefill-on-CPU via f32 GEMM — CONDITIONAL / PENDING

Do NOT implement now. Gate: a host f32 blocked GEMM (or OpenBLAS link) must
measure ≥0.9× vs the current all-GPU prefill on E-series shapes before this
section activates. Rationale: prefill is compute-bound; q4_0 CPU GEMV wins
only when VRAM cannot hold the layers at all, and even then prompt processing
on CPU may dominate wall time. Findings doc priority order puts this below
E2B parity + M9 perf.

If it lands: batch the pinned shuttle to `batch*dim` f32 rows, replace
`tt_cpu_gemv` chain with `sgemm(A[m,k], X[k,b])` per projection, reuse the
same boundary logic with `pos += b`. Est. +150 LOC + optional `-lopenblas`.

---

## 6. Phases, effort, test plan

| Phase | Content | Effort |
|---|---|---|
| P1 | offload.h/.c plan builder + classify + engine-create wiring (hook A), host-twin Tensors, `keep_host()` | 0.5–1 day |
| P2 | host_ops helpers (§3) + unit parity tests vs device twins (dump inputs via TT_DUMP_X-style hooks, compare) | 1 day |
| P3 | forward_layers CPU branch + pinned lifecycle (hook B), host KV slabs + scatter/attention-host | 1–2 days |
| P4 | rollback/session dual-device execution (zero-range dispatch), SWA compact on host slabs | 0.5 day |
| P5 | E2B split sweep benchmark + parity harness; docs | 0.5 day |

Total ≈ 3.5–5 days single dev. Priority per findings doc: BELOW E2B parity
and M9 perf; build only if E4B demand persists.

Test plan:
1. **Parity sweep**: fixed prompt set, greedy, `--layer-split N` for N ∈
   {0(all-GPU control), mid, n_layers-1}; require argmax token sequence
   identical across all splits on E2B model (f32 KV both sides → deterministic).
   Compare logits rms via existing `eng_rms`/`qwen2_debug_copy_logits` (:1821).
2. **Helper parity**: randomized x/g vectors, h_rmsnorm/h_swiglu_gelu/
   attention_host vs device kernel outputs, max-abs-diff < 1e-5.
3. **Rollback under hybrid**: force specdec draft rejection at a CPU layer
   boundary; verify post-rollback logits match a fresh shorter-context run.
4. **Perf**: tokens/s vs N sweep on RTX 3050 4GB + DDR4-3200; expect the
   findings-doc curve (5–9 t/s band for E4B split); record best-N.
5. **Page-fault sanity**: cold start first-token latency reported separately;
   confirm steady-state matches warm (mmap faults amortize after pass 1).

---

## 7. Risks

1. **Stream sync stalls** — two blocking syncs/token at boundaries; 20–50µs
   each → ≤100µs/token overhead, fine vs ~150–200ms/token at target t/s, but
   fatal if someone later adds per-layer alternation (findings: ~500 t/s
   ceiling collapse). Mitigation: enforce contiguous-split-only invariant in
   plan builder (reject non-contiguous override sets in v1 with clear error).
2. **mmap page faults during decode** — first pass over CPU weights faults
   every page; cold first token could be seconds. Mitigation: optional warmup
   pass (touch weights at load, mlock if RSS budget allows); measure cold vs
   warm in test 5. Fault storms during concurrent GC/malloc pressure remain
   possible — accept for v1, document.
3. **norm_offset/gamma duplication** — five device call sites + new host
   helper risk drift (gemma woff semantics). Mitigation: h_rmsnorm is the
   single host implementation; add a static_assert-style runtime check that
   `tr.norm_offset==0` unless family is gemma (matches device behavior).
4. **Attention host port correctness** — SWA windowing + GQA repeat-interleave
   subtleties; highest-bug-density item. Mitigation: helper parity test 2
   includes SWA edge cases (window == valid_len, window crossing slot 0).
5. **Graph capture interplay** — blocking memcpy illegal inside captured
   region (precedent: g_capturing guard at :1381). CPU-layer branch must sit
   OUTSIDE any capture; guard with same `!g_capturing` check.
6. **dtype coverage gap** — cpu_backend supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K
   only; F16/F32/BF16 weights on CPU layers need either dequant-at-load or
   host GEMV extensions. Mitigation: plan builder rejects unsupported dtypes
   on CPU-resident tensors with actionable error.

## Open questions

- Q1: does E4B demand actually persist after M9 perf? (go/no-go gate before P1)
- Q2: host KV slabs malloc'd vs part of a single mmap arena (session-restore
  simplicity favors one arena; defer to P4).
- Q3: `--ot` full regex vs suffix-subset sufficient for v1? (llama.cpp users
  expect real regex; subset documented clearly if chosen)
- Q4: thread count policy — reuse chat loop's n_threads or dedicated env?
