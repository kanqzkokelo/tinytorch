# M6.3 Performance Engineering Implementation Plan

> Execution note: implement this task-by-task.

**Goal:** Take the correct tinytorch Qwen2 decode engine from 48 tok/s to ≥100 tok/s decode-only on the RTX 3050 laptop, without ever breaking greedy-output parity against the llama.cpp oracle.

**Architecture:** Three stacked optimizations, each independently verified: (1) a locked-in benchmark harness with an anti-fake validity assertion, (2) CUDA Graph replay of the whole decode step with position driven by a device scalar, (3) targeted kernel fusion and GEMV bandwidth work. Every optimization phase ends by re-running the logit-parity gate; any gate failure reverts the change before moving on.

**Tech Stack:** CUDA 12 (sm_86), C11 engine (`kernels/qwen2_cuda.cu`, `kernels/gemv_q4_cuda.cu`), Python3 + numpy for gates/harnesses, pinned llama.cpp build at `oracle/llama.cpp` as ground truth.

**Working branch:** `m6-correctness` (already checked out; do NOT create a worktree — the oracle directory and built binaries live in this checkout and are gitignored).

**Baseline numbers (measured 2026-08-23, record in every commit message):**
- Decode-only: **48 tok/s** (eager, ~170 kernel launches/token, legacy-stream barriers removed)
- llama.cpp reference on this box: ~58 tok/s tg (short ctx)
- Memory floor: ~400 MB weights/token ÷ 176 GB/s ≈ 2.5 ms/token ⇒ ceiling ≈ 400 tok/s

---

## Task 0: Lock in fixtures + formal parity gates (do first — everything else verifies against these)

The ad-hoc parity tooling lives in `/tmp` and dies on reboot. Make it permanent and wired into `verify.sh`.

**Files:**
- Create: `tests/fixtures/gen_fixtures.py`
- Create: `tests/fixtures/parity_set.json` (generated, committed)
- Create: `tests/fixtures/oracle_lg_*.bin` (generated, committed — 7 × 592 KB, fine for git)
- Modify: `tools/dump_logits.c` (no code change; it is already correct)
- Modify: `scripts/verify.sh`
- Create: `tests/gate_m6_logit_parity.py`

**Step 1: Fixture generator**

Write `tests/fixtures/gen_fixtures.py`. It must tokenize each prompt with `llama-tokenize`, dump oracle logits with `/tmp/oracle_logits` (move that source into `tools/oracle_logits.c` first so it survives), and write JSON:

```python
#!/usr/bin/env python3
"""Generate committed parity fixtures. Requires oracle/llama.cpp build."""
import subprocess, os, json, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
BIN = os.path.join(ROOT, "oracle/llama.cpp/build/bin")
MODEL = os.path.join(ROOT, "data/models/qwen2.5-0.5b-instruct-q4_0.gguf")
env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(BIN))
PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "My name is Maria. I like to eat",
    "The three primary colors are red,",
    "Once upon a time in a distant kingdom",
    "The largest planet in the solar system is",
    "Photosynthesis is the process by which",
]
out = []
for i, p in enumerate(PROMPTS):
    r = subprocess.run([f"{BIN}/llama-tokenize", "-m", MODEL, "-p", p],
                       capture_output=True, text=True, env=env)
    import re
    ids = [int(m.group(1)) for line in r.stdout.splitlines()
           if (m := re.match(r"\s*(\d+)\s*->", line))]
    assert ids, f"tokenize failed for {p!r}"
    ol = os.path.join(ROOT, "build/oracle_logits")
    r2 = subprocess.run([ol, MODEL, p, "--dump", f"tests/fixtures/oracle_lg_{i}.bin"],
                        capture_output=True, text=True, env=env)
    top8 = [l for l in r2.stdout.splitlines() if l.startswith("TOP8")][0]
    m = re.search(r"\((\d+),([\d.]+)\)", top8)
    out.append({"prompt": p, "tokens": ids,
                "oracle_argmax": int(m.group(1)), "oracle_val": float(m.group(2))})
json.dump(out, open("tests/fixtures/parity_set.json", "w"), indent=1)
print(f"wrote {len(out)} fixtures")
```

**Step 2: Move the oracle tool into the repo**

Copy `/tmp/oracle_logits.c` to `tools/oracle_logits.c` unchanged. Add a Makefile target:

```makefile
$(BUILD)/oracle_logits: tools/oracle_logits.c | $(BUILD)
	$(CC) -O2 -I $(HOME)/Storage/repos/nnfromscratch/oracle/llama.cpp/include \
	  -I $(HOME)/Storage/repos/nnfromscratch/oracle/llama.cpp/ggml/src \
	  -o $@ $< -L $(HOME)/Storage/repos/nnfromscratch/oracle/llama.cpp/build/bin -lllama \
	  -Wl,-rpath,$(CURDIR)/oracle/llama.cpp/build/bin
```

(If the deprecated-API warnings annoy you, add `-Wno-deprecated-declarations`.)

Build it: `make build/oracle_logits` — expected: binary appears, no errors.

**Step 3: Generate and inspect fixtures**

Run: `mkdir -p tests/fixtures && python3 tests/fixtures/gen_fixtures.py`
Expected: `wrote 7 fixtures`; `ls tests/fixtures/` shows `parity_set.json` + 7 `.bin` files (~592 KB each).

Sanity-check one: `python3 -c "import json; d=json.load(open('tests/fixtures/parity_set.json')); print(d[0]['tokens'], d[0]['oracle_argmax'])"`
Expected: `[785, 6722, 315, 9625, 374] 12095`

**Step 4: The logit-parity gate**

Write `tests/gate_m6_logit_parity.py`:

```python
#!/usr/bin/env python3
"""Gate Q2: teacher-forced logits parity vs oracle fixtures.
Pass requires, per prompt: top-1 match AND argmax-delta <= 0.35
AND median|dlogit| <= 0.15 over >= 85% of prompts."""
import json, subprocess, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FIX = os.path.join(ROOT, "tests/fixtures")
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    env.get("LD_LIBRARY_PATH", "")]))

data = json.load(open(f"{FIX}/parity_set.json"))
npass = 0
for i, d in enumerate(data):
    ids = ",".join(map(str, d["tokens"]))
    r = subprocess.run(["build/dump_logits", ids, f"/tmp/ours_lg_{i}.bin"],
                       capture_output=True, text=True, timeout=300, env=env)
    am_line = next(l for l in r.stdout.splitlines() if l.startswith("ARGMAX"))
    parts = am_line.split()
    ours_am, ours_v = int(parts[1]), float(parts[2])
    ref = np.fromfile(f"{FIX}/oracle_lg_{i}.bin", dtype="<f4")
    ours = np.fromfile(f"/tmp/ours_lg_{i}.bin", dtype="<f4")[:len(ref)]
    med = float(np.median(np.abs(ours - ref)))
    am_d = abs(ours_v - d["oracle_val"])
    top1_ok = ours_am == d["oracle_argmax"]
    ok = top1_ok and am_d <= 0.35 and med <= 0.15
    npass += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {d['prompt'][:40]!r:44s} "
          f"top1={'Y' if top1_ok else 'N'} argmax_d={am_d:.3f} median={med:.4f}")
frac = npass / len(data)
print(f"\nGate Q2-lite: {npass}/{len(data)} (need >= 0.85)")
sys.exit(0 if frac >= 0.85 else 1)
```

**Step 5: Wire into verify.sh**

In `scripts/verify.sh`, the m61 case already exists from earlier edits; confirm it reads:

```bash
  m61) run "m6-logits-parity" python3 tests/gate_m6_logit_parity.py ;;
```

(replace the older `gate_m6_parity.py --min-agree 0.7` line if present — the text gate had parsing fragility; the logits gate supersedes it. Keep `gate_m6_parity.py` around as a manual sanity tool.)

**Step 6: Run the gate**

Run: `./scripts/verify.sh m61`
Expected: `7/7 PASS`, exit 0. If any prompt fails here, STOP — fix correctness before any perf work.

**Step 7: Commit**

```bash
git add tests/fixtures tools/oracle_logits.c tools/dump_logits.c tests/gate_m6_logit_parity.py scripts/verify.sh Makefile
git commit -m "M6-gates: committed parity fixtures + logit-parity gate (verify.sh m61)"
```

---

## Task 1: Anti-fake benchmark harness (bench_llm.py)

Every future speed claim must come from this script. It times decode-only throughput AND re-validates output quality in the same process.

**Files:**
- Create: `bench/bench_llm.py`
- Modify: `examples/run_llm_gpu.c` (add machine-readable stats line)

**Step 1: Stats line in run_llm_gpu**

Modify the final printf in `examples/run_llm_gpu.c` to emit a parseable line (keep human lines too):

```c
printf("\"\\n[gen: %d tokens | decode %.1f tok/s | incl prefill %.1f tok/s | greedy]\\n",
       gen_count, gen_count / dec, gen_count / tot);
printf("STATS tokens=%d prefill=%d decode_us=%.0f\\n",
       gen_count, n_prompt,
       (unsigned long)(dec * 1e6));
```

**Step 2: The harness**

Write `bench/bench_llm.py`: runs `build/run_llm_gpu <prompt> <tokens>` N=7 times (1 warmup discarded), parses `STATS` lines, reports median decode tok/s, then runs the parity assertion (reuse `tests/gate_m6_logit_parity.py` logic via import or subprocess) and exits nonzero if parity fails:

```python
#!/usr/bin/env python3
"""Anti-fake LLM benchmark: median-of-N decode throughput + mandatory parity check.
Usage: bench_llm.py [--runs 7] [--tokens 128] [--prompt TEXT]"""
import subprocess, os, sys, statistics, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=7)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--prompt", default="Explain quantum computing in one sentence.")
args = ap.parse_args()

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    env.get("LD_LIBRARY_PATH", "")]))

rates = []
for run in range(args.runs):
    r = subprocess.run(["build/run_llm_gpu", args.prompt, str(args.tokens)],
                       capture_output=True, text=True, timeout=600, env=env)
    stats = [l for l in r.stdout.splitlines() if l.startswith("STATS")]
    assert stats, f"run {run}: no STATS line\n{r.stdout[-500:]}"
    t = int(stats[0].split("decode_us=")[1])
    rates.append(args.tokens / (t / 1e6))
med = statistics.median(rates)
print(f"decode: median {med:.1f} tok/s over {args.runs} runs "
      f"(min {min(rates):.1f}, max {max(rates):.1f})")

# ANTI-FAKE: throughput only counts if parity still holds
g = subprocess.run([sys.executable, "tests/gate_m6_logit_parity.py"],
                   capture_output=True, text=True, timeout=1200, env=env)
print(g.stdout.strip().splitlines()[-1] if g.stdout else g.stderr[-200:])
sys.exit(0 if (g.returncode == 0) else 1)
```

**Step 3: Baseline measurement**

Run: `python3 bench/bench_llm.py`
Expected: `median ~45-50 tok/s` and `Gate Q2-lite: 7/7`. Record the number in `bench/results.md` under a new heading `## LLM decode (M6.3 optimization ladder)` with date/GPU/clock context.

**Step 4: Commit**

```bash
git add bench/bench_llm.py examples/run_llm_gpu.c bench/results.md
git commit -m "M6.3: anti-fake bench harness; eager baseline ~XX tok/s recorded"
```

---

## Task 2: Device-driven position (prerequisite for graphs)

CUDA Graphs bake in scalar kernel arguments. `pos` is passed by value today; move it to device memory so a captured graph reads the current position at replay time.

**Files:**
- Modify: `kernels/qwen2_cuda.cu` (engine struct + kernels + step functions)

**Step 1: Add `d_pos` and switch kernels to pointer-based position**

Engine struct gains `int *d_pos;`. Allocate in `qwen2_engine_create` (`cudaMalloc(&e->d_pos, sizeof(int))`, init to 0 via `cudaMemsetAsync`). Free it in `qwen2_engine_free`.

Change `k_rope` to take `const int *__restrict__ d_pos` and read `const int pos = *d_pos;` as its first statement. Change `k_flash_gqa` likewise (loop bound `t <= pos`, slot math uses `pos`). Update both launch sites in `forward_layers` to pass `e->d_pos` instead of `e->pos`.

**Step 2: Slot addressing moves device-side**

K/V GEMV outputs can no longer go straight into the ring-buffer slot (host no longer knows the slot). Add staging buffers `d_k_stage`, `d_v_stage` (each `n_kv_heads*head_dim` floats) and a scatter kernel:

```cuda
__global__ void k_kv_scatter(const float *__restrict__ kst,
                             const float *__restrict__ vst,
                             float *__restrict__ Kc, float *__restrict__ Vc,
                             const int *__restrict__ d_pos,
                             int n_kv_heads, int head_dim, int max_ctx) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int kvdim = n_kv_heads * head_dim;
    if (i >= kvdim) return;
    const int slot = (*d_pos) % max_ctx;
    Kc[(long)slot * kvdim + i] = kst[i];
    Vc[(long)slot * kvdim + i] = vst[i];
}
```

In `forward_layers`: GEMVs for k/v write to `d_k_stage`/`d_v_stage`; RoPE-K applies to `d_k_stage`; then `k_kv_scatter` copies both into the cache. Flash attention is unchanged except the position read.

**Step 3: advance() updates device position, not host-only**

```c
static int advance(Qwen2Engine *e, int tok) {
    int rc = embed_token(e, tok);            /* embed BEFORE pos++ : embedding is position-independent */
    if (rc) return rc;
    rc = forward_layers(e);
    if (rc) return rc;
    e->pos++;
    cudaMemcpyAsync(e->d_pos, &e->pos, sizeof(int), cudaMemcpyHostToDevice, e->stream);
    return 0;
}
```

Note the ordering subtlety: `forward_layers` must run while `*d_pos` still holds the CURRENT slot, and the increment lands after. Initialize `d_pos` to 0 at create. In `qwen2_engine_prefill`, also push `e->pos` to `d_pos` once at entry (in case a previous session left it stale).

**Step 4: Verify parity unchanged**

Rebuild: `make run_llm_gpu chat_llm_gpu build/dump_logits`
Run: `./scripts/verify.sh m61`
Expected: PASS 7/7. If FAIL, the pos-ordering above is wrong — debug before proceeding.

**Step 5: Measure + commit**

Run: `python3 bench/bench_llm.py` (expect roughly unchanged, ±10%).
Append result line to `bench/results.md`. Commit:
```bash
git add -A && git commit -m "M6.3: device-scalar position (graph-readiness); parity intact"
```

---

## Task 3: CUDA Graph replay of the decode step

**Files:**
- Modify: `include/qwen2_engine.h` (new API)
- Modify: `kernels/qwen2_cuda.cu`

**Step 1: Graph lifecycle API**

Header additions:

```c
/* Capture the whole decode step (embed->layers->norm->logits->argmax) into a
 * CUDA graph. Requires TTConfig.max_ctx-bounded KV ring. Returns 0 on success. */
int qwen2_engine_graph_capture(Qwen2Engine *e);
void qwen2_engine_graph_free(Qwen2Engine *e);
```

Engine struct gains: `cudaGraphExec_t graph_exec; int graph_ready;` plus pinned host int `*h_sampled;` and its device mirror already exists (`d_out`).

**Step 2: Warmup + capture**

Implementation sketch for `qwen2_engine_graph_capture`:

```c
int qwen2_engine_graph_capture(Qwen2Engine *e) {
    /* warm up on a side stream: graph capture forbids legacy-default-stream ops */
    for (int i = 0; i < 3; i++) {
        if (advance(e, 151643)) return -1;   /* any valid token */
        qwen2_engine_next(e);
    }
    cudaStreamSynchronize(e->stream);

    cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal);
    /* one full step, exactly mirroring advance()+next()'s sampling stage */
    embed_token(e, e->pending_tok);          /* see step 3 */
    forward_layers(e);
    k_rmsnorm<<<1,256,256*sizeof(float),e->stream>>>(e->d_x, e->d_out_norm,
                                                     e->d_xn, e->cfg.dim, e->cfg.rms_eps);
    tt_logits_dispatch(e->d_out_w, e->out_is_q8, e->d_xn, e->d_logits,
                       e->cfg.vocab, e->cfg.dim, e->stream);
    k_argmax_partial<<<256,128,0,e->stream>>>(e->d_logits, e->cfg.vocab,
                                              e->d_bvals, e->d_bidxs);
    k_argmax_final<<<1,1,0,e->stream>>>(e->d_bvals, e->d_bidxs, 256, e->d_out);
    cudaGraph_t graph;
    cudaStreamEndCapture(e->stream, &graph);
    cudaError_t err = cudaGraphInstantiate(&e->graph_exec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);
    e->graph_ready = (err == cudaSuccess);
    return e->graph_ready ? 0 : (int)err;
}
```

**Step 3: The pending-token problem**

A captured step must embed a token chosen at replay time. Solve by splitting `advance`: add `int d_next_tok;` device int; `k_embed_q4_0` gets a new variant taking `const int *d_tok` instead of a host value:

```cuda
__global__ void k_embed_q4_0_dyn(const BlockQ4_0 *W, const int *d_tok,
                                 float *dx, int dim) {
    k_embed_q4_0(W, *d_tok, dx, dim);   /* device-side indirection */
}
```

Host flow per generated token becomes:
1. `cudaMemcpyAsync(e->d_next_tok, &id, 4, H2D, e->stream)` (token from previous sample)
2. `cudaGraphLaunch(e->graph_exec, e->stream)`
3. `cudaMemcpyAsync(e->h_sampled, e->d_out, 4, D2H, e->stream)` (pinned buffer)
4. `cudaStreamSynchronize(e->stream)` — the ONLY sync, once per token
5. read `*h_sampled`

Rewrite `qwen2_engine_next()` as exactly this when `graph_ready`, keeping the eager path as fallback (`TT_NO_GRAPH=1` env forces fallback — keep it for debugging forever).

Prefill stays eager (variable length). After prefill, prime the pipeline: run one eager `next()`-style sampling WITHOUT advancing, set `d_next_tok` to that id, then capture.

Careful bookkeeping: `e->pos++` cannot happen on the host inside a captured region. Move the position increment INTO the captured step as a tiny final kernel:

```cuda
__global__ void k_pos_inc(int *d_pos) { (*d_pos)++; }
```

appended at the end of the captured sequence, and remove the `cudaMemcpyAsync(d_pos...)` from `advance`'s graph path (keep it in the eager path).

**Step 4: Correctness check**

Rebuild all three binaries. Run: `./scripts/verify.sh m61` — expected PASS 7/7 (dump_logits uses eager path; that alone doesn't prove the graph path).

Then prove the GRAPH path: `TT_RAW_PROMPT=1 ./build/run_llm_gpu "The capital of France is" 16` must print the identical continuation ("Paris, and it is the capital of Europe…"). Run all five known prompts manually. Any divergence = capture bug (most likely: stale d_pos ordering or the pending-token plumbing).

**Step 5: Measure + commit**

Run: `python3 bench/bench_llm.py`
Expected: **≥ 80 tok/s** (launch overhead collapses to one graph launch + one sync per token).
Record in results.md. Commit: `git add -A && git commit -m "M6.3: cudaGraph replay of decode step; XX tok/s; parity intact"`

---

## Task 4: Fuse residual-add + RMSNorm

Two full passes over `dim` floats become one.

**Files:**
- Modify: `kernels/qwen2_cuda.cu`

**Step 1: New kernel**

```cuda
/* y = rmsnorm(x + r) * g ; x updated in place to x+r first */
__global__ void k_add_rmsnorm(float *__restrict__ x, const float *__restrict__ r,
                              const float *__restrict__ g, float *__restrict__ y,
                              int dim, float eps) {
    extern __shared__ float s[];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float v = x[i] + r[i];
        x[i] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) s[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++) t += s[w];
        s[0] = rsqrtf(t / (float)dim + eps);
    }
    __syncthreads();
    const float inv = s[0];
    for (int i = tid; i < dim; i += blockDim.x) y[i] = x[i] * inv * g[i];
}
```

**Step 2: Replace the two-step sequences in `forward_layers`**

Attention block: delete `tt_gemv(w->o,...)` into `d_xn` + separate `k_add`; instead call the o-GEMV into `d_xn`, then `k_add_rmsnorm(d_x, d_xn, w->ffn_norm, e->d_xn, ...)` producing xn directly (this merges residual #1 + ffn-norm).

MLP block: down-GEMV into `d_xn`, then `k_add_rmsnorm(d_x, d_xn, next layer's attn_norm … )` — CAREFUL: the next norm belongs to the NEXT iteration. Simplest correct form: keep MLP-end as plain `k_add` + let next iteration's step-1 rmsnorm stay as-is, and only apply the fused kernel at the attention boundary (one fusion site, half the risk). Do ONLY the attention-boundary fusion in this task.

Net effect per layer: 2 kernels → 1, and one fewer full-tensor read/write round trip.

**Step 3: Verify + measure + commit**

`./scripts/verify.sh m61` → PASS required. `TT_RAW_PROMPT=1 ./build/run_llm_gpu` spot-check France prompt text unchanged. `python3 bench/bench_llm.py` — expect +3–8%.
Commit: `git commit -am "M6.3: fused residual+rmsnorm at attention boundary"`

---

## Task 5: LM-head q8_0 GEMV vectorization

The logits projection reads 144 MB/token — the single largest traffic item. Current kernel loads `int8` scalars. Vectorize 4-at-a-time.

**Files:**
- Modify: `kernels/gemv_q4_cuda.cu`

**Step 1: Vectorized inner loop**

Replace the q8_0 inner loop with 32-bit loads and one `__dp4a` (sm_61+ integer dot product instruction):

```cuda
#include <cuda_dp4a.h>  /* or rely on intrinsics.h via cuda_fp16.h chain */

__global__ void k_logits_q8_0(const BlockQ8_0 *__restrict__ W,
                              const float *__restrict__ x,
                              float *__restrict__ logits, int vocab, int K) {
    const int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= vocab) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const BlockQ8_0 *rowW = W + (long)v * nb;
    int acc = 0;                                  /* int accumulator for dp4a */
    for (int b = lane; b < nb; b += 32) {
        BlockQ8_0 blk = rowW[b];
        const float *xb = x + b * 32;
        const uint32_t *q4 = reinterpret_cast<const uint32_t *>(blk.qs);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t qq = q4[i];
            int xi[4];
            xi[0] = __float_as_int(xb[i*4+0]); xi[1] = __float_as_int(xb[i*4+1]);
            xi[2] = __float_as_int(xb[i*4+2]); xi[3] = __float_as_int(xb[i*4+3]);
            acc = __dp4a(qq, *reinterpret_cast<int*>(xi), acc);
        }
        sum_f += __half2float(blk.d) * (float)acc_for_this_block;  /* see note */
    }
    ...
}
```

IMPLEMENTATION NOTE (read before coding): `__dp4a(a,b,c)` computes c + Σ a[i]*b[i] over four int8 lanes of each 32-bit operand. The clean formulation: accumulate `acc_i32 += dp4a(qword, xword_as_int, 0)` per group of 4, multiply the BLOCK subtotal by `d` once per block, add into the fp32 sum. Because x is fp32, you must convert x to int8-scale first — NOT viable directly. The actually-viable variant: keep fp32 FMA loop but load qs via `uint32_t` and unpack with `__byte_perm`, converting 4 int8s to float via `__int2float_rn` on extracted bytes — measure whether this beats the scalar loop; if gain < 5%, REVERT and record. Alternative simpler win: increase `blockDim.y` rows-per-block from 16 to 32 for better L2 streaming of x, and add `#pragma unroll 4` on the b-loop. TRY THE SIMPLE VARIANT FIRST; only reach for dp4a if profiling shows compute-bound (it won't — this is bandwidth-bound).

**Decision rule for this task:** try variants in order (a) rows-per-block 16→32, (b) `#pragma unroll`, (c) uint32 byte-extract loads. Keep whatever measures fastest under `bench_llm.py`; revert anything slower. One variant per commit.

**Step 2: Verify + measure + commit per variant**

After each variant: `./scripts/verify.sh m61` must PASS, then `python3 bench/bench_llm.py`.
Final commit: `git commit -am "M6.3: lm-head GEMV tuning -> XX tok/s"`.

---

## Task 6: rmsnorm multi-block width (small, safe)

`k_rmsnorm` launches 1 block × 256 threads: one SM does all the work serially over 896 floats while 13 SMs idle. For dim=896 this is ~2μs either way — MEASURE FIRST with nsight or a timed loop before touching. Only proceed if the profile shows rmsnorm > 3% of step time. Expected outcome: SKIP (documented as measured-not-worth-it). YAGNI applies to kernels too.

Run: `python3 bench/bench_llm.py` before and after any change; keep only if >2% improvement. Commit or document skip.

---

## Task 7: Final validation + documentation

**Files:**
- Modify: `bench/results.md`
- Modify: `TODO.md`
- Modify: `PLAN_M6.md` (mark completed phases)

**Step 1: Full gate sweep**

Run: `./scripts/verify.sh m0 && ./scripts/verify.sh m1 && ./scripts/verify.sh m61`
Expected: all green (CPU gates prove the gemm.c/stream work didn't regress anything).

**Step 2: Results table**

Append to `bench/results.md`:

```markdown
## LLM decode ladder (M6.3) — RTX 3050 laptop, Qwen2.5-0.5B-Instruct q4_0/q8_0-head
| config                        | decode tok/s | parity     |
|-------------------------------|--------------|------------|
| eager (post stream-unify)     | 48           | 7/7 top1   |
| + cudaGraph replay            | XX           | 7/7 top1   |
| + fused add+rmsnorm           | XX           | 7/7 top1   |
| + lm-head tuning              | XX           | 7/7 top1   |
| llama.cpp reference (tg)      | ~58          | —          |
```

**Step 3: Commit**

```bash
git add -A && git commit -m "M6.3 complete: XX tok/s decode (from 48), parity maintained throughout"
```

---

## Rollback rules

- Any m61 gate failure after a change: `git stash` or `git revert HEAD`, re-run gate, confirm green, then diagnose offline.
- Never stack more than one unverified optimization.
- The eager path (`TT_NO_GRAPH=1`) is permanent debugging infrastructure — do not delete it when graphs land.

## Out of scope (explicitly)

- fp16/int8 KV cache (changes numerics; revisit only if P2 fails later)
- Batching (single-user terminal is the product)
- Tensor-core paths for GEMV (memory-bound; WMMA buys nothing at bs=1)
- Speculative decoding (different project scale)
