# Fused Matmul + Bias + Activation Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Close the decode gap to llama.cpp CUDA (currently 0.71× on qwen2.5-0.5b-q4_0) by reducing kernel launch overhead and the number of round-trips through global memory in the per-layer epilogue.

**HARD GATE (Task 0):** Before doing any fusion work, run `nsys` to measure the actual inter-kernel gap. If the average gap is <1µs, the entire fusion thesis is wrong and we should pivot to GEMV-internal optimization (vectorization, L2 cache hints, dequant in registers). If the gap is >2µs, fusion is worth pursuing.

**Architecture (if gate passes):** Move from ~30 launches per layer to ~10 by fusing small epilogue kernels. The decode is currently ~230 tok/s vs llama.cpp's 325 tok/s. The remaining 0.5B model is bandwidth-bound on the GEMV; fusion is a secondary lever.

**Tech Stack:** CUDA C (Ampere, sm_86), existing tt_gemv_typed scalar V2 kernels, graph capture already in place (caches launch sequence).

**Expected gain:** If launch overhead is the bottleneck: 10-20% decode speedup (230 → 250-275 tok/s). If bandwidth-bound (more likely): 2-5%. Final ratio target: ≥0.75× of llama.cpp CUDA on qwen2.5-0.5b-q4_0.

**Constraints:**
- m61 7/7, m84, ple gates must remain green
- Graph capture must continue to work (RE-CAPTURE after kernel changes)
- K-quants (Q4_K, Q5_K, Q6_K) must not regress — they already use scalar path
- Q4_0/Q8_0 V2 path is the target for fusion
- Do NOT touch speculative decoding (rejected by Qwen as break-even)
- **Bias-add is NOT in qwen2.5-0.5b's Q/K/V/O or gate/up — no bias-fusion tasks needed** (verify by reading GGUF metadata first in Task 0)
- **RoPE requires paired-element access — fusion kernel must handle this correctly**

---

## Background

**Per-layer launch count (verified empirically):** ~30 launches per layer × 24 layers = ~720 launches per decode step. With graph capture, launches are batched into a single `cudaGraphLaunch` call, so the actual cost is graph-node execution, not launch overhead. **This is the critical insight the plan hinges on: graph capture already eliminates launch overhead. The remaining cost is per-node execution time and global memory traffic between nodes.**

**Implication:** Fusion's value is NOT launch overhead reduction (graph capture handles that). Fusion's value is reducing global memory traffic by keeping intermediate values in shared memory or registers across fused operations.

---

## Task 0: HARD GATE — measure actual launch overhead and bias presence

**Files:**
- Create: `data/profile/launch_gap.log` (from `nsys`)
- Create: `data/profile/bias_inventory.md`

**Why this is first:** Per Qwen review, the original plan overestimated launch overhead. Graph capture already eliminates per-kernel launch latency; what's left is the per-node execution time and global memory traffic between nodes. Verify before doing any work.

**Step 1: Check if any model in bench has biases**

```bash
cd ~/Storage/repos/nnfromscratch
python3 -c "
import os
for f in ['data/testmodels/qwen2.5-0.5b-instruct-q4_0.gguf',
          'data/testmodels/qwen3-0.6b-q8_0.gguf',
          'data/testmodels/llama-3.2-1b-q4_0.gguf',
          'data/testmodels/smollm2-135m-f16.gguf']:
    if not os.path.exists(f):
        print(f'{os.path.basename(f)}: MISSING')
        continue
    with open(f, 'rb') as fp:
        data = fp.read(1<<20)  # first 1MB
    has = {n: (n.encode() in data) for n in ['attn_q.bias', 'attn_k.bias', 'attn_v.bias', 'ffn_gate.bias', 'ffn_up.bias']}
    print(f'{os.path.basename(f)}: {has}')
"
```

**Expected output:** All False for qwen2.5, qwen3, llama-3.2, smollm2. **If True for any model, that model's bias-fusion targets are real; otherwise all bias-fusion work is skipped.**

Document result in `data/profile/bias_inventory.md`.

**Step 2: Measure actual launch overhead with nsys**

```bash
cd ~/Storage/repos/nnfromscratch
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
mkdir -p data/profile
nsys profile --stats=true --force-overwrite=true -o data/profile/launch_gap.nsys-rep \
  env TT_MODEL=data/testmodels/qwen2.5-0.5b-instruct-q4_0.gguf \
      ./build/run_llm_gpu "The quick brown fox" 16 2>&1 | tail -50
```

**Step 3: Read the stats output**

Look for the "CUDA Kernel" table. Note:
- Total kernels per decode step (should be ~700-800 for 16-token decode)
- Average kernel duration (if <1µs, fusion is wasted effort)
- Total GPU time per decode step (should match ~4ms per token for 230 tok/s)

**Step 4: Decision**

- If average kernel duration >5µs: **proceed with fusion plan** (launch overhead is real)
- If average kernel duration 1-5µs: **partial proceed** — only fuse the largest kernels (SwiGLU, RMSNorm, RoPE)
- If average kernel duration <1µs: **ABORT the plan** — pivot to GEMV internal optimization (vectorization, register-resident dequant)

**Step 5: Commit decision + raw data**

```bash
git add data/profile/
git commit -m "profile: nsys launch-gap measurement + bias inventory (gate)"
```

---

## Task 1: Profile to confirm the bottleneck

**Files:**
- Read: `data/bench/results_cuda_vs_cuda_v2.md`
- Create: `data/profile/nsight_decode.log` (from running `ncu`)

**Step 1: Run a single decode pass under ncu**

```bash
cd ~/Storage/repos/nnfromscratch
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
mkdir -p data/profile
ncu --launch-skip 100 --launch-count 30 \
  --csv \
  --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_xu.sum \
  --target-processes all \
  env TT_MODEL=data/testmodels/qwen2.5-0.5b-instruct-q4_0.gguf \
      ./build/run_llm_gpu "The quick brown fox" 16 \
  > data/profile/nsight_decode.log 2>&1
```

**Step 2: Identify the top 3 kernels by time**

```bash
grep -A 1 "^Process" data/profile/nsight_decode.log | head -20
grep "k_\|tt_gemv" data/profile/nsight_decode.log | head -30
```

**Step 3: Document findings**

Write a 5-line summary to `data/profile/bottleneck_summary.md` listing the top 3 time-consuming kernels and their % of total decode time.

**Step 4: Commit**

```bash
git add data/profile/bottleneck_summary.md
git commit -m "profile: identify top decode kernels via ncu"
```

---

## Task 2: Fuse RMSNorm + first GEMV into single launch (only if Task 0 shows >2µs gap)

**Files:**
- Modify: `kernels/qwen2_cuda.cu` — find the sequence: `k_rmsnorm(...)` followed by `tt_gemv_typed(...)` for the Q projection
- Add: new function `tt_fused_rmsnorm_gemv_q4_0(...)` in `kernels/gemv_q4_cuda.cu`

**Pre-flight:** Confirm Task 0 result shows >2µs gap. If not, SKIP this task.

**Architecture decision:** The Q GEMV is the first matmul of every layer. Currently we do RMSNorm to produce `d_xn` (fp32, dim=896), then read d_xn in Q GEMV. If we fuse: read the input `d_x` (fp32), compute RMSNorm in shared memory, immediately feed into the GEMV's dequant loop. This saves:
- 1 kernel launch (graph-captured, so cost is ~0.5-2µs)
- 1 write+read of d_xn (896 floats = 3.5KB) — minor

**Per Qwen review:** Fusing RMSNorm into GEMV may serialize compute and memory. Test carefully. If the fused kernel is slower than the unfused pair, REVERT.

**Step 1: Find the current RMSNorm + Q GEMV sequence**

```bash
cd ~/Storage/repos/nnfromscratch
grep -n "k_rmsnorm\|tt_gemv_typed(w->q.ptr" kernels/qwen2_cuda.cu | head -20
```

The sequence is in `qwen2_engine_decode_step` or similar function. Look for the pre-attn RMSNorm followed by the Q projection.

**Step 2: Write the fused kernel**

In `kernels/gemv_q4_cuda.cu`, add:
```cuda
/* M9.5+ fused: RMSNorm + Q4_0 GEMV. One launch reads d_x (fp32),
 * normalizes in shared memory, then dispatches to V2 GEMV. */
__global__ void k_fused_rmsnorm_q4_0_gemv_v2(
    const float * __restrict__ d_x,    /* input (raw, pre-norm) */
    const float * __restrict__ d_gamma, /* RMSNorm weight */
    const BlockQ4_0 * __restrict__ W,   /* Q4_0 weights, M rows x K cols */
    float * __restrict__ d_y,           /* output (M floats) */
    int dim,                            /* K = model dim (e.g. 896) */
    int M,                              /* output rows (e.g. attn_qout) */
    float eps                           /* RMSNorm eps (1e-5 or 1e-6) */
) {
    extern __shared__ float smem_x[];   /* size = dim * sizeof(float) */

    // Phase 1: compute RMS, normalize, store in shared
    int tid = threadIdx.x;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        local_sum_sq += d_x[i] * d_x[i];
    }
    // block-wide reduction (warp shuffle + shared mem)
    __shared__ float ssum;
    // ... use cub::BlockReduce or manual reduction ...
    // normalize: smem_x[i] = d_x[i] * rsqrt(ssum/dim + eps) * d_gamma[i]
    __syncthreads();

    // Phase 2: V2 GEMV reading from shared memory
    // (2-rows-per-warp pattern, smem_x as input)
    // ... V2 implementation ...
}
```

For 0.5B model: dim=896, M_q=896. Block of 256 threads. Shared mem = 896*4 = 3.5KB. Fits in 1 block per output row range.

**Step 3: Wire up the dispatch**

In the engine, replace the pre-attn `k_rmsnorm` + Q-GEMV pair with one call to the fused kernel. **Re-capture the graph** (graph state is invalidated).

**Step 4: Verify gates**

```bash
cd ~/Storage/repos/nnfromscratch
./scripts/ci_local.sh
./scripts/verify.sh m61
./scripts/verify.sh ple
./scripts/verify.sh m84
```

Expected: all PASS. m61 catches any numerical drift from the fused path.

**Step 5: Bench + compare**

```bash
bash tools/bench_cuda_vs_cuda.sh
```

If qwen2.5-q4_0 ratio DROPS: REVERT (the fusion is a regression, abort).
If it goes UP by ≥3%: keep. If <3%: keep but note the small gain.

**Step 6: Commit (only if it helps)**

```bash
git add kernels/gemv_q4_cuda.cu kernels/qwen2_cuda.cu
git commit -m "M9.5+: fuse RMSNorm + Q4_0 GEMV (one launch for pre-attn norm + Q proj)"
```

If regressed: `git revert HEAD`.

---

## Task 3: Fuse Q/K RoPE into single launch (skip if Q/K are already in one launch)

**Files:**
- Modify: `kernels/qwen2_cuda.cu` — find the RoPE sequence
- (Optional) Add: new kernel `k_rope_qk_neox_fused` that applies RoPE to Q and K in one launch

**Pre-flight check (no work needed if already fused):**

```bash
cd ~/Storage/repos/nnfromscratch
grep -n "k_rope\|k_rope_qk\|rope_neox\|rope_gptj" kernels/qwen2_cuda.cu | head -10
```

If `k_rope_qk` already takes both `d_q` and `d_k`: SKIP this task.

**Architecture (revised after Qwen review):**
- qwen2.5/qwen3/llama-3.2/smollm2: **NO biases** in Q/K/V/O or gate/up/down. Bias-fusion tasks removed.
- RoPE pairing: Neox pairs adjacent elements within a head. A naive "one thread per element" fusion is WRONG because it loses the pair. Use one thread per element but read/write the pair via two iterations, OR use one warp per head with lane-shuffles.

**Step 1: If separate, write the fused kernel**

```cuda
__global__ void k_rope_qk_neox_fused(
    float *d_q, float *d_k, int q_len, int k_len, int HD,
    const float *freqs, int n_heads_q, int n_heads_k
) {
    /* One thread per (head, dim_half) — handle the pair (d, d+HD/2) together */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = n_heads_q * (HD / 2);
    if (idx >= total_q) return;
    int h = idx / (HD / 2);
    int half = idx % (HD / 2);
    float f = freqs[h * (HD/2) + half];
    float cosf = cosf(f), sinf = sinf(f);

    int d_lo = h * HD + half;          /* first half of head */
    int d_hi = h * HD + half + (HD/2); /* second half (the pair) */

    /* Q */
    if (d_lo < q_len) {
        float qa = d_q[d_lo], qb = d_q[d_hi];
        d_q[d_lo] = qa * cosf - qb * sinf;
        d_q[d_hi] = qa * sinf + qb * cosf;
    }
    /* K */
    if (d_lo < k_len && h < n_heads_k) {
        float ka = d_k[d_lo], kb = d_k[d_hi];
        d_k[d_lo] = ka * cosf - kb * sinf;
        d_k[d_hi] = ka * sinf + kb * cosf;
    }
}
```

**Step 2: Wire up (only if Q and K RoPE are currently separate launches)**

Replace the two `k_rope_qk` calls with one `k_rope_qk_neox_fused`. **Re-capture graph.**

**Step 3: Verify + bench + commit (if changed)**

```bash
cd ~/Storage/repos/nnfromscratch
make build/run_llm_gpu
./scripts/verify.sh m61 && ./scripts/verify.sh ple && ./scripts/verify.sh m84
bash tools/bench_cuda_vs_cuda.sh
git add kernels/qwen2_cuda.cu
git commit -m "M9.5+: fuse Q+K RoPE (Neox, no bias) into single launch"
```

If no change needed, document the skip and move on.

---

## Task 4: End-to-end verification + bench

**Files:**
- Read: `data/bench/results_cuda_vs_cuda_v2.md`
- Update: `data/profile/bottleneck_summary.md` with post-fusion numbers

**Step 1: Run the full bench**

```bash
cd ~/Storage/repos/nnfromscratch
bash tools/bench_cuda_vs_cuda.sh
cat data/bench/results_cuda_vs_cuda_v2.md
```

**Step 2: Run the oracle comparison**

```bash
cd ~/Storage/repos/nnfromscratch
python3 -c "
import json
rows = [json.loads(l) for l in open('data/bench/results_cuda_vs_cuda_v2.jsonl') if 'ratio' in l]
geomean = 1
for r in rows: geomean *= r['ratio']
geomean = geomean ** (1/len(rows))
print(f'geomean: {geomean:.3f}, wins: {sum(1 for r in rows if r[\"ratio\"] >= 1.0)}/{len(rows)}')
for r in rows: print(f\"  {r['name']:<40} {r['ratio']:.2f}\")
"
```

**Target:** geomean ≥ 0.65×, qwen2.5-q4_0 ≥ 0.75×. **If we did Task 0 honestly and the gate failed (gap <1µs), we may need to ABORT the whole plan and pivot to GEMV-internal optimization instead.**

**Step 3: Final commit if any docs/bench artifacts changed**

```bash
cd ~/Storage/repos/nnfromscratch
git add data/bench/ data/profile/
git commit -m "bench: post-M9.5+ fusion results (geomean XX, ratio Y)"
```

---

## Out of Scope (Day 2+ candidates)

These were identified but not pursued for day 1:
- **Flash attention** (Option A) — expected 5-10% gain, complex, defer
- **Full layer mega-kernel** (Option C) — multi-day, defer
- **Self-speculative decoding** — break-even per Qwen, skip
- **MMQ-style WMMA for Q4_0** — our prior attempt regressed, defer
- **LM head 4-rows-per-warp V4** — small, defer
- **GEMV-internal optimization** (vectorization, register-resident dequant) — if Task 0 gate fails, this becomes Day 1's plan

## Risks & Mitigations

- **Risk:** Task 0 gate shows <1µs gap → fusion is wasted. **Mitigation:** Task 0 is the explicit gate; we abort cleanly.
- **Risk:** RMSNorm fusion breaks graph capture. **Mitigation:** graph capture re-runs lazily on first decode step. Check `[qwen2-engine] decode-step graph captured` log line. If absent, fall back to eager.
- **Risk:** m61 numerical drift. **Mitigation:** m61 golden test is bit-exact. If drift >1e-4, the change is incorrect — revert.
- **Risk:** Per-model regressions (e.g. fusion helps qwen2.5 but hurts smollm2 like the QKV attempt). **Mitigation:** bench after each task. If any model drops >5%, REVERT immediately and isolate to that model via dtype-specific dispatch.
- **Risk:** RMSNorm-into-GEMV serialization (per Qwen). **Mitigation:** bench after Task 2. If slower, REVERT.

## Success Criteria

- m61 7/7, m84 6/7, ple 3/3 — all gates remain green
- qwen2.5-0.5b-q4_0 ratio ≥ 0.75× (was 0.71×)
- Geomean across 4 bench models ≥ 0.60× (was 0.57×)
- No model regresses >5% from its current ratio
- Task 0 gate passes (or we pivot to GEMV-internal opt)
- All changes committed in atomic, isolated commits
