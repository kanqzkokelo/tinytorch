# Batched Prefill FlashAttention Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Replace the per-token serial flash loop in `prefill_batched_gemm` with a single FlashAttention-style kernel that processes all `n` query tokens × `ctx` Q8_0 KV blockwise. Lift prefill throughput at $N \ge 2048$ from 24.9 tok/s to **2,000+ tok/s**, reaching llama.cpp parity (~11,000 tok/s at $N=2169$ for 0.5B on RTX 3050, but realistic target 2-4k tok/s for our Q8_0 KV path).

**Architecture:**
- New kernel `k_prefill_flash_q8_0` in `kernels/qwen2_cuda.cu`: grid = `(S, n_kv_heads, query_tile)`, where `S = ceil(ctx / BC)`, `query_tile = ceil(n / BR)`. Each block tiles `BR` query rows × `BC` KV columns through smem with online softmax, dequantizing Q8_0 on the fly.
- GQA: 7 query heads per KV head, grouped warps in same block (blockDim = `G * 32` = 224 threads). Reuse the `k_fa2_q8_split` smem layout from `tools/micro_fa2_q8.cu` but extend the q-tile dimension to BR rows per warp.
- Two-pass flash with FP32 output accumulator written to `d_Att[query, head, head_dim]`.
- Drop the existing per-token `for (i = 0; i < n; i++) k_flash_gqa_q8_0<<<...>>>(d_Q + i*attn_qout, ...)` loop in `prefill_batched_gemm` and replace with one `k_prefill_flash_q8_0` launch per layer.

**Tech Stack:** CUDA C++ (sm_86, Q8_0 KV, GQA 14/2/128), C host engine (`kernels/qwen2_cuda.cu`), FP32 attention accumulator.

**Why this, why now:** The current prefill path is $O(n^2)$ because the attention loop runs serial `k_flash_gqa_q8_0` once per prefill token, each doing $O(\text{ctx})$ work. At $n=7209$ the 24-layer prefill attention cost is $24 \times 7209 \times 7209/2 \approx 624\text{M}$ token-pairs. llama.cpp does this in a single batched flash kernel per layer, reducing it to $24 \times 7209 \times 7209 / \text{parallelism} \approx 5\text{ms}$ on H100. Our target: drop $n=7209$ prefill from $290\text{s}$ ($24.9$ tok/s) to under $4\text{s}$ ($>2\text{k tok/s}$).

---

## Task 1: Microbench `k_prefill_flash_q8_0` Kernel

**Files:**
- Create: `tools/micro_prefill_flash.cu`
- Test: Build and run `build/micro_prefill_flash`

**Step 1: Write `tools/micro_prefill_flash.cu`**

Implement `k_prefill_flash_q8_0`:
- `__global__ void k_prefill_flash_q8_0(const float* q, const BlockQ8_0* Kc, const BlockQ8_0* Vc, float* att_out, int n_queries, int n_kv_heads, int n_heads, int head_dim, int ctx_len, float scale, int window, int BR, int BC)`
- Grid: `dim3(S, n_kv_heads, num_q_tiles)` where `S = ceil(ctx/BC)`, `num_q_tiles = ceil(n/BR)`.
- Block: `G * 32` threads, `G = n_heads / n_kv_heads` (7 for qwen2.5-0.5b).
- Each warp owns one query head; each warp processes `BR/32` query rows. So `BR = 32` (one row per lane, simplest) or `BR = 64` (two rows per lane).
- Smem: `BlockQ8_0 sK[BC * (head_dim/32)]` and same for V. `float sQ[BR * head_dim]` for q tile.
- Online softmax: per-row `m`, `l`, `acc[head_dim]` updated as we sweep KV blocks. Rescale `acc *= exp(m_old - m_new)` on each block.
- Dequantize Q8_0 on read: `(float)q8_val * block->d`, where `block->d = __half2float(...)`.
- Output: write `att_out[query, head, head_dim] = acc / l` (final normalize).

Reference comparison:
- `time_baseline`: the existing per-token serial loop, doing `n_queries` launches of `k_flash_gqa_q8_0`. Allocate the same scratch d_Q/d_Att of size `n_queries * (n_heads * head_dim)`.
- `time_batched`: one `k_prefill_flash_q8_0` launch.

Test across:
- `n_queries ∈ {32, 64, 128, 256, 512, 1024, 2048}`
- `ctx ∈ {512, 1024, 2048, 4096}`
- shape `n_heads=14, n_kv_heads=2, head_dim=128`

Compute `speedup = time_baseline / time_batched` (Target: $\ge 10\times$ at $n=2048$, $\text{ctx}=2048$).

Correctness: compare `att_out` against a CPU FP32 reference implementation of standard attention. Target: `max_abs_error < 1e-2`, zero NaN/Inf.

**Step 2: Build and run microbench**

```bash
cd ~/Storage/repos/nnfromscratch
nvcc -O3 -arch=native --resource-usage -Iinclude -Isrc -o build/micro_prefill_flash tools/micro_prefill_flash.cu -L$HOME/mmcuda/lib -lcudart
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
./build/micro_prefill_flash
```

Target: at $n=2048$, $\text{ctx}=2048$ the batched kernel should be $\le 0.6\text{ms}$ (vs $2048 \times 0.06\text{ms} = 122\text{ms}$ for the serial loop = **$>200\times$ speedup**).

**Step 3: Commit**

```bash
cd ~/Storage/repos/nnfromscratch
git add tools/micro_prefill_flash.cu
git commit -m "prefill: microbench batched FlashAttention Q8_0 kernel (target 10x+ vs per-token serial)"
```

---

## Task 2: Wire `k_prefill_flash_q8_0` into Engine (replace per-token flash loop)

**Files:**
- Modify: `kernels/qwen2_cuda.cu` — copy `k_prefill_flash_q8_0` from `tools/micro_prefill_flash.cu`; replace per-token flash loop in `prefill_batched_gemm` (around line 2722)

**Step 1: Copy the kernel and launcher into `kernels/qwen2_cuda.cu`**
- Add `__global__ void k_prefill_flash_q8_0(...)` and a `static void launch_prefill_flash(...)` host wrapper near the other prefill helpers.
- The wrapper computes grid `dim3(S, KV_l, num_q_tiles)`, allocates `BR=64`, `BC=64` (or whatever the microbench tuned for sm86), and calls the kernel.

**Step 2: Replace the per-token flash loop**

Current code (line ~2720):
```c
if (e->use_q8_kvcache) {
    k_flash_gqa_q8_0<<<H_l, 32, 0, e->stream>>>(
        d_Q + (long)i * attn_qout, Kl_q8, Vl_q8, d_Att + (long)i * attn_qout,
        d_pos_i, H_l, KV_l, HDl, c->max_ctx, scale_l, swa_l);
} else {
    k_flash_gqa<<<H_l, 32, 0, e->stream>>>(
        d_Q + (long)i * attn_qout, Kl_f, Vl_f, d_Att + (long)i * attn_qout,
        d_pos_i, H_l, KV_l, HDl, c->max_ctx, scale_l, swa_l);
}
```

Replace with:
```c
if (e->use_q8_kvcache) {
    launch_prefill_flash(d_Q, Kl_q8, Vl_q8, d_Att, n, H_l, KV_l, HDl,
                         ctx_so_far, scale_l, swa_l, e->stream);
} else {
    /* keep the FP32 fallback path as-is for now — Q8 path is the bottleneck */
    for (int i = 0; i < n; i++) {
        k_flash_gqa<<<H_l, 32, 0, e->stream>>>(
            d_Q + (long)i * attn_qout, Kl_f, Vl_f, d_Att + (long)i * attn_qout,
            d_pos_i, H_l, KV_l, HDl, c->max_ctx, scale_l, swa_l);
    }
}
```

`ctx_so_far = e->pos` (the host mirror, set by `cudaMemcpy` in prefill). Note: during prefill, `*d_pos` is the position *before* this layer's first new token; KV cache already has $e \to \text{pos}-1$ populated, plus the new token $i$ is just-scattered at slot $\text{pos}+i$. Since this attention runs AFTER all `n` `k_kv_scatter_q8_0` writes for the layer, `ctx_so_far = e->pos + n`.

Read the surrounding scatter/rope loop to confirm the position semantics before integrating; if not exact, set `ctx_so_far = e->pos + n`.

**Step 3: Verify test gates**

```bash
cd ~/Storage/repos/nnfromscratch
make -j4
./scripts/ci_local.sh
./scripts/verify.sh m61
./scripts/verify.sh ple
```

Expected: ALL PASS. The m61/ple gates run short-prompt tests (≤32 tokens), so the new path is exercised but the O($n^2$) degradation is irrelevant there.

**Step 4: Commit**

```bash
cd ~/Storage/repos/nnfromscratch
git add kernels/qwen2_cuda.cu
git commit -m "prefill: wire batched FlashAttention Q8_0 into prefill_batched_gemm (replace per-token loop)"
```

---

## Task 3: Drop unused FP32 KV cache when Q8_0 KV enabled

**Files:**
- Modify: `kernels/qwen2_cuda.cu` — `qwen2_engine_create` around line 1639-1641

**Step 1: Conditionally skip FP32 KV allocation**

```c
const long cache_per = (long)max_kv * cfg->max_ctx * cfg->head_dim;
if (!e->use_q8_kvcache) {
    cudaMalloc(&e->d_kc, cache_per * cfg->n_layers * sizeof(float));
    cudaMalloc(&e->d_vc, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_kc, 0, cache_per * cfg->n_layers * sizeof(float));
    cudaMemset(e->d_vc, 0, cache_per * cfg->n_layers * sizeof(float));
}
```

This frees $\sim 252\text{MB}$ of VRAM at $\text{max\_ctx}=10240$ when Q8_0 KV is enabled, unblocking larger prefill batches and providing headroom for the batched prefill's per-token activation buffers (which grow $\sim 200\text{MB}$ at $n=7209$).

Verify nothing reads `d_kc` / `d_vc` when `use_q8_kvcache` is true (the FP32 path at line ~2010 only runs when `!e->use_q8_kvcache`, which is the case we keep). Existing code at line 2733 already gates the FP32 path similarly.

**Step 2: Run gates again**

```bash
cd ~/Storage/repos/nnfromscratch
./scripts/ci_local.sh
./scripts/verify.sh m61
./scripts/verify.sh ple
```

**Step 3: Commit**

```bash
git add kernels/qwen2_cuda.cu
git commit -m "engine: skip FP32 KV cache allocation when Q8_0 KV enabled (free 252MB @ 10k ctx)"
```

---

## Task 4: Benchmark & Verify Long-Context Prefill + Decode

**Files:**
- Modify: `data/profile/prefill_long_ctx_results.md` (new)

**Step 1: Run prefill throughput sweep**

For $N \in \{512, 1024, 2048, 4096, 7209\}$ (with the long-prompt `formatted[262144]` and `prompt_tokens[16384]` buffer edits already in place from yesterday's run):

```bash
cd ~/Storage/repos/nnfromscratch
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
for N in 512 1024 2048 4096 7209; do
    python3 -c "print('The history of quantum computing dates back to the early 1980s when physicist Richard Feynman and computer scientist Paul Benioff suggested that quantum mechanics could be harnessed for computation. Traditional computers process information using bits. ' * $((N/5)))" > /tmp/prompt_$N.txt
    TT_MAX_CTX=10240 TT_Q8_KV=1 TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf \
        ./build/run_llm_gpu "$(cat /tmp/prompt_$N.txt)" 16 2>&1 | grep -E "prompt:|STATS"
done
```

**Step 2: Run decode tok/s at $N=7209$ post-prefill**

Same setup, after prefill completes, the 16 generated tokens give decode tok/s. Should land **>120 tok/s** (vs 97 today; target 200+).

**Step 3: Run llama.cpp oracle at $N=2048, 7209$ for direct comparison**

```bash
llama-completion -ngl 99 -m data/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    -f /tmp/prompt_2048.txt -n 32 --no-display-prompt 2>&1 | tail
```

**Step 4: Write up results**

```bash
cd ~/Storage/repos/nnfromscratch
git add -f data/profile/prefill_long_ctx_results.md
git commit -m "bench: long-context prefill + decode results after batched flash attention"
```

---

## Success Criteria

- Microbench at $n=2048$, $\text{ctx}=2048$: $\text{time\_batched} \le 0.6\text{ms}$ (vs $122\text{ms}$ serial) = **$\ge 200\times$ speedup**.
- All correctness gates (`ci_local`, `m61`, `ple`) remain GREEN.
- Prefill at $N=2048$ reaches **$\ge 1{,}500$ tok/s** (vs 141 today) on RTX 3050.
- Prefill at $N=7209$ reaches **$\ge 2{,}000$ tok/s** (vs 24.9 today).
- Decode at $N=7209$ sustained context reaches **$\ge 120$ tok/s** (vs 97 today).
- No regression in short-ctx decode (must stay $\ge 260$ tok/s at $N=30$).

## Out of Scope

- FA2 decode attention (separate effort; was reverted at `da6f805` last night).
- Tensor Core WMMA for the projection GEMMs in prefill (already shipped at commit `099d244` / `3836b2c`).
- Dropping the FP32 fallback path entirely (keep it as the `!use_q8_kvcache` branch).
- Public push / v0.1.0 tagging (deferred per user).
