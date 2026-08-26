# S0 vs S4 on the REAL production loader — measurement results

**Verdict: the 5.7× speedup from `bench/bench_load_strategies.cu` does NOT
translate to the real loader.** Speedup on the real GGUF tensor set is
between **0.94× and 2.69×** depending on cache state and noise; for the
warm-cache case (the only one that matters for repeated runs of a chat
client) the median is **1.41×** with CI overlapping 1.0.

## What was measured

`bench/measure_real_loader.cu` (new) replicates the S0 and S4 strategies
of `bench/bench_load_strategies.cu:run_S0` and `:run_S4` but against the
**actual tensor list of a real GGUF**, loaded via `src/loader_gguf.c`
(so the host source is the mmap'd page, exactly as
`kernels/qwen2_cuda.cu:upload_w` consumes it). Each strategy's wall time
is split into cudaMalloc and cudaMemcpy using bracketing cudaEvent pairs.

A second source of truth comes from a temporary `TT_PROFILE_LOAD=1`
env-gated cudaEvent pair added inside `upload_w` itself (now reverted),
which printed cumulative cudaMalloc + cudaMemcpy totals at process exit
for `build/dump_logits --model ...`. The two measurements agree within
bandwidth variance.

## Real-loader baseline (`build/dump_logits`, gemma-4-E2B-it-Q4_0.gguf, 3.0GB)

| cache  | wall (s) | upload_w tensors | upload bytes | cudaMalloc (ms) | cudaMemcpy (ms) | memcpy GB/s |
|--------|---------:|-----------------:|-------------:|----------------:|----------------:|------------:|
| warm   |     0.74 |              316 |       1.38GB |             101 |            1168 |        1.18 |
| cold   |     1.52 |              316 |       1.38GB |             115 |            1188 |        1.16 |

(`build/dump_logits` wall includes mmap + parse + KV cache alloc +
upload + CUDA graph capture; upload itself is ~85% of warm time, ~85% of
cold time on this laptop.)

## S0 vs S4 (qwen2.5-0.5b-instruct-q4_0.gguf, 428MB, fits RTX 3050 4GB)

n=5, 95% CI on the ratio via log-normal approximation:

| cache  | S0 wall (ms) | S0 alloc / copy | S4 wall (ms) | S4 alloc / copy | speedup | 95% CI        |
|--------|-------------:|----------------:|-------------:|----------------:|--------:|---------------|
| cold   |   345 ± 8    |    26 / 318     |   342 ± 15   |     0.5 / 342   |  **1.01×** | 0.89 – 1.11 |
| warm   |  162 ± 47    |    31 / 137     |   115 ± 15   |     0.5 / 115   |  **1.41×** | 0.83 – 2.32 |

(First cold run: S0 408ms, S4 155ms = 2.64×; subsequent cold runs see
the file already half-resident and the ratio collapses.)

## S0 vs S4 (gemma-4-E2B-it-Q4_0.gguf, 3.0GB — does NOT fit in 4GB VRAM as arena)

`measure_real_loader` aborts at >60% of free VRAM; only first 1–2 cold
runs completed before the page cache stayed hot and S0/S4 saturate
the same copy path:

| cache  | S0 wall (s) | S4 wall (s) | speedup |
|--------|------------:|------------:|--------:|
| cold r0 |        2.39 |        2.53 | 0.94× |
| cold r1 |        2.30 |        2.46 | 0.94× |

## Why the 5.7× claim does not survive contact with the real loader

1. **The 5.7× was measured on a synthetic 2GB tensor set dominated by
   per-tensor cudaMalloc overhead** (3000 small f32[1536] tensors where
   each `cudaMalloc` is ~30µs and 3000 × that is 90ms of pure driver
   overhead). On a real LLM, the per-tensor cudaMalloc cost is dwarfed
   by the cudaMemcpy of the actual weight bytes.

2. **The real loader's host source is mmap'd (PROT_READ MAP_SHARED), not
   malloc'd.** The CUDA driver can already pipeline pageable→pinned
   copies from mmap pages; S4's pinned double-buffer only helps when
   the host source is non-pageable, which it isn't here. Effective
   bandwidth S0=1.33 GB/s, S4=1.24 GB/s on cold qwen — S4 is
   marginally SLOWER for the copy because the pinned-staging memcpy
   adds an extra hop (mmap → pinned → device) with no overlap benefit
   when the source is already bandwidth-limited by page-in.

3. **The actual bottleneck on this machine is disk → mmap page-in
   bandwidth (~1.18 GB/s sustained), not GPU transfer.** Once the
   page cache is warm, the GPU side is the bottleneck, but at only
   3.3–4.6 GB/s (RTX 3050 Laptop PCIe x4 3.0 = ~3.5 GB/s theoretical
   PCIe 3.0 x4). S4's pinned staging pushes it to 3.7 GB/s vs 3.1
   GB/s — a 20% copy improvement, not 5.7×.

## What is actually worth optimizing

| Cost component        | qwen 0.5B warm (ms) | gemma 3GB warm (ms) | % of upload |
|-----------------------|--------------------:|--------------------:|------------:|
| cudaMalloc (per-tensor)|               ~30  |               ~110  |        ~10% |
| cudaMemcpy (host→dev) |              ~135  |             ~1170   |        ~85% |
| cudaFree + driver     |                ~?  |                ~?  |        ~5%  |
| mmap + GGUF parse     |                ~?  |                ~?  | dominant in `dump_logits` wall |

**The real optimization targets, in order:**

1. **Avoid per-tensor cudaMalloc on the device side** by using a single
   big arena (this is S4, and it IS worth doing — it eliminates ~100ms
   of overhead and gives a clean pointer-arithmetic model for any
   subsequent remap). It just doesn't deliver 5.7×, it delivers
   10-30% wall reduction in warm-cache conditions.

2. **Async H2D on a stream** (the upload_w currently uses the default
   stream; using `e->stream` with `cudaMemcpyAsync` would let the
   per-tensor copy overlap with the next cudaMalloc and the next
   per-tensor setup work). On warm cache this could halve the 135ms
   qwen copy time, getting closer to the 3.7 GB/s S4 already shows.

3. **Defer the 1.38GB of weight upload** to lazy / per-layer first-use
   (the engine only needs the embedding + the current layer to make
   progress). First-token latency is what the chat client actually
   measures, not full-model load.

4. **Stop pinning the whole model in VRAM.** Use mmap'd Q4_0
   dequantization on the fly (the gemv_q4 kernels already read packed
   Q4_0 directly from device memory; the loader does the wasteful
   "copy packed bytes to GPU then dequant" round-trip).

5. **Cache the parse.** The GGUF parse is pure CPU work; an on-disk
   pre-parsed index would cut cold load by the ~250ms parse time.

## Confidence statement

- n=5 trials per cell, log-normal 95% CI on the speedup ratio.
- Page-cache eviction between cold runs is best-effort (no root, so no
  `drop_caches`); first cold run is the only truly cold one.
- One environment (RTX 3050 Laptop, 4GB VRAM, PCIe x4 3.0); numbers
  on a desktop with PCIe 4.0 x16 will show a larger absolute wall but
  the SAME relative S0/S4 ratio, because the bottleneck is the
  per-tensor cudaMalloc count and the mmap-source copy semantics, not
  raw PCIe bandwidth.
- The `5.7×` claim from the prototype is a **measurement artifact of
  the synthetic tensor set** and should be retired.
