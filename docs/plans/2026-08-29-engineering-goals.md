# Engineering Goals & Technical Roadmaps (nnfromscratch)

## Overview
This document outlines three high-bar engineering milestones for `nnfromscratch`. Each target addresses a fundamental performance or architectural boundary in low-level CUDA LLM inference, complete with concrete substeps, hardware math, and verification criteria.

---

## Goal 1: The Zero-Overhead 4GB Appliance (7B/8B on Consumer Edge)

### Objective
Enable execution of $7\text{B}\text{--}8\text{B}$ parameter LLMs ($Q3\_K\_M$ / $Q4\_K\_M$) with a sustained $8{,}192$-token context window within a strict $4.0\text{ GB}$ VRAM physical budget (e.g., RTX 3050 Laptop), with zero host-memory swapping.

### Physics & VRAM Budget Math
- **7B Model Weights ($Q3\_K\_M$)**: $\sim 3.0\text{ GB}$
- **Activation Buffers (Chunked)**: $\le 45\text{ MB}$ (512-token chunk max)
- **8k KV Cache ($Q4\_0$ quantized KV, 28 layers, 4 KV heads, 128 dim)**:
  $$\text{KV Size} = 28 \times 2 \times 4 \times 128 \times 8192 \times 0.5625\text{ bytes} \approx 132\text{ MB}$$
- **Total Footprint**: $3.0\text{ GB} + 0.045\text{ GB} + 0.132\text{ GB} + 0.15\text{ GB (CUDA runtime)} \approx 3.33\text{ GB} \le 4.0\text{ GB}$

### Technical Substeps
1. **Substep 1.1: $Q3\_K\_M$ & $Q4\_K\_M$ CUDA GEMV Dequantization**
   - Port 2-rows-per-warp $Q3\_K$ and $Q4\_K\_M$ kernels to `kernels/gemv_typed.cu`.
   - Ensure coalesced 128-bit memory reads (`uint4`) on sub-block scale lookups.
   - Verify dequantization error against CPU golden reference ($< 5\times 10^{-7}$ relative error).
2. **Substep 1.2: $Q4\_0$ Quantized KV Cache Layout**
   - Implement `k_kv_scatter_q4_0` to pack FP32 key/value vectors into 4-bit nibbles + FP16 scale block on write.
   - Implement `k_fa2_q4_split` online softmax decode attention reading 4-bit KV blocks.
   - Validate numerical tolerance ($< 10^{-3}$ max absolute logit delta vs FP16 KV).
3. **Substep 1.3: Static Buffer Reclamation & Zero-Alloc Ingestion**
   - Eliminate all dynamic `cudaMalloc` calls during runtime; pre-allocate one unified scratch buffer at startup.
   - Clamp maximum batch prefill memory using the 512-token chunked pipeline.

4. **Substep 1.4: Hybrid Layer Offloading (Split GPU/CPU Layer Execution)**
   - Allow configurable split: $N_{\text{gpu}}$ layers on GPU VRAM and $N_{\text{cpu}}$ layers in System RAM (using the multi-threaded AVX2 CPU backend in `src/cpu_backend.c`).
   - Pipeline layer boundary transitions via pinned host memory (`cudaMemcpyAsync`) with zero GPU execution stalls.
   - Enables executing $8\text{B } Q4\_K\_M$ ($\approx 4.5\text{ GB}$) and $14\text{B } Q4\_K\_M$ ($\approx 8.5\text{ GB}$) on a 4GB GPU laptop by loading the bulk of layers into VRAM and offloading remainder layers to 16GB/32GB System RAM.
### Verification Gate
- Load `Llama-3.1-8B-Instruct-Q3_K_M.gguf` on a 4GB GPU device.
- Prefill an 8,192-token prompt and generate 64 tokens without OOM or fallback to CPU RAM.

---

## Goal 2: Microsecond-Level Graph Orchestration (Peak Bandwidth Decode)

### Objective
Sustain $\ge 95\%$ of physical memory bandwidth theoretical maximum ($\sim 168\text{ GB/s}$ out of $176\text{ GB/s}$ on RTX 3050) across all decode steps from context length $32$ to $8{,}192$, eliminating all host-device synchronization gaps.

### Performance Breakdown Math
- Model Read Traffic ($0.5\text{B } Q4\_0$): $279.6\text{ MB}$ per step
- Target Step Time at $95\%$ Bandwidth:
  $$t_{\text{target}} = \frac{279.6\text{ MB}}{168\text{ GB/s}} \approx 1.66\text{ ms} \implies \approx 602\text{ tokens/sec}$$

### Technical Substeps
1. **Substep 2.1: Full-Step CUDA Graph Capture with Host-Pipelined Sampling**
   - Incorporate all 24 layers, normalization kernels, RoPE, and LM-head logits into a single static CUDA Graph.
   - Move argmax/sampling kernel directly to the tail of the device graph.
   - Implement asynchronous double-buffered token delivery (`d_sampled -> h_sampled`) over pinned host memory.
2. **Substep 2.2: Fused Normalization & Quantized Weight Epilogues**
   - Fuse RMSNorm directly into the input stage of the first GEMV warp tile to eliminate intermediate memory round-trips.
   - Fuse SwiGLU / GeGLU activation functions into FFN-gate and FFN-up output registers.
3. **Substep 2.3: Zero-Overhead KV Cache Pointers in Static Replay**
   - Maintain static device-side pointer tables for KV cache layers so graph topology remains invariant to position increments.

### Verification Gate
- Measure single-token decode latency on `qwen2.5-0.5b-instruct-q4_0` via `nvprof` / `nsys`.
- Confirm kernel launch overhead between layers is $< 0.5\ \mu\text{s}$ and total step time reaches $\le 3.5\text{ ms/tok}$.

---

## Goal 3: Zero-Dependency Portability & Reference Correctness

### Objective
Maintain a standalone CUDA C LLM inference engine that compiles with a single command on any POSIX system with standard NVCC, has zero third-party dependencies, and maintains bit-exact parity across major LLM architectures.

### Architecture Support Scope
1. **Qwen2 / Qwen2.5 / Qwen3**: NeOX RoPE, SiLU activation, per-head QK-RMSNorm, bias support.
2. **LLaMA-3 / 3.1 / 3.2**: GPT-J RoPE with frequency-factor scaling, GQA attention, 128k vocabulary.
3. **Gemma / Gemma-2 / Gemma-4**: GeGLU activation, logit softcapping, sliding-window attention (SWA), per-layer embeddings.
4. **Mistral / SmolLM2**: Standard GQA LLaMA architecture variants.

### Technical Substeps
1. **Substep 3.1: Self-Contained Architecture Registry & Trait Resolvers**
   - Parse all model metadata directly from raw GGUF headers without python wrappers.
   - Auto-detect activation types, RoPE formulas, and layer geometries at model load time.
2. **Substep 3.2: Automated Golden Parity Test Harness**
   - Expand `tests/test_engine_golden.py` to compare logits against reference outputs across all supported families.
   - Enforce gate criterion: top-1 argmax match on $100\%$ of standard benchmark prompts with median logit delta $< 0.15$.
3. **Substep 3.3: One-Line Single Binary Build System**
   - Maintain a Makefile with zero external library linkages beyond `libc`, `libpthread`, and `libcuda/libcudart`.
   - Build time target: $< 15\text{ seconds}$ from clean checkout.

### Verification Gate
- Run `bash scripts/ci_local.sh && bash scripts/verify.sh all`.
- Complete all checks with zero errors and zero runtime dependencies.
