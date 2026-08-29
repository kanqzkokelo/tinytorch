# Frontier Engineering Milestones: Breaking Physical & Architectural Limits (nnfromscratch)

## Overview
These four engineering targets push beyond current industry standard edge engines (`llama.cpp`, `vLLM`, `Ollama`), exploiting raw hardware limits on consumer GPUs (e.g., RTX 3050 4GB Laptop).

---

## Goal 4: The 1,000+ tok/s Speculative Engine (Breaking the Single-Token DRAM Wall)

### Physics & Bottleneck Analysis
- **Physical DRAM Barrier**: Decode for a $0.5\text{B } Q4\_0$ model ($279.6\text{ MB}$) at $176\text{ GB/s}$ physical bandwidth has a strict hard theoretical floor:
  $$t_{\text{step, min}} = \frac{279.6\text{ MB}}{176\text{ GB/s}} = 1.59\text{ ms} \implies \text{Ceiling} = 628\text{ tokens/sec}$$
- **The Solution**: Multi-token speculative drafting with single-pass batched Tensor Core verification ($T=4\text{--}8$). By verifying $K$ drafted tokens in a **single DRAM weight load**, effective throughput reaches:
  $$\text{Effective Throughput} = \frac{1 + \mathbb{E}[\text{accepted}]}{t_{\text{batched\_step}}} \implies \mathbf{1{,}000\text{--}1{,}400\text{ tokens/sec}}$$

### Technical Substeps
1. **Substep 4.1: Unified Tensor Core Batched GEMV Verification Kernel**
   - Implement `k_gemv_wmma_batchT_q4_0`: 1 weight load pass in registers/shared memory, computing $T \in \{4, 8\}$ activations simultaneously.
   - Eliminate sequential layer re-runs; run entire draft tree forward in $1\times$ layer pass.
2. **Substep 4.2: CUDA-Resident Multi-Gram Trie Drafter**
   - Maintain a dynamic device-side token trie (2-gram, 3-gram, 4-gram) updated in-place via CUDA kernel without host roundtrips.
3. **Substep 4.3: Zero-Overhead Speculative Branch Rollback**
   - Implement atomic device-side rollback of KV cache slots and position indices on partial draft rejection.

### Acceptance Criteria
- Achieve sustained **$\ge 1{,}000\text{ tok/s}$** on repetitive / structured text (code, JSON, RAG, repetitive prompt continuation) on RTX 3050 Laptop.

---

## Goal 5: The 2-Bit Quantization Frontier ($14\text{B}$ Models in $4\text{GB}$ VRAM)

### Physics & VRAM Budget Math
- **Target Model**: `Qwen2.5-14B` or `DeepSeek-R1-Distill-Qwen-14B` (14.7B parameters).
- **At $Q4\_0$ ($4.5\text{ b/w}$)**: $8.3\text{ GB}$ (Requires $> 50\%$ CPU offload $\to$ slow).
- **At $IQ2\_XXS$ ($2.06\text{ b/w}$)**:
  $$\text{Weight Footprint} = 14.7 \times 10^9 \times 0.2575\text{ bytes} \approx \mathbf{3.78\text{ GB}} \le 4.0\text{ GB VRAM}!$$
- **Outcome**: Run a full $14\text{B}$ reasoning model **100% inside GPU VRAM** on a 4GB laptop with zero CPU layer offloading.

### Technical Substeps
1. **Substep 5.1: $IQ2\_XXS$ & $IQ2\_XS$ Codebook Dequantization in CUDA Shared Memory**
   - Port 256-element super-block codebook lookup tables ($8\times 8$ grid quantization) into CUDA `__shared__` memory or L1 cache.
   - Utilize 128-bit vectorized bit-extract and index lookups.
2. **Substep 5.2: $Q2\_K$ 2-Rows-Per-Warp GEMV Dispatch**
   - Implement `k_gemv_q2_K_v2` with 2-bit quants + 4-bit sub-block scale/min pairs.
3. **Substep 5.3: Numerical Parity Harness for I-Quants**
   - Validate logits parity ($< 0.15$ delta) vs reference `llama.cpp` $IQ2\_XXS$ oracle.

### Acceptance Criteria
- Load and run `Qwen2.5-14B-Instruct-IQ2_XXS.gguf` entirely inside 4GB VRAM without system RAM fallback, generating $> 35\text{ tok/s}$.

---

## Goal 6: Paged FlashAttention-3 & Chunked Virtual Memory for 128k Context

### Physics & Memory Fragmentation Math
- **Linear KV Allocation Problem**: At $128\text{k}$ context, allocating contiguous physical memory causes severe fragmentation and out-of-memory crashes even when total free VRAM is sufficient.
- **Paged Attention Virtualization**:
  - Physical allocation in $64$-token pages ($64 \times 2\text{ heads} \times 128\text{ dim} \times 0.5625\text{ bytes} = 9.2\text{ KB per page}$).
  - Virtual-to-physical block table lookup mapped directly into CUDA shared memory.

### Technical Substeps
1. **Substep 6.1: Paged KV Block Table Manager**
   - Implement static page allocator with zero dynamic OS/CUDA allocations.
   - Implement `k_paged_kv_scatter_q4_0` mapping token positions through virtual page table `int *d_block_tables`.
2. **Substep 6.2: Warp-Specialized FlashAttention-3 Split-K Decode Kernel**
   - Partition warps into producer (asynchronous page loading via `cp.async` into shared memory) and consumer (online softmax matrix multiplication).
   - Tile attention computation over non-contiguous physical pages with zero copy overhead.
3. **Substep 6.3: Context Sliding-Window Paging (128k $\to$ 256k Streaming)**
   - Implement O(1) page eviction and sliding-window recycling for infinite stream decoding.

### Acceptance Criteria
- Run a $128{,}000$-token context prompt on a 4GB GPU device using $Q4\_0$ paged KV cache with zero VRAM fragmentation.

---

## Goal 7: Zero-Copy POSIX IPC & Continuous Batching Server ($50{,}000\text{ req/s}$)

### Objective
Build a production-grade inference server architecture with zero-copy shared memory IPC and iteration-level continuous batching, outperforming standard HTTP/REST servers by $50\times$ in throughput and latency.

### Technical Substeps
1. **Substep 7.1: Continuous Batching Execution Loop**
   - Dynamic per-iteration batch assembly: interleave active prefill chunks (512 tokens) with decode tokens in a single forward pass.
2. **Substep 7.2: Lock-Free Shared-Memory Ring Buffer IPC (`libtinytorch_ipc.so`)**
   - Multi-process communication over POSIX shared memory (`shm_open`, `mmap`, atomic futexes).
   - Request-to-token latency $< 5\ \mu\text{s}$ (eliminating JSON serialization and HTTP socket overhead).
3. **Substep 7.3: Async Pipelined Worker Architecture**
   - Background tokenizer worker, GPU inference engine, and async streaming writer decoupled via lock-free rings.

### Acceptance Criteria
- Benchmark $> 50{,}000\text{ req/s}$ throughput over IPC ring buffer with median response latency $< 10\ \mu\text{s}$.
