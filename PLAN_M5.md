# PLAN_M5.md — Milestone 5: High-Speed 2B LLM GGUF Inference Engine

## Objective
Implement a lightweight, zero-overhead C + CUDA Transformer inference engine in `tinytorch` capable of loading GGUF `q4_0` quantized models (e.g. Qwen2.5-0.5B / 1.5B, Llama-3.2-1B) and generating text at **> 130 tokens/sec**, outperforming standard `llama.cpp` CUDA backend on RTX 3050.

---

## 1. Core Components

### A. GGUF Parser (`src/loader_gguf.c`, `include/loader_gguf.h`)
- Reads GGUF v2/v3 header, magic number (`GGUF`), key-value metadata (architecture, context length, embedding dim, head count, layer count).
- Memory-maps (`mmap`) tensor metadata and binary payload.
- Resolves tensor offset tables into CUDA memory allocations.

### B. Fused INT4 (`q4_0`) GEMV CUDA Kernel (`kernels/gemv_q4_cuda.cu`)
- **Block Layout**: `q4_0` quantizes 32 FP16 values per block into 16 bytes of packed nibbles + 1 FP16 scale (`d`).
- **Kernel Strategy**:
  1. Threads load 128-bit `uint4` vectors directly from VRAM (16 bytes = 32 packed weights).
  2. In-register nibble unpacking: `low_nibble = byte & 0x0F - 8`, `high_nibble = (byte >> 4) - 8`.
  3. Single-pass multiply-accumulate with FP16 activation vector $x$.
  4. Warp shuffle reduction (`__shfl_down_sync`) to aggregate dot product across 32 lanes.

### C. Fused Transformer Operators (`src/ops_llm.c`, `include/ops_llm.h`)
- `tt_rmsnorm_cuda`: $y_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum x_k^2 + \epsilon}} \times \gamma_i$. Fused block reduction.
- `tt_rope_cuda`: In-place Rotary Position Embeddings applied to Query and Key projections.
- `tt_swiglu_cuda`: Fused activation $y = (x \cdot w_{gate} \cdot \text{sigmoid}(x \cdot w_{gate})) \cdot (x \cdot w_{up})$.
- `tt_kvcache`: Fixed contiguous $K, V$ allocation in GPU VRAM with zero runtime allocation overhead.

### D. Benchmark & Verification Harness (`tests/gate_llm_bench.py`)
- Downloads `Qwen2.5-0.5B-Instruct-GGUF` (~390MB) or `Qwen2.5-1.5B-Instruct-GGUF` (~980MB).
- Runs token generation loop in `tinytorch` and measures exact end-to-end throughput in tokens/sec.
- Compares head-to-head against `llama.cpp` CUDA backend on identical prompts.

---

## 2. Milestone 5 Verification Gate Criteria
- **Gate A (GGUF & Op Parity)**: All LLM ops (`rmsnorm`, `rope`, `swiglu`, `q4_0_gemv`) match golden Python/NumPy outputs (`rtol=1e-3`).
- **Gate B (LLM Throughput)**: `tinytorch` achieves **> 120 tokens/sec** on 0.5B/1.5B INT4 models on RTX 3050 GPU.
