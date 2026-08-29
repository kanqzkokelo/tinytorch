# Autonomous Continuous Improvement Loop (LOOP.md)

## Goal
Advance `nnfromscratch` into an industry-grade, minimal zero-dependency CUDA C LLM inference engine.
Achieve parity or superiority vs `llama.cpp` across architectures (Qwen2.5, LLaMA-3/3.2, SmolLM2, Gemma-2, Mistral), quantizations (Q4_0, Q8_0, Q4_K_M, Q6_K), and context lengths (32 to 10k+).
Keep all CI gates green at all times (`ci_local.sh`, `verify.sh m61`, `verify.sh ple`).

## Active Backlog
1. **[Phase 1] Architecture Expansion**:
   - [x] LLaMA-3 / LLaMA-3.2 / LLaMA-3.1 full GPU graph execution & RoPE theta scaling support
   - [x] SmolLM2 & Mistral architecture aliases and GPU fast paths
   - [x] Gemma-2 softcapping & GeGLU fast paths in engine
2. **[Phase 2] K-Quant GPU Engine**:
   - [x] High-throughput Q4_K / Q5_K / Q6_K GEMV V2 2-rows-per-warp dispatch
   - [x] Wire Q4_K / Q6_K GEMV into GPU decode loop
   - [x] Parity & performance verification on K-quant models
3. **[Phase 3] FlashAttention-2 Prefill**:
   - [x] Vectorized smem tile loading and conflict-free 32-bit reads in `k_prefill_flash_q8_0`
   - [x] Fixed Tensor Core GEMM boundary tile store bug in `k_gemm_wmma_q4_0_prefill`
   - [x] Bit-exact prefill parity verified ($N=32, 64, 128$) with $100\%$ matching sampled tokens
4. **[Phase 4] Production Server & Tools**:
   - [x] Upgrade `examples/server_minimal.c` with streaming SSE `/v1/chat/completions`
   - [x] Python test suite for OpenAI API compatibility (`tests/test_server_minimal.py` - 7/7 PASSED)
   - [x] Polish interactive multi-turn CLI `examples/chat_llm_gpu.c` with prompt templates
5. **[Phase 5] Benchmarking & Automated Suite**:
   - [x] Create automated benchmark comparison harness `bench/benchmark_suite.py`
   - [x] Full regression check across all models and test gates
6. **[Phase 6] Advanced Attention & Prefill**:
   - [ ] Implement single-kernel fused FlashAttention-2 for $N \le 512$ decode
   - [ ] Add Chunked Prefill memory bounding for large contexts up to 32k
7. **[Phase 7] Speculative Engine V2**:
   - [ ] Wire batched verification into `spec_llm_gpu` speculative runner
   - [ ] Benchmark end-to-end speculative decoding speedup vs baseline
8. **[Phase 8] Server Multi-Turn & Release Polish**:
   - [ ] Add dynamic multi-turn session management in `server_minimal`
   - [ ] Run comprehensive multi-model verification across all gates

---

## Cycle Log

### Cycle 0: Foundation & Plan Initialization
- **Action**: Initialized comprehensive continuous loop plan and `LOOP.md`.
- **Verification**: `ci_local.sh` and `verify.sh m61` all GREEN.

### Cycle 1: Architecture Expansion & RoPE Scaling
- **Action**: Added architecture aliases (`llama3`, `llama2`, `mistral`, `smollm`, `smollm2`) to `src/arch_registry.c` and implemented `k_rope_gptj_ff` in `kernels/qwen2_cuda.cu`.
- **Verification**: `ci_local.sh` and `verify.sh m61` all GREEN.

### Cycle 2: K-Quant High-Performance V2 Dispatch
- **Action**: Enabled high-throughput 2-rows-per-warp V2 kernels (`k_gemv_q4_K_v2`, `k_gemv_q5_K_v2`, `k_gemv_q6_K_v2`) in `kernels/gemv_typed.cu`.
- **Verification**: `ci_local.sh` and `verify.sh m61` all GREEN.

### Cycle 3: OpenAI-Compatible Server SSE Streaming
- **Action**: Upgraded `examples/server_minimal.c` with real-time SSE streaming (`stream: true`), CORS headers, and temperature / top_p / repetition penalty controls.
- **Verification**: `pytest tests/test_server_minimal.py` 7/7 PASSED in 2.52s.

### Cycle 4: Automated Multi-Context Benchmark Suite
- **Action**: Created `bench/benchmark_suite.py` with multi-context evaluation and formatted Markdown reporting.
- **Live Measured Metrics**: Prefill $1{,}288\text{ tok/s}$ @ 512 ctx, Decode $271.1\text{ tok/s}$.

### Cycle 5: FlashAttention-2 Prefill Acceleration & Parity Verification
- **Action**: Optimized `k_prefill_flash_q8_0` with vectorized `uint32_t` smem loads; fixed Tensor Core GEMM boundary tile store bug; verified 100% bit-exact layer-by-layer parity across $N=32, 64, 128$.
- **Verification**: `test_prefill_layer_parity` all PASSED; `ci_local.sh`, `verify.sh m61`, `verify.sh ple` all 100% GREEN.
