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
   - [ ] Microbenchmark Tensor Core FlashAttention-2 prefill kernel (`tools/micro_fa2_prefill.cu`)
   - [ ] Wire FlashAttention-2 prefill into `qwen2_engine_prefill`
   - [ ] Long-prompt prefill throughput evaluation
4. **[Phase 4] Production Server & Tools**:
   - [x] Upgrade `examples/server_minimal.c` with streaming SSE `/v1/chat/completions`
   - [x] Python test suite for OpenAI API compatibility (`tests/test_server_minimal.py` - 7/7 PASSED)
   - [x] Polish interactive multi-turn CLI `examples/chat_llm_gpu.c` with prompt templates
5. **[Phase 5] Benchmarking & Automated Suite**:
   - [x] Create automated benchmark comparison harness `bench/benchmark_suite.py`
   - [x] Full regression check across all models and test gates

---

## Cycle Log

### Cycle 0: Foundation & Plan Initialization
- **Action**: Initialized comprehensive 5-phase continuous loop plan and `LOOP.md`.
- **Baseline**: Qwen2.5-0.5B Q4_0 with Q8_0 KV Cache and FlashAttention-2 Decode.
- **Verification**: `ci_local.sh` and `verify.sh m61` all GREEN.

### Cycle 1: Architecture Expansion & RoPE Scaling
- **Action**: Added architecture aliases (`llama3`, `llama2`, `mistral`, `smollm`, `smollm2`) to `src/arch_registry.c` and implemented `k_rope_gptj_ff` in `kernels/qwen2_cuda.cu` for frequency-factor scaled GPT-J RoPE.
- **Verification**: `ci_local.sh` and `verify.sh m61` all GREEN.

### Cycle 2: K-Quant High-Performance V2 Dispatch
- **Action**: Enabled high-throughput 2-rows-per-warp V2 kernels (`k_gemv_q4_K_v2`, `k_gemv_q5_K_v2`, `k_gemv_q6_K_v2`) in `kernels/gemv_typed.cu` for K-quant models.
- **Verification**: `ci_local.sh` and `verify.sh m61` all GREEN.

### Cycle 3: OpenAI-Compatible Server SSE Streaming
- **Action**: Upgraded `examples/server_minimal.c` with real-time SSE streaming (`stream: true`), CORS headers, and temperature / top_p / repetition penalty controls.
- **Verification**: `pytest tests/test_server_minimal.py` 7/7 PASSED in 2.52s.

### Cycle 4: Automated Multi-Context Benchmark Suite
- **Action**: Created `bench/benchmark_suite.py` with multi-context evaluation and formatted Markdown reporting.
- **Live Measured Metrics** (`qwen2.5-0.5b-instruct-q4_0`):
  - $N=32$: Prefill **326.5 tok/s** | Decode **233.9 tok/s** ($4.28\text{ ms/tok}$)
  - $N=512$: Prefill **1,288.3 tok/s** | Decode **271.1 tok/s** ($3.69\text{ ms/tok}$)
  - $N=1024$: Prefill **1,140.4 tok/s** | Decode **257.7 tok/s** ($3.88\text{ ms/tok}$)
  - $N=1558$: Prefill **990.5 tok/s** | Decode **242.8 tok/s** ($4.12\text{ ms/tok}$)
- **Verification**: All gates (`ci_local.sh`, `verify.sh m61`, `verify.sh ple`) 100% GREEN.
