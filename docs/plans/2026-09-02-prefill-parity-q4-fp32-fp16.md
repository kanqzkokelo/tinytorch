# Comprehensive Prefill Parity Plan (Q4, FP32, FP16 & Static Arena)

> Execution note: implement this plan task-by-task, one task per commit.

**Goal:** Reach multi-thousand tok/s prefill across ALL remaining data formats (Q4_0 KV, FP32 KV, FP16 KV) to close the prefill gap with llama.cpp, matching what Q8_0 KV achieved ($16,000\text{--}20,000\text{ tok/s}$).

---

## Root Causes Identified

1. **Q4 KV Cache in prefill is completely un-wired**:
   - In `prefill_batched_gemm` lines 3720-3760, when `e->use_q4_kvcache=1`, it falls through `else` and executes **FP32 scatter** and **FP32 flash**!
   - Result: Q4 KV prefill gets 0% speedup, stuck at 880 tok/s.
2. **`k_prefill_flash_fp32` uses uncoalesced scalar loads**:
   - Tile size `BC_PREFILL_FP32` is only 32.
   - Loads `sK[tok * head_dim + d] = Kc[g_idx]` element-by-element (32-bit scalar) instead of 128-bit `float4` loads.
3. **Driver Allocation Tax (10 `cudaMalloc` + 10 `cudaFree` per prefill)**:
   - Every single prefill call executes 10 `cudaMalloc` and 10 `cudaFree` calls for `d_X`, `d_Xn`, `d_Q`, `d_K`, `d_V`, `d_Att`, `d_H`, `d_G`, `d_U`, `d_pos_batch`.
   - On NVIDIA Linux driver, each call takes global locks and causes driver stalls. Pre-allocating in `Qwen2Engine` eliminates this overhead.

---

## Task 1: Static Arena for `prefill_batched_gemm` (Kill 20 Driver Malloc/Frees)

**Files:**
- Modify: `kernels/qwen2_cuda.cu`
- Modify: `include/qwen2_engine.h`

**Details:**
- In `qwen2_engine_create`, allocate prefill workspace buffers sized to `CHUNK_SIZE = 512` (or `max_prefill_chunk = 512`):
  `d_pf_X`, `d_pf_Xn`, `d_pf_Q`, `d_pf_K`, `d_pf_V`, `d_pf_Att`, `d_pf_H`, `d_pf_G`, `d_pf_U`, `d_pf_pos_batch`.
- In `prefill_batched_gemm`, reuse these persistent pointers instead of calling `cudaMalloc` and `cudaFree`.
- In `qwen2_engine_free`, free them once at destruction.

**Verification:**
```bash
./scripts/ci_local.sh
./scripts/verify.sh m61
```

---

## Task 2: Vectorize `k_prefill_flash_fp32` with `float4` (128-bit) Loads

**Files:**
- Modify: `kernels/qwen2_cuda.cu` (`k_prefill_flash_fp32`)

**Details:**
- Replace scalar loop `Kc[g_idx]` with `reinterpret_cast<const float4*>` vectorized 128-bit loads.
- Ensure `head_dim` (64 or 128) is divided by 4 for `float4`.
- Enlarge `BC_PREFILL_FP32` from 32 to 64 if shared memory permits ($64 \times 128 \times 4 \times 2 = 64\text{ KB} \le 100\text{ KB}$ on sm_86 Ampere).
- Measure FP32 pp512 tok/s jump from $870 \to 2{,}000+\text{ tok/s}$.

**Verification:**
```bash
python3 -c "print('The history of quantum computing dates back to the early 1980s. ' * 25)" > /tmp/pp512.txt
TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "$(cat /tmp/pp512.txt)" 8
```

---

## Task 3: Implement `k_prefill_flash_q4_0` and Batched Q4 KV Scatter

**Files:**
- Modify: `kernels/qwen2_cuda.cu`

**Details:**
- Implement `k_kv_scatter_q4_0_batched`: scatters $N$ tokens of K and V into `d_kc_q4` and `d_vc_q4` (`BlockQ4_0` format).
- Implement `k_prefill_flash_q4_0`: batched prefill flash attention directly reading `BlockQ4_0` KV cache in shared memory tiles (`BR_PREFILL=8`, `BC_PREFILL=64`).
- Wire into `prefill_batched_gemm`:
  `if (e->use_q4_kvcache) { k_prefill_flash_q4_0<<<...>>>(...); }`
- Measure Q4 pp512 tok/s jump from $880 \to 15{,}000+\text{ tok/s}$.

**Verification:**
```bash
TT_Q4_KV=1 TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "$(cat /tmp/pp512.txt)" 8
```

---

## Task 4: Full Multi-Type Prefill Benchmark & Doc Update

**Files:**
- Update: `LOOP.md` Cycle 18
- Update: `data/bench/results_scoreboard.md`

**Target:**
- FP32 pp512 $\ge 2{,}000\text{ tok/s}$
- Q4 pp512 $\ge 15{,}000\text{ tok/s}$
- Q8 pp512 $\ge 20{,}000\text{ tok/s}$
- All gates (`ci_local`, `m61`, `ple`) 100% GREEN.
