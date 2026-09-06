# FP16 Tensor-Core Prefill GEMM — Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** prefill 6.8k → 9k+ tok/s @pp759 by moving prefill GEMMs from FP32 SIMT path onto FP16 tensor cores. All prior work used `CUDA_R_32F` + `CUBLAS_COMPUTE_32F` (~5.5 of 7.5 TFLOPS SIMT peak); tensor peak is ~50 TFLOPS. GPT-5.6 prescription: FP16 shadow weights + FP16 inputs + FP32 accumulate (`CUBLAS_COMPUTE_32F_FAST_16F`).

**Architecture:** After model load (or lazily on first big prefill), dequantize each Q4 layer weight once to FP16 device buffer (~700MB, fits 4GB with graceful ABORT fallback if free <200MB — same pattern as banked `415b357`). New flag `TT_CUBLAS_FP16=1`: prefill GEMM call sites use `cublasGemmEx` (`CUDA_R_16F` in, `CUBLAS_COMPUTE_32F_FAST_16F`, FP32 or FP16 out) on FP16 shadow. Row-major mapping via OP_T/OP_N (exact lda/ldb/ldc per GPT-5.6 note in `data/profile/gpt56_code_solution.md` — read it). Optional N=759→768 padding only if edge tiles measure slow; keep exact-N first.

**Tech Stack:** cuBLAS (`kernels/cublas_ref.cu` — dlopen pattern, NO Makefile changes), prefill dispatch (`kernels/qwen2_cuda.cu`), gates (`ci_local.sh`, `verify.sh m61`).

**Global env:**
```bash
cd ~/Storage/repos/nnfromscratch
export PATH=$HOME/mmcuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf
```

**Clock discipline:** 2 warmups discarded + 3 measured = HOT median. Discard outliers >15% slow.

**Reference numbers (locked):**
- Baseline combo `TT_CUBLAS_PRE=1 TT_FA2_PRE=1`: 6768.6 hot median (112ms; o+mlp 76.15, flash 14.47)
- GEMM floor (FP32 path, all algos): ~83ms. Target FP16 path: 25–35ms → total ~60–70ms → 10–12k
- Known risk: FP16 weight rounding flipped pp759 argmax in first cuBLAS attempt (short prompts OK). FP32 accumulate mitigates; parity gate is binding.

---

### Task 1: Build FP16 shadow + tensor path (the bet)

**Files:** `kernels/cublas_ref.cu`, `kernels/qwen2_cuda.cu`, `src/` + `include/` as needed (shadow alloc). Do NOT touch docs/tests/Makefile/LOOP.md/data.

**Steps:**
1. Recon `kernels/cublas_ref.cu` (dlopen handle pattern from `415b357` — see `git show 415b357 --stat` for files touched) + current `tt_cublas_prefill_nt` wrapper.
2. FP16 shadow: dequant Q4→half once per weight (all 7 projections × 24 layers + verify size ~700MB–1GB); `nvidia-smi` free >200MB or ABORT with fallback to default path (log `[cublas-fp16] ABORT...`).
3. `TT_CUBLAS_FP16=1` routes prefill GEMMs to `cublasGemmEx` FP16/TENSOR_OP path (exact transpose/leading-dims per `data/profile/gpt56_code_solution.md`).
4. Correctness FIRST: short prompt coherent + greedy first-8 identical vs default; then pp759.

**Acceptance (ALL or `git checkout kernels/ src/ include/`, NO commit):**
- Combo `TT_CUBLAS_FP16=1 TT_FA2_PRE=1` pp759 HOT median ≥9000 tok/s
- Greedy first-8 IDENTICAL vs default: short + pp759; 400-tok probe argmax match, maxdiff <0.02
- `ci_local.sh` GREEN + `verify.sh m61` PASS + backfill PASS (default + combo flags)
- Commit: `prefill: FP16 shadow + tensor-core GEMM (TT_CUBLAS_FP16) 6.8k->9k+`

Time-box 90 min. If FP16 parity fails after trying FP32-out variant: report maxdiff, revert, NO commit.

---

### Task 2: Verify + document (Cycle 30)
- Independent hot-median reproduce + parity + gates → KEEP/REVERT.
- `LOOP.md` Cycle 30: numbers, parity data, next wall.
