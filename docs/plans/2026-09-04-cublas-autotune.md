# cuBLAS Algo Autotune — Implementation Plan

> Execution note: implement this plan task-by-task, one task per commit.

**Goal:** prefill 6.4–6.8k → 7.5–8.5k tok/s @pp759 by autotuning cuBLAS algorithm selection per GEMM shape. o+mlp 76.7ms (65% of prefill) is the whole game; prior fusion attempt proved algo choice swings 10x (0.5 vs 5.5 TFLOPS).

**Architecture:** At engine init (or first big prefill), benchmark candidate algos for each distinct GEMM shape (gate/up 4864×896, down 896×4864, qkv/o 896×896, N≈512/759) via `cublasGemmEx` algo parameter or cublasLt heuristics; cache winning algo per shape; use for all prefill GEMMs under `TT_CUBLAS_PRE=1`. Revert to heuristic default if no algo beats it by ≥10%.

**Tech Stack:** cuBLAS/cuBLASLt (`kernels/cublas_ref.cu`), prefill dispatch (`kernels/qwen2_cuda.cu`), profiler (`TT_PROFILE=1`), gates (`ci_local.sh`, `verify.sh m61`).

**Global env (every shell task):**
```bash
cd ~/Storage/repos/nnfromscratch
export PATH=$HOME/mmcuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf
```

**Clock discipline (no sudo, clocks float 682MHz–1965MHz):** every benchmark = 2 warmup runs discarded + 3 measured, report HOT median. If first measured run >15% slower than other two, discard + run one more.

**Reference numbers (locked):**
- Combo `TT_CUBLAS_PRE=1 TT_FA2_PRE=1` pp759: ~6.4–6.8k tok/s (~115–118ms)
- Stage split: qkv 7.3 / o+mlp 76.7 / flash 14.6 / swiglu 6.1 / norm+resid ~5 / gap ~5ms
- Prior data point: fused gate+up M=9728 hit cuBLAS algo fallback → 0.5 TFLOPS (algo choice dominates)

---

### Task 1: Lock baseline + per-shape GEMM microbenchmark (read-only + scratch)

**Files:** none committed (scratch test program under /tmp only)

**Step 1: Regenerate prompt + confirm baseline**
```bash
python3 -c "print('The quick brown fox jumps over the lazy dog near the river bank while soft rain falls on the quiet village below the hills. ' * 30)" > /tmp/pp759.txt
for i in 1 2; do TT_CUBLAS_PRE=1 TT_FA2_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | tail -1; done
for i in 1 2 3; do TT_CUBLAS_PRE=1 TT_FA2_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
TT_PROFILE=1 TT_CUBLAS_PRE=1 TT_FA2_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -iE "gemm|mlp|flash|total" | head -6
```
Expected: hot median ~6.4–6.8k; o+mlp ~76ms.

**Step 2: Write /tmp/cublas_shape_bench (scratch, NOT in repo):** time each prefill GEMM shape standalone (gate 4864×896×N, down 896×4864×N, o 896×896×N; N=512 and 759) with current default algo, 100 iters, report ms + effective TFLOPS. Identifies which shape is worst vs roofline.

**Step 3: Report table** (shape | N | ms | TFLOPS) + baseline numbers. No commit.

---

### Task 2: Algo autotune + wire-in (the bet)

**Files:**
- Modify: `kernels/cublas_ref.cu` (autotune table + selection)
- Modify: `kernels/qwen2_cuda.cu` (call sites use tuned algo — flag-gated)
- Test: greedy-compare + backfill + gates

**Step 1: Implement autotune** (runs once at first big prefill or engine init under `TT_CUBLAS_PRE=1`):
- Candidate set: `CUBLAS_GEMM_DEFAULT`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, plus `cublasGemmEx` algo IDs 0–7 where valid for FP32/compute_32F; or cublasLt `cublasLtMatmulAlgoGetHeuristic` top-4 per shape.
- Per distinct shape (M,K,N): time 20 iters each candidate (after 5 warmup), keep winner iff ≥10% faster than default; else keep default.
- Cache in static table keyed by (M,K,N,transA,transB); print chosen algos to stderr once (`[cublas-tune] ...`).

**Step 2: Wire call sites** so prefill GEMMs pass their shape key and use the tuned algo. Default path (no flag) untouched.

**Step 3: Verify + benchmark (acceptance ALL):**
```bash
make -j4 build/run_llm_gpu 2>&1 | tail -2
TT_CUBLAS_PRE=1 TT_FA2_PRE=1 ./build/run_llm_gpu "Paris is the capital" 8 2>&1 | tail -1   # greedy-identical vs no-flag
for i in 1 2; do TT_CUBLAS_PRE=1 TT_FA2_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 >/dev/null 2>&1; done
for i in 1 2 3; do TT_CUBLAS_PRE=1 TT_FA2_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
```
Acceptance: HOT median ≥7000 tok/s AND greedy first-8 identical (short + pp759) AND `ci_local.sh` GREEN AND `verify.sh m61` PASS AND backfill PASS (default + flags).

**Step 4: Commit ONLY if ALL met:** `prefill: per-shape cuBLAS algo autotune (...)`. Else `git checkout kernels/ src/ include/`, NO commit, report best-per-shape table.

---

### Task 3: Verify + document

**Files:** `LOOP.md` (Cycle 29 entry) — owner commits docs.
- Independent hot-median reproduce, greedy check, gates. KEEP/REVERT verdict.
- Doc: per-shape winners + TFLOPS, final prefill number, next wall named.
