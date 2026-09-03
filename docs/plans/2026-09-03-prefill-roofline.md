# Prefill Roofline Chase Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Take prefill from ~2.8k tok/s (WMMA FP16) to ~8-11k tok/s (~4x, llama.cpp parity) by climbing the GEMM roofline; document the physics wall (~50k tok/s) honestly.

**Architecture:** Prefill is 78% GEMM (o+mlp stage alone ~200ms @pp759). Three independent attacks, each gated on measured speedup with greedy-identical output: (1) INT8 tensor GEMM — Q4 nibbles map exactly to int8 (q-8), per-block float scale folds into epilogue, ~2x tensor rate of FP16; (2) fuse SwiGLU elementwise into down-projection X-load, killing the d_H round-trip (2×n×FF×4B per layer); (3) autotune CTA tiles (128×32 vs 256×64) + split-K on N, keep the winner. Each phase commits only on ≥15% measured prefill gain with gates green; else revert.

**Tech Stack:** CUDA WMMA (`nvcuda::wmma`, sm_86, `mma.sync` s8/s16 via `wmma::experimental` or inline PTX), `kernels/gemv_q4_cuda.cu`, `kernels/qwen2_cuda.cu`, profiler (`TT_PROFILE=1`), gates (`ci_local.sh`, `verify.sh m61`).

**Global env (every shell task):**
```bash
cd ~/Storage/repos/nnfromscratch
export PATH=$HOME/mmcuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf
```

**Reference numbers (lock these as baseline in Phase 0):**
- CTA tile today: 128×32, 8 warps, `mma.m16n16k16` FP16, X fp32→half on load, Q4→half dequant, FP32 out (`kernels/gemv_q4_cuda.cu:1580-1700`, wrapper `:2007-2013`)
- FFN sequence: gate GEMM → up GEMM → `k_swiglu_apply` (`kernels/qwen2_cuda.cu:2531`) → down GEMM (`:4514-4530`)
- Dispatch: `prefill_gemm_fn` pointer (`:4205-4211`), `TT_USE_WMMA_PRE=1` + `n>=64` selects WMMA
- K dims on this model: dim 896, FF 4864 (both %32==0 — required by K/32 tiling)

---

### Task 1: Lock baseline (read-only)

**Files:** none (measure only)

**Step 1: Regenerate prompt + measure 3-run medians**

```bash
python3 -c "print('The quick brown fox jumps over the lazy dog near the river bank while soft rain falls on the quiet village below the hills. ' * 30)" > /tmp/pp759.txt
for i in 1 2 3; do ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
for i in 1 2 3; do TT_USE_WMMA_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
```

Expected: FP32 ~1400, WMMA ~2300-2800.

**Step 2: Stage breakdown**

```bash
TT_PROFILE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -iE "qkv|mlp|flash|norm|total" | head -8
```

Expected: GEMM ~78%, flash ~21%.

**Step 3: Record numbers in `data/profile/roofline_baseline.md` (create, 10 lines: prefill medians + stage ms). Commit docs.**

```bash
git add data/profile/roofline_baseline.md
git commit -m "docs: prefill roofline baseline (WMMA 2.8k, GEMM 78%)"
```

---

### Task 2: INT8 WMMA prefill GEMM (the 2x bet)

**Files:**
- Modify: `kernels/gemv_q4_cuda.cu` (new kernel + wrapper after `tt_gemm_wmma_q4_0_prefill:2007`)
- Modify: `kernels/qwen2_cuda.cu:4205-4211` (dispatch: `TT_USE_WMMA8_PRE=1` + n>=64 + q4_0-only → new fn)
- Test: greedy-compare + `build/test_qcache_backfill`

**Step 1: Write `k_gemm_wmma_q4_int8_prefill` (new kernel, do NOT modify the FP16 one)**

Math: Q4 nibble `q in 0..15` → int8 `v = q-8` (exact, no rounding). Per-32-block: `out += v * d_w` where `d_w` = block fp16 scale. X row: dynamic quant `xs = x / (max|x|/127)`, int8. Accumulate `wmma s8` → int32, epilogue: `y = acc * d_w * (max|X|/127)`. Same 128×32 CTA / 8-warp skeleton as the FP16 kernel — copy it, change: `s_X` to `int8` (quantize on load with row absmax via 2-pass: first pass absmax in registers, `__shfl_sync` row-max, second pass store), `s_W` to `int8` (`q-8` direct, NO fp scale multiply), fragments `wmma::experimental::precision::s8`, accumulator int32, epilogue multiplies `d_w * x_scale`.

```c
// X-load sketch (per row r in 32x32 tile):
float amax = 0; for (c in 0..31) amax = fmaxf(amax, fabsf(dX[g_n*K + k_base + c]));
amax = row_reduce_max(amax); float xinv = 127.0f / fmaxf(amax, 1e-12f);
for (c in 0..31) s_X[r][c] = (int8_t)__float2int_rn(dX[...] * xinv);
// store xinv per row in smem float s_XS[32] for epilogue
// W-load sketch: s_W[m][k] = (int8_t)((j&1) ? (qs>>4)-8 : (qs&15)-8); s_WD[m] = __half2float(d)
// Epilogue: y = (float)acc * s_WD[m] * (s_XS[row]/127)
```

K/32 tiling requirement unchanged (K %32==0 holds: 896, 4864).

**Step 2: Wrapper + dispatch (mirror existing pattern)**

```c
int tt_gemm_wmma_q4_int8_prefill(const void *dW, const float *dX_NxK, float *dY_NxM,
                                 int M, int K, int N, cudaStream_t stream) {
    dim3 grid((M + 127) / 128, (N + 31) / 32);
    dim3 block(256);
    k_gemm_wmma_q4_int8_prefill<<<grid, block, 0, stream>>>(dW, dX_NxK, dY_NxM, M, K, N);
    return (int)cudaGetLastError();
}
```
Dispatch: `if (getenv("TT_USE_WMMA8_PRE") && n >= 64) prefill_gemm_fn = tt_gemm_wmma_q4_int8_prefill;` (q4_0-only call sites, same guard as WMMA).

**Step 3: Build + correctness first**

```bash
make -j4 build/run_llm_gpu 2>&1 | tail -2
TT_USE_WMMA8_PRE=1 ./build/run_llm_gpu "Paris is the capital" 8 2>&1 | tail -1
./build/run_llm_gpu "Paris is the capital" 8 2>&1 | tail -1
```

Expected: both coherent English, first-8 tokens identical (int8 X-quant noise must not flip argmax on this prompt).

**Step 4: Benchmark (acceptance: ≥40% faster than FP16-WMMA o+mlp)**

```bash
TT_PROFILE=1 TT_USE_WMMA8_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -iE "mlp|total" | head -3
for i in 1 2 3; do TT_USE_WMMA8_PRE=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
```

Expected: o+mlp stage down ~40-50%, prefill ≥3800 tok/s.

**Step 5: Commit ONLY if Step 4 passes + gates green**

```bash
./scripts/ci_local.sh 2>&1 | tail -1
./scripts/verify.sh m61 2>&1 | tail -2
make build/test_qcache_backfill && TT_USE_WMMA8_PRE=1 ./build/test_qcache_backfill 2>&1 | tail -1
git add kernels/gemv_q4_cuda.cu kernels/qwen2_cuda.cu
git commit -m "prefill: INT8 WMMA GEMM (Q4 nibbles exact, X dynamic quant) pp759 2.8k->3.8k+"
```

If Step 4 fails (<40% or wrong output): `git checkout kernels/` and document the measured numbers in the report — do NOT commit.

---

### Task 3: Fuse SwiGLU into down-projection X-load

**Files:**
- Modify: `kernels/gemv_q4_cuda.cu` (new kernel `k_gemm_wmma_down_fused` OR fuse into winner of Task 2)
- Modify: `kernels/qwen2_cuda.cu:4530` (down call site: pass d_G + d_U instead of d_H when flag set)
- Test: greedy-compare

**Step 1: Write fused down kernel**

Copy the Task-2-winning down GEMM kernel. Change ONLY the X-load: instead of reading `d_H[row*FF + k]`, compute `silu(g)*u` on the fly:

```c
// fused X element (then quantize to int8/half exactly as the base kernel does):
float g = d_G[row * FF_l + k_global];
float u = d_U[row * FF_l + k_global];
float h = (g / (1.0f + expf(-g))) * u;   // silu; act_gelu variant: use tanh approx from k_swiglu_apply:2531
// ... feed h into the existing quantize-and-store path
```

This kills the `d_H` write (down-GEMM... no: kills `k_swiglu_apply`'s read of d_G+d_U AND write of d_H, plus down-GEMM's read of d_H): saves 3×n×FF×4B traffic per layer (2 reads + 1 write → 0.5 read of G+U... precisely: G,U still read once each by fused kernel instead of twice total). Net saved: 1×n×FF×4B per layer ≈ 11MB @759 ×24 layers ≈ 270MB (~5-8% of prefill traffic).

Gate with `TT_FUSE_SWIGLU=1` (independent flag so it composes with WMMA8).

**Step 2: Wire call site** (`qwen2_cuda.cu` down projection): `if (fuse_swiglu && w->down.dtype == TTQ_Q4_0) fused_fn(w->down.ptr, d_G, d_U, d_Xn, dim, FF_l, n, act_gelu, stream); else <existing>`. Skip `k_swiglu_apply` launch when fused.

**Step 3: Verify + benchmark (acceptance: greedy-identical 8 tokens + ≥10% prefill gain)**

```bash
make -j4 build/run_llm_gpu 2>&1 | tail -2
TT_USE_WMMA8_PRE=1 TT_FUSE_SWIGLU=1 ./build/run_llm_gpu "Paris is the capital" 8 2>&1 | tail -1
for i in 1 2 3; do TT_USE_WMMA8_PRE=1 TT_FUSE_SWIGLU=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
```

**Step 4: Commit ONLY if acceptance met + `ci_local.sh` GREEN + m61 PASS.** Else revert `git checkout kernels/`.

---

### Task 4: Autotune tiles + split-K, keep winner

**Files:**
- Modify: `kernels/gemv_q4_cuda.cu` (parameterize CTA_M/CTA_N via template or `#define` variants)
- Test: `tools/bench_prefill.c` or direct runs (check `make build/bench_prefill` exists)

**Step 1: Benchmark current tile vs 256×64**

Add `#define CTA_M 128 / CTA_N 32` variant + `256/64` variant of the winning kernel (Task 2/3). Measure o+mlp stage ms via `TT_PROFILE=1` @pp759, 3 runs each. Keep winner iff ≥10% better.

**Step 2: Split-K on N for short-n only if occupancy-bound**

Check: `n=759 → N-tiles = 24`; M=4864 → 38 CTAs; total 912 CTAs — occupancy fine, SKIP split-K. Only implement if measured SM underutilization (`nvidia-smi dmon` SM% <60% during prefill). Document the reading either way.

**Step 3: Commit winner (gates green) or document no-op.**

---

### Task 5: Final matrix + honest wall documentation

**Files:**
- Modify: `LOOP.md` (new Cycle entry), `data/profile/roofline_baseline.md` (append final)

**Step 1: 3-run medians**

```bash
for i in 1 2 3; do TT_USE_WMMA8_PRE=1 TT_FUSE_SWIGLU=1 ./build/run_llm_gpu "$(cat /tmp/pp759.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
```

**Step 2: Gates sweep**

```bash
./scripts/ci_local.sh 2>&1 | tail -1
./scripts/verify.sh m61 2>&1 | tail -2
```

**Step 3: Document**: per-task shipped/reverted with measured numbers, final vs llama.cpp (7941-11236), physics wall math (50 TFLOPS ceiling → ~50k tok/s absolute cap, we reached X%), next bottleneck named (flash 21%? decode?).

**Step 4: Commit docs.**
