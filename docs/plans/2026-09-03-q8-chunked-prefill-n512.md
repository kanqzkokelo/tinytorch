# Q8 Chunked Prefill N>512 Fix Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Make `TT_Q8_KV=1` prefill work for prompts longer than 512 tokens (currently `embed rc=716` on chunk 2).

**Architecture:** `qwen2_engine_prefill` splits `n > 512` into 512-token chunks calling `prefill_batched_gemm` per chunk. Chunk 1 succeeds, chunk 2 faults with stale `716` (real fault in chunk-2 batched kernels, not embed). Likely the same `BlockQ8_0` 34B misaligned-read class fixed in `2c23659`, but in a position-dependent path (slot addressing with advanced `e->pos`), or a chunk-2-only grid/S bug. Fix in place, keep single-chunk behavior identical.

**Tech Stack:** CUDA C, sm_86, `BlockQ8_0`, `qwen2_cuda.cu`.

---

### Task 1: Reproduce chunk-2 failure and isolate the faulting kernel

**REPRO FINDINGS (2026-09-03, inline):**
- `hello*1024` (1033 tok) Q8: FAIL (`embed rc=716 tok=23811`, then `tok=151644`). Base-sentence 2079 tok: FAIL same way.
- `fox*60+Hi` (1510 tok) Q8: PASS (67k tok/s). `hello*500` (509 tok): PASS (24-27k). `hello*504` (513 tok): FAIL deterministic 3/3.
- Boundary is EXACTLY 512/513: 509 = single chunk PASS; 513 = 512-chunk + 1-token tail (`chunk_len<32` → per-token `advance()` path) FAIL.
- So the fault is in the TAIL `advance()` at pos≈512 under Q8 (stale 716 surfaces at next embed), NOT in chunk-2 batched kernels. Suspect: eager `S` sizing (`S=(ctx+63)/64` vs graph-locked S) with 304 empty slices, or Q8 split/combine empty-slice handling — see hunt #3 P1-4. Fixer: instrument tail `advance()` (sync+check after each Q8 kernel in `forward_layers`), not the chunk loop.

**Files:**
- Read: `kernels/qwen2_cuda.cu:4840-4880` (`qwen2_engine_prefill` chunk loop)
- Test: `build/run_llm_gpu` with 1024-token prompt

**Step 1: Build a 1024-token prompt file**

```bash
cd ~/Storage/repos/nnfromscratch
python3 -c "print('The quick brown fox jumps over the lazy dog near the river bank while soft rain falls on the quiet village below the hills. ' * 60 + 'Hi.')" > /tmp/pp1024.txt
wc -c /tmp/pp1024.txt
```

**Step 2: Reproduce the failure**

```bash
export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
TT_MAX_CTX=10240 TT_Q8_KV=1 TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "$(cat /tmp/pp1024.txt)" 8 2>&1 | tail -3
```

Expected: FAIL with `[qwen2-engine] embed rc=716 ...` + `prefill failed`.

**Step 3: Confirm chunk 1 alone succeeds (512-token prefix)**

```bash
head -c 4500 /tmp/pp1024.txt > /tmp/pp512b.txt
TT_MAX_CTX=10240 TT_Q8_KV=1 TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "$(cat /tmp/pp512b.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1
```

Expected: PASS with multi-thousand tok/s prefill. This proves the fault is chunk-2-specific.

**Step 4: Commit nothing (repro only). Report which chunk fails.**

---

### Task 2: Instrument chunk 2 to find the faulting kernel

**Files:**
- Modify (temporarily): `kernels/qwen2_cuda.cu` `prefill_batched_gemm` — add sync+check
- Test: rebuild + rerun Task 1 Step 2

**Step 1: Add clear-sync-check around the Q8 flash launch in `prefill_batched_gemm`**

Find the `k_prefill_flash_q8_0<<<...>>>` launch site. Immediately after it, insert:
```c
{ cudaStreamSynchronize(e->stream); cudaError_t ce = cudaGetLastError(); if (ce != cudaSuccess) fprintf(stderr, "[CHUNK-DBG] flash q8 fail: %s (n=%d pos=%d)\n", cudaGetErrorString(ce), n, e->pos); }
```
Do the same after the `k_kv_scatter_q8_0_batched<<<...>>>` launch.

**Step 2: Rebuild and rerun**

```bash
export PATH=$HOME/mmcuda/bin:$PATH
make -j4 build/run_llm_gpu 2>&1 | tail -2
TT_MAX_CTX=10240 TT_Q8_KV=1 TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "$(cat /tmp/pp1024.txt)" 8 2>&1 | grep -E "CHUNK-DBG|embed rc" | head -5
```

Expected: `[CHUNK-DBG]` line names the real faulting kernel (flash or scatter), proving embed is innocent.

**Step 3: Remove the debug prints (revert to clean tree)**

```bash
git checkout kernels/qwen2_cuda.cu
```

**Step 4: Report the faulting kernel name + args (n, pos).**

---

### Task 3: Fix the chunk-2 Q8 fault

**Files:**
- Modify: `kernels/qwen2_cuda.cu` (faulting kernel from Task 2)

**Step 1: Read the faulting kernel and compare chunk-1 vs chunk-2 inputs**

Chunk 2 differs from chunk 1 in exactly two ways: `e->pos` is advanced (+512) and `toks` pointer is offset. Check:
- Slot addressing: `slot = (e_pos + local) % max_ctx` — any `%` or divide assuming pos==0?
- Grid/S sizing: any `S` or `num_q_tiles` computed from `n` (chunk_len, same both chunks — fine) vs from `e->pos`?
- `BlockQ8_0` reads: any `uint32_t`/`uint16_t` cast of `qs` (offset 2, misaligned → 716 on sm_86)? The `2c23659` fix covered `k_prefill_flash_q8_0` smem path — check the scatter kernel and any second-chunk-only path for the same pattern.

**Step 2: Apply the minimal fix (byte-wise copy pattern)**

```c
// WRONG (faults 716 on sm_86): misaligned 4B load at qs offset 2
((uint32_t*)&bk.qs[0])[j]
// RIGHT: byte-wise copy
for (int j = 0; j < 16; j++) { sK_q[row_off + j] = bk.qs[j]; }
```

**Step 3: Rebuild and verify chunk 2 passes**

```bash
make -j4 build/run_llm_gpu 2>&1 | tail -2
TT_MAX_CTX=10240 TT_Q8_KV=1 TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "$(cat /tmp/pp1024.txt)" 8 2>&1 | tail -2
```

Expected: clean English output, no `rc=716`, no `prefill failed`.

**Step 4: Commit**

```bash
git add kernels/qwen2_cuda.cu
git commit -m "prefill: fix Q8 chunk-2 fault at N>512 (same misalign class as 2c23659)"
```

---

### Task 4: Benchmark + gates + docs

**Files:**
- Modify: `LOOP.md` (Cycle 20 entry)

**Step 1: 3-run benchmark Q8 pp1024**

```bash
for i in 1 2 3; do TT_MAX_CTX=10240 TT_Q8_KV=1 TT_MODEL=data/models/qwen2.5-0.5b-instruct-q4_0.gguf ./build/run_llm_gpu "$(cat /tmp/pp1024.txt)" 8 2>&1 | grep -oE "prefill [0-9.]+ tok/s" | head -1; done
```

Expected: multi-thousand tok/s, stable across runs.

**Step 2: Run gates**

```bash
./scripts/ci_local.sh
./scripts/verify.sh m61
```

Expected: ALL PASS.

**Step 3: Document in LOOP.md (Cycle 20 entry)**

Record: faulting kernel, one-line root cause, pp1024 Q8 tok/s before (FAIL) vs after, gates status.

**Step 4: Commit**

```bash
git add LOOP.md
git commit -m "docs: Cycle 20 - Q8 chunked prefill N>512 fixed (pp1024 Q8 multi-k tok/s)"
```
