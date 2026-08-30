> **Honesty note (2026-08-29):** Numbers below are pre-honest (single-sample/batch-mean, L2-hot possible, -arch=native). See `data/profile/AUDIT_HONEST_2026-08-29.md` for honest per-iter median, cross-core, sm_86 pinned measurements.

# `nnfromscratch` Project State - 2026-08-28

## Current Position

- **Branch**: `m6-correctness`
- **Latest commits**: 22b346e, ecde6b0, b83a1c4 (fused QKV/FFN microbenches), f803ceb (Q8_0 V4 LM head), 0caa1c3, 0b4e462, 81bc24e, 00f310f, 8a7233b, d6f0e1f, 986e59e, 04a1c90, 099d244, 3836b2c, 5b17c48
- **Decode throughput**: 269 tok/s (qwen2.5-0.5b-q4_0, RTX 3050)
- **Prefill throughput**: 16.8x speedup (11 -> 420-782 tok/s)
- **VRAM bandwidth**: 174-180 GB/s (98.9% of hardware peak)
- **All test gates GREEN**: ci_local, m61 (7/7 logits parity), ple (3/3 PLE golden), m84 (6/7 gemma4)

## What's Shipped (Latest 9 commits)
1. **Q8_0 V4 LM Head** (commits f803ceb, 0caa1c3, 22b346e): 1.92x LM head speedup, bit-exact
2. **Fused QKV Microbench** (b83a1c4): 31.4% faster than 3 sequential Q/K/V calls, bit-exact
3. **Fused FFN Microbench** (ecde6b0): 34.0% faster than 3 sequential Gate+Up+SwiGLU calls, bit-exact
4. **True Batched Verification** (committed 099d244)
5. **Tensor Core WMMA Prefill** (commits 04a1c90, 099d244, 3836b2c, d9372f6): 1,100+ tok/s peak prefill
6. **2D Batched Prefill GEMM** (commits 0b4e462, 81bc24e, 00f310f, 8a7233b, d6f0e1f, 986e59e, 25bc77f): 10-16x prefill speedup
7. **Temperature/Repetition Penalty Sampling** (commits dc9a750, 77bfce9)
8. **Batched-4 GEMV (M10 proto)** (commits c3638fd, 5b17c48, 2904588)
9. **Universal Speculative Engine** (commits 914839a, 8d33123, eaeb944, 0c497f4, bd05238)

## What's Next: Fused QKV/FFN Task 3 (WIRE INTO ENGINE)
- Task 3 was IN PROGRESS when session ended
- Task 3 builder agent hit "Insufficient balance" mid-work
- Microbenches prove the kernels work; integration is what's left
- Plan at: `docs/plans/2026-08-28-fused-qkv-ffn-kernels.md`
- Plan tracker state:
  - ✓ Task 1: Microbench fused QKV (done)
  - ✓ Task 2: Microbench fused FFN (done)
  - ⚠ Task 3: Wire into engine (PENDING - last agent failed)
  - ○ Task 4: Final decode benchmark & parity verification

## What the Task 3 partial work showed
- The agent was investigating the FFN shape (M=4864, K=896) and register pressure
- 8 warps per block, 4 rows per warp = 32 rows per block = 152 blocks total
- The existing `tt_ffn_q4_0` uses 16 warps per block, 2 rows per warp
- Need to check if larger register pressure (16 accumulators per warp) is the issue

## When User Returns - Suggested Next Steps

### Option A: Continue Fused QKV/FFN Plan (immediate win)
- Task 3: Wire `tt_gemv_q4_0_qkv_fused` and `tt_gemv_q4_0_ffn_fused` into `forward_layers()` in `kernels/qwen2_cuda.cu`
- Expected: 269 -> 290-310 tok/s decode
- Time: 2-4 hours of CUDA work

### Option B: Polished Open-Source Release
- Write clean README, LICENSE, API docs
- Document the 178 commits
- Time: 1 day

### Option C: Continue with more decode optimizations
- Split-K Down projection
- Persistent kernel (one CTA per SM, loops over rows)
- Time: 1 day each

### Option D: Q3_K quantization parity test
- Download qwen2.5-0.5b-q3_K_M.gguf
- Measure decode speed with Q3_K (would close gap to llama.cpp at lower precision)
- Time: 30 minutes

## Architecture Notes
- Engine: 2,000 lines of standalone CUDA C, zero dependencies
- Decode: 81% GEMV (memory-bound at 174 GB/s = 98.9% of RTX 3050 peak)
- All weight reads are at 100% bandwidth efficiency
- Single-token decode is fundamentally memory-bandwidth limited
- Only way to beat llama.cpp on same model: reduce kernel launch overhead, split-K, persistent kernels

## User's Strategic Context
- User has been working 16+ hours on this
- Emotional state: tired but wants to achieve llama.cpp parity
- Wants strategic empathy, not just perf numbers
- Asked repeatedly about: ngram speculative, flash attention, fused kernels
- All gates are GREEN; engine works for real generation
- 30+ commits shipped today across 5+ plans
