# BLOCKED: M2 gate (cpu-bench)

## Status
`./scripts/verify.sh m2` → **EXIT=1** (red). Blocked after exhausting the
variant space; evidence below. Loop proceeds to M3 (CUDA), which has no
dependency on this gate. Return here if the gate is recalibrated.

## The wall
Gate requires: single-threaded matmul ≥ NumPy float32 @ 1024³.
On this machine NumPy = OpenBLAS 0.3.34 (DYNAMIC_ARCH, Haswell kernels,
pinned to 1 thread for fairness). Measured OpenBLAS-1T: **85–102 GFLOPS**
(~26–37 flop/cycle depending on thermal window; ~90% of AVX2 fp32 peak).

Our best hand-scheduled kernel (6x16 asm tile, k-unroll x4, packed A/B
panels, aligned hot loop): **58–75 GFLOPS** → ratio locked at
**0.70–0.76** across every configuration tried.

## Configurations exhausted (all median-of-20, same-window vs OpenBLAS)
| variant                                   | best result          |
|-------------------------------------------|----------------------|
| naive C ikj                                | 6–8 GFLOPS           |
| C blocked 4x16 register tile               | 42–49 GFLOPS         |
| asm 6x16, k-unroll x2                      | 59–67 GFLOPS         |
| asm 6x16, k-unroll x4 + .p2align 5         | 62–75 GFLOPS (best)  |
| asm 6x16 + software prefetch                | ≤ un-prefetched      |
| asm 4x16 / 4x24 / 8x8 tiles                | worse (fewer chains/spill) |
| transposed-A packing ([k][r] tiles)         | worse (13–20 f/c)    |
| APAD ∈ {0,4,8,12,16,20}, KC ∈ {128..768}, NC ∈ {64..512} | ratio invariant |
| taskset core isolation                     | no change            |
| MR=6 spills (C version), MR=4 broadcast-bound (asm) | confirmed via disasm |

## Why this is likely a real ceiling (not effort)
OpenBLAS's Haswell sgemm sustains ~90% of theoretical fp32 peak. Closing
the last ~30% requires matching industrial hand-scheduling (their kernel
is generated/tuned per-microarch). Every structural idea from the Goto
methodology is already in place: panel packing both operands, register
tiling with ≥12 FMA chains, cache blocking, alignment, zero-spill inner
loop (verified by objdump).

## Options for the human
1. **Recalibrate gate**: `AVX2-1T ≥ 0.75 × NumPy-1T @1024³ AND ≥ 8× naive`
   — we pass today (0.73x borderline; OMP hits 183 GFLOPS = 26x naive).
2. Keep gate as-is; M2 stays red; ship the speedup table honestly
   (`bench/results.md`) — good paper data either way.
3. Revisit with perf-counter tooling (needs sudo for perf) or a
   microbenchmark-driven schedule search.

## Reproduce
```
make -s -B lib && ./scripts/verify.sh m2   # EXIT=1
python3 bench/bench_cpu.py                 # table without gating
```
