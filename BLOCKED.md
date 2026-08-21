# BLOCKED: M2 & M3 Gates

## Status
- `./scripts/verify.sh m2` → **EXIT=1** (red). CPU AVX2 matmul vs OpenBLAS-1T.
- `./scripts/verify.sh m3` → **EXIT=1** (red). GPU parity PASS, CUDA bench RED (4.8x < 10x target).

---

## M2 — CPU Performance

### The wall
Gate requires: single-threaded matmul ≥ NumPy float32 @ 1024³.
On this machine NumPy = OpenBLAS 0.3.34 (DYNAMIC_ARCH, Haswell kernels,
pinned to 1 thread for fairness). Measured OpenBLAS-1T: **85–102 GFLOPS**
(~26–37 flop/cycle depending on thermal window; ~90% of AVX2 fp32 peak).

Our best hand-scheduled kernel (6x16 asm tile, k-unroll x4, packed A/B
panels, aligned hot loop): **58–75 GFLOPS** → ratio locked at
**0.70–0.76** across every configuration tried.

### Configurations exhausted (all median-of-20, same-window vs OpenBLAS)
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

---

## M3 — CUDA Backend

### Status
- **Gate A (gpu-parity): PASS** — all kernels (naive, tiled, fp16 WMMA) match CPU reference (`allclose(1e-3)` for fp32, `2e-2` for fp16 WMMA due to half input quantization).
- **Gate B (cuda-bench): RED** — tiled reaches ~2100 GFLOPS (50.7% of cuBLAS), but ratio vs naive is **4.8x** (target ≥ 10x).

### Why Gate B is RED
On Ampere (RTX 3050 laptop, 2MB L2 cache), the canonical naive kernel achieves **~385 GFLOPS** (~4.3% of 9.1TF peak) because L2 caches B column-panels across thread blocks automatically. To hit 10x naive (3850 GFLOPS), tiled would need >84% of cuBLAS.
- Naive: 386 GFLOPS
- Tiled (64x64 SMEM tile, 4x4 subtile): 2112 GFLOPS (4.7x naive, 50.7% cuBLAS)
- WMMA fp16 (Tensor Cores): 2873 GFLOPS (68.9% cuBLAS)
- cuBLAS (reference): 4169 GFLOPS

---

## Options for the human
1. **Recalibrate M2 gate**: `AVX2-1T ≥ 0.75 × NumPy-1T @1024³ AND ≥ 8× naive` (passes today: OMP hits 183 GFLOPS = 26x naive).
2. **Recalibrate M3 gate**: `tiled ≥ 4.5× naive AND ≥ 45% cuBLAS @1024³` (passes today: 4.8x naive, 50.7% cuBLAS).
3. Keep gates as-is; ship the full ladder tables in `bench/results.md` (valuable paper data showing realistic performance curves against industrial baselines).

## Reproduce
```
make -s -B lib cuda cublas
./scripts/verify.sh m0   # PASS (EXIT=0)
./scripts/verify.sh m1   # PASS (EXIT=0)
./scripts/verify.sh m2   # RED  (EXIT=1, 0.73x OpenBLAS-1T)
./scripts/verify.sh m3   # RED  (Gate A PASS, Gate B 4.8x < 10x naive)
```
