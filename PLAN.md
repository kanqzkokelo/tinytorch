# tinytorch — neural network framework in pure C + CUDA

Autonomous build. The coding loop implements milestones; `scripts/verify.sh` decides truth.

## Loop rules (non-negotiable)
1. Trust ONLY exit codes from `./scripts/verify.sh` — never self-assessment.
2. Commit after every green gate. Message: `M<n>: <gate> PASS`.
3. >20 consecutive red commits on one gate → stop, write `BLOCKED.md`, exit.
4. Dependencies allowed: CMake, OpenMP, CUDA toolkit, pybind11 (M4 only). Nothing else.
5. Benchmarks: median of 20 runs, warmup 5, note GPU clocks in results.

## Environment
- GPU: RTX 3050 laptop, sm_86, 4GB VRAM, fp16 tensor cores OK
- nvcc flag: `-gencode arch=compute_86,code=sm_86`
- Reference oracle: NumPy (CPU tests), PyTorch (spot checks only)

## Milestones & gates

### M0 — Tensor library (C)
- `Tensor`: float32 data, shape, strides, refcount
- Ops: add, mul, scalar ops, matmul (naive), relu, softmax
- **Gate:** every op matches NumPy reference, allclose(atol=1e-5, rtol=1e-5)

### M1 — Autograd
- Reverse-mode AD over op DAG, `backward()` via topological sort
- **Gate A:** gradcheck vs central finite differences, rel err < 1e-4 on random tensors
- **Gate B:** 2-layer MLP trains MNIST ≥ 97% in ≤ 10 epochs (scripted assertion)

### M2 — CPU performance
- AVX2 + OpenMP matmul kernels
- **Gate:** single-threaded matmul ≥ NumPy float32 on 1024³; report speedup table

### M3 — CUDA backend
- Naive kernel → tiled shared-memory → fp16 WMMA
- **Gate A:** GPU output vs CPU allclose(atol=1e-3) for every kernel
- **Gate B:** tiled ≥ 10x naive; report % of cuBLAS sgemm
- Benchmark ladder chart goes in README

### M4 — Conv + bindings + polish
- conv2d, maxpool, avgpool; train CNN on CIFAR-10
- **Gate A:** CIFAR-10 ≥ 70% test acc
- **Gate B:** pybind11 demo: `python examples/train_mnist.py` works end-to-end
- README: benchmark tables, loss-curve GIF, zero-dependency badge

## Layout
```
src/        C sources
include/    headers
kernels/    .cu files
tests/      differential + gradcheck tests
bench/      benchmark scripts, results.md
examples/   training demos
scripts/    verify.sh (single entry point, exit code = truth)
```

## Human involvement (~5 min/day)
`git log --oneline && cat bench/results.md` — if green and advancing, do nothing.
