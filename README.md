# tinytorch

C tensor library with autograd, built from scratch. No dependencies beyond
a C compiler, OpenMP, NumPy, and (optionally) pybind11. Split out of the
`nnfromscratch` monorepo; inference-engine code lives in `tinyinference`.

What works: contiguous row-major tensors (`strides` metadata is informational only;
`reshape` returns a copy, not a view), elementwise / matmul / softmax / relu ops,
reverse-mode autograd with gradcheck, im2col conv2d + maxpool for CNNs,
AVX2+FMA blocked sgemm (single + OpenMP paths), pybind11 bindings.

## Build

```
make lib        # build/libtinytorch.so
make pybind     # Python extension module
make ci         # syntax-check all sources + rebuild lib
```

## Quickstart

```python
from tests.torch_py import Tensor, Node
x = Node.leaf([[1.0, 2.0], [3.0, 4.0]])
y = (x * x).sum()
y.backward()
```

## Gates

```
./scripts/verify.sh m0   # ops vs NumPy
./scripts/verify.sh m1   # gradcheck + MNIST MLP (>=97%, downloads ~11MB)
./scripts/verify.sh m2   # CPU sgemm vs NumPy @1024^3
./scripts/verify.sh m3   # CUDA tiled GEMM (needs NVIDIA GPU)
./scripts/verify.sh m4   # CIFAR-10 CNN (>=70%) + pybind smoke
```

## Numbers

M2 AVX2 sgemm beats NumPy float32 single-threaded at 1024^3; M3 tiled
CUDA kernel targets >=10x naive and reports % of cuBLAS sgemm. See
`bench/results.md` for the recorded ladder.

## Layout

`src/tensor.c`, `src/autograd.c`, `src/ops.c`, `src/ops_spatial.c`,
`src/gemm.c`, `src/bindings.cpp` — headers in `include/`, gates in
`tests/`, GEMM harnesses in `bench/`, sweep driver in
`tools/bench_gemm_sweep.c`.
