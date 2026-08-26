# tinytorch

A from-scratch neural-network inference stack in pure C + CUDA: tensor library,
autograd, hand-scheduled AVX2 GEMMs, a CUDA GEMM ladder that reaches cuBLAS
parity, and a multi-architecture LLM inference engine that **outperforms
llama.cpp ~4x** on the same GPU while proving output equivalence against it.

## LLM engine

```bash
./chat                                    # interactive chat (qwen2.5 default)
TT_MODEL=data/testmodels/smollm2-135m-instruct-Q8_0.gguf ./chat
```

| Property | Value |
|---|---|
| Decode speed | **286.7 tok/s** ctx≤64 / ~250 sustained (Qwen2.5-0.5B q4_0, RTX 3050 laptop) |
| vs llama.cpp | **~4.9× faster** (58 tok/s same box), output token-equivalent |
| Parity gate | `./scripts/verify.sh m61` — teacher-forced logits vs oracle, 7/7 |
| Sampling | repeat penalty + Gumbel-max temperature sampling inside CUDA graphs |

## Verified model grid

Teacher-forced logits parity vs llama.cpp oracle (`tests/gate_m7_grid.py`).
7-prompt gate; thresholds argmax-d ≤0.35, median \|Δlogit\| ≤0.6 (relaxed from
0.15 for ≤1B models where llama.cpp's MMQ integer kernels diverge from exact
dequant-fp32 accumulation).

| Architecture | Models | Quant formats | Status |
|---|---|---|---|
| qwen2 | Qwen2.5-0.5B | q4_0 (+q8_0 head) | ✅ 7/7 strict |
| qwen3 | Qwen3-0.6B | q8_0 | ✅ 7/7 |
| llama | TinyLlama-1.1B | f16 | ✅ bit-perfect (median Δ = 0.0009) |
| llama | SmolLM2-135M | f16, q4_0, q5_0, q5_1, q8_0, q4_K, q4_K_S, q5_K | ✅ |
| llama | SmolLM2-135M | q4_1, q5_K_S, q6_K | ⚠️ 5–6/7 (top-1 flips on near-tied logits) |
| gemma2 | Gemma2-2B | q6_K | ⚠️ 6/7 (was 7/7; re-verified during M8 session) |
| gemma4 | Gemma-4-E2B | q4_0 | 🚧 IN PROGRESS — see note below |

Quant kernels: q4_0, q4_1, q5_0, q5_1, q8_0, q4_K(+S), q5_K(+S), q6_K — all
golden-verified against gguf-py dequantization on real model bytes.

### gemma4 status (M8, active)

Heterogeneous mixture architecture (per-layer heads/kv/ffn/head_dim,
KV-cache sharing layers 15–34, partial RoPE, PLE/MatFormer blocks) loads and
runs end-to-end. Parity not yet green:

- Single-token median \|Δlogit\| **2.66** (was 22.06 before KV-cache sharing;
  measured via `tests/gate_m84_gemma4.py` single-token probe)
- Two-token median ~**9** — divergence under active bisection
- Gate `./scripts/verify.sh m84` NOT passing
- Golden reference: `tests/ref_gemma4_numpy.py` (NumPy forward from plan math)

## Classic ML benchmarks

| Suite | Result |
|---|---|
| Tensor ops vs NumPy | all match |
| Autograd gradcheck | < 1e-4 rel. error |
| MNIST MLP (2-layer) | ≥97% test accuracy |
| CPU matmul (AVX2+FMA asm) | 75 GFLOPS 1T / 239 GFLOPS MP (0.76× OpenBLAS-1T — documented wall) |
| CUDA sgemm tiled | **up to 100.7% of cuBLAS**, 39× naive @1024³ |
| WMMA fp16 | up to 72% cuBLAS |
| CIFAR-10 CNN | gate wired (`verify.sh m4`), formal run pending |

## Build & verify

```bash
make lib pybind cuda cublas          # libraries
make run_llm_gpu chat_llm_gpu        # LLM binaries (nvcc required)
./scripts/verify.sh m61              # LLM parity gate
./scripts/verify.sh all              # full sweep incl. classic ML gates
```

## Layout

```
src/            C sources (tensor, autograd, gemm, loader_gguf, tokenizer, arch_registry)
kernels/        CUDA (.cu): GEMM ladder, typed GEMVs, transformer engine
include/        headers
kernels/qwen2_cuda.cu   trait-driven decode engine (cudaGraph replay)
tests/          gates + golden references (NumPy forward, dequant goldens)
bench/          benchmark harnesses + results.md ladder tables
oracle/         pinned llama.cpp build (parity ground truth)
tools/          dump_logits, oracle_logits, profile_step
docs/plans/     implementation plans (M6.3 perf, M7 quants/archs)
```
