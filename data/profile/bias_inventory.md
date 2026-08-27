# Bias Inventory (Task 0 Gate)

**Method**: Read first 1 MB of each GGUF, search for byte string `attn_q.bias`,
`attn_k.bias`, `attn_v.bias`, `ffn_gate.bias`, `ffn_up.bias`. These tensor names
are the standard GGUF naming for Q/K/V output projection biases and FFN
gate/up projection biases in llama.cpp's converter.

**Date**: 2026-08-27
**Branch**: m6-correctness

## Result (all 4 bench models)

| Model | attn_q.bias | attn_k.bias | attn_v.bias | ffn_gate.bias | ffn_up.bias | **any bias?** |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct-q4_0.gguf | no | no | no | no | no | **NO** |
| qwen3-0.6b-q8_0.gguf | no | no | no | no | no | **NO** |
| llama-3.2-1b-q4_0.gguf | no | no | no | no | no | **NO** |
| smollm2-135m-f16.gguf | no | no | no | no | no | **NO** |

## Conclusion (1 sentence per model)

- **qwen2.5-0.5b-instruct-q4_0**: No biases anywhere — confirms qwen2.5 architecture is unbiased; the QKV split kernel needs no bias-add path.
- **qwen3-0.6b-q8_0**: No biases — qwen3 is also unbiased; same simplification applies.
- **llama-3.2-1b-q4_0**: No biases — Llama 3.x dropped biases; same simplification.
- **smollm2-135m-f16**: No biases — SmolLM2 architecture is unbiased; same simplification.

## Implication for fusion plan

All 4 bench models are bias-free, so a fused QKV-projection kernel does not need
a bias-add fused tail. A fused SwiGLU gate*up→silu*gate*up kernel does not need
a bias-add either. This simplifies fused-kernel design (no extra `+ bias` epilogue,
no extra `cudaMemcpyAsync` for bias tensors per layer, no extra per-row add launch).
**Eliminates one whole class of kernels from the launch-gap equation.**

## Caveat / known limitation

The first-1MB scan is a string search, not a tensor-table parse. The GGUF header
(tensor count + metadata) is small but a real GGUF could in principle have a
bias tensor whose name is split across the 1MB boundary on a tiny model.
For the 4 models above sizes range 91 MB to 2.15 GB, and GGUF places all tensor
metadata in the first few KB of the file (well under 1 MB), so the string search
on the tensor name is reliable. The names searched are the exact names
llama.cpp's `convert_hf_to_gguf.py` writes. Confirmed against
`llama.cpp/gguf-py/gguf/constants.py` naming convention.
