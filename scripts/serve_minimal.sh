#!/usr/bin/env bash
# scripts/serve_minimal.sh -- M12 P1 env-setup wrapper for server_minimal.
#
# Sets LD_LIBRARY_PATH so the binary can find libtinytorch.so / CUDA runtime,
# picks a default model, and execs the server. Honors caller-provided env
# overrides (TT_MODEL, TT_LISTEN, TT_MAX_CTX, ...).
#
# Usage:
#   ./scripts/serve_minimal.sh                    # defaults
#   TT_MODEL=path/to/other.gguf ./scripts/serve_minimal.sh
#   TT_LISTEN=0.0.0.0:9000 ./scripts/serve_minimal.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Locate build artifacts. Server binary lives in build/ (matches other
# examples like chat_llm_gpu -> build/chat_llm_gpu).
BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BUILD_DIR}/server_minimal"
if [[ ! -x "${BIN}" ]]; then
    echo "[serve_minimal] ${BIN} not found. Build it first with:" >&2
    echo "    ${NVCC:-nvcc} -O3 -gencode arch=compute_86,code=sm_86 \\" >&2
    echo "      -I\${HOME}/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/include \\" >&2
    echo "      -Iinclude -Isrc -L\${HOME}/mmcuda/lib -o ${BIN} \\" >&2
    echo "      examples/server_minimal.c src/loader_gguf.c src/arch_registry.c \\" >&2
    echo "      src/dequant_ref.c src/tokenizer_bpe.c src/chat_template.c \\" >&2
    echo "      src/samplers.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \\" >&2
    echo "      kernels/qwen2_cuda.cu -lcudart -lpthread" >&2
    exit 1
fi

# Default model. Caller's TT_MODEL wins.
export TT_MODEL="${TT_MODEL:-data/models/qwen2.5-0.5b-instruct-q4_0.gguf}"
if [[ ! -f "${TT_MODEL}" ]]; then
    echo "[serve_minimal] model not found: ${TT_MODEL}" >&2
    exit 1
fi

# LD path: the cuda toolkit, the local build/ for libtinytorch.so, and the
# mmcuda install where the cuda runtime libs live.
export LD_LIBRARY_PATH="${BUILD_DIR}:${HOME}/mmcuda/lib:${LD_LIBRARY_PATH:-}"

# Reasonable defaults if the caller did not set them. The server enforces
# hard caps; these are just defaults.
export TT_LISTEN="${TT_LISTEN:-127.0.0.1:8080}"
export TT_MAX_CTX="${TT_MAX_CTX:-1024}"
export TT_GREEDY="${TT_GREEDY:-1}"     # server tests assume deterministic output

echo "[serve_minimal] model=${TT_MODEL} listen=${TT_LISTEN} ctx=${TT_MAX_CTX}" >&2
exec "${BIN}"
