#!/usr/bin/env bash
# Downloads one small model per architecture (ungated HF repos only),
# then generates a full quant matrix from the smallest via llama-quantize.
# M7 Task 0: see docs/plans/2026-08-23-m7-all-quants-all-archs.md
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/testmodels && cd data/testmodels
Q="$PWD/../../oracle/llama.cpp/build/bin/llama-quantize"

fetch() { # url outfile  (resume + retries)
  [ -f "$2" ] && { echo "have $2"; return; }
  curl -L --fail --retry 5 --retry-delay 3 -C - -o "$2" "$1"
}

# --- per-architecture representatives ---
# NOTE substitutions vs original plan (original repos now gated/deleted on HF):
#  smollm2: ggml-org/SmolLM2-135M-Instruct-GGUF -> unsloth/SmolLM2-135M-Instruct-GGUF
fetch https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-F16.gguf smollm2-135m-f16.gguf
fetch https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf qwen3-0.6b-q8_0.gguf
#  tinyllama: bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF (401) -> andrijdavid mirror
fetch https://huggingface.co/andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-f16.gguf tinyllama-f16.gguf
# gemma2 mirror (ungated); if URL 404s, substitute any ungated gemma2-2b GGUF f16
#  bartowski repo live but no f16 file (and f16 ~5.2GB busts the 4GB VRAM /
#  2.5GB-per-model envelope) -> use Q6_K (closest-to-lossless that fits)
fetch https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q6_K.gguf gemma2-2b-q6_k.gguf || echo "WARN: gemma2 fetch failed - resolve manually"

# --- quant matrix from the smallest model ---
for q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q4_K_S Q5_K Q5_K_S Q6_K; do
  [ -f "smollm2-135m-instruct-$q.gguf" ] || "$Q" smollm2-135m-f16.gguf "smollm2-135m-instruct-$q.gguf" "$q" 2>/dev/null || echo "quant $q failed"
done
ls -la
