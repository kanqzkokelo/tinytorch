#!/usr/bin/env bash
# scripts/ci_local.sh = 30-second sanity without GPU.
#
# Same checks as .github/workflows/ci.yml, runnable locally in seconds.
# Covers: gcc syntax-check of all C sources, libqwen object build,
# samplers / specdec-sim / chat-template unit tests, cpu_backend
# synthetic run (golden-vs-GGUF part auto-skips without model files).
#
# NOT covered (need NVIDIA GPU, run manually):
#   make cuda / cublas / run_llm_gpu / chat_llm_gpu / dump_logits
#   python3 tests/test_gpu_parity.py tests/gate_m6_logit_parity.py
#   python3 tests/gate_tokenizer.py        # needs oracle binaries
#   ./scripts/verify.sh m3                 # GPU parity + bench
#
# Usage: ./scripts/ci_local.sh          (exit 0 = green)
set -uo pipefail
cd "$(dirname "$0")/.."

T0=$(date +%s)
FAIL=0
LOGDIR=build/ci_logs
mkdir -p "$LOGDIR"

run() { # run <name> <cmd...>
  local name="$1"; shift
  echo "=== $name ==="
  if "$@" 2>&1 | tee "$LOGDIR/$name.log"; then
    echo "PASS: $name"
  else
    echo "FAIL: $name (log: $LOGDIR/$name.log)"
    FAIL=1
  fi
}

CC_BIN="${CC:-gcc}"

# 1. Syntax-only check, every C source, repo include paths.
echo "=== compile-syntax ==="
SYN_FAIL=0
for f in src/*.c; do
  if ! $CC_BIN -std=c11 -fsyntax-only -DTT_IN_LIB -Iinclude -Isrc "$f" \
      >"$LOGDIR/syntax-$(basename "$f").log" 2>&1; then
    echo "FAIL: syntax $f"; cat "$LOGDIR/syntax-$(basename "$f").log"
    SYN_FAIL=1
  fi
done
[ "$SYN_FAIL" -eq 0 ] && echo "PASS: compile-syntax ($(ls src/*.c | wc -l) files)" || FAIL=1

# 2. Build libqwen objects with gcc (no link of CUDA deps).
run build-gcc bash -c '
  mkdir -p build/obj && rc=0
  for f in src/*.c; do
    gcc -O1 -mavx2 -mfma -fopenmp -std=c11 -fPIC -DTT_IN_LIB -Iinclude -Isrc \
        -c "$f" -o "build/obj/$(basename "${f%.c}").o" || rc=1
  done
  exit $rc'

# 3. Unit gates (pure C + Python, no GPU, no model files).
run test-samplers      python3 tests/test_samplers.py
run test-specdec-sim   python3 tests/test_specdec_sim.py
run test-chat-template python3 tests/test_chat_template.py

# 5. kvcache unit tests (pure C, self-contained binary).
if [ ! -x build/test_kvcache ]; then
  $CC_BIN -std=c99 -O2 -Wall -Wextra -Isrc -o build/test_kvcache \
      src/kvcache.c tests/test_kvcache.c -lm
fi
run test-kvcache build/test_kvcache

# 4. cpu_backend: compile CLI per include/cpu_backend.h line, then test.
#    Golden-vs-GGUF correctness skips gracefully when no model committed;
#    synthetic benchmark still validates the binary end-to-end.
if [ ! -x build/dequant_ref ] || [ src/dequant_ref.c -nt build/dequant_ref ]; then
  $CC_BIN -O3 -std=c11 -Iinclude -DTTQ_MAIN -o build/dequant_ref src/dequant_ref.c src/loader_gguf.c -lm
fi
if [ ! -x build/cpu_backend ] || [ src/cpu_backend.c -nt build/cpu_backend ]; then
  gcc -O3 -mavx2 -mfma -fopenmp -std=c11 -Iinclude \
      -DCPU_BACKEND_MAIN -o build/cpu_backend src/cpu_backend.c -lm
fi
run test-cpu-backend python3 tests/test_cpu_backend.py
if [ ! -x build/bench_ipc ] || [ src/tinytorch_ipc.c -nt build/bench_ipc ] || [ tools/bench_ipc_throughput.c -nt build/bench_ipc ]; then
  $CC_BIN -O3 -std=c11 -Wall -Wextra -Iinclude src/tinytorch_ipc.c tools/bench_ipc_throughput.c -lpthread -lrt -o build/bench_ipc
fi
run test-ipc build/bench_ipc

SEC=$(( $(date +%s) - T0 ))
if [ "$FAIL" -eq 0 ]; then
  echo "ci_local: GREEN (${SEC}s)"
else
  echo "ci_local: RED (${SEC}s)"
fi
exit $FAIL
