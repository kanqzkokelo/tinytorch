#!/usr/bin/env bash
# Single source of truth. Exit 0 = gate green, nonzero = red.
# Usage: ./scripts/verify.sh [m0|m1|m2|m3|m4|m61|all]
set -uo pipefail
cd "$(dirname "$0")/.."

GATE="${1:-all}"
FAIL=0

run() { # run <name> <cmd...>
  local name="$1"; shift
  echo "=== $name ==="
  if "$@"; then echo "PASS: $name"; else echo "FAIL: $name"; FAIL=1; fi
}

case "$GATE" in
  m61) run "m6-logits-parity" python3 tests/gate_m6_logit_parity.py ;;
  m0)  run "unit-vs-numpy"   python3 tests/test_ops.py ;;
  m1)  run "gradcheck"       python3 tests/test_grad.py
       run "mnist-mlp"       python3 tests/gate_mnist_mlp.py ;;
  m2)  run "cpu-bench"       python3 bench/bench_cpu.py --gate ;;
  m3)  run "gpu-parity"      python3 tests/test_gpu_parity.py
       run "cuda-bench"      python3 bench/bench_cuda.py --gate ;;
  m4)  run "cifar-cnn"       python3 tests/gate_cifar_cnn.py
       run "pybind-demo"     python3 examples/train_mnist.py --smoke ;;
  all) for g in m0 m1 m2 m3 m4 m61; do "$0" "$g"; done ;;
  *) echo "unknown gate: $GATE"; exit 2 ;;
esac

exit $FAIL
