#!/usr/bin/env bash
# module_smoke.sh -- build-together smoke for the new M8 C modules.
#
# For each module:
#   1. strict object compile: -Wall -Wextra -Wpedantic -std=c99 -Iinclude
#   2. run the module's own test harness (sequential)
# Prints PASS/FAIL per module + total time. Must finish < 60s.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CC=${CC:-cc}
STRICT="-Wall -Wextra -Wpedantic -std=c99 -Iinclude -Isrc"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

START=$(date +%s.%N)
FAILS=0

run_module() {
    local name="$1"; shift
    local t0=$(date +%s.%N)
    if "$@" >"$TMP/$name.log" 2>&1; then
        local t1=$(date +%s.%N)
        printf 'PASS  %-14s (%.1fs)\n' "$name" "$(echo "$t1 - $t0" | bc)"
    else
        FAILS=$((FAILS + 1))
        printf 'FAIL  %-14s -- tail of log:\n' "$name"
        tail -15 "$TMP/$name.log" | sed 's/^/      /'
    fi
}

strict_compile() {
    # $1 = module name, $2 = source file
    $CC $STRICT -c "$2" -o "$TMP/$1.o" 2>"$TMP/$1.compile.log"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "compile failed:" >&2
        cat "$TMP/$1.compile.log" >&2
    elif [ -s "$TMP/$1.compile.log" ]; then
        echo "  [warn] $1 strict-compile warnings:" >&2
        head -5 "$TMP/$1.compile.log" | sed 's/^/         /' >&2
    fi
    return $rc
}

echo "== module_smoke: strict compile check =="
for m in samplers chat_template specdec kvcache moe_router cpu_backend; do
    if ! strict_compile "$m" "src/$m.c"; then
        echo "FAIL  $m (object compile)"
        FAILS=$((FAILS + 1))
    fi
done

echo
echo "== harnesses (sequential) =="

# samplers: python harness compiles its own SAMPLERS_MAIN CLI
run_module samplers python3 tests/test_samplers.py

# chat_template: ctypes harness builds its own .so
run_module chat_template python3 tests/test_chat_template.py

# specdec: pure-python simulator harness
run_module specdec python3 tests/test_specdec_sim.py

# kvcache: documented line -- src + test together, own main in test file
kv_bin="$TMP/test_kvcache"
if $CC -std=c99 -O2 -Wall -Wextra -Iinclude -Isrc -o "$kv_bin" \
       src/kvcache.c tests/test_kvcache.c -lm 2>"$TMP/kvcache_build.log"; then
    run_module kvcache "$kv_bin"
else
    echo "FAIL  kvcache (harness build)"; cat "$TMP/kvcache_build.log"; FAILS=$((FAILS+1))
fi

# moe_router: documented line
moe_bin="$TMP/test_moe_router"
if $CC -std=c99 -O2 -Wall -Wextra -Iinclude -Isrc -o "$moe_bin" \
       src/moe_router.c tests/test_moe_router.c -lm 2>"$TMP/moe_build.log"; then
    run_module moe_router "$moe_bin"
else
    echo "FAIL  moe_router (harness build)"; cat "$TMP/moe_build.log"; FAILS=$((FAILS+1))
fi

# cpu_backend: standalone driver per include/cpu_backend.h doc line;
# test_cpu_backend.py expects the binary at build/cpu_backend and the
# golden CLI at build/dequant_ref.
mkdir -p build
cb_bin="build/cpu_backend"
if ! make -s build/dequant_ref >"$TMP/dq_build.log" 2>&1; then
    echo "FAIL  cpu_backend (dequant_ref golden CLI build)"; FAILS=$((FAILS+1))
fi
if $CC -O3 -mavx2 -mfma -fopenmp -std=gnu11 -Iinclude -DCPU_BACKEND_MAIN \
       -o "$cb_bin" src/cpu_backend.c -lm 2>"$TMP/cb_build.log"; then
    run_module cpu_backend python3 tests/test_cpu_backend.py
else
    echo "FAIL  cpu_backend (harness build)"; cat "$TMP/cb_build.log"; FAILS=$((FAILS+1))
fi

END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)
printf '\nTotal: %d fail(s), %.1fs\n' "$FAILS" "$ELAPSED"
[ "$FAILS" -eq 0 ]
