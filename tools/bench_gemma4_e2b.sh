#!/usr/bin/env bash
# bench_gemma4_e2b.sh — honest decode-throughput benchmark for gemma-4-E2B.
#
# Anti-fake rules honored:
#   * Phase 0 parity guard: refuses to publish numbers unless
#     tests/gate_m84_gemma4.py passes (--force overrides, loudly warned).
#   * Methodology mirrors bench/bench_llm.py: short prompt (ctx <= 64),
#     warmup pass, timed >=128-token greedy decode, median across repeats.
#   * Our arm reports decode-only tok/s from the engine STATS line.
#   * Oracle arm uses llama-completion with identical token budget.
#
# Usage:
#   tools/bench_gemma4_e2b.sh [--model PATH] [--repeats N] [--skip-oracle]
#                             [--force] [--dry-run]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="data/models/gemma-4-E2B-it-Q4_0.gguf"
REPEATS=3
SKIP_ORACLE=0
FORCE=0
DRY_RUN=0
PROMPT="Explain quantum computing in one sentence."
GEN_TOKENS=128
WARMUP_TOKENS=16
GATE="tests/gate_m84_gemma4.py"
ORACLE_BIN="oracle/llama.cpp/build/bin/llama-completion"
ENGINE_BIN="build/run_llm_gpu"
RESULTS_JSONL="data/bench/results_gemma4.jsonl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --repeats)    REPEATS="$2"; shift 2 ;;
        --skip-oracle) SKIP_ORACLE=1; shift ;;
        --force)      FORCE=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

export LD_LIBRARY_PATH="$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$PWD/oracle/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

[[ -x "$ENGINE_BIN" ]] || { echo "missing engine: $ENGINE_BIN" >&2; exit 1; }
[[ -f "$MODEL" ]]      || { echo "missing model: $MODEL" >&2; exit 1; }

echo "== gemma-4-E2B decode benchmark =="
echo "model:   $MODEL"
echo "repeats: $REPEATS  gen_tokens: $GEN_TOKENS  warmup: $WARMUP_TOKENS"
[[ $FORCE -eq 1 ]]   && echo "mode:    FORCED (parity gate bypassed)"
[[ $DRY_RUN -eq 1 ]] && echo "mode:    DRY-RUN (no commands executed)"

# ---------------------------------------------------------------- phase 0
PARITY_STATUS="not-run"
GATE_CMD=(python3 "$GATE" --model "$MODEL")
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] ${GATE_CMD[*]}"
    PARITY_STATUS="dry-run-skipped"
elif [[ $FORCE -eq 0 ]]; then
    echo "-- phase 0: parity gate ($GATE) --"
    if GATE_OUT=$("${GATE_CMD[@]}" 2>&1); then
        PARITY_STATUS="pass: $(echo "$GATE_OUT" | tail -n1)"
        echo "gate PASS: $(echo "$GATE_OUT" | tail -n1)"
    else
        PARITY_STATUS="FAIL: $(echo "$GATE_OUT" | tail -n1)"
        {
            echo "******************************************************************"
            echo "* ANTI-FAKE GUARD: parity gate FAILED — refusing to benchmark.   *"
            echo "* gate said: $(echo "$GATE_OUT" | tail -n1)"
            echo "* re-run with --force to override (numbers will be recorded      *"
            echo "* as parity_status=forced-unverified in the JSONL).              *"
            echo "******************************************************************"
        } >&2
        exit 1
    fi
else
    PARITY_STATUS="forced-unverified"
    echo "!! WARNING: --force given — publishing numbers WITHOUT parity proof !!" >&2
fi

ENGINE_MED=""
ORACLE_MED="skipped"
SPEEDUP="n/a"

# ------------------------------------------------------------- our engine
# CLI (examples/run_llm_gpu.c): run_llm_gpu PROMPT N_TOKENS, model via
# TT_MODEL env var. Greedy by construction. Decode-only tok/s comes from
# the STATS tokens=/decode_us= line (mirrors bench/bench_llm.py).
engine_rates=()
for ((i = 0; i < REPEATS; i++)); do
    if [[ $i -eq 0 ]]; then n=$WARMUP_TOKENS; tag="warmup";
    else n=$GEN_TOKENS; tag="timed $((i))"; fi
    echo "-- engine $tag: $n tokens --"
    ENGINE_CMD=(env TT_MODEL="$MODEL" "$ENGINE_BIN" "$PROMPT" "$n")
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] ${ENGINE_CMD[*]}"
        TOK=$n; DUS=4000000; RATE="DRY"
    else
        OUT=$("${ENGINE_CMD[@]}")
        STATS=$(echo "$OUT" | grep '^STATS ') || { echo "no STATS line" >&2; exit 1; }
        TOK=$(echo "$STATS" | sed 's/.* tokens=\([0-9]*\).*/\1/')
        DUS=$(echo "$STATS" | sed 's/.* decode_us=\([0-9.]*\).*/\1/')
        RATE=$(python3 -c "print(f'{$TOK / ($DUS / 1e6):.2f}')")
    fi
    echo "   tokens=$TOK decode_us=$DUS -> $RATE tok/s"
    [[ $i -gt 0 && $DRY_RUN -eq 0 ]] && engine_rates+=("$RATE")
done
if [[ $DRY_RUN -eq 1 ]]; then
    ENGINE_MED="DRY"
else
    ENGINE_MED=$(python3 -c "
import statistics, sys
print(f'{statistics.median([float(x) for x in sys.argv[1:]]):.2f}')" \
        "${engine_rates[@]}")
fi
echo "engine median: $ENGINE_MED tok/s over $REPEATS timed runs"

# ---------------------------------------------------------------- oracle
if [[ $SKIP_ORACLE -eq 0 ]]; then
    oracle_rates=()
    for ((i = 0; i < REPEATS; i++)); do
        echo "-- oracle run $((i + 1))/$REPEATS: $GEN_TOKENS tokens --"
        ORACLE_CMD=("$ORACLE_BIN" -m "$MODEL" -p "$PROMPT" -n "$GEN_TOKENS" \
                    --temp 0 --top-k 1 --perf)
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[dry-run] ${ORACLE_CMD[*]}"
            ORATE="DRY"; src="dry-run"
        else
            T0=$(date +%s%N)
            OUT=$("${ORACLE_CMD[@]}" 2>&1)
            T1=$(date +%s%N)
            WALL_MS=$(( (T1 - T0) / 1000000 ))
            # prefer libllama decode-only timing (--perf), else wall clock
            EVAL_MS=$(echo "$OUT" \
                | grep -oP 'eval time.*?eval_time *= *\K[0-9.]+' | tail -n1 || true)
            if [[ -n "${EVAL_MS:-}" ]]; then
                ORATE=$(python3 -c "print(f'{$GEN_TOKENS / ($EVAL_MS / 1000):.2f}')")
                src="decode-only(perf)"
            else
                ORATE=$(python3 -c "print(f'{$GEN_TOKENS / ($WALL_MS / 1000):.2f}')")
                src="wall-clock(incl prefill+load)"
            fi
            oracle_rates+=("$ORATE")
        fi
        echo "   $src -> $ORATE tok/s"
    done
    if [[ $DRY_RUN -eq 1 ]]; then
        ORACLE_MED="DRY"
    else
        ORACLE_MED=$(python3 -c "
import statistics, sys
print(f'{statistics.median([float(x) for x in sys.argv[1:]]):.2f}')" \
            "${oracle_rates[@]}")
    fi
    echo "oracle median: $ORACLE_MED tok/s"
fi

# ---------------------------------------------------------------- report
NOTES="greedy, ctx<=64, decode-only(STATS), warmup=${WARMUP_TOKENS}t"
if [[ $SKIP_ORACLE -eq 0 && $DRY_RUN -eq 0 ]]; then
    SPEEDUP=$(python3 -c "print(f'{$ENGINE_MED / $ORACLE_MED:.2f}x')")
fi
printf '\n%-28s %10s  %s\n' "engine" "tok/s" "notes"
printf '%-28s %10s  %s\n' "---------" "-----" "-----"
printf '%-28s %10s  %s\n' "ours (run_llm_gpu)" "$ENGINE_MED" "$NOTES"
[[ $SKIP_ORACLE -eq 0 ]] \
    && printf '%-28s %10s  %s\n' "llama-completion" "$ORACLE_MED" \
        "greedy(--temp 0), same budget"
printf 'speedup ours/oracle: %s\n' "$SPEEDUP"

# ---------------------------------------------------------------- jsonl
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
DATE_ISO="$(date -Iseconds)"
JSON_LINE=$(python3 - "$ENGINE_MED" "$ORACLE_MED" "$SPEEDUP" <<PYEOF
import json, sys
engine, oracle, speedup = sys.argv[1], sys.argv[2], sys.argv[3]
def num(s):
    try: return float(s.rstrip("x"))
    except ValueError: return None
print(json.dumps({
    "date": "$DATE_ISO",
    "git_head": "$HEAD_SHA",
    "model": "$MODEL",
    "bench": "bench_gemma4_e2b",
    "repeats": $REPEATS,
    "gen_tokens": $GEN_TOKENS,
    "warmup_tokens": $WARMUP_TOKENS,
    "prompt": "$PROMPT",
    "engine_tok_s_median": num(engine),
    "oracle_tok_s_median": num(oracle),
    "speedup": num(speedup),
    "parity_status": """$PARITY_STATUS""",
}))
PYEOF
)
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] mkdir -p data/bench && echo '$JSON_LINE' >> $RESULTS_JSONL"
else
    mkdir -p data/bench
    echo "$JSON_LINE" >> "$RESULTS_JSONL"
    echo "appended: $RESULTS_JSONL"
fi
