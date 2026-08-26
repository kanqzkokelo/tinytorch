#!/usr/bin/env bash
# bench_gemma4_decode.sh — capture gemma-4 E2B decode tok/s + graph-capture state.
#
# Purpose: when C5 graph-capture re-enable lands, this script captures the win.
#  - engine:    ./build/run_llm_gpu  (32 tokens, x3)
#  - oracle:    oracle/llama.cpp/build/bin/llama-completion  (x3, decode-only)
#  - chat UX:   ./build/chat_llm_gpu  (multi-turn smoke, capture reply)
#
# Outputs (gitignored):
#   data/bench/results_gemma4_c5.md      (markdown table)
#   data/bench/results_gemma4_c5.jsonl   (one JSON per invocation)
#
# Usage: tools/bench_gemma4_decode.sh [label]
#   label is appended to the JSONL row (e.g. "before", "after-c5").
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LABEL="${1:-run}"
MODEL="data/models/gemma-4-E2B-it-Q4_0.gguf"
ENGINE="./build/run_llm_gpu"
CHAT="./build/chat_llm_gpu"
ORACLE="oracle/llama.cpp/build/bin/llama-completion"
REPEATS=3
PROMPT="What is the capital of France?"
NTOK=32
RES_DIR="data/bench"
MD="$RES_DIR/results_gemma4_c5.md"
JSONL="$RES_DIR/results_gemma4_c5.jsonl"

export LD_LIBRARY_PATH="$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$PWD/oracle/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$RES_DIR"

[[ -f "$MODEL" ]]   || { echo "missing model: $MODEL" >&2; exit 1; }
[[ -x "$ENGINE" ]]  || { echo "missing engine: $ENGINE" >&2; exit 1; }
[[ -x "$ORACLE" ]]  || { echo "missing oracle: $ORACLE" >&2; exit 1; }
[[ -x "$CHAT" ]]    || echo "warn: chat binary missing, will skip chat UX" >&2

# ---------------------------------------------------------------- helpers
# Extract decode tok/s from a STATS line. Falls back to wall-clock if no STATS.
parse_engine_stats() {
    local out="$1"
    local stats tok dus rate
    stats=$(echo "$out" | grep '^STATS ' || true)
    if [[ -n "$stats" ]]; then
        tok=$(echo "$stats" | sed 's/.* tokens=\([0-9]*\).*/\1/')
        dus=$(echo "$stats" | sed 's/.* decode_us=\([0-9.]*\).*/\1/')
        rate=$(python3 -c "print(f'{${tok} / (${dus} / 1e6):.2f}')")
    else
        rate="0.00"
    fi
    echo "$rate"
}

# Detect graph-capture status from startup log.
# Matches either positive ("graph captured" / "cuda graph") or negative
# ("fell back to eager" / "eager mode" / "graph disabled") markers.
parse_graph_status() {
    local out="$1"
    local lower
    lower=$(echo "$out" | tr '[:upper:]' '[:lower:]')
    if grep -qE 'graph captured|cuda graph|graph_capture: ok' <<<"$lower"; then
        echo "graph captured"
    elif grep -qE 'fell back to eager|eager mode|graph disabled|no graph' <<<"$lower"; then
        echo "fell back to eager"
    else
        echo "unknown"
    fi
}

# Extract decode-only tok/s from llama-completion --perf output.
# libllama emits "eval time = X ms" inside "eval time" stanza. Fall back
# to wall-clock (incl prefill+load) if --perf line absent.
parse_oracle_rate() {
    local out="$1"
    local eval_ms rate src=""
    # Real format: "eval time =      64.94 ms /     1 runs   (   64.94 ms per token,    15.40 tokens per second)"
    # Extract first numeric after "eval time =". Also pull "N runs" so we can
    # report actual decoded-token count (the `-n 32` budget may be cut short by EOS).
    eval_ms=$(echo "$out" | grep -oP 'eval time\s*=\s*\K[0-9.]+' | head -n1 || true)
    runs=$(echo "$out" | grep -oP 'eval time\s*=\s*[0-9.]+\s*ms\s*/\s*\K[0-9]+' | head -n1 || true)
    runs=${runs:-1}
    if [[ -n "${eval_ms:-}" ]] && python3 -c "import sys; sys.exit(0 if float('$eval_ms')>0 else 1)" 2>/dev/null; then
        rate=$(python3 -c "print(f'{${runs} / (${eval_ms} / 1000):.2f}')")
        src="decode-only(perf, ${runs} tok)"
    else
        # caller passes wall time separately; here we just signal fail
        rate="0.00"; src="missing-perf"
    fi
    echo "$rate|$src"
}

# ---------------------------------------------------------------- engine
echo "== engine: $ENGINE =="
engine_rates=()
engine_graph="unknown"
engine_graph_lines=""
for ((i=0; i<REPEATS; i++)); do
    echo "-- engine run $((i+1))/$REPEATS --"
    out=$(env TT_MODEL="$MODEL" "$ENGINE" "$PROMPT" "$NTOK" 2>&1)
    rate=$(parse_engine_stats "$out")
    g=$(parse_graph_status "$out")
    engine_rates+=("$rate")
    [[ "$g" != "unknown" && "$engine_graph" == "unknown" ]] && engine_graph="$g"
    engine_graph_lines+="run$((i+1))=$g "
    echo "   tok/s=$rate  graph=$g"
done
engine_med=$(python3 -c "
import statistics, sys
vals=[float(x) for x in sys.argv[1:] if float(x)>0]
print(f'{statistics.median(vals):.2f}' if vals else '0.00')" "${engine_rates[@]}")
echo "engine median: $engine_med tok/s  (graph=$engine_graph)"

# ---------------------------------------------------------------- oracle
echo
echo "== oracle: $ORACLE =="
oracle_rates=()
oracle_srcs=()
for ((i=0; i<REPEATS; i++)); do
    echo "-- oracle run $((i+1))/$REPEATS --"
    t0=$(date +%s%N)
    out=$("$ORACLE" -m "$MODEL" -p "$PROMPT" -n "$NTOK" --temp 0 --top-k 1 -no-cnv --perf 2>&1)
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    parsed=$(parse_oracle_rate "$out")
    orate="${parsed%%|*}"
    src="${parsed##*|}"
    if [[ "$orate" == "0.00" || "$src" == "missing-perf" ]]; then
        orate=$(python3 -c "print(f'{${NTOK} / (${wall_ms} / 1000):.2f}')")
        src="wall-clock(incl prefill+load)"
    fi
    oracle_rates+=("$orate")
    oracle_srcs+=("$src")
    echo "   tok/s=$orate  src=$src"
done
oracle_med=$(python3 -c "
import statistics, sys
print(f'{statistics.median([float(x) for x in sys.argv[1:]]):.2f}')" "${oracle_rates[@]}")
echo "oracle median: $oracle_med tok/s"

# ---------------------------------------------------------------- chat UX
echo
echo "== chat UX: $CHAT =="
chat_reply="(chat binary unavailable)"
chat_reply_multi=""
chat_reply_assistant=""
if [[ -x "$CHAT" ]]; then
    # Pipe both prompts; give binary time to finish prefill+decode of both turns
    chat_out=$( (printf 'hi\n2+2?\n/exit\n'; sleep 90) | TT_MODEL="$MODEL" "$CHAT" 2>&1 )
    # head -20 per task spec (will usually be prefill noise because chat binary
    # logs per-layer PLE phases; assistant text typically appears later)
    chat_head=$(echo "$chat_out" | head -20)
    # All non-`[`-prefixed lines (drops engine logs); includes banners + assistant text
    chat_reply_multi=$(echo "$chat_out" \
        | grep -v '^\[' \
        | grep -v '^$' \
        | tr '\n' ' ' \
        | sed 's/  */ /g' \
        | cut -c1-400)
    # Just the assistant replies: take text-only view, drop PLE markers inline,
    # and extract content that follows "User >". Assistant tokens stream between
    # PLE log lines so a single-line awk isn't sufficient; instead we collapse
    # all non-`[`-prefixed runs (banner+prompt+assistant) to one string.
    chat_reply_assistant=$(echo "$chat_out" \
        | grep -v '^\[' \
        | grep -v '^$' \
        | grep -v 'tinytorch chat' \
        | grep -v 'Type .* to quit' \
        | grep -v '=====' \
        | tr '\n' ' ' \
        | sed 's/  */ /g' \
        | cut -c1-400)
    chat_reply=$(echo "$chat_head" | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-200)
    echo "   head 20 (spec): $chat_reply"
    echo "   text (no logs): $chat_reply_multi"
    echo "   assistant turns: $chat_reply_assistant"
fi

# ---------------------------------------------------------------- speedup
speedup=$(python3 -c "
e, o = float('$engine_med'), float('$oracle_med')
print(f'{e/o:.2f}x' if o>0 else 'n/a')")

# ---------------------------------------------------------------- jsonl
date_iso="$(date -Iseconds)"
head_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
{
    echo "engine_graph_lines=$engine_graph_lines"
    echo "oracle_srcs=${oracle_srcs[*]}"
} >/dev/null

python3 - "$JSONL" <<PYEOF
import json, sys
jsonl = sys.argv[1]
row = {
    "date": "$date_iso",
    "label": "$LABEL",
    "git_head": "$head_sha",
    "model": "$MODEL",
    "prompt": "$PROMPT",
    "gen_tokens": int("$NTOK"),
    "repeats": int("$REPEATS"),
    "engine_tok_s_median": float("$engine_med"),
    "oracle_tok_s_median": float("$oracle_med"),
    "speedup": "$speedup",
    "graph_capture_status": "$engine_graph",
    "chat_ux_reply": """$chat_reply""",
    "chat_ux_reply_text": """$chat_reply_multi""",
    "chat_ux_assistant_turns": """$chat_reply_assistant""",
}
with open(jsonl, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"appended: {jsonl}")
PYEOF

# ---------------------------------------------------------------- markdown
ts="$(date -Iseconds)"
{
    echo "# gemma-4 E2B decode benchmark"
    echo
    echo "- run label: \`${LABEL}\`"
    echo "- when: ${ts}"
    echo "- git: \`${head_sha}\`"
    echo "- model: \`${MODEL}\`"
    echo "- prompt: \`${PROMPT}\`"
    echo "- n_tokens: ${NTOK}  repeats: ${REPEATS}"
    echo
    echo "| arm | tok/s (median) | source | graph capture |"
    echo "|---|---:|---|---|"
    echo "| ours (run_llm_gpu) | ${engine_med} | decode-only (STATS) | ${engine_graph} |"
    echo "| llama-completion   | ${oracle_med} | $(IFS='/'; echo "${oracle_srcs[*]}") | n/a |"
    echo
    echo "**speedup ours / oracle: ${speedup}**"
    echo
    echo "## Chat UX reply"
    echo
    echo "### head -20 (raw)"
    echo
    echo '```'
    echo "${chat_reply}"
    echo '```'
    echo
    echo "### text-only (assistant content)"
    echo
    echo '```'
    echo "${chat_reply_multi}"
    echo '```'
    echo
    echo "### assistant turns (after each \`User >\` prompt)"
    echo
    echo '```'
    echo "${chat_reply_assistant}"
    echo '```'
    echo
    echo "_engine per-run graph detection: ${engine_graph_lines}_"
} > "$MD"
echo
echo "wrote: $MD"

# Final console summary
echo
echo "===== summary (${LABEL}) ====="
printf '  engine: %s tok/s  graph=%s\n' "$engine_med" "$engine_graph"
printf '  oracle: %s tok/s\n' "$oracle_med"
printf '  speedup: %s\n' "$speedup"
echo "============================="
