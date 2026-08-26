#!/usr/bin/env bash
# Honest "ours vs llama.cpp" scoreboard.
#
# Anti-fake rule: a model is benchmarked ONLY after tests/gate_m7_arch.py
# passes on it (quick form: 1 prompt). Models gated red are skipped, not fudged.
#
# Protocol mirrors bench/bench_llm.py methodology:
#   ours:   build/run_llm_gpu, TT_GREEDY=1, TT_RAW_PROMPT=1,
#           warmup 16 tok, tg-128 timed x3, median tok/s from STATS decode_us
#   oracle: llama-completion -n 128 --temp 0 --top-k 1 --perf --no-jinja
#           (libllama "eval time" decode-only line preferred)
#   pp512:  ~512-token filler prompt, timed once per engine
#           (ours: incl-prefill rate minus decode rate from [gen:] line;
#            oracle: --perf prompt_eval line)
#
# Results appended as JSONL to data/bench/results_scoreboard.jsonl.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS="data/bench/results_scoreboard.jsonl"
mkdir -p data/bench
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STAMP="$(date -Is)"
PROMPT_RAW="Explain quantum computing in one sentence."
ORACLE="oracle/llama.cpp/build/bin/llama-completion"
GATE="tests/gate_m7_arch.py"
MAX_CTX=1024

export LD_LIBRARY_PATH="$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export TT_GREEDY=1
export TT_RAW_PROMPT=1

MODELS=(
  data/testmodels/smollm2-135m-instruct-Q4_0.gguf
  data/testmodels/smollm2-135m-instruct-Q8_0.gguf
  data/models/qwen2.5-0.5b-instruct-q4_0.gguf
  data/testmodels/qwen3-0.6b-q8_0.gguf
  data/testmodels/llama-3.2-1b-q4_0.gguf
)

median3() { sort -g | sed -n '2p'; }

# run_ours <model> <prompt> <tokens>; echoes engine stdout; retries on OOM.
run_ours() {
  local model="$1" prompt="$2" tokens="$3" attempt out rc
  for attempt in 1 2 3 4 5 6; do
    out="$(TT_MODEL="$model" build/run_llm_gpu "$prompt" "$tokens" 2>/tmp/scoreboard_err.txt)"
    rc=$?
    if [ $rc -eq 0 ] && grep -q '^STATS ' <<<"$out"; then
      printf '%s\n' "$out"; return 0
    fi
    if grep -qiE 'out of memory|cudaErrorMemoryAllocation|OOM' /tmp/scoreboard_err.txt "$out" 2>/dev/null; then
      echo "[scoreboard] OOM (attempt $attempt/5+1), waiting 60s: $(basename "$model")" >&2
      sleep 60
    else
      echo "[scoreboard] ours failed rc=$rc: $(tail -c 200 /tmp/scoreboard_err.txt)" >&2
      return $rc
    fi
  done
  return 1
}

append_jsonl() {  # append_jsonl <json-string>
  printf '{%s, %s}\n' "$1" "\"date\": \"$STAMP\", \"git_head\": \"$GIT_HEAD\", \"parity_status\": \"gate-passed\"}" >> "$RESULTS"
}

echo "model | gate | ours_tg128 | oracle_tg128 | ratio | ours_pp512 | oracle_pp512"
for m in "${MODELS[@]}"; do
  name="$(basename "$m")"
  sz=$(stat -c%s "$m")
  if [ "$sz" -gt 1677721600 ]; then echo "SKIP >1.6GB: $name" >&2; continue; fi

  # --- anti-fake gate (quick form, 1 prompt) ---
  if ! python3 "$GATE" --model "$m" --prompt "The capital of France is" >/tmp/scoreboard_gate.txt 2>&1; then
    echo "[scoreboard] GATE FAIL, skipping $name"; tail -3 /tmp/scoreboard_gate.txt >&2
    continue
  fi
  echo "[scoreboard] gate PASS: $name" >&2

  # --- ours: warmup 16 tok ---
  run_ours "$m" "$PROMPT_RAW" 16 >/dev/null || { echo "SKIP warmup fail: $name" >&2; continue; }

  # --- ours: tg-128 x3, median ---
  rates=(); pps=""
  for i in 1 2 3; do
    out="$(run_ours "$m" "$PROMPT_RAW" 128)" || break
    stats=$(grep '^STATS ' <<<"$out" | head -1)
    tok=$(awk '{for(i=1;i<=NF;i++) if($i~/^tokens=/){split($i,a,"=");print a[2]}}' <<<"$stats")
    dus=$(awk '{for(i=1;i<=NF;i++) if($i~/^decode_us=/){split($i,a,"=");print a[2]}}' <<<"$stats")
    rates+=("$(awk -v t="$tok" -v d="$dus" 'BEGIN{print t/(d/1e6)}')")
    if [ -z "$pps" ]; then
      pf=$(awk '{for(i=1;i<=NF;i++) if($i~/^prefill=/){split($i,a,"=");print a[2]}}' <<<"$stats")
      dec=$(grep -oP 'decode \K[0-9.]+' <<<"$out" | head -1)
      inc=$(grep -oP 'incl prefill \K[0-9.]+' <<<"$out" | head -1)
      # prefill wall-time = total - decode window; pp rate = prompt_tokens / that
      [ -n "$pf" ] && [ "$pf" -gt 0 ] 2>/dev/null && [ -n "$dec" ] && [ -n "$inc" ] && \
        pps=$(awk -v p="$pf" -v t="$tok" -v d="$dec" -v i="$inc" \
          'BEGIN{pt=t/i-t/d; print (pt>0)?sprintf("%.1f",p/pt):""}')
    fi
  done
  [ ${#rates[@]} -eq 3 ] || { echo "SKIP ours tg128 fail: $name" >&2; continue; }
  ours_tg=$(printf '%s\n' "${rates[@]}" | median3)

  # --- oracle: tg-128 x3, median of libllama eval-time rate ---
  orates=(); opps=""
  for i in 1 2 3; do
    oout="$($ORACLE -m "$m" -no-cnv -p "$PROMPT_RAW" -n 128 --temp 0 --top-k 1 --perf 2>&1)"
    # perf lines: "... common_perf_print:        eval time = 25.37 ms / 7 runs (... 275.95 tokens per second)"
    ev=$(awk '/eval time/ && !/prompt eval/ {if (match($0, /[0-9.]+ tokens per second/)) print substr($0, RSTART)+0}' <<<"$oout" | tail -1)
    if [ -z "$ev" ]; then
      # fallback: wall-clock minus estimated prefill (prompt_eval ms line), stated in JSONL
      tot=$(grep -oP 'total time\s*=\s*\K[0-9.]+' <<<"$oout" | tail -1)
      pe=$(grep -oP 'prompt eval time\s*=\s*\K[0-9.]+' <<<"$oout" | tail -1)
      nt=$(grep -oP 'eval time\s*=\s*[0-9.]+ ms\s*/\s*\K[0-9]+' <<<"$oout" | tail -1)
      [ -n "$tot" ] && [ -n "$pe" ] && [ -n "$nt" ] && \
        ev=$(awk -v t="$tot" -v p="$pe" -v n="$nt" 'BEGIN{print n/((t-p)/1000)}')
      ometh="wall-clock minus prompt_eval"
    else ometh="libllama eval-time line"; fi
    [ -n "$ev" ] && orates+=("$ev")
    if [ -z "$opps" ]; then
      opps=$(awk '/prompt eval time/ {if (match($0, /[0-9.]+ tokens per second/)) print substr($0, RSTART)+0}' <<<"$oout" | tail -1)
    fi
  done
  [ ${#orates[@]} -eq 3 ] || { echo "SKIP oracle fail: $name" >&2; continue; }

  # --- pp512: ~512-token filler prompt, timed once per engine ---
  FILLER="The quick brown fox jumps over the lazy dog near the river bank while soft rain falls on the quiet village below the hills. "
  PP_PROMPT=""
  for r in $(seq 1 21); do PP_PROMPT="$PP_PROMPT$FILLER"; done   # ~504 tok
  opps512=""; opfs512=""
  out="$(run_ours "$m" "$PP_PROMPT" 1)" && {
    st=$(grep '^STATS ' <<<"$out" | head -1)
    pft=$(awk '{for(i=1;i<=NF;i++) if($i~/^prefill=/){split($i,a,"=");print a[2]}}' <<<"$st")
    tk=$(awk '{for(i=1;i<=NF;i++) if($i~/^tokens=/){split($i,a,"=");print a[2]}}' <<<"$st")
    d2=$(grep -oP 'decode \K[0-9.]+' <<<"$out" | head -1)
    i2=$(grep -oP 'incl prefill \K[0-9.]+' <<<"$out" | head -1)
    [ -n "$d2" ] && [ -n "$i2" ] &&       opfs512=$(awk -v p="$pft" -v t="$tk" -v d="$d2" -v i="$i2" \
        'BEGIN{pt=t/i-t/d; print (pt>0)?sprintf("%.1f",p/pt):""}')
  }
  oout="$($ORACLE -m "$m" -no-cnv -p "$PP_PROMPT" -n 1 --temp 0 --top-k 1 --perf 2>&1)"
  opps512=$(awk '/prompt eval time/ {if (match($0, /[0-9.]+ tokens per second/)) print substr($0, RSTART)+0}' <<<"$oout" | tail -1)
  oracle_tg=$(printf '%s\n' "${orates[@]}" | median3)
  ratio=$(awk -v a="$ours_tg" -v b="$oracle_tg" 'BEGIN{printf "%.3f", a/b}')
  rec="\"model\": \"$m\", \"name\": \"$name\", \"size_bytes\": $sz, \"ours_tg128_med_tok_s\": $ours_tg, \"oracle_tg128_med_tok_s\": $oracle_tg, \"ratio_ours_over_oracle\": $ratio, \"ours_pp512_tok_s\": ${opfs512:-null}, \"oracle_pp512_tok_s\": ${opps512:-null}, \"ours_pp_shortprompt_tok_s\": ${pps:-null}, \"oracle_timing_method\": \"$ometh\", \"protocol\": {\"ours\": \"run_llm_gpu TT_GREEDY=1 TT_RAW_PROMPT=1 warmup16 tg128x3 median STATS decode_us\", \"oracle\": \"llama-completion -no-cnv -n 128 --temp 0 --top-k 1 --perf x3 median\", \"ours_device\": \"cuda\", \"oracle_device\": \"cpu\"}"
  append_jsonl "$rec"
  echo "$name | PASS | $ours_tg | $oracle_tg | $ratio | ${opfs512:--} | ${opps512:--}"
done

echo "--- results appended to $RESULTS ---"
