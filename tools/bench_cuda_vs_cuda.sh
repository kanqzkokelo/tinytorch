#!/usr/bin/env bash
# tools/bench_cuda_vs_cuda.sh — honest CUDA-vs-CUDA scoreboard, v2.
#
# Re-measures the engine against llama.cpp CUDA (RTX 3050 4GB) after the
# M9.0 PLE-fused V2 (5bcd88e) and split-K flash (4985d00) landings.
#
# Protocol:
#   ours:   build/run_llm_gpu, TT_GREEDY=1, TT_RAW_PROMPT=1
#           warmup 16 tok; tg-128 timed x3, median tok/s from STATS decode_us
#   oracle: llama-completion -no-cnv -n 128 --temp 0 --top-k 1 --perf
#           x3, median of libllama eval-time rate
#   chat:   gemma only — engine chat_llm_gpu vs llama-completion -p "hi\n2+2?\n"
#
# 3 runs per measurement, median. Models that fail to load or exceed VRAM
# are skipped and noted in the report. Output: data/bench/results_cuda_vs_cuda_v2.jsonl
# (gitignored) + a markdown scoreboard.
#
# Usage: tools/bench_cuda_vs_cuda.sh           # full sweep
#        tools/bench_cuda_vs_cuda.sh <model>   # one model only
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS="data/bench/results_cuda_vs_cuda_v2.jsonl"
REPORT="data/bench/results_cuda_vs_cuda_v2.md"
mkdir -p data/bench
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STAMP="$(date -Is)"

# CUDA runtime path for oracle (libcudart.so.12 / libcublas.so.12).
export LD_LIBRARY_PATH="$HOME/mmcuda/lib:${LD_LIBRARY_PATH:-}"
# Engine flags.
export TT_GREEDY=1
export TT_RAW_PROMPT=1

ORACLE="oracle/llama.cpp/build-cuda/bin/llama-completion"
MAX_BYTES=1677721600   # 1.6 GiB hard cap for in-VRAM test
TG=128                 # decode length (matches column header "tg128")
RUNS=3
PROMPT_RAW='Explain quantum computing in one sentence.'

# All candidates the user listed; resolved to first-existing file.
declare -a CANDIDATES=(
  "data/models/qwen2.5-0.5b-instruct-q4_0.gguf"
  "data/testmodels/qwen3-0.6b-q8_0.gguf"
  "data/testmodels/llama-3.2-1b-q4_0.gguf"
  "data/testmodels/smollm2-135m-f16.gguf"
  "data/models/gemma-4-E2B-it-Q6_K.gguf"
  "data/models/gemma-4-E2B-it-Q4_0.gguf"
  "data/testmodels/tinyllama-f16.gguf"
)
# Resolve to actually-existing files (whichever the user listed in any case).
MODELS=()
for c in "${CANDIDATES[@]}"; do
  if [ -f "$c" ]; then MODELS+=("$c")
  else
    base="$(basename "$c")"
    hit="$(find data -name "$base" 2>/dev/null | head -1)"
    [ -n "$hit" ] && MODELS+=("$hit")
  fi
done
[ "${1:-}" ] && MODELS=("$1")   # CLI: restrict to one model

median3() { sort -g | sed -n '2p'; }

# Engine runner. Streams stdout; on OOM sleeps 60s up to 5 attempts.
run_engine() {
  local model="$1" prompt="$2" tokens="$3" attempt rc
  for attempt in 1 2 3 4 5 6; do
    local out
    out="$(TT_MODEL="$model" ./build/run_llm_gpu "$prompt" "$tokens" 2>/tmp/cvc_err.txt)"
    rc=$?
    if [ $rc -eq 0 ] && grep -q '^STATS ' <<<"$out"; then
      # Merge stdout + stderr so graph-status line is reachable by downstream parsers.
      { printf '%s\n' "$out"; cat /tmp/cvc_err.txt; }
      return 0
    fi
    if grep -qiE 'out of memory|cudaErrorMemoryAllocation|OOM' /tmp/cvc_err.txt 2>/dev/null; then
      echo "[cvc] OOM attempt $attempt: $(basename "$model")" >&2
      sleep 60
    else
      echo "[cvc] engine fail rc=$rc: $(tail -c 300 /tmp/cvc_err.txt)" >&2
      return $rc
    fi
  done
  return 1
}

# Extract tok/s from STATS line; tokens / (decode_us/1e6).
extract_engine_rate() {
  awk '
    /^STATS / {
      for (i=1;i<=NF;i++) {
        split($i,a,"=")
        if (a[1]=="tokens") t=a[2]
        if (a[1]=="decode_us") d=a[2]
      }
      if (t>0 && d>0) printf "%.3f\n", t/(d/1e6)
    }'
}

# Extract graph-status line: "graph captured" or "fell back to eager" or "Graph: ...".
extract_graph_status() {
  awk '
    /decode-step graph captured/ {g="graph_captured"; next}
    /fell back to eager/ {g="eager_fallback"; next}
    /graph capture failed/ {g="eager_fallback"; next}
    /cudaGraph replay ON/ {g="graph_captured"; next}
    /cudaGraph replay OFF/ {g="eager_fallback"; next}
    END{print g?g:"unknown"}'
}

# Oracle: tok/s from libllama "eval time" line; fallback to wall-clock.
run_oracle() {
  local model="$1" prompt="$2" tokens="$3"
  "$ORACLE" -m "$model" -no-cnv -p "$prompt" -n "$tokens" --temp 0 --top-k 1 --perf 2>&1
}
extract_oracle_rate() {
  awk '
    /eval time/ && !/prompt eval/ {
      if (match($0, /[0-9.]+ tokens per second/)) print substr($0,RSTART,RLENGTH)+0
    }' | tail -1
}
extract_oracle_pp_rate() {
  awk '/prompt eval time/ {if (match($0,/[0-9.]+ tokens per second/)) print substr($0,RSTART,RLENGTH)+0}' | tail -1
}

: > "$RESULTS"
echo "model | ours_tg128 | oracle_tg128 | ratio | graph | chat_excerpt" > "$REPORT"
echo "------|-------------|--------------|-------|-------|---------------" >> "$REPORT"

declare -a RATIOS=()   # for geomean
SKIPPED=()

for m in "${MODELS[@]}"; do
  name="$(basename "$m")"
  sz=$(stat -c%s "$m" 2>/dev/null || echo 0)
  echo ">>> $name ($sz bytes)" >&2

  # Size gate.
  if [ "$sz" -gt "$MAX_BYTES" ]; then
    echo "SKIP >1.6GB: $name" >&2
    SKIPPED+=("$name: file > 1.6 GB")
    echo "$name | SKIP | - | - | - | - | -" >> "$REPORT"
    continue
  fi

  # Sanity load: 4 tokens.
  if ! run_engine "$m" "$PROMPT_RAW" 4 >/dev/null 2>&1; then
    echo "SKIP load fail: $name" >&2
    SKIPPED+=("$name: engine failed to load / generate")
    echo "$name | SKIP | - | - | - | - | -" >> "$REPORT"
    continue
  fi

  # Warmup 16 tok.
  run_engine "$m" "$PROMPT_RAW" 16 >/dev/null 2>&1 || true

  # --- ours: tg-128 x3, median ---
  erates=()
  eout_last=""
  etok_actual=""
  for i in $(seq 1 $RUNS); do
    eout="$(run_engine "$m" "$PROMPT_RAW" $TG)" || { erates=(); break; }
    eout_last="$eout"
    r="$(extract_engine_rate <<<"$eout")"
    [ -n "$r" ] && erates+=("$r")
    [ -z "$etok_actual" ] && etok_actual="$(awk '/^STATS /{for(i=1;i<=NF;i++)if($i~/^tokens=/){split($i,a,"=");print a[2];exit}}' <<<"$eout")"
  done
  if [ ${#erates[@]} -ne $RUNS ]; then
    echo "SKIP engine tg fail: $name" >&2
    SKIPPED+=("$name: engine tg$TG not measurable")
    echo "$name | SKIP | - | - | - | - | -" >> "$REPORT"
    continue
  fi
  ours_tg="$(printf '%s\n' "${erates[@]}" | median3)"
  ours_tg_note=""
  # If the engine only emitted 1 token (current M9 regression), the "rate" is a
  # per-forward-pass time, not a sustained decode rate. Annotate.
  if [ -n "$etok_actual" ] && [ "$etok_actual" -lt 8 ] 2>/dev/null; then
    ours_tg_note="(tokens_actual=$etok_actual/128 — per-pass)"
  fi
  graph_active="$(extract_graph_status <<<"$eout_last")"

  # --- oracle: tg-128 x3, median ---
  orates=()
  for i in $(seq 1 $RUNS); do
    oout="$(run_oracle "$m" "$PROMPT_RAW" $TG)" || { orates=(); break; }
    r="$(extract_oracle_rate <<<"$oout")"
    [ -n "$r" ] && orates+=("$r")
  done
  if [ ${#orates[@]} -ne $RUNS ]; then
    echo "SKIP oracle tg fail: $name" >&2
    SKIPPED+=("$name: oracle tg$TG not measurable")
    echo "$name | SKIP | - | - | - | - | -" >> "$REPORT"
    continue
  fi
  oracle_tg="$(printf '%s\n' "${orates[@]}" | median3)"
  ratio="$(awk -v a="$ours_tg" -v b="$oracle_tg" 'BEGIN{printf "%.3f", a/b}')"

  # --- chat (gemma only) ---
  chat_excerpt=""
  if [[ "$name" == gemma-* ]]; then
    echo "[cvc] chat: engine for $name" >&2
    ce="$(printf 'hi\n2+2?\n/exit\n' | TT_MODEL="$m" ./build/chat_llm_gpu 2>&1 | tail -c 800)"
    echo "[cvc] chat: oracle for $name" >&2
    co="$(run_oracle "$m" 'hi
2+2?' 50 | tail -c 800)"
    # Trim to ~120 chars
    chat_excerpt="engine=\"$(printf '%s' "$ce" | tr -d '\n' | head -c 60 | sed 's/"/\\"/g')\" oracle=\"$(printf '%s' "$co" | tr -d '\n' | head -c 60 | sed 's/"/\\"/g')\""
  fi

  # --- write JSONL + table row ---
  safe_chat="$(printf '%s' "$chat_excerpt" | sed 's/"/\\"/g')"
  rec="$(printf '{"ts":"%s","git_head":"%s","model":"%s","name":"%s","size_bytes":%s,"engine_tg128":%s,"engine_tokens_actual":%s,"oracle_tg128":%s,"ratio":%s,"engine_graph_active":"%s","engine_token_count_from_stats":true,"chat_excerpt":"%s","oracle_binary":"%s","device":"cuda","protocol":{"ours":"run_llm_gpu TT_GREEDY=1 TT_RAW_PROMPT=1 warmup16 tg128x3 median STATS decode_us","oracle":"llama-completion -no-cnv -n 128 --temp 0 --top-k 1 --perf x3 median"}}' \
    "$STAMP" "$GIT_HEAD" "$m" "$name" "$sz" "$ours_tg" "${etok_actual:-null}" "$oracle_tg" "$ratio" "$graph_active" "$safe_chat" "$ORACLE")"
  printf '%s\n' "$rec" >> "$RESULTS"

  echo "$name | $ours_tg $ours_tg_note | $oracle_tg | $ratio | $graph_active | ${chat_excerpt:--}" >> "$REPORT"
  RATIOS+=("$ratio")
done

# --- geomean ---
if [ ${#RATIOS[@]} -gt 0 ]; then
  gm="$(awk -v r="${RATIOS[*]}" 'BEGIN{n=split(r,a," "); p=1; for(i=1;i<=n;i++) p*=a[i]+0; printf "%.3f", p^(1/n)}')"
  echo "" >> "$REPORT"
  echo "**Geomean ratio (engine/oracle) = $gm** across ${#RATIOS[@]} models." >> "$REPORT"
  if [ "$(awk -v g="$gm" 'BEGIN{print (g>=1.0)?"1":"0"}')" = "1" ]; then
    echo "=> engine >= llama.cpp CUDA on decode (claim holds)" >> "$REPORT"
  else
    echo "=> engine < llama.cpp CUDA on decode (gap = $(awk -v g="$gm" 'BEGIN{printf "%.1f%%", (1-g)*100}'); roadmap target)" >> "$REPORT"
  fi
else
  echo "" >> "$REPORT"
  echo "**No models measured successfully.**" >> "$REPORT"
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "" >> "$REPORT"
  echo "## Skipped" >> "$REPORT"
  for s in "${SKIPPED[@]}"; do echo "- $s" >> "$REPORT"; done
fi

# --- gemma chat probe (any gemma model that loads, regardless of size cap) ---
echo "" >> "$REPORT"
echo '## Gemma chat probe (`hi\n2+2?\n/exit`)' >> "$REPORT"
GEMMA_MODEL=""
for c in "${CANDIDATES[@]}"; do
  if [[ "$(basename "$c")" == gemma-* ]]; then
    if [ -f "$c" ]; then GEMMA_MODEL="$c"; break; fi
    hit="$(find data -name "$(basename "$c")" 2>/dev/null | head -1)"
    [ -n "$hit" ] && { GEMMA_MODEL="$hit"; break; }
  fi
done
if [ -z "$GEMMA_MODEL" ]; then
  echo "- no gemma model file present" >> "$REPORT"
else
  echo "- using $(basename "$GEMMA_MODEL")" >> "$REPORT"
  # If Q6_K is the chosen one (corrupt), fall back to Q4_0 even though it exceeds the bench size cap.
  if [[ "$(basename "$GEMMA_MODEL")" == gemma-*-Q6_K.gguf ]] && [ -f "data/models/gemma-4-E2B-it-Q4_0.gguf" ]; then
    echo "- Q6_K is corrupted on disk, falling back to Q4_0 for chat probe only" >> "$REPORT"
    GEMMA_MODEL="data/models/gemma-4-E2B-it-Q4_0.gguf"
  fi
  ce="$(printf 'hi\n2+2?\n/exit\n' | TT_MODEL="$GEMMA_MODEL" ./build/chat_llm_gpu 2>&1 | tail -c 2000)"
  # Filter out the per-layer PLE debug spam; keep only the reply + timing line.
  ce_filtered="$(printf '%s' "$ce" | grep -v '\[PLE\]' | tail -c 600)"
  ce_reply="$(printf '%s' "$ce_filtered" | sed -n '/User >/,/User >/{//!p}' | tail -c 400 | tr -d '\n' | sed 's/"/\\"/g' | head -c 150)"
  ce_short="$(printf '%s' "$ce_filtered" | tail -c 200 | tr -d '\n' | sed 's/"/\\"/g' | head -c 100)"
  co="$(oracle/llama.cpp/build-cuda/bin/llama-completion -m "$GEMMA_MODEL" -no-cnv -p 'hi
2+2?' -n 50 --temp 0 --top-k 1 --perf 2>&1 | tail -c 2500)"
  # The actual reply sits between the prompt echo and the perf line.
  co_reply="$(printf '%s' "$co" | sed -n '/2+2?/,/common_perf_print/{//!p}' | head -c 400 | tr -d '\n' | sed 's/"/\\"/g' | head -c 150)"
  co_runs="$(printf '%s' "$co" | awk '
    /eval time/ && !/prompt eval/ {
      n=split($0,a," "); for(i=1;i<=n;i++) if (a[i]=="runs" && i>1) {print a[i-1]; exit}
    }')"
  ce_tokens="$(printf '%s' "$ce" | grep -oP '\[\K[0-9]+(?= tokens )' | head -1)"
  co_tokens="$co_runs"
  echo "- engine reply text: \`${ce_reply:-$ce_short}\` (${ce_tokens:-?} tokens)" >> "$REPORT"
  echo "- oracle reply text: \`${co_reply:-N/A}\` (${co_tokens:-?} runs)" >> "$REPORT"
  if [ -n "$ce_tokens" ] && [ "$ce_tokens" -gt 1 ] 2>/dev/null; then
    echo "- engine chat: **multi-token** ($ce_tokens tokens)" >> "$REPORT"
  else
    echo "- engine chat: **1-token** (tokenizer fix has not landed)" >> "$REPORT"
  fi
fi

echo "" >&2
echo "--- wrote $(wc -l < "$RESULTS") rows to $RESULTS ---" >&2
echo "--- report at $REPORT ---" >&2
cat "$REPORT"
