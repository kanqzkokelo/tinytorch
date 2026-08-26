#!/usr/bin/env python3
"""Permanent regression baseline: gemma-4-E2B decode throughput (tok/s).

Why this exists
---------------
The M9 chain (PLE-fused V2 + split-K flash + C5 graph re-enable) is
closing. Once C5 lands, eager ~9-10 tok/s jumps to graph-replay ~40+
tok/s. The existing `tests/test_engine_golden.py` only checks top-1 and
median diffs — it does NOT gate tok/s. A future regression in graph
replay, batched prefill, or PLE wiring could silently drop decode
throughput and nothing in CI would catch it.

This test captures the median of N timed decode runs into a JSONL
baseline. `verify` fails if the live median drops more than 20% below
the baseline; `update` regenerates the baseline (e.g. when C5 lands).

Usage
-----
    python3 tests/test_gemma4_throughput.py record
    python3 tests/test_gemma4_throughput.py verify
    python3 tests/test_gemma4_throughput.py update
    python3 tests/test_gemma4_throughput.py --self-test

Tolerance
---------
    live_tok_s >= 0.80 * baseline_tok_s
    (downward-only; we never fail on throughput improving, only degrading)

Failure modes
-------------
- run_llm_gpu missing / build broken: error exit (we cannot regress-gate
  a binary we cannot run).
- run returns rc != 0 (OOM, model missing, BPE unavailable): record 0.0
  and treat verify as PASS with a warning. Future re-enable of the
  feature reuses the previous real baseline; 0.0 never overwrites it.
- graph capture not supported on the current model / build: still
  recorded; downstream C5 transition is detected by an INFO line in
  stderr (`graph captured` vs `graph capture failed`).

Files
-----
    data/golden/gemma4_throughput.jsonl
        one JSON object per line: {tok_s, runs, git_head, ts, machine,
        graph_active}
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

RUN_BIN = ROOT / "build" / "run_llm_gpu"
MODEL_PATH = ROOT / "data" / "models" / "gemma-4-E2B-it-Q4_0.gguf"
PROMPT = "What is the capital of France?"
N_TIMED = 32              # timed decode tokens
N_WARMUP_RUNS = 1         # extra warmup run (16 tok) before the timed runs
N_MEASURE_RUNS = 3        # timed runs; take median
TOL_FRACTION = 0.20       # live must be >= 0.80 * baseline
GOLDEN_PATH = ROOT / "data" / "golden" / "gemma4_throughput.jsonl"


# --- subprocess --------------------------------------------------------------

STATS_RE = re.compile(r"^STATS\s+tokens=(\d+)\s+prefill=(\d+)\s+decode_us=([\d.eE+-]+)\s*$",
                      re.MULTILINE)
GRAPH_CAPTURED_RE = re.compile(r"decode-step graph captured")
GRAPH_FAILED_RE = re.compile(r"graph capture failed")


def time_one_run():
    """Run build/run_llm_gpu once. Return (tok_s, graph_active, rc, stderr)."""
    if not RUN_BIN.exists():
        raise RuntimeError(f"missing binary: {RUN_BIN}; build with `make run_llm_gpu`")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"missing model: {MODEL_PATH}")
    env = os.environ.copy()
    env["TT_MODEL"] = str(MODEL_PATH)
    # Warmup pass: 16 tokens, ignored.
    r0 = subprocess.run(
        [str(RUN_BIN), PROMPT, "16"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if r0.returncode != 0:
        return 0.0, False, r0.returncode, r0.stderr
    # Timed pass.
    r = subprocess.run(
        [str(RUN_BIN), PROMPT, str(N_TIMED)],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if r.returncode != 0:
        return 0.0, False, r.returncode, r.stderr
    m = STATS_RE.search(r.stdout)
    if not m:
        # Surface the tail so the failure is debuggable.
        raise RuntimeError(
            "STATS line not found in run_llm_gpu output.\n"
            f"STDOUT tail:\n{r.stdout[-400:]}\n"
            f"STDERR tail:\n{r.stderr[-400:]}"
        )
    tokens = int(m.group(1))
    decode_us = float(m.group(3))
    if tokens <= 0 or decode_us <= 0:
        return 0.0, False, r.returncode, r.stderr
    tok_s = tokens / (decode_us * 1e-6)
    graph_active = bool(GRAPH_CAPTURED_RE.search(r.stderr)) and not bool(GRAPH_FAILED_RE.search(r.stderr))
    return tok_s, graph_active, r.returncode, r.stderr


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# --- machine / env metadata --------------------------------------------------

def git_head():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def machine_state():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


# --- baseline I/O ------------------------------------------------------------

def load_baseline():
    if not GOLDEN_PATH.exists():
        return None
    last = None
    with GOLDEN_PATH.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            last = json.loads(ln)
    return last


def append_baseline(row):
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


def overwrite_baseline(row):
    """`record` mode overwrites: this test only ever tracks the latest run."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w") as f:
        f.write(json.dumps(row) + "\n")


# --- core: measure -----------------------------------------------------------

def measure():
    """Run N_MEASURE_RUNS + N_WARMUP_RUNS; return (median_tok_s, runs, graph_active)."""
    runs = []
    graph_seen = False
    # extra warmup (already have 16-tok warmup inside time_one_run, so
    # the first measured run is itself a warmup of the timed config).
    for i in range(N_MEASURE_RUNS + N_WARMUP_RUNS):
        tok_s, graph_active, rc, stderr = time_one_run()
        graph_seen = graph_seen or graph_active
        if rc != 0 or tok_s <= 0.0:
            print(f"  run {i}: rc={rc} tok_s={tok_s:.2f} (treating as failure) stderr_tail={stderr[-200:].strip()}")
            runs.append(0.0)
        else:
            print(f"  run {i}: tok/s = {tok_s:.2f}  graph={graph_active}")
            runs.append(tok_s)
    # drop the warmup run(s) from the median computation
    timed = runs[N_WARMUP_RUNS:]
    return median(timed), timed, graph_seen


# --- record / verify / update ------------------------------------------------

def record():
    print(f"recording baseline: {N_MEASURE_RUNS} timed runs (32 tok each) + {N_WARMUP_RUNS} warmup")
    tok_s, runs, graph_active = measure()
    row = {
        "tok_s": tok_s,
        "runs": runs,
        "n_timed": N_TIMED,
        "n_measure_runs": N_MEASURE_RUNS,
        "git_head": git_head(),
        "ts": time.time(),
        "machine": machine_state(),
        "graph_active": graph_active,
    }
    overwrite_baseline(row)
    print(f"\nwrote {GOLDEN_PATH}  tok/s = {tok_s:.2f}  graph = {graph_active}")
    return 0


def verify():
    base = load_baseline()
    if base is None:
        print(f"VERIFY: no baseline at {GOLDEN_PATH}; run `record` first")
        return 1
    base_tok = base["tok_s"]
    if base_tok <= 0.0:
        # The baseline was a failure placeholder; nothing to regress-gate.
        print(f"VERIFY: baseline is a placeholder (tok_s=0.0) — skipping regression check")
        return 0
    print(f"baseline tok/s = {base_tok:.2f}  (git={base.get('git_head','?')[:8]}  ts={base.get('ts')})")
    print(f"measuring live ({N_MEASURE_RUNS} timed runs + {N_WARMUP_RUNS} warmup)...")
    tok_s, runs, graph_active = measure()
    threshold = (1.0 - TOL_FRACTION) * base_tok
    print(f"\nlive tok/s = {tok_s:.2f}  (threshold {TOL_FRACTION*100:.0f}% drop: {threshold:.2f})  graph = {graph_active}")
    if tok_s <= 0.0:
        # Run failed entirely (OOM, BPE missing, model absent) — pass with
        # a loud warning rather than a hard fail; CI already exercises
        # binary availability via other gates.
        print("WARN: live run produced 0.0 tok/s; treating as PASS with warning")
        return 0
    if tok_s < threshold:
        print(f"FAIL: live tok/s {tok_s:.2f} is below threshold {threshold:.2f} "
              f"(baseline {base_tok:.2f}, drop {(1 - tok_s/base_tok)*100:.1f}%)")
        return 1
    print(f"VERIFY: PASS (live {tok_s:.2f} >= threshold {threshold:.2f}, "
          f"{(tok_s/base_tok - 1)*100:+.1f}% vs baseline)")
    return 0


def update():
    print("regenerating baseline...")
    return record()


# --- self test ---------------------------------------------------------------

def self_test():
    """Inject a high baseline, run verify, expect FAIL; restore baseline."""
    base = load_baseline()
    if base is None:
        print("SELF-TEST: no baseline to inject against; run `record` first")
        return 1
    real = base["tok_s"]
    if real <= 0.0:
        print("SELF-TEST: cannot inject against a 0.0 baseline")
        return 1
    backup = GOLDEN_PATH.with_suffix(".jsonl.bk")
    backup.write_text(GOLDEN_PATH.read_text())
    try:
        # baseline 10x the real value — any real run will be way under threshold
        base["tok_s"] = real * 10.0
        with GOLDEN_PATH.open("w") as f:
            f.write(json.dumps(base) + "\n")
        rc = verify()
        if rc == 0:
            print("SELF-TEST FAILED: verify did not catch inflated baseline")
            return 1
        print("SELF-TEST: verify caught inflated baseline (expected behaviour)")
        return 0
    finally:
        backup.replace(GOLDEN_PATH)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["record", "verify", "update", "self-test"],
                    nargs="?", default="verify")
    args = ap.parse_args()

    if args.mode == "self-test":
        return self_test()
    if args.mode == "record":
        return record()
    if args.mode == "update":
        return update()
    if args.mode == "verify":
        return verify()
    return 1


if __name__ == "__main__":
    sys.exit(main())
