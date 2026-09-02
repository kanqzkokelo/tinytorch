#!/usr/bin/env python3
"""Anti-fake LLM benchmark: median-of-N decode throughput + mandatory parity check.
Usage: bench_llm.py [--runs 7] [--tokens 128] [--prompt TEXT] [--ctx N]

--ctx N pads the prompt with a repeated neutral sentence to reach ~N prompt
tokens; the measured prefill size from STATS is the ground truth and is
reported alongside the rate."""
import subprocess, os, sys, statistics, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=7)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--prompt", default="Explain quantum computing in one sentence.")
ap.add_argument("--ctx", type=int, default=0,
                help="pad prompt to ~N tokens with a repeated neutral sentence")
ap.add_argument("--max-ctx", type=int, default=0,
                help="set TT_MAX_CTX (KV cache capacity) for ctx > 1024 prompts")
ap.add_argument("--raw", action="store_true",
                help="set TT_RAW_PROMPT=1 (skip chat template)")
args = ap.parse_args()

FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
          "while soft rain falls on the quiet village below the hills. ")  # ~24 tok
prompt = args.prompt
if args.ctx > 0:
    # chat template + base prompt eat ~32 tokens; filler sentence ~24 tokens
    reps = max(0, round((args.ctx - 32) / 24))
    if reps:
        prompt = FILLER * reps + "Hi."

env = dict(os.environ)
if args.max_ctx > 0:
    env["TT_MAX_CTX"] = str(args.max_ctx)
if args.raw:
    env["TT_RAW_PROMPT"] = "1"

env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    env.get("LD_LIBRARY_PATH", "")]))

rates = []
prefills = []
for run in range(args.runs):
    r = subprocess.run(["build/run_llm_gpu", prompt, str(args.tokens)],
                       capture_output=True, text=True, timeout=600, env=env)
    if r.returncode != 0:
        print(f"run {run}: engine exit {r.returncode}\n{r.stderr[-300:]}")
        sys.exit(1)
    stats = [l for l in r.stdout.splitlines() if l.startswith("STATS")]
    if not stats:
        print(f"run {run}: no STATS line\n{r.stdout[-500:]}")
        sys.exit(1)
    fields = dict(kv.split("=", 1) for kv in stats[0].split()[1:])
    tokens_actual = int(fields["tokens"])
    decode_us = float(fields["decode_us"])
    prefills.append(int(fields.get("prefill", -1)))
    rates.append(tokens_actual / (decode_us / 1e6))
med = statistics.median(rates)
pf = f" prefill={statistics.median(prefills):.0f}" if args.ctx > 0 else ""
print(f"decode: median {med:.1f} tok/s over {args.runs} runs (min {min(rates):.1f}, max {max(rates):.1f}){pf}")

g = subprocess.run([sys.executable, "tests/gate_m6_logit_parity.py"],
                   capture_output=True, text=True, timeout=1200, env=env)
tail = g.stdout.strip().splitlines()[-1] if g.stdout.strip() else g.stderr[-200:]
print(tail)
sys.exit(0 if g.returncode == 0 else 1)
