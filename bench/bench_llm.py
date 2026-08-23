#!/usr/bin/env python3
"""Anti-fake LLM benchmark: median-of-N decode throughput + mandatory parity check.
Usage: bench_llm.py [--runs 7] [--tokens 128] [--prompt TEXT]"""
import subprocess, os, sys, statistics, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=7)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--prompt", default="Explain quantum computing in one sentence.")
args = ap.parse_args()

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    env.get("LD_LIBRARY_PATH", "")]))

rates = []
for run in range(args.runs):
    r = subprocess.run(["build/run_llm_gpu", args.prompt, str(args.tokens)],
                       capture_output=True, text=True, timeout=600, env=env)
    stats = [l for l in r.stdout.splitlines() if l.startswith("STATS")]
    assert stats, f"run {run}: no STATS line\n{r.stdout[-500:]}"
    fields = dict(kv.split("=", 1) for kv in stats[0].split()[1:])
    tokens_actual = int(fields["tokens"])
    decode_us = float(fields["decode_us"])
    rates.append(tokens_actual / (decode_us / 1e6))
med = statistics.median(rates)
print(f"decode: median {med:.1f} tok/s over {args.runs} runs (min {min(rates):.1f}, max {max(rates):.1f})")

g = subprocess.run([sys.executable, "tests/gate_m6_logit_parity.py"],
                   capture_output=True, text=True, timeout=1200, env=env)
tail = g.stdout.strip().splitlines()[-1] if g.stdout.strip() else g.stderr[-200:]
print(tail)
sys.exit(0 if g.returncode == 0 else 1)
