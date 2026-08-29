#!/usr/bin/env python3
"""Automated Benchmark Suite for nnfromscratch CUDA engine.

Evaluates prefill throughput (tok/s), decode throughput (tok/s),
latency per token (ms/tok), and memory footprint across context lengths.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_BIN = REPO_ROOT / "build" / "run_llm_gpu"

ENV_BASE = dict(os.environ)
ENV_BASE["PATH"] = f"{os.path.expanduser('~/mmcuda/bin')}:{ENV_BASE.get('PATH', '')}"
ENV_BASE["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    ENV_BASE.get("LD_LIBRARY_PATH", "")
]))


def run_benchmark(model_path: str, prompt_text: str, gen_tokens: int = 16,
                  max_ctx: int = 10240, q8_kv: int = 1) -> dict[str, float]:
    env = dict(ENV_BASE)
    env["TT_MODEL"] = model_path
    env["TT_MAX_CTX"] = str(max_ctx)
    env["TT_Q8_KV"] = str(q8_kv)

    cmd = [str(BUILD_BIN), prompt_text, str(gen_tokens)]
    t0 = time.monotonic()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    wall_sec = time.monotonic() - t0

    if res.returncode != 0:
        raise RuntimeError(f"run_llm_gpu failed (rc={res.returncode}):\n{res.stderr}")

    out = res.stdout + "\n" + res.stderr
    stats_m = re.search(r"STATS tokens=(\d+) prefill=(\d+) decode_us=(\d+) prefill_us=(\d+) prefill_tok_s=([\d\.]+)", out)
    if not stats_m:
        # Fallback to gen line
        gen_m = re.search(r"gen:\s+(\d+)\s+tokens\s+\|\s+decode\s+([\d\.]+)\s+tok/s\s+\|\s+prefill\s+([\d\.]+)\s+tok/s", out)
        if not gen_m:
            raise RuntimeError(f"Could not parse STATS line from output:\n{out[-500:]}")
        tokens = int(gen_m.group(1))
        dec_tok_s = float(gen_m.group(2))
        pref_tok_s = float(gen_m.group(3))
        return {
            "tokens": tokens,
            "prefill_tok_s": pref_tok_s,
            "decode_tok_s": dec_tok_s,
            "decode_ms_tok": 1000.0 / dec_tok_s if dec_tok_s > 0 else 0.0,
            "wall_sec": wall_sec,
        }

    tokens = int(stats_m.group(1))
    prefill_toks = int(stats_m.group(2))
    decode_us = float(stats_m.group(3))
    prefill_us = float(stats_m.group(4))
    prefill_tok_s = float(stats_m.group(5))

    dec_tok_s = (tokens / (decode_us * 1e-6)) if decode_us > 0 else 0.0
    return {
        "tokens": tokens,
        "prefill_tokens": prefill_toks,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": dec_tok_s,
        "decode_ms_tok": (decode_us / 1000.0 / tokens) if tokens > 0 else 0.0,
        "wall_sec": wall_sec,
    }


def make_prompt(n_tokens_approx: int) -> str:
    base_sentence = "The history of quantum computing dates back to the early 1980s when physicist Richard Feynman and computer scientist Paul Benioff suggested that quantum mechanics could be harnessed for computation. "
    reps = max(1, n_tokens_approx // 35)
    return base_sentence * reps


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Benchmark Suite")
    parser.add_argument("--model", default="data/models/qwen2.5-0.5b-instruct-q4_0.gguf")
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32, 128, 512, 1024, 1558, 3000, 7200])
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Benchmarking nnfromscratch CUDA Engine ===")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Contexts: {args.contexts}")
    print(f"Tokens to generate: {args.gen_tokens}")
    print()

    results = []
    for ctx in args.contexts:
        prompt = make_prompt(ctx) if ctx > 32 else "The capital of France is"
        print(f"Running context target ~{ctx} tokens...", end="", flush=True)
        try:
            r = run_benchmark(model_path, prompt, gen_tokens=args.gen_tokens)
            results.append((ctx, r))
            print(f" DONE: Prefill={r['prefill_tok_s']:.1f} tok/s | Decode={r['decode_tok_s']:.1f} tok/s ({r['decode_ms_tok']:.2f} ms/tok)")
        except Exception as e:
            print(f" FAILED: {e}")

    print("\n### Benchmark Summary Table\n")
    print("| Context (Tokens) | Prefill (tok/s) | Decode (tok/s) | Latency (ms/tok) |")
    print("|:---:|:---:|:---:|:---:|")
    for ctx, r in results:
        print(f"| {ctx} | {r['prefill_tok_s']:.1f} | {r['decode_tok_s']:.1f} | {r['decode_ms_tok']:.2f} ms |")
    print()


if __name__ == "__main__":
    main()
