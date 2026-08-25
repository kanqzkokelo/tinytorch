#!/usr/bin/env python3
"""M7 per-architecture logit-parity gate (generalizes gate_m6_logit_parity.py).

For each prompt:
  1. tokenize with the ORACLE tokenizer (llama-tokenize --ids, parse_special on)
  2. teacher-force those ids through our engine (build/dump_logits)
  3. compare against fresh oracle logits (build/oracle_logits --dump)

Pass per prompt: top-1 match AND |argmax-delta| <= 0.35 AND median|dlogit| <= 0.15.
Gate passes at >= 85% of prompts (same thresholds as M6).

Usage:
  python3 tests/gate_m7_arch.py --model data/testmodels/tinyllama-f16.gguf \
      [--prompt "text"] [--max-delta 0.35] [--quiet]
Multiple --prompt flags allowed; defaults to a small built-in ASCII set.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ORACLE_TOK = "oracle/llama.cpp/build/bin/llama-tokenize"
ORACLE_LOGITS = "build/oracle_logits"
DUMP = "build/dump_logits"

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    env.get("LD_LIBRARY_PATH", "")]))

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "My name is Maria. I like to eat",
    "The three primary colors are red,",
    "Once upon a time in a distant kingdom",
    "Photosynthesis is the process by which",
    "2 + 2 =",
]


def oracle_tokenize(model, prompt):
    r = subprocess.run([ORACLE_TOK, "-m", model, "--ids", "-p", prompt],
                       capture_output=True, text=True, timeout=120, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed: {r.stderr.strip()[:200]}")
    return [int(t) for t in r.stdout.strip().strip("[]").split(",") if t.strip()]


def engine_logits(model, ids, tag):
    out = f"/tmp/m7_ours_{tag}.bin"
    r = subprocess.run([DUMP, "--model", model,
                        ",".join(map(str, ids)), out],
                       capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise RuntimeError("engine dump_logits exit=%d\n%s" %
                           (r.returncode, "\n".join(r.stderr.splitlines()[-5:])))
    am = [l for l in r.stdout.splitlines() if l.startswith("ARGMAX")]
    if not am:
        raise RuntimeError("no ARGMAX line from dump_logits")
    parts = am[0].split()
    return np.fromfile(out, dtype="<f4"), int(parts[1]), float(parts[2])


def oracle_ref(model, ids, tag):
    out = f"/tmp/m7_oracle_{tag}.bin"
    id_csv = ",".join(map(str, ids))
    r = subprocess.run([ORACLE_LOGITS, model, "--ids", id_csv, "--dump", out],
                       capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise RuntimeError("oracle_logits exit=%d\n%s" %
                           (r.returncode, "\n".join(r.stderr.splitlines()[-3:])))
    return np.fromfile(out, dtype="<f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", action="append", default=[],
                    help="prompt to test (repeatable)")
    ap.add_argument("--max-delta", type=float, default=0.35)
    ap.add_argument("--median-max", type=float, default=0.15)
    ap.add_argument("--min-pass-frac", type=float, default=0.85)
    args = ap.parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS

    npass = 0
    for i, prompt in enumerate(prompts):
        try:
            ids = oracle_tokenize(args.model, prompt)
            ours, ours_am, ours_v = engine_logits(args.model, ids, str(i))
            ref = oracle_ref(args.model, ids, str(i))
        except (RuntimeError, OSError) as exc:
            print(f"[FAIL] {prompt[:40]!r:44s} {exc}")
            continue
        if len(ours) < len(ref):
            print(f"[FAIL] {prompt[:40]!r:44s} vocab mismatch ours={len(ours)} ref={len(ref)}")
            continue
        ours_c = ours[:len(ref)]
        med = float(np.median(np.abs(ours_c - ref)))
        d = abs(ours_v - float(ref[ours_am]))
        top1 = ours_am == int(np.argmax(ref))
        ok = top1 and d <= args.max_delta and med <= args.median_max
        npass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt[:40]!r:44s} "
              f"top1={'Y' if top1 else 'N'} argmax_d={d:.3f} median={med:.4f}")

    frac = npass / len(prompts) if prompts else 0
    print(f"\ngate m7-arch [{os.path.basename(args.model)}]: {npass}/{len(prompts)} "
          f"(need >= {args.min_pass_frac})")
    sys.exit(0 if frac >= args.min_pass_frac else 1)


if __name__ == "__main__":
    main()
