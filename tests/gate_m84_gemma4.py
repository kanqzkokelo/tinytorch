#!/usr/bin/env python3
"""M8.4 gemma-4-E2B logit-parity gate (mirrors gate_m7_arch.py).

Prompts are explicit token-id lists (no tokenizer dependency), fed to both
binaries via --ids:
  - oracle:   build/oracle_logits MODEL --ids <csv> --dump /tmp/m84_oracle_N.bin
  - engine:   build/dump_logits --model MODEL <csv> /tmp/m84_ours_N.bin
Full logits (float32 LE, vocab 262144) compared per prompt.

Relaxed house-rule tolerance for gemma-4-E2B (KV-share work in flight):
pass per prompt = top-1 argmax ID match AND median|dlogit| <= 0.6.
Gate passes at >= 6/7 prompts.

Usage:
  python3 tests/gate_m84_gemma4.py [--model data/models/gemma-4-E2B-it-Q4_0.gguf]
      [--ids 2,2202] [--quick]
--ids replaces the default prompt list with a single prompt (debugging).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ORACLE_LOGITS = "build/oracle_logits"
DUMP = "build/dump_logits"

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    os.path.join(os.getcwd(), "oracle/llama.cpp/build/bin"),
    env.get("LD_LIBRARY_PATH", "")]))

DEFAULT_MODEL = "data/models/gemma-4-E2B-it-Q4_0.gguf"

# Deterministic mid-vocab id sequences; BOS=2. Content irrelevant.
DEFAULT_ID_PROMPTS = [
    [2, 2202],
    [2, 9302, 1110],
    [2, 5103, 7841],
    [2, 1110, 2202, 9302],
    [2, 15003, 402],
    [2, 6890, 12055, 304],
    [2, 3305, 9980],
]

VOCAB = 262144


def engine_logits(model, ids, tag):
    out = f"/tmp/m84_ours_{tag}.bin"
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
    out = f"/tmp/m84_oracle_{tag}.bin"
    r = subprocess.run([ORACLE_LOGITS, model,
                        "--ids", ",".join(map(str, ids)), "--dump", out],
                       capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise RuntimeError("oracle_logits exit=%d\n%s" %
                           (r.returncode, "\n".join(r.stderr.splitlines()[-3:])))
    return np.fromfile(out, dtype="<f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ids", default=None,
                    help="single prompt as comma-separated token ids (debug)")
    ap.add_argument("--quick", action="store_true",
                    help="run only the first 2 prompts")
    ap.add_argument("--median-max", type=float, default=0.6)
    ap.add_argument("--min-pass", type=int, default=6)
    args = ap.parse_args()

    prompts = ([[int(t) for t in args.ids.split(",")]] if args.ids
               else DEFAULT_ID_PROMPTS[:2] if args.quick else DEFAULT_ID_PROMPTS)

    print(f"gate m84 gemma4 [{os.path.basename(args.model)}] "
          f"({len(prompts)} prompts, need >= {args.min_pass})\n")
    header = (f"{'prompt':28s} {'n_tok':>5s} {'top1':>4s} "
              f"{'argmax_id':>9s} {'ref_id':>7s} {'am_d':>7s} {'median':>8s} {'result':>7s}")
    print(header)
    print("-" * len(header))

    npass = 0
    for i, ids in enumerate(prompts):
        label = ",".join(map(str, ids))[:26]
        try:
            ours, ours_am, ours_v = engine_logits(args.model, ids, str(i))
            ref = oracle_ref(args.model, ids, str(i))
        except (RuntimeError, OSError) as exc:
            print(f"{label:28s} {len(ids):5d} {'-':>4s} "
                  f"{'-':>9s} {'-':>7s} {'-':>7s} {'-':>8s} {'ERROR':>7s}")
            print(f"       {exc}")
            continue
        if len(ref) != VOCAB or len(ours) < len(ref):
            print(f"{label:28s} {len(ids):5d} vocab mismatch "
                  f"ours={len(ours)} ref={len(ref)} {'FAIL':>7s}")
            continue
        ref_am = int(np.argmax(ref))
        med = float(np.median(np.abs(ours[:len(ref)] - ref)))
        d = abs(ours_v - float(ref[ours_am]))
        top1 = ours_am == ref_am
        ok = top1 and med <= args.median_max
        npass += ok
        print(f"{label:28s} {len(ids):5d} {'Y' if top1 else 'N':>4s} "
              f"{ours_am:9d} {ref_am:7d} {d:7.3f} {med:8.4f} "
              f"{'PASS' if ok else 'FAIL':>7s}")

    print(f"\ngate m84-gemma4: {npass}/{len(prompts)} pass "
          f"(need >= {args.min_pass}, median <= {args.median_max})")
    sys.exit(0 if npass >= args.min_pass else 1)


if __name__ == "__main__":
    main()
