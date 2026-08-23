#!/usr/bin/env python3
"""Gate Q2: teacher-forced logits parity vs oracle fixtures.
Pass requires, per prompt: top-1 match AND argmax-delta <= 0.35
AND median|dlogit| <= 0.15 over >= 85% of prompts."""
import json, subprocess, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FIX = os.path.join(ROOT, "tests/fixtures")
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    env.get("LD_LIBRARY_PATH", "")]))

def fail(i, d, why):
    print(f"[FAIL] {d['prompt'][:40]!r:44s} {why}")


data = json.load(open(f"{FIX}/parity_set.json"))
npass = 0
for i, d in enumerate(data):
    ids = ",".join(map(str, d["tokens"]))
    try:
        r = subprocess.run(["build/dump_logits", ids, f"/tmp/ours_lg_{i}.bin"],
                           capture_output=True, text=True, timeout=300, env=env)
    except (subprocess.TimeoutExpired, OSError) as exc:
        fail(i, d, f"engine did not run: {exc}")
        continue
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-5:]) or "(no stderr)"
        hint = ("build/dump_logits missing? build first"
                if not os.path.exists("build/dump_logits")
                else "GPU or driver issue?")
        fail(i, d, f"exit={r.returncode}\n  stderr tail:\n{tail}\n  hint: {hint}")
        continue
    am_lines = [l for l in r.stdout.splitlines() if l.startswith("ARGMAX")]
    if not am_lines:
        tail = "\n".join(r.stderr.splitlines()[-5:]) or "(no stderr)"
        hint = ("build/dump_logits missing? build first"
                if not os.path.exists("build/dump_logits")
                else "GPU or driver issue?")
        fail(i, d, f"no ARGMAX line\n  stderr tail:\n{tail}\n  hint: {hint}")
        continue
    parts = am_lines[0].split()
    ours_am, ours_v = int(parts[1]), float(parts[2])
    ref = np.fromfile(f"{FIX}/oracle_lg_{i}.bin", dtype="<f4")
    if not os.path.exists(f"/tmp/ours_lg_{i}.bin"):
        fail(i, d, "no logits dump written by engine")
        continue
    ours_full = np.fromfile(f"/tmp/ours_lg_{i}.bin", dtype="<f4")
    if len(ours_full) < len(ref):
        fail(i, d, f"vocab-size mismatch: ours={len(ours_full)} ref={len(ref)} "
                   f"(oracle vocab {len(ref)} vs engine vocab {len(ours_full)})")
        continue
    ours = ours_full[:len(ref)]
    med = float(np.median(np.abs(ours - ref)))
    am_d = abs(ours_v - d["oracle_val"])
    top1_ok = ours_am == d["oracle_argmax"]
    ok = top1_ok and am_d <= 0.35 and med <= 0.15
    npass += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {d['prompt'][:40]!r:44s} "
          f"top1={'Y' if top1_ok else 'N'} argmax_d={am_d:.3f} median={med:.4f}")
frac = npass / len(data)
print(f"\nGate Q2-lite: {npass}/{len(data)} (need >= 0.85)")
sys.exit(0 if frac >= 0.85 else 1)
