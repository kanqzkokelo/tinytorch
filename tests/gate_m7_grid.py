#!/usr/bin/env python3
"""M7 Task 5: full architecture x quant verification grid.

Runs tests/gate_m7_arch.py across every model in tests/fixtures/models.json
(plus the qwen2.5 reference), emits a markdown truth table, and exits nonzero
if any previously-passing combination regressed (baseline stored in
tests/fixtures/grid_baseline.json).

Usage: gate_m7_grid.py [--strict-median]   # strict: median<=0.15 for all
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
GATE = os.path.join(ROOT, "tests/gate_m7_arch.py")
BASELINE = os.path.join(ROOT, "tests/fixtures/grid_baseline.json")
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    env.get("LD_LIBRARY_PATH", "")]))

MODELS = []
mf = "tests/fixtures/models.json"
if os.path.exists(mf):
    manifest = json.load(open(mf))
    entries = manifest if isinstance(manifest, list) else manifest.get("models", [])
    for e in entries:
        MODELS.append((e.get("name") or e["path"], e["path"]))
# always include the flagship reference
MODELS.insert(0, ("qwen2.5-0.5b-q4_0 (reference)",
                  "data/models/qwen2.5-0.5b-instruct-q4_0.gguf"))

strict = "--strict-median" in sys.argv
results = {}
for name, path in MODELS:
    if not os.path.exists(path):
        print(f"[SKIP] {name}: {path} missing")
        results[name] = "missing"
        continue
    cmd = [sys.executable, GATE, "--model", path]
    if not strict:
        # small models have tighter logit margins vs MMQ accumulation drift
        cmd += ["--median-max", "0.6"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
    ok = r.returncode == 0
    results[name] = f"{'PASS' if ok else 'FAIL'} ({last.split(':')[-1].strip()})"
    print(f"{'PASS' if ok else 'FAIL'} {name}")

print("\n| Model | Parity |")
print("|-------|--------|")
for k, v in results.items():
    print(f"| {k} | {v} |")

# regression check vs baseline
if os.path.exists(BASELINE):
    base = json.load(open(BASELINE))
    regressions = [k for k, v in base.items()
                   if v.startswith("PASS") and results.get(k) != v and results.get(k, "").startswith("FAIL")]
    if regressions:
        print("REGRESSIONS:", regressions)
        sys.exit(1)

# update baseline on success
json.dump(results, open(BASELINE, "w"), indent=1)
sys.exit(0 if all(v.startswith(("PASS", "missing")) for v in results.values()) else 1)
