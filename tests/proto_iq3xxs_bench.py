#!/usr/bin/env python3
"""IQ3_XXS prototype bench driver.

Build:  $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 -Iinclude \
        tests/proto_iq3xxs.cu -o /tmp/proto_iq3xxs
Run:    python tests/proto_iq3xxs_bench.py
"""
import subprocess, sys, os, time, json

BIN = "/tmp/proto_iq3xxs"
SHAPES = [
    ("o+mlp",   11008,  1536),
    ("lm-head", 262144, 1536),
    ("smoke",    4096,  1536),
]

def main():
    if not os.path.exists(BIN):
        print(f"missing {BIN} — build first", file=sys.stderr); return 1
    rows = []
    for name, M, K in SHAPES:
        t0 = time.time()
        out = subprocess.run([BIN, str(M), str(K)], capture_output=True, text=True)
        dt = time.time() - t0
        if out.returncode != 0:
            print(f"FAIL {name}: {out.stderr}", file=sys.stderr); return out.returncode
        # parse last two bench lines
        plain, shm = None, None
        for line in out.stderr.splitlines():
            if "[bench] plain:" in line: plain = line
            if "[bench] shm:"   in line: shm   = line
        print(f"=== {name}  M={M} K={K}  ({dt:.1f}s wall) ===")
        print(plain); print(shm); print()
        rows.append({"name": name, "M": M, "K": K,
                     "plain": plain.split("W:")[1].split("GB/s")[0].strip() if plain else None,
                     "shm":   shm.split("W:")[1].split("GB/s")[0].strip()   if shm   else None})
    print("--- summary ---")
    print(f"{'shape':<10} {'M':>7} {'K':>5}  plain_GB/s  shm_GB/s")
    for r in rows:
        print(f"{r['name']:<10} {r['M']:>7} {r['K']:>5}  {r['plain']:>10}  {r['shm']:>8}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
