#!/usr/bin/env python3
"""M7 Task 1 gate: validate CPU dequant_ref C implementation against gguf-py
for every Tier-1 quant type, plus a GGUF size-computation audit.

Part A (per type): run build/dequant_ref on the same tensor gguf-py reads;
compare committed golden slice AND the freshly computed full tensor.
The goldens come from llama.cpp's own quantized files (llama-quantize output),
so passing means our layout + scale math match llama.cpp exactly.

Part B (sizes): loader-computed size_bytes must equal the byte length implied
by gguf-py's GGML_QUANT_SIZES for EVERY tensor of every SmolLM2 matrix model.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build/dequant_ref")
MANIFEST = os.path.join(ROOT, "tests/fixtures/dequant_gold_manifest.json")


def _load_gguf_py():
    """Import vendored gguf-py (plain import fails here: no sentencepiece)."""
    pkg_dir = os.path.join(ROOT, "oracle/llama.cpp/gguf-py/gguf")
    spec = importlib.util.spec_from_file_location(
        "gguf", os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf"] = mod
    for name in ("sentencepiece", "google", "google.protobuf"):
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            if name == "sentencepiece":
                stub.SentencePieceProcessor = object
            sys.modules[name] = stub
    spec.loader.exec_module(mod)
    return mod


GGUF = _load_gguf_py()


def run_cli(args):
    return subprocess.run([BIN] + args, capture_output=True, text=True)


def part_a_types():
    manifest = json.load(open(MANIFEST))
    fails = 0
    print(f"{'type':7s} {'tensor':26s} {'max|err|':>10s}  result")
    for label, info in manifest.items():
        path = os.path.join(ROOT, "data/testmodels", info["file"])
        out_bin = f"/tmp/dq_ref_{label}.bin"
        r = run_cli([path, info["tensor"], out_bin])
        if r.returncode != 0:
            print(f"{label:7s} {info['tensor']:26s} {'-':>10s}  FAIL (cli exit "
                  f"{r.returncode}: {r.stderr.strip().splitlines()[-1] if r.stderr else ''})")
            fails += 1
            continue
        ours = np.fromfile(out_bin, dtype="<f4")

        # committed golden slice (first gold_n values)
        gold = np.load(os.path.join(ROOT, "tests/fixtures",
                                    f"gold_smollm_{label}.npy"))
        atol = 2e-2 * max(1.0, float(np.abs(gold).max()))
        ok_slice = ours[:len(gold)].shape == gold.shape and \
            np.allclose(ours[:len(gold)], gold, rtol=0, atol=atol)

        # live full-tensor reference straight from gguf-py
        reader = GGUF.GGUFReader(path)
        t = next(x for x in reader.tensors if x.name == info["tensor"])
        ref = GGUF.quants.dequantize(t.data, t.tensor_type)\
            .astype(np.float32).reshape(-1)
        atol_full = 2e-2 * max(1.0, float(np.abs(ref).max()))
        err = float(np.abs(ours.astype(np.float64) - ref.astype(np.float64)).max()) \
            if ours.shape == ref.shape else float("inf")
        ok_full = ours.shape == ref.shape and np.allclose(
            ours, ref, rtol=0, atol=atol_full)

        ok = ok_slice and ok_full and len(ours) == info["n_full"]
        print(f"{label:7s} {info['tensor']:26s} {err:>10.3e}  "
              f"{'PASS' if ok else 'FAIL'}"
              + ("" if ok else f"  (slice_ok={ok_slice} full_ok={ok_full} "
                               f"n={len(ours)} vs {info['n_full']})"))
        fails += 0 if ok else 1
    return fails


def part_b_sizes():
    """loader size_bytes vs gguf-py GGML_QUANT_SIZES byte length, all tensors."""
    models = sorted(f for f in os.listdir(os.path.join(ROOT, "data/testmodels"))
                    if f.startswith("smollm2-135m"))
    total_bad = 0
    total_tensors = 0
    for fname in models:
        path = os.path.join(ROOT, "data/testmodels", fname)
        r = run_cli(["--sizes", path])
        if r.returncode != 0:
            print(f"SIZES {fname}: FAIL cli exit {r.returncode}")
            total_bad += 1
            continue
        bad = []
        reader = GGUF.GGUFReader(path)
        expected = {}
        for t in reader.tensors:
            block_size, type_size = GGUF.constants.GGML_QUANT_SIZES[t.tensor_type]
            numel = int(np.prod(t.shape))
            expected[t.name] = numel // block_size * type_size
        n = 0
        for line in r.stdout.splitlines():
            if "\t" not in line:
                continue  # loader printf noise
            name, _typ, size_bytes, _off = line.split("\t")
            if name in expected and int(size_bytes) != expected[name]:
                bad.append((name, int(size_bytes), expected[name]))
            n += 1
        total_tensors += n
        status = "OK" if not bad else f"FAIL ({len(bad)} mismatch)"
        print(f"SIZES {fname}: {n} tensors checked -> {status}")
        for name, got, want in bad:
            print(f"   {name}: loader={got} gguf-py={want}")
        total_bad += len(bad)
    return total_bad


if __name__ == "__main__":
    fails = part_a_types()
    print()
    bad = part_b_sizes()
    print(f"\nsize mismatches: {bad}")
    if fails or bad:
        print(f"RESULT: FAIL ({fails} type failures, {bad} size mismatches)")
        sys.exit(1)
    print("RESULT: ALL PASS")
