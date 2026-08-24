#!/usr/bin/env python3
"""Generate golden dequant fixtures for every Tier-1 quant type.

For each type T: read a tensor that is ACTUALLY stored in type T from
data/testmodels/smollm2-135m-instruct-<T>.gguf (K-quant files keep
token_embd.weight at Q8_0, so we pick the largest tensor of the target type),
dequantize it with gguf-py (llama.cpp's own reference), and save the first
GOLD_N float32 values to tests/fixtures/gold_smollm_<T>.npy plus a manifest.

NOTE on committed size: a full token_embd slice would be >100 MB per file, so
we commit only GOLD_N values (256 K-quant super-blocks worth) — enough to pin
layout, nibble order, and scale math. tests/test_dequant_golden.py additionally
re-dequantizes the FULL tensor live via gguf-py and compares all elements.
"""
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gguf_py():
    """Import vendored gguf-py. Plain `import gguf` fails on this box, so load
    its __init__.py explicitly under the 'gguf' name (relative imports work)."""
    import importlib.util
    pkg_dir = os.path.join(ROOT, "oracle/llama.cpp/gguf-py/gguf")
    spec = importlib.util.spec_from_file_location(
        "gguf", os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf"] = mod
    # gguf.vocab imports sentencepiece/protobuf, which are not installed here;
    # we only need GGUFReader/quants, so stub the optional deps.
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


_gguf = _load_gguf_py()
GGUFReader, GGMLQuantizationType, quants = _gguf.GGUFReader, _gguf.GGMLQuantizationType, _gguf.quants

GOLD_N = 1 << 16  # 65536 floats = 256 KB each committed

# type label -> (gguf file, GGML type code)
MODELS = {
    "F16":    ("smollm2-135m-f16.gguf",              GGMLQuantizationType.F16),
    "Q4_0":   ("smollm2-135m-instruct-Q4_0.gguf",    GGMLQuantizationType.Q4_0),
    "Q4_1":   ("smollm2-135m-instruct-Q4_1.gguf",    GGMLQuantizationType.Q4_1),
    "Q5_0":   ("smollm2-135m-instruct-Q5_0.gguf",    GGMLQuantizationType.Q5_0),
    "Q5_1":   ("smollm2-135m-instruct-Q5_1.gguf",    GGMLQuantizationType.Q5_1),
    "Q8_0":   ("smollm2-135m-instruct-Q8_0.gguf",    GGMLQuantizationType.Q8_0),
    # Q4_K_S / Q5_K_S files store tensors with the same on-disk layout codes
    # (Q4_K=12 / Q5_K=13); we still test their files explicitly.
    "Q4_K":   ("smollm2-135m-instruct-Q4_K.gguf",    GGMLQuantizationType.Q4_K),
    "Q4_K_S": ("smollm2-135m-instruct-Q4_K_S.gguf",  GGMLQuantizationType.Q4_K),
    "Q5_K":   ("smollm2-135m-instruct-Q5_K.gguf",    GGMLQuantizationType.Q5_K),
    "Q5_K_S": ("smollm2-135m-instruct-Q5_K_S.gguf",  GGMLQuantizationType.Q5_K),
    "Q6_K":   ("smollm2-135m-instruct-Q6_K.gguf",    GGMLQuantizationType.Q6_K),
}


def pick_tensor(reader, qtype):
    """Largest tensor stored exactly in `qtype` (>= 4096 elems)."""
    best = None
    for t in reader.tensors:
        if t.tensor_type != qtype:
            continue
        n = int(np.prod(t.shape))
        if n < 4096:
            continue
        if best is None or n > best_n:
            best, best_n = t, n
    return best


def main():
    manifest = {}
    for label, (fname, qtype) in MODELS.items():
        path = os.path.join(ROOT, "data/testmodels", fname)
        r = GGUFReader(path)
        t = pick_tensor(r, qtype)
        assert t is not None, f"{label}: no {qtype.name} tensor in {fname}"
        full = quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
        out = os.path.join(ROOT, f"tests/fixtures/gold_smollm_{label}.npy")
        np.save(out, full[:GOLD_N])
        manifest[label] = {
            "file": fname,
            "tensor": t.name,
            "qtype": int(t.tensor_type),
            "n_full": int(full.size),
            "gold_n": min(GOLD_N, int(full.size)),
        }
        print(f"{label:6s} {t.name:28s} n={full.size:>9d} gold={out}")
    mpath = os.path.join(ROOT, "tests/fixtures/dequant_gold_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
