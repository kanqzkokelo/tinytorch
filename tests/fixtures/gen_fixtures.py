#!/usr/bin/env python3
"""Generate committed parity fixtures. Requires oracle/llama.cpp build."""
import subprocess, os, json, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
BIN = os.path.join(ROOT, "oracle/llama.cpp/build/bin")
MODEL = os.path.join(ROOT, "data/models/qwen2.5-0.5b-instruct-q4_0.gguf")
env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(BIN))
PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "My name is Maria. I like to eat",
    "The three primary colors are red,",
    "Once upon a time in a distant kingdom",
    "The largest planet in the solar system is",
    "Photosynthesis is the process by which",
]
out = []
for i, p in enumerate(PROMPTS):
    r = subprocess.run([f"{BIN}/llama-tokenize", "-m", MODEL, "-p", p],
                       capture_output=True, text=True, env=env)
    ids = [int(m.group(1)) for line in r.stdout.splitlines()
           if (m := re.match(r"\s*(\d+)\s*->", line))]
    assert ids, f"tokenize failed for {p!r}"
    ol = os.path.join(ROOT, "build/oracle_logits")
    r2 = subprocess.run([ol, MODEL, p, "--dump", f"tests/fixtures/oracle_lg_{i}.bin"],
                        capture_output=True, text=True, env=env)
    top8 = [l for l in r2.stdout.splitlines() if l.startswith("TOP8")][0]
    m = re.search(r"\((\d+),([\d.]+)\)", top8)
    out.append({"prompt": p, "tokens": ids,
                "oracle_argmax": int(m.group(1)), "oracle_val": float(m.group(2))})
json.dump(out, open("tests/fixtures/parity_set.json", "w"), indent=1)
print(f"wrote {len(out)} fixtures")
