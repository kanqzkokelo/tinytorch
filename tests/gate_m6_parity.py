#!/usr/bin/env python3
"""Gate Q2-lite — greedy-decoding parity vs llama.cpp oracle.

Runs both engines on identical raw prompts and compares generated text.
Exit 0 only if >= threshold of prompts match token-exactly (whitespace-trimmed
prefix comparison over the llama.cpp continuation).

Usage: gate_m6_parity.py [--model PATH] [--min-agree 0.8] [--tokens 16]
Requires: build/run_llm_gpu, oracle/llama.cpp/build/bin/llama-cli
"""
import subprocess, sys, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "data/models/qwen2.5-0.5b-instruct-q4_0.gguf")
OURS = os.path.join(ROOT, "build/run_llm_gpu")
LLAMA = os.path.join(ROOT, "oracle/llama.cpp/build/bin/llama-cli")

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "My name is Maria. I like to eat",
    "The three primary colors are red,",
    "Once upon a time in a distant kingdom",
    "The largest planet in the solar system is",
    "Photosynthesis is the process by which",
]

def run_ours(prompt, n):
    env = dict(os.environ, TT_RAW_PROMPT="1")
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        os.path.expanduser("~/mmcuda/lib"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
        env.get("LD_LIBRARY_PATH", "")]))
    r = subprocess.run([OURS, prompt, str(n)], capture_output=True, text=True, timeout=180, env=env)
    # stdout layout: <banner lines>\n<streamed tokens>\"\n[gen: ...]
    out = r.stdout.split("\n")
    # drop banner lines (contain [run]/[GGUF]/[BPE] markers) and stats tail
    toks = [l for l in out if "[gen:" not in l and not l.startswith(("[run]", "[GGUF]", "[BPE]"))]
    txt = "".join(toks).split('"')[0]
    return txt

def run_llama(prompt, n):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(ROOT, "oracle/llama.cpp/build/bin")
    r = subprocess.run([LLAMA, "-m", MODEL, "-p", prompt, "-n", str(n),
                        "--temp", "0", "-no-cnv"],
                       capture_output=True, text=True, timeout=180, env=env)
    lines = [l for l in r.stdout.splitlines() if l.strip() and not l.startswith(("llama_perf", "="))]
    return lines[-1] if lines else ""

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

def main():
    min_agree = 0.8
    n_tokens = 16
    args = sys.argv[1:]
    if "--min-agree" in args: min_agree = float(args[args.index("--min-agree")+1])
    if "--tokens" in args: n_tokens = int(args[args.index("--tokens")+1])

    agree = 0
    total = 0
    for p in PROMPTS:
        ours = run_ours(p, n_tokens)
        ref = run_llama(p, n_tokens)
        total += 1
        o, r = norm(ours or ""), norm(ref)
        ok = o == r or (len(r) > 20 and o.startswith(r[:max(20, len(r)-3)])) or \
             (len(o) > 20 and r.startswith(o[:max(20, len(o)-3)]))
        status = "MATCH" if ok else "DIFFER"
        if ok: agree += 1
        print(f"[{status}] {p!r}")
        print(f"   ours : {o[:110]}")
        print(f"   llama: {r[:110]}")
    print(f"\nparity: {agree}/{total} (threshold {min_agree:.0%})")
    sys.exit(0 if agree / max(total,1) >= min_agree else 1)

if __name__ == "__main__":
    main()
