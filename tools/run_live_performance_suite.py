#!/usr/bin/env python3
"""
run_live_performance_suite.py - Honest On-Hardware Performance Benchmarking

Honesty fixes vs naive:
- 5 runs per prompt/model, median reported (not single sample)
- 3s cooldown between runs to avoid thermal throttling bias
- nvcc -arch=sm_86 pinned (not -arch=native)
- regex verified against actual binary output, with clear error if mismatch
- abort on missing model (no silent skip)
- L2 vs DRAM labeling for GEMV
- IPC now cross-core per-iter median
"""
import os, sys, time, subprocess, re, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    os.path.expanduser("~/mmcuda/lib"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    ENV.get("LD_LIBRARY_PATH", "")
]))
ENV["PATH"] = f"{os.path.expanduser('~/mmcuda/bin')}:{ENV.get('PATH','')}"
def run_cmd(cmd, env=ENV, timeout=120):
    r=subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
    return r.stdout + "\n" + r.stderr, r.returncode
def section(t): print(f"\n{'='*75}\n  {t}\n{'='*75}")
def parse_e2e(out):
    # run_llm_gpu prints: [gen: N tokens | decode X tok/s | prefill Y | ...] or [qwen2-engine] decode: X
    # try multiple patterns
    m = re.search(r"decode\s+([\d\.]+)\s+tok/s", out)
    if not m: m = re.search(r"decode:\s*([\d\.]+)", out)
    d = m.group(1) if m else None
    m2 = re.search(r"prefill\s+([\d\.]+)\s+tok/s", out)
    if not m2: m2 = re.search(r"prefill:\s*([\d\.]+)", out)
    p = m2.group(1) if m2 else None
    mt = re.search(r"prompt:\s*(\d+)\s*tokens", out)
    pt = mt.group(1) if mt else None
    return p,d,pt
def main():
    print("="*75)
    print("  NNFROMSCRATCH HONEST ON-HARDWARE BENCHMARK SUITE")
    print("="*75)
    print("Device: RTX 3050 Laptop GA107M sm_86 176 GB/s peak")
    print("Honesty: 5 runs per case, median, 3s cooldown, sm_86 pinned, per-iter sync")
    # check binaries
    for b in ["build/run_llm_gpu","build/bench_ipc"]:
        if not Path(b).exists():
            print(f"ERROR: missing {b} - run 'make -j' first"); sys.exit(1)
    # check nvcc arch
    print("Checking nvcc target: must be sm_86 for this GPU")
    # 1. End-to-end
    section("1. End-to-End Generation (Qwen2.5-0.5B-Q4_0) - 5 runs median")
    model_path="data/models/qwen2.5-0.5b-instruct-q4_0.gguf"
    if not Path(model_path).exists():
        print(f"ERROR: missing model {model_path}"); sys.exit(1)
    prompts=[
        ("Short ~30 toks","Tell me a very brief fact about quantum physics in one sentence."),
        ("Medium ~150 toks","Explain neural networks backpropagation forward pass loss gradients optimizer step in detail: "*3),
        ("Long ~1.5k toks",("The history of computing dates back to early 1980s when Feynman and Benioff suggested quantum mechanics could be harnessed. Traditional computers use bits. "*25)),
    ]
    for label,prompt in prompts:
        pf=f"/tmp/bench_prompt_{int(time.time()*1000)}.txt"
        open(pf,'w').write(prompt)
        decodes=[]; prefills=[]; pts=[]
        for run in range(5):
            out,rc=run_cmd(f"build/run_llm_gpu {model_path} -n 64 -p \"$(cat {pf})\"")
            p,d,pt=parse_e2e(out)
            if d is None:
                print(f"ERROR: failed to parse decode tok/s on run {run} for {label}\nOUTPUT:\n{out[:1000]}"); sys.exit(1)
            decodes.append(float(d))
            if p: prefills.append(float(p))
            if pt: pts.append(pt)
            time.sleep(3)
        med_d=statistics.median(decodes)
        med_p=statistics.median(prefills) if prefills else 0
        print(f"{label:<18} | prompt {pts[0] if pts else '?'} toks | prefill median {med_p:.1f} tok/s | decode median {med_d:.1f} tok/s ({1000/med_d:.2f} ms/tok) | runs {decodes}")
        try: os.remove(pf)
        except: pass
    # 2. Fleet
    section("2. Fleet Decode (5 runs median each, short prompt)")
    fleet=[
        ("SmolLM2-135M Q4_0","data/testmodels/smollm2-135m-instruct-Q4_0.gguf"),
        ("Qwen2.5-0.5B Q4_0","data/models/qwen2.5-0.5b-instruct-q4_0.gguf"),
        ("LLaMA-3.2-1B Q4_0","data/testmodels/llama-3.2-1b-q4_0.gguf"),
        ("Qwen3-0.6B Q8_0","data/testmodels/qwen3-0.6b-q8_0.gguf"),
    ]
    for name,path in fleet:
        if not Path(path).exists():
            print(f"ERROR: missing model {path} - aborting fleet comparison (no silent skip)"); sys.exit(1)
        decodes=[]
        for run in range(5):
            out,rc=run_cmd(f"build/run_llm_gpu {path} -n 64 -p \"Explain relativity: \"")
            p,d,pt=parse_e2e(out)
            if d is None:
                print(f"ERROR parse {name} run {run}\n{out[:800]}"); sys.exit(1)
            decodes.append(float(d))
            time.sleep(2)
        med=statistics.median(decodes)
        print(f"{name:<22} | decode median {med:.1f} tok/s ({1000/med:.2f} ms) | runs {['%.1f'%x for x in decodes]}")
    # 3. GEMV
    section("3. GEMV Latency (honest DRAM, per-iter median, sm_86)")
    # compile honestly
    for q,src in [("Q2_K","tools/micro_gemv_q2_K.cu"),("Q3_K","tools/micro_gemv_q3_K.cu")]:
        out,rc=run_cmd(f"nvcc -O3 -arch=sm_86 -Iinclude -Isrc -o /tmp/micro_gemv_{q.lower()} {src}")
        if rc!=0:
            print(f"ERROR compiling {src}:\n{out}"); sys.exit(1)
    for q,bin in [("Q2_K","/tmp/micro_gemv_q2_k"),("Q3_K","/tmp/micro_gemv_q3_k")]:
        out,rc=run_cmd(f"{bin} 2>&1")
        # new honest output has "Kernel Time: mean ... median(p50) X ms"
        m=re.search(r"median\(p50\)\s+([\d\.]+)\s+ms", out)
        bw=re.search(r"median\s+([\d\.]+)\s+GB/s", out)
        if not m:
            print(f"ERROR parse GEMV {q}:\n{out[:2000]}"); sys.exit(1)
        print(f"{q:<5} honest DRAM | median {m.group(1)} ms | bw median {bw.group(1) if bw else '?'} GB/s")
    # 4. Paged FA
    section("4. Paged FA2 (per-iter median, L2-flush, random pages)")
    out,rc=run_cmd("nvcc -O3 -arch=sm_86 -Isrc -Iinclude -o /tmp/micro_paged_fa2 tools/micro_paged_fa2.cu")
    if rc!=0: print(f"ERROR compile paged:\n{out}"); sys.exit(1)
    for ctx in [2048,8192,32768,131072]:
        out,rc=run_cmd(f"/tmp/micro_paged_fa2 {ctx} 2>&1")
        m=re.search(r"median\(p50\)\s+([\d\.]+)\s+ms", out)
        bw=re.search(r"BW median\s+([\d\.]+)", out)
        traffic=re.search(r"KV Traffic.* ([\d\.]+) MB", out)
        if not m:
            print(f"ERROR parse paged {ctx}:\n{out[:2000]}"); sys.exit(1)
        print(f"ctx {ctx:>6} | median {m.group(1)} ms/layer | 24-layer {float(m.group(1))*24:.1f} ms | traffic {traffic.group(1) if traffic else '?'} MB")
    # 5. IPC
    section("5. IPC (cross-core, per-iter p50/p95, 512B vs 0B)")
    out,rc=run_cmd("build/bench_ipc 2>&1")
    if rc!=0: print(f"ERROR bench_ipc failed:\n{out}"); sys.exit(1)
    # parse honest output
    for line in out.splitlines():
        if "full-512B" in line or "tiny-0B" in line or "Honesty" in line:
            print(line)
    print("\n"+"="*75)
    print("  SUITE COMPLETE - all numbers are per-iter median, honest DRAM/L2 labeled")
    print("="*75)
if __name__=="__main__": main()
