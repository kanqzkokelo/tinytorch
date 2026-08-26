# RESUME — M8 gemma4 (fresh-session handoff)

## What works
- gemma4-E2B loads + runs end-to-end, all mechanisms implemented:
  per-layer heterogeneous geometry (heads/kv/ffn/head_dim arrays), partial
  RoPE (full-attn 1e6+freq_factors / SWA 1e4 plain), plain V-norm,
  attention scale 1.0, PLE/MatFormer blocks, KV-cache sharing.
- Single-token median |Δlogit| = 2.66 (was 22.06 pre-KV-share), measured by
  tests/gate_m84_gemma4.py single-token probe.
- Regression fleet re-verified this session: m61 7/7, chat-multiturn PASS,
  qwen3 0.047, gemma2 6/7, tinyllama 7/7 bit-perfect.

## What's broken
- Two-token median |Δlogit| ~9 → m84 parity gate NOT green.
- Bisection of two-token divergence is IN FLIGHT (agent running) — check its
  result before starting anything; do not duplicate.

## Exact next steps
1. Collect two-token bisection result from running agent; fix root cause.
2. ./scripts/verify.sh m84   # until ≥6/7 green
3. TT_MODEL=data/models/gemma-4-E2B-it-Q4_0.gguf ./chat   # smoke
4. tg-128 bench vs llama.cpp oracle; target ≥25–30 tok/s (Windows baseline 17).
5. README grid: flip gemma4 row to ✅ with real numbers.

## Key file:line pointers
- shared_kv_layers meta parse: src/loader_gguf.c:181
- pl_src mapping + kv_shared cache skip: kernels/qwen2_cuda.cu forward_layers
  ~line 1025 (`const int kv_shared = e->has_pl_embd && e->pl_src[l] >= 0;`)
- Golden NumPy reference: tests/ref_gemma4_numpy.py
- Parity gate: tests/gate_m84_gemma4.py (wired as verify.sh m84)
- Plan spec: docs/plans/2026-08-24-m8-gemma4-port.md

## Key facts
- gemma4-E2B: dim=1536 L=35 H=8 KV=1 HD=256 vocab=262144 ffn=6144/12288(mixed)
- KV sharing: layers 15–34 reuse L13(swa)/L14(full) caches
- BF16 loader size bug fixed (was half → NaN); TTQ_BF16 GEMV/embed kernels exist

## Blocked (do not attempt)
- E4B: 5.15GB weights > 4GB VRAM — see BLOCKED.md.
