# TODO — M8: Gemma-4 port (E2B target)

Spec: docs/plans/2026-08-24-m8-gemma4-port.md (complete forward math inside)

## Done
- [x] M8.1 Config derivation: head_dim via gcd(q_rows,k_rows)=256, hidden_dim
      from ffn_down tensor K (=6144); metadata lies on gemma4, shapes win.
      Verified dim=1536 L=35 H=8 KV=1 HD=256 vocab=262144.
- [x] Heterogeneous per-layer geometry: heads/kv/ffn/head_dim arrays
      (pl_hd 256/512, ffn 6144/12288); staging buffers sized to maxima.
- [x] Partial RoPE: rope_freqs[256] factors divide theta — pairs with ≥32
      factors → 1e30 → identity. Full-attn layers base 1e6 + _factors;
      SWA layers base 1e4 plain.
- [x] Plain V-norm (ones-gamma trick) + attention scale 1.0 trait.
- [x] KV-cache sharing: layers 15–34 reuse L13(swa)/L14(full) caches; loader
      parses shared_kv_layers meta (src/loader_gguf.c:181; verified against
      oracle llama-model.cpp:2502).
- [x] BF16 loader bug fixed (size was half → NaN); TTQ_BF16 GEMV/embed kernels.
- [x] NumPy golden reference tests/ref_gemma4_numpy.py.
- [x] Single-token median |Δlogit| 22.06 → 2.66 after KV-share
      (tests/gate_m84_gemma4.py single-token probe).

## Open (in order)
- [ ] Two-token divergence bisection (~9 median) — agent running; do not
      duplicate work, check its result first.
- [ ] M8.4 Parity gate green: ./scripts/verify.sh m84
      (tests/gate_m84_gemma4.py) — blocked on two-token fix.
- [ ] Chat smoke E2B: TT_MODEL=<gemma-E2B gguf> ./chat (thinking-channel
      markers in stop strings).
- [ ] tg-128 benchmark vs llama.cpp; target ≥25–30 tok/s (Windows baseline 17).
- [ ] Final README grid update once m84 passes.

## Deferred
- E4B support: 5.15GB > 4GB VRAM — see BLOCKED.md.
- CUDA graph capture for gemma4 path (PLE host stage).
