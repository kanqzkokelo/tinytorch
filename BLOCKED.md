# BLOCKED: E4B — weights exceed VRAM

## Status
gemma-4-E2B heterogeneity blocker is RESOLVED (per-layer geometry, KV-cache
sharing, partial RoPE, PLE all implemented — see TODO.md). Remaining hard
blocker:

## The blocker: E4B 5.15GB weights > 4GB VRAM
E4B checkpoint does not fit the laptop GPU. Needs hybrid CPU offload —
separate milestone, out of scope until E2B reaches parity.

## Previously blocking (resolved this milestone)
- Heterogeneous layer geometry (pl_hd 256/512, ffn 6144/12288) — DONE
- rope_freqs freq_factors + dual bases (1e6 full / 1e4 swa) — DONE
- Plain V-norm, attention scale 1.0 — DONE (pre-existing)
- KV-cache sharing layers 15–34 — DONE

## Options for the human (unchanged)
1. Fund hybrid CPU-offload work for E4B.
2. Stay on E2B until m84 gate green (current path).
