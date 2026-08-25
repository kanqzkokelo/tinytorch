# BLOCKED: M8 Gemma-4 (E2B/E4B) — requires per-layer heterogeneous architecture

## Status
Engine loads and runs gemma-4-E2B but produces wrong output. Root cause
discovered via full tensor inventory (TT_DEBUG upload log):

## The blocker: gemma4-E2B is a HETEROGENEOUS mixture model
- Layers 0-3, 5-8, ... : q_out=2048 (8 heads x 256), k/v_out=256 (1 kv head),
  ffn=6144
- Layers 4, 9, 14, 19, 24, 29 (every 5th): q_out=4096 (16 heads x 256),
  k/v_out=512 (2 kv heads), ffn=6144
- Layer 15+: ffn=12288 (double-wide MLP), others 6144
- All layers also carry inp_gate [256,1536], proj [1536,256],
  layer_output_scale [1], plus top-level per_layer_token_embd [262144, 8960]
  (MatFormer per-layer inputs, scaled by sqrt(256), projected per layer)

The engine assumes ONE uniform layer geometry (single TTConfig). Supporting
gemma4 requires per-layer dims: separate head counts, kv counts, head dims,
and ffn widths per layer group — an engine capability addition comparable in
size to the entire M6 rewrite.

Additionally needed once unblocked:
- rope_freqs.weight learned frequency factors (global attention layers)
- dual rope bases (1e6 global / 1e4 swa)
- sliding_window_pattern handling
- v plain-rmsnorm, attention scale 1.0 (both already implemented)

## What IS complete (committed)
- Full forward math extraction (docs/plans/2026-08-24-m8-gemma4-port.md)
- Metadata array parser, trait entry, typed uploads incl. bf16 GEMV,
  PLE pipeline scaffold, sandwich norms, embed sqrt scaling
- Oracle upgraded to latest llama.cpp which SUPPORTS gemma4 natively
  (reference logits available for parity iteration when unblocked)

## Options for the human
1. Fund the per-layer-geometry refactor (est. 1-2 sessions) then gemma4 works.
2. Target a uniform-layer gemma-family model instead (e.g. gemma2-2b — DONE).
3. Accept partial support: run E2B's uniform layers only (not a real model).

## Evidence
Full tensor inventory captured in-session (TT_DEBUG upload log): layers 0-34
inventoried; heterogeneity confirmed by shape diffs across layer groups.
