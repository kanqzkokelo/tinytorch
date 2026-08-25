# TODO — M8: Gemma-4 port (E2B target)

Spec: docs/plans/2026-08-24-m8-gemma4-port.md (complete forward math inside)
Groundwork already committed: gemma4 registry entry, array-metadata parser,
gcd-based head_dim derivation, E2B model downloaded (3.04GB, verified complete).

- [ ] M8.1 Config derivation: hidden_dim from ffn_down tensor K (=6144, NOT meta
      12288); verify heads=8/kv=1/hd=256 derive correctly
- [ ] M8.2 Uploads + PLE pipeline: per_layer_model_proj/pl_proj_norm uploads,
      host+device gamma copies, d_ple_cache [ctx x 35x256], eager PLE compute
      at embed time (gemv -> D2H -> rmsnorm slices -> +pe*16 -> *1/sqrt2 -> H2D)
- [ ] M8.3 Forward additions: inp_gate GEMV -> gelu -> *PLE slice -> pl_proj
      GEMV -> norm(?) -> residual; layer_output_scale multiply; V plain RMSNorm;
      attention scale 1.0 (trait)
- [ ] M8.4 Parity gate: tests/gate_m7_arch.py --model gemma-E2B >=6/7
      (debug: single-token first, then two-token; NumPy golden = math table in plan)
- [ ] M8.5 Chat smoke: ./chat with TT_MODEL=gemma E2B; add thinking-channel
      markers (<|channel>thought etc.) to stop strings; README grid update

## Open questions to resolve empirically (from plan)
- pl_proj@g normalization gamma source (per_layer_proj_norm top-level assumed)
- rope base per layer: try all-global 1e6 first; freq_factors second
- layer_output_scale raw vs transformed

## Deferred
- E4B support (5.15GB > 4GB VRAM): needs hybrid CPU offload — separate milestone
- CUDA graph capture for gemma4 path (PLE host stage) — optimization later
- qwen3 note: DONE (7/7). Oracle upgraded to 0a5ac49b.
