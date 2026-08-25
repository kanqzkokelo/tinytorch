# SESSION STATE — M8 gemma4 port IN PROGRESS

## What works NOW (committed)
- Engine LOADS AND RUNS gemma-4-E2B-it-Q4_0.gguf end-to-end, no crash
  (config derivation from tensors works: dim=1536 L=35 H=8 KV=1 HD=256 vocab=262144)
- All prior gates green: m61 7/7, tinyllama bit-perfect, smollm2 matrix, qwen3 7/7,
  gemma2 7/7, chat working
- Oracle upgraded to latest llama.cpp (0a5ac49b): supports qwen3 AND gemma4 archs
- Grid runner: tests/gate_m7_grid.py + README truth table committed

## WHAT'S LEFT for gemma4 parity (current gap: single-token median |dlogit| ~10)
The novel pieces are IMPLEMENTED but one or more details are wrong:
1. PLE pipeline (embed_token -> compute_ple): per_layer_model_proj GEMV on GPU ->
   D2H -> per-256-slice rmsnorm w/ pl_proj_norm_host -> add pe_row*sqrt(256) ->
   *1/sqrt2 -> H2D to d_ple_row. VERIFY against llama.cpp
   project_per_layer_inputs() (oracle src/models/gemma4.cpp:424-455).
2. Forward MatFormer block (forward_layers, has_pl_embd section):
   inp_gate gemv -> gelu -> *PLE slice -> pl_proj gemv -> **MISSING: rmsnorm with
   blk.N.post_norm.weight** (build code line ~353 uses layers[il].per_layer_post_norm
   = blk.N.post_norm.weight [1536]) -> residual.
3. NOT YET WIRED: rope_freqs factors for global layers (rope_freqs.weight [256]
   scales theta per-dim); attention scale IS wired via trait (1.0).
4. V plain norm + QK norms wired via ones-vector/trait ✓ verify active.

## Debug protocol
Single token "hi" ids=[2,2202]: ours ARGMAX 198580@29.33 vs oracle TOP1 9079@19.29.
Bisect: dump x after embed (before any layer) vs numpy dequant(token_embd row)*sqrt(dim).
Then after layer 0 vs hand-computed. NumPy golden = math table in
docs/plans/2026-08-24-m8-gemma4-port.md.

## Also pending
- E4B file (5.15GB .incomplete) exceeds VRAM - delete or keep for future offload work
- TODO.md has full M8 task list with open questions
