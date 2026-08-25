# SESSION STATE — resume here

# RESUME (latest — supersedes below)
## Oracle UPGRADED to latest llama.cpp 0a5ac49b (qwen3+gemma4 aware). m61 still green.
## qwen3-0.6b gate: 7/7 PASS. Grid runner: tests/gate_m7_grid.py (11/14 green).
## README truth table written.
## NEXT: M8 gemma4 port — FULL SPEC at docs/plans/2026-08-24-m8-gemma4-port.md
## (all math extracted from llama.cpp build_gemma4; E2B downloaded complete 3.04GB;
##  old gemma-4 files in data/models/*.incomplete were truncated downloads - delete)
## Registry/loader already carry gemma4 entry + array-metadata parsing (committed).

---

## Branch m6-correctness. M7 Task 4 COMPLETE for llama/gemma2 families.

## Verified grid (gate_m7_arch.py, teacher-forced logits vs llama.cpp oracle)
- tinyllama-f16 (llama): 7/7 BIT-PERFECT (median dlogit 0.0009)
- smollm2 x Q4_0/Q5_0/Q8_0: 7/7 each; Q6_K 6/7 (relaxed tol 0.6 for 135M margins)
- gemma2-2b-q6_k: 7/7 (top1 all match)
- qwen2.5-0.5b q4_0/q8_0-head: m61 gate 7/7 standing

## REMAINING M7
1. qwen3-0.6b: BLOCKED on oracle age — pinned llama.cpp predates arch 'qwen3'.
   Fix: update oracle checkout (rebuild tools), regenerate qwen fixtures.
2. Task 5: grid runner (tests/gate_m7_grid.py) + README truth table.
3. Chat smoke on remaining models if desired.

## THEN M8: gemma4 support (user goal: gemma-4-E4B_q4_0-it.gguf, data/models/)
- E4B = 5.15GB q4_0: DOES NOT FIT 4GB VRAM. E2B (1.46GB) fits — target it first.
- Needs: per-layer input embeddings (gemma4.embedding_length_per_layer_input=256),
  heterogeneous head dims (global 512 vs SWA 256 per layer), dual rope bases
  (1e6 global / 1e4 swa), sliding_window pattern, softcap 30, SP tokenizer
  ("gemma4" ggml model -> ensure tokenizer sp_mode covers it).
- User expectation: >=25-30 tok/s (Windows llama.cpp baseline was 17).
- NOTE E4B hybrid offload = separate engineering if E2B insufficient.

## Gotchas (accumulated)
- rope: llama-family = GPTJ interleaved; qwen/gemma = NEOX half-split (empirical)
- gate_chat.py pins TT_GREEDY=1; ctx accounting infers from turn-2 delta
  (turn 1 includes system prompt ~33 tokens)
- oracle_logits supports --ids mode (input alignment essential for SP models)
- K-quants need n_per_row %256==0; gemma2 key_length makes Q/O GEMV dims
  n_heads*HD != dim — never assume square projections
- pi subagents die silently on long GPU tasks — run in-session directly
