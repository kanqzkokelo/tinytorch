# SESSION STATE — resume here (end of 2026-08-24, late)

## Branch state: m6-correctness @ 992e20d, tree clean

## DONE (M7)
- Task -1 multi-turn fix (3 root causes: q4_0 nibble pairing + unsigned underflow,
  lm-head grid/blockDim mismatch, missing special-token encode)
- Task 0 fleet: 14 ggufs incl. 10-quant smollm2 matrix (data/testmodels/, gitignored)
- Task 1 golden dequant all types (gguf-py verified; caught real q5_K field-order bug)
- Task 2 CUDA GEMV dispatch all types (golden-verified, maxerr ~1e-6)
- Task 3 traits: rope style / activation / qk-norm / softcap / swa / norm-offset;
  qwen2 byte-identical; loader suffix-based keys + arch string
- Task 4 MOSTLY: SP tokenizer done (encode matches llama-tokenize);
  tinyllama-f16 BIT-PERFECT parity (7/7, median 0.0009) with rope=GPTJ;
  smollm2 x10 quant matrix 63/70 prompts at relaxed tolerance (see results.md)

## NEXT (M7 remaining)
1. qwen3-0.6b gate (forward already runs clean; just run tests/gate_m7_arch.py --model data/testmodels/qwen3-0.6b-q8_0.gguf)
2. gemma2-2b-q6_k bring-up (SP decode works via sp_mode; check softcap/SWA/norm-offset traits vs oracle; gemma tokenizer.ggml.model may differ — check sp_mode triggers)
3. smollm2 chat smoke (TT_MODEL=... ./chat)
4. silence optional-bias stderr spam behind TT_DEBUG
5. Task 5 grid runner + README truth table

## THEN: gemma4-E4B milestone (user's actual goal!)
- data/models/gemma-4-E4B_q4_0-it.gguf = 5.15GB — DOES NOT FIT 4GB VRAM
- E2B variant (1.46GB) fits and is the realistic target (~60-90 tok/s potential)
- Needs: heterogeneous head dims per layer (global 512 vs SWA 256), per-layer
  input embeddings (256-dim MatFormer stream), dual rope bases (1e6/1e4),
  softcap 30, SWA 512, SP tokenizer ("gemma4" ggml model => ensure sp_mode)
- User expects >=25-30 tok/s (their Windows llama.cpp baseline was 17)

## Gotchas
- pi subagents die silently on long tasks — work directly in-session
- gate_chat.py pins TT_GREEDY=1; sampling is chat-level only
- oracle_logits supports --ids mode now (input alignment matters for SP models!)
- llama-family = GPTJ rope (empirically proven); qwen/gemma = NEOX... NOTE:
  registry comment cites llama.cpp ROPE_TYPE_NORM=interleaved — kept GPTJ after A/B;
  clean up the contradictory comment in src/arch_registry.c line ~52 someday
