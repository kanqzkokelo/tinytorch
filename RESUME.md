# SESSION STATE — resume here (end of 2026-08-24)

## Done & committed (branch m6-correctness)
- M6.1 parity, M6.3 perf: 286.7 tok/s ctx<=64 / ~250 sustained, parity 7/7 always
- chat: ./chat launcher, sampling (TT_TEMP/TT_REPEAT_PENALTY/TT_GREEDY), multi-turn fix
- M7 Task -1..3 committed: test fleet (14 gguf), golden dequant (10 types),
  CUDA GEMV dispatch (golden-verified), trait-driven forward pass (qwen2 byte-identical)

## IN FLIGHT — M7 Task 4 (per-arch bring-up), UNCOMMITTED working tree:
- tools/dump_logits.c: --model PATH / TT_MODEL support (DONE, works)
- kernels/qwen2_cuda.cu: small edits (+35/-12) — review before trusting
- tests/gate_m7_arch.py: generalized parity gate (NEW, untested?)

## Next steps (in order)
1. Review/commit the WIP diff above
2. SP tokenizer backend for llama-family vocab (decode ▁->space+<0xNN>; encode greedy score match)
   in tokenizer_bpe.c — this is why tinyllama text is garbage
3. Bring-up order w/ tests/gate_m7_arch.py: tinyllama-f16 -> smollm2 x10 quants ->
   qwen3-0.6b -> gemma2-q6_k (hardest; softcap/SWA/norm-offset already trait'd)
4. Then plan Tasks 5 (grid + README)

## Gotchas learned
- pi subagents died silently twice on Task 4 — do it directly in-session
- gate_chat.py now pins TT_GREEDY=1 (determinism); sampling is chat-only feature
- llama-quantize needed building in oracle tree (done once, persists)
- K-quants require n_per_row %256==0 (dispatcher enforces)
- lm-head blockDim sweep lesson: grid must match actual b.y (old bug class)
