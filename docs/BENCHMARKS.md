# Benchmarks — Qwen2.5-0.5B-Instruct-Q4_0, RTX 3050 Laptop (sm_86, 4GB)

HOT medians, 2 warmups + 3 measured. Prefill prompt 752 tokens; decode gen 128.
`verify.sh m61` PASS (7/7 parity + chat), backfill PASS, 5/5 greedy-match default vs best.

| Config | Prefill tok/s | Prefill ms | Decode tok/s |
|---|---|---|---|
| Default | 1416.5 | 530.9 | 224.2 |
| `TT_CUBLAS_PRE=1 TT_FA2_PRE=1` | 6408.3 | 117.3 | — |
| `TT_CUBLAS_FP16=1 TT_FA2_PRE=1` (best) | 7736.6 | 97.2 | 212.8 |

Best = **5.5x** default prefill. Decode flags add small overhead (~5%); prefill flags target prefill only.
DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M: coherent, "The answer is 144."
