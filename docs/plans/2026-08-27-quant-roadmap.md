
## CORRECTION (main-session verification, 2026-08-26)
The earlier "gemma-3n: 30 layers, hd=256 everywhere" note conflates gemma-3n
with OUR file. Ground truth for data/models/gemma-4-E2B-it-Q4_0.gguf
(verified from tensors + metadata): 35 layers, SWA layers hd=256, global
layers hd=512, shared_kv_layers=20 → only 15 KV-owning layers. KV sharing
is ALREADY IMPLEMENTED in our engine (commit 2052181 lineage) — item (1)
of the KV list above is done, not pending.
