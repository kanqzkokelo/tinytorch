# RESUME — M8 gemma4 port (active debugging)

## State: gemma4-E2B loads+runs end-to-end; segfault in PLE host path
- compute_ple() crashes at/after D2H of d_ple_tmp
- Likely: pe_full[64*1024] buffer vs row=8960 should fit; check ttq_dequant
  bounds on q5_K row_bytes calc (row/256*176 for 8960 vals = 6160B)
- Debug prints partially added (TT_DEBUG grep "PLE stage")
- All other M7/M8 work committed and stable

## Next steps (in order)
1. Fix compute_ple segfault (add fprintf before/after each stage)
2. Verify PLE values against NumPy golden
3. Wire MatFormer block into forward_layers (inp_gate->gelu->*PLE->proj->norm)
4. Parity gate >=6/7 on E2B
5. Chat smoke + README grid update

## Key facts
- gemma4-E2B: dim=1536 L=35 H=8 KV=1 HD=256 vocab=262144 ffn=6144/12288(mixed)
- per_layer_token_embd: [8960, 262144] type 13 (Q5_K)
- per_layer_model_proj: [1536, 8960] type 30 (BF16)
- All gates m61 + chat-multiturn + grid still PASS
