# Supported Model Families

GGUF `general.architecture` keys resolved by `src/arch_registry.c` (`kArchTable`).
Traits: RoPE variant · activation · softcap · SWA · tied embeddings · QK-norm.
Unknown keys → resolve fails (`-1`); KV overrides (`sliding_window`, `final_logit_softcapping`) apply where present.

## Fully supported (kernels + loader + chat template)

| GGUF arch | Family / models | Notes |
|---|---|---|
| `qwen2` | Qwen2 / Qwen2.5 (0.5B–72B) | NEOX half-split RoPE, SwiGLU, QKV-bias auto-detect |
| `qwen3`, `qwen3_moe` | Qwen3 dense + MoE | + per-head QK-RMSNorm (eps from `attention.key_epsilon`, default 1e-6) |
| `llama`, `llama2`, `llama3` | Llama 1/2/3.x, Mistral-7B, TinyLlama, SmolLM | GPTJ RoPE, SwiGLU, untied head |
| `mistral` | Mistral-7B (native key) | Alias of llama path |
| `smollm`, `smollm2` | SmolLM 135M–1.7B | Alias of llama path |
| `tinyllama` | TinyLlama (keeps HF name) | Alias of llama path |
| `granite` | Granite dense | Alias of llama path |
| `gemma` | Gemma 1 | NEOX, GeGLU, tied embeddings |
| `gemma2` | Gemma 2 (9B/27B) | + SWA 4096 default, logit softcap 30.0 |
| `gemma3` | Gemma 3 | SWA 4096, softcap 30.0, tied |
| `gemma4` | Gemma 4 (E2B/E4B) | QK-norm every layer, attn scale 1.0, MatFormer per-layer embeddings |
| `internlm2` | InternLM 2 | GPTJ + SwiGLU, untied (Tier-1, verified vs llama.cpp) |
| `xverse` | XVERSE | GPTJ + SwiGLU, untied (Tier-1) |
| `exaone` | EXAONE 3.x | NEOX + SwiGLU, tied (Tier-1) |
| `ernie4_5` | ERNIE 4.5 dense | GPTJ + SwiGLU, tied (Tier-1, underscore key) |

## Chat templates (`src/chat_template.c`)
- QWEN2/3: ChatML + `im_end`
- GEMMA: `start_of_turn`; GEMMA4: `<|turn|>`/`<turn|>`
- LLAMA3: headers + `eot_id`

## GPU notes
- RTX 3050/3090 (sm_86): native. RTX 4090 (sm_89): cubins shipped.
- 4 GB VRAM fits ≤1B FP32-shadow prefill comfortably; 7B+ needs bigger cards.
- FP16-shadow prefill (`TT_CUBLAS_FP16`), FA2 flash (`TT_FA2_PRE`) are opt-in flags.

## Explicitly NOT supported (roadmap)
- `qwen3next`, `qwen35`/`qwen35moe` (Qwen3.8-27B): Gated DeltaNet linear attention — needs new kernels
- `deepseek2` family: MLA latent attention — needs new kernel
- `phi2/3`, `falcon`, `mpt`, `bloom`, `gpt2`, `starcoder`, `cohere`, …: LayerNorm+bias path — needs LN kernel (~20 fams, biggest unlock)
- `mamba`/`jamba`, `rwkv*`: SSM / time-mix kernels
- `qwen2vl`/`gemma4-asst`/VLMs, TTS, encoders: multimodal, out of scope
- MoE execution (`qwen3_moe` resolves traits; expert dispatch engine pending)
