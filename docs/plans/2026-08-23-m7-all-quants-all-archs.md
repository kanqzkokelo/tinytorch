# M7: All Practical Quants + All Major Model Architectures Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this task-by-task.

**Goal:** tinytorch loads and correctly runs every widely-downloaded GGUF quantization format across the major open-model families (Llama/Mistral, Qwen2/3, Gemma, Phi, SmolLM), verified against llama.cpp oracle parity per model×quant combination.

**Architecture:** Three layers of generalization. (1) A dequant dispatch layer — CPU golden functions validated against gguf-py, then CUDA kernels per quant type behind a single `tt_matvec_typed(weights, type_tag, x, out)` entry point, replacing today's hardcoded q4_0/q8_0 pair. (2) An architecture-trait layer — TTConfig grows per-family fields (rope style, activation, norm placement, bias map, sliding window, vocab/tied-embedding rules) and `forward_layers` becomes a trait-driven builder instead of Qwen2-only code. (3) A verification grid — one small model per architecture, quantized locally to every supported format via `llama-quantize`, each combination run through the existing logit-parity gate machinery.

**Tech Stack:** Existing engine (`src/loader_gguf.c`, `kernels/*`, `include/qwen2_engine.h`), llama.cpp oracle build (`oracle/llama.cpp` — includes `llama-quantize` for generating test matrices), `gguf-py` from the oracle tree as dequant ground truth, Python3/numpy gates.

**Hardware envelope:** RTX 3050 laptop 4 GB — every test model must be ≤2.5 GB on disk.

**Scope tiers (be honest about these):**
- **Tier 1 (this plan's commitment):** quants q4_0✓, q8_0✓, f16, f32, **q4_1, q5_0, q5_1, q4_K, q5_K, q6_K**; architectures **Qwen2✓, Llama/Mistral/TinyLlama/SmolLM (llama-family), Qwen3, Gemma/Gemma2** (incl. its SentencePiece tokenizer).
- **Tier 2 (follow-up plan):** i-quants (iq2_xxs…iq4_xs), Phi-3/Phi-2, GPT-NeoX/J (interleaved-RoPE legacy), StableLM.
- **Out of scope until asked:** MoE (Mixtral/DeepSeek-MoE), Mamba/state-space, multimodal vision towers.

---

## Task -1: Fix multi-turn chat degradation (known regression, blocks chat-facing work)

Observed 2026-08-23 (`cff5bd8`): after ~200 ctx tokens, generation collapses into
repetitive degenerate output ("!!!!!…") across turns; single-turn is clean.
Suspect: cross-turn KV/pending-token interaction with the graph-replay pipeline
(the Task-3 "prefill flush" path), OR genuine long-context model behavior at
q4_0 — distinguish first!

**Step 1:** Reproduce minimally: `printf 'hi!\nWhat is 2+2?\n/exit\n' | ./chat` and
a longer two-turn session; capture at which ctx position degeneration starts.
**Step 2:** A/B with `TT_NO_GRAPH=1 ./build/chat_llm_gpu` — if clean, the bug is
in graph/pending plumbing; if identical, it is cache-state or model behavior.
**Step 3:** If plumbing: audit pending-flush vs pos accounting at turn boundary.
If model/cache: dump per-layer hidden at turn 2 start vs fresh-session equivalent
(NumPy ref supports this) and bisect.
**Step 4:** Gate: two-turn coherence test added to `tests/gate_chat.py` expectations;
commit fix.

## Task 0: Test fleet + local quant matrix (unblocks everything)

**Files:**
- Create: `scripts/setup_test_fleet.sh`
- Create: `tests/fixtures/models.json`

**Step 1: Write the fleet script**

```bash
#!/usr/bin/env bash
# Downloads one small model per architecture (ungated HF repos only),
# then generates a full quant matrix from the smallest via llama-quantize.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/testmodels && cd data/testmodels
Q="$PWD/../../oracle/llama.cpp/build/bin/llama-quantize"

fetch() { # url outfile
  [ -f "$2" ] && { echo "have $2"; return; }
  curl -L --fail -o "$2" "$1"
}

# --- per-architecture representatives ---
fetch https://huggingface.co/ggml-org/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-f16.gguf smollm2-135m-f16.gguf
fetch https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf qwen3-0.6b-q8_0.gguf
fetch https://huggingface.co/bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-f16.gguf tinyllama-f16.gguf
# gemma2 mirror (ungated); if URL 404s, substitute any ungated gemma2-2b GGUF f16
fetch https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-f16.gguf gemma2-2b-f16.gguf || echo "WARN: gemma2 fetch failed - resolve manually"

# --- quant matrix from the smallest model ---
for q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q4_K_S Q5_K Q5_K_S Q6_K; do
  [ -f "smollm2-135m-instruct-$q.gguf" ] || "$Q" smollm2-135m-f16.gguf "smollm2-135m-instruct-$q.gguf" "$q" 2>/dev/null || echo "quant $q failed"
done
ls -la
```

**Step 2:** `chmod +x scripts/setup_test_fleet.sh && ./scripts/setup_test_fleet.sh`
Expected: 4 base models + 10 SmolLM2 quants in `data/testmodels/`. Any gated/404 repo: substitute an ungated mirror, note substitution in `tests/fixtures/models.json`.

**Step 3:** Write `tests/fixtures/models.json` manifest: `{name, path, arch (from gguf meta), tokenizer_model (gpt2|llama), notes}` for every file. Commit the manifest, never the models (data/ is gitignored).

**Step 4:** Commit: `git add scripts/setup_test_fleet.sh tests/fixtures/models.json && git commit -m "M7: test fleet + local quant-matrix generator"`

---

## Task 1: CPU golden dequant for every Tier-1 type (TDD foundation)

**Files:**
- Create: `tests/test_dequant_golden.py`
- Create: `src/dequant_ref.c` + `include/dequant_ref.h`

**Step 1: Golden test first (it will fail)**

```python
#!/usr/bin/env python3
"""Validate src/dequant_ref.c against gguf-py for every Tier-1 quant type."""
import sys, types, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "../oracle/llama.cpp/gguf-py"))
pkg = types.ModuleType("gguf"); pkg.__path__ = [
    os.path.join(os.path.dirname(__file__), "../oracle/llama.cpp/gguf-py/gguf")]
sys.modules["gguf"] = pkg
from gguf.quants import dequantize
import numpy as np, subprocess, json

TYPES = ["Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q4_K","Q5_K","Q6_K"]
MODEL = sys.argv[1]   # any gguf containing a big weight tensor
fails = 0
for t in TYPES:
    r = subprocess.run(["build/dequant_ref", MODEL, "token_embd.weight", t,
                        "/tmp/dq_ref.bin"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ours = np.fromfile("/tmp/dq_ref.bin", dtype="<f4")
    # reference: quantize f16 source to type t via llama-quantize, read back w/ gguf-py
    ref = np.load(f"/tmp/dq_gold_{t}.npy")
    ok = np.allclose(ours, ref, atol=2e-2 * max(1,np.abs(ref).max()))
    print(("PASS " if ok else "FAIL ") + t)
    fails += 0 if ok else 1
sys.exit(1 if fails else 0)
```

(Golden `.npy`s are produced once by a helper that quantizes SmolLM2-f16 to each type and reads rows via gguf-py — write `tests/fixtures/gen_dequant_gold.py` doing exactly that.)

**Step 2: Run** — `python3 tests/test_dequant_golden.py data/testmodels/smollm2-135m-instruct-Q4_0.gguf`
Expected: FAIL (`build/dequant_ref` doesn't exist).

**Step 3: Implement `dequant_ref.c`**

A CLI + reusable C function: reads named tensor from GGUF (reuse `loader_gguf.c` parsing), re-quantizes nothing — instead: load the f16 source tensor, apply GGML's quantize algorithm per type IN C (port `quantize_row_q4_1_ref`, `quantize_row_q5_0_ref`, … from ggml-quants.h semantics), then immediately dequantize, writing floats. Round-tripping through our own quantizer validates we understood the FORMAT; comparing against gguf-py-dequantized files catches layout misunderstandings. Reference implementations to port live at `oracle/llama.cpp/ggml/src/ggml-quants.c` — copy the math, credit in comments, no linking.

Block layouts cheat-sheet (from ggml-quants.h):
- q4_1: `d,f (fp16 fp16); qs[16]` — 20 B/32 val, value = d*x + f
- q5_0: `d(fp16); qh[4](5th bits); qs[16]` — 22 B/32
- q5_1: `d,f; qh[4]; qs[16]` — 24 B/32
- q6_K: super-blocks of 256: 16 sub-blocks `{8×int8 scales(signed), ql[64], qh[32]}` — 210 B/256
- q4_K/q5_K: super-blocks of 256, sub-block 32, `d,dmin` pairs, 6-bit values with per-subblock min offset — see `block_q4_K`/`block_q5_K` structs

**Step 4:** Build target `build/dequant_ref` in Makefile. Run Step-2 command → all PASS.

**Step 5:** `git commit -am "M7: CPU golden dequant round-trip for all Tier-1 types"`

---

## Task 2: Type-tagged weight upload + typed GEMV dispatch (CUDA)

**Files:**
- Modify: `include/loader_gguf.h` (size computation for ALL types — today non-q4_0/q8_0 still wrong!)
- Modify: `kernels/qwen2_cuda.cu`, `kernels/gemv_q4_cuda.cu` (rename mental model → `gemv_typed.cu`)
- Modify: `include/qwen2_engine.h`

**Step 1:** Fix `loader_gguf.c` size computation to cover q4_1/q5_0/q5_1/K-quants (mirror Task 1 layouts). Add a unit check inside `gen_fixtures`-style test asserting size matches actual file offsets for every tensor of the SmolLM2 quant matrix.

**Step 2:** Engine struct: replace `const BlockQ4_0*` pointers with `void *ptr; int dtype;` pairs (small `TTensor` struct). Upload becomes type-blind memcpy.

**Step 3:** Dispatch entry point:

```c
/* y[M] = W[M,K] @ x[K], W in any supported dtype */
int tt_gemv_typed(const void *W, int dtype, const float *x, float *y,
                  int M, int K, cudaStream_t stream);
```

Implement kernels per type: reuse existing q4_0/q8_0; new `k_gemv_q5_0` (bit-unpack 5 values via qh byte + two nibble planes), `k_gemv_q5_1` (same + min add), `k_gemv_q4_K/q5_K/q6_K` (super-block: one warp per 256-value super-block, sub-block scale applied per 32-lane segment — port scale math from Task 1 C reference; K-quants' `d,dmin` and signed 6-bit/8-bit scales are the tricky part, the CPU reference from Task 1 is the executable spec).

**Step 4:** Per-type golden micro-test on GPU: extend `tools/dump_logits`-style harness OR simpler — new `tools/test_gemv_typed.cu`: random x, real W rows, GPU result vs CPU `dequant_ref` dot-product, per type, atol 1e-2 relative. Run for every type × three shapes {(128,896),(4864,896),(896,4864)}.

**Step 5:** `./scripts/verify.sh m61` must stay green (Qwen path unchanged when fed q4_0/q8_0).

**Step 6:** Commit per type: `M7: CUDA GEMV for <type> (golden-verified)`.

---

## Task 3: Architecture traits — generalize the forward pass

**Files:**
- Modify: `include/qwen2_engine.h` → rename conceptually to engine traits (keep filename, add fields)
- Modify: `kernels/qwen2_cuda.cu`
- Create: `src/arch_registry.c` (key-name tables per family)

**Step 1: Extend TTConfig**

```c
typedef enum { ROPE_NEOX, ROPE_GPTJ } RopeStyle;
typedef enum { ACT_SILU, ACT_GELU } Activation;
typedef struct {
    /* existing dims... */
    RopeStyle rope;
    Activation act;
    int has_qkv_bias;        /* per-tensor detection overrides */
    int attn_softcap;        /* gemma2: final logits tanh softcap */
    float softcap_value;
    int swa_size;            /* gemma2 sliding window, 0=off */
    int tied_embeddings;
    int qk_norm_rms;         /* qwen3: rmsnorm on q/k heads pre-rope */
    float qk_norm_eps;
    int gemma_norm_offset;   /* gemma: (1 + rmsnorm(x)) */
} TTraits;
```

**Step 2: Key-name registry** (`src/arch_registry.c`): map architecture string from GGUF `general.architecture` → tensor-name templates + defaults. Families: `qwen2`✓, `llama` (covers mistral/tinyllama/smollm), `qwen3`, `gemma2`. Loader already reads `general.architecture` implicitly via key prefixes — make it explicit and stored in GGUFModel.

**Step 3: Kernel additions**
- `k_rope_gptj` (interleaved pairing) — you bisected this convention already; both variants exist conceptually.
- `k_qk_rmsnorm` (per-head rmsnorm over head_dim, qwen3).
- GELU activation variant in fused SwiGLU kernel → `tt_ffn_typed(gate,up,W,act,...)`.
- Gemma2 items: softcap on logits (`tanh(x/c)*c`), SWA cap in flash loop (`t >= pos-swa` skip), norm offset constant. These are ~20 lines total across two kernels.

**Step 4: Refactor `forward_layers`** to branch ONLY on traits (if/else per variation point — do NOT build an inheritance system; four families don't justify it):
- bias adds: `if (w->q_bias) …` (already conditional)
- rope style: select kernel pointer
- qwen3: insert q/k-norm between projection and rope
- gemma2: SWA bound in flash, softcap before argmax, norm offset in k_rmsnorm (pass `offset` param)

**Step 5:** Verify NO regression: `./scripts/verify.sh m61` green; France prompt unchanged.

**Step 6:** Commit: `M7: trait-driven forward pass; qwen2 behavior unchanged`.

---

## Task 4: Per-architecture bring-up (one model each, oracle-gated)

**Files:**
- Modify: `examples/chat_llm_gpu.c`, `examples/run_llm_gpu.c` (model path from argv/env)
- Create: `tests/gate_m7_arch.py`

**Step 1:** Generalize the parity gate: `gate_m7_arch.py --model PATH [--tokens-from llama-tokenize]` — tokenizes prompt with `llama-tokenize`, feeds ids to `build/dump_logits` (add model-path argv there), compares against fresh `oracle_logits` dump. Reuses threshold logic from `gate_m6_logit_parity.py` (extract shared helper or duplicate the 30 lines — duplication fine here).

**Step 2: Bring-up order (stop-and-fix each until parity):**
1. **TinyLlama f16** (llama family): expect immediate success after key-names — llama arch has NO qkv biases (trait says none), input OneWire embed, output head separate or tied. Gate: top-1 match, maxΔ ≤0.35.
2. **TinyLlama quant matrix** (Task 0's 10 quants): exercises Task 2 dispatch across all types through a REAL model. Every combo must pass the same gate.
3. **Qwen3-0.6B**: exercises qk-norm trait. Note qwen3 key prefix differences (`qwen3.` vs `qwen2.`).
4. **SmolLM2 f16 + quants**: llama-family again, different tokenizer nuances (check `tokenizer.ggml.model` == gpt2 ✓ our BPE).
5. **Gemma2-2b**: hardest — SentencePiece tokenizer (new decode path: tokens ARE raw sp pieces; space = "▁" prefix rule, byte fallback `<0xNN>` same), geglu?? (gemma2 uses gelu — actually GeGLU via `ffn_gate` with gelu), softcap, SWA, norm offset, tied embeddings. Budget a full session for gemma alone.

**Step 3:** Tokenizer SP-backend (only needed for gemma): in `tokenizer_bpe.c`, detect `tokenizer.ggml.model == "llama"` → decode passthrough of raw pieces with "▁"→space replacement + byte-fallback `<0xNN>`; encode via longest-piece match on ▁-prefixed vocab + scores ranking (SP is unigram — greedy score-max descent, NOT BPE merges; ~60 lines). Honest scope call: encode quality for SP is approximate without Viterbi; greedy-longest is acceptable for chat use, note limitation.

**Step 4:** Record every combo in `bench/results.md` grid: rows=architectures, cols=quants, cells=PASS/FAIL + tok/s.

**Step 5:** Commits per architecture: `M7: <family> bring-up — parity vs oracle (<model>, N quants)`.

---

## Task 5: The full verification grid + README truth-table

**Files:**
- Create: `tests/gate_m7_grid.py` (runs Task-4 gate across models.json × available quants)
- Modify: `README.md` (create proper README finally)

**Step 1:** Grid runner iterates manifest, emits markdown matrix, exit nonzero on any regression vs previous run (store last results in `tests/fixtures/grid_baseline.json`).

**Step 2:** Wire `verify.sh m7` = grid runner. Full sweep runtime budget: ~15 min (each model×quant loads in seconds).

**Step 3:** README: supported-archs × supported-quants truth table, per-combo parity status, quick-start (`./chat`), benchmark ladder.

**Step 4:** `git commit -m "M7: full grid green — X archs × Y quants verified"`.

---

## Standing rules

- Oracle parity per model is THE acceptance test — coherent-looking text is not evidence.
- No quant kernel lands without the CPU golden round-trip (Task 1 pattern).
- 4 GB VRAM discipline: refuse to load models >3 GB with clear error.
- Every new env var documented in README (TT_NO_GRAPH precedent).

## Explicit non-goals (Tier 2/3, separate plans)

i-quants, Phi/NeoX/J families, MoE routing, Mamba, multimodal, continuous batching, speculative decoding.
