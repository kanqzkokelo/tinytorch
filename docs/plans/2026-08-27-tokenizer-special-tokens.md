# gemma chat still 1-token even after `<bos>` fix — discovered followup bug

## Status
- Chat fix landed in `e4b6cae`: `fmt_gemma` no longer emits literal `<bos>` (was the
  original diagnosis — SP-mode tokenizer splits it into subword pieces and
  garbles the prompt prefix).
- ci_local GREEN, m61 7/7, all chat-template golden tests pass.
- **CHAT STILL 1-TOKENS**: gemma-4-E2B with the fix emits "▁The" (token 818) at
  pos=0, then EOS (token 106) at pos=1. The model genuinely thinks the turn
  is over after the first word.

## Root cause (discovered but not fixed here)

`src/tokenizer_bpe.c::bpe_encode` has **no special-token handling**. The gemma
vocabulary's special tokens (e.g. `<start_of_turn>` id 105, `<end_of_turn>`
id 106, `<bos>` id 2) are registered as `added_tokens` in the GGUF metadata
and the SP tokenizer is supposed to detect `<...>` literal spans in the input
text and emit them as single token ids. Our implementation does:

```c
// src/tokenizer_bpe.c, the bpe_encode path:
// 1. BPE-byte-fallback-encode the entire input text
// 2. (no step 2: no special-token table)
```

So the rendered prompt fragment `<start_of_turn>model\n` becomes 3-4
subword pieces (`▁start`, `_of`, `_turn`, `>`, `▁model`, etc.) instead of the
two single-token ids the model was trained to expect. The model reads the
garbled sequence, sees something that doesn't match its training distribution,
and the heuristic "we're past the start of a turn" fires — emitting `<end_of_turn>`
after the first plausible word.

## Fix (when desired)

1. Read the GGUF metadata `tokenizer.ggml.added_tokens` array (struct
   `llama_vocab::special_tokens` in llama.cpp; ours has no equivalent loader
   surface — needs to be added to `GGUFModel`).
2. Build a small hash table: `name → id` for the added tokens, populated
   at tokenizer load time from that metadata.
3. In `bpe_encode`, before doing the byte-BPE path, scan the input text for
   `<...>` literal spans that match a key in the table; emit the corresponding
   id; skip the BPE path for that span. This is the standard llama.cpp
   `llama_tokenize_internal` approach (see `oracle/llama.cpp/src/llama-vocab.cpp`
   `tokenize_add`).
4. Test: the test_chat_template.py passes (it tests the FORMAT, not the
   tokenization), so add a `tests/test_tokenizer_special_tokens.py` golden
   test that:
   - Loads a gemma-4 GGUF, encodes "What is <start_of_turn>user\nhi<end_of_turn>"
   - Asserts that the encoded ids contain the SINGLE `<start_of_turn>` id
     (not the BPE split)
   - Asserts the same for `<end_of_turn>`, `<bos>`, `<|begin_of_text|>`
5. Once `bpe_encode` honors special tokens, the chat should produce real
   multi-token gemma replies.

## Effort
~0.5-1 day: ~30 lines of hash table + 30 lines of scan-and-emit, plus the
golden test. The GGUF metadata structure is already in `loader_gguf.c` —
just need to surface it to `tokenizer_bpe.c` (probably via a new function
returning an array of `{name, id}` pairs).

## Status check
The user is asleep; the chat fix is in but does not by itself restore
multi-token replies. The next session's first action should be the
special-token handling above. After that, the chat will work properly
and the graph-capture work / M9.1 batched prefill can be re-attempted
without the chat-template issue being confounded.
