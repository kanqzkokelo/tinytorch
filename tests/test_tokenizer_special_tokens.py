#!/usr/bin/env python3
"""Tokenizer special-token handling test (M9 follow-up).

Loads a real GGUF (default: data/models/gemma-4-E2B-it-Q4_0.gguf) and
verifies that src/tokenizer_bpe.c's bpe_encode() emits the verbatim
single-token id for each "<...>" / "<|...|>" / "<|...>" / "<...|>"
span that exists in the vocab. Prior to the fix, these spans were
greedy-split into subword pieces, garbling chat prompts and causing
the gemma-4 chat to 1-token after a single word.

The same test passes for any model whose vocab contains the test
spans (qwen2 with `<|im_start|>`/`<|im_end|>`, llama-3 with
`<|begin_of_text|>`/`<|eot_id|>`, gemma-2 with `<start_of_turn>`/
`<end_of_turn>`, gemma-4 with `<|turn|>`/`<turn|>`/etc.).

The test is self-contained: it compiles a small driver linked against
src/tokenizer_bpe.c, src/loader_gguf.c, src/dequant_ref.c using
gcc -shared -fPIC, drives it via subprocess, and parses the output.

Run:  python3 tests/test_tokenizer_special_tokens.py
"""

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO, "data", "models", "gemma-4-E2B-it-Q4_0.gguf")
HDR_DIR = os.path.join(REPO, "include")
SRC_DIR = os.path.join(REPO, "src")

# gemma-4 actual vocab special tokens (verified from GGUF metadata):
#   id 2  <bos>         id 1   <eos>
#   id 105 <|turn>      id 106 <turn|>     (asymmetric open/close)
#   id 46  <|tool>      id 47  <tool|>
#   id 98  <|think|>
SPECIAL_TOKENS = [
    ("<bos>",       2),
    ("<eos>",       1),
    ("<|turn>",     105),
    ("<turn|>",     106),
    ("<|tool>",     46),
    ("<tool|>",     47),
    ("<|think|>",   98),
]

# Full chat-prompt fragment as the gemma-4 chat template would render it
# (using gemma-4's actual turn tokens — the local fmt_gemma still uses
# gemma-2 names; that's a chat-template bug out of scope here).
CHAT_FRAGMENT = "<|turn>user\nhi<turn|>\n<|turn>model\n"

DRIVER_C = r"""
#include "tokenizer_bpe.h"
#include "loader_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Protocol (matches gate_tokenizer.py for consistency):
 *   - input: NUL-delimited lines, one case per entry
 *   - each case: "ENCODE\t<text>" or "DECODE\t<id>"
 *   - output: one line per case, "CASE <idx> ENCODE n=N id1 id2 ..." or
 *             "CASE <idx> DECODE n=N <text>" */
static char *read_all_stdin(size_t *out_len) {
    size_t cap = 1 << 16, len = 0;
    char *buf = malloc(cap);
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, stdin)) > 0) {
        len += r;
        if (len == cap) { cap *= 2; buf = realloc(buf, cap); }
    }
    *out_len = len;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf < cases.nul\n", argv[0]); return 2; }
    GGUFModel *m = gguf_load(argv[1]);
    if (!m) return 1;
    BPETokenizer *tok = bpe_tokenizer_init(m);
    if (!tok) { gguf_free(m); return 1; }

    size_t len;
    char *buf = read_all_stdin(&len);
    char *p = buf, *end = buf + len;

    int idx = 0;
    while (p < end) {
        char *term = memchr(p, '\0', (size_t)(end - p));
        if (!term) term = end;
        char saved = *term;
        *term = '\0';
        char *tab = strchr(p, '\t');
        if (tab) {
            *tab = '\0';
            if (strcmp(p, "ENCODE") == 0) {
                const char *txt = tab + 1;
                int ids[8192];
                int n = bpe_encode(tok, txt, ids, 8192);
                printf("CASE %d ENCODE n=%d", idx, n);
                for (int i = 0; i < n; i++) printf(" %d", ids[i]);
                printf("\n");
            } else if (strcmp(p, "DECODE") == 0) {
                int id = atoi(tab + 1);
                int dlen = 0;
                const char *d = bpe_decode_token(tok, id, &dlen);
                printf("CASE %d DECODE n=%d %.*s\n", idx, dlen, dlen, d);
            } else {
                printf("CASE %d UNKNOWN %s\n", idx, p);
            }
        }
        *term = saved;
        p = term + 1;
        idx++;
    }
    free(buf);
    bpe_tokenizer_free(tok);
    gguf_free(m);
    return 0;
}
"""


def build_driver():
    tmp = tempfile.mkdtemp(prefix="tt_special_tok_")
    src = os.path.join(tmp, "driver.c")
    exe = os.path.join(tmp, "driver")
    with open(src, "w") as f:
        f.write(DRIVER_C)
    cmd = ["gcc", "-O2", "-DTT_IN_LIB",
           "-I", HDR_DIR, "-I", SRC_DIR,
           "-o", exe, src,
           os.path.join(SRC_DIR, "tokenizer_bpe.c"),
           os.path.join(SRC_DIR, "loader_gguf.c"),
           os.path.join(SRC_DIR, "dequant_ref.c"),
           "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write("driver build failed:\n" + r.stderr)
        raise RuntimeError("driver build failed")
    return exe


def run_cases(driver, model, cases):
    """cases: list of (op, value). op in {'ENCODE', 'DECODE'}.
    Returns dict[int] -> ('ENCODE', n, ids) | ('DECODE', n, text)."""
    # NUL-delimited so newlines in the text don't terminate the case.
    parts = []
    for op, txt, _id in cases:
        if op == "ENCODE":
            parts.append(b"ENCODE\t" + txt.encode("utf-8") + b"\x00")
        elif op == "DECODE":
            parts.append(b"DECODE\t%d\x00" % _id)
    payload = b"".join(parts)
    r = subprocess.run([driver, model], input=payload, capture_output=True, timeout=180)
    r_stdout = r.stdout.decode("utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("driver rc=%d: %s" % (r.returncode, r.stderr[:400]))
    out = {}
    for line in r_stdout.splitlines():
        if not line.startswith("CASE "):
            continue
        parts = line.split(maxsplit=3)
        idx = int(parts[1])
        op = parts[2]
        if op == "ENCODE":
            # "n=N id1 id2 ..."
            rest = parts[3].split()
            n = int(rest[0][2:])  # "n=N" -> N
            ids = [int(x) for x in rest[1:1 + n]]
            out[idx] = ("ENCODE", n, ids)
        elif op == "DECODE":
            rest = parts[3].split(maxsplit=1)
            n = int(rest[0][2:])
            text = rest[1] if len(rest) > 1 else ""
            out[idx] = ("DECODE", n, text)
    return out


FAILURES = []
def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"SKIP: model not found at {MODEL_PATH}")
        return 0

    print(f"== model: {os.path.basename(MODEL_PATH)} ==")
    driver = build_driver()

    # Build (op, payload, expected_id) tuples. For ENCODE we expect that
    # the special-token id appears AS A SINGLE TOKEN in the output (i.e.
    # the verbatim string was matched by the longest-match scan, not split
    # into subword pieces). For DECODE we expect the literal string back.
    cases = []
    # First N are ENCODEs, then N are DECODEs.
    enc_cases = list(SPECIAL_TOKENS) + [("<|turn>user\nhi<turn|>\n<|turn>model\n", None)]
    dec_cases = list(SPECIAL_TOKENS)
    for tok, _id in enc_cases[:-1]:
        cases.append(("ENCODE", tok, _id))
    cases.append(("ENCODE", enc_cases[-1][0], enc_cases[-1][1]))
    for tok, _id in dec_cases:
        cases.append(("DECODE", tok, _id))

    results = run_cases(driver, MODEL_PATH, cases)

    print("\n== 1. Single-token special-token encoding ==")
    for i, (tok, _id) in enumerate(enc_cases[:-1]):
        op, n, ids = results[i]
        check(f"ENCODE({tok!r}) contains id={_id}",
              _id in ids,
              f"got ids={ids[:8]}..." if len(ids) > 8 else f"got ids={ids}")
        # Also: should be a SINGLE token (only the leading BOS may precede it
        # for SP-mode models). At minimum, _id should appear once.
        check(f"ENCODE({tok!r}) id={_id} appears exactly once",
              ids.count(_id) == 1,
              f"count={ids.count(_id)}, ids={ids}")

    print("\n== 2. Chat-prompt fragment contains turn tokens as single ids ==")
    i = len(enc_cases) - 1  # the chat fragment is the last ENCODE
    op, n, ids = results[i]
    check(f"CHAT fragment contains <|turn|> id=105",
          105 in ids,
          f"ids={ids}")
    check(f"CHAT fragment contains <turn|> id=106",
          106 in ids,
          f"ids={ids}")
    # Verify the turn ids appear right after the BOS (typical gemma-4 layout)
    try:
        idx_turn_open = ids.index(105)
        idx_turn_close = ids.index(106)
        check("CHAT fragment <|turn|> (105) precedes <turn|> (106)",
              idx_turn_open < idx_turn_close,
              f"open@{idx_turn_open} close@{idx_turn_close}")
        check("CHAT fragment <|turn|> (105) appears >= 2 times (open + model prompt)",
              ids.count(105) >= 2,
              f"count={ids.count(105)}")
    except ValueError as e:
        FAILURES.append("CHAT fragment ordering check")
        print(f"  FAIL  ordering ({e})")

    print("\n== 3. Round-trip decode ==")
    for j, (tok, _id) in enumerate(dec_cases):
        # DECODE cases follow all the ENCODE cases in `cases` (index offset = len(enc_cases))
        k = len(enc_cases) + j
        op, n, text = results[k]
        check(f"DECODE(id={_id}) == {tok!r}",
              text == tok,
              f"got {text!r}")

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
