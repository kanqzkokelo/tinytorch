#!/usr/bin/env python3
"""Tokenizer correctness gate: our dual-mode tokenizer vs llama.cpp oracle.

For every local GGUF model in data/testmodels/ <= 1.5 GB:
  1. Encode 10 probe strings through OUR engine tokenizer (bpe_encode, CPU).
  2. Encode the same strings through the llama-tokenize oracle binary
     (vocab-only load, --no-escape so bytes are verbatim; BOS follows each
     model's own metadata convention on both sides).
  3. Require exact id-sequence equality per string.
  4. Decode roundtrip check on our side: ids -> text -> ids identity
     (leading BOS excluded from both legs).

PASS bar: 100% id equality per model. Tokenizers are exact-match domain.

STATUS (M6.1): pre-tokenization gaps CLOSED. src/tokenizer_bpe.c now ports the
llama.cpp unicode.cpp custom splitters verbatim per tokenizer.ggml.pre family
(qwen2 / llama-bpe+ignore_merges / smollm two-pass / GPT-2 default), with
\p{L}/\p{N}/\s classification from the oracle's own unicode_ranges_flags table
(src/tokenizer_uni_table.inc). \s+(?!\S) trailing-space attachment replicated.
All gate models are tokenizer.ggml.model=="gpt2" (BPE) — smollm2 included —
so SP/ugm unigram Viterbi remains out of scope; sp_mode still uses greedy
longest-piece matching and is NOT exercised by this gate.
BOS handling mirrors llama-vocab.cpp for both modes.
"""

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "data", "testmodels")
ORACLE = os.path.join(REPO, "oracle", "llama.cpp", "build", "bin", "llama-tokenize")
MAX_MODEL_BYTES = int(1.5e9)

DRIVER_C = r"""
#include "tokenizer_bpe.h"
#include "loader_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf < strings.nul\n", argv[0]); return 2; }
    GGUFModel *m = gguf_load(argv[1]);
    if (!m) return 1;
    BPETokenizer *tok = bpe_tokenizer_init(m);
    if (!tok) { gguf_free(m); return 1; }

    size_t len;
    char *buf = read_all_stdin(&len);
    int idx = 0;
    for (size_t p = 0; p < len; ) {
        const char *s = buf + p;
        size_t sl = strlen(s);
        p += sl + 1;

        int ids[8192];
        int n = bpe_encode(tok, s, ids, 8192);

        /* decode roundtrip: drop leading BOS, ids -> text -> ids */
        int start = (n > 0 && ids[0] == tok->bos_id) ? 1 : 0;
        size_t cap = 64;
        for (int i = start; i < n; i++) cap += (size_t)tok->token_lens[i] + 4;
        char *txt = malloc(cap);
        size_t tl = 0;
        for (int i = start; i < n; i++) {
            int pl = 0;
            const char *piece = bpe_decode_token(tok, ids[i], &pl);
            memcpy(txt + tl, piece, (size_t)pl);
            tl += (size_t)pl;
        }
        txt[tl] = '\0';
        int ids2[8192];
        int n2 = bpe_encode(tok, txt, ids2, 8192);
        int start2 = (n2 > 0 && ids2[0] == tok->bos_id) ? 1 : 0;
        int rt_ok = (n - start) == (n2 - start2);
        if (rt_ok)
            for (int i = 0; start + i < n; i++)
                if (ids[start + i] != ids2[start2 + i]) { rt_ok = 0; break; }

        printf("TOKCASE %d\n", idx++);
        printf("TOKIDS");
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");
        printf("TOKRT %d\n", rt_ok);
        fflush(stdout);
        free(txt);
    }
    bpe_tokenizer_free(tok);
    gguf_free(m);
    return 0;
}
"""

# NUL-delimited at runtime; order matters and is reported per case.
PROBE_STRINGS = [
    ("ascii",          b"Hello, world! This is a tokenizer test."),
    ("unicode-accents", "héllo wörld — ünïcode ñ".encode("utf-8")),
    ("emoji-cjk",      "emoji 👍🚀 test 中文 日本語 한국".encode("utf-8")),
    ("multi-space",    b"a    b     c    d"),
    ("leading-space",  b" leading space matters"),
    ("code-snippet",   b"def foo(x):\n    return x*2 + [1, 2]"),
    ("special-text",   b"<start_of_turn>"),
    ("empty",          b""),
    ("single-char",    b"x"),
    ("very-long-word", b"supercalifragilisticexpialidociousantidisestablishmentarianism"),
]


def discover_models():
    models = []
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.endswith(".gguf"):
            continue
        path = os.path.join(MODEL_DIR, name)
        if os.path.getsize(path) > MAX_MODEL_BYTES:
            print(f"SKIP {name}: > 1.5GB")
            continue
        models.append((name, path))
    return models


def build_driver(tmpdir):
    src = os.path.join(tmpdir, "gate_tok_driver.c")
    exe = os.path.join(tmpdir, "gate_tok_driver")
    with open(src, "w") as f:
        f.write(DRIVER_C)
    cmd = ["gcc", "-O2", "-DTT_IN_LIB", "-I", os.path.join(REPO, "include"),
           "-o", exe, src,
           os.path.join(REPO, "src", "tokenizer_bpe.c"),
           os.path.join(REPO, "src", "loader_gguf.c"),
           os.path.join(REPO, "src", "dequant_ref.c"),
           "-lm"]
    subprocess.run(cmd, check=True, capture_output=True)
    return exe


def run_ours(driver, model_path):
    payload = b"".join(s + b"\x00" for _, s in PROBE_STRINGS)
    proc = subprocess.run([driver, model_path], input=payload,
                          capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"driver rc={proc.returncode}: {proc.stderr.decode()[:400]}")
    ids_by_case, rt_by_case = {}, {}
    case = None
    for line in proc.stdout.decode(errors="replace").splitlines():
        if line.startswith("TOKCASE "):
            case = int(line.split()[1])
        elif line.startswith("TOKIDS"):
            ids_by_case[case] = [int(x) for x in line.split()[1:]]
        elif line.startswith("TOKRT "):
            rt_by_case[case] = line.split()[1] == "1"
    return ids_by_case, rt_by_case


def run_oracle(model_path, text_bytes, tmpdir):
    f = tempfile.NamedTemporaryFile(dir=tmpdir, delete=False)
    f.write(text_bytes)
    f.close()
    try:
        proc = subprocess.run(
            [ORACLE, "-m", model_path, "-f", f.name, "--ids", "--no-escape"],
            capture_output=True, timeout=300)
    finally:
        os.unlink(f.name)
    out = proc.stdout.decode().strip()
    if not out.startswith("["):
        raise RuntimeError(f"oracle bad output: {proc.stderr.decode()[:400]}")
    inner = out[1:out.rindex("]")].strip()
    return [] if not inner else [int(x) for x in inner.split(",")]


def main():
    if not os.access(ORACLE, os.X_OK):
        print(f"FAIL: oracle missing: {ORACLE}")
        return 1
    models = discover_models()
    if not models:
        print("FAIL: no models found")
        return 1

    with tempfile.TemporaryDirectory(prefix="gate_tok_") as tmpdir:
        driver = build_driver(tmpdir)
        total_models = total_cases = failed_models = 0

        for name, path in models:
            total_models += 1
            print(f"\n=== {name} ===")
            try:
                ours, rts = run_ours(driver, path)
                oracle_ids = {
                    i: run_oracle(path, s, tmpdir) for i, (_, s) in enumerate(PROBE_STRINGS)
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                failed_models += 1
                continue

            fails = 0
            for i, (label, _) in enumerate(PROBE_STRINGS):
                total_cases += 1
                o, m = oracle_ids[i], ours.get(i)
                rt = rts.get(i, False)
                if m != o or not rt:
                    fails += 1
                    print(f"  FAIL [{label}]")
                    print(f"    oracle: {o}")
                    print(f"    ours:   {m}")
                    if not rt:
                        print(f"    decode roundtrip: FAIL")
            if fails:
                failed_models += 1
                print(f"  RESULT: {len(PROBE_STRINGS) - fails}/{len(PROBE_STRINGS)} cases match")
            else:
                print(f"  RESULT: {len(PROBE_STRINGS)}/{len(PROBE_STRINGS)} cases match, roundtrip OK")

    print(f"\n{'=' * 40}")
    print(f"models={total_models} passed={total_models - failed_models} "
          f"failed={failed_models} cases_total={total_cases}")
    if failed_models:
        print("GATE: FAIL (100% exact-match bar)")
        return 1
    print("GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
