#!/usr/bin/env python3
"""Golden-string tests for src/chat_template.c (per-family chat formatting).

Builds a standalone shared lib from src/chat_template.c and asserts exact
formatted outputs transcribed from the official HF chat_template Jinja:

  - google/gemma-3-4b-it   (mirror unsloth/gemma-3-4b-it)  <start_of_turn>
  - Qwen/Qwen2.5-Instruct  ChatML <|im_start|>/<|im_end|>
  - Qwen/Qwen3-Instruct    + <think>-channel strip / empty-think flag
  - meta-llama/Llama-3.2-1B-Instruct (mirror unsloth/...)  header/eot style

Run:  python3 tests/test_chat_template.py
(no repo libs needed -- only cc; the repo tokenizer encode-check is skipped
because bpe_encode requires a full GGUFModel from loader_gguf).
"""

import ctypes
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "chat_template.c")
HDR_DIR = os.path.join(REPO, "include")

# ---- enum values must match chat_template.h --------------------------------
TT_CHAT_QWEN2, TT_CHAT_QWEN3, TT_CHAT_GEMMA, TT_CHAT_GEMMA4, TT_CHAT_LLAMA3 = range(5)


class TTMsg(ctypes.Structure):
    _fields_ = [("role", ctypes.c_char_p), ("content", ctypes.c_char_p)]


class TTOpts(ctypes.Structure):
    _fields_ = [
        ("add_generation_prompt", ctypes.c_int),
        ("keep_think", ctypes.c_int),
        ("add_empty_think", ctypes.c_int),
        ("add_bos_text", ctypes.c_int),
        ("date_string", ctypes.c_char_p),
    ]


def build_lib():
    tmp = tempfile.mkdtemp(prefix="tt_chat_tpl_")
    so = os.path.join(tmp, "libchat_template.so")
    subprocess.run(
        ["cc", "-std=c11", "-Wall", "-Wextra", "-O2", "-I", HDR_DIR,
         "-shared", "-fPIC", "-o", so, SRC],
        check=True,
    )
    lib = ctypes.CDLL(so)
    lib.tt_chat_format_ex.restype = ctypes.c_int
    lib.tt_chat_format_ex.argtypes = [
        ctypes.c_int, ctypes.POINTER(TTMsg), ctypes.c_int,
        ctypes.POINTER(TTOpts), ctypes.c_char_p, ctypes.c_size_t]
    lib.tt_chat_format.restype = ctypes.c_int
    lib.tt_chat_format.argtypes = [
        ctypes.c_int, ctypes.POINTER(TTMsg), ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.tt_chat_family_from_arch.restype = ctypes.c_int
    lib.tt_chat_family_from_arch.argtypes = [ctypes.c_char_p]
    lib.tt_chat_stop_string.restype = ctypes.c_char_p
    lib.tt_chat_stop_string.argtypes = [ctypes.c_int]
    return lib


LIB = build_lib()


def fmt(fam, conv, opts=None):
    """conv: list of (role, content). Returns (rendered_string, retcode)."""
    n = len(conv)
    arr = (TTMsg * n)(*[TTMsg(r.encode(), c.encode() if c else None)
                        for r, c in conv]) if n else None
    buf = ctypes.create_string_buffer(65536)
    if opts is None:
        rc = LIB.tt_chat_format(fam, arr, n, buf, len(buf))
    else:
        rc = LIB.tt_chat_format_ex(fam, arr, n, ctypes.byref(opts),
                                   buf, len(buf))
    return buf.value.decode("utf-8"), rc


def opts(**kw):
    d = dict(add_generation_prompt=1, keep_think=0, add_empty_think=0,
             add_bos_text=1, date_string=None)
    d.update(kw)
    if isinstance(d["date_string"], str):
        d["date_string"] = d["date_string"].encode()
    return TTOpts(**d)


FAILURES = []


def check(name, got, want, rc=0):
    if got != want or rc < 0:
        FAILURES.append(name)
        print(f"FAIL {name} (rc={rc})")
        print(f"  want: {want!r}")
        print(f"  got : {got!r}")
    else:
        print(f"ok   {name}")


SYS = "You are a helpful assistant."
U1, A1 = "What is 2+2?", "It is 4."
U2, A2 = "And times three?", "Twelve."
U3, A3 = "Thanks!", "Anytime."
UNI = "héllo 世界 🚀 — naïve café"

# ============================ QWEN2 (ChatML) ================================
# From Qwen/Qwen2.5-Instruct: '<|im_start|>{role}\n{content}<|im_end|>\n'
# per message, optional leading system block, trailing generation prompt
# '<|im_start|>assistant\n'.

check("qwen2 single+system golden",
      fmt(TT_CHAT_QWEN2, [("system", SYS), ("user", U1)])[0],
      f"<|im_start|>system\n{SYS}<|im_end|>\n"
      f"<|im_start|>user\n{U1}<|im_end|>\n"
      f"<|im_start|>assistant\n")

check("qwen2 multiturn 3 exchanges golden",
      fmt(TT_CHAT_QWEN2,
          [("user", U1), ("assistant", A1),
           ("user", U2), ("assistant", A2),
           ("user", U3)])[0],
      f"<|im_start|>user\n{U1}<|im_end|>\n"
      f"<|im_start|>assistant\n{A1}<|im_end|>\n"
      f"<|im_start|>user\n{U2}<|im_end|>\n"
      f"<|im_start|>assistant\n{A2}<|im_end|>\n"
      f"<|im_start|>user\n{U3}<|im_end|>\n"
      f"<|im_start|>assistant\n")

check("qwen2 no-system golden",
      fmt(TT_CHAT_QWEN2, [("user", U1)])[0],
      f"<|im_start|>user\n{U1}<|im_end|>\n<|im_start|>assistant\n")

check("qwen2 empty msg golden",
      fmt(TT_CHAT_QWEN2, [("user", "")])[0],
      "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n")

check("qwen2 unicode golden",
      fmt(TT_CHAT_QWEN2, [("user", UNI)])[0],
      f"<|im_start|>user\n{UNI}<|im_end|>\n<|im_start|>assistant\n")

check("qwen2 no-gen-prompt golden",
      fmt(TT_CHAT_QWEN2, [("user", U1)],
          opts(add_generation_prompt=0))[0],
      f"<|im_start|>user\n{U1}<|im_end|>\n")

# ============================ QWEN3 (ChatML + think) ========================
# Same shape as qwen2; <think>..</think> stripped from assistant history
# unless keep_think; add_empty_think mirrors enable_thinking=false which
# pre-fills an empty think block after the generation prompt.

THINKY = "<think>let me count 2,3,4...</think>" + A1

check("qwen3 strips think from history golden",
      fmt(TT_CHAT_QWEN3,
          [("user", U1), ("assistant", THINKY), ("user", U2)])[0],
      f"<|im_start|>user\n{U1}<|im_end|>\n"
      f"<|im_start|>assistant\n{A1}<|im_end|>\n"
      f"<|im_start|>user\n{U2}<|im_end|>\n"
      f"<|im_start|>assistant\n")

check("qwen3 keep_think preserves blocks golden",
      fmt(TT_CHAT_QWEN3,
          [("user", U1), ("assistant", THINKY), ("user", U2)],
          opts(keep_think=1))[0],
      f"<|im_start|>user\n{U1}<|im_end|>\n"
      f"<|im_start|>assistant\n{THINKY}<|im_end|>\n"
      f"<|im_start|>user\n{U2}<|im_end|>\n"
      f"<|im_start|>assistant\n")

check("qwen3 unterminated think dropped golden",
      fmt(TT_CHAT_QWEN3,
          [("assistant", "<think>dangling..." )])[0],
      "<|im_start|>assistant\n<|im_end|>\n<|im_start|>assistant\n")

check("qwen3 add_empty_think golden",
      fmt(TT_CHAT_QWEN3, [("user", U1)], opts(add_empty_think=1))[0],
      f"<|im_start|>user\n{U1}<|im_end|>\n"
      f"<|im_start|>assistant\n<think>\n\n</think>\n\n")

# ============================ GEMMA / GEMMA4 ================================
# From google/gemma-3-4b-it: literal <bos>; leading system folded into the
# first user turn as '{system}\n\n'; roles user/model; content |trim'd;
# '<end_of_turn>\n' terminator; generation prompt '<start_of_turn>model\n'.

g_conv_sys = fmt(TT_CHAT_GEMMA, [("system", SYS), ("user", U1)])
check("gemma single+system golden",
      g_conv_sys[0],
      f"<bos><start_of_turn>user\n{SYS}\n\n{U1}<end_of_turn>\n"
      f"<start_of_turn>model\n")

check("gemma4 identical to gemma golden",
      fmt(TT_CHAT_GEMMA4, [("system", SYS), ("user", U1)])[0],
      g_conv_sys[0])

check("gemma multiturn 3 exchanges golden",
      fmt(TT_CHAT_GEMMA,
          [("user", U1), ("assistant", A1),
           ("user", U2), ("assistant", A2),
           ("user", U3)])[0],
      f"<bos><start_of_turn>user\n{U1}<end_of_turn>\n"
      f"<start_of_turn>model\n{A1}<end_of_turn>\n"
      f"<start_of_turn>user\n{U2}<end_of_turn>\n"
      f"<start_of_turn>model\n{A2}<end_of_turn>\n"
      f"<start_of_turn>user\n{U3}<end_of_turn>\n"
      f"<start_of_turn>model\n")

check("gemma no-system golden",
      fmt(TT_CHAT_GEMMA, [("user", U1)])[0],
      f"<bos><start_of_turn>user\n{U1}<end_of_turn>\n"
      f"<start_of_turn>model\n")

check("gemma trims whitespace golden",
      fmt(TT_CHAT_GEMMA, [("user", "  padded  \n")])[0],
      "<bos><start_of_turn>user\npadded<end_of_turn>\n"
      "<start_of_turn>model\n")

check("gemma empty msg golden",
      fmt(TT_CHAT_GEMMA, [("user", "")])[0],
      "<bos><start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n")

check("gemma unicode golden",
      fmt(TT_CHAT_GEMMA, [("user", UNI)])[0],
      f"<bos><start_of_turn>user\n{UNI}<end_of_turn>\n"
      f"<start_of_turn>model\n")

check("gemma bos off (engine injects BOS id) golden",
      fmt(TT_CHAT_GEMMA, [("user", U1)], opts(add_bos_text=0))[0],
      f"<start_of_turn>user\n{U1}<end_of_turn>\n<start_of_turn>model\n")

# ============================ LLAMA3 ========================================
# From meta-llama/Llama-3.2-1B-Instruct: <|begin_of_text|>; system block
# '<|start_header_id|>system<|end_header_id|>\n\n{dates}{sys}<|eot_id|>';
# turns '{hdr}{role}{eoh}\n\n{content|trim}<|eot_id|>'; generation prompt
# assistant header. Date preamble only when date_string set (see DEVIATION
# note in chat_template.c).

check("llama3 single+system+date golden",
      fmt(TT_CHAT_LLAMA3, [("system", SYS), ("user", U1)],
          opts(date_string="26 Jul 2024"))[0],
      f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
      f"Cutting Knowledge Date: December 2023\n"
      f"Today Date: 26 Jul 2024\n\n"
      f"{SYS}<|eot_id|>"
      f"<|start_header_id|>user<|end_header_id|>\n\n{U1}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n")

check("llama3 no-system no-dates golden",
      fmt(TT_CHAT_LLAMA3, [("user", U1)])[0],
      f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
      f"{U1}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n")

check("llama3 multiturn 3 exchanges golden",
      fmt(TT_CHAT_LLAMA3,
          [("user", U1), ("assistant", A1),
           ("user", U2), ("assistant", A2),
           ("user", U3)])[0],
      f"<|begin_of_text|>"
      f"<|start_header_id|>user<|end_header_id|>\n\n{U1}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n{A1}<|eot_id|>"
      f"<|start_header_id|>user<|end_header_id|>\n\n{U2}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n{A2}<|eot_id|>"
      f"<|start_header_id|>user<|end_header_id|>\n\n{U3}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n")

check("llama3 empty msg golden",
      fmt(TT_CHAT_LLAMA3, [("user", "")])[0],
      "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
      "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

check("llama3 unicode golden",
      fmt(TT_CHAT_LLAMA3, [("user", UNI)])[0],
      f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
      f"{UNI}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n")

# ============================ misc API ======================================

for arch, fam in [("qwen2", TT_CHAT_QWEN2), ("qwen3", TT_CHAT_QWEN3),
                  ("gemma", TT_CHAT_GEMMA), ("gemma2", TT_CHAT_GEMMA),
                  ("gemma4", TT_CHAT_GEMMA4), ("llama", TT_CHAT_LLAMA3)]:
    got = LIB.tt_chat_family_from_arch(arch.encode())
    if got != fam:
        FAILURES.append(f"arch:{arch}")
        print(f"FAIL arch map {arch}: want {fam} got {got}")
    else:
        print(f"ok   arch map {arch}")

if LIB.tt_chat_family_from_arch(b"falcon") != -1:
    FAILURES.append("arch unknown")
    print("FAIL unknown arch should be -1")
else:
    print("ok   unknown arch -> -1")

stops = {TT_CHAT_QWEN2: "<|im_end|>", TT_CHAT_QWEN3: "<|im_end|>",
         TT_CHAT_GEMMA: "<end_of_turn>", TT_CHAT_GEMMA4: "<end_of_turn>",
         TT_CHAT_LLAMA3: "<|eot_id|>"}
for fam, s in stops.items():
    got = LIB.tt_chat_stop_string(fam)
    if not got or got.decode() != s:
        FAILURES.append(f"stop:{fam}")
        print(f"FAIL stop string fam={fam}: want {s} got {got}")
    else:
        print(f"ok   stop string fam={fam}")

# snprintf-style sizing: exact-size buffer succeeds, size-1 truncates.
conv = [("system", SYS), ("user", U1)]
full, rc = fmt(TT_CHAT_QWEN2, conv)
need = LIB.tt_chat_format(TT_CHAT_QWEN2, None, 0, None, 0)  # NULL probe
arr = (TTMsg * 2)(TTMsg(b"system", SYS.encode()), TTMsg(b"user", U1.encode()))
exact = ctypes.create_string_buffer(full.encode() + b"\x00")
rc_exact = LIB.tt_chat_format(TT_CHAT_QWEN2, arr, 2, exact,
                              len(full) + 1)
tiny = ctypes.create_string_buffer(len(full))  # one byte short
need_len = LIB.tt_chat_format(TT_CHAT_QWEN2, arr, 2, tiny, len(tiny))
if rc_exact == len(full) and exact.value.decode() == full \
        and need_len == len(full) > len(tiny) - 1:
    print("ok   snprintf sizing/truncation semantics")
else:
    FAILURES.append("sizing")
    print(f"FAIL sizing: rc_exact={rc_exact} need_len={need_len} "
          f"len(full)={len(full)} roundtrip={exact.value.decode() == full}")

if LIB.tt_chat_format(99, None, 0, ctypes.create_string_buffer(8), 8) == -1:
    print("ok   bad family -> -1")
else:
    FAILURES.append("bad family")
    print("FAIL bad family should return -1")

# ---- history helper --------------------------------------------------------
lib_hist_init = LIB.tt_chat_history_init
lib_hist_push = LIB.tt_chat_history_push
lib_hist_fmt = LIB.tt_chat_history_format
HIST_SIZE = 64 * (16 + 4096) + 4  # struct: int n + role/content arrays


class Hist(ctypes.Union):
    _fields_ = [("_bytes", ctypes.c_char * HIST_SIZE)]


class TTHistory(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int),
                ("role", (ctypes.c_char * 16) * 64),
                ("content", (ctypes.c_char * 4096) * 64)]


lib_hist_init.argtypes = [ctypes.POINTER(TTHistory)]
lib_hist_push.argtypes = [ctypes.POINTER(TTHistory), ctypes.c_char_p,
                          ctypes.c_char_p]
lib_hist_push.restype = ctypes.c_int
lib_hist_fmt.argtypes = [ctypes.POINTER(TTHistory), ctypes.c_int,
                         ctypes.POINTER(TTOpts), ctypes.c_char_p,
                         ctypes.c_size_t]
lib_hist_fmt.restype = ctypes.c_int

h = TTHistory()
lib_hist_init(ctypes.byref(h))
lib_hist_push(ctypes.byref(h), b"system", SYS.encode())
lib_hist_push(ctypes.byref(h), b"user", U1.encode())
buf = ctypes.create_string_buffer(65536)
o = opts()
rc = lib_hist_fmt(ctypes.byref(h), TT_CHAT_QWEN2, ctypes.byref(o),
                  buf, len(buf))
check("history accumulation matches direct format",
      buf.value.decode(),
      fmt(TT_CHAT_QWEN2, [("system", SYS), ("user", U1)])[0])

# ---- optional tokenizer encode-check ---------------------------------------
# Skipped by design: bpe_encode needs a full GGUFModel (loader_gguf + weights).
# Wire up later inside examples/chat_llm_gpu.c where the model is loaded:
#   formatted prompt -> bpe_encode(tok, ...) must return > 0 with no
#   byte-fallback explosion (compare n_tokens vs strlen heuristic).

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all chat-template golden tests passed")
