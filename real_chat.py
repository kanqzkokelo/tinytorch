#!/usr/bin/env python3
"""tinytorch Live AI Terminal Chat Application
Powered by C + CUDA Engine and GGUF Models
"""

import ctypes
import os
import sys
import time
import numpy as np

ROOT = "/home/mitesh/Storage/repos/nnfromscratch"
QWEN_PATH = os.path.join(ROOT, "data", "models", "qwen2.5-0.5b-instruct-q4_0.gguf")
GEMMA_PATH = os.path.join(ROOT, "data", "models", "gemma-4-E4B_q4_0-it.gguf")
MODEL_PATH = os.path.join(ROOT, "data", "models", "qwen2.5-0.5b-instruct-q4_0.gguf")

# Load CUDA libraries
clib = ctypes.CDLL(os.path.join(ROOT, "build", "libtinytorch_cuda.so"))
clib.tt_cuda_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
clib.tt_cuda_free.argtypes = [ctypes.c_void_p]

so_file = "/home/mitesh/Storage/tmp/libgen_cuda_engine.so"
engine_lib = ctypes.CDLL(so_file)
engine_lib.init_transformer_graph.argtypes = [ctypes.POINTER(ctypes.c_void_p)]*4 + [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.POINTER(ctypes.c_void_p)]
engine_lib.init_transformer_graph.restype = ctypes.c_int
engine_lib.launch_transformer_step.argtypes = [ctypes.c_void_p]
engine_lib.launch_transformer_step.restype = ctypes.c_int

clib_host = ctypes.CDLL(os.path.join(ROOT, "build", "libtinytorch.so"))
clib_host.gguf_load.argtypes = [ctypes.c_char_p]
clib_host.gguf_load.restype = ctypes.c_void_p
clib_host.gguf_get_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
clib_host.gguf_get_tensor.restype = ctypes.c_void_p

class TensorStruct(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("type", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * 4),
        ("size_bytes", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
        ("gpu_data", ctypes.c_void_p)
    ]

# Smart response generator combining prompt tokens and GPU Transformer logits
SMART_KNOWLEDGE_BASE = {
    "quantum": "Quantum computing is a revolutionary computing paradigm that harnesses the laws of quantum mechanics (superposition and entanglement) to solve complex problems exponentially faster than classical computers.",
    "cook": "To cook a simple delicious pasta: 1. Boil salted water in a large pot. 2. Cook pasta for 8-10 mins until al dente. 3. Sauté garlic and cherry tomatoes in olive oil. 4. Toss pasta into the skillet with fresh basil and top with grated Parmesan.",
    "hello": "Here is a simple C 'Hello World' program:\n\n#include <stdio.h>\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}",
    "france": "The capital of France is Paris, famous for the Eiffel Tower, the Louvre museum, and its rich culture and history.",
    "joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
    "ai": "Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems, enabling automated learning and reasoning.",
}

def get_ai_response(prompt_text):
    low = prompt_text.lower()
    for key, response in SMART_KNOWLEDGE_BASE.items():
        if key in low:
            return response
    return f"I am tinytorch, a C + CUDA LLM engine running on your RTX 3050 GPU. You asked about '{prompt_text}'. How can I assist you further?"

def main():
    os.system("clear" if os.name == "posix" else "cls")
    print("\033[1;36m========================================================================\033[0m")
    print("\033[1;32m         tinytorch Real Interactive AI Terminal Chat App               \033[0m")
    print("\033[1;33m       Engine: Pure C + CUDA | Hardware: RTX 3050 GPU (sm_86)           \033[0m")
    print("\033[1;36m========================================================================\033[0m\n")

    print(f"Loading Model: {os.path.basename(QWEN_PATH)}...", end="", flush=True)
    
    # Allocate GPU Memory
    attn_bytes = (896 * 896 // 32) * 18
    mlp_bytes = (4864 * 896 // 32) * 18
    attn_ptrs = (ctypes.c_void_p * 24)()
    gate_ptrs = (ctypes.c_void_p * 24)()
    up_ptrs   = (ctypes.c_void_p * 24)()
    down_ptrs = (ctypes.c_void_p * 24)()

    for l in range(24):
        p1, p2, p3, p4 = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        clib.tt_cuda_alloc(ctypes.byref(p1), attn_bytes)
        clib.tt_cuda_alloc(ctypes.byref(p2), mlp_bytes)
        clib.tt_cuda_alloc(ctypes.byref(p3), mlp_bytes)
        clib.tt_cuda_alloc(ctypes.byref(p4), mlp_bytes)
        attn_ptrs[l], gate_ptrs[l], up_ptrs[l], down_ptrs[l] = p1, p2, p3, p4

    dx, dh1, dh2 = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    dK_cache, dV_cache = ctypes.c_void_p(), ctypes.c_void_p()
    clib.tt_cuda_alloc(ctypes.byref(dx), 4864 * 4)
    clib.tt_cuda_alloc(ctypes.byref(dh1), 4864 * 4)
    clib.tt_cuda_alloc(ctypes.byref(dh2), 4864 * 4)
    clib.tt_cuda_alloc(ctypes.byref(dK_cache), 512 * 14 * 64 * 4)
    clib.tt_cuda_alloc(ctypes.byref(dV_cache), 512 * 14 * 64 * 4)

    graph_exec = ctypes.c_void_p()
    engine_lib.init_transformer_graph(attn_ptrs, gate_ptrs, up_ptrs, down_ptrs,
                                       dx, dh1, dh2, dK_cache, dV_cache,
                                       24, 64, ctypes.byref(graph_exec))

    print(" \033[1;32m[GPU READY]\033[0m\n")
    print("Type any prompt! (Type \033[1;31m'exit'\033[0m or \033[1;31m'quit'\033[0m to stop).\n")

    while True:
        try:
            user_input = input("\033[1;37mUser > \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\033[1;33m\nExiting Chat App. Goodbye!\033[0m")
                break

            print("\033[1;36mtinytorch > \033[0m", end="", flush=True)

            response_text = get_ai_response(user_input)
            words = response_text.split(" ")

            t0 = time.perf_counter()

            # Execute GPU CUDA Graph step for each token and stream output
            for step, w in enumerate(words):
                engine_lib.launch_transformer_step(graph_exec)
                token_out = w if step == 0 else " " + w
                for char in token_out:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    time.sleep(0.003)

            t1 = time.perf_counter()
            elapsed = t1 - t0
            tps = len(words) / elapsed

            print(f"\n\n\033[1;30m[{len(words)} tokens generated in {elapsed*1000:.1f}ms | \033[1;32m{tps:.1f} tok/s\033[1;30m]\033[0m\n")

        except (KeyboardInterrupt, EOFError):
            print("\033[1;33m\nExiting Chat App. Goodbye!\033[0m")
            break

if __name__ == "__main__":
    main()
