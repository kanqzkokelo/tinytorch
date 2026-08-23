#!/usr/bin/env python3
"""NumPy golden reference for Qwen2-family decode (PRD_M6 M6.1 Step 1).

Pure-python GGUF parse + exact GGML q4_0 dequant + full transformer forward.
Usage: ref_qwen2_numpy.py [--model PATH] [--tokens ID,ID,...] [--top5]
"""
import struct, sys
import numpy as np

MODEL = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf"

def load_gguf(path):
    f = open(path, "rb")
    magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    assert magic == 0x46554747, hex(magic)

    def r_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_val(t):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8:
            r_str(); return
        if t == 9:
            (it,) = struct.unpack("<I", f.read(4))
            (al,) = struct.unpack("<Q", f.read(8))
            for _ in range(al): skip_val(it)
            return
        f.read(sizes[t])

    kv = {}
    for _ in range(n_kv):
        key = r_str()
        (t,) = struct.unpack("<I", f.read(4))
        pos = f.tell()
        if key.endswith(("embedding_length","feed_forward_length","block_count",
                         "attention.head_count","attention.head_count_kv")):
            v = struct.unpack("<i", f.read(4))[0]
            kv[key] = v
        elif key.endswith("rms_epsilon"):
            kv[key] = struct.unpack("<f", f.read(4))[0]
        elif key.endswith(("rope.freq_base","rope_freq_base")):
            kv[key] = struct.unpack("<f", f.read(4))[0]
        f.seek(pos); skip_val(t)

    tensors = {}
    for _ in range(n_tensors):
        name = r_str()
        (nd) = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q"*nd, f.read(8*nd))
        (dtype,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        tensors[name] = (dims, dtype, off)

    base = (f.tell() + 31) & ~31
    mm = open(path, "rb").read()          # models are < 2GB here; simple
    return kv, tensors, mm, base

def deq_f16(b):
    return struct.unpack("<e", b)[0]

def dequant_q4_0(buf, numel):
    nb = numel // 32
    arr = np.frombuffer(buf[:nb*18], dtype=np.uint8).reshape(nb, 18)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    qs = arr[:, 2:].astype(np.int32)
    lo = (qs & 0x0F) - 8
    hi = (qs >> 4) - 8
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = lo * d
    out[:, 16:] = hi * d
    return out.reshape(-1)[:numel]

def dequant_q8_0(buf, numel):
    nb = numel // 32
    arr = np.frombuffer(buf[:nb*34], dtype=np.uint8).reshape(nb, 34)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    qs = arr[:, 2:].copy().view(np.int8).astype(np.float32)
    return (qs * d).reshape(-1)[:numel]

def get_w(kv, tensors, mm, base, name):
    dims, dtype, off = tensors[name]
    numel = int(np.prod(dims))
    if dtype == 0:
        raw = mm[base+off : base+off+numel*4]
        return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(dims[::-1])
    if dtype == 1:
        raw = mm[base+off : base+off+numel*2]
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(dims[::-1])
    if dtype == 2:
        raw = mm[base+off : base+off+(numel//32)*18]
        return dequant_q4_0(raw, numel).reshape(dims[::-1])
    if dtype == 8:
        raw = mm[base+off : base+off+(numel//32)*34]
        return dequant_q8_0(raw, numel).reshape(dims[::-1])
    raise ValueError(f"dtype {dtype}")

def rmsnorm(x, g, eps):
    return x / np.sqrt(np.mean(x*x) + eps) * g

def rope(x, pos, head_dim, base):
    # x: [heads, head_dim]; rotate halves
    half = head_dim // 2
    out = x.copy()
    for h in range(x.shape[0]):
        for i in range(half):
            freq = base ** (-2.0 * i / head_dim)
            ang = pos * freq
            c, s = np.cos(ang), np.sin(ang)
            v0, v1 = x[h, i], x[h, i + half]
            out[h, i] = v0*c - v1*s
            out[h, i + half] = v0*s + v1*c
    return out

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def main():
    args = sys.argv[1:]
    model_path = MODEL
    toks = [785, 6722, 315, 9625, 374]   # "The capital of France is"
    show_top5 = "--top5" in args
    if "--model" in args: model_path = args[args.index("--model")+1]
    if "--tokens" in args:
        toks = [int(t) for t in args[args.index("--tokens")+1].split(",")]

    kv, tensors, mm, base = load_gguf(model_path)
    L   = kv[next(k for k in kv if k.endswith("block_count"))]
    dim = kv[next(k for k in kv if k.endswith("embedding_length"))]
    nh  = kv[next(k for k in kv if k.endswith("attention.head_count"))]
    nkv = kv.get(next((k for k in kv if k.endswith("attention.head_count_kv")), None), nh) or nh
    hd  = dim // nh
    eps = kv.get(next((k for k in kv if k.endswith("rms_epsilon")), ""), 1e-6)
    rbase = kv.get(next((k for k in kv if k.endswith(("rope.freq_base","rope_freq_base"))), ""), 1e6)
    print(f"[ref] layers={L} dim={dim} heads={nh} kv_heads={nkv} hd={hd} eps={eps} rope={rbase}")

    embd  = get_w(kv, tensors, mm, base, "token_embd.weight")
    out_n = get_w(kv, tensors, mm, base, "output_norm.weight")
    out_w = get_w(kv, tensors, mm, base, "output.weight") if "output.weight" in tensors else embd

    cap_t = len(toks) + (32 if "--gen" in args else 1)
    if "--layers" in args:
        L = int(args[args.index("--layers")+1])
        print(f"[ref] override: using first {L} layers")
    cache_k = np.zeros((L, cap_t, nkv, hd), dtype=np.float32)
    cache_v = np.zeros((L, cap_t, nkv, hd), dtype=np.float32)
    x = None
    for t_idx, tok in enumerate(toks):
        x = embd[tok].astype(np.float32)
        for l in range(L):
            p = f"blk.{l}."
            an = get_w(kv, tensors, mm, base, p+"attn_norm.weight")
            xn = rmsnorm(x, an, eps)
            Wq = get_w(kv, tensors, mm, base, p+"attn_q.weight")
            Wk = get_w(kv, tensors, mm, base, p+"attn_k.weight")
            Wv = get_w(kv, tensors, mm, base, p+"attn_v.weight")
            bq = get_w(kv, tensors, mm, base, p+"attn_q.bias") if p+"attn_q.bias" in tensors else 0.0
            bk = get_w(kv, tensors, mm, base, p+"attn_k.bias") if p+"attn_k.bias" in tensors else 0.0
            bv = get_w(kv, tensors, mm, base, p+"attn_v.bias") if p+"attn_v.bias" in tensors else 0.0
            Wo = get_w(kv, tensors, mm, base, p+"attn_output.weight")

            q = (Wq @ xn + bq).reshape(nh, hd)
            k = (Wk @ xn + bk).reshape(nkv, hd)
            v = (Wv @ xn + bv).reshape(nkv, hd)

            cache_k[l, t_idx] = rope(k, t_idx, hd, rbase)
            cache_v[l, t_idx] = v

            q = rope(q, t_idx, hd, rbase)
            group = nh // nkv
            att = np.empty((nh, hd), dtype=np.float32)
            for h in range(nh):
                kvh = h // group
                scores = cache_k[l, :t_idx+1, kvh] @ q[h] / np.sqrt(hd)
                probs = softmax(scores)
                att[h] = probs @ cache_v[l, :t_idx+1, kvh]

            x = x + Wo.reshape(dim, dim) @ att.reshape(-1)

            fn = get_w(kv, tensors, mm, base, p+"ffn_norm.weight")
            xn = rmsnorm(x, fn, eps)
            Wg = get_w(kv, tensors, mm, base, p+"ffn_gate.weight")
            Wu = get_w(kv, tensors, mm, base, p+"ffn_up.weight")
            Wd = get_w(kv, tensors, mm, base, p+"ffn_down.weight")
            gate = Wg @ xn
            h = gate / (1.0 + np.exp(-gate)) * (Wu @ xn)
            x = x + Wd @ h
            print(f"[ref] tok {t_idx} layer {l}: |x|={np.linalg.norm(x):.3f} x[:3]={x[:3]}")

    logits = out_w @ rmsnorm(x, out_n, eps)
    top = np.argsort(logits)[::-1][:5]
    print("[ref] top5:", [(int(i), float(logits[i])) for i in top])

    n_gen = 16
    if "--gen" in args:
        n_gen = int(args[args.index("--gen")+1])
        eos = {151643, 151645}
        gen = []
        for g in range(n_gen):
            nxt = int(np.argmax(logits))
            if nxt in eos: break
            gen.append(nxt)
            t_idx = len(toks) + g
            x = embd[nxt].astype(np.float32)
            for l in range(L):
                p = f"blk.{l}."
                an = get_w(kv, tensors, mm, base, p+"attn_norm.weight")
                xn = rmsnorm(x, an, eps)
                Wq = get_w(kv, tensors, mm, base, p+"attn_q.weight")
                Wk = get_w(kv, tensors, mm, base, p+"attn_k.weight")
                Wv = get_w(kv, tensors, mm, base, p+"attn_v.weight")
                bq = get_w(kv, tensors, mm, base, p+"attn_q.bias") if p+"attn_q.bias" in tensors else 0.0
                bk = get_w(kv, tensors, mm, base, p+"attn_k.bias") if p+"attn_k.bias" in tensors else 0.0
                Wo = get_w(kv, tensors, mm, base, p+"attn_output.weight")
                q = (Wq @ xn + bq).reshape(nh, hd)
                k = (Wk @ xn + bk).reshape(nkv, hd)
                v = (Wv @ xn).reshape(nkv, hd)
                cache_k[l, t_idx] = rope(k, t_idx, hd, rbase)
                cache_v[l, t_idx] = v
                q = rope(q, t_idx, hd, rbase)
                group = nh // nkv
                att = np.empty((nh, hd), dtype=np.float32)
                for h in range(nh):
                    kvh = h // group
                    scores = cache_k[l, :t_idx+1, kvh] @ q[h] / np.sqrt(hd)
                    probs = softmax(scores)
                    att[h] = probs @ cache_v[l, :t_idx+1, kvh]
                x = x + Wo.reshape(dim, dim) @ att.reshape(-1)
                fn = get_w(kv, tensors, mm, base, p+"ffn_norm.weight")
                xn = rmsnorm(x, fn, eps)
                Wg = get_w(kv, tensors, mm, base, p+"ffn_gate.weight")
                Wu = get_w(kv, tensors, mm, base, p+"ffn_up.weight")
                Wd = get_w(kv, tensors, mm, base, p+"ffn_down.weight")
                gate = Wg @ xn
                h2 = gate / (1.0 + np.exp(-gate)) * (Wu @ xn)
                x = x + Wd @ h2
            logits = out_w @ rmsnorm(x, out_n, eps)

    import struct as _s
    f = open("data/models/qwen2.5-0.5b-instruct-q4_0.gguf","rb")
    # decode ids via tokenizer tables parsed quickly: reuse gguf kv scan is heavy;
    # fall back to printing ids only
    print("[ref] generated ids:", gen)

if __name__ == "__main__":
    main()
