#!/usr/bin/env python3
"""NumPy golden reference for gemma4-E2B decode, single position.

Mirrors llama.cpp build_gemma4 exactly (oracle src/models/gemma4.cpp).
Prints per-layer ||x|| to compare against engine TT_TRACE output.
Usage: ref_gemma4_numpy.py [--model PATH] [--token ID]
"""
import struct, sys
import numpy as np

def load_gguf(path):
    f = open(path, "rb")
    magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    assert magic == 0x46554747

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
    import mmap as _mmap
    _f = open(model_path, "rb")
    mm = _mmap.mmap(_f.fileno(), 0, prot=_mmap.PROT_READ)
    globals()['_keep_mm_file'] = _f   # prevent GC closing fd
    for _ in range(n_kv):
        key = r_str()
        (t,) = struct.unpack("<I", f.read(4))
        pos = f.tell()
        if key.endswith(("embedding_length","block_count")):
            kv[key] = struct.unpack("<i", f.read(4))[0]
        elif key.endswith("rms_epsilon"):
            kv[key] = struct.unpack("<f", f.read(4))[0]
        elif key.endswith(("freq_base","softcapping","sliding_window",
                           "key_length","value_length","dimension_count")):
            try: kv[key] = struct.unpack("<f", f.read(4))[0]
            except Exception: pass
        f.seek(pos); skip_val(t)

    tensors = {}
    for _ in range(n_tensors):
        name = r_str()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q"*nd, f.read(8*nd))
        (dtype,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        tensors[name] = (dims, dtype, off)
    base = (f.tell() + 31) & ~31
    import mmap
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return kv, tensors, mm, base

def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j+4] & 63
    return ((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4),
            (q[j+4] >> 4) | ((q[j-0] >> 6) << 4))

def deq_q4_K(buf, numel):
    nb = numel // 256
    out = np.empty(numel, dtype=np.float32)
    p = 0
    for i in range(nb):
        blk = buf[i*144:(i+1)*144]
        d = struct.unpack("<e", blk[0:2])[0]
        mn = struct.unpack("<e", blk[2:4])[0]
        sc_b = blk[4:16]; q = np.frombuffer(blk[16:], dtype=np.uint8)
        is_ = 0; o = 0
        for j in range(0, 256, 64):
            sc0, m0 = get_scale_min_k4(is_, sc_b)
            sc1, m1 = get_scale_min_k4(is_+1, sc_b)
            d1, mm0 = d*sc0, mn*m0
            d2, mm1 = d*sc1, mn*m1
            lo = (q[o:o+32] & 0xF).astype(np.float32)
            hi = (q[o:o+32] >> 4).astype(np.float32)
            out[p:p+32]   = d1*lo - mm0
            out[p+32:p+64] = d2*hi - mm1
            p += 64; o += 32; is_ += 2
    return out

def deq_q5_K(buf, numel):
    nb = numel // 256
    out = np.empty(numel, dtype=np.float32)
    for i in range(nb):
        blk = buf[i*176:(i+1)*176]
        d = struct.unpack("<e", blk[0:2])[0]
        mn = struct.unpack("<e", blk[2:4])[0]
        sc_b = blk[4:16]
        qh = np.frombuffer(blk[16:48], dtype=np.uint8)
        ql = np.frombuffer(blk[48:], dtype=np.uint8)
        is_ = 0; p = 0; u1, u2 = 1, 2
        base = i*256
        for j in range(0, 256, 64):
            sc0, m0 = get_scale_min_k4(is_, sc_b)
            sc1, m1 = get_scale_min_k4(is_+1, sc_b)
            d1, mm0 = d*sc0, mn*m0
            d2, mm1 = d*sc1, mn*m1
            lo = (ql[p:p+32] & 0xF).astype(np.float32) + np.where(qh[:32] & u1, 16, 0)
            hi = (ql[p:p+32] >> 4).astype(np.float32) + np.where(qh[:32] & u2, 16, 0)
            out[base+j:base+j+32]    = d1*lo - mm0
            out[base+j+32:base+j+64] = d2*hi - mm1
            p += 32; is_ += 2; u1 <<= 2; u2 <<= 2
    return out

def get_w(tensors, mm, base, name):
    dims, dtype, off = tensors[name]
    numel = int(np.prod(dims))
    raw = lambda n: mm[base+off : base+off+n]
    if dtype == 0:
        return np.frombuffer(raw(numel*4), dtype="<f4").astype(np.float32).reshape(dims[::-1])
    if dtype == 1:
        return np.frombuffer(raw(numel*2), dtype="<f2").astype(np.float32).reshape(dims[::-1])
    if dtype == 30:
        b = np.frombuffer(raw(numel*2), dtype="<u2").astype(np.uint32)
        f = (b << 16).view(np.float32) if hasattr((b<<16), 'view') else None
        f = (b << np.uint32(16))
        return f.view(np.float32).reshape(dims[::-1]) if False else \
               np.frombuffer(((b << np.uint32(16))).tobytes(), dtype="<f4").reshape(dims[::-1])
    if dtype == 2:
        nb = numel // 32
        arr = np.frombuffer(raw(nb*18), dtype=np.uint8).reshape(nb, 18)
        d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb,1)
        qs = arr[:, 2:].astype(np.int32)
        out = np.empty((nb,32), dtype=np.float32)
        out[:, :16] = ((qs & 0xF) - 8) * d
        out[:, 16:] = ((qs >> 4) - 8) * d
        return out.reshape(-1)[:numel].reshape(dims[::-1])
    if dtype == 3:
        nb = numel // 32
        arr = np.frombuffer(raw(nb*20), dtype=np.uint8).reshape(nb, 20)
        d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb,1)
        m = arr[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb,1)
        q8 = arr[:, 4:]
        out = np.empty((nb,32), dtype=np.float32)
        out[:, :16] = ((q8 & 0xF).astype(np.float32) * d) + m
        out[:, 16:] = ((q8 >> 4).astype(np.float32) * d) + m
        return out.reshape(-1)[:numel].reshape(dims[::-1])
    if dtype == 12:
        return deq_q4_K(raw((numel//256)*144), numel).reshape(dims[::-1])
    if dtype == 13:
        return deq_q5_K(raw((numel//256)*176), numel).reshape(dims[::-1])
    raise ValueError(f"{name}: dtype {dtype}")

def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt(np.mean(x*x) + eps) * g

def main():
    args = sys.argv[1:]
    model_path = "data/models/gemma-4-E2B-it-Q4_0.gguf"
    tok = 2
    if "--model" in args: model_path = args[args.index("--model")+1]
    if "--token" in args: tok = int(args[args.index("--token")+1])

    # authoritative parse via gguf-py reader (hand parser drifts on this file)
    from gguf.gguf_reader import GGUFReader
    rd = GGUFReader(model_path)
    tensors = {}
    base_abs = min(int(t.data_offset) for t in rd.tensors)  # aligned data section start
    for t in rd.tensors:
        dims_ne = tuple(int(d) for d in np.array(t.shape).astype(int))  # ne order [K,M]
        tensors[t.name] = (dims_ne, int(t.tensor_type), int(t.data_offset))
    base = 0
    L = 35; dim = 1536
    kv = {}
    import mmap as _mmap
    _f = open(model_path, "rb")
    mm = _mmap.mmap(_f.fileno(), 0, prot=_mmap.PROT_READ)
    globals()['_keep_mm_file'] = _f   # prevent GC closing fd
    print(f"[ref-g4] layers={L} dim={dim} token={tok}")

    def get_row(name, r):
        dims, dtype, off = tensors[name]
        K = dims[0]                       # ne[0] = row width
        raw = lambda n: mm[base+off+r*(dtype_rowbytes(dtype,K)) : base+off+(r+1)*dtype_rowbytes(dtype,K)]
        if dtype == 12: return deq_q4_K(raw((K//256)*144), K)
        if dtype == 13: return deq_q5_K(raw((K//256)*176), K)
        if dtype == 30:
            b = np.frombuffer(raw(K*2), dtype="<u2").astype(np.uint32)
            return np.frombuffer((b << np.uint32(16)).tobytes(), dtype="<f4")
        if dtype == 2: return deq_q4_K(raw((K//32)*18), K)
        raise ValueError(dtype)

    embd  = None   # row-accessed
    ple_t = None
    plnorm= get_w(tensors, mm, base, "per_layer_proj_norm.weight")
    def dtype_rowbytes(dt, K):
        return {12:(K//256)*144, 13:(K//256)*176, 30:K*2, 2:(K//32)*18, 3:(K//32)*20}[dt]
    plproj= get_w(tensors, mm, base, "per_layer_model_proj.weight")
    rope_f= get_w(tensors, mm, base, "rope_freqs.weight")
    out_n = get_w(tensors, mm, base, "output_norm.weight")
    print(f"[ref-g4] plproj dims={tensors['per_layer_model_proj.weight'][0]}")

    W = {}
    for l in range(L):
        W[l] = {k: get_w(tensors, mm, base, f"blk.{l}.{k}.weight")
                for k in ["attn_norm","attn_q","attn_q_norm","attn_k","attn_k_norm",
                          "attn_v","attn_output","post_attention_norm",
                          "ffn_norm","ffn_gate","ffn_up","ffn_down","post_ffw_norm",
                          "inp_gate","proj","post_norm","layer_output_scale"]}

    def gemv(Wm, x):   # Wm stored [K,M] (gguf ne reversed) -> y = x @ Wm? gguf: W[M_rows=out,K], row-major K-contig
        # numpy: our arrays are reshaped dims[::-1]: dims=[K,M] -> shape [M,K]; y = x @ W.T
        return x @ Wm.T

    x = get_row("token_embd.weight", tok).astype(np.float32) * np.sqrt(np.float32(dim))

    # PLE pipeline (build once from the EMBEDDING — not residual)
    row = plproj.shape[0]                       # 8960
    proj = gemv(plproj, x) * (1.0/np.sqrt(dim)) # [8960]
    proj = proj.reshape(L, 256)
    proj = proj / np.sqrt(np.sum(proj*proj, axis=1, keepdims=True)/256 + 1e-6) * plnorm
    pe = get_row("per_layer_token_embd.weight", tok).reshape(L, 256)
    PLE = (proj + pe) * (1.0/np.sqrt(2.0))
    print(f"[ref-g4] PLE[0][:4]={PLE[0][:4]}")

    pos = 0
    for l in range(L):
        wl = W[l]
        hd_l = wl["attn_k"].shape[0]      # k rows = kv_heads*hd; kv_heads=1 here
        nh_l = wl["attn_q"].shape[0] // hd_l
        is_full = hd_l > 256
        rbase = 1e6 if is_full else 1e4

        if l == 0:
            def _p(tag, v): print(f"[g4-L0] {tag} rms={np.sqrt(np.mean(v*v)):.4f}")
            _p("x_in", x)
        xn = rmsnorm(x, wl["attn_norm"])
        if l == 0: _p("xn", xn)
        q  = gemv(wl["attn_q"], xn).reshape(nh_l, hd_l)
        k  = gemv(wl["attn_k"], xn).reshape(1, hd_l)
        v  = gemv(wl["attn_v"], xn).reshape(1, hd_l)

        # per-head rmsnorm q/k
        q = q / np.sqrt(np.mean(q*q, axis=1, keepdims=True) + 1e-6) * wl["attn_q_norm"]
        k = k / np.sqrt(np.mean(k*k, axis=1, keepdims=True) + 1e-6) * wl["attn_k_norm"]
        v = v / np.sqrt(np.mean(v*v, axis=1, keepdims=True) + 1e-6)

        def rope_neox(h_, base_):
            half = h_.shape[-1]//2
            i = np.arange(half)
            freq = base_ ** (-2.0*i/h_.shape[-1])
            ff = rope_f[:half] if is_full else np.ones_like(freq)
            ang = pos * freq / ff
            c, s = np.cos(ang), np.sin(ang)
            out = h_.copy()
            out[..., :half] = h_[..., :half]*c - h_[..., half:]*s
            out[..., half:] = h_[..., :half]*s + h_[..., half:]*c
            return out
        q = rope_neox(q, rbase); k = rope_neox(k, rbase)

        scores = (q @ k[0].T) * 1.0       # attn scale 1
        att = np.exp(scores - scores.max(axis=-1, keepdims=True))
        att = att / att.sum(axis=-1, keepdims=True)
        if l == 0: _p("v", v)
        ao = v[0][None, :].repeat(q.shape[0], axis=0)   # pos=0: att=1 per head
        if l == 0: _p("ao", ao)
        attn_out = gemv(wl["attn_output"], ao.reshape(-1))
        if l == 0: _p("attn_out_raw", attn_out)
        x = x + rmsnorm(attn_out, wl["post_attention_norm"])
        if l == 0: _p("x_after_attn", x)

        xn2 = rmsnorm(x, wl["ffn_norm"])
        g = gemv(wl["ffn_gate"], xn2)
        g = 0.5*g*(1+np.tanh(0.7978845608028654*(g+0.044715*g*g*g)))
        u = gemv(wl["ffn_up"], xn2)
        mlp = gemv(wl["ffn_down"], g*u)
        if l == 0:
            _p("mlp_raw", mlp)
            _p("gu", g*u)
        x = x + rmsnorm(mlp, wl["post_ffw_norm"])

        pe_in = x.copy()
        gg = gemv(wl["inp_gate"], pe_in)
        gg = 0.5*gg*(1+np.tanh(0.7978845608028654*(gg+0.044715*gg*gg*gg)))
        pp = gemv(wl["proj"], gg*PLE[l])
        if l == 0:
            _p("ggPLE", gg*PLE[l]); _p("pp_prenorm", pp)
        pp = rmsnorm(pp, wl["post_norm"])
        if l == 0: _p("pp_postnorm", pp)
        if l == 0:
            _p("pp", pp); _p("pe_in", pe_in)
        x = pe_in + pp
        print(f"[ref-g4] L{l} pre_scale_rms={np.sqrt(np.mean(x*x)):.4f}")
        x = x * float(wl["layer_output_scale"].reshape(-1)[0])

    xf = rmsnorm(x, out_n)
    # logits: chunked row-wise dot to avoid 3GB dequant
    dims_e, dte, offe = tensors["token_embd.weight"]
    K_e = dims_e[0]; rb = dtype_rowbytes(dte, K_e)
    best = []
    for r0 in range(0, dims_e[1], 4096):
        rows = np.stack([get_row("token_embd.weight", r) for r in range(r0, min(r0+4096, dims_e[1]))])
        lg = rows @ xf / 30.0
        best.append(np.tanh(lg)*30.0)
    logits = np.concatenate(best)
    logits = np.tanh(logits) * 30.0
    top = np.argsort(-logits)[:8]
    print("[ref-g4] TOP8:", " ".join(f"({i},{logits[i]:.4f})" for i in top))

if __name__ == "__main__":
    main()
