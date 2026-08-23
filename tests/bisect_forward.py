#!/usr/bin/env python3
"""Bisect forward-pass format assumptions against oracle logits dump.
Sweeps: q4_0 nibble pairing x weight orientation x RoPE convention.
Usage: bisect_forward.py /tmp/oracle_logits.bin
"""
import struct, sys
import numpy as np

MODEL = "data/models/qwen2.5-0.5b-instruct-q4_0.gguf"
import os
TOKS = [int(t) for t in os.environ.get("BISECT_TOKENS","785,6722,315,9625,374").split(",")]
ORACLE_TOP1 = int(os.environ.get("BISECT_ORACLE","12095"))

def load_gguf(path):
    f = open(path, "rb")
    magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    def r_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)
    def skip_val(t):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: r_str(); return
        if t == 9:
            (it,) = struct.unpack("<I", f.read(4))
            (al,) = struct.unpack("<Q", f.read(8))
            for _ in range(al): skip_val(it)
            return
        f.read(sizes[t])
    kv = {}
    for _ in range(n_kv):
        key = r_str().decode()
        (t,) = struct.unpack("<I", f.read(4))
        pos = f.tell()
        if key.endswith(("embedding_length","feed_forward_length","block_count",
                         "attention.head_count","attention.head_count_kv")):
            kv[key] = struct.unpack("<i", f.read(4))[0]
        elif key.endswith("rms_epsilon"):
            kv[key] = struct.unpack("<f", f.read(4))[0]
        elif key.endswith(("rope.freq_base","rope_freq_base")):
            kv[key] = struct.unpack("<f", f.read(4))[0]
        f.seek(pos); skip_val(t)
    tensors = {}
    for _ in range(n_tensors):
        name = r_str().decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q"*nd, f.read(8*nd))
        dtype, off = struct.unpack("<IQ", f.read(12))
        tensors[name] = (dims, dtype, off)
    base = (f.tell() + 31) & ~31
    mm = open(path, "rb").read()
    return kv, tensors, mm, base

kv, tensors, mm, base = load_gguf(MODEL)
L, dim = 24, 896
nh, nkv = 14, 2
hd = 64
eps = 1e-6
rbase = 1e6

_cache = {}
_dq = {}   # (name, nibble) -> (matrix, dims)
def get_raw(name):
    if name in _cache: return _cache[name]
    dims, dtype, off = tensors[name]
    numel = int(np.prod(dims))
    if dtype == 2:
        nb = numel // 32
        raw = mm[base+off : base+off+nb*18]
    elif dtype == 8:
        nb = numel // 32
        raw = mm[base+off : base+off+nb*34]
    elif dtype == 0:
        raw = mm[base+off : base+off+numel*4]
    else:
        raise ValueError(dtype)
    _cache[name] = (dims, dtype, raw, numel)
    return _cache[name]

def dequant(name, nibble):
    key = (name, nibble)
    if key in _dq: return _dq[key]
    dims, dtype, raw, numel = get_raw(name)
    if dtype == 0:
        shape = dims[::-1] if len(dims) > 1 else (dims[0], 1)
        r = (np.frombuffer(raw, "<f4").astype(np.float32).reshape(shape).reshape(-1), dims)
        _dq[key] = r; return r
    if dtype == 2:
        nb = len(raw)//18
        arr = np.frombuffer(raw, np.uint8).reshape(nb,18)
        d = arr[:,:2].copy().view(np.float16).astype(np.float32)
        qs = arr[:,2:].astype(np.int32)
        lo = ((qs & 0xF) - 8).astype(np.float32)*d
        hi = ((qs >> 4) - 8).astype(np.float32)*d
        out = np.empty((nb,32), np.float32)
        if nibble == 'split': out[:,:16]=lo; out[:,16:]=hi
        else:                 out[:,0::2]=lo; out[:,1::2]=hi
        v = out.reshape(-1)[:numel]
    else:  # q8_0
        nb = len(raw)//34
        arr = np.frombuffer(raw, np.uint8).reshape(nb,34)
        d = arr[:,:2].copy().view(np.float16).astype(np.float32)
        qs = arr[:,2:].copy().view(np.int8).astype(np.float32)
        v = (qs*d).reshape(-1)[:numel]
    r = (v.reshape(dims[1], dims[0]), dims)   # (ne1 rows, ne0 cols)
    _dq[key] = r; return r

_flat = {}
def flat(name):
    """flat dequantized vector + gguf dims tuple"""
    if name not in _flat:
        v, dims = dequant(name, NIBBLE)
        _flat[name] = (v.reshape(-1), dims)
    return _flat[name]

def matvec(name, x, orient):
    """orient='rows_out' : mem row (ne0 floats) = one output -> y = M(ne1,ne0) @ x
       orient='cols_out' : mem column j     = one output -> y = M(ne0,ne1).T @ x"""
    v, dims = flat(name)
    ne0, ne1 = dims[0], dims[1]
    if orient == 'rows_out':
        assert ne0 == x.shape[0], (name, dims, x.shape)
        return v.reshape(ne1, ne0) @ x
    else:
        assert ne0 == x.shape[0], (name, dims, x.shape)
        return v.reshape(ne0, ne1).T @ x

def rope(x, pos, conv):
    half = hd // 2
    out = x.copy()
    i = np.arange(half)
    freqs = rbase ** (-2.0*i/hd) * pos
    c, s = np.cos(freqs), np.sin(freqs)
    if conv == 'half':
        out[:, :half] = x[:, :half]*c - x[:, half:]*s
        out[:, half:] = x[:, :half]*s + x[:, half:]*c
    else:  # interleaved pairs (2i, 2i+1)
        out[:, 0::2] = x[:, 0::2]*c - x[:, 1::2]*s
        out[:, 1::2] = x[:, 1::2]*s + x[:, 0::2]*c
    return out

def rmsnorm(x, g):
    return x/np.sqrt(np.mean(x*x)+eps)*g

def softmax(x):
    e = np.exp(x-x.max()); return e/e.sum()

def run(nibble, orient, conv, kvmap='div', head='out', attn='on'):
    global NIBBLE
    NIBBLE = nibble
    embd,_ = dequant("token_embd.weight", nibble)
    out_n,_ = dequant("output_norm.weight", nibble)
    has_out = "output.weight" in tensors
    out_w = dequant("output.weight", nibble)[0] if has_out else embd
    if head == 'embd': out_w = embd

    ck = np.zeros((L, len(TOKS), nkv, hd), np.float32)
    cv = np.zeros((L, len(TOKS), nkv, hd), np.float32)
    x = None
    for ti, tok in enumerate(TOKS):
        x = embd[tok].astype(np.float32)
        for l in range(L):
            p = f"blk.{l}."
            xn = rmsnorm(x, dequant(p+"attn_norm.weight", nibble)[0])
            q = matvec(p+"attn_q.weight", xn, orient)
            k = matvec(p+"attn_k.weight", xn, orient)
            v = matvec(p+"attn_v.weight", xn, orient)
            bq = dequant(p+"attn_q.bias", nibble)[0] if p+"attn_q.bias" in tensors else 0.0
            bk = dequant(p+"attn_k.bias", nibble)[0] if p+"attn_k.bias" in tensors else 0.0
            bv = dequant(p+"attn_v.bias", nibble)[0] if p+"attn_v.bias" in tensors else 0.0
            q = (q+bq).reshape(nh,hd); k = (k+bk).reshape(nkv,hd); v = (v+bv).reshape(nkv,hd)
            ck[l,ti] = rope(k, ti, conv)
            cv[l,ti] = v
            q = rope(q, ti, conv)
            g = nh//nkv
            att = np.empty((nh,hd), np.float32)
            if attn == 'on':
                for h in range(nh):
                    kvh = h//g if kvmap=='div' else h%nkv
                    scores = ck[l,:ti+1,kvh] @ q[h]/np.sqrt(hd)
                    att[h] = softmax(scores) @ cv[l,:ti+1,kvh]
            x = x + matvec(p+"attn_output.weight", att.reshape(-1), orient)
            xn = rmsnorm(x, dequant(p+"ffn_norm.weight", nibble)[0])
            gate = matvec(p+"ffn_gate.weight", xn, orient)
            h2 = gate/(1.0+np.exp(-gate))*matvec(p+"ffn_up.weight", xn, orient)
            x = x + matvec(p+"ffn_down.weight", h2, orient)
    logits = out_w @ rmsnorm(x, out_n)
    top = np.argsort(logits)[::-1][:5]
    return [(int(i), round(float(logits[i]),2)) for i in top]

import itertools, sys
oracle_top1 = ORACLE_TOP1
print(f"{'nib':6s} {'ornt':6s} {'rope':6s} {'kvm':5s} {'head':5s} -> top1")
best=None
for nibble, orient, conv, kvmap, head in itertools.product(
        ('split',), ('rows_out','cols_out'), ('half','interleaved'),
        ('div','mod'), ('out','embd')):
    try:
        top = run(nibble, orient, conv, kvmap, head)
        hit = " <<< MATCH" if top[0][0]==oracle_top1 else ""
        print(f"{nibble:6s} {orient:6s} {conv:6s} {kvmap:5s} {head:5s} -> {top[0]} {top[1]}{hit}", flush=True)
    except Exception as e:
        print(f"{nibble:6s} {orient:6s} {conv:6s} {kvmap:5s} {head:5s} -> FAIL {e}", flush=True)

# diagnostics: attention off
top = run('split','rows_out','half','div','out',attn='off')
print("DIAG attn-off:", top[0])
