#!/usr/bin/env python3
"""M1 Gate B: 2-layer MLP trains MNIST >= 97% test acc in <= 10 epochs.

MLP: 784 -> 128 (relu) -> 10, biases folded via a constant-ones input
column (framework has no broadcast add). Loss: softmax CE, gradient
seeded analytically at the logits node: (softmax - onehot) / batch.
"""
import gzip
import os
import struct
import sys
import time
import urllib.request

import numpy as np

from torch_py import Node

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", 16),
    "train_labels": ("train-labels-idx1-ubyte.gz", 8),
    "test_images": ("t10k-images-idx3-ubyte.gz", 16),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", 8),
}

MAX_EPOCHS = 10
TARGET_ACC = 0.97
BATCH = 128
LR = 0.1
MOMENTUM = 0.9


def fetch():
    os.makedirs(DATA, exist_ok=True)
    out = {}
    for key, (fname, off) in FILES.items():
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(f"{BASE}/{fname}", path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if "images" in key:
            magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
            assert magic == 2051, f"bad idx magic {magic}"
            out[key] = np.frombuffer(raw, np.uint8, offset=off)\
                .reshape(n, rows * cols)
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=off)
    return out


def onehot(y, k=10):
    z = np.zeros((y.size, k), dtype=np.float32)
    z[np.arange(y.size), y] = 1.0
    return z


def pad_ones(x):
    return np.ascontiguousarray(
        np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)],
                       axis=1))


def forward_logits(xb, w1, w2):
    """xb:(B,784) w1:(785,H) w2:(H+1,10) -> logits node (graph-connected)."""
    h = Node.leaf(xb, False).padones().matmul(w1).relu()
    return h.padones().matmul(w2)


def evaluate(x, y, w1, w2, bs=2000):
    correct = 0
    for i in range(0, x.shape[0], bs):
        logits = forward_logits(x[i:i + bs], w1, w2)
        pred = logits.value().argmax(axis=1)
        correct += int((pred == y[i:i + bs]).sum())
    return correct / x.shape[0]


def main():
    d = fetch()
    mean, std = 0.1307, 0.3081
    xtr = ((d["train_images"].astype(np.float32) / 255.0) - mean) / std
    ytr = d["train_labels"].astype(np.int64)
    xte = ((d["test_images"].astype(np.float32) / 255.0) - mean) / std
    yte = d["test_labels"].astype(np.int64)
    print(f"train={xtr.shape} test={xte.shape}")

    rng = np.random.default_rng(1234)
    H = 256
    w1v = (rng.standard_normal((785, H)) * np.sqrt(2.0 / 785))\
        .astype(np.float32)
    w2v = (rng.standard_normal((H + 1, 10)) * np.sqrt(2.0 / (H + 1)))\
        .astype(np.float32)
    v1 = np.zeros_like(w1v)
    v2 = np.zeros_like(w2v)

    order = np.arange(xtr.shape[0])
    best = 0.0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(order)
        losses = []
        for i in range(0, xtr.shape[0] - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            w1 = Node.leaf(w1v, True)
            w2 = Node.leaf(w2v, True)
            logits = forward_logits(xtr[idx], w1, w2)
            probs = logits.softmax()
            p = probs.value()
            y = onehot(ytr[idx])
            loss = float(-np.log(np.maximum(
                p[np.arange(BATCH), ytr[idx]], 1e-12)).mean())
            losses.append(loss)
            # dCE/dlogits = (softmax - onehot) / batch, seeded at logits
            logits.backward((p - y) / BATCH)
            g1, g2 = w1.grad(), w2.grad()
            v1[:] = MOMENTUM * v1 - LR * g1
            v2[:] = MOMENTUM * v2 - LR * g2
            w1v += v1
            w2v += v2
            for n in (w1, w2, logits, probs):
                n.release()

        acc = evaluate(xte, yte,
                       Node.leaf(w1v, False), Node.leaf(w2v, False))
        # note: evaluate takes nodes; wrap again properly below
        best = max(best, acc)
        print(f"epoch {epoch:2d}  loss={np.mean(losses):.4f}  "
              f"test_acc={acc:.4f}  ({time.time()-t0:.0f}s)")
        if acc >= TARGET_ACC:
            print(f"\nGATE B GREEN: {acc:.4f} >= {TARGET_ACC} "
                  f"at epoch {epoch} (<= {MAX_EPOCHS})")
            return 0

    print(f"\nGATE B RED: best_acc={best:.4f} < {TARGET_ACC} "
          f"after {MAX_EPOCHS} epochs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
