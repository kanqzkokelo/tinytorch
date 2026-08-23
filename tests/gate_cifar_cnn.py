#!/usr/bin/env python3
"""M4 Gate A: train CNN on CIFAR-10 >= 70% test accuracy.

Architecture:
  (B, 3, 32, 32)
  -> Conv2D(3 -> 32, 3x3, p=1) -> ReLU -> MaxPool2D(2x2, s=2)  # (B, 32, 16, 16)
  -> Conv2D(32 -> 64, 3x3, p=1) -> ReLU -> MaxPool2D(2x2, s=2) # (B, 64, 8, 8)
  -> Conv2D(64 -> 128, 3x3, p=1) -> ReLU -> MaxPool2D(2x2, s=2)# (B, 128, 4, 4 = 2048)
  -> Reshape(B, 2048)
  -> Linear(2048 -> 256) -> ReLU                              # (B, 256)
  -> Linear(256 -> 10) -> Softmax CE                           # (B, 10)
"""
import gc
import gzip
import os
import pickle
import sys
import tarfile
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

try:
    import tinytorch_pybind as tt
except ImportError:
    import subprocess
    subprocess.run(["make", "-s", "pybind"], cwd=ROOT, check=True)
    import tinytorch_pybind as tt

DATA = os.path.join(ROOT, "data")
TAR_PATH = os.path.join(DATA, "cifar-10-python.tar.gz")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

TARGET_ACC = 0.70
MAX_EPOCHS = 5
BATCH_SIZE = 256
SAMPLES_PER_EPOCH = 25600
LR = 0.003


def fetch_cifar():
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(TAR_PATH):
        print("Downloading CIFAR-10...", flush=True)
        urllib.request.urlretrieve(URL, TAR_PATH)
    batch_dir = os.path.join(DATA, "cifar-10-batches-py")
    if not os.path.exists(batch_dir):
        print("Extracting CIFAR-10...", flush=True)
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=DATA)

    def load_b(path):
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32)
        y = np.array(d[b"labels"], dtype=np.int64)
        return x, y

    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_b(os.path.join(batch_dir, f"data_batch_{i}"))
        xs.append(x)
        ys.append(y)
    xtr = np.concatenate(xs, axis=0)
    ytr = np.concatenate(ys, axis=0)
    xte, yte = load_b(os.path.join(batch_dir, "test_batch"))
    return xtr, ytr, xte, yte


def onehot(y, k=10):
    z = np.zeros((y.size, k), dtype=np.float32)
    z[np.arange(y.size), y] = 1.0
    return z


class Adam:
    def __init__(self, params, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1.0 - self.b2 ** self.t) / (1.0 - self.b1 ** self.t))
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


def forward_cnn(x_arr, w1, b1, w2, b2, w3, b3, w_fc1, w_fc2):
    x = tt.Node.leaf(x_arr, False)
    h1 = x.conv2d(w1, b1, stride_h=1, stride_w=1, pad_h=1, pad_w=1).relu().maxpool2d(2, 2, 2, 2) # (B, 32, 16, 16)
    h2 = h1.conv2d(w2, b2, stride_h=1, stride_w=1, pad_h=1, pad_w=1).relu().maxpool2d(2, 2, 2, 2) # (B, 64, 8, 8)
    h3 = h2.conv2d(w3, b3, stride_h=1, stride_w=1, pad_h=1, pad_w=1).relu().maxpool2d(2, 2, 2, 2) # (B, 128, 4, 4)
    b_size = x_arr.shape[0]
    h3_flat = h3.reshape([b_size, 2048])

    fc1 = h3_flat.padones().matmul(w_fc1).relu()                        # (B, 256)
    logits = fc1.padones().matmul(w_fc2)                                # (B, 10)
    return logits


def evaluate(x_test, y_test, w1, b1, w2, b2, w3, b3, w_fc1, w_fc2, mean, std, bs=512):
    correct = 0
    total = x_test.shape[0]
    for i in range(0, total, bs):
        xb = (x_test[i:i + bs] / 255.0 - mean) / std
        logits = forward_cnn(xb, w1, b1, w2, b2, w3, b3, w_fc1, w_fc2)
        pred = logits.value().argmax(axis=1)
        correct += int((pred == y_test[i:i + bs]).sum())
    return correct / total


def main():
    xtr_raw, ytr, xte_raw, yte = fetch_cifar()
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)

    print(f"CIFAR-10 loaded: train={xtr_raw.shape}, test={xte_raw.shape}", flush=True)

    rng = np.random.default_rng(1234)

    w1_v = (rng.standard_normal((32, 3, 3, 3)) * np.sqrt(2.0 / 27)).astype(np.float32)
    b1_v = np.zeros(32, dtype=np.float32)

    w2_v = (rng.standard_normal((64, 32, 3, 3)) * np.sqrt(2.0 / 288)).astype(np.float32)
    b2_v = np.zeros(64, dtype=np.float32)

    w3_v = (rng.standard_normal((128, 64, 3, 3)) * np.sqrt(2.0 / 576)).astype(np.float32)
    b3_v = np.zeros(128, dtype=np.float32)

    w_fc1_v = (rng.standard_normal((2049, 256)) * np.sqrt(2.0 / 2049)).astype(np.float32)
    w_fc2_v = (rng.standard_normal((257, 10)) * np.sqrt(2.0 / 257)).astype(np.float32)

    params = [w1_v, b1_v, w2_v, b2_v, w3_v, b3_v, w_fc1_v, w_fc2_v]
    opt = Adam(params, lr=LR)

    order = np.arange(xtr_raw.shape[0])
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(order)
        losses = []
        if epoch % 2 == 0:
            opt.lr *= 0.85

        sub_order = order[:SAMPLES_PER_EPOCH]
        for i in range(0, SAMPLES_PER_EPOCH - BATCH_SIZE + 1, BATCH_SIZE):
            idx = sub_order[i:i + BATCH_SIZE]
            xb = (xtr_raw[idx] / 255.0 - mean) / std
            yb = ytr[idx]

            w1 = tt.Node.leaf(w1_v, True)
            b1 = tt.Node.leaf(b1_v, True)
            w2 = tt.Node.leaf(w2_v, True)
            b2 = tt.Node.leaf(b2_v, True)
            w3 = tt.Node.leaf(w3_v, True)
            b3 = tt.Node.leaf(b3_v, True)
            w_fc1 = tt.Node.leaf(w_fc1_v, True)
            w_fc2 = tt.Node.leaf(w_fc2_v, True)

            logits = forward_cnn(xb, w1, b1, w2, b2, w3, b3, w_fc1, w_fc2)
            probs = logits.softmax()
            p = probs.value()

            loss = float(-np.log(np.maximum(p[np.arange(BATCH_SIZE), yb], 1e-12)).mean())
            losses.append(loss)

            logits.backward((p - onehot(yb)) / float(BATCH_SIZE))

            grads = [w1.grad(), b1.grad(), w2.grad(), b2.grad(),
                     w3.grad(), b3.grad(), w_fc1.grad(), w_fc2.grad()]
            opt.step(grads)

            del w1, b1, w2, b2, w3, b3, w_fc1, w_fc2, logits, probs, grads
            if i % (BATCH_SIZE * 5) == 0:
                gc.collect()

        eval_n = 2000 if epoch < MAX_EPOCHS else 10000
        acc = evaluate(xte_raw[:eval_n], yte[:eval_n],
                       tt.Node.leaf(w1_v, False), tt.Node.leaf(b1_v, False),
                       tt.Node.leaf(w2_v, False), tt.Node.leaf(b2_v, False),
                       tt.Node.leaf(w3_v, False), tt.Node.leaf(b3_v, False),
                       tt.Node.leaf(w_fc1_v, False), tt.Node.leaf(w_fc2_v, False),
                       mean, std)

        best_acc = max(best_acc, acc)
        print(f"epoch {epoch:2d}/{MAX_EPOCHS} | loss = {np.mean(losses):.4f} | val_acc = {acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if acc >= TARGET_ACC:
            full_acc = evaluate(xte_raw, yte,
                                tt.Node.leaf(w1_v, False), tt.Node.leaf(b1_v, False),
                                tt.Node.leaf(w2_v, False), tt.Node.leaf(b2_v, False),
                                tt.Node.leaf(w3_v, False), tt.Node.leaf(b3_v, False),
                                tt.Node.leaf(w_fc1_v, False), tt.Node.leaf(w_fc2_v, False),
                                mean, std)
            print(f"Full 10k test acc = {full_acc:.4f}", flush=True)
            if full_acc >= TARGET_ACC:
                print(f"\nGATE A GREEN: CIFAR-10 test acc {full_acc:.4f} >= {TARGET_ACC} at epoch {epoch}", flush=True)
                return 0

    if best_acc >= TARGET_ACC:
        full_acc = evaluate(xte_raw, yte,
                            tt.Node.leaf(w1_v, False), tt.Node.leaf(b1_v, False),
                            tt.Node.leaf(w2_v, False), tt.Node.leaf(b2_v, False),
                            tt.Node.leaf(w3_v, False), tt.Node.leaf(b3_v, False),
                            tt.Node.leaf(w_fc1_v, False), tt.Node.leaf(w_fc2_v, False),
                            mean, std)
        if full_acc >= TARGET_ACC:
            print(f"\nGATE A GREEN: CIFAR-10 best acc {full_acc:.4f} >= {TARGET_ACC}", flush=True)
            return 0

    print(f"\nGATE A RED: best acc {best_acc:.4f} < {TARGET_ACC}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
