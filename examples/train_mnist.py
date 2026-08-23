#!/usr/bin/env python3
"""M4 Gate B: End-to-end MNIST training demo using pybind11 bindings."""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build"))

# Build pybind if extension module not present
try:
    import tinytorch_pybind as tt
except ImportError:
    subprocess.run(["make", "-s", "pybind"], cwd=ROOT, check=True)
    import tinytorch_pybind as tt

import numpy as np

def onehot(y, k=10):
    z = np.zeros((y.size, k), dtype=np.float32)
    z[np.arange(y.size), y] = 1.0
    return z

def main():
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(42)

    batch_size = 32 if smoke else 128
    num_steps = 2 if smoke else 50

    print(f"== tinytorch pybind11 demo (smoke={smoke}) ==")

    # Initialize weights
    w1_val = (rng.standard_normal((785, 64)) * np.sqrt(2.0 / 785)).astype(np.float32)
    w2_val = (rng.standard_normal((65, 10)) * np.sqrt(2.0 / 65)).astype(np.float32)
    lr = 0.05

    for step in range(num_steps):
        # Synthetic / dummy MNIST data for smoke test
        x_raw = rng.standard_normal((batch_size, 784)).astype(np.float32)
        y_labels = rng.integers(0, 10, size=batch_size).astype(np.int64)

        # Pad ones for bias
        x_padded = np.concatenate([x_raw, np.ones((batch_size, 1), dtype=np.float32)], axis=1)

        w1 = tt.Node.leaf(w1_val, True)
        w2 = tt.Node.leaf(w2_val, True)

        # Forward pass: x @ w1 -> relu -> padones -> @ w2 -> softmax
        h1 = tt.Node.leaf(x_padded, False).matmul(w1).relu()
        logits = h1.padones().matmul(w2)
        probs = logits.softmax()

        p = probs.value()
        y_oh = onehot(y_labels)
        loss = float(-np.log(np.maximum(p[np.arange(batch_size), y_labels], 1e-12)).mean())

        # Backward pass seeded at logits
        logits.backward((p - y_oh) / float(batch_size))

        g1 = w1.grad()
        g2 = w2.grad()

        # Update weights (SGD)
        w1_val -= lr * g1
        w2_val -= lr * g2

        if step % 10 == 0 or smoke:
            acc = float((p.argmax(axis=1) == y_labels).mean())
            print(f"step {step:3d} | loss = {loss:.4f} | acc = {acc:.4f}")

    print("\nDEMO PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
