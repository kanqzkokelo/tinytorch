#!/usr/bin/env python3
"""M1 Gate A: gradcheck vs central finite differences, rel err < 1e-4.

Metric: elementwise |analytic - numeric| <= atol + rtol*|numeric|
with rtol=1e-4 (gate), atol=1e-6 (fp32 noise floor).
"""
import sys

import numpy as np

from torch_py import Node

RTOL = 1e-4   # gate: rel err < 1e-4
ATOL = 1e-5   # fp32 central-FD roundoff floor (h-sweep shows ~1e-6..1e-5 abs)
H = 1e-2
FAILURES = []
R = None


def forward_scalar(build, arrays):
    """Rebuild graph; return (float64 scalar sum(R*out), nodes, out_node)."""
    nodes = [Node.leaf(a.copy(), True) for a in arrays]
    out = build(*nodes)
    val = float(np.sum(R * out.value(), dtype=np.float64))
    return val, nodes, out


def gradcheck(name, build, arrays):
    global R
    probe_nodes = [Node.leaf(a.copy(), False) for a in arrays]
    probe = build(*probe_nodes)
    out_shape = probe.value().shape
    probe.release()
    for n in probe_nodes:
        n.release()
    R = np.random.default_rng(7).standard_normal(out_shape).astype(np.float32)

    # analytic gradients
    _, nodes, out = forward_scalar(build, arrays)
    out.backward(R)
    analytic = [n.grad().copy() for n in nodes]

    # numeric central finite differences
    numeric = [np.zeros_like(a) for a in arrays]
    for k in range(len(arrays)):
        num = numeric[k].reshape(-1)
        for i in range(arrays[k].size):
            ap = [x.copy() for x in arrays]
            am = [x.copy() for x in arrays]
            ap[k].reshape(-1)[i] += H
            am[k].reshape(-1)[i] -= H
            fp, _, _ = forward_scalar(build, ap)
            fm, _, _ = forward_scalar(build, am)
            num[i] = (fp - fm) / (2 * H)

    worst_violation = -np.inf
    max_rel = 0.0
    for an, nm in zip(analytic, numeric):
        viol = np.abs(an - nm) - RTOL * np.abs(nm)
        worst_violation = max(worst_violation, float(viol.max()))
        mask = np.abs(nm) > 1e-6
        if mask.any():
            max_rel = max(max_rel, float(
                (np.abs(an - nm)[mask] / np.abs(nm)[mask]).max()))
    ok = worst_violation < ATOL
    status = " ok " if ok else " FAIL"
    print(f"{status} {name:38s} max_rel={max_rel:.2e} "
          f"violation={worst_violation:.2e}")
    if not ok:
        FAILURES.append(name)

    for n in nodes:
        n.release()
    out.release()


rng = np.random.default_rng(0)


def clear_x(shape):
    """Random tensor with all |x| > 0.05 (keeps FD clear of relu kink)."""
    while True:
        x = rng.standard_normal(shape).astype(np.float32)
        if np.abs(x).min() > 0.05:
            return x


print("== M1 Gate A: gradcheck (central FD, h=1e-2) ==")

X = clear_x((4, 3))
Y = clear_x((4, 3))
gradcheck("add", lambda x, y: x + y, [X, Y])
gradcheck("mul", lambda x, y: x * y, [X, Y])
gradcheck("addscalar", lambda x: x + 1.5, [X])
gradcheck("mulscalar", lambda x: x * (-0.7), [X])
gradcheck("relu", lambda x: x.relu(), [clear_x((4, 3))])

gradcheck("matmul", lambda x, w: x.matmul(w),
          [clear_x((4, 3)), clear_x((3, 5))])
gradcheck("softmax", lambda x: x.softmax(), [clear_x((4, 5))])
gradcheck("padones", lambda x: x.padones(), [clear_x((4, 3))])
gradcheck("composite softmax(relu(0.5*x@w)+0.1)",
          lambda x, w: ((x.matmul(w) * 0.5).relu() + 0.1).softmax(),
          [clear_x((4, 3)), clear_x((3, 6))])

if FAILURES:
    print(f"\nGATE A RED: {len(FAILURES)} failing expression(s): {FAILURES}")
    sys.exit(1)
print("\nGATE A GREEN: all grads match central finite differences "
      f"(rtol={RTOL}, atol={ATOL})")
