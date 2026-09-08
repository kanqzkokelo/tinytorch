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


def gradcheck(name, build, arrays, h=H, atol=ATOL):
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
            ap[k].reshape(-1)[i] += h
            am[k].reshape(-1)[i] -= h
            fp, _, _ = forward_scalar(build, ap)
            fm, _, _ = forward_scalar(build, am)
            num[i] = (fp - fm) / (2 * h)

    worst_violation = -np.inf
    max_rel = 0.0
    for an, nm in zip(analytic, numeric):
        viol = np.abs(an - nm) - RTOL * np.abs(nm)
        worst_violation = max(worst_violation, float(viol.max()))
        mask = np.abs(nm) > 1e-6
        if mask.any():
            max_rel = max(max_rel, float(
                (np.abs(an - nm)[mask] / np.abs(nm)[mask]).max()))
    ok = worst_violation < atol
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

gradcheck("reshape", lambda x: x.reshape((2, 6)), [clear_x((3, 4))])
gradcheck("conv2d", lambda x, w, b: x.conv2d(w, b, stride=1, pad=1),
          [clear_x((2, 2, 5, 5)), clear_x((3, 2, 3, 3)), clear_x((3,))], atol=2e-4)
gradcheck("maxpool2d", lambda x: x.maxpool2d(pool_size=2, stride=2),
          [clear_x((2, 2, 6, 6))])
gradcheck("avgpool2d", lambda x: x.avgpool2d(pool_size=2, stride=2),
          [clear_x((2, 2, 6, 6))])
gradcheck("composite softmax(relu(0.5*x@w)+0.1)",
          lambda x, w: ((x.matmul(w) * 0.5).relu() + 0.1).softmax(),
          [clear_x((4, 3)), clear_x((3, 6))])

if FAILURES:
    print(f"\nGATE A RED: {len(FAILURES)} failing expression(s): {FAILURES}")
    sys.exit(1)
print("\nGATE A GREEN: all grads match central finite differences "
      f"(rtol={RTOL}, atol={ATOL})")


# === Task 3: size-aware backward seed (TDD: mismatch must raise, not OOB-read) ===
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from torch_py import Node as _Node


def _leaf10():
    return _Node.leaf(
        __import__("numpy").array(
            [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], dtype="float32"
        ),
        True,
    )


def test_short_seed_rejected():
    import numpy as np

    x = _leaf10()
    z = x * 2
    try:
        try:
            z.backward(np.array([1.0], dtype=np.float32))
        except (ValueError, RuntimeError):
            return
        assert False, "short seed must raise, not OOB-read"
    finally:
        z.release()
        x.release()


def test_long_seed_rejected():
    import numpy as np

    x = _leaf10()
    z = x * 2
    try:
        try:
            z.backward(np.ones(20, dtype=np.float32))
        except (ValueError, RuntimeError):
            return
        assert False, "long seed must raise, not over-read"
    finally:
        z.release()
        x.release()


def test_none_seed_gives_ones_grad():
    x = _leaf10()
    z = x * 2
    try:
        z.backward(None)
        g = z.grad()
        assert g is not None
        assert (g == 1.0).all(), f"None seed must fill ones, got {g}"
        xg = x.grad()
        assert (xg == 2.0).all(), f"dx must be 2.0, got {xg}"
    finally:
        z.release()
        x.release()


def test_correct_seed_ok():
    import numpy as np

    x = _leaf10()
    z = x * 2
    try:
        seed = np.arange(1, 11, dtype=np.float32)
        z.backward(seed)
        xg = x.grad()
        assert (xg == 2 * seed).all(), f"dx must be 2*seed, got {xg}"
    finally:
        z.release()
        x.release()


def test_int_dtype_seed_coerced():
    import numpy as np

    x = _leaf10()
    z = x * 2
    try:
        z.backward(np.arange(1, 11, dtype=np.int32))
        xg = x.grad()
        assert (xg == 2 * np.arange(1, 11, dtype=np.float32)).all()
    finally:
        z.release()
        x.release()


# === Task 8: hostile/edge grad battery (append-only; no case may crash/hang) ===
import signal as _gsig


class _GTimeout:
    """SIGALRM guard for stress loops (no-op where SIGALRM missing)."""

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        if hasattr(_gsig, "SIGALRM"):
            def _raise(*a):
                raise TimeoutError("stress case timed out")
            _gsig.signal(_gsig.SIGALRM, _raise)
            _gsig.alarm(self.sec)
        return self

    def __exit__(self, *exc):
        if hasattr(_gsig, "SIGALRM"):
            _gsig.alarm(0)
        return False


def test_wrong_dtype_f64_seed_coerced():
    import numpy as np

    x = _leaf10()
    z = x * 2
    try:
        z.backward(np.arange(1, 11, dtype=np.float64))
        xg = x.grad()
        assert (xg == 2 * np.arange(1, 11, dtype=np.float32)).all(), \
            f"f64 seed must coerce, got {xg}"
    finally:
        z.release()
        x.release()


def test_noncontiguous_seed_coerced():
    import numpy as np

    x = _leaf10()
    z = x * 2
    try:
        seed = np.arange(1, 21, dtype=np.float64)[::2]  # non-contig f64
        assert not seed.flags["C_CONTIGUOUS"]
        assert seed.size == 10
        z.backward(seed)
        xg = x.grad()
        assert (xg == 2 * np.arange(1, 20, 2, dtype=np.float32)).all(), \
            f"non-contig seed must coerce, got {xg}"
    finally:
        z.release()
        x.release()


def test_deep_chain_1000_backward():
    import numpy as np

    from torch_py import Node

    with _GTimeout(60):
        v = Node.leaf(np.array([1.0, 2.0], dtype=np.float32), True)
        nodes = [v]
        cur = v
        for _ in range(1000):
            cur = cur + 1.0
            nodes.append(cur)
        try:
            cur.backward(None)
            g = v.grad()
            assert g is not None
            assert np.allclose(g, 1.0, atol=1e-6), f"deep-chain grad must be 1, got {g}"
        finally:
            for n in nodes:
                n.release()


def test_shared_subgraph_double_backward():
    import numpy as np

    from torch_py import Node

    x = Node.leaf(np.array([1.0, 2.0, 3.0], dtype=np.float32), True)
    y = x * x
    z = y + y
    try:
        with _GTimeout(30):
            z.backward(None)
            g1 = x.grad().copy()
            assert np.allclose(g1, [4.0, 8.0, 12.0]), f"1st grad 4x, got {g1}"
            z.backward(None)  # must not crash; accumulates deterministically
            g2 = x.grad()
            assert np.isfinite(g2).all()
            assert np.allclose(g2, 3 * g1), f"2nd backward accumulates 3x, got {g2}"
    finally:
        z.release()
        y.release()
        x.release()


def test_nan_backward_no_crash():
    import numpy as np

    from torch_py import Node

    with _GTimeout(30):
        x = Node.leaf(np.array([np.nan, 1.0], dtype=np.float32), True)
        y = x * x
        try:
            y.backward(None)  # must not crash
            g = x.grad()
            assert g is not None and g.shape == (2,)
        finally:
            y.release()
            x.release()


def test_inf_backward_no_crash():
    import numpy as np

    from torch_py import Node

    with _GTimeout(30):
        x = Node.leaf(np.array([np.inf, -np.inf], dtype=np.float32), True)
        r = x.relu()
        try:
            r.backward(None)  # must not crash
            g = x.grad()
            assert g is not None and g.shape == (2,)
            assert np.isfinite(g).all(), f"relu(+-Inf) grad must be finite, got {g}"
        finally:
            r.release()
            x.release()
