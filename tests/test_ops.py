#!/usr/bin/env python3
"""M0 gate: every op must match NumPy reference, allclose(atol=1e-5, rtol=1e-5)."""
import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.run(["make", "-s", "lib"], cwd=ROOT, check=True)

lib = ctypes.CDLL(os.path.join(ROOT, "build", "libtinytorch.so"))
F32P = ctypes.POINTER(ctypes.c_float)
I64P = ctypes.POINTER(ctypes.c_long)

lib.tt_fromdata.restype = ctypes.c_void_p
lib.tt_fromdata.argtypes = [F32P, I64P, ctypes.c_int]
lib.tt_new.restype = ctypes.c_void_p
lib.tt_new.argtypes = [I64P, ctypes.c_int]
lib.tt_retain.restype = ctypes.c_void_p
lib.tt_retain.argtypes = [ctypes.c_void_p]
lib.tt_release.argtypes = [ctypes.c_void_p]
lib.tt_data.restype = F32P
lib.tt_data.argtypes = [ctypes.c_void_p]
lib.tt_numel.restype = ctypes.c_long
lib.tt_numel.argtypes = [ctypes.c_void_p]
lib.tt_ndim.restype = ctypes.c_int
lib.tt_ndim.argtypes = [ctypes.c_void_p]
lib.tt_shape.argtypes = [ctypes.c_void_p, I64P]
lib.tt_strides.argtypes = [ctypes.c_void_p, I64P]
for name in ("tt_relu", "tt_softmax", "tt_matmul_fast"):
    f = getattr(lib, name)
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p]
for name in ("tt_add", "tt_mul", "tt_matmul"):
    f = getattr(lib, name)
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.tt_matmul_omp.restype = ctypes.c_void_p
lib.tt_matmul_omp.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_int]
for name in ("tt_addscalar", "tt_mulscalar"):
    f = getattr(lib, name)
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_float]

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        print(f" FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


def make(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ndim = arr.ndim
    shape = (ctypes.c_long * max(ndim, 1))(*arr.shape)
    t = lib.tt_fromdata(arr.ctypes.data_as(F32P), shape, ndim)
    assert t, f"tt_fromdata failed for shape {arr.shape}"
    return t


def to_np(t, shape):
    n = int(lib.tt_numel(t))
    buf = np.ctypeslib.as_array(lib.tt_data(t), shape=(n,))
    return buf.copy().reshape(shape)


def run_unary(name, fn, x):
    t_in = make(x)
    t_out = fn(t_in)
    got = to_np(t_out, x.shape)
    ref = np.maximum(x, 0.0)
    ok = np.allclose(got, ref, atol=1e-5, rtol=1e-5)
    check(f"{name}{list(x.shape)}", ok,
          f"max_err={np.abs(got - ref).max():.3e}" if not ok else "")
    lib.tt_release(t_in)
    lib.tt_release(t_out)


def run_binary(name, fn, x, y):
    tx, ty = make(x), make(y)
    t_out = fn(tx, ty)
    got = to_np(t_out, x.shape)
    ref = {"add": np.add, "mul": np.multiply}[name](x, y)
    ok = np.allclose(got, ref, atol=1e-5, rtol=1e-5)
    check(f"{name}{list(x.shape)}", ok,
          f"max_err={np.abs(got - ref).max():.3e}" if not ok else "")
    lib.tt_release(tx)
    lib.tt_release(ty)
    lib.tt_release(t_out)


def run_scalar(name, fn, x, s):
    tx = make(x)
    t_out = fn(tx, ctypes.c_float(s))
    got = to_np(t_out, x.shape)
    ref = {"addscalar": np.add, "mulscalar": np.multiply}[name](x, s)
    ok = np.allclose(got, ref, atol=1e-5, rtol=1e-5)
    check(f"{name}{list(x.shape)}+{s}", ok,
          f"max_err={np.abs(got - ref).max():.3e}" if not ok else "")
    lib.tt_release(tx)
    lib.tt_release(t_out)


def softmax_ref(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def run_softmax(x):
    tx = make(x)
    t_out = lib.tt_softmax(tx)
    got = to_np(t_out, x.shape)
    ref = softmax_ref(x)
    ok = np.allclose(got, ref, atol=1e-5, rtol=1e-5)
    sums_ok = np.allclose(got.sum(axis=-1), 1.0, atol=1e-5)
    check(f"softmax{list(x.shape)}", ok and sums_ok,
          f"max_err={np.abs(got - ref).max():.3e}" if not ok else "")
    lib.tt_release(tx)
    lib.tt_release(t_out)


def run_matmul(a, b):
    ta, tb = make(a), make(b)
    t_out = lib.tt_matmul(ta, tb)
    got = to_np(t_out, (a.shape[0], b.shape[1]))
    ref = a @ b
    ok = np.allclose(got, ref, atol=1e-5, rtol=1e-5)
    check(f"matmul{list(a.shape)}x{list(b.shape)}", ok,
          f"max_err={np.abs(got - ref).max():.3e}" if not ok else "")
    lib.tt_release(ta)
    lib.tt_release(tb)
    lib.tt_release(t_out)


rng = np.random.default_rng(42)

print("== elementwise ==")
for shape in [(8,), (3, 4), (2, 3, 4)]:
    x = rng.standard_normal(shape).astype(np.float32)
    y = rng.standard_normal(shape).astype(np.float32)
    run_binary("add", lib.tt_add, x, y)
    run_binary("mul", lib.tt_mul, x, y)
    run_scalar("addscalar", lib.tt_addscalar, x, 2.5)
    run_scalar("mulscalar", lib.tt_mulscalar, x, -0.5)
    run_unary("relu", lib.tt_relu, x)
    run_unary("relu", lib.tt_relu, np.array([-3.0, 0.0, 2.5], dtype=np.float32))

print("== matmul ==")
run_matmul(rng.standard_normal((3, 4)), rng.standard_normal((4, 5)))
run_matmul(rng.standard_normal((16, 16)), rng.standard_normal((16, 16)))
run_matmul(rng.standard_normal((1, 7)), rng.standard_normal((7, 1)))
run_matmul(rng.standard_normal((32, 64)), rng.standard_normal((64, 17)))

print("== matmul fast/omp (AVX2 kernel incl. edge sizes) ==")
for shape_a, shape_b in [((3, 4), (4, 5)), ((17, 13), (13, 5)),
                         ((7, 7), (7, 7)), ((5, 300), (300, 3)),
                         ((9, 256), (256, 33)), ((64, 512), (512, 16))]:
    a = rng.standard_normal(shape_a)
    b = rng.standard_normal(shape_b)
    ref = a @ b
    for name, fn in (("fast", lib.tt_matmul_fast),
                     ("omp", lambda x, y: lib.tt_matmul_omp(x, y, 4))):
        ta, tb = make(a), make(b)
        t_out = fn(ta, tb)
        got = to_np(t_out, ref.shape)
        ok = np.allclose(got, ref, atol=1e-4, rtol=1e-4)
        check(f"matmul-{name}{list(shape_a)}x{list(shape_b)}", ok,
              f"max_err={np.abs(got - ref).max():.3e}" if not ok else "")
        lib.tt_release(ta)
        lib.tt_release(tb)
        lib.tt_release(t_out)

print("== softmax ==")
run_softmax(rng.standard_normal((5,)))
run_softmax(rng.standard_normal((3, 6)))
run_softmax(rng.standard_normal((2, 3, 4)) + 100.0)  # stability
run_softmax(np.full((4,), -1e9, dtype=np.float32))

print("== shape/stride metadata ==")
x = rng.standard_normal((2, 3, 4)).astype(np.float32)
t = make(x)
shp = (ctypes.c_long * 3)()
stp = (ctypes.c_long * 3)()
lib.tt_shape(ctypes.c_void_p(t), shp)
lib.tt_strides(ctypes.c_void_p(t), stp)
check("shape", list(shp) == [2, 3, 4], f"got {list(shp)}")
check("strides", list(stp) == [12, 4, 1], f"got {list(stp)}")
check("ndim", lib.tt_ndim(ctypes.c_void_p(t)) == 3)
check("numel", int(lib.tt_numel(ctypes.c_void_p(t))) == 24)
lib.tt_release(t)

print("== error paths ==")
a = make(rng.standard_normal((3, 4)))
b = make(rng.standard_normal((5, 6)))
check("matmul-mismatch->NULL", not lib.tt_matmul(a, b))
c = make(rng.standard_normal((2, 3)))
check("add-shape-mismatch->NULL", not lib.tt_add(a, c))
lib.tt_release(a)
lib.tt_release(b)
lib.tt_release(c)

print("== lifecycle (retain/release) ==")
t = make(rng.standard_normal((4,)))
u = lib.tt_retain(ctypes.c_void_p(t))
lib.tt_release(ctypes.c_void_p(u))
lib.tt_release(ctypes.c_void_p(t))
print("  ok  retain/release no crash")

if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S)")
    sys.exit(1)
print("\nALL M0 OP TESTS PASS")

# === Task 1 hostile param validation (TDD: must return NULL, never crash) ===
lib.tt_conv2d.restype = ctypes.c_void_p
lib.tt_conv2d.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
lib.tt_maxpool2d.restype = ctypes.c_void_p
lib.tt_maxpool2d.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
lib.tt_avgpool2d.restype = ctypes.c_void_p
lib.tt_avgpool2d.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4


class _CTensor(ctypes.Structure):
    _fields_ = [("data", F32P),
                ("shape", ctypes.POINTER(ctypes.c_long)),
                ("strides", ctypes.POINTER(ctypes.c_long)),
                ("ndim", ctypes.c_int),
                ("numel", ctypes.c_long),
                ("refcount", ctypes.c_int)]


def _t4(*shape):
    return make(rng.standard_normal(shape).astype(np.float32))


def test_hostile_conv_zero_stride():
    a, w = _t4(1, 1, 8, 8), _t4(2, 1, 3, 3)
    try:
        assert not lib.tt_conv2d(a, w, None, 0, 1, 0, 0), "stride_h=0 must return NULL"
    finally:
        lib.tt_release(a); lib.tt_release(w)


def test_hostile_conv_neg_stride():
    a, w = _t4(1, 1, 8, 8), _t4(2, 1, 3, 3)
    try:
        assert not lib.tt_conv2d(a, w, None, -1, 1, 0, 0), "stride_h=-1 must return NULL"
    finally:
        lib.tt_release(a); lib.tt_release(w)


def test_hostile_conv_zero_kernel():
    a, w = _t4(1, 1, 8, 8), _t4(2, 1, 3, 3)
    tw = _CTensor.from_address(w)
    assert tw.ndim == 4 and tw.shape[2] == 3  # layout sanity
    tw.shape[2] = 0  # hostile: HH=0 (allocator would never build this)
    try:
        assert not lib.tt_conv2d(a, w, None, 1, 1, 0, 0), "HH=0 must return NULL"
    finally:
        tw.shape[2] = 3
        lib.tt_release(a); lib.tt_release(w)


def test_hostile_conv_neg_pad():
    a, w = _t4(1, 1, 8, 8), _t4(2, 1, 3, 3)
    try:
        assert not lib.tt_conv2d(a, w, None, 1, 1, -1, 0), "pad_h=-1 must return NULL"
    finally:
        lib.tt_release(a); lib.tt_release(w)


def test_hostile_maxpool_zero_pool():
    a = _t4(1, 1, 8, 8)
    try:
        assert not lib.tt_maxpool2d(a, 0, 2, 2, 2), "pool_h=0 must return NULL"
    finally:
        lib.tt_release(a)


def test_hostile_avgpool_zero_stride():
    a = _t4(1, 1, 8, 8)
    try:
        assert not lib.tt_avgpool2d(a, 2, 2, 0, 2), "stride_h=0 must return NULL"
    finally:
        lib.tt_release(a)


# === Task 2 hostile alloc validation (TDD: huge dims -> NULL, never hang/OOM) ===
def _try_new(dims):
    arr = (ctypes.c_long * len(dims))(*dims)
    return lib.tt_new(arr, len(dims))


def test_hostile_huge_dims_wraparound():
    # 2**40 * 2**40 wraps long numel to 0 pre-fix -> must be NULL
    assert not _try_new((2**40, 2**40)), "2**40 dims must return NULL"


def test_hostile_huge_dims_product_overflow():
    # 2**62 * 2 overflows LONG_MAX -> must be NULL
    assert not _try_new((2**62, 2)), "numel overflow must return NULL"


def test_hostile_huge_dims_over_cap():
    # 2**42 elements: no long overflow, but insane -> reject before calloc
    assert not _try_new((2**21, 2**21)), "over-cap alloc must return NULL"


def test_hostile_huge_single_dim():
    assert not _try_new((2**40,)), "single huge dim must return NULL"


def test_hostile_negative_dim():
    assert not _try_new((-3, 4)), "negative dim must return NULL"


def test_hostile_zero_dim():
    assert not _try_new((0, 4)), "zero dim must return NULL"


def test_hostile_fromdata_huge_rejected():
    big = (ctypes.c_long * 2)(2**40, 2**40)
    assert not lib.tt_fromdata(None, big, 2), "fromdata huge dims must return NULL"


# === Task 4: pybind NULL -> ValueError (TDD: shape mismatch must raise) ===
def _load_pybind():
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _sys.path.insert(0, _os.path.join(ROOT, "build"))
    try:
        import tinytorch_pybind as _tt
    except ImportError:
        _sp.run(["make", "-s", "pybind"], cwd=ROOT, check=True)
        import tinytorch_pybind as _tt
    return _tt


def test_shape_mismatch_raises():
    import numpy as _np

    _tt = _load_pybind()
    a = _tt.Node.leaf(_np.ones((2, 3), dtype=_np.float32), False)
    b = _tt.Node.leaf(_np.ones((4, 5), dtype=_np.float32), False)
    try:
        c = a.matmul(b)
    except ValueError:
        return
    raise AssertionError("matmul shape mismatch must raise ValueError")


def test_ctypes_mismatch_raises():
    import os as _os2
    import sys as _sys

    import numpy as _np

    _sys.path.insert(0, _os2.path.dirname(_os2.path.abspath(__file__)))
    from torch_py import Node as _CNode

    a = _CNode.leaf(_np.ones((2, 3), dtype=_np.float32), False)
    b = _CNode.leaf(_np.ones((4, 5), dtype=_np.float32), False)
    try:
        try:
            a.matmul(b)
        except ValueError:
            return
        raise AssertionError("ctypes matmul mismatch must raise ValueError")
    finally:
        a.release()
        b.release()
