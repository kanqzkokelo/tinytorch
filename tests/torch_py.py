"""Shared ctypes bindings for libtinytorch (tensors + autograd)."""
import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

F32P = ctypes.POINTER(ctypes.c_float)
I64P = ctypes.POINTER(ctypes.c_long)


def _load():
    subprocess.run(["make", "-s", "lib"], cwd=ROOT, check=True)
    lib = ctypes.CDLL(os.path.join(ROOT, "build", "libtinytorch.so"))

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

    for name in ("ag_leaf", "ag_add", "ag_mul", "ag_addscalar",
                 "ag_mulscalar", "ag_matmul", "ag_relu", "ag_softmax",
                 "ag_padones", "ag_retain"):
        f = getattr(lib, name)
        f.restype = ctypes.c_void_p
        if name in ("ag_addscalar", "ag_mulscalar"):
            f.argtypes = [ctypes.c_void_p, ctypes.c_float]
        elif name == "ag_leaf":
            f.argtypes = [ctypes.c_void_p, ctypes.c_int]
        else:
            f.argtypes = [ctypes.c_void_p] * (
                2 if name in ("ag_add", "ag_mul", "ag_matmul") else 1)

    lib.ag_value.restype = ctypes.c_void_p
    lib.ag_value.argtypes = [ctypes.c_void_p]
    lib.ag_grad.restype = ctypes.c_void_p
    lib.ag_grad.argtypes = [ctypes.c_void_p]
    lib.ag_requires_grad.restype = ctypes.c_int
    lib.ag_requires_grad.argtypes = [ctypes.c_void_p]
    lib.ag_backward.argtypes = [ctypes.c_void_p]
    lib.ag_backward_from.argtypes = [ctypes.c_void_p, F32P]
    lib.ag_release.argtypes = [ctypes.c_void_p]
    return lib


LIB = _load()


class Tensor:
    """Owns a C Tensor; releases on GC."""

    def __init__(self, ptr):
        self.ptr = ptr

    @classmethod
    def from_np(cls, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ndim = arr.ndim
        shape = (ctypes.c_long * max(ndim, 1))(*arr.shape)
        ptr = LIB.tt_fromdata(arr.ctypes.data_as(F32P), shape, ndim)
        assert ptr, "tt_fromdata failed"
        return cls(ptr)

    def to_np(self):
        n = int(LIB.tt_numel(self.ptr))
        assert n > 0 or True
        ndim = int(LIB.tt_ndim(self.ptr))
        shp = (ctypes.c_long * max(ndim, 1))()
        LIB.tt_shape(ctypes.c_void_p(self.ptr), shp)
        shape = tuple(shp)[:ndim] if ndim else ()
        buf = np.ctypeslib.as_array(LIB.tt_data(ctypes.c_void_p(self.ptr)),
                                    shape=(max(n, 1),))
        return buf.copy().reshape(shape)

    def release(self):
        if self.ptr:
            LIB.tt_release(ctypes.c_void_p(self.ptr))
            self.ptr = None


class Node:
    """Owns an AGNode reference; call .release() when done."""

    def __init__(self, ptr):
        self.ptr = ptr

    @classmethod
    def leaf(cls, arr, requires_grad=True):
        t = Tensor.from_np(arr)
        n = cls(LIB.ag_leaf(t.ptr, 1 if requires_grad else 0))
        t.release()  # node holds its own reference
        return n

    def value(self):
        return Tensor(LIB.ag_value(ctypes.c_void_p(self.ptr))).to_np()

    def grad(self):
        gp = LIB.ag_grad(ctypes.c_void_p(self.ptr))
        if not gp:
            return None
        return Tensor(gp).to_np()

    def backward(self, seed=None):
        if seed is None:
            LIB.ag_backward(ctypes.c_void_p(self.ptr))
        else:
            seed = np.ascontiguousarray(seed, dtype=np.float32)
            LIB.ag_backward_from(ctypes.c_void_p(self.ptr),
                                 seed.ctypes.data_as(F32P))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Node(LIB.ag_addscalar(self.ptr, ctypes.c_float(other)))
        return Node(LIB.ag_add(self.ptr, other.ptr))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Node(LIB.ag_mulscalar(self.ptr, ctypes.c_float(other)))
        return Node(LIB.ag_mul(self.ptr, other.ptr))

    def __rmul__(self, other):
        return self.__mul__(other)

    def matmul(self, other):
        return Node(LIB.ag_matmul(self.ptr, other.ptr))

    def relu(self):
        return Node(LIB.ag_relu(self.ptr))

    def softmax(self):
        return Node(LIB.ag_softmax(self.ptr))

    def padones(self):
        return Node(LIB.ag_padones(self.ptr))

    def release(self):
        if self.ptr:
            LIB.ag_release(ctypes.c_void_p(self.ptr))
            self.ptr = None
