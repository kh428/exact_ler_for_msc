"""Single-thread native arithmetic in F_q[x]/x^(K+1)."""
import ctypes
from .native_build import build_library
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
U = ctypes.c_uint64
PTR = ctypes.POINTER(U)


class PolynomialField:
    def __init__(self, ring, build_dir=None):
        path, self.evidence = build_library("modular_polynomial_endpoints.cpp", build_dir)
        self.lib = ctypes.CDLL(str(path))
        self.ring, self.prime = ring, ring.prime
        self.one = (1 << 64) % self.prime
        self.inverse_R = pow(self.one, -1, self.prime)
        self.lib.msc_poly_convert.argtypes = [PTR, ctypes.c_size_t, U, ctypes.c_int]
        self.lib.msc_poly_add.argtypes = [PTR, PTR, ctypes.c_size_t, U]
        self.lib.msc_poly_broadcast.argtypes = [PTR, PTR, ctypes.c_uint, ctypes.c_size_t,
                                                PTR, U, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, U]
        for name in ('msc_poly_convert', 'msc_poly_add', 'msc_poly_broadcast'):
            getattr(self.lib, name).restype = ctypes.c_int

    def pointer(self, array):
        if array.dtype != np.uint64 or not array.flags.c_contiguous or array.size > 11*(1 << 26):
            raise ValueError('Native polynomial array must be contiguous uint64 within its cap')
        return array.ctypes.data_as(PTR)

    def convert(self, array, encode):
        if self.lib.msc_poly_convert(self.pointer(array), array.size, self.prime, int(encode)):
            raise ValueError('Native polynomial conversion rejected its inputs')

    def add(self, dest, source):
        if dest.shape != source.shape:
            raise ValueError('Polynomial addition shapes differ')
        if self.lib.msc_poly_add(self.pointer(dest), self.pointer(source), dest.size, self.prime):
            raise ValueError('Native polynomial addition rejected its inputs')

    def broadcast(self, dest, source, source_scope, output_scope, old_degree, assignment=None):
        source_scope, output_scope = tuple(source_scope), tuple(output_scope)
        assignment = {} if assignment is None else assignment
        width, source_degree = dest.shape[-1], source.shape[-1]-1
        if (dest.shape[:-1] != (2,)*len(output_scope) or source.shape[:-1] != (2,)*len(source_scope)
                or not set(source_scope) <= set(output_scope) | set(assignment)):
            raise ValueError('Native polynomial broadcast scope mismatch')
        strides = {v: 1 << (len(source_scope)-i-1) for i, v in enumerate(source_scope)}
        fixed = sum(strides[v]*int(bit) for v, bit in assignment.items() if v in strides)
        masks, cumulative = [], 0
        for v in reversed(output_scope):
            cumulative ^= strides.get(v, 0)
            masks.append(cumulative)
        masks = np.array(masks, dtype=np.uint64)
        status = self.lib.msc_poly_broadcast(self.pointer(dest), self.pointer(source), len(output_scope),
                                              1 << len(source_scope), self.pointer(masks), fixed,
                                              width, old_degree, source_degree, self.prime)
        if status:
            raise ValueError('Native polynomial broadcast rejected its inputs: '+str(status))
        return min(width-1, old_degree+source_degree)

    def encode_polynomial(self, value, degree=None):
        value = self.ring(value)
        degree = value.degree if degree is None else degree
        array = np.zeros(degree+1, dtype=np.uint64)
        array[:len(value.coefficients)] = [v*self.one % self.prime for v in value.coefficients]
        return array

    def decode_polynomial(self, values):
        return self.ring.coefficients(int(v)*self.inverse_R % self.prime for v in values)
