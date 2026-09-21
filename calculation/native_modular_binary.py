"""Exact modular arithmetic for binary-factor contractions."""
import ctypes
from .native_build import build_library
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


class BinaryField:
    def __init__(self, prime, build_dir=None):
        if not 2 < prime < 1 << 62 or prime % 2 == 0:
            raise ValueError('Expected an odd certified prime below 2^62')
        self.prime = prime
        self.one = (1 << 64) % prime
        self.inverse_R = pow(self.one, -1, prime)
        self.build_dir = build_dir
        path, self.evidence = build_library("modular_binary.cpp", build_dir)
        self.library = ctypes.CDLL(str(path))
        self.ptr = ctypes.POINTER(ctypes.c_uint64)
        self.library.msc_binary_convert.argtypes = [self.ptr, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_int]
        self.library.msc_binary_add.argtypes = [self.ptr, self.ptr, ctypes.c_size_t, ctypes.c_uint64]
        self.library.msc_binary_broadcast.argtypes = [self.ptr, self.ptr, ctypes.c_uint, ctypes.c_size_t,
                                                       self.ptr, ctypes.c_uint64, ctypes.c_uint64]
        for name in ('msc_binary_convert', 'msc_binary_add', 'msc_binary_broadcast'):
            getattr(self.library, name).restype = ctypes.c_int

    def pointer(self, array):
        assert array.dtype == np.uint64 and array.flags.c_contiguous and array.size <= 1 << 26
        return array.ctypes.data_as(self.ptr)

    def convert(self, array, *, encode):
        if self.library.msc_binary_convert(self.pointer(array), array.size, self.prime, int(encode)):
            raise ValueError('Native residue conversion refused')
        return array

    def add(self, destination, source):
        assert destination.shape == source.shape
        if self.library.msc_binary_add(self.pointer(destination), self.pointer(source), destination.size, self.prime):
            raise ValueError('Native addition refused')

    def scalar_multiply(self, left, right):
        return int(left)*int(right)*self.inverse_R % self.prime

    def broadcast(self, destination, source, scope, output_scope, assignment=None):
        assignment = {} if assignment is None else assignment
        scope, output_scope = tuple(scope), tuple(output_scope)
        if (not set(scope) <= set(output_scope) | set(assignment) or
            set(output_scope) & set(assignment) or len(set(scope)) != len(scope) or
            len(set(output_scope)) != len(output_scope) or any(b not in (0, 1) for b in assignment.values())):
            raise ValueError('Invalid broadcast or fixed-index mapping')
        if destination.size != 1 << len(output_scope) or source.size != 1 << len(scope):
            raise ValueError('Broadcast shapes disagree with scopes')
        # NumPy arrays use C order, so the last named scope bit is the LSB.
        strides = {v: 1 << (len(scope)-i-1) for i, v in enumerate(scope)}
        fixed = sum(strides[v]*b for v, b in assignment.items() if v in strides)
        value, masks = 0, []
        for v in reversed(output_scope):
            value ^= strides.get(v, 0)
            masks.append(value)
        flips = np.array(masks, dtype=np.uint64)
        status = self.library.msc_binary_broadcast(self.pointer(destination), self.pointer(source),
                                                    len(output_scope), source.size, self.pointer(flips), fixed, self.prime)
        if status:
            raise ValueError(f'Native exact binary product refused: {status}')
