"""Exact phase-response and compatible-class batch arithmetic.

The prepared inputs contain the already compiled quantum phase form. A batch
recomputes its contribution with integer Gaussian sums; it is not a full series.
"""
import ctypes as ct
from fractions import Fraction
import random
import time
import numpy as np
from calculation.cyclotomic import omega
from .phase_leaf import Kernel, PreparedSource, dyadic_descriptor

def compress(values):
    basis={}
    for original in values:
        mask=original;label=0
        while mask:
            i=mask.bit_length()-1
            if i in basis:
                a,b=basis[i];mask^=a;label^=b
            else:basis[i]=(mask,label^(1<<len(basis)));break
    def encode(mask):
        result=0
        while mask:
            a,b=basis[mask.bit_length()-1];mask^=a;result^=b
        return result
    return [encode(v) for v in values],len(basis)


def histogram_sum(limbs):
    answer = omega(0)*0
    for i in np.nonzero(limbs.reshape((-1, 2)).any(axis=1))[0]:
        v = int(limbs[2*i]) + (int(limbs[2*i+1]) << 64)
        if v >> 127:
            v -= 1 << 128
        answer += dyadic_descriptor(int(i)//8-256, int(i)%8)*v
    return answer


def fraction(value):
    """Require an exactly rational, real C8 value; never discard a component."""
    d = value.exact()
    assert d['imag']['a'] == '0' and d['imag']['b'] == '0', d
    assert d['real']['b'] == '0', d
    return Fraction(int(d['real']['a']), 1 << d['real']['denominator_power_of_two'])


def project_signatures(signatures, bits, seed=166016):
    rng = random.Random(seed)
    rank = max(signatures, default=0).bit_length()
    masks = [rng.getrandbits(rank) for _ in range(bits)]
    values = [sum(((s & m).bit_count() % 2) << i for i, m in enumerate(masks))
              for s in signatures]
    return values, masks


class SixthContext:
    """One immutable prepared source, safely shared by native worker threads."""
    def __init__(self, plan, labels, signatures, powers, library, bits=14):
        self.kernel = Kernel(plan, 'A', library)
        self.source = PreparedSource(self.kernel, labels)
        self.bits = bits
        signatures, self.rank = compress(signatures)
        assert self.rank <= 128
        project, self.masks = project_signatures(signatures, bits)
        ss = np.array([[s & ((1 << 64)-1), s >> 64] for s in signatures], dtype=np.uint64)
        pp = np.array(project, dtype=np.uint32)
        ww = np.array(powers, dtype=np.uint64)
        assert ww.shape == (len(labels), 4)
        u, u32 = ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint32)
        lib = self.kernel.lib
        lib.sixth_make.argtypes = [ct.c_void_p, ct.c_void_p, u, u32, u, ct.c_int, ct.c_int]
        lib.sixth_make.restype = ct.c_void_p
        lib.sixth_zero.argtypes = [ct.c_void_p, ct.c_uint64, u]
        lib.sixth_run.argtypes = [ct.c_void_p, ct.c_int, ct.c_uint32,
                                 ct.c_uint64, ct.c_uint64, ct.c_uint64, u, u]
        lib.sixth_stop.argtypes = [ct.c_void_p]
        lib.sixth_delete.argtypes = [ct.c_void_p]
        self.ptr = lib.sixth_make(self.kernel.ptr, self.source.ptr, ss.ctypes.data_as(u),
                                 pp.ctypes.data_as(u32), ww.ctypes.data_as(u), len(labels), bits)
        if not self.ptr:
            raise ValueError('Unsupported sixth-order source')
        self.zero_counts = None

    def prepare_zero(self, limit=5_000_000):
        assert self.zero_counts is None
        out = np.zeros(2, dtype=np.uint64)
        code = self.kernel.lib.sixth_zero(self.ptr, limit, out.ctypes.data_as(ct.POINTER(ct.c_uint64)))
        if code:
            raise RuntimeError(f'Zero-source guard {code}')
        self.zero_counts = [int(v) for v in out]
        return self.zero_counts

    def run(self, kind, index=0, begin=0, end=0, limit=5_000_000):
        limbs = np.zeros((7, 8192), dtype=np.uint64)
        stats = np.zeros(10, dtype=np.uint64)
        wall, cpu = time.monotonic(), time.thread_time()
        u = ct.POINTER(ct.c_uint64)
        code = self.kernel.lib.sixth_run(self.ptr, kind, index, begin, end, limit,
                                        limbs.ctypes.data_as(u), stats.ctypes.data_as(u))
        if code:
            raise RuntimeError(f'Sixth-order guard {code}: {(kind,index,begin,end)}')
        values = [histogram_sum(row) for row in limbs]
        return dict(kind=kind, index=index, begin=begin, end=end,
                    sums=[str(fraction(v)) for v in values[:4]],
                    lower_sums=[str(fraction(v)) for v in values[4:]],
                    stats=[int(v) for v in stats],
                    core_seconds=time.thread_time()-cpu, wall_seconds=time.monotonic()-wall)

    def stop(self):
        self.kernel.lib.sixth_stop(self.ptr)

    def close(self):
        if getattr(self, 'ptr', None):
            self.kernel.lib.sixth_delete(self.ptr)
            self.ptr = None
        self.source.close()
        self.kernel.close()

