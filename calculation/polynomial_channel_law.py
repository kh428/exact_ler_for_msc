"""Exact physical-weight Pauli-label polynomials and native endpoint operations."""
from collections import Counter
import ctypes
from fractions import Fraction
import math
import time
import numpy as np
from .character_surrogate import canonical_basis
from .native_modular_polynomial import PolynomialField, PTR, U
from .rational_binary_network import residue


class EndpointField(PolynomialField):
    def __init__(self, ring, build_run):
        super().__init__(ring, build_run)
        self.lib.msc_endpoint_walsh.argtypes = [PTR, ctypes.c_size_t, ctypes.c_uint,
                                                ctypes.c_uint, U, ctypes.c_int]
        self.lib.msc_endpoint_walsh.restype = ctypes.c_int
        self.lib.msc_endpoint_first_mix.argtypes = [PTR, PTR, ctypes.POINTER(ctypes.c_int32),
                                                     ctypes.POINTER(ctypes.c_uint32), PTR,
                                                     ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, U]
        self.lib.msc_endpoint_first_mix.restype = ctypes.c_int

    def walsh(self, array, bits, inverse=False):
        width = array.shape[-1]
        assert 1 <= width <= 11 and array.size % width == 0
        status = self.lib.msc_endpoint_walsh(self.pointer(array), array.size//width,
                                              width, bits, self.prime, int(inverse))
        if status:
            raise ValueError('Polynomial endpoint Walsh rejected inputs: '+str(status))

    def first_mix(self, prefix, check, tags, syndromes, output, first):
        width = self.ring.degree+1
        assert prefix.shape == (256, width) and check.shape == (4096, width)
        assert output.shape == (8, 8, 4, width)
        assert tags.dtype == np.int32 and tags.flags.c_contiguous and tags.shape[1:] == (4096, 8)
        assert syndromes.dtype == np.uint32 and syndromes.shape == (4096,) and syndromes.flags.c_contiguous
        status = self.lib.msc_endpoint_first_mix(self.pointer(prefix), self.pointer(check),
                  tags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                  syndromes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), self.pointer(output),
                  first, len(tags), width, self.prime)
        if status:
            raise ValueError('Polynomial first mixture rejected inputs: '+str(status))


def polynomial_law(channels, bits, ring, field, selected=None):
    """Complete coefficient laws, then optionally select accepted boundary labels.

    Zero characters carry (1+x)^N. Nontrivial damping characters carry
    (1+(1-c)x), with c=4/3,16/15,2. Equal profiles are only an evaluation
    cache; all Fourier characters are included in the inverse transform.
    """
    if not 0 <= bits <= 18:
        raise ValueError('Endpoint label-law cap is 18 bits')
    began = time.perf_counter()
    n, K, q = len(channels), ring.degree, ring.prime
    kinds = ('depol1', 'depol2', 'flip')
    groups = Counter((c['kind'], canonical_basis(c['labels'])) for c in channels)
    characters = np.arange(1 << bits, dtype=np.uint64)
    counts = np.zeros((len(characters), 3), dtype=np.int32)
    for (kind, basis), multiplicity in groups.items():
        assert all(0 <= v < 1 << bits for v in basis)
        active = np.zeros(len(characters), dtype=bool)
        for v in basis:
            active |= (np.bitwise_count(characters & np.uint64(v)) % 2).astype(bool)
        counts[:, kinds.index(kind)] += multiplicity*active
    unique, inverse = np.unique(counts, axis=0, return_inverse=True)
    slopes = [residue(v,q) for v in (Fraction(-1,3),Fraction(-1,15),Fraction(-1),Fraction(1))]
    cache = {}
    def power(index, count):
        key = index, count
        if key not in cache:
            cache[key] = ring.coefficients(math.comb(count,k)*pow(slopes[index],k,q)
                                           for k in range(min(K,count)+1))
        return cache[key]
    profiles = np.zeros((len(unique),K+1),dtype=np.uint64)
    for at, row in enumerate(unique):
        powers = [int(v) for v in row]
        powers.append(n-sum(powers))
        assert powers[-1] >= 0
        value = ring(1)
        for kind, exponent in enumerate(powers):
            value *= power(kind, exponent)
        profiles[at,:len(value.coefficients)] = value.coefficients
    values = np.ascontiguousarray(profiles[inverse])
    # Walsh addition and inverse scaling work on ordinary residues too.
    field.walsh(values,bits,inverse=True)
    for k in range(K+1):
        assert sum(map(int,values[:,k])) % q == (math.comb(n,k) % q if k <= n else 0)
    output = values if selected is None else np.ascontiguousarray(values[np.asarray(selected,dtype=np.int64)])
    return output, {'physical_channels': n, 'label_bits': bits, 'degree': K,
                    'character_profiles': len(profiles), 'all_characters': len(characters),
                    'normalization_coefficients_checked': K+1, 'selected_entries': len(output),
                    'seconds': time.perf_counter()-began,
                    'arithmetic': 'All Fourier characters, exact truncated physical-weight polynomials modulo q'}
