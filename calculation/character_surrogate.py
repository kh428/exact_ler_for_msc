"""Compile exact uniform-noise polynomials from a coherent Pauli response.

The response depends on data Q_X, Q_Z modulo Z stabilizers, ancilla Z flips,
and one check-sign bit. A Walsh transform keeps all interference inside that
response and averages independent physical channels in the character basis.
For DEPOLARIZE2(p), every nontrivial character has eigenvalue q=1-16p/15.
Grouping characters by their number of active locations gives A(q), B(q).
There is no fault-weight or Pauli-path truncation. The flat transform is
explicitly limited to small label spaces; it is not proposed for 49 bits.
"""

from collections import Counter
from fractions import Fraction
import math
import time
import numpy as np
from .css_boundary import span
from .dyadic import make
from .fault_sets import gf2_rank


def parity_preimages(rows, n):
    """One linear right inverse for z -> (row . z)_row over GF(2)."""
    generators = []
    for target in range(len(rows)):
        augmented = [[row, int(i == target)] for i, row in enumerate(rows)]
        pivots = []
        for col in range(n):
            pivot = next((i for i in range(len(pivots), len(rows)) if augmented[i][0] >> col & 1), None)
            if pivot is None:
                continue
            k = len(pivots)
            augmented[k], augmented[pivot] = augmented[pivot], augmented[k]
            for i in range(len(rows)):
                if i != k and augmented[i][0] >> col & 1:
                    augmented[i][0] ^= augmented[k][0]
                    augmented[i][1] ^= augmented[k][1]
            pivots.append(col)
        if len(pivots) != len(rows):
            raise ValueError("Parity rows are not independent")
        solution = sum(augmented[i][1] << col for i, col in enumerate(pivots))
        assert all((solution & row).bit_count() % 2 == int(i == target) for i, row in enumerate(rows))
        generators.append(solution)
    return span(generators)


def canonical_basis(vectors):
    rows = {}
    for value in vectors:
        while value:
            pivot = value.bit_length()-1
            if pivot in rows:
                value ^= rows[pivot]
            else:
                rows[pivot] = value
                break
    for pivot in sorted(rows):
        for other in rows:
            if other != pivot and rows[other] >> pivot & 1:
                rows[other] ^= rows[pivot]
    return tuple(rows[p] for p in sorted(rows, reverse=True))


def walsh_inplace(values):
    width = 1
    while width < len(values):
        blocks = values.reshape(-1, 2*width)
        left = blocks[:, :width].copy()
        blocks[:, :width] = left + blocks[:, width:]
        blocks[:, width:] = left - blocks[:, width:]
        width *= 2


def compile_surrogate(model, boundary, channels, initial_phase=7, max_bits=22):
    started = time.perf_counter()
    n = boundary.n
    z_rows = boundary.checks + (boundary.all,)
    nz = len(z_rows)
    ancillas = model.spec["ancillas"]
    bits = n+nz+len(ancillas)+1
    if bits > max_bits:
        raise ValueError(f"Flat character space has {bits} bits; exceeds the declared {max_bits}-bit pilot limit")
    if model.spec["frame"] != (0, 0) or model.spec["effective_sign"] != 1:
        raise ValueError("This compiler expects the zero central record and clean reference frame")
    def label(signature):
        fx, fz, flip = signature
        x, z = model.data_masks(fx, fz)
        value = x | (sum(((z & row).bit_count() % 2) << i for i, row in enumerate(z_rows)) << n)
        value |= sum(((fz >> q) & 1) << (n+nz+i) for i, q in enumerate(ancillas))
        return value | (flip << (bits-1))
    response = []
    words = boundary.words
    z_preimages = parity_preimages(z_rows, n)
    for x in (*words, *(word ^ boundary.all for word in words)):
        for z_label, z in enumerate(z_preimages):
            for flip in (0, 1):
                a, b = boundary.instrument(x, z, -1 if flip else 1, initial_phase)
                index = x | (z_label << n) | (flip << (bits-1))
                response.append((index, a, b))
    denominator = max(value.k for _, a, b in response for value in (a, b))
    length = 1 << bits
    arrays = [np.zeros(length, dtype=np.int64) for _ in range(4)]
    for index, a, b in response:
        for offset, value in ((0, a), (2, b)):
            arrays[offset][index] = value.a << (denominator-value.k)
            arrays[offset+1][index] = value.b << (denominator-value.k)
    # Integer overflow is excluded before either the transform or final sums.
    bounds = [sum(abs(int(value)) for value in values) for values in arrays]
    if max(bounds)*length >= 2**62:
        raise ValueError("Integer transform bound exceeds the conservative int64 budget")
    groups = Counter()
    all_columns = []
    for channel in channels:
        columns = tuple(label(channel["outcomes"][code-1]["signature"]) for code in (1, 2, 4, 8))
        # Verify the 15-outcome channel's linear label map in full, rather
        # than assuming it from four chosen columns.
        for code, entry in enumerate(channel["outcomes"], 1):
            expected = 0
            for i, column in enumerate(columns):
                if code >> i & 1:
                    expected ^= column
            assert label(entry["signature"]) == expected
        groups[canonical_basis(columns)] += 1
        all_columns.extend(columns)
    characters = np.arange(length, dtype=np.uint64)
    active_count = np.zeros(length, dtype=np.int16)
    for basis, multiplicity in groups.items():
        active = np.zeros(length, dtype=bool)
        for column in basis:
            active |= (np.bitwise_count(characters & np.uint64(column)) % 2).astype(bool)
        active_count += multiplicity * active
    coefficient_arrays = []
    for values in arrays:
        walsh_inplace(values)
        coefficients = np.zeros(len(channels)+1, dtype=np.int64)
        np.add.at(coefficients, active_count, values)
        coefficient_arrays.append(coefficients)
    polynomials = {}
    for name, offset in (("A", 0), ("B", 2)):
        polynomials[name] = [make(int(coefficient_arrays[offset][k]), int(coefficient_arrays[offset+1][k]), bits+denominator)
                             for k in range(len(channels)+1)]
    return polynomials, {"label_bits": bits, "measured_GF2_noise_label_rank": gf2_rank(all_columns),
                         "characters": length, "coherent_response_entries_evaluated": len(response),
                         "physical_channels": len(channels), "distinct_channel_character_subspaces": len(groups),
                         "response_common_denominator_power": denominator,
                         "integer_sum_bounds": bounds, "integer_accumulation_bound": max(bounds)*length,
                         "polynomial_variable": "q = 1 - 16 p / 15", "seconds": time.perf_counter()-started,
                         "label_definition": "data X mask; data Z parities against CSS checks and Z_L; ancilla Z flips; central flip"}


def evaluate(coefficients, p):
    """Return the exact rational and sqrt(2) coefficients at rational p."""
    q = 1-Fraction(16, 15)*p
    a = b = Fraction(0)
    for coefficient in reversed(coefficients):
        a = a*q + Fraction(coefficient.a, 1 << coefficient.k)
        b = b*q + Fraction(coefficient.b, 1 << coefficient.k)
    return a, b


def taylor(coefficients, order):
    a = b = Fraction(0)
    for degree, coefficient in enumerate(coefficients):
        if degree >= order:
            factor = math.comb(degree, order)*Fraction(-16, 15)**order
            a += Fraction(coefficient.a, 1 << coefficient.k)*factor
            b += Fraction(coefficient.b, 1 << coefficient.k)*factor
    return a, b
