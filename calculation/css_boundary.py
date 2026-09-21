"""Exact code-boundary reduction for products of rotated X operators in a CSS code.

Only applies when the input is supported in the specified +1 code space.
The X and Z checks have identical binary supports. This is not a licence to
drop nonnormalizer terms inside a noisy circuit before returning to that boundary.
"""

import math


def span(generators):
    values = [0]
    for generator in generators:
        values += [value ^ generator for value in values]
    if len(set(values)) != len(values):
        raise ValueError("Generators must be independent")
    return values


def rref(rows, n):
    rows = list(rows)
    pivots = []
    for column in range(n):
        pivot = next((i for i in range(len(pivots), len(rows)) if rows[i] >> column & 1), None)
        if pivot is None:
            continue
        k = len(pivots)
        rows[k], rows[pivot] = rows[pivot], rows[k]
        for i in range(len(rows)):
            if i != k and rows[i] >> column & 1:
                rows[i] ^= rows[k]
        pivots.append(column)
    return rows[:len(pivots)], pivots


def nullspace(rows, n):
    reduced, pivots = rref(rows, n)
    basis = []
    for free in range(n):
        if free in pivots:
            continue
        vector = 1 << free
        for row, pivot in zip(reduced, pivots):
            if row >> free & 1:
                vector |= 1 << pivot
        basis.append(vector)
    return basis


def rotated_x_logical(n, checks, angles, normalizer_masks=None):
    """2x2 code restriction of product_q (cos(angle_q) X_q+sin(angle_q) Y_q).

    Logical |0> is the uniform superposition over the binary span of X checks;
    logical |1>=X_all|0>. Enumerate only Z masks commuting with every X check.
    This returns the restriction, even if the physical operator leaks outside
    the code. A separate leakage check is essential before calling it logical.
    """
    masks = span(nullspace(checks, n)) if normalizer_masks is None else normalizer_masks
    upper, lower = [], []
    cosines = [math.cos(angle) for angle in angles]
    sines = [math.sin(angle) for angle in angles]
    for z in masks:
        coefficient = 1.0
        for q in range(n):
            coefficient *= sines[q] if z >> q & 1 else cosines[q]
        w = z.bit_count()
        phase = (1, 1j, -1, -1j)[w % 4]
        lower.append(coefficient * phase)
        upper.append(coefficient * phase * (-1)**w)
    # Sum real and imaginary parts separately with compensated scalar summation.
    def total(values):
        return complex(math.fsum(complex(v).real for v in values),
                       math.fsum(complex(v).imag for v in values))
    return [[0j, total(upper)], [total(lower), 0j]]


def sign_violations(checks, flipped):
    return [i for i, row in enumerate(checks)
            if (row.bit_count() // 2 + (row & flipped).bit_count()) % 2]
