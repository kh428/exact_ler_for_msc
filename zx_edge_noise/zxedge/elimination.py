"""Linear elimination that keeps long parity equations sparse and never removes open variables.

The repository's eliminate_linear_constraints (alternatives/calculation/pauli_polynomial.py) substitutes XOR
constraints into the factors while no factor grows beyond max_factor_bits; when it stops, every remaining parity
equation becomes one dense XOR factor over all of its variables, refused above max_factor_bits.

This version is the same up to that point. A remaining equation v1 + v2 + ... + vm = c is then written as a chain of
three-variable XORs with helper variables, a1 = v1 + v2, a2 = a1 + v3, ..., a(m-2) + vm = c, which is exact and keeps
every factor small. A factor that no substitution touched is kept as it is, whatever its size. Variables in protected
(the open labels of a stage network) are never substituted away.
"""

import numpy as np

from . import REPO  # noqa: F401  (puts the repository on the path)
from calculation.truncated_polynomial import PolynomialNetwork        # noqa: E402


def eliminate(net, *, max_factor_bits=18, max_source_entries=1 << 26, protected=()):
    constant = 1 << len(net.labels)
    variable_mask = constant - 1
    equations, factors = [], []
    original_variables = len(net.variables)
    for f in net.factors:
        nonzero = [i for i, value in enumerate(f.values.flat) if value]
        c = f.values.flat[nonzero[0]] if nonzero else net.ring(0)
        parity = nonzero[0].bit_count() % 2 if nonzero else 0
        if (len(nonzero) * 2 == f.values.size and all(f.values.flat[i] == c for i in nonzero)
                and all((i.bit_count() % 2 == parity) == bool(v) for i, v in enumerate(f.values.flat))):
            equations.append(sum(1 << v for v in f.scope) ^ (constant if parity else 0))
            net.scalar *= c
        else:
            factors.append(([1 << v for v in f.scope], f.values))
    variables = set(net.variables)
    eliminated = 0
    while equations:
        if constant in equations:
            raise ValueError('Inconsistent affine conditions')
        equations = list(dict.fromkeys(e for e in equations if e))
        if not equations:
            break
        supports = []
        for expressions, _ in factors:
            support = 0
            for e in expressions:
                support |= e & variable_mask
            supports.append(support)
        occurrence = {v: [i for i, s in enumerate(supports) if s >> v & 1] for v in variables}
        best = None
        for eq in equations:
            bits = eq & variable_mask
            while bits:
                bit = bits & -bits
                bits ^= bit
                v = bit.bit_length() - 1
                if v in protected:
                    continue
                delta, largest = 0, 0
                for i in occurrence[v]:
                    support = 0
                    for e in factors[i][0]:
                        support |= (e ^ eq if e & bit else e) & variable_mask
                    width = support.bit_count()
                    largest = max(largest, width)
                    delta += (1 << width) - (1 << supports[i].bit_count())
                score = (largest, delta, (eq & variable_mask).bit_count(), v, eq)
                if best is None or score < best:
                    best = score
        if best is None:                          # every equation left is on protected variables only
            break
        largest, _, _, v, eq = best
        if largest > max_factor_bits:
            break
        bit = 1 << v
        factors = [([e ^ eq if e & bit else e for e in expressions], table) for expressions, table in factors]
        equations = [e ^ eq if e & bit else e for e in equations]
        variables.remove(v)
        eliminated += 1
    result = PolynomialNetwork(net.ring)
    result.labels = list(net.labels)
    result.variables = variables
    result.scalar = net.scalar
    source_entries = 0
    for expressions, table in factors:
        support = 0
        for e in expressions:
            support |= e & variable_mask
        scope = tuple(i for i in range(len(net.labels)) if support >> i & 1)
        source_entries += 1 << len(scope)
        untouched = all(not e & constant and (e & variable_mask).bit_count() == 1 for e in expressions)
        if (len(scope) > max_factor_bits and not untouched) or source_entries > max_source_entries:
            raise RuntimeError('Affine substitution exceeds the source-array guard')
        if untouched and [e & variable_mask for e in expressions] == [1 << v for v in scope]:
            result.add(scope, table)                 # nothing substituted: the table as it is
            continue
        vals = np.arange(1 << len(scope), dtype=np.uint64)
        indices = []
        for e in expressions:
            mask = sum(1 << (len(scope) - 1 - j) for j, w in enumerate(scope) if e >> w & 1)
            indices.append((np.bitwise_count(vals & np.uint64(mask)) % 2).astype(np.intp) ^ int(bool(e & constant)))
        result.add(scope, table[tuple(indices)] if expressions else table)
    chained = 0
    for eq in equations:
        scope = [w for w in sorted(variables) if eq >> w & 1]
        parity = int(bool(eq & constant))
        if len(scope) <= 3:
            result.xor(tuple(scope), parity)
            continue
        chained += 1
        acc = scope[0]
        for w in scope[1:-1]:
            nxt = result.variable('xor-chain')
            result.xor((acc, w, nxt), 0)
            acc = nxt
        result.xor((acc, scope[-1]), parity)
    cleanup = result.simplify(protected=[v for v in protected if v in result.variables])
    return result, dict(original_variables=original_variables, eliminated=eliminated,
                        residual_equations=len(equations), chained_equations=chained,
                        expanded_source_entries=source_entries, cleanup=cleanup)
