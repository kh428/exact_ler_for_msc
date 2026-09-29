"""Closed, defect-free Pauli webs of a whole circuit and the record checks they imply.

A closed web has no support on any output wire. Defect-free means every spider
keeps its phase: at a T spider no opposite-colour support and even same-colour
parity, whatever the T angle is. For such a web w the diagram with records r obeys
    D_r = lambda(r) D_r,   lambda(r) = lambda_0 * (-1)^{sum over record spiders with
                                       opposite-colour support of r},
so D_r = 0 unless that record parity equals b, where lambda_0 = (-1)^b. The scalar
lambda_0 uses the package's analytic local scalars plus the transpose sign of each
contracted Y edge, exactly as in src/generate.py.
"""

from src.generate import local_scalar
from src.linear import echelon, kernel, rank
from src.semiweb import BITS, constraints


def incidence(graph):
    inc = {k: [] for k in graph['nodes']}
    for j, e in enumerate(graph['edges']):
        inc[e['u']].append(j)
        inc[e['v']].append(j)
    return inc


def closed_web_basis(graph):
    rows = constraints(graph, require_web=True)
    inc = incidence(graph)
    for k, n in graph['nodes'].items():
        if n['kind'] == 'boundary':
            (j,) = inc[k]
            rows += [1 << (2 * j), 1 << (2 * j + 1)]
    return kernel(rows, 2 * len(graph['edges'])), rows


def labels_at(vector, js):
    return tuple('IXZY'[(vector >> (2 * j)) & 3] for j in js)


def record_equation(graph, inc, vector):
    """(record support bitmask, b) implied by one closed defect-free web."""
    support, exponent = 0, 0
    for k, n in graph['nodes'].items():
        if n['kind'] == 'boundary':
            assert all(((vector >> (2 * j)) & 3) == 0 for j in inc[k]), 'web is not closed'
            continue
        labels = labels_at(vector, inc[k])
        if n['kind'] == 'H':
            a, b = labels
            assert BITS[a] == BITS[b][::-1], 'Hadamard constraint violated'
            exponent += 4 if a == 'Y' else 0          # H Y H = -Y
            continue
        opposite = 0 if n['kind'] == 'Z' else 1
        o = BITS[labels[0]][opposite]
        exponent += local_scalar(n['kind'], n['phase'], labels)
        if 'record' in n and o:
            support ^= 1 << n['record']
    exponent += 4 * sum(((vector >> (2 * j)) & 3) == 3 for j in range(len(graph['edges'])))
    exponent %= 8
    if exponent not in (0, 4):
        raise AssertionError(f'closed web with non-real scalar exponent {exponent}')
    return support, exponent // 4


def web_checks(graph):
    basis, rows = closed_web_basis(graph)
    inc = incidence(graph)
    equations, trivial = [], 0
    for w in basis:
        support, b = record_equation(graph, inc, w)
        if support == 0:
            # A web touching no record must have scalar +1, or the diagram would vanish.
            assert b == 0, 'record-free closed web with scalar -1'
            trivial += 1
        else:
            equations.append((support, b))
    return dict(web_dimension=len(basis), record_free_webs=trivial, equations=equations,
                constraint_rows=len(rows), nodes=len(graph['nodes']), edges=len(graph['edges']))


def affine_rank(equations, nrec):
    return rank([v | (b << nrec) for v, b in equations])


def reduced_basis(equations, nrec):
    """Canonical reduced row-echelon basis, for readable output and comparison."""
    pivots = echelon([v | (b << nrec) for v, b in equations])
    keys = sorted(pivots)
    for p in keys:
        for q in keys:
            if q != p and (pivots[q] >> p) & 1:
                pivots[q] ^= pivots[p]
    out = []
    for p in sorted(pivots, reverse=True):
        row = pivots[p]
        out.append((row & ((1 << nrec) - 1), (row >> nrec) & 1))
    return out
