"""Closed webs of the factory diagram, their records and their scalars."""

from src.generate import local_scalar
from src.linear import kernel
from src.semiweb import BITS, constraints

from .diagram import incidence


def closed_webs(g):
    """Basis of the closed, defect-free webs: no label on the output leg."""
    rows = constraints(g, require_web=True)
    inc = incidence(g)
    for k, n in g['nodes'].items():
        if n['kind'] == 'boundary':
            (j,) = inc[k]
            rows += [1 << (2 * j), 1 << (2 * j + 1)]
    return kernel(rows, 2 * len(g['edges']))


def labels_at(vector, js):
    return tuple('IXZY'[(vector >> (2 * j)) & 3] for j in js)


def labelling(g, overlay):
    """The vector of a web drawn by hand: one Pauli for each labelled edge."""
    vector = 0
    for j, e in enumerate(g['edges']):
        x, z = BITS[overlay.get(frozenset((e['u'], e['v'])), 'I')]
        vector |= (x << (2 * j)) | (z << (2 * j + 1))
    return vector


def scalar_exponent(g, inc, vector):
    """lambda_0 as a power of exp(i pi/4): the local scalars, and a sign for each contracted Y."""
    exponent = 0
    for k, n in g['nodes'].items():
        if n['kind'] != 'boundary':
            exponent += local_scalar(n['kind'], n['phase'], labels_at(vector, inc[k]))
    for j, e in enumerate(g['edges']):
        roles = (g['nodes'][e['u']]['role'], g['nodes'][e['v']]['role'])
        if ((vector >> (2 * j)) & 3) == 3 and 'output' not in roles:
            exponent += 4
    return exponent % 8


def record_support(g, inc, vector):
    """The records a web reads, as a bit mask: those whose measurement leg carries X or Y."""
    support = 0
    for k, n in g['nodes'].items():
        if 'record' in n and BITS[labels_at(vector, inc[k][:1])[0]][0]:
            support ^= 1 << n['record']
    return support


def by_records(g, basis):
    """Split a web basis into one web for each record pivot, and the webs that read no record."""
    inc = incidence(g)
    pivots, free = {}, []
    for w in basis:
        s = record_support(g, inc, w)
        while s:
            top = s.bit_length() - 1
            if top not in pivots:
                pivots[top] = (s, w)
                break
            s ^= pivots[top][0]
            w ^= pivots[top][1]
        if not s:
            free.append(w)
    return pivots, free


def solve(pivots, target):
    """The web that reads exactly the records in `target`."""
    s, w = target, 0
    while s:
        top = s.bit_length() - 1
        if top not in pivots:
            raise ValueError('No web reads these records')
        s ^= pivots[top][0]
        w ^= pivots[top][1]
    return w
