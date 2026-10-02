"""The tensor of the factory diagram, in exact arithmetic.

Every phase is a multiple of pi/4, so each entry is an integer combination of 1, w, w^2 and
w^3, where w = exp(i pi/4) and w^4 = -1. An entry is stored as those four integers. One
positive constant, fixed by the shape of the diagram, is left out of every tensor.
"""

import numpy as np


def times_unit(value, exponent):
    """value * w^exponent. The last axis of value holds the four coefficients."""
    value = np.asarray(value)
    for _ in range(exponent % 8):
        value = np.concatenate([-value[..., 3:], value[..., :3]], axis=-1)
    return value


def product(a, b):
    """Product of two entries."""
    out = [0, 0, 0, 0]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[(i + j) % 4] += int(x) * int(y) * (-1 if i + j >= 4 else 1)
    return out


def proportional(u, v):
    """Whether two states of the output qubit are multiples of one another."""
    return product(u[0], v[1]) == product(u[1], v[0])


class Tensor:
    """The tensor on the output leg, by a sum over one bit for each wire and each gadget.

    Z spiders joined by a wire share one bit. A Z spider of phase a contributes w^a when its
    bit is 1. An X spider of phase 0 keeps the assignments in which its legs have even parity.
    """

    def __init__(self, g):
        nodes = g['nodes']
        parent = {k: k for k, n in nodes.items() if n['kind'] != 'X'}

        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        legs = {k: [] for k, n in nodes.items() if n['kind'] == 'X'}
        for e in g['edges']:
            a, b = e['u'], e['v']
            if a in parent and b in parent:
                parent[find(a)] = find(b)
            elif a in parent or b in parent:
                hub, other = (b, a) if a in parent else (a, b)
                legs[hub].append(other)
            else:
                raise ValueError('An edge joins two X spiders')
        classes = sorted({find(k) for k in parent})
        index = {c: i for i, c in enumerate(classes)}
        assign = np.arange(1 << len(classes))
        bit = {k: (assign >> index[find(k)]) & 1 for k in parent}
        keep = np.ones(len(assign), dtype=bool)
        for hub, others in legs.items():
            if nodes[hub]['phase'] % 8:
                raise ValueError('Only X spiders of phase 0 are supported')
            parity = np.zeros(len(assign), dtype=np.int64)
            for other in others:
                parity ^= bit[other]
            keep &= parity == 0
        self.g = g
        self.variables = len(classes)
        self.assignments = int(keep.sum())
        self.bit = {k: v[keep] for k, v in bit.items()}
        self.outputs = sorted(k for k, n in nodes.items() if n['kind'] == 'boundary')
        self.key = sum(self.bit[k] << i for i, k in enumerate(self.outputs))

    def __call__(self, records=0, phases=None, shift=None):
        """Tensor for one record pattern. `phases` replaces the T phases, `shift` adds to phases."""
        exponent = np.zeros(self.assignments, dtype=np.int64)
        for k, n in self.g['nodes'].items():
            if n['kind'] != 'Z':
                continue
            phase = n['phase']
            if phases is not None and 'site' in n:
                phase = phases[n['site']]
            if shift:
                phase += shift.get(k, 0)
            if 'record' in n:
                phase += 4 * ((records >> n['record']) & 1)
            if phase % 8:
                exponent += (phase % 8) * self.bit[k]
        size = 1 << len(self.outputs)
        counts = np.bincount(self.key * 8 + exponent % 8, minlength=8 * size).reshape(size, 8)
        return counts[:, :4] - counts[:, 4:]

    def live(self, phases=None):
        """The record patterns that occur: those whose tensor is not zero."""
        return [r for r in range(16) if self(r, phases).any()]


def fixed_checks(live):
    """The parities of the four records that take one value on every pattern that occurs."""
    return {c for c in range(1, 16) if len({(c & r).bit_count() & 1 for r in live}) == 1}
