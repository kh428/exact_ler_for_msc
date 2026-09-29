"""Linear web constraints and exact local phase defects.

Phases are integer multiples of pi/4. Pauli bits on edge j are x at 2j
and z at 2j+1. Global scalars are checked separately, never guessed from bits.
"""

from .linear import kernel, xor_sum

BITS = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}


def incidence(graph):
    return {v: [j for j, e in enumerate(graph['edges']) if v in (e['u'], e['v'])]
            for v in graph['nodes']}


def constraints(graph, *, require_web=False):
    rows = []
    for v, es in incidence(graph).items():
        node = graph['nodes'][v]
        if node['kind'] == 'boundary':
            continue
        if node['kind'] == 'H':
            if len(es) != 2:
                raise ValueError('A Hadamard must have two legs')
            a, b = es
            rows.extend([(1 << (2*a)) ^ (1 << (2*b+1)),
                         (1 << (2*a+1)) ^ (1 << (2*b))])
            continue
        if node['kind'] not in ('X', 'Z') or not es:
            raise ValueError('Only nonempty X/Z spiders and Hadamards are supported')
        opposite_bit = 0 if node['kind'] == 'Z' else 1
        opposite = [1 << (2*j + opposite_bit) for j in es]
        same = [1 << (2*j + 1-opposite_bit) for j in es]
        rows.extend(opposite[0] ^ x for x in opposite[1:])
        if require_web:
            phase = node['phase'] % 8
            if phase % 2:
                rows.append(opposite[0])
                rows.append(xor_sum(same))
            else:
                rows.append(xor_sum(same) ^ (opposite[0] if phase % 4 else 0))
    return [r for r in rows if r]


def basis(graph, *, require_web=False):
    return kernel(constraints(graph, require_web=require_web), 2*len(graph['edges']))


def highlighted(graph):
    return sum((BITS[e['pauli']][0] << (2*j)) | (BITS[e['pauli']][1] << (2*j+1))
               for j, e in enumerate(graph['edges']))


def edge_paulis(graph, vector):
    return ['IXZY'[(vector >> (2*j)) & 3] for j in range(len(graph['edges']))]


def defects(graph, vector):
    if any((r & vector).bit_count() & 1 for r in constraints(graph)):
        raise ValueError('This labelling is not a semiweb')
    out = {}
    for v, es in incidence(graph).items():
        node = graph['nodes'][v]
        if node['kind'] in ('boundary', 'H'):
            continue
        opposite_bit = 0 if node['kind'] == 'Z' else 1
        o = (vector >> (2*es[0]+opposite_bit)) & 1
        parity = sum((vector >> (2*j+1-opposite_bit)) & 1 for j in es) % 2
        delta = (-2*o*node['phase'] + 4*parity) % 8
        if delta:
            out[v] = delta if delta <= 4 else delta-8
    return out

