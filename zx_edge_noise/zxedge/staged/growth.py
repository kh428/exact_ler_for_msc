"""Labels of the growth faults, their leaves in the paper's growth tree, and the leaf tables.

The growth stage (source lines 189 to 387 of the d=5 SOFT file) is Clifford. Every Pauli fault in it has a label in
the 75-bit space of the paper (Appendix E): 49 compatibility bits (a deterministic measurement flipped), then 26
interface bits (input reference X and Z checks, output X and Z faces, logical XX and ZZ). The labels come from the
compiler of the growth Choi instrument that wrote data/models/growth_frame_model.json, here with each channel given
by explicit generator Paulis, so that any Pauli channel (an edge, a source depolariser, a flip) is compiled the same
way; for the source channels it reproduces that file exactly.

The paper's tree has 569 leaves: W, 567 groups of channels with a common label span, and F. A channel of another
noise model is put in a leaf whose span contains its labels; a leaf table is then the XOR convolution of its channels'
laws, each 1 + x (sum_f w_f [label f] + w_0 [0]), a polynomial in x truncated at degree K, indexed as the tree's maps
expect: entry i is the label sum_j bit_j(i) basis[j].
"""
import json
from fractions import Fraction

import numpy as np

from .. import REPO
from . import STABILIZER  # noqa: F401
from calculation.fault_sets import gf2_rank                        # noqa: E402
from pauli import anticommutes                                     # noqa: E402
from support import advance, basis, compress                       # noqa: E402

GROWTH = (189, 388)
FREE_RECORDS = (22, 24, 31, 33)


def stage_model():
    d3 = json.loads((REPO / 'data/inputs/d3_soft_source_noise_20260905.json').read_text())
    d5 = json.loads((REPO / 'data/inputs/d5_soft_source_noise_20260905.json').read_text())
    return d3, d5


def growth_rows(d3, d5, physical_n=42):
    """The Choi reference of growth_frame_model.json: Bell pairs on the d=3 data, outputs as listed there."""
    references = list(range(physical_n, physical_n + len(d3['data'])))
    initial = [(0, 1 << q, 1) for q in range(physical_n) if q not in d3['data']]
    for data, reference in zip(d3['data'], references):
        pair = (1 << data) | (1 << reference)
        initial += [(pair, 0, 1), (0, pair, 1)]
    outputs = []
    for axis in (0, 1):
        for check in d3['checks']:
            mask = sum(1 << references[i] for i in range(len(references)) if check >> i & 1)
            outputs.append((mask, 0, 1) if axis == 0 else (0, mask, 1))
    for axis in (0, 1):
        for face in d5['faces']:
            mask = sum(1 << q for q in face)
            outputs.append((mask, 0, 1) if axis == 0 else (0, mask, 1))
    mask = sum(1 << q for q in references + d5['data'])
    outputs += [(mask, 0, 1), (0, mask, 1)]
    return initial, outputs, references


def generator_paulis(loc):
    """The generators of a location's faults as (x mask, z mask), and its outcome weights by code."""
    if loc['kind'] == 'law':
        gens = []
        for g in loc['generators']:
            (q, s), = g.items()
            gens.append(((1 << q) if s in 'XY' else 0, (1 << q) if s in 'YZ' else 0))
        return gens, dict(loc['weights'])
    q = loc['qubits']
    if loc['kind'] == 'flip':
        a = loc['axis']
        return [((1 << q[0]) if a in 'XY' else 0, (1 << q[0]) if a in 'YZ' else 0)], {1: Fraction(1)}
    if loc['kind'] == 'depolarizing':
        gens = [g for v in q for g in (((1 << v), 0), (0, (1 << v)))]
        n = (1 << len(gens)) - 1
        return gens, {c: Fraction(1, n) for c in range(1, n + 1)}
    if loc['kind'] == 'record':
        return [None], {1: Fraction(1)}
    raise ValueError(loc['kind'])


def events(circuit, locations):
    """The circuit as compiler events, each location right after the gate it follows."""
    after = {}
    for i, loc in enumerate(locations):
        after.setdefault(loc['after'], []).append(i)
    ev, lines = [], []

    def noise(g):
        for i in after.get(g, ()):
            ev.append(('NOISE', i))
            lines.append(locations[i]['line'])
    noise(-1)
    for g, gate in enumerate(circuit['gates']):
        op, line = gate['op'], gate['line']
        if op in ('R', 'RX'):
            ev.append(('RESET', gate['q'], 'Z' if op == 'R' else 'X', 1))
        elif op in ('M', 'MX'):
            ev.append(('MEASURE', gate['q'], 'Z' if op == 'M' else 'X', gate['record']))
        elif op == 'MPP':
            x = sum(1 << q for q in gate['qubits']) if gate['pauli'] in 'XY' else 0
            z = sum(1 << q for q in gate['qubits']) if gate['pauli'] in 'YZ' else 0
            ev.append(('MEASURE_PAULI', x, z, gate['record']))
        elif op == 'CX':
            ev.append(('CX', gate['c'], gate['t']))
        elif op == 'FEEDBACK':
            ev.append(('FEEDBACK', gate['axis'], gate['q'], gate['record']))
        elif op in ('H', 'S', 'S_DAG', 'T', 'T_DAG', 'X', 'Y', 'Z'):
            ev.append((op, gate['q']))
        else:
            raise ValueError(op)
        lines.append(line)
        noise(g)
    return ev, lines


def xor_all(values):
    result = 0
    for value in values:
        result ^= value
    return result


def compile_growth(circuit, locations, *, n=49):
    """The 75-bit label of every outcome of every growth location (any Pauli channel).

    Returns (channels, model): channels is a list of dicts with the location index, generator labels, outcome labels
    by code and weights; model has the constraint and output forms and the audit of the Choi reference.
    """
    d3, d5 = stage_model()
    initial_rows, output_rows, _ = growth_rows(d3, d5)
    start_line, stop_line = GROWTH
    state = basis(initial_rows, n)
    assert not state['zero'] and len(state['rows']) == n
    x, z = [0] * n, [0] * n
    bit_count = len(FREE_RECORDS)
    record_forms = {record: 1 << i for i, record in enumerate(FREE_RECORDS)}
    channels, by_location, readout = [], {}, {}
    for i, loc in enumerate(locations):
        if not start_line <= loc['line'] < stop_line:
            continue
        gens, weights = generator_paulis(loc)
        item = dict(location=i, gens=gens, weights=weights, columns=list(range(bit_count, bit_count + len(gens))))
        bit_count += len(gens)
        channels.append(item)
        by_location[i] = item
        if loc['kind'] == 'record':
            readout[loc['record']] = readout.get(loc['record'], 0) ^ (1 << item['columns'][0])
    constraints, audit, random_count = [], [], 0
    ev, lines = events(circuit, locations)
    for event, line in zip(ev, lines):
        if not start_line <= line < stop_line:
            continue
        name, *args = event
        if name == 'NOISE':
            item = by_location.get(args[0])
            if item is None or locations[args[0]]['kind'] == 'record':
                continue
            for (px, pz), col in zip(item['gens'], item['columns']):
                for q in range(n):
                    if px >> q & 1:
                        x[q] ^= 1 << col
                    if pz >> q & 1:
                        z[q] ^= 1 << col
            continue
        if name == 'CX':
            c, t = args
            x[t] ^= x[c]
            z[c] ^= z[t]
        elif name == 'H':
            q, = args
            x[q], z[q] = z[q], x[q]
        elif name in ('S', 'S_DAG'):
            q, = args
            z[q] ^= x[q]
        elif name == 'FEEDBACK':
            axis, q, record = args
            value = record_forms.get(record, 0)
            if axis in 'XY':
                x[q] ^= value
            if axis in 'YZ':
                z[q] ^= value
            continue
        elif name == 'RESET':
            q, axis, sign = args
            fixed = []
            for candidate in 'XYZ':
                key = ((1 << q) if candidate in 'XY' else 0, (1 << q) if candidate in 'YZ' else 0)
                if compress({key: 1}, state, n) in ({(0, 0): 1}, {(0, 0): -1}):
                    fixed.append(candidate)
            if not fixed:
                raise ValueError((line, q, 'Reset frame erasure requires a product Choi site'))
            x[q] = z[q] = 0
        elif name in ('MEASURE', 'MEASURE_PAULI'):
            if name == 'MEASURE':
                q, axis, record = args
                key = ((1 << q) if axis in 'XY' else 0, (1 << q) if axis in 'YZ' else 0)
            else:
                px, pz, record = args
                key = (px, pz)
            commutator = xor_all(z[q] for q in range(n) if key[0] >> q & 1) ^ \
                xor_all(x[q] for q in range(n) if key[1] >> q & 1)
            delta = commutator ^ record_forms.get(record, 0) ^ readout.get(record, 0)
            pivots = [g for g in state['rows'] if anticommutes(g[:2], key)]
            if pivots:
                pivot = min(pivots, key=lambda g: ((g[0] | g[1]).bit_count(), g))
                for j in range(n):
                    if pivot[0] >> j & 1:
                        x[j] ^= delta
                    if pivot[1] >> j & 1:
                        z[j] ^= delta
                random_count += 1
                audit.append(dict(line=line, record=record, classification='random_half'))
            else:
                if compress({key: 1}, state, n) != {(0, 0): 1}:
                    raise ValueError((line, record, 'The chosen zero reference record is impossible'))
                constraints.append(delta)
                audit.append(dict(line=line, record=record, classification='deterministic_constraint'))
            event = ('POST', *key, 1)
        elif name not in ('X', 'Y', 'Z'):
            raise ValueError((line, 'Non-Clifford or unsupported Choi operation', event))
        state = advance(state, event, n)
        if state['zero'] or len(state['rows']) != n:
            raise ValueError('Choi reference lost purity or became zero')
    output_forms = []
    for row in output_rows:
        if compress({row[:2]: row[2]}, state, n) != {(0, 0): 1}:
            raise ValueError((row, 'Output generator does not support the reference Choi state'))
        output_forms.append(xor_all(z[q] for q in range(n) if row[0] >> q & 1) ^
                            xor_all(x[q] for q in range(n) if row[1] >> q & 1))
    forms = constraints + output_forms
    columns = [sum(((form >> j) & 1) << i for i, form in enumerate(forms)) for j in range(bit_count)]
    for item in channels:
        gl = [columns[j] for j in item['columns']]
        item['generator_labels'] = gl
        item['labels'] = {code: xor_all(g for j, g in enumerate(gl) if code >> j & 1) for code in item['weights']}
    model = dict(constraint_bits=len(constraints), output_bits=len(output_forms), label_bits=len(forms),
                 random_measurements=random_count, noise_label_rank=gf2_rank(columns[len(FREE_RECORDS):]),
                 free_record_labels=columns[:len(FREE_RECORDS)], audit=audit)
    return channels, model


def leaf_groups():
    """The 569 leaves of the paper's tree with their label bases (calculation/point.py of the repository)."""
    from calculation.point import growth_groups
    return growth_groups()


def reduce(vector, rows):
    """vector reduced against an echelon dict {pivot: row}; 0 when it is in the span."""
    while vector:
        h = vector.bit_length() - 1
        if h not in rows:
            return vector
        vector ^= rows[h]
    return 0


def echelon(vectors):
    rows = {}
    for v in vectors:
        v = reduce(v, rows)
        if v:
            rows[v.bit_length() - 1] = v
    return rows


def coordinates(basis_vectors, vector):
    """The bits c with vector = sum_j c_j basis[j] (basis linearly independent)."""
    rows, combos = {}, {}
    for j, b in enumerate(basis_vectors):
        v, c = b, 1 << j
        while v:
            h = v.bit_length() - 1
            if h not in rows:
                rows[h], combos[h] = v, c
                break
            v ^= rows[h]
            c ^= combos[h]
    v, c = vector, 0
    while v:
        h = v.bit_length() - 1
        if h not in rows:
            raise ValueError('not in the span')
        v ^= rows[h]
        c ^= combos[h]
    return c


def assign(channels, groups):
    """Put each channel with nonzero labels in the smallest leaf whose span contains all its labels."""
    spans = [(len(g['basis']), f, echelon(g['basis'])) for f, g in enumerate(groups) if 0 < f < len(groups) - 1]
    spans.sort()
    placed, homeless, identity = {}, [], []
    for item in channels:
        labels = [v for v in item['labels'].values() if v]
        if not labels:
            identity.append(item)
            continue
        for _, f, rows in spans:
            if all(reduce(v, rows) == 0 for v in labels):
                placed.setdefault(f, []).append(item)
                break
        else:
            homeless.append(item)
    return placed, homeless, identity


def leaf_table(basis_vectors, items, degree, prime):
    """The polynomial table of a leaf, shape (K+1, 2^r) mod prime, entry i = label sum_j bit_j(i) basis[j]."""
    r = len(basis_vectors)
    n = 1 << r
    K = degree
    chars = np.arange(n, dtype=np.int64)
    spectrum = np.zeros((K + 1, n), dtype=object)
    spectrum[0, :] = 1
    for item in items:
        w0 = 1 - sum(item['weights'].values())
        factor = np.full(n, w0, dtype=object)
        for code, w in item['weights'].items():
            c = coordinates(basis_vectors, item['labels'][code])
            sign = np.array([(-1) ** bin(c & int(ch)).count('1') for ch in chars], dtype=object)
            factor = factor + w * sign
        factor = np.array([int(Fraction(v).numerator) * pow(Fraction(v).denominator, -1, prime) % prime
                           for v in factor], dtype=object)
        new = spectrum.copy()
        for k in range(1, K + 1):
            new[k] = (spectrum[k] + spectrum[k - 1] * factor) % prime
        spectrum = new
    table = np.zeros((K + 1, n), dtype=np.uint64)
    inv = pow(n, -1, prime)
    for k in range(K + 1):
        values = [int(v) for v in spectrum[k]]
        h = 1
        while h < n:                               # Walsh transform (its own inverse up to 1/n)
            for i in range(0, n, 2 * h):
                for j in range(i, i + h):
                    a, b = values[j], values[j + h]
                    values[j], values[j + h] = (a + b) % prime, (a - b) % prime
            h *= 2
        table[k] = [v * inv % prime for v in values]
    return table
