"""Frozen double-checking stages, their ZX graphs and semiweb generators."""

import hashlib
from pathlib import Path
import re

from .linear import echelon, kernel
from .semiweb import BITS, constraints, highlighted, incidence

ROOT = Path(__file__).resolve().parents[1]


def read_stage(distance):
    filename = {3: 'd3_native_t.txt',
                5: 'd5_native_t.stim'}[distance]
    path = ROOT / 'circuits' / filename
    lines = path.read_text().splitlines()
    start_text = {3: 'RX 14 11 6 2 8 1',
                  5: 'RX 24 31 2 6 8 10 20 22 39 1 12 18 37 33 14 26 28 30 41'}[distance]
    start = lines.index(start_text)
    groups = []
    for i, line in enumerate(lines[start:], start):
        match = re.fullmatch(r'(RX|T_DAG|T|CX|MX)(?:\([^)]*\))? (.*)', line)
        if not match:
            continue
        name, operands = match.groups()
        qs = list(map(int, operands.split()))
        if name in ('T', 'T_DAG'):
            if groups and groups[-1]['name'] == 'PHASE':
                group = groups[-1]
            else:
                group = dict(name='PHASE', phases={}, source_lines=[])
                groups.append(group)
            assert not set(qs) & group['phases'].keys()
            group['phases'].update({q: 1 if name == 'T' else -1 for q in qs})
            group['source_lines'].append(i + 1)
        else:
            groups.append(dict(name=name, support=qs, source_lines=[i + 1]))
        if name == 'MX' and len(qs) == len(groups[0]['support']):
            break
    data = sorted(groups[1]['phases'])
    ancillas = sorted(groups[0]['support'])
    center = next(g['support'][0] for g in groups if g['name'] == 'MX')
    assert set(data).isdisjoint(ancillas)
    assert set(groups[-1]['support']) == set(ancillas)
    assert groups[-2]['phases'] == {q: -v for q, v in groups[1]['phases'].items()}
    assert len(data) == {3: 7, 5: 19}[distance]
    return dict(distance=distance, groups=groups, data=data, ancillas=ancillas,
                wires=sorted(data + ancillas), center=center,
                source_file=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_lines=[start + 1, groups[-1]['source_lines'][-1]])


def graph_for(stage, incoming=None):
    """Canonical circuit-oriented semiweb for an incoming Pauli string.

    A Pauli at a measured wire ends there; fresh preparations start with I.
    Measurement-sign defects are retained, rather than interpreted as passes.
    """
    incoming = incoming or {}
    nodes, edges, last = {}, [], {}
    frame = {q: BITS[incoming.get(q, 'I')] for q in stage['wires']}
    ys = {q: -i for i, q in enumerate(stage['wires'])}
    def add(kind, phase, q, column, role):
        key = f'n{len(nodes)}'
        nodes[key] = dict(kind=kind, phase=phase, qubit=q, x=column,
                          y=ys[q], role=role)
        return key
    def join(a, b, bits):
        pauli = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[bits]
        edges.append(dict(u=a, v=b, pauli=pauli))
    for q in stage['data']:
        last[q] = add('boundary', 0, q, 0, 'input')
    center_measure = next(i for i, g in enumerate(stage['groups']) if g['name'] == 'MX')
    for gi, g in enumerate(stage['groups']):
        name = g['name']
        if name == 'RX':
            for q in g['support']:
                assert q not in last
                last[q] = add('Z', 0, q, gi, 'prepare')
                frame[q] = (0, 0)
        elif name == 'PHASE':
            for q, phase in sorted(g['phases'].items()):
                node = add('Z', phase, q, gi, 'entry' if gi < center_measure else 'exit')
                join(last[q], node, frame[q])
                last[q] = node
        elif name == 'CX':
            for c, t in zip(g['support'][::2], g['support'][1::2]):
                control, target = add('Z', 0, c, gi, 'control'), add('X', 0, t, gi, 'target')
                xc, zc = frame[c]; xt, zt = frame[t]
                join(last[c], control, frame[c]); join(last[t], target, frame[t])
                join(control, target, (xc, zt))
                frame[c] = (xc, zc ^ zt); frame[t] = (xt ^ xc, zt)
                last[c], last[t] = control, target
        elif name == 'MX':
            for q in g['support']:
                node = add('Z', 0, q, gi, 'measure')
                join(last.pop(q), node, frame[q])
        else:
            raise ValueError(name)
    for q in stage['data']:
        node = add('boundary', 0, q, len(stage['groups']) - 1, 'output')
        join(last.pop(q), node, frame[q])
    assert not last
    return dict(nodes=nodes, edges=edges)


def boundary_constraints(graph, roles):
    rows = []
    for node, es in incidence(graph).items():
        if graph['nodes'][node]['role'] in roles:
            assert len(es) == 1
            rows.extend([1 << (2 * es[0]), 1 << (2 * es[0] + 1)])
    return rows


def full_basis(stage, graph):
    rows = constraints(graph)
    nbits = 2 * len(graph['edges'])
    input_rows = boundary_constraints(graph, {'input'})
    closed_rows = boundary_constraints(graph, {'input', 'output'})
    interior = kernel(rows + closed_rows, nbits)
    closed_webs = kernel(constraints(graph, require_web=True) + closed_rows, nbits)
    # Prefer actual defect-free webs first in the internal part of the basis.
    internal = list(closed_webs)
    pivots = echelon(internal)
    def extend(v, pivots):
        residue = v
        while residue:
            top = residue.bit_length() - 1
            if top not in pivots:
                pivots[top] = residue
                return True
            residue ^= pivots[top]
        return False
    for v in sorted(interior, key=lambda w: (w.bit_count(), w)):
        if extend(v, pivots):
            internal.append(v)
    incoming = []
    for q in stage['data']:
        for pauli in ('X', 'Z'):
            incoming.append(dict(kind='incoming', label=f'{pauli}{q}',
                                 vector=highlighted(graph_for(stage, {q: pauli}))))
    pivots = echelon([r['vector'] for r in incoming] + internal)
    boundary = []
    for v in sorted(kernel(rows + input_rows, nbits), key=lambda w: (w.bit_count(), w)):
        if extend(v, pivots):
            boundary.append(dict(kind='output', label=f'Output {len(boundary) + 1}', vector=v))
    closed = [dict(kind='internal', label=f'Internal {i + 1}', vector=v)
              for i, v in enumerate(internal)]
    records = incoming + boundary + closed
    expected = nbits - len(echelon(rows))
    assert len(records) == len(echelon(r['vector'] for r in records)) == expected
    for r in records:
        assert all(not ((row & r['vector']).bit_count() & 1) for row in rows)
        if r['kind'] in ('output', 'internal'):
            assert all(not ((row & r['vector']).bit_count() & 1) for row in input_rows)
        if r['kind'] == 'internal':
            assert all(not ((row & r['vector']).bit_count() & 1) for row in closed_rows)
    return records, dict(semiweb_dimension=expected,
                         pauli_web_dimension=nbits - len(echelon(constraints(graph, require_web=True))),
                         incoming=len(incoming), output_only=len(boundary), internal_only=len(closed),
                         internal_webs=len(closed_webs), nodes=len(graph['nodes']), edges=len(graph['edges']))
