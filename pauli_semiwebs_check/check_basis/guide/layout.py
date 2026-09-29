"""Positions for the whole-circuit ZX graph: one row per qubit, time to the right.

Operations are packed as soon as possible, in source order. A two-qubit gate or a
Pauli product blocks every row between its outermost qubits, so no connection
crosses a spider. Single-qubit operations from one source line share a column, so
each T layer and each measurement layer stays in one column. A barrier line starts
a new stage: nothing from that line on is drawn left of anything before it.
Columns holding a printed phase are wider.
"""

from collections import defaultdict

SINGLE = {'R', 'RX', 'PHASE', 'M', 'MX', 'FEEDX', 'FEEDZ'}
PHASE_WIDTH = 1.7


def op_rows(op, row):
    if op['op'] in ('CX', 'CZ'):
        return [row[op['c']], row[op['t']]]
    if op['op'] == 'MPP':
        return [row[q] for _, q in op['factors']]
    return [row[op['q']]]


def layout(parsed, g, barriers=()):
    ops = parsed['ops']
    barriers = sorted(barriers)
    qubits = sorted({n['qubit'] for n in g['nodes'].values() if n['qubit'] is not None})
    row = {q: i for i, q in enumerate(qubits)}

    frontier, column = defaultdict(int), {}
    i = 0
    while i < len(ops):
        while barriers and ops[i]['line'] >= barriers[0]:
            barriers.pop(0)
            top = max(frontier.values(), default=0)
            for r in range(len(qubits)):
                frontier[r] = top
        if ops[i]['op'] in SINGLE:
            j = i
            while j < len(ops) and ops[j]['op'] in SINGLE and ops[j]['line'] == ops[i]['line']:
                j += 1
            rows = [op_rows(ops[k], row)[0] for k in range(i, j)]
            c, seen = max(frontier[r] for r in rows) + 1, set()
            for k, r in zip(range(i, j), rows):
                if r in seen:                  # the same qubit twice on one line
                    c, seen = c + 1, set()
                column[k] = c
                seen.add(r)
            for r in rows:
                frontier[r] = c
            i = j
            continue
        rows = op_rows(ops[i], row)
        span = range(min(rows), max(rows) + 1)
        width = 3 if ops[i]['op'] == 'MPP' and ops[i]['factors'][0][0] == 'Y' else 1
        c = max(frontier[r] for r in span) + 1
        column[i] = c
        for r in span:
            frontier[r] = c + width - 1
        i += 1
    ncols = max(frontier.values())

    # Offsets within a column, in units of that column's width.
    discarded = {(n['op'], n['qubit']) for n in g['nodes'].values() if n['role'] == 'discard'}
    placed = {}
    for k, n in g['nodes'].items():
        role, op = n['role'], n['op']
        if role == 'output':
            continue
        c, f = column[op], 0.0
        if role == 'discard':
            f = -.25
        elif role == 'prepare' and (op, n['qubit']) in discarded:
            f = .25
        elif role in ('reprepare', 'implicit_zero'):
            f = -.42
        elif role in ('product_wire', 'product') and ops[op]['factors'][0][0] == 'Y':
            c += 1
        elif role == 'y_basis_out':
            c += 2
        if role == 'product':
            f += .5
        placed[k] = (c, f)

    wide = defaultdict(bool)
    for k, (c, f) in placed.items():
        if g['nodes'][k]['phase'] and g['nodes'][k]['kind'] in ('X', 'Z'):
            wide[c] = True
    width = {c: PHASE_WIDTH if wide[c] else 1.0 for c in range(1, ncols + 1)}
    centre, x = {}, 0.0
    for c in range(1, ncols + 1):
        centre[c] = x + width[c] / 2
        x += width[c]
    total = x

    # Outputs sit in one extra column. A page shows whole columns, so the only
    # connections cut at its edges are wires.
    left = {c: centre[c] - width[c] / 2 for c in centre}
    right = {c: centre[c] + width[c] / 2 for c in centre}
    left[ncols + 1], right[ncols + 1] = total, total + 1.2
    node_column = {k: placed[k][0] if k in placed else ncols + 1 for k in g['nodes']}

    pos = {}
    for k, n in g['nodes'].items():
        if n['role'] == 'output':
            pos[k] = (total + .6, -row[n['qubit']])
            continue
        c, f = placed[k]
        if n['kind'] == 'H' or n['role'] == 'product':
            nbrs = [e['v'] if e['u'] == k else e['u'] for e in g['edges'] if k in (e['u'], e['v'])]
            ys = [row[g['nodes'][m]['qubit']] for m in nbrs]
            y = (min(ys) + max(ys)) / 2
            if y == int(y):                    # keep the spider off a wire
                y += .5
            pos[k] = (centre[c] + f * width[c], -y)
        else:
            pos[k] = (centre[c] + f * width[c], -row[n['qubit']])

    line_column = defaultdict(list)
    for i, op in enumerate(ops):
        line_column[op['line']].append(column[i])
    line_x = {line: (centre[min(cs)] - width[min(cs)] / 2, centre[max(cs)] + width[max(cs)] / 2)
              for line, cs in line_column.items()}
    return dict(pos=pos, qubits=qubits, row=row, total=total, line_x=line_x,
                columns=ncols, column_of_op=column, node_column=node_column, left=left, right=right)
