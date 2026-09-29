"""ZX graph of an entire noiseless circuit, with every measurement record exposed.

Conventions match src/double_check_stage.py: a CNOT is a Z spider on the control
joined to an X spider on the target, and a diagonal phase gate diag(1, e^{i theta})
is a two-legged Z spider of phase theta. Phases are in units of pi/4.

Measurement records enter as spiders whose phase is r*pi:
  M  outcome r   -> effect <r|,   a one-legged X spider;
  MX outcome r   -> effect <+-|,  a one-legged Z spider;
  MPP Z product  -> Z spiders on the wires joined to a central X spider of phase r*pi;
  MPP X product  -> X spiders on the wires joined to a central Z spider of phase r*pi;
  MPP Y product  -> the X product conjugated by S_dag before and S after, per qubit;
  CX rec q       -> X^r, a two-legged X spider of phase r*pi;
  CZ rec q       -> Z^r, a two-legged Z spider of phase r*pi.
Record spiders are built at the baseline phase 0 and tagged with their record.
A qubit measured without reset and used again is re-prepared in |r>, a second
record spider for the same record.

A reset applied to a qubit still in use discards it. The oracle establishes that
each such qubit is in one known eigenstate of X or Z for every phase at the
preceding T gates, so the discard is exactly the post-selected effect onto that
state, a one-legged spider with no record.
"""

PREP = {'R': 'X', 'RX': 'Z'}            # |0> is an X spider, |+> a Z spider
EFFECT_FOR_STATE = {'X': 'Z', 'Z': 'X'}  # effect onto an X eigenstate is a Z spider


def build(parsed, discard, *, phase_override=None):
    nodes, edges = {}, []
    last, fresh, measured = {}, {}, {}
    current = [None]                        # index of the operation being translated
    counter = iter(range(10 ** 9))          # keys stay unique when fresh preparations are deleted

    def add(kind, phase, role, q=None, line=None, **extra):
        key = f'n{next(counter)}'
        nodes[key] = dict(kind=kind, phase=phase % 8, role=role, qubit=q, line=line, op=current[0], **extra)
        return key

    def join(a, b):
        edges.append(dict(u=a, v=b))

    def wire(q, line):
        """Return the open end of qubit q, creating it if needed."""
        if q in measured:
            kind, record = measured.pop(q)
            last[q] = add(kind, 0, 'reprepare', q, line, record=record)
        elif q not in last:
            last[q] = add('X', 0, 'implicit_zero', q, line)
        fresh[q] = False
        return last[q]

    def extend(q, node, line):
        join(wire(q, line), node)
        last[q] = node

    for i, op in enumerate(parsed['ops']):
        current[0] = i
        name, line = op['op'], op['line']
        if name in PREP:
            q = op['q']
            measured.pop(q, None)
            if q in last:
                if fresh.get(q):
                    del nodes[last[q]]
                else:
                    basis, value = discard[i]
                    end = add(EFFECT_FOR_STATE[basis], 0 if value == 1 else 4, 'discard', q, line)
                    join(last[q], end)
                del last[q]
            last[q] = add(PREP[name], 0, 'prepare', q, line)
            fresh[q] = True
        elif name == 'PHASE':
            phase = op['phase']
            if op['site'] is not None and phase_override is not None:
                phase = phase_override[op['site']]
            extend(op['q'], add('Z', phase, 'phase', op['q'], line, gate=op['gate'], site=op['site']), line)
        elif name == 'CX':
            c, t = op['c'], op['t']
            zc, xt = add('Z', 0, 'control', c, line), add('X', 0, 'target', t, line)
            extend(c, zc, line)
            extend(t, xt, line)
            join(zc, xt)
        elif name == 'CZ':
            a, b = op['c'], op['t']
            za, zb = add('Z', 0, 'cz', a, line), add('Z', 0, 'cz', b, line)
            h = add('H', 0, 'cz_hadamard', None, line)
            extend(a, za, line)
            extend(b, zb, line)
            join(za, h)
            join(h, zb)
        elif name in ('FEEDX', 'FEEDZ'):
            extend(op['q'], add(name[-1], 0, 'feedforward', op['q'], line, record=op['record']), line)
        elif name in ('M', 'MX'):
            q = op['q']
            kind = 'X' if name == 'M' else 'Z'
            join(wire(q, line), add(kind, 0, 'measure', q, line, record=op['record']))
            del last[q]
            measured[q] = (kind, op['record'])
        elif name == 'MPP':
            pauli = op['factors'][0][0]
            central = add('X' if pauli == 'Z' else 'Z', 0, 'product', None, line,
                          record=op['record'], product=''.join(f'{p}{q}' for p, q in op['factors']))
            for _, q in op['factors']:
                if pauli == 'Z':
                    w = add('Z', 0, 'product_wire', q, line)
                    extend(q, w, line)
                    join(w, central)
                elif pauli == 'X':
                    w = add('X', 0, 'product_wire', q, line)
                    extend(q, w, line)
                    join(w, central)
                else:
                    extend(q, add('Z', 6, 'y_basis_in', q, line), line)
                    w = add('X', 0, 'product_wire', q, line)
                    extend(q, w, line)
                    join(w, central)
                    extend(q, add('Z', 2, 'y_basis_out', q, line), line)
        else:
            raise ValueError(f'Unsupported operation {name} at line {line}')

    current[0] = None
    for q in sorted(last):
        if fresh.get(q):
            del nodes[last[q]]
        else:
            join(last[q], add('boundary', 0, 'output', q, None))
    used = {e['u'] for e in edges} | {e['v'] for e in edges}
    assert used == set(nodes), 'every node must be connected'
    return dict(nodes=nodes, edges=edges)
