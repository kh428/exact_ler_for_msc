"""Closed Pauli webs and semiwebs of the 15-to-1 factory, with every claim checked.

Run from this folder:  python -B -m factory.run [--output]
Without --output nothing is written. With it, results/factory.json and webs/*.tikz are
rewritten.

The factory is that of Litinski, Quantum 3, 205 (2019), for the 15-to-1 protocol of Bravyi
and Kitaev, Phys. Rev. A 71, 022316 (2005). Its S-gate proxy has every T state replaced by
an S state. Each of the four X measurements is a closed Pauli web of the proxy. Read on the
diagram with T states, the same labels are a closed semiweb with defects at T states.
"""

import argparse
import json

from src.linear import kernel, rank
from src.semiweb import constraints, defects

from . import HERE, statevector
from .diagram import incidence, read, supports, with_phase
from .draw import tikz
from .tensor import Tensor, fixed_checks, product, proportional, times_unit
from .webs import by_records, closed_webs, labelling, labels_at, record_support, scalar_exponent, solve

DIAGRAMS = HERE / 'diagrams'
FACTORY = DIAGRAMS / 'factory_t.tikz'                     # the factory, drawn with T states
HAND_DRAWN = [DIAGRAMS / f'proxy_web_r{i}.tikz' for i in (1, 2, 3, 4)]   # the proxy, one web each
LAYOUT = HAND_DRAWN[0]                                    # node positions with room for a web
ANGLE = {2: '+pi/2', -2: '-pi/2', 4: 'pi', 1: '+pi/4', -1: '-pi/4', 3: '+3pi/4', -3: '-3pi/4'}
SHIFTS = {'+pi/2': 2, '+pi': 4, '+3pi/2': 6}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def pattern(r):
    """A record pattern as r_1 r_2 r_3 r_4."""
    return ''.join(str((r >> i) & 1) for i in range(4))


def analyse():
    """Return the results and the drawings, after checking every statement in the README."""
    base = read(LAYOUT)
    nodes = base['nodes']
    roles = [n['role'] for n in nodes.values()]
    require((roles.count('T'), roles.count('hub'), roles.count('measure'), roles.count('prepare'),
             roles.count('output')) == (15, 11, 4, 1, 1), 'unexpected factory diagram')
    require(all(nodes[k]['phase'] == 2 for k in base['sites']), 'the layout drawing is not the proxy')
    drawn = read(FACTORY)
    require(all(drawn['nodes'][k]['phase'] == 1 for k in drawn['sites']), 'factory_t.tikz has other phases')
    require(supports(drawn) == supports(base), 'the two drawings are different factories')

    columns = supports(base)
    matrix = [[int(w in s) for s in columns] for w in range(5)]
    weights = [sum(row) for row in matrix]
    pair = [[sum(a * b for a, b in zip(matrix[i], matrix[j])) for j in range(5)] for i in range(5)]
    triple = sorted({sum(a * b * c for a, b, c in zip(matrix[i], matrix[j], matrix[k]))
                     for i in range(5) for j in range(i + 1, 5) for k in range(j + 1, 5)})
    require(weights == [7, 8, 8, 8, 8], 'row weights')
    require({pair[i][j] for i in range(1, 5) for j in range(1, 5) if i != j} == {4}, 'pair overlaps')
    require({sum(a * b * c for a, b, c in zip(matrix[i], matrix[j], matrix[k]))
             for i in range(1, 5) for j in range(i + 1, 5) for k in range(j + 1, 5)} == {2}, 'triple overlaps')
    require(sorted(sum(1 << (w - 1) for w in s if w) for s in columns) == list(range(1, 16)),
            'the columns on wires 1 to 4 are not the fifteen nonzero vectors')

    proxy, actual = with_phase(base, 2), with_phase(base, 1)
    inc = incidence(base)
    nedge = len(base['edges'])

    # Closed Pauli webs: four in the proxy, one for each record, and none with T states.
    proxy_basis = closed_webs(proxy)
    pivots, free = by_records(proxy, proxy_basis)
    actual_basis = closed_webs(actual)
    semiweb_dimension = 2 * nedge - rank(constraints(actual))
    require((len(proxy_basis), len(pivots), len(free)) == (4, 4, 0), 'proxy webs')
    require(len(actual_basis) == 0, 'the T diagram has a closed Pauli web')

    # The tensor of each diagram, and the same factory as five qubits.
    tensor_t, tensor_s = Tensor(actual), Tensor(proxy)
    require((tensor_t.variables, tensor_t.assignments) == (16, 32), 'tensor size')
    for tensor, phase in ((tensor_t, 1), (tensor_s, 2)):
        for r in range(16):
            require(tensor(r).tolist() == statevector.output(columns, [phase] * 15, r),
                    'the diagram and the five-qubit calculation differ')
    out_t, out_s = tensor_t(0), tensor_s(0)
    for tensor, out, power in ((tensor_t, out_t, -1), (tensor_s, out_s, -2)):
        require(tensor.live() == [0], 'a record other than 0000 occurs')
        require(out[0].any() and (out[1] == times_unit(out[0], power)).all(), 'output state')

    # Every parity of the four records: a Pauli web of the proxy, a semiweb with T states.
    checks = []
    for c in range(1, 16):
        v = solve(pivots, c)
        require(record_support(proxy, inc, v) == c and not defects(proxy, v), 'proxy web')
        marks = defects(actual, v)
        require(all(nodes[k]['role'] == 'T' for k in marks), 'a defect away from the T states')
        lam_s, lam_t = scalar_exponent(proxy, inc, v), scalar_exponent(actual, inc, v)
        # The semiweb identity D_r = lambda_0 (-1)^(c.r) D_{r,delta}, and D_{r,delta} = D_r itself.
        for r in range(16):
            sign = 4 * ((c & r).bit_count() & 1)
            lhs, shifted = tensor_t(r), tensor_t(r, shift=marks)
            require((lhs == times_unit(shifted, lam_t + sign)).all(), 'semiweb identity')
            require((shifted == lhs).all(), 'the shifted diagram differs')
            require((tensor_s(r) == times_unit(tensor_s(r), lam_s + sign)).all(), 'proxy web identity')
        require(len(marks) == 8 and set(marks.values()) == {2} and lam_t == lam_s == 0, 'defects')
        if c & (c - 1) == 0:        # one check: the measured X, pushed back until the T states take it as Y
            for k, n in nodes.items():
                if len(inc[k]) == 1:
                    wanted = 'X' if n.get('record') == c.bit_length() - 1 else 'Y' if k in marks else 'I'
                    require(labels_at(v, inc[k]) == (wanted,), 'a check web does not end as stated')
        sites = {nodes[k]['site'] for k in marks}
        expected = {s for s in range(15) if sum((c >> (w - 1)) & 1 for w in columns[s] if w) % 2}
        require(sites == expected, 'the defects are not the T states with odd overlap')
        checks.append(dict(parity=c, records=[i + 1 for i in range(4) if (c >> i) & 1], vector=v, marks=marks,
                           labelled_edges=sum(((v >> (2 * j)) & 3) != 0 for j in range(nedge))))
    singles = [d for d in checks if len(d['records']) == 1]

    # The four webs drawn by hand on the proxy are the computed ones.
    hand_drawn = {}
    for path in HAND_DRAWN:
        old = read(path)
        shape = lambda g: ({k: (n['x'], n['y'], n['kind'], n['phase']) for k, n in g['nodes'].items()},
                           [frozenset(e.values()) for e in g['edges']])
        require(shape(old) == shape(base), f'{path.name} is another diagram')
        v = labelling(base, old['overlay'])
        require(not any((row & v).bit_count() & 1 for row in constraints(proxy, require_web=True)),
                f'{path.name}: the drawn labels are not a Pauli web')
        c = record_support(proxy, inc, v)
        require(v == solve(pivots, c) and (c & (c - 1)) == 0, f'{path.name}: not a computed web')
        hand_drawn[path.name] = c.bit_length()
    require(sorted(hand_drawn.values()) == [1, 2, 3, 4], 'hand-drawn webs')

    # One T state at a time, moved through the three other Clifford angles.
    sweep, survivors = {}, set(range(1, 16))
    for site in range(15):
        wires = sorted(w for w in columns[site] if w)
        column = sum(1 << (w - 1) for w in wires)
        entry = dict(wires=sorted(columns[site]))
        for name, shift in SHIFTS.items():
            phases = [1] * 15
            phases[site] = (1 + shift) % 8
            live = tensor_t.live(phases)
            require(live == statevector.live(columns, phases), 'the two calculations differ')
            kept = fixed_checks(live)
            survivors &= kept
            lost = sorted(i + 1 for i in range(4) if (1 << i) not in kept)
            if shift == 4:      # a Z error on this T state: its checks flip, and stay determined
                require(live == [column] and not lost, 'a Z error is not read as its column')
            else:               # T becomes T S or T_dag: the checks on its wires become random
                require(live == [0, column] and lost == wires, 'checks lost')
                require(kept == {c for c in range(1, 16) if (c & column).bit_count() % 2 == 0}, 'parities kept')
            entry[name] = dict(records_that_occur=[pattern(r) for r in live], random_checks=lost)
        sweep[str(site + 1)] = entry
    require(not survivors, 'a parity is fixed for every Clifford angle')

    # The web of the output: Y on the output leg and no record.
    (out_node,) = [k for k, n in nodes.items() if n['kind'] == 'boundary']
    (out_edge,) = inc[out_node]
    open_webs = kernel(constraints(proxy, require_web=True), 2 * nedge)
    require(len(open_webs) == 5, 'open webs of the proxy')
    (w,) = [x for x in open_webs if (x >> (2 * out_edge)) & 3]
    w ^= solve(pivots, record_support(proxy, inc, w))
    require(record_support(proxy, inc, w) == 0 and labels_at(w, [out_edge]) == ('Y',), 'output web')
    out_marks = defects(actual, w)
    require(len(out_marks) == 7 and set(out_marks.values()) == {2} and not defects(proxy, w), 'output defects')
    require({nodes[k]['site'] for k in out_marks} == {s for s in range(15) if 0 in columns[s]},
            'the output defects are not the T states on wire 0')
    shifted = tensor_t(0, shift=out_marks)
    y_on = lambda d: [times_unit(d[1], 6), times_unit(d[0], 2)]      # Y = [[0, -i], [i, 0]]
    lam = {}
    for name, d, other in (('proxy', out_s, out_s), ('T', out_t, shifted)):
        y = y_on(other)
        (lam[name],) = [e for e in range(8) if all((d[i] == times_unit(y[i], e)).all() for i in (0, 1))]
        require(lam[name] == scalar_exponent(proxy if name == 'proxy' else actual, inc, w), 'output scalar')
    require(lam['proxy'] == 4, 'the proxy output is not the -1 eigenstate of Y')
    require(not proportional(shifted, out_t), 'the shifted diagram is the original one')
    require((shifted[0] == out_t[0]).all() and (shifted[1] == times_unit(out_t[1], -2)).all(),
            'the seven shifts are not S_dag on the output')

    # The shifts as operators on the five wires. A shift of +pi/2 on a T state multiplies a basis
    # state by i to the parity of its bits on that T state's wires. Over the T states of one wire
    # the exponent is 7 x_0 mod 4 for the output wire, S_dag on the output, and 0 for a measured wire.
    for wire in range(5):
        for x in range(32):
            exponent = sum(sum((x >> w) & 1 for w in s) % 2 for s in columns if wire in s) % 4
            require(exponent == (3 * (x & 1) if wire == 0 else 0), 'the shifts of one wire')

    # The outputs as spiders, and the operator of which each is an eigenstate.
    require(product(out_s[0], [1, 0, -1, 0]) == product(out_s[1], [1, 0, 1, 0]),
            'the proxy output is not the X spider of phase pi/2')       # (1 + i)|0> + (1 - i)|1>
    y_s_dag = [times_unit(out_t[1], 4), times_unit(out_t[0], 2)]        # Y S_dag = [[0, -1], [i, 0]]
    require(all((y_s_dag[i] == times_unit(out_t[i], 3)).all() for i in (0, 1)),
            'the output is not an eigenstate of Y S_dag = exp(3 i pi/4) (X - Y)/sqrt2')

    drawings = {}
    for d in singles:
        i = d['records'][0]
        drawings[f'proxy_check{i}.tikz'] = tikz(
            base, d['vector'], phase=2, records=d['parity'],
            heading=f'15-to-1 factory, S-gate proxy: closed Pauli web of the check r_{i}')
        drawings[f't_check{i}.tikz'] = tikz(
            base, d['vector'], phase=1, records=d['parity'], marks=d['marks'],
            heading=f'15-to-1 factory, T states: closed semiweb of the check r_{i}, 8 defects of +pi/2')
    drawings['proxy_diagram.tikz'] = tikz(base, 0, phase=2, heading='15-to-1 factory, S-gate proxy')
    drawings['t_diagram.tikz'] = tikz(base, 0, phase=1, heading='15-to-1 factory, T states')
    drawings['proxy_output.tikz'] = tikz(
        base, w, phase=2, output='Y', heading='15-to-1 factory, S-gate proxy: open Pauli web with Y on the output')
    drawings['t_output.tikz'] = tikz(
        base, w, phase=1, output='Y', marks=out_marks,
        heading='15-to-1 factory, T states: the output web as a semiweb, 7 defects of +pi/2')

    by_site = lambda marks: {str(nodes[k]['site'] + 1): ANGLE[v] for k, v in sorted(
        marks.items(), key=lambda kv: nodes[kv[0]]['site'])}
    results = dict(
        diagrams=dict(factory=FACTORY.name, layout=LAYOUT.name, hand_drawn_proxy_webs=hand_drawn),
        graph=dict(nodes=len(nodes), edges=nedge, t_states=15, measurements=4,
                   summed_bits=tensor_t.variables, assignments_kept=tensor_t.assignments),
        matrix=dict(rows=matrix, row_weights=weights, pair_overlaps=pair, triple_overlaps=triple),
        t_states={str(i + 1): sorted(s) for i, s in enumerate(columns)},
        proxy=dict(closed_pauli_webs=len(proxy_basis), record_free=len(free), records_that_occur=['0000'],
                   output='|0> - i|1>', output_tensor=out_s.tolist()),
        t_diagram=dict(closed_pauli_webs=len(actual_basis), semiweb_dimension=semiweb_dimension,
                       records_that_occur=['0000'], output='|0> + e^{-i pi/4}|1>',
                       output_tensor=out_t.tolist()),
        checks=[dict(records=d['records'], labelled_edges=d['labelled_edges'], defects=by_site(d['marks']),
                     lambda_0_exponent=0, shifted_diagram_equals_original=True) for d in checks],
        single_t_state_clifford_sweep=dict(
            note='one T state moved by a Clifford angle; wires include wire 0, checks are r_1 to r_4',
            t_states=sweep, parities_fixed_for_every_angle=[]),
        output_web=dict(label='Y', defects=by_site(out_marks), lambda_exponent=dict(proxy=lam['proxy'], t=lam['T']),
                        shifted_diagram_equals_original=False, shifted_output_tensor=shifted.tolist()),
        output=dict(
            proxy=dict(state='|0> - i|1>', plus_one_eigenstate_of='-Y',
                       spider='X spider of phase +pi/2, which is a Z spider of phase +pi/2 followed by a '
                              'Hadamard and, up to a scalar, a Z spider of phase -pi/2'),
            t=dict(state='|0> + e^{-i pi/4}|1>',
                   plus_one_eigenstate_of='(X - Y)/sqrt2 = T_dag X T = e^{-3 i pi/4} Y S_dag',
                   pauli_expectation_values='X: 1/sqrt2, Y: -1/sqrt2, Z: 0',
                   spider='Z spider of phase -pi/4'),
            seven_shifts_of_the_output_wire='S_dag on the output, on all 32 basis states',
            eight_shifts_of_each_measured_wire='the identity, on all 32 basis states'),
        arithmetic='exact: each tensor entry is four integers, the coefficients of 1, w, w^2, w^3 with '
                   'w = exp(i pi/4); one positive constant is left out of every tensor')
    return results, drawings


def summary(results):
    lines = [f"{results['graph']['nodes']} nodes, {results['graph']['edges']} edges, "
             f"{results['graph']['summed_bits']} summed bits",
             f"S-gate proxy: {results['proxy']['closed_pauli_webs']} closed Pauli webs, "
             f"{results['proxy']['record_free']} without a record; output {results['proxy']['output']}",
             f"T states: {results['t_diagram']['closed_pauli_webs']} closed Pauli webs; semiweb space of "
             f"dimension {results['t_diagram']['semiweb_dimension']}; output {results['t_diagram']['output']}",
             'every parity of the four records: 8 defects of +pi/2 with T states, lambda_0 = 1,',
             '  and the eight shifted phases leave the diagram unchanged']
    for d in results['checks']:
        if len(d['records']) == 1:
            lines.append(f"  r_{d['records'][0]}: {d['labelled_edges']} labelled edges, "
                         f"defects at T states {', '.join(d['defects'])}")
    lines += ['one T state moved by +pi/2 or +3pi/2: exactly the checks on its wires become random',
              'one T state moved by +pi: its checks flip and stay determined',
              f"output web: Y on the output, 7 defects at T states {', '.join(results['output_web']['defects'])}; "
              'the shifted diagram differs',
              'the seven shifts act as S_dag on the output, so the output is an eigenstate of Y S_dag,',
              '  which is (X - Y)/sqrt2 up to a phase: the output is the Z spider of phase -pi/4',
              'proxy output: the X spider of phase +pi/2, the -1 eigenstate of Y']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', action='store_true', help='rewrite results/factory.json and webs/*.tikz')
    args = parser.parse_args()
    results, drawings = analyse()
    print(summary(results))
    if args.output:
        (HERE / 'results').mkdir(exist_ok=True)
        (HERE / 'webs').mkdir(exist_ok=True)
        (HERE / 'results' / 'factory.json').write_text(json.dumps(results, indent=1) + '\n')
        for name, text in drawings.items():
            (HERE / 'webs' / name).write_text(text)
        print(f'wrote results/factory.json and {len(drawings)} drawings in webs/')


if __name__ == '__main__':
    main()
