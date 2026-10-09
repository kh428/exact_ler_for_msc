"""The two endpoint tables of the d=5 calculation, from stage networks of the circuit itself.

W (first stage, source lines before 189): the network ends with open labels on the seven d=3 data qubits, restricted
to the 256 operators S_g L_Q of the d=3 code's normaliser (g: which X and Z checks, Q: X_L and Z_L parts). Its value
V(g, Q) is the coefficient of that label in the accepted (unnormalised) output state. The table of the paper,
W(s, t) over the input syndrome s and logical tag t, is the inverse Walsh transform of V / r, with r the coefficient
of the ideal code state; the tag decomposition exists when no Z_L part survives (V(g, Z_L) = 0), which is checked.

F (final stage, source lines from 388): the network starts from an open unit label S_g L_Q on the nineteen d=5 data
qubits (g: which of the 9 X and 9 Z faces, Q as above) and imposes the stage's detectors, each without the records
of earlier stages (every one of those is fixed to 0 in an accepted shot, see audit_cut), and the observable for B.
F(u) = sum over (g, Q) of (-1)^(u.(g,Q)) r5(g, Q) R(g, Q), with r5 the ideal d=5 code state.

Index conventions are those of the paper's tree: W bit 0-2 input X checks, 3-5 Z checks, 6 XX, 7 ZZ; F bit 0-8
output X faces, 9-17 Z faces, 18 XX, 19 ZZ, faces in the order of the repository's d5 model.
"""
import numpy as np

from .growth import stage_model, GROWTH

FIRST_STOP, FINAL_START = GROWTH


def gate_range(circuit, lo_line, hi_line):
    idx = [g for g, gate in enumerate(circuit['gates']) if lo_line <= gate['line'] < hi_line]
    return idx[0], idx[-1] + 1


def stage_locations(locations, lo_line, hi_line):
    return [loc for loc in locations if lo_line <= loc['line'] < hi_line]


def record_lines(circuit):
    return {g['record']: g['line'] for g in circuit['gates'] if g['op'] in ('M', 'MX', 'MPP')}


def stage_detectors(circuit, lo_line, hi_line, *, drop_outside):
    """Detectors with a record in the stage; records of other stages dropped (drop_outside) or the detector skipped."""
    lines = record_lines(circuit)
    out = []
    for d in circuit['detectors']:
        inside = [r for r in d['records'] if lo_line <= lines[r] < hi_line]
        if not inside:
            continue
        if len(inside) < len(d['records']) and not drop_outside:
            continue
        out.append(dict(d, records=inside))
    return out


def code_labels(data, faces_or_checks, coefficient_bits):
    """For each data qubit, the coefficient bits that make up its x and z label bits.

    coefficient c: bits 0..m-1 X-type generators, m..2m-1 Z-type generators, 2m logical X, 2m+1 logical Z."""
    m = len(faces_or_checks)
    xs, zs = {}, {}
    for q in data:
        inside = [k for k, f in enumerate(faces_or_checks) if q in f]
        xs[q] = inside + [2 * m]
        zs[q] = [m + k for k in inside] + [2 * m + 1]
    assert coefficient_bits == 2 * m + 2
    return xs, zs


def xor_chain(compiler, variables):
    acc = None
    for v in variables:
        acc = compiler.xor(acc, v, 'code-label') if acc is not None else v
    return acc


def faces3():
    d3, _ = stage_model()
    data = d3['data']
    return data, [[data[i] for i in range(len(data)) if check >> i & 1] for check in d3['checks']]


def faces5():
    _, d5 = stage_model()
    return d5['data'], d5['faces']


def w_finish(compiler):
    data, checks = faces3()
    xs, zs = code_labels(data, checks, 8)
    coeff = [compiler.variable(f'out{j}') for j in range(8)]
    for q in data:
        x, z = compiler.wires.pop(q)
        for bit, parts in ((x, xs[q]), (z, zs[q])):
            combo = xor_chain(compiler, [coeff[j] for j in parts])
            link = compiler.xor(bit, combo, 'out-link')
            if link is not None:
                compiler.net.fixed(link, 0)
    return coeff


def f_start(compiler):
    data, faces = faces5()
    xs, zs = code_labels(data, faces, 20)
    coeff = [compiler.variable(f'in{j}') for j in range(20)]
    for q in data:
        compiler.wires[q] = (xor_chain(compiler, [coeff[j] for j in xs[q]]),
                             xor_chain(compiler, [coeff[j] for j in zs[q]]))
    return coeff


def label_bits(c, data, faces):
    """The x and z masks over data (in data order) of the coefficient vector c."""
    m = len(faces)
    xs, zs = code_labels(data, faces, 2 * m + 2)
    a = sum(1 << i for i, q in enumerate(data) if sum(c >> j & 1 for j in xs[q]) % 2)
    b = sum(1 << i for i, q in enumerate(data) if sum(c >> j & 1 for j in zs[q]) % 2)
    return a, b


def ideal_signs(data, faces):
    """(-1)^{|A and B|} for every coefficient vector: the sign relating the network's label coefficient
    Tr(Z^B X^A rho) to Tr(X^A Z^B rho) = Tr(L_Q rho) on the code space."""
    m = len(faces)
    n = 1 << (2 * m + 2)
    out = np.empty(n, dtype=np.int64)
    for c in range(n):
        a, b = label_bits(c, data, faces)
        out[c] = -1 if bin(a & b).count('1') % 2 else 1
    return out


def walsh(values, prime, inverse=False):
    """Walsh-Hadamard transform over the last axis of an object or int array, modulo prime."""
    v = np.array(values, dtype=object) % prime
    n = v.shape[-1]
    h = 1
    while h < n:
        a = v.reshape(v.shape[:-1] + (n // (2 * h), 2, h))
        left, right = a[..., 0, :].copy(), a[..., 1, :].copy()
        a[..., 0, :] = (left + right) % prime
        a[..., 1, :] = (left - right) % prime
        v = a.reshape(v.shape)
        h *= 2
    if inverse:
        v = v * pow(n, -1, prime) % prime
    return v


def _echelon(rows):
    piv = {}
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h not in piv:
                piv[h] = v
                break
            v ^= piv[h]
    return piv


def _in_span(v, piv):
    while v:
        h = v.bit_length() - 1
        if h not in piv:
            return False
        v ^= piv[h]
    return True


def audit_cut(circuit, growth_audit, free_records):
    """The stage conditions against the declared detectors, as linear forms on the record flips.

    The stages impose: the detectors of the first stage that use only its records; every growth record that is not a
    free random outcome (each deterministic one, and each that reads the input syndrome, which the tree matches with
    W); the final-stage detectors without their earlier records. The cut is exact when these span the same space as
    all declared detectors. Returns the two ranks and whether each set implies the other.
    """
    lines = record_lines(circuit)

    def stage(r):
        return 0 if lines[r] < FIRST_STOP else 1 if lines[r] < FINAL_START else 2

    def vec(records):
        v = 0
        for r in records:
            v ^= 1 << r
        return v

    declared = [vec(d['records']) for d in circuit['detectors']]
    growth = sorted(r for r in lines if stage(r) == 1)
    classes = {a['record']: a['classification'] for a in growth_audit if 'record' in a}
    assert all(r in classes for r in growth), 'every growth record is classified'
    first = [vec(d['records']) for d in circuit['detectors'] if all(stage(r) == 0 for r in d['records'])]
    middle = [1 << r for r in growth if r not in free_records]
    final = [vec([r for r in d['records'] if stage(r) == 2]) for d in circuit['detectors']
             if any(stage(r) == 2 for r in d['records'])]
    staged = first + middle + final
    pd, ps = _echelon(declared), _echelon(staged)
    return dict(declared_rank=len(pd), stage_rank=len(ps),
                stages_implied_by_detectors=all(_in_span(v, pd) for v in staged),
                detectors_implied_by_stages=all(_in_span(v, ps) for v in declared),
                free_records_in_detectors=[r for r in free_records if any(v >> r & 1 for v in declared)])
