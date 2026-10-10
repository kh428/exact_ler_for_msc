"""The actual d=3 double check of the SOFT circuit (data/inputs/clifft_d3_p001.stim, lines 131-188), propagated
backwards one moment at a time, and the d=5 central readout.

Records are numbered from 1 over the whole file, as in Fig. 19 of the paper: r8 is the central MX 7, r9-r14 are
MX 14 11 6 2 8 1. Checks:
  * noiseless: X7 reaches the data as R = 2^(-7/2) prod_q (X_q + Y_q), and on the code space R = M_L (Eq. (12)-(13));
  * noisy: the factor picked up by r8 equals the one from Stim's detector error model of the S proxy;
  * d=5: X25 reaches the data as a product of 19 forks, 1024 of the 2^19 strings survive, and they add to M_L.

  python -B real_check.py
"""
import json
from fractions import Fraction
from pathlib import Path

from engine import (Poly, Operator, ONE, parse, records, step_ops, apply_backward, string_tex, letter, set_letter,
                    anticommutes)

REPO = Path(__file__).resolve().parents[2]  # the repository root (this file is explainer/prop/real_check.py)
D3_FILE = REPO / 'data/inputs/clifft_d3_p001.stim'
D5_FILE = REPO / 'data/inputs/soft_cultivation_d5_p0005.stim'
FIRST, LAST = 131, 188                      # the double check, source lines

CODES = {
    3: dict(sites=[0, 3, 7, 9, 10, 12, 13], faces=[[0, 3, 7, 10], [3, 7, 9, 12], [7, 10, 12, 13]]),
    5: dict(sites=[0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40],
            faces=[[0, 3, 7, 11], [9, 13, 15, 19, 21, 27], [17, 25, 32, 36], [3, 7, 9, 13], [15, 21, 23, 29],
                   [19, 25, 27, 32, 34, 38], [7, 11, 13, 17, 19, 25], [21, 27, 29, 34], [32, 36, 38, 40]]),
}


def mask(qs):
    m = 0
    for q in qs:
        m |= 1 << q
    return m


def rowspace(faces):
    out = {0}
    for f in faces:
        fm = mask(f)
        out |= {v ^ fm for v in out}
    return out


def logical_action(key, code):
    """A data string on the code space: None if it flips a stabiliser, else (sign, 'I'|'X'|'Y'|'Z')."""
    x, z = key
    sites = mask(code['sites'])
    assert (x | z) & ~sites == 0, 'support outside the data'
    for f in code['faces']:
        fm = mask(f)
        if bin(x & fm).count('1') % 2 or bin(z & fm).count('1') % 2:
            return None
    rows = code.setdefault('_rows', rowspace(code['faces']))
    a = x not in rows
    b = z not in rows
    k = bin(x & z).count('1')               # P = i^k X^x Z^z
    if a and b:                             # X_L Z_L = -i Y_L
        power = (k - 1) % 4
        assert power in (0, 2)
        return (1 if power == 0 else -1), 'Y'
    assert k % 2 == 0
    sign = 1 if k % 4 == 0 else -1
    return sign, ('X' if a else 'Z' if b else 'I')


S_ = Poly.sym('s')
T_MEAN = {'I': ONE, 'X': S_, 'Y': -S_, 'Z': Poly()}    # <T_L^dag| L |T_L^dag>


def code_mean(O, code, E=(0, 0)):
    """<T_L^dag| E^dag O E |T_L^dag> for an operator on the data."""
    total = Poly()
    for key, c in O.terms.items():
        act = logical_action(key, code)
        if act is None:
            continue
        sign, L = act
        if anticommutes(key, E):
            sign = -sign
        total = total + c * T_MEAN[L] * sign
    return total


def moments(ops):
    """Group the operations of the region by TICK: [(first line, last line, [op indices])]."""
    out, cur = [], []
    for i, (line, name, arg, targets) in enumerate(ops):
        if not (FIRST <= line <= LAST):
            continue
        if name == 'TICK':
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        out.append(cur)
    return out


GATE_TEXT = {'CX': 'CX', 'T': 'T', 'T_DAG': 'T_DAG', 'RX': 'RX', 'R': 'R', 'MX': 'MX', 'M': 'M'}


def run_d3(parity, noisy):
    text = D3_FILE.read_text()
    ops = parse(text)
    rec_of, _ = records(ops)
    allsteps = step_ops(ops)
    by_op = {}
    for st in allsteps:
        by_op.setdefault(st['op'], []).append(st)
    O = Operator.identity()
    hist = []
    for mom in reversed(moments(ops)):
        for i in reversed(mom):
            for st in reversed(by_op.get(i, [])):
                apply_backward(O, st, rec_of, set(parity), noisy)
        hist.append((mom, O.copy()))
    return ops, rec_of, hist, O


def factor_common(O):
    """(common Poly, {key: rest}) when every coefficient is the same monomial times +-1 or +-1/sqrt2 powers."""
    vals = list(O.terms.values())
    if not vals:
        return Poly(), {}
    first = vals[0]
    if all(v == first or v == -first for v in vals):
        return first, {k: (1 if v == first else -1) for k, v in O.terms.items()}
    return None, None


def describe(O, order):
    """For the widget: up to 8 terms with TeX, the count and the letters each qubit carries."""
    terms = sorted(O.terms.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    letters = {}
    for (x, z), c in terms:
        for q in range((x | z).bit_length()):
            l = letter(x, z, q)
            if l != 'I':
                letters.setdefault(q, set()).add(l)
    out = dict(count=len(terms), letters={str(q): ''.join(sorted(v)) for q, v in sorted(letters.items())})
    if len(terms) <= 8:
        out['terms'] = [[string_tex(k, order), c.tex()] for k, c in terms]
    return out


def main():
    code = CODES[3]
    text = D3_FILE.read_text()
    ops = parse(text)
    rec_of, nrec = records(ops)
    rec_line = {r: (ops[i][0], ops[i][1], ops[i][3][j]) for (i, j), r in rec_of.items()}
    central = [r for r, (line, name, t) in rec_line.items() if line == 154][0]
    aux = {int(t): r for r, (line, name, t) in rec_line.items() if line == 180}
    print('central readout record', central, 'aux readout records', aux)
    assert central == 8 and aux == {14: 9, 11: 10, 6: 11, 2: 12, 8: 13, 1: 14}

    # noiseless X7: one string through the CNOTs, 128 after the entry layer, M_L on the code space
    _, _, hist, O = run_d3([central], noisy=False)
    counts = [len(o) for _, o in hist]
    print('terms after each moment, backwards:', counts)
    R = Operator({(0, 0): ONE})
    for q in code['sites']:
        nxt = Operator()
        for key, c in R.terms.items():
            for l, cc in (('X', S_), ('Y', S_)):
                nxt.add(set_letter(*key, q, l), c * cc)
        R = nxt
    assert O.terms == R.terms, 'X7 does not reach R = 2^(-7/2) prod (X+Y)'
    survivors = [k for k in O.terms if logical_action(k, code) is not None]
    acts = [logical_action(k, code) for k in survivors]
    print('survivors', len(survivors), 'as +X_L:', acts.count((1, 'X')), 'as -Y_L:', acts.count((-1, 'Y')))
    assert len(survivors) == 16 and acts.count((1, 'X')) == 8 and acts.count((-1, 'Y')) == 8
    assert code_mean(O, code) == ONE
    E = {'I': (0, 0), 'X7': set_letter(0, 0, 7, 'X'), 'Z7': set_letter(0, 0, 7, 'Z'), 'Y7': set_letter(0, 0, 7, 'Y'),
         'X0X3X10': (mask([0, 3, 10]), 0)}
    for name, e in E.items():
        print(f'  <E^dag R E> for E={name}:', code_mean(O, code, e).tex())

    # noisy: the factor collected by r8
    _, _, histn, On = run_d3([central], noisy=True)
    mean = code_mean(On, code)
    print('noisy <r8 term>:', mean.tex())
    # Stim check on the S proxy
    stim_factor = stim_check(central)
    lam = mean.value(0.001)
    print('noise factor at p=1e-3: engine', repr(lam), 'stim DEM', repr(stim_factor))
    assert abs(lam - stim_factor) < 1e-13

    # every readout of the check: which reach the data, and how many strings
    for r in [central] + sorted(aux.values()):
        _, _, _, Or = run_d3([r], noisy=False)
        print(f'  r{r}: {len(Or)} strings at the start;', 'mean', code_mean(Or, code).tex() if all(
            ((k[0] | k[1]) & ~mask(code['sites'])) == 0 for k in Or.terms) else 'off the data')
    _, _, _, Od = run_d3([8, 11], noisy=False)
    print('detector r8+r11:', len(Od), 'strings,', [string_tex(k) for k in Od.terms])
    d5_central()


def stim_check(central):
    """Noise factor of the central readout from Stim's detector error model, T replaced by S."""
    import stim
    lines = D3_FILE.read_text().splitlines()
    region = []
    for i, raw in enumerate(lines, start=1):
        if FIRST <= i <= LAST:
            s = raw.strip()
            if s.startswith(('DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS')):
                continue
            s = s.replace('T_DAG', 'S_DAG') if s.startswith('T_DAG') else (('S' + s[1:]) if s.startswith('T ') else s)
            region.append(s)
    code = CODES[3]
    # input: the encoded -Y_L... eigenstate the proxy check measures with +1: prepare it from stabilisers
    stabs = []
    for f in code['faces']:
        stabs.append(stim.PauliString('*'.join(f'X{q}' for q in f)))
        stabs.append(stim.PauliString('*'.join(f'Z{q}' for q in f)))
    # the readout X7 reaches prod_q (S X S^dag)_q = prod Y_q; choose the logical eigenstate with <prod Y> = +1
    stabs.append(stim.PauliString('*'.join(f'Y{q}' for q in code['sites'])))
    n = 15
    for q in range(n):
        if q not in code['sites']:
            stabs.append(stim.PauliString('*'.join([f'Z{q}'])))
    tab = stim.Tableau.from_stabilizers(stabs, allow_redundant=False, allow_underconstrained=False)
    prep = tab.to_circuit('elimination')
    circ = prep + stim.Circuit('\n'.join(region))
    # the central readout is the first measurement of the region
    meas = [k for k, ins in enumerate(circ.flattened()) if ins.name in ('M', 'MX', 'MPP')]
    nmeas = circ.num_measurements
    first_region = nmeas - 7                 # MX 7 then MX of six ancillas
    circ.append('OBSERVABLE_INCLUDE', [stim.target_rec(first_region - nmeas)], 0)
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    factor = 1.0
    for ins in dem.flattened():
        if ins.type == 'error' and any(t.is_logical_observable_id() for t in ins.targets_copy()):
            factor *= 1 - 2 * ins.args_copy()[0]
    return factor


def d5_central():
    """X25 through the d=5 fold, then the entry layer: 2^19 strings; their sum on the code space."""
    text = D5_FILE.read_text()
    ops = parse(text)
    rec_of, _ = records(ops)
    steps = step_ops(ops)
    central_line = 415
    central = [r for (i, j), r in rec_of.items() if ops[i][0] == central_line][0]
    entry_lines = (391, 392)
    O = Operator.identity()
    for st in reversed(steps):
        if st['line'] > central_line:
            continue
        if st['line'] in entry_lines or st['line'] < entry_lines[0]:
            break
        apply_backward(O, st, rec_of, {central}, noisy=False)
    assert len(O) == 1
    (key, coef), = O.terms.items()
    code = CODES[5]
    x, z = key
    data = mask(code['sites'])
    anc = [q for q in range((x | z).bit_length()) if (x | z) >> q & 1 and not data >> q & 1]
    print('d=5: X25 before the entry layer:', string_tex(key), 'coefficient', coef.tex(), 'ancillas', anc)
    assert z == 0 and x & data == data
    # entry layer: T on six sites, T_DAG on thirteen. Backwards: T gives (X - Y)/sqrt2, T_DAG gives (X + Y)/sqrt2.
    six = mask([0, 11, 17, 19, 36, 40])
    sites = code['sites']
    count = {('X', 1): 0, ('X', -1): 0, ('Y', 1): 0, ('Y', -1): 0}
    surv = 0
    for ymask_idx in range(1 << 19):
        y = 0
        for b, q in enumerate(sites):
            if ymask_idx >> b & 1:
                y |= 1 << q
        act = logical_action((data, y), code)
        if act is None:
            continue
        surv += 1
        sign, L = act
        sign *= -1 if bin(y & six).count('1') % 2 else 1
        count[(L, sign)] += 1
    print('d=5 survivors', surv, count)
    cx = count[('X', 1)] - count[('X', -1)]
    cy = count[('Y', 1)] - count[('Y', -1)]
    # sum = 2^(-19/2) (cx X_L + cy Y_L); M_L = (X_L - Y_L)/sqrt2 needs cx = 2^9 and cy = -2^9
    print('d=5 coefficient sums: X_L', cx, 'Y_L', cy)
    assert surv == 1024 and cx == 512 and cy == -512
    return dict(survivors=surv, counts={f'{L}{"+" if s > 0 else "-"}': v for (L, s), v in count.items()})


if __name__ == '__main__':
    main()
