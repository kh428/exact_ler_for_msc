"""Recompute every number shown by the propagation pages of the Atlas and by the film, and check each one.

Each piece is computed here and checked before it is written:
  small   the worked circuits, step by step as propagation trees; density-matrix check at three noise strengths
  real    the d=3 double check of the SOFT circuit, moment by moment, for four readouts; Stim check of the noise
  d5      the d=5 central readout: 2^19 strings, 1024 survivors, M_L
  toy     a one-qubit double check from propagation to exact series, residues modulo small primes and CRT;
          the residues are also obtained by contracting the transfer-matrix network modulo each prime
  stored  the repository's own residues for the d=3 and d=5 series, rebuilt into 32/75 and 574/375

The result is written to outputs/explainer/prop_data.json and compared with the copy embedded in
exact_ler_atlas.html. From the repository root, with numpy and stim installed (requirements-zx.txt):

  python3 -B explainer/prop/build_prop_data.py
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from engine import (Poly, Operator, ONE, parse, records, step_ops, apply_backward, expect_zero, density_check,
                    string_tex, letter, set_letter, mul_strings, TABLES, LETTERS)
import small_examples as SE
import real_check as RC

HERE = Path(__file__).resolve().parent
REPO = RC.REPO
OUT = REPO / 'outputs' / 'explainer' / 'prop_data.json'
PAGE = HERE.parent / 'exact_ler_atlas.html'


def frac_tex(v):
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    sign = '-' if v < 0 else ''
    return f'{sign}\\tfrac{{{abs(v.numerator)}}}{{{v.denominator}}}'


def frac_str(v):
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f'{v.numerator}/{v.denominator}'


def poly_terms(c):
    """A coefficient for the browser: [[s, e1, e2, ef, num, den], ...]."""
    return [[k[0], k[1], k[2], k[3], v.numerator, v.denominator] for k, v in sorted(c.t.items())]

# ------------------------------------------------------------------ small examples as propagation trees


def column_label(name, arg, targets):
    if name in ('DEPOLARIZE1',):
        return '\\mathcal N_1'
    if name == 'DEPOLARIZE2':
        return '\\mathcal N_2'
    if name.endswith('_ERROR'):
        return f'{name[0]}_p'
    return {'T': 'T', 'T_DAG': 'T^\\dagger', 'S': 'S', 'S_DAG': 'S^\\dagger', 'H': 'H', 'X': 'X', 'Y': 'Y',
            'Z': 'Z'}.get(name, name)


def tree_example(name):
    ex = SE.EXAMPLES[name]
    text = ex['text'].replace('(p)', '(0.001)')
    ops = parse(text)
    rec_of, nrec = records(ops)
    steps = step_ops(ops)
    parity = set(ex['parity'])
    # columns: one per operation line, in circuit order
    cols = []
    for st in steps:
        line, op, arg, targets = ops[st['op']]
        qs = [int(t) for t in targets]
        cols.append(dict(kind=st['kind'], gate=op, qubits=qs, noisy=arg is not None and st['kind'] != 'unitary',
                         label=column_label(op, arg, targets),
                         records=[rec_of[(st['op'], j)] for j in range(len(targets))] if st['kind'] == 'measure' else []))
    # propagate term by term to record edges
    O = Operator.identity()
    layers = [dict(col=len(cols), nodes=[dict(s='I', c='1', v=[[0, 0, 0, 0, 1, 1]], key=[0, 0])], edges=[])]
    keys_prev = [(0, 0)]
    for ci in reversed(range(len(steps))):
        st = steps[ci]
        children = {}
        edges = []
        new = Operator()
        for pi, key in enumerate(keys_prev):
            coef = O.terms.get(key)
            if coef is None:
                continue
            single = Operator({key: ONE})
            apply_backward(single, st, rec_of, parity, noisy=True)
            for ck, fac in single.terms.items():
                children.setdefault(ck, len(children))
                edges.append([pi, ck, fac])
                new.add(ck, coef * fac)
            if not single.terms:
                edges.append([pi, None, None])
        keys = list(children)
        nodes = []
        for k in keys:
            c = new.terms.get(k, Poly())
            nodes.append(dict(s=string_tex(k), c=c.tex(), v=poly_terms(c), key=list(k), zero=c.is_zero()))
        edge_out = []
        for pi, ck, fac in edges:
            if ck is None:
                edge_out.append(dict(a=pi, b=-1, f='0', cls='kill'))
                continue
            ftex = fac.tex()
            fork = any(k[0] for k in fac.t)
            cls = 'minus' if ftex.startswith('-') else ('plus' if fork else 'plain')
            edge_out.append(dict(a=pi, b=keys.index(ck), f=ftex, cls=cls))
        layers.append(dict(col=ci, nodes=nodes, edges=edge_out))
        O = Operator({k: v for k, v in new.terms.items()})
        keys_prev = keys
    mean = expect_zero(O)
    for p in (0.0, 0.01, 0.137):
        ref = density_check(ex['text'].replace('(p)', f'({p})'), ex['parity'], p)
        assert abs(ref - mean.value(p)) < 1e-12, (name, p)
    # the leftmost layer holds only strings of I and Z on |0>-initialised wires, after the resets: here only I
    return dict(title=ex['title'], lesson=ex['lesson'], qubits=ex['qubits'], cols=cols, layers=layers,
                mean=mean.tex(), mean_v=poly_terms(mean), records=ex['parity'])

# ------------------------------------------------------------------ the real d=3 check


def monomial_ratio(noisy, clean):
    """noisy = clean * l1^a l2^b f^c: return (a, b, c)."""
    (k1, v1), = noisy.t.items()
    (k0, v0), = clean.t.items()
    assert k1[0] == k0[0] and v1 == v0, (noisy.t, clean.t)
    return k1[1] - k0[1], k1[2] - k0[2], k1[3] - k0[3]


def mono_tex(e):
    a, b, c = e
    out = ''
    for name, k in (('\\lambda_1', a), ('\\lambda_2', b), ('f', c)):
        if k == 1:
            out += name
        elif k > 1:
            out += f'{name}^{{{k}}}'
    return out or '1'


WIRES = [6, 7, 8, 9, 2, 3, 1, 0, 11, 10, 14, 13, 12]
MOMENT_NAMES = ['\\text{reset}', 'D^\\dagger', '\\mathtt C_1', '\\mathtt C_2', '\\mathtt C_3', '\\mathtt C_4',
                'r_8', '\\text{reset}', '\\mathtt C_4', '\\mathtt C_3', '\\mathtt C_2', '\\mathtt C_1', 'D',
                'r_{9\\ldots14}']


def real_d3():
    code = RC.CODES[3]
    text = RC.D3_FILE.read_text()
    ops = parse(text)
    rec_of, _ = records(ops)
    moms = RC.moments(ops)
    moments = []
    for mi, mom in enumerate(moms):
        gates, noise = [], []
        for i in mom:
            line, name, arg, targets = ops[i]
            if name in ('DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS'):
                continue
            entry = dict(line=line, name=name, arg=arg, targets=targets)
            (noise if name in ('DEPOLARIZE1', 'DEPOLARIZE2', 'Z_ERROR', 'X_ERROR') else gates).append(entry)
        moments.append(dict(name=MOMENT_NAMES[mi], lines=[ops[mom[0]][0], ops[mom[-1]][0]], gates=gates, noise=noise))
    readouts = {'r8': [8], 'r9': [9], 'r11': [11], 'r8r11': [8, 11]}
    obs = {}
    for name, parity in readouts.items():
        _, _, hist0, O0 = RC.run_d3(parity, noisy=False)
        _, _, hist1, O1 = RC.run_d3(parity, noisy=True)
        steps = []
        prev = (0, 0, 0)
        for (mom, o0), (_, o1) in zip(hist0, hist1):
            assert set(o0.terms) == set(o1.terms)
            if o0.terms:
                k = next(iter(o0.terms))
                tot = monomial_ratio(o1.terms[k], o0.terms[k])
                for kk in o0.terms:
                    assert monomial_ratio(o1.terms[kk], o0.terms[kk]) == tot
            else:
                tot = prev
            gained = tuple(t - s for t, s in zip(tot, prev))
            d = RC.describe(o0, None)
            single = len(o0) == 1
            entry = dict(count=len(o0), letters=d['letters'], gained=mono_tex(gained), total=mono_tex(tot),
                         gained_e=list(gained), total_e=list(tot))
            if single:
                (k, c), = o0.terms.items()
                entry['string'] = string_tex(k, order=lambda q: WIRES.index(q) if q in WIRES else 99)
                entry['coef'] = c.tex()
            else:
                fx = factored(o0, code['sites'])
                assert fx is not None, name
                entry['factored'] = fx
            steps.append(entry)
            prev = tot
        # at the start: on the data only?
        data = RC.mask(code['sites'])
        on_data = all(((k[0] | k[1]) & ~data) == 0 for k in O0.terms)
        final = dict(count=len(O0), on_data=on_data)
        if on_data:
            acts = [RC.logical_action(k, code) for k in O0.terms]
            final['survivors'] = sum(a is not None for a in acts)
            final['as'] = {f'{"+" if s > 0 else "-"}{L}': sum(1 for a in acts if a == (s, L))
                           for s in (1, -1) for L in 'IXYZ' if any(a == (s, L) for a in acts)}
            final['mean'] = {}
            for ename, e in {'I': (0, 0), 'X_7': set_letter(0, 0, 7, 'X'), 'Y_7': set_letter(0, 0, 7, 'Y'),
                             'Z_7': set_letter(0, 0, 7, 'Z'), 'X_0X_3X_{10}': (RC.mask([0, 3, 10]), 0),
                             'X_0': set_letter(0, 0, 0, 'X'), 'Z_0Z_3': (0, RC.mask([0, 3]))}.items():
                m0 = RC.code_mean(O0, code, e)
                m1 = RC.code_mean(O1, code, e)
                final['mean'][ename] = dict(clean=m0.tex(), noisy=m1.tex(), clean_v=poly_terms(m0), noisy_v=poly_terms(m1))
        obs[name] = dict(parity=parity, steps=steps, final=final)
    # the paper's statements
    assert obs['r8']['final']['survivors'] == 16 and obs['r8']['final']['as'] == {'+X': 8, '-Y': 8}
    assert obs['r8']['final']['mean']['I']['clean'] == '1' and obs['r8']['final']['mean']['X_7']['clean'] == '0'
    assert obs['r8']['final']['mean']['I']['noisy'] == '\\lambda_1^{20}\\lambda_2^{12}f^{5}'
    assert obs['r8r11']['final']['count'] == 1
    stim_factor = RC.stim_check(8)
    lam = 1 - 4e-3 / 3
    assert abs(lam ** 20 * (1 - 16e-3 / 15) ** 12 * (1 - 2e-3) ** 5 - stim_factor) < 1e-13
    return dict(wires=WIRES, data=code['sites'], ancillas=[6, 8, 2, 1, 11, 14], moments=moments, readouts=obs,
                stim_factor_p001=stim_factor)

# ------------------------------------------------------------------ the toy pipeline

TOY = """RX 0
T_DAG 0
DEPOLARIZE1(p) 0
RX 1
Z_ERROR(p) 1
T 0
DEPOLARIZE1(p) 0
CX 1 0
DEPOLARIZE2(p) 1 0
T_DAG 0
DEPOLARIZE1(p) 0
MX(p) 1
T 0
MX 0
"""
TOY_LOCATIONS = 6       # injection, reset flip, three depolarisers, readout flip


def poly_p_value(c, p):
    return sum(float(v) * p ** i for i, v in enumerate(c))


def p_to_x(cp, N):
    """A(p) as a polynomial in p -> A-hat(x) = A(x/(1+x)) (1+x)^N, exactly."""
    out = [Fraction(0)] * (N + 1)
    for i, v in enumerate(cp):
        # p^i (1+x)^N / (1+x)^i = x^i (1+x)^(N-i)
        for j in range(N - i + 1):
            out[i + j] += v * math.comb(N - i, j)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def quotient(a, e, K):
    L = []
    for k in range(K + 1):
        v = (e[k] if k < len(e) else 0) - sum((a[j] if j < len(a) else 0) * L[k - j] for j in range(1, k + 1))
        L.append(Fraction(v))
    return L


def crt(residues, primes):
    x, m = 0, 1
    for r, q in zip(residues, primes):
        t = (r - x) * pow(m, -1, q) % q
        x += m * t
        m *= q
    return x, m


def toy():
    text = TOY
    ops = parse(text.replace('(p)', '(0.001)'))
    rec_of, nrec = records(ops)
    assert nrec == 2
    parities = {'r1': [1], 'r2': [2], 'r1r2': [1, 2]}
    means = {}
    for name, par in parities.items():
        O = Operator.identity()
        for st in reversed(step_ops(ops)):
            apply_backward(O, st, rec_of, set(par), noisy=True)
        means[name] = expect_zero(O)
    A = (ONE + means['r1']) * Fraction(1, 2)
    B = (ONE + means['r1'] - means['r2'] - means['r1r2']) * Fraction(1, 4)
    Ap, Bp = A.in_p(), B.in_p()
    for p in (0.0, 0.01, 0.05, 0.2):
        t = text.replace('(p)', f'({p})')
        e1 = density_check(t, [1], p)
        e2 = density_check(t, [2], p)
        e12 = density_check(t, [1, 2], p)
        assert abs((1 + e1) / 2 - poly_p_value(Ap, p)) < 1e-12
        assert abs((1 + e1 - e2 - e12) / 4 - poly_p_value(Bp, p)) < 1e-12
    # the response table: no injection noise, incoming E inserted as a gate
    table = {}
    base = text.replace('DEPOLARIZE1(p) 0\nRX 1', '{E}RX 1', 1)
    assert '{E}' in base
    for E in 'IXYZ':
        t = base.replace('{E}', '' if E == 'I' else f'{E} 0\n')
        ops_e = parse(t.replace('(p)', '(0.001)'))
        rec_e, _ = records(ops_e)
        m = {}
        for name, par in parities.items():
            O = Operator.identity()
            for st in reversed(step_ops(ops_e)):
                apply_backward(O, st, rec_e, set(par), noisy=True)
            m[name] = expect_zero(O)
        FA = (ONE + m['r1']) * Fraction(1, 2)
        FB = (ONE + m['r1'] - m['r2'] - m['r1r2']) * Fraction(1, 4)
        for pv in (0.01, 0.07):                     # every entry against a density matrix, not only their sum
            te = t.replace('(p)', f'({pv})')
            for name, par in parities.items():
                assert abs(density_check(te, par, pv) - m[name].value(pv)) < 1e-12, (E, name, pv)
        table[E] = dict(FA=FA, FB=FB, means={k: v.tex() for k, v in m.items()})
    # Eq. (19) in miniature: A = sum_E mu(E) F_A(E), with mu from the injection depolariser
    mu = {'I': [Fraction(1), Fraction(-1)], 'X': [Fraction(0), Fraction(1, 3)], 'Y': [Fraction(0), Fraction(1, 3)],
          'Z': [Fraction(0), Fraction(1, 3)]}

    def pmul(a, b):
        r = [Fraction(0)] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                r[i + j] += x * y
        return r

    def padd(a, b):
        n = max(len(a), len(b))
        return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]

    A2, B2 = [Fraction(0)], [Fraction(0)]
    for E in 'IXYZ':
        A2 = padd(A2, pmul(mu[E], table[E]['FA'].in_p()))
        B2 = padd(B2, pmul(mu[E], table[E]['FB'].in_p()))
    trim = lambda v: v[:max(i for i, c in enumerate(v) if c) + 1] if any(v) else [Fraction(0)]
    assert trim(A2) == trim(Ap) and trim(B2) == trim(Bp), 'sum over E does not give A and B'
    # series in x
    N = TOY_LOCATIONS
    a = p_to_x(Ap, N)
    e = p_to_x(Bp, N)
    K = 8
    L = quotient(a + [Fraction(0)] * (K + 1 - len(a)), e + [Fraction(0)] * (K + 1 - len(e)), K)
    # integers U_k = scale_k a_k with scale_k = 4 * 15^k, residues modulo small primes q = 1 (mod 8)
    primes = [17, 41, 73, 89, 97]
    rows = []
    for k in range(len(a)):
        scale = 4 * 15 ** k
        for key, v in (('a', a[k]), ('e', e[k] if k < len(e) else Fraction(0))):
            U = v * scale
            assert U.denominator == 1 and 0 <= U.numerator <= scale * math.comb(N, k)
            U = U.numerator
            res = [U % q for q in primes]
            need = 1
            while math.prod(primes[:need]) <= scale * math.comb(N, k):
                need += 1
            x, m = crt(res[:need], primes[:need])
            assert x == U
            rows.append(dict(k=k, key=key, value=frac_str(v), scale=scale, U=U, bound=scale * math.comb(N, k),
                             residues=res, need=need))
    # the same residues from the network itself: transfer matrices in x-form, contracted modulo each prime
    net_check = toy_network_residues(primes, N)
    for qi, q in enumerate(primes):
        for key, series in (('a', a), ('e', e)):
            for k, v in enumerate(series):
                assert net_check[q][key][k] == (v.numerator * pow(v.denominator, -1, q)) % q, (q, key, k)
    return dict(text=TOY, locations=N,
                A_p=[frac_str(v) for v in Ap], B_p=[frac_str(v) for v in Bp],
                A_tex=A.tex(), B_tex=B.tex(),
                means={k: v.tex() for k, v in means.items()},
                table={E: dict(FA=[frac_str(v) for v in d['FA'].in_p()], FB=[frac_str(v) for v in d['FB'].in_p()],
                               FA_tex=d['FA'].tex(), FB_tex=d['FB'].tex(), means=d['means'])
                       for E, d in table.items()},
                mu={E: [frac_str(v) for v in m] for E, m in mu.items()},
                a=[frac_str(v) for v in a], e=[frac_str(v) for v in e], L=[frac_str(v) for v in L],
                primes=primes, rows=rows, sqrt2={q: sqrt2_mod(q) for q in primes})


def sqrt2_mod(q):
    for r in range(1, q):
        if r * r % q == 2:
            return min(r, q - r)
    raise ValueError(q)


def toy_network_residues(primes, N):
    """Contract the toy's Pauli-transfer network modulo q, every entry a polynomial in x truncated at x^N:
    each location is diag(1 + x, lambda-hat, ...) with lambda-hat = 1 - x/3, 1 - x/15 or 1 - x."""
    text = TOY.replace('(p)', '(0.001)')
    ops = parse(text)
    rec_of, _ = records(ops)
    steps = step_ops(ops)
    out = {}
    for q in primes:
        r2 = sqrt2_mod(q)
        inv_r2 = pow(r2, -1, q)
        K = N

        def pm(a, b):
            r = [0] * (K + 1)
            for i, x in enumerate(a):
                if x:
                    for j, y in enumerate(b[:K + 1 - i]):
                        r[i + j] = (r[i + j] + x * y) % q
            return r

        def coef_mod(c):
            """A Poly with only s (no noise symbols) as an integer mod q."""
            tot = 0
            for (s, a1, a2, af), v in c.t.items():
                assert a1 == a2 == af == 0
                tot += v.numerator * pow(v.denominator, -1, q) * (inv_r2 if s else 1)
            return tot % q

        def lam_hat(kind, word):
            third, fifteenth = pow(3, -1, q), pow(15, -1, q)
            if all(l == 'I' for l in word):
                return [1, 1] + [0] * (K - 1)
            if kind == 'DEPOLARIZE1':
                return [1, (-third) % q] + [0] * (K - 1)
            if kind == 'DEPOLARIZE2':
                return [1, (-fifteenth) % q] + [0] * (K - 1)
            axis = kind[0]
            return ([1, 1] if word in (axis,) else [1, q - 1]) + [0] * (K - 1)

        results = {}
        for key, par in (('r1', {1}), ('r2', {2}), ('r1r2', {1, 2})):
            # operator: {string: polynomial mod q}
            O = {(0, 0): [1] + [0] * K}
            for st in reversed(steps):
                new = {}
                if st['kind'] == 'noise':
                    for g in st['groups']:
                        new = {}
                        for k_, c in O.items():
                            word = ''.join(letter(*k_, qq) for qq in g)
                            new[k_] = pm(c, lam_hat(st['gate'], word))
                        O = new
                    continue
                if st['kind'] == 'measure':
                    for j in reversed(range(len(st['targets']))):
                        qq = int(st['targets'][j])
                        pk = set_letter(0, 0, qq, 'X' if st['gate'] == 'MX' else 'Z')
                        r = rec_of[(st['op'], j)]
                        new = {}
                        for k_, c in O.items():
                            from engine import anticommutes
                            if anticommutes(k_, pk):
                                continue
                            if r in par:
                                power, nk = mul_strings(pk, k_)
                                cc = c if power == 0 else [(-v) % q for v in c]
                                if st['p'] is not None:
                                    cc = pm(cc, [1, q - 1] + [0] * (K - 1))      # f-hat = 1 - x
                                new[nk] = [(u + v) % q for u, v in zip(new.get(nk, [0] * (K + 1)), cc)]
                            else:
                                cc = c
                                if st['p'] is not None:
                                    cc = pm(cc, [1, 1] + [0] * (K - 1))          # an unread flip: 1 + x
                                new[k_] = [(u + v) % q for u, v in zip(new.get(k_, [0] * (K + 1)), cc)]
                        O = new
                    continue
                single = Operator()
                acc = {}
                for k_, c in O.items():
                    t = Operator({k_: ONE})
                    apply_backward(t, st, rec_of, par, noisy=False)
                    for nk, f in t.terms.items():
                        fm = coef_mod(f)
                        acc[nk] = [(u + fm * v) % q for u, v in zip(acc.get(nk, [0] * (K + 1)), c)]
                O = {k_: c for k_, c in acc.items() if any(c)}
            mean = [0] * (K + 1)
            for (x, z), c in O.items():
                if x == 0:
                    mean = [(u + v) % q for u, v in zip(mean, c)]
            results[key] = mean
        # A-hat = ((1+x)^N + mean_r1)/2, B-hat = ((1+x)^N + r1 - r2 - r1r2)/4, (1+x)^N for the identity parity
        onepx = [math.comb(N, k) % q for k in range(K + 1)]
        inv2, inv4 = pow(2, -1, q), pow(4, -1, q)
        a_hat = [(u + v) * inv2 % q for u, v in zip(onepx, results['r1'])]
        e_hat = [(u + v - w - y) * inv4 % q for u, v, w, y in zip(onepx, results['r1'], results['r2'], results['r1r2'])]
        out[q] = {'a': a_hat, 'e': e_hat}
    return out

# ------------------------------------------------------------------ the repository's stored residues


def stored():
    out = {}
    for dist, key, k in ((3, 'B', 2), (5, 'B', 3), (3, 'A', 1), (5, 'A', 10), (5, 'B', 10)):
        series = json.loads((REPO / f'data/series/d{dist}.json').read_text())
        recs = series['modular_records']
        qs = [int(r['certificate']['prime']) for r in recs]
        count = series['reconstruction_primes']
        scale = 2 ** series['conservative_dyadic_power'] * 15 ** k
        N = series['retained_physical_locations']
        height = scale * math.comb(N, k)
        rows = []
        for r, q in zip(recs, qs):
            if dist == 5:
                chars = sorted(r['characters'], key=lambda c: c['character'])
                ch = [int(c['coefficients'][key][k]) for c in chars]
                res = sum(ch) * pow(8, -1, q) % q
                rows.append(dict(q=str(q), k_odd=r['certificate']['odd_k'], m=r['certificate']['power_of_two'],
                                 witness=r['certificate']['witness'], characters=[str(v) for v in ch], residue=str(res)))
            else:
                res = int(r['coefficients'][key][k])
                rows.append(dict(q=str(q), k_odd=r['certificate']['odd_k'], m=r['certificate']['power_of_two'],
                                 witness=r['certificate']['witness'], residue=str(res)))
        scaled = [scale * int(row['residue']) % q for row, q in zip(rows, qs)]
        U, M = crt(scaled[:count], qs[:count])
        assert M > height and 0 <= U <= height
        for q, s in zip(qs[count:], scaled[count:]):
            assert U % q == s
        value = Fraction(U, scale)
        assert value == Fraction(series['raw_coefficients'][key][k])
        for row, q, s in zip(rows, qs, scaled):
            row['scaled'] = str(s)
        out[f'd{dist}_{key}{k}'] = dict(distance=dist, key=key, k=k, scale=str(scale),
                                       scale_tex=f'2^{{{series["conservative_dyadic_power"]}}}\\cdot15^{{{k}}}',
                                       N=N, height=str(height), height_bits=height.bit_length(), primes=rows,
                                       reconstruction=count, U=str(U), modulus_bits=M.bit_length(),
                                       value=frac_str(value))
    assert out['d5_B3']['value'] == '574/375' and out['d3_B2']['value'] == '32/75'
    assert out['d5_A10']['value'] == '7771271684969588874728818721617/147622500000000'
    # a_10 at d=5 needs all three primes: its scaled residues differ
    qs5 = sorted(int(r['q']) for r in out['d5_A10']['primes'])
    assert int(out['d5_A10']['U']) > qs5[-1] * qs5[-2], 'a_10 at d=5 must need all three primes'
    assert int(out['d5_B10']['U']) < qs5[0] * qs5[1]
    return out


def gate_tables():
    """U^dag P U for the explorer: {gate: {word: [[word, coefficient TeX], ...]}}."""
    out = {}
    for g in ('H', 'S', 'S_DAG', 'T', 'T_DAG', 'CX', 'CZ'):
        out[g] = {w: [[nw, c.tex()] for nw, c in terms] for w, terms in TABLES[g].items()}
    # spot checks against the paper
    assert out['T']['X'] == [['X', '\\frac{1}{\\sqrt2}'], ['Y', '-\\frac{1}{\\sqrt2}']]
    assert out['T_DAG']['X'] == [['X', '\\frac{1}{\\sqrt2}'], ['Y', '\\frac{1}{\\sqrt2}']]
    assert out['CX']['XI'] == [['XX', '1']] and out['CX']['IZ'] == [['ZZ', '1']]
    return out


def toy_transfer():
    """The toy circuit as transfer matrices on the 16 two-qubit labels, one per column and readout parity.
    Label index = l(q0) + 4 l(q1), l: I=0, X=1, Y=2, Z=3."""
    text = TOY.replace('(p)', '(0.001)')
    ops = parse(text)
    rec_of, _ = records(ops)
    steps = step_ops(ops)
    L = 'IXYZ'
    keys = []
    for i in range(16):
        x = z = 0
        for q, l in ((0, L[i % 4]), (1, L[i // 4])):
            x, z = set_letter(x, z, q, l)
        keys.append((x, z))
    index = {k: i for i, k in enumerate(keys)}
    cols = []
    for st in steps:
        line, op, arg, targets = ops[st['op']]
        cols.append(dict(gate=op, qubits=[int(t) for t in targets], label=column_label(op, arg, targets),
                         kind=st['kind'], noisy=arg is not None))
    mats = {}
    for name, par in (('r1', {1}), ('r2', {2}), ('r1r2', {1, 2})):
        per = []
        for st in steps:
            entries = []
            for j, key in enumerate(keys):
                single = Operator({key: ONE})
                apply_backward(single, st, rec_of, par, noisy=True)
                for nk, c in single.terms.items():
                    entries.append([index[nk], j, poly_terms(c)])
            per.append(entries)
        mats[name] = per
    # the input |00>: <P> = 1 for strings of I and Z
    init = [1 if (L[i % 4] in 'IZ' and L[i // 4] in 'IZ') else 0 for i in range(16)]
    # check: backward contraction reproduces the engine's means
    def value(entries_list, p):
        v = [0.0] * 16
        v[0] = 1.0
        for entries in reversed(entries_list):
            w = [0.0] * 16
            for r, c_, terms in entries:
                w[r] += v[c_] * Poly({(a, b, cc, d): Fraction(n, m) for a, b, cc, d, n, m in terms}).value(p)
            v = w
        return sum(a * b for a, b in zip(init, v))
    for p in (0.0, 0.03):
        t = TOY.replace('(p)', f'({p})')
        for name, par in (('r1', [1]), ('r2', [2]), ('r1r2', [1, 2])):
            assert abs(value(mats[name], p) - density_check(t, par, p)) < 1e-12
    return dict(cols=cols, labels=[L[i % 4] + L[i // 4] for i in range(16)], init=init, mats=mats)


def factored(O, data_sites):
    """TeX of an operator that is (prod over data of (X_q +- Y_q)/sqrt2) times one fixed string elsewhere."""
    data = RC.mask(data_sites)
    rest = None
    signs = {}
    for (x, z), c in O.terms.items():
        r = (x & ~data, z & ~data)
        if rest is None:
            rest = r
        if r != rest:
            return None
    n = len(data_sites)
    # find per-site sign by comparing a term with Y on one site to the all-X term
    allx = (data | rest[0], rest[1])
    if allx not in O.terms:
        return None
    c0 = O.terms[allx]
    for q in data_sites:
        k = set_letter(*allx, q, 'Y')
        if k not in O.terms:
            return None
        signs[q] = '+' if O.terms[k] == c0 else '-' if O.terms[k] == -c0 else None
        if signs[q] is None:
            return None
    # rebuild and compare
    R = Operator({rest: ONE})
    for q in data_sites:
        nxt = Operator()
        for key, c in R.terms.items():
            nxt.add(set_letter(*key, q, 'X'), c)
            nxt.add(set_letter(*key, q, 'Y'), c if signs[q] == '+' else -c)
        R = nxt
    scaled = Operator({k: v * c0 for k, v in R.terms.items()})
    if scaled.terms != O.terms:
        return None
    plus = [q for q in data_sites if signs[q] == '+']
    minus = [q for q in data_sites if signs[q] == '-']
    rest_tex = string_tex(rest) if rest != (0, 0) else ''
    body = ''
    if plus:
        body += f'\\prod_{{q\\in\\mathcal D}}(X_q+Y_q)' if not minus else ''.join(f'(X_{{{q}}}+Y_{{{q}}})' for q in plus)
    if minus:
        body += ''.join(f'(X_{{{q}}}-Y_{{{q}}})' for q in minus)
    ctex = c0.tex()
    if len(c0.t) == 1:
        (sb, e1, e2, ef), v = next(iter(c0.t.items()))
        mag = abs(v)
        if e1 == e2 == ef == 0 and mag.numerator == 1 and mag.denominator & (mag.denominator - 1) == 0:
            k = 2 * (mag.denominator.bit_length() - 1) + sb
            ctex = ('-' if v < 0 else '') + f'2^{{-{k}/2}}'
    return dict(tex=f'{ctex}\\,{body}' + (f'\\;{rest_tex}' if rest_tex else ''), coef=ctex, rest=rest_tex)


def detector_table():
    """Every declared d=3 detector carried back through the whole noiseless circuit: the number of strings after
    each operation, the forks and merges at T layers, and the mean at the start, next to the number of defects of
    its drawn semiweb in the repository's detector-web guide (pauli_semiwebs_check/detector_webs/index.tsv)."""
    import csv
    import re
    text = RC.D3_FILE.read_text()
    ops = parse(text)
    rec_of, nrec = records(ops)
    steps = step_ops(ops)
    dets, k = [], 0
    for i, (line, name, arg, targets) in enumerate(ops):
        if name in ('M', 'MX', 'MPP'):
            k += len(targets)
        if name == 'DETECTOR':
            recs = sorted(k + 1 + int(re.match(r'rec\[(-\d+)\]', t).group(1)) for t in targets if t.startswith('rec'))
            dets.append(dict(line=line, records=recs))
    index = {int(r['detector']): r for r in csv.DictReader(
        (REPO / 'pauli_semiwebs_check/detector_webs/index.tsv').open(), delimiter='\t') if r['circuit'] == '3'}
    tline = {ops[st['op']][0]: st['gate'] for st in steps if st['gate'] in ('T', 'T_DAG')}
    out = []
    for d, det in enumerate(dets):
        row = index[d]
        assert ' '.join(str(r - 1) for r in det['records']) == row['records']
        O = Operator.identity()
        trace, forks = [], []
        for st in reversed(steps):
            before = len(O)
            apply_backward(O, st, rec_of, set(det['records']), noisy=False)
            trace.append([st['line'], len(O)])
            if st['gate'] in ('T', 'T_DAG') and len(O) != before:
                forks.append([st['line'], st['gate'], before, len(O)])
        mean = expect_zero(O)
        assert mean == ONE, (d, mean.tex())
        defects = int(row['defects'])
        most = max(c for _, c in trace)
        assert (defects == 0) == (most == 1), d
        out.append(dict(detector=d, line=det['line'], records=det['records'], defects=defects, most=most,
                        forks=forks, trace=trace))
    webs = [r['detector'] for r in out if r['defects'] == 0]
    assert [r['detector'] for r in out if r['defects']] == [7, 14, 16, 18] and len(webs) == 16
    return dict(detectors=out, t_lines=sorted(tline), last_line=max(l for l, *_ in ops))


def check_strengths():
    from engine import single_strength
    for path in (RC.D3_FILE, RC.D5_FILE):
        single_strength(parse(path.read_text()))
    for ex in SE.EXAMPLES.values():
        single_strength(parse(ex['text'].replace('(p)', '(0.001)')))
    single_strength(parse(TOY.replace('(p)', '(0.001)')))


def main():
    check_strengths()
    data = dict(small={name: tree_example(name) for name in SE.EXAMPLES}, real=real_d3(), d5=RC.d5_central(),
                toy=toy(), stored=stored(), gates=gate_tables(), toynet=toy_transfer(), dets=detector_table())
    text = json.dumps(data, separators=(',', ':'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print('wrote', OUT.relative_to(REPO), OUT.stat().st_size // 1024, 'KB')
    t = data['toy']
    print('toy A(p) =', t['A_p'], ' B(p) =', t['B_p'])
    print('toy a_k =', t['a'], ' e_k =', t['e'], ' L_k =', t['L'])
    for k, v in data['stored'].items():
        print(k, v['value'], 'from', len(v['primes']), 'primes, U =', v['U'][:30], '...')
    embedded = PAGE.read_text().split('window.ATLAS_PROP = ', 1)[1].split(';</script>', 1)[0]
    assert embedded == text, 'the recomputed data differ from those embedded in ' + PAGE.name
    print('identical to the data embedded in', PAGE.name)


if __name__ == '__main__':
    main()
