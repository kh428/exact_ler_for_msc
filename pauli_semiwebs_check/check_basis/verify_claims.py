"""Independent checks of every scientific statement in the detector-web guide.

Run from pauli_semiwebs_check:  python -B -m check_basis.verify_claims
Each check prints PASS or FAIL with the numbers behind it.

 1. Readout algebra: T^dag Y T = T X T^dag = (X+Y)/sqrt2, T Y T^dag = -T^dag X T.
 2. The noiseless logical readout (T layer, then MPP Y_L) measures the same transversal
    operator as the last double check (entry layer, X check, inverse layer), read from
    the source lines.
 3. d=3 statevector: with the paper's readout every record is deterministic, with the
    bare MPP Y_L readout of ../pauli_semiwebs/circuits/d3_native_t.txt the Y record is not.
 4. The paper's exact series (data/series in this repository) start with a_0 = 1 and
    e_0 = 0 at d=3 and d=5: the noiseless T circuit passes every detector and reads
    the logical correctly.
 5. A closed labelling that keeps the phase of every non-T spider and has no
    opposite-colour support on any T spider only reaches the web span: every closed
    semiweb for a phase-dependent detector has opposite-colour support on a T spider.
 6. Each phase-dependent detector is random for some Clifford choice of the T angles.
 7. Each drawn web's coset has 2^6 = 64 proxy webs.
 8. Each phase-dependent detector contains a central double-check readout, or an X
    stabilizer measured before and after the d=5 double check.
"""

import csv
import math
from pathlib import Path

import numpy as np
import stim

from src.linear import kernel, rank, xor_sum
from . import REPO, SEMIWEBS, circuit, graph, oracle, statevector, webs
from .guide.select import detector_webs, record_support

SERIES = REPO / 'data' / 'series'
# Lines of the last double check's entry layer and of the logical readout layer.
LAYERS = {3: dict(entry=[135], exit=[177], readout=[192]),
          5: dict(entry=[391, 392], exit=[441, 442], readout=[491, 492])}
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)


def gate(phase):
    return np.diag([1, np.exp(1j * np.pi * phase / 4)])


def report(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def check_algebra():
    T, Td = gate(1), gate(-1)
    h = (X + Y) / math.sqrt(2)
    ok = (np.allclose(Td @ Y @ T, h) and np.allclose(T @ X @ Td, h)
          and np.allclose(T @ Y @ Td, -(Td @ X @ T)) and np.allclose(T @ h @ Td, Y))
    return report('1 readout algebra', ok, 'T^dag Y T = T X T^dag = (X+Y)/sqrt2 and T H_XY T^dag = Y')


def layer(parsed, lines):
    return {s['q']: s['phase'] for s in parsed['tsites'] if s['line'] in lines}


def check_readout_operator(key):
    parsed = circuit.load(key)
    data = sorted(q for op in parsed['ops'] if op['op'] == 'MPP' and op['factors'][0][0] == 'Y'
                  for _, q in op['factors'])
    L = LAYERS[key]
    entry, exit_, readout = (layer(parsed, L[k]) for k in ('entry', 'exit', 'readout'))
    ok = sorted(entry) == sorted(exit_) == sorted(readout) == data
    ok &= all((entry[q] + exit_[q]) % 8 == 0 for q in data)       # exit undoes entry
    sign = 1
    for q in data:
        U, V = gate(entry[q]), gate(readout[q])
        check, read = U.conj().T @ X @ U, V.conj().T @ Y @ V       # measured before the layer
        ratio = [read[i, j] / check[i, j] for i in range(2) for j in range(2) if abs(check[i, j]) > 1e-12]
        ok &= np.allclose(ratio, ratio[0]) and np.isclose(abs(ratio[0]), 1) and np.isclose(ratio[0].imag, 0)
        sign *= round(ratio[0].real)
    ok &= sign == 1
    return report(f'2 d={key} readout = last double check', ok,
                  f'{len(data)} data qubits, exit layer inverts entry, site-by-site operators agree, overall sign {sign:+d}')


def check_statevector():
    paper = circuit.SOURCES[3].read_text()
    bare = (SEMIWEBS / 'circuits' / 'd3_native_t.txt').read_text()
    ok, notes = True, []
    for label, text, want_random in (('paper readout', paper, set()), ('bare MPP Y_L', bare, {14})):
        randoms = set()
        for seed in range(6):
            p, prob, bit = statevector.run(text, seed)
            randoms |= {r for r, x in prob.items() if x < 1 - 1e-9}
            fired = [d['index'] for d in p['detectors'] if sum(bit[r] for r in d['records']) % 2]
            ok &= not fired
        ok &= randoms == want_random
        if want_random:
            p14 = max(prob[14], 1 - prob[14])
            ok &= np.isclose(p14, math.cos(math.pi / 8) ** 2)
            notes.append(f'{label}: only r14 random, likely outcome {p14:.4f} = cos^2(pi/8)')
        else:
            notes.append(f'{label}: all {len(prob)} records deterministic, no detector fires')
    return report('3 d=3 statevector', ok, '; '.join(notes))


def check_series():
    ok, notes = True, []
    for d in (3, 5):
        rows = {r['k']: r for r in csv.DictReader(open(SERIES / f'd{d}_coefficients.csv'))}
        a0, e0 = rows['0']['a_k_exact'], rows['0']['e_k_exact']
        ok &= a0 == '1' and e0 == '0'
        first = min(int(k) for k, r in rows.items() if r['e_k_exact'] != '0')
        notes.append(f'd={d}: a_0={a0}, e_0={e0}, first nonzero e_k at k={first}')
    return report('4 exact series, noiseless order', ok, '; '.join(notes))


def phase_free_constraints(g):
    """Web rows at every spider except T spiders, where only o = 0 is imposed."""
    inc = webs.incidence(g)
    rows = []
    for k, n in g['nodes'].items():
        es = inc[k]
        if n['kind'] == 'boundary':
            rows += [1 << (2 * es[0]), 1 << (2 * es[0] + 1)]
            continue
        if n['kind'] == 'H':
            a, b = es
            rows += [(1 << (2 * a)) ^ (1 << (2 * b + 1)), (1 << (2 * a + 1)) ^ (1 << (2 * b))]
            continue
        opp = 0 if n['kind'] == 'Z' else 1
        opposite = [1 << (2 * j + opp) for j in es]
        same = [1 << (2 * j + 1 - opp) for j in es]
        rows += [opposite[0] ^ x for x in opposite[1:]]
        if n['phase'] % 2:
            rows.append(opposite[0])                  # no opposite-colour support, parity free
        else:
            rows.append(xor_sum(same) ^ (opposite[0] if n['phase'] % 4 else 0))
    return [r for r in rows if r], inc


def check_opposite_colour_needed(key):
    parsed = circuit.load(key)
    g = graph.build(parsed, oracle.discard_states(parsed))
    rows, inc = phase_free_constraints(g)
    relaxed = [record_support(g, inc, w) for w in kernel(rows, 2 * len(g['edges']))]
    web_supports = [v for v, _ in webs.web_checks(g)['equations']]
    r_web, r_relaxed = rank(web_supports), rank(relaxed)
    r_union = rank(web_supports + relaxed)
    outside = [d for d in parsed['detectors']
               if rank(web_supports + [sum(1 << r for r in d['records'])]) > r_web]
    ok = r_web == r_relaxed == r_union
    return report(f'5 d={key} opposite colour on a T spider is needed', ok,
                  f'record span with T parity freed = {r_relaxed}, web span = {r_web}, '
                  f'{len(outside)} declared detectors have supports outside it')


def check_random_somewhere(key):
    parsed = circuit.load(key)
    dep = [d for d in select_cache(key) if not d['phase_independent']]
    found = {}
    for label, phases in oracle.assignments(parsed['tsites'], 20260929, 40):
        c = stim.Circuit(circuit.substituted_text(parsed['text'], parsed['tsites'], phases)).without_noise()
        s = c.compile_sampler().sample(128)
        for d in dep:
            if d['index'] in found:
                continue
            parity = np.bitwise_xor.reduce(s[:, d['records']], axis=1)
            if parity.min() != parity.max():
                found[d['index']] = label
        if len(found) == len(dep):
            break
    ok = len(found) == len(dep)
    return report(f'6 d={key} phase-dependent detectors are random somewhere', ok,
                  f'{len(found)} of {len(dep)} random under some Clifford choice of the T angles')


_cache = {}


def select_cache(key):
    if key not in _cache:
        _cache[key] = detector_webs(key)
    return _cache[key]['detectors']


def check_coset(key):
    select_cache(key)
    free = _cache[key]['record_free']
    return report(f'7 d={key} coset size', free == 6, f'{free} record-free proxy webs, coset of {2 ** free}')


def check_anatomy(key):
    parsed = circuit.load(key)
    recs = parsed['records']
    central = {r['index'] for r in recs if r['kind'] == 'MX' and r['target'] in ('7', '25')
               and any(op['op'] == 'MX' and op['q'] == int(r['target']) and op['record'] == r['index']
                       for op in parsed['ops'])}
    central = {i for i in central if recs[i]['line'] in (154, 157, 415)}
    last_check = 386 if key == 5 else 131
    ok, lines = True, []
    for d in select_cache(key):
        if d['phase_independent']:
            continue
        rs = d['records']
        has_central = bool(set(rs) & central)
        before = [r for r in rs if recs[r]['kind'] == 'MX' and recs[r]['line'] < last_check and r not in central]
        after = [r for r in rs if recs[r]['kind'] == 'MPP' and recs[r]['target'].startswith('X')]
        across = bool(before) and bool(after)
        ok &= has_central or across
        lines.append(f"D{d['index']}:{'readout' if has_central else ''}{'+' if has_central and across else ''}"
                     f"{'X across' if across else ''}")
    return report(f'8 d={key} anatomy', ok, ', '.join(lines))


def main():
    results = [check_algebra()]
    for key in (3, 5):
        results.append(check_readout_operator(key))
    results.append(check_statevector())
    results.append(check_series())
    for key in (3, 5):
        results += [check_opposite_colour_needed(key), check_random_somewhere(key), check_coset(key),
                    check_anatomy(key)]
    print(f'\n{sum(results)} of {len(results)} checks pass')
    return sum(results), len(results)


if __name__ == '__main__':
    passed, total = main()
    raise SystemExit(0 if passed == total else 1)
