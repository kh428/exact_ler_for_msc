"""One drawn web for every declared detector.

Every declared detector is a closed, defect-free Pauli web of the S proxy, the
circuit with each T replaced by S and each T_dag by S_dag (Clifford completeness).
The proxy graph has the same nodes and edges as the T graph, and only the T sites
differ in phase. Read on the T graph, a proxy web therefore stays defect-free at
every spider except T spiders. There, an edge pair X in, Y out (or Y in, X out)
is defect-free at S but carries a defect of pi/2 at T and -pi/2 at T_dag.

Two proxy webs with the same records differ by a record-free web, so the webs of
one detector form a coset of the record-free webs (64 elements in both circuits).
The drawn web has the fewest T defects in its coset, then the fewest labelled edges.
A detector is phase-independent exactly when that minimum is zero.
"""

from itertools import combinations

from src.generate import local_scalar
from src.semiweb import BITS
from .. import circuit, graph, oracle, webs


def defects(g, inc, vector):
    """{node: delta in units of pi/4}, as src.semiweb.defects, with a fast incidence."""
    out = {}
    for k, n in g['nodes'].items():
        if n['kind'] not in ('X', 'Z'):
            continue
        es = inc[k]
        opposite = 0 if n['kind'] == 'Z' else 1
        o = (vector >> (2 * es[0] + opposite)) & 1
        parity = sum((vector >> (2 * j + 1 - opposite)) & 1 for j in es) % 2
        delta = (-2 * o * n['phase'] + 4 * parity) % 8
        if delta:
            out[k] = delta if delta <= 4 else delta - 8
    return out


def scalar_exponent(g, inc, vector):
    """lambda_0 as an exponent in units of pi/4, with the original phases."""
    exponent = 0
    for k, n in g['nodes'].items():
        if n['kind'] == 'boundary':
            continue
        labels = webs.labels_at(vector, inc[k])
        if n['kind'] == 'H':
            exponent += 4 if labels[0] == 'Y' else 0
        else:
            exponent += local_scalar(n['kind'], n['phase'], labels)
    exponent += 4 * sum(((vector >> (2 * j)) & 3) == 3 for j in range(len(g['edges'])))
    return exponent % 8


def record_support(g, inc, vector):
    support = 0
    for k, n in g['nodes'].items():
        if 'record' in n and n['kind'] in ('X', 'Z'):
            if BITS[webs.labels_at(vector, inc[k][:1])[0]][0 if n['kind'] == 'Z' else 1]:
                support ^= 1 << n['record']
    return support


def detector_webs(key):
    parsed = circuit.load(key)
    discard = oracle.discard_states(parsed)
    g = graph.build(parsed, discard)
    pg = graph.build(parsed, discard, phase_override=oracle.proxy_phases(parsed['tsites']))
    assert list(pg['nodes']) == list(g['nodes']) and pg['edges'] == g['edges']
    inc = webs.incidence(g)
    basis, _ = webs.closed_web_basis(pg)

    # Eliminate on record supports, keeping the web that produces each pivot row.
    pivots, free = {}, []
    for w in basis:
        s = record_support(pg, inc, w)
        while s:
            top = s.bit_length() - 1
            if top not in pivots:
                pivots[top] = (s, w)
                break
            s ^= pivots[top][0]
            w ^= pivots[top][1]
        if not s:
            free.append(w)
    assert len(free) <= 12, 'coset too large to enumerate'
    cosets = [0]
    for f in free:
        cosets += [c ^ f for c in cosets]

    # The T-graph web span, to label each detector independently of the drawing.
    t_eqs = webs.web_checks(g)['equations']
    nrec = len(parsed['records'])

    out = []
    for d in parsed['detectors']:
        target = sum(1 << r for r in d['records'])
        s, w = target, 0
        while s:
            top = s.bit_length() - 1
            s ^= pivots[top][0]
            w ^= pivots[top][1]
        best = None
        for c in cosets:
            v = w ^ c
            dft = defects(g, inc, v)
            assert all(g['nodes'][k].get('site') is not None for k in dft), 'defect away from a T spider'
            score = (len(dft), sum(((v >> (2 * j)) & 3) != 0 for j in range(len(g['edges']))))
            if best is None or score < best[0]:
                best = (score, v, dft)
        (count, edges), v, dft = best
        assert record_support(g, inc, v) == target
        proxy_exponent = scalar_exponent(pg, inc, v)
        assert proxy_exponent in (0, 4), 'proxy web with a non-real scalar'
        in_span = oracle.contains(t_eqs, nrec, target, proxy_exponent // 4)
        assert in_span == (count == 0), 'defect count disagrees with the web span'
        out.append(dict(index=d['index'], line=d['line'], records=d['records'], vector=v,
                        defects=dft, labelled_edges=edges, proxy_value=proxy_exponent // 4,
                        scalar_pi_over_4=scalar_exponent(g, inc, v), phase_independent=in_span))
    return dict(parsed=parsed, graph=g, proxy_graph=pg, incidence=inc, detectors=out,
                record_free=len(free))


if __name__ == '__main__':
    for key in (3, 5):
        r = detector_webs(key)
        dep = [d for d in r['detectors'] if not d['phase_independent']]
        print(key, 'record-free', r['record_free'], 'phase-dependent', [d['index'] for d in dep])
        for d in r['detectors']:
            print(f"  D{d['index']:<3} recs {d['records']} value {d['proxy_value']} "
                  f"edges {d['labelled_edges']} defects {sorted(d['defects'].values())} "
                  f"lambda0 e^(i pi {d['scalar_pi_over_4']}/4)")
