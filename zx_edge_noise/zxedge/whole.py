"""Exact series of A, B and P_L for a whole circuit contracted as one network (circuits of d=3 size).

  python -B -m zxedge.whole --circuit clifft_d3_p001.stim --noise specs/edges_uniform.json --degree 10 --primes 8 \\
      --output results/d3_edges_uniform.json
"""
import argparse, json, time
from pathlib import Path

from . import INPUTS
from . import circuit as C, noise as N, series as S
from .network import build, to_engine, PolynomialRing
from .elimination import eliminate
from .engine import plan, contract

PLANS = {}


def planned(E, minutes, target):
    """A cotengra plan for this structure, found once and reused (every prime gives the same structure)."""
    key = tuple(tuple(i) for i, _ in E.tensors), tuple(E.output)
    if key not in PLANS:
        PLANS[key] = plan(E, minutes=minutes, target=target)[0]
    return PLANS[key]


def series_mod(circuit, locations, degree, prime, root2, *, minutes=0.1, target=27):
    """The coefficients of A-hat and B-hat modulo one prime."""
    ring = PolynomialRing(prime, degree)
    out = {}
    for observable in ('A', 'B'):
        net, _, _ = build(circuit, locations, ring, root2, observable=observable)
        net, _ = eliminate(net)
        E = to_engine(net, ring)
        E.simplify(log=False)
        out[observable] = [int(v) for v in contract(E, planned(E, minutes, target), log_every=0)]
    return out


def exact(circuit, locations, degree, count, **kw):
    """A-hat, B-hat and P_L as exact fractions from count primes (the last held out), and the residues."""
    runs = {}
    for prime, root2 in S.primes(count):
        runs[prime] = dict(sqrt2=root2, **series_mod(circuit, locations, degree, prime, root2, **kw))
    A = S.reconstruct({p: r['A'] for p, r in runs.items()})
    B = S.reconstruct({p: r['B'] for p, r in runs.items()})
    if None in A or None in B:
        raise RuntimeError('more primes are needed')
    return A, B, S.ratio(B, A), runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--circuit', default='clifft_d3_p001.stim', help='a file, or a name in ../data/inputs')
    ap.add_argument('--noise', help='JSON noise specification (default: uniform Pauli on every edge)')
    ap.add_argument('--degree', type=int, default=6)
    ap.add_argument('--primes', type=int, default=7)
    ap.add_argument('--output')
    a = ap.parse_args()
    path = Path(a.circuit) if Path(a.circuit).exists() else INPUTS / a.circuit
    circuit = C.load(path)
    spec = N.read(a.noise or {})
    locations, discard = N.locations(circuit, spec)
    t0 = time.time()
    A, B, L, runs = exact(circuit, locations, a.degree, a.primes)
    result = S.record(path, spec, 'whole', len(locations), len(discard), a.degree, A, B, L, runs,
                      law_norm=N.law_norm(locations), seconds=round(time.time() - t0))
    print(json.dumps({k: result[k] for k in ('circuit', 'locations', 'degree', 'P_L', 'seconds')}, indent=1))
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
