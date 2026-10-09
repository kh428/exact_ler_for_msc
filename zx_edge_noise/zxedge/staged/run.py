"""The d=5 SOFT circuit in three stages, at one prime or reconstructed exactly over several primes.

  python -B -m zxedge.staged.run --noise specs/edges_uniform.json --degree 7 --primes 9 --workers 4 \
      --output results/d5_edges_uniform.json

For each prime: the first-stage table W (stage-1 network, 8 open coefficients), the final-stage responses F_A and F_B
(final network, 20 open coefficients), the growth leaves (growth channels placed in the leaves of the repository's
tree), and the eight characters of the tree. A-hat and B-hat are series in x = p/(1-p) with every location's
no-fault factor removed. An existing output file is resumed: primes already in it are not recomputed.
"""
import argparse, json, math, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import INPUTS, OUTPUTS
from .. import circuit as C, noise as N, series as S
from ..network import build, to_engine, PolynomialRing
from ..elimination import eliminate
from ..engine import plan, contract
from . import endpoints as EP, growth as G

SOURCE = INPUTS / 'soft_cultivation_d5_p0005.stim'
BUILD = OUTPUTS / 'native'
PLANS = OUTPUTS / 'plans'


class SavedPlan:
    """What engine.contract needs from a cotengra tree, kept on disk as plain data."""
    def __init__(self, data):
        self.inputs = [tuple(t) for t in data['inputs']]
        self.sliced_inds = list(data['sliced'])
        self.path = [tuple(s) for s in data['path']]
        self.width, self.flops = data['width'], data['flops']

    @classmethod
    def from_tree(cls, tree):
        return cls(dict(inputs=[tuple(t) for t in tree.inputs], sliced=list(tree.sliced_inds),
                        path=[tuple(s) for s in tree.get_path()], width=float(tree.contraction_width()),
                        flops=float(tree.total_flops())))

    def get_path(self):
        return self.path

    def contraction_width(self):
        return self.width


def planned(E, name, minutes, target, log=print):
    """A plan for this network structure: from disk if this structure was planned before, else found and saved."""
    import hashlib
    key = repr((tuple(tuple(i) for i, _ in E.tensors), tuple(E.output), target))
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    PLANS.mkdir(parents=True, exist_ok=True)
    path = PLANS / f'{name}_{digest}.json'
    if path.exists():
        data = json.loads(path.read_text())
        if data['key'] == key:
            return SavedPlan(data)
    t0 = time.time()
    tree, unsliced = plan(E, minutes=minutes, target=target)
    saved = SavedPlan.from_tree(tree)
    path.write_text(json.dumps(dict(key=key, inputs=saved.inputs, sliced=saved.sliced_inds, path=saved.path,
                                    width=saved.width, flops=saved.flops)))
    log(f'  planned {name}: width {saved.width:.0f} (unsliced {unsliced[0]:.0f}), 10^{math.log10(saved.flops):.2f} '
        f'flops, {2 ** len(saved.sliced_inds)} slices, {time.time() - t0:.0f} s')
    return saved


def bit_reverse_index(n):
    """flat engine index (output 0 most significant) of each coefficient vector c (bit j = output j)."""
    c = np.arange(1 << n, dtype=np.int64)
    out = np.zeros_like(c)
    for j in range(n):
        out |= ((c >> j) & 1) << (n - 1 - j)
    return out


def stage_values(circuit, locs, ring, root2, *, gates, detectors, observable, start=None, finish=None, name,
                 minutes, target, log):
    net, opens, _ = build(circuit, locs, ring, root2, gates=gates, detectors=detectors, observable=observable,
                          start=start, finish=finish)
    net, _ = eliminate(net, protected=opens)
    E = to_engine(net, ring, opens)
    E.simplify(log=False)
    tree = planned(E, name, minutes, target, log)
    t0 = time.time()
    total = contract(E, tree, log_every=0)
    log(f'  contracted {name}: {time.time() - t0:.0f} s')
    n = len(opens)
    flat = total.reshape(1 << n, ring.degree + 1)
    return np.ascontiguousarray(flat[bit_reverse_index(n)].T.astype(np.int64))       # (K+1, 2^n), index c


def label_signs(data, faces):
    """(-1)^{|A(c) and B(c)|} for every coefficient vector c, vectorised."""
    m = len(faces)
    xs, zs = EP.code_labels(data, faces, 2 * m + 2)
    nbits = 2 * m + 2
    xv = [sum(1 << i for i, q in enumerate(data) if j in xs[q]) for j in range(nbits)]
    zv = [sum(1 << i for i, q in enumerate(data) if j in zs[q]) for j in range(nbits)]
    A = np.zeros(1 << nbits, dtype=np.int64)
    B = np.zeros(1 << nbits, dtype=np.int64)
    for j in range(nbits):
        A[1 << j:2 << j] = A[:1 << j] ^ xv[j]
        B[1 << j:2 << j] = B[:1 << j] ^ zv[j]
    parity = np.bitwise_count((A & B).astype(np.uint64)) % 2
    return np.where(parity == 1, -1, 1)


def walsh_np(values, prime, inverse=False):
    """Walsh-Hadamard transform along the last axis of an int64 array of residues below 2^21."""
    v = values.copy() % prime
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


def first_table(circuit, locs, prime, root2, K, minutes, log):
    """W in the paper's leaf index (K+1, 256), the logical coefficients tau(Q) and the Z_L check."""
    ring = PolynomialRing(prime, K)
    gates = EP.gate_range(circuit, 0, EP.FIRST_STOP)
    sl = EP.stage_locations(locs, 0, EP.FIRST_STOP)
    dets = EP.stage_detectors(circuit, 0, EP.FIRST_STOP, drop_outside=False)
    V = stage_values(circuit, sl, ring, root2, gates=gates, detectors=dets, observable=None, finish=EP.w_finish,
                     name=f'W_{tag(locs)}', minutes=minutes, target=26, log=log)
    data, checks = EP.faces3()
    sign = label_signs(data, checks)
    Q = np.arange(256) >> 6
    tau = {}
    for qv in range(4):
        vals = {int(V[0, c] * sign[c]) % prime for c in range(256) if Q[c] == qv}
        assert len(vals) == 1, ('the ideal first-stage state is not a code state', qv, vals)
        tau[qv] = vals.pop()
    assert tau[0] == 1 and tau[2] == 0
    zl = np.count_nonzero(V[:, Q == 2] % prime)
    if zl:
        raise ValueError(f'the first-stage output has a Z_L part ({zl} nonzero entries): no four-state table')
    inv = {qv: pow(tau[qv], -1, prime) for qv in (0, 1, 3)}
    hat = np.zeros_like(V)
    for c in range(256):
        if Q[c] != 2:
            hat[:, c] = V[:, c] * (sign[c] % prime) % prime * inv[Q[c]] % prime
    return walsh_np(hat, prime, inverse=True), tau


def final_tables(circuit, locs, prime, root2, K, tau, minutes, target, log):
    ring = PolynomialRing(prime, K)
    gates = EP.gate_range(circuit, EP.FINAL_START, 10 ** 6)
    sl = EP.stage_locations(locs, EP.FINAL_START, 10 ** 6)
    dets = EP.stage_detectors(circuit, EP.FINAL_START, 10 ** 6, drop_outside=True)
    data, faces = EP.faces5()
    sign = label_signs(data, faces) % prime
    Q = np.arange(1 << 20) >> 18
    r5 = sign * np.array([tau[int(qv)] for qv in range(4)])[Q] % prime
    out = []
    for obs in 'AB':
        R = stage_values(circuit, sl, ring, root2, gates=gates, detectors=dets, observable=obs, start=EP.f_start,
                         name=f'F{obs}_{tag(locs)}_t{target}', minutes=minutes, target=target, log=log)
        out.append(walsh_np(R * r5 % prime, prime))
    return tuple(out)


def tag(locs):
    kinds = sorted({loc['kind'] for loc in locs})
    return '-'.join(kinds) + f'-{len(locs)}'


def growth_leaves(circuit, locs, prime, K):
    channels, model = G.compile_growth(circuit, locs)
    groups = G.leaf_groups()
    placed, homeless, identity = G.assign(channels, groups)
    if homeless:
        raise ValueError(f'{len(homeless)} growth channels fit no leaf of the paper tree')
    leaves = {}
    for f in range(1, len(groups) - 1):
        basis = groups[f]['basis']
        if f in placed:
            leaves[f] = G.leaf_table(basis, placed[f], K, prime).astype(np.int64)
        else:
            t = np.zeros((K + 1, 1 << len(basis)), dtype=np.int64)
            t[0, 0] = 1
            leaves[f] = t
    return leaves, len(identity), len(channels)


def _character(args):
    prime, leaves, K, ch = args
    from .tree import Tree
    tree = Tree(prime, K, BUILD)
    clean = {f: (tuple(np.ascontiguousarray(v.astype(np.uint64)) for v in t) if f == 568
                 else np.ascontiguousarray(t.astype(np.uint64))) for f, t in leaves.items()}
    return tree.evaluate(clean, ch)


def one_prime(circuit, locs, prime, root2, K, *, workers=1, minutes=2, target=24, log=print):
    t0 = time.time()
    W, tau = first_table(circuit, locs, prime, root2, K, minutes, log)
    FA, FB = final_tables(circuit, locs, prime, root2, K, tau, minutes, target, log)
    leaves, identity, nchan = growth_leaves(circuit, locs, prime, K)
    leaves[0] = W
    leaves[568] = (FA, FB)
    log(f'  endpoints and {nchan} growth channels ({identity} identity) ready: {time.time() - t0:.0f} s')
    jobs = [(prime, leaves, K, ch) for ch in range(8)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_character, jobs))
    else:
        rows = [_character(j) for j in jobs]
    inv8 = pow(8, -1, prime)
    total = {}
    for name in 'AB':
        vals = [sum(r[name][k] for r in rows) * inv8 % prime for k in range(K + 1)]
        # growth channels with no effect contribute (1 + x) each
        for _ in range(identity):
            vals = [(vals[k] + (vals[k - 1] if k else 0)) % prime for k in range(K + 1)]
        total[name] = vals
    log(f'  prime {prime}: {time.time() - t0:.0f} s')
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--noise', help='JSON noise specification (default: uniform Pauli on every edge)')
    ap.add_argument('--degree', type=int, default=3)
    ap.add_argument('--primes', type=int, default=1)
    ap.add_argument('--workers', type=int, default=1, help='processes for the eight tree characters')
    ap.add_argument('--minutes', type=float, default=2, help='planning time per stage network')
    ap.add_argument('--target', type=int, default=24, help='largest intermediate of the final networks, log2')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    circuit = C.load(SOURCE)
    spec = N.read(a.noise or {})
    locs, discard = N.locations(circuit, spec)
    out_path = Path(a.output)
    runs = {}
    if out_path.exists():
        old = json.loads(out_path.read_text())
        assert old['noise'] == spec and old['degree'] == a.degree, 'the output file is for another calculation'
        runs = {r['prime']: dict(sqrt2=r['sqrt2'], A=r['A'], B=r['B']) for r in old['residues']}
    t0 = time.time()

    def save():
        A = S.reconstruct({p: r['A'] for p, r in runs.items()}) if len(runs) > 1 else [None]
        B = S.reconstruct({p: r['B'] for p, r in runs.items()}) if len(runs) > 1 else [None]
        done = None not in A and None not in B
        L = S.ratio(B, A) if done else []
        result = S.record(SOURCE, spec, 'staged', len(locs), len(discard), a.degree,
                          A if done else [], B if done else [], L, runs, law_norm=N.law_norm(locs),
                          complete=done, seconds=round(time.time() - t0))
        out_path.write_text(json.dumps(result, indent=1) + '\n')
        return result

    for prime, root2 in S.primes(a.primes):
        if prime in runs:
            continue
        print(f'prime {prime}', flush=True)
        runs[prime] = dict(sqrt2=root2, **one_prime(circuit, locs, prime, root2, a.degree, workers=a.workers,
                                                    minutes=a.minutes, target=a.target))
        result = save()
    result = save()
    print(json.dumps({k: result[k] for k in ('complete', 'locations', 'degree', 'P_L')}, indent=1))


if __name__ == '__main__':
    main()
