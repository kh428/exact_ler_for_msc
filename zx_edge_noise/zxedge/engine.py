"""Exact contraction of series-valued networks with a cotengra plan.

A tensor is a float64 array of shape (2,)*n + (K+1,): the coefficients of x^0 .. x^K of a series truncated at x^K,
each an integer in [0, q) for a prime q = 1 (mod 8) below 2^21 (so i and sqrt2 are field elements). A pairwise
contraction is, for each pair of orders (o1, o2) with o1 + o2 <= K, a batched matrix product in float64 over chunks of
at most 2048 summed terms: products are below 2^42 and their sums below 2^53, so every value is an exact integer, reduced
mod q after each chunk.

Before planning, exact structural simplifications (no rounding is involved anywhere):
  fixed   an index whose value 1 (or 0) makes some tensor vanish identically is fixed to the other value;
  diag    a tensor that vanishes off the diagonal of two of its indices merges them (a hyperedge);
  absorb  a tensor whose indices are all in another tensor is multiplied into it (series product entry by entry);
  dangle  an index in one tensor only, not open, is summed there; one in no tensor gives a factor 2.

  from zxedge.engine import Net, plan, contract
"""
import itertools, math, time
import numpy as np

CHUNK = 2048


def series_mul(a, b, K, q):
    """Entry-wise product of series arrays a[..., K+1], b[..., K+1] (broadcast), truncated at x^K, mod q."""
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape))
    for o1 in range(K + 1):
        if not a[..., o1].any():
            continue
        for o2 in range(K + 1 - o1):
            out[..., o1 + o2] += np.fmod(a[..., o1] * b[..., o2], q)
    return np.fmod(out, q)


def degree(arr):
    nz = [o for o in range(arr.shape[-1]) if arr[..., o].any()]
    return max(nz) if nz else 0


class Net:
    """tensors: list of [inds tuple, array]; output: open indices; scalar: series (K+1,); K, q."""
    def __init__(self, tensors, output, scalar, K, q, free=0):
        self.tensors = [[tuple(i), np.asarray(a, dtype=np.float64)] for i, a in tensors]
        self.output = tuple(output)
        self.K, self.q = K, q
        self.scalar = np.zeros(K + 1)
        self.scalar[:len(scalar)] = np.asarray(scalar[:K + 1], dtype=np.float64) % q
        self.fixed = {}
        self.factor2 = free

    # -- structural simplification -------------------------------------------------------------------------------
    def _where(self):
        w = {}
        for k, (inds, _) in enumerate(self.tensors):
            for v in inds:
                w.setdefault(v, []).append(k)
        return w

    def _fix(self, v, val):
        self.fixed[v] = val
        for t in self.tensors:
            if v in t[0]:
                ax = t[0].index(v)
                t[1] = np.take(t[1], val, axis=ax)
                t[0] = t[0][:ax] + t[0][ax + 1:]

    def _merge(self, a, b):
        """Indices a and b take equal values: b becomes a everywhere."""
        for t in self.tensors:
            if b not in t[0]:
                continue
            if a in t[0]:
                ia, ib = t[0].index(a), t[0].index(b)
                d = np.diagonal(t[1], axis1=ia, axis2=ib)          # diagonal axis goes last, before nothing: move it
                d = np.moveaxis(d, -1, 0)                            # (diag, rest..., K+1)
                rest = [v for v in t[0] if v not in (a, b)]
                t[0] = (a,) + tuple(rest)
                t[1] = d
            else:
                t[0] = tuple(a if v == b else v for v in t[0])

    def _merge_flip(self, a, b):
        """b = NOT a: flip b everywhere, then merge it into a."""
        for t in self.tensors:
            if b in t[0]:
                t[1] = np.flip(t[1], axis=t[0].index(b))
        self._merge(a, b)

    def _pair(self, ka, kb, keep):
        """Contract tensors ka and kb exactly, summing their shared indices not in keep."""
        from .engine import program, run, degree
        (ia, A), (ib, B) = self.tensors[ka], self.tensors[kb]
        steps, final = program([ia, ib], tuple(sorted(keep)), [(0, 1)], set())
        res, _ = run([A, B], [degree(A), degree(B)], steps, self.K, self.q)
        return final, res

    def simplify(self, log=True, diag_max=14, split_max=12, splits=True):
        K, q = self.K, self.q
        out = set(self.output)
        t0 = time.time()
        passes = 0
        while True:
            passes += 1
            changed = False
            keep = []
            for inds, arr in self.tensors:                       # scalars
                if not inds:
                    self.scalar = series_mul(self.scalar, arr, K, q)
                    changed = True
                else:
                    keep.append([inds, arr])
            self.tensors = keep
            w = self._where()
            for v, ks in w.items():                               # dangling indices
                if len(ks) == 1 and v not in out:
                    t = self.tensors[ks[0]]
                    ax = t[0].index(v)
                    t[1] = np.fmod(t[1].sum(axis=ax), q)
                    t[0] = t[0][:ax] + t[0][ax + 1:]
                    changed = True
            if changed:
                continue
            fixes = {}                                            # fixed columns
            for inds, arr in self.tensors:
                for ax, v in enumerate(inds):
                    if v in out or v in fixes:
                        continue
                    z0 = not np.take(arr, 0, axis=ax).any()
                    z1 = not np.take(arr, 1, axis=ax).any()
                    if z0 and z1:
                        raise ZeroDivisionError('a tensor vanishes: the network is zero')
                    if z0 or z1:
                        fixes[v] = 1 if z0 else 0
            for v, val in fixes.items():
                self._fix(v, val)
            if fixes:
                continue
            parent = {}                                           # diagonal pairs, merged with a union-find
            def find(v):
                while parent.get(v, v) != v:
                    v = parent[v]
                return v
            merged = 0
            flipped = set()                                   # one flip per index per pass (stale views)
            for inds, arr in [(t[0], t[1]) for t in self.tensors]:
                if len(inds) < 2 or len(inds) > diag_max:
                    continue
                for i, j in itertools.combinations(range(len(inds)), 2):
                    off = np.take(np.take(arr, 0, axis=i), 1, axis=j - 1).any() or \
                          np.take(np.take(arr, 1, axis=i), 0, axis=j - 1).any()
                    on = np.take(np.take(arr, 0, axis=i), 0, axis=j - 1).any() or \
                         np.take(np.take(arr, 1, axis=i), 1, axis=j - 1).any()
                    if off and on:
                        continue
                    a, b = find(inds[i]), find(inds[j])
                    if a == b or (a in out and b in out) or (a in flipped or b in flipped):
                        continue
                    if b in out:
                        a, b = b, a
                    if off:                                   # antidiagonal: b = NOT a
                        self._merge_flip(a, b)
                        flipped.update((a, b))
                    else:
                        self._merge(a, b)
                    parent[b] = a
                    merged += 1
            if merged:
                continue
            w = {v: set(ks) for v, ks in self._where().items()}   # absorb contained tensors
            dead = set()
            for k in sorted(range(len(self.tensors)), key=lambda k: len(self.tensors[k][0])):
                inds, arr = self.tensors[k]
                cands = set.intersection(*(w[v] for v in inds)) - {k} - dead
                if not cands:
                    continue
                host = max(cands, key=lambda c: len(self.tensors[c][0]))
                hi, ha = self.tensors[host]
                perm = [inds.index(v) for v in hi if v in inds]
                shape = [2 if v in inds else 1 for v in hi] + [K + 1]
                sub = np.transpose(arr, perm + [len(inds)]).reshape(shape)
                self.tensors[host][1] = series_mul(ha, sub, K, q)
                dead.add(k)
                for v in inds:
                    w[v].discard(k)
            if dead:
                self.tensors = [t for k, t in enumerate(self.tensors) if k not in dead]
                continue
            w = self._where()                                     # rank simplification: pairs that do not grow
            done = False
            for v, ks in w.items():
                if len(ks) != 2 or v in out:
                    continue
                ka, kb = ks
                ia, ib = self.tensors[ka][0], self.tensors[kb][0]
                keep = {u for u in set(ia) | set(ib) if u in out or len(w[u]) > 2 or not (u in ia and u in ib)}
                res = [u for u in dict.fromkeys(ia + ib) if u in keep]
                if len(res) <= max(len(ia), len(ib)):
                    final, arr = self._pair(ka, kb, keep)
                    self.tensors = [t for k, t in enumerate(self.tensors) if k not in (ka, kb)] + [[final, arr]]
                    done = True
                    break
            if done:
                continue
            if splits:                                            # exact low-rank splits
                from .splitter import best_split, apply_split
                if not hasattr(self, '_unsplittable'):
                    self._unsplittable = set()
                    self._bonds = 0
                new, did = [], False
                for inds, arr in self.tensors:
                    key = (inds, arr.tobytes().__hash__())
                    if did or key in self._unsplittable:
                        new.append([inds, arr])
                        continue
                    best = best_split(inds, arr, K, q, max_n=split_max)
                    if best is None:
                        self._unsplittable.add(key)
                        new.append([inds, arr])
                        continue
                    names = [f'bond{self._bonds + k}' for k in range(best[5])]
                    self._bonds += best[5]
                    for t in apply_split(inds, arr, K, q, *best[1:], names):
                        new.append([t[0], t[1]])
                    did = True
                self.tensors = new
                if did:
                    continue
            break
        if log:
            sizes = sorted(len(i) for i, _ in self.tensors)
            print(f'simplified in {passes} passes, {time.time() - t0:.0f} s: {len(self.tensors)} tensors, '
                  f'{len(self._where())} indices, largest {sizes[-3:]}, fixed {len(self.fixed)}', flush=True)

    def structure(self):
        inputs = [tuple(i) for i, _ in self.tensors]
        size = {v: 2 for i in inputs for v in i}
        return inputs, self.output, size


def plan(net, minutes=5, target=24, seed=0, recipe='combo'):
    import cotengra as ctg
    import warnings
    warnings.filterwarnings('ignore')
    inputs, output, size = net.structure()
    opt = ctg.HyperOptimizer(methods=['kahypar', 'greedy'], minimize='combo', reconf_opts={}, max_time=60 * minutes,
                             max_repeats=10 ** 7, parallel=False, progbar=False, on_trial_error='ignore')
    tree = opt.search(inputs, output, size)
    unsliced = (tree.contraction_width(), math.log10(tree.total_flops()))
    if tree.contraction_width() > target:
        tree = tree.slice_and_reconfigure_forest(target_size=2 ** target, minimize='flops', parallel=False)
    return tree, unsliced


def program(inputs, output, path, sliced):
    """The steps of a contraction path with the sliced indices removed: per step (i, j, result inds, spec)."""
    tensors = [tuple(v for v in t if v not in sliced) for t in inputs]
    out = set(output) - set(sliced)
    ref = {}
    for t in tensors:
        for v in t:
            ref[v] = ref.get(v, 0) + 1
    steps = []
    for i, j in path:
        if i > j:
            i, j = j, i
        ib, ia = tensors.pop(j), tensors.pop(i)
        sa, sb = set(ia), set(ib)
        for v in sa | sb:
            ref[v] -= (v in sa) + (v in sb)
        keep = {v for v in sa | sb if v in out or ref[v] > 0}
        adrop = [v for v in ia if v not in sb and v not in keep]
        bdrop = [v for v in ib if v not in sa and v not in keep]
        ia2 = tuple(v for v in ia if v not in adrop)
        ib2 = tuple(v for v in ib if v not in bdrop)
        sa2, sb2 = set(ia2), set(ib2)
        batch = [v for v in ia2 if v in sb2 and v in keep]
        summed = [v for v in ia2 if v in sb2 and v not in keep]
        akeep = [v for v in ia2 if v not in sb2]
        bkeep = [v for v in ib2 if v not in sa2]
        res = tuple(batch + akeep + bkeep)
        for v in res:
            ref[v] = ref.get(v, 0) + 1
        steps.append(dict(i=i, j=j, ia=ia, ib=ib, adrop=[ia.index(v) for v in adrop], bdrop=[ib.index(v) for v in bdrop],
                          pa=[ia2.index(v) for v in batch + akeep + summed], pb=[ib2.index(v) for v in batch + summed + bkeep],
                          nb=len(batch), na=len(akeep), ns=len(summed), nn=len(bkeep), res=res))
        tensors.append(res)
    return steps, tensors[0] if tensors else ()


def modmatmul(A, B, q):
    s = A.shape[-1]
    if s <= CHUNK:
        return np.fmod(np.matmul(A, B), q)
    out = np.zeros(A.shape[:-1] + B.shape[-1:])
    for a in range(0, s, CHUNK):
        out += np.fmod(np.matmul(A[..., a:a + CHUNK], B[..., a:a + CHUNK, :]), q)
        out = np.fmod(out, q)
    return out


def run(arrays, degs, steps, K, q):
    arrays, degs = list(arrays), list(degs)
    for st in steps:
        i, j = st['i'], st['j']
        B, db = arrays.pop(j), degs.pop(j)
        A, da = arrays.pop(i), degs.pop(i)
        if st['adrop']:
            A = np.fmod(A.sum(axis=tuple(st['adrop'])), q)
        if st['bdrop']:
            B = np.fmod(B.sum(axis=tuple(st['bdrop'])), q)
        nb, na, ns, nn = (1 << st['nb']), (1 << st['na']), (1 << st['ns']), (1 << st['nn'])
        A2 = np.transpose(A, st['pa'] + [A.ndim - 1]).reshape(nb, na, ns, K + 1)
        B2 = np.transpose(B, st['pb'] + [B.ndim - 1]).reshape(nb, ns, nn, K + 1)
        dc = min(K, da + db)
        C = np.zeros((nb, na, nn, K + 1))
        Bs = [np.ascontiguousarray(B2[..., o]) for o in range(db + 1)]
        for o1 in range(da + 1):
            Ao = np.ascontiguousarray(A2[..., o1])
            if not Ao.any():
                continue
            for o2 in range(min(db, K - o1) + 1):
                C[..., o1 + o2] += modmatmul(Ao, Bs[o2], q)
        C = np.fmod(C, q)
        arrays.append(C.reshape((2,) * len(st['res']) + (K + 1,)))
        degs.append(dc)
    return arrays[0], degs[0]


def contract(net, tree, log_every=64, slices=None):
    """The value of the network: an array (2,)*len(output) + (K+1,) mod q, in the order of net.output."""
    K, q = net.K, net.q
    inputs = [tuple(i) for i, _ in net.tensors]
    assert [tuple(t) for t in tree.inputs] == inputs, 'the plan is for another structure'
    sliced = list(tree.sliced_inds)
    path = tree.get_path()
    steps, final = program(inputs, net.output, path, set(sliced))
    out_unsliced = [v for v in net.output if v not in sliced]
    out_sliced = [v for v in net.output if v in sliced]
    total = np.zeros((2,) * len(net.output) + (K + 1,))
    t0 = time.time()
    n = 1 << len(sliced)
    lo, hi = slices if slices else (0, n)
    for s in range(lo, min(hi, n)):
        val = {v: (s >> k) & 1 for k, v in enumerate(sliced)}
        arrs, degs = [], []
        for inds, arr in net.tensors:
            idx = tuple(val[v] if v in val else slice(None) for v in inds) + (slice(None),)
            a = arr[idx] if any(v in val for v in inds) else arr
            arrs.append(a)
            degs.append(degree(a))
        res, _ = run(arrs, degs, steps, K, q) if steps else (arrs[0], degs[0])
        perm = [final.index(v) for v in out_unsliced]
        res = np.transpose(res, perm + [len(final)])
        sel = tuple(val[v] if v in val else slice(None) for v in net.output) + (slice(None),)
        if out_sliced:
            total[sel] = np.fmod(total[sel] + res, q)
        else:
            total = np.fmod(total + res, q)
        if log_every and (s + 1) % log_every == 0:
            print(f'  slice {s + 1}/{n}, {time.time() - t0:.0f} s', flush=True)
    if slices:                                   # a partial sum: the scalar factors are applied when the parts are added
        return total
    total = series_mul(total, net.scalar.reshape((1,) * len(net.output) + (K + 1,)), K, q)
    total = np.fmod(total * pow(2, net.factor2, q), q)
    return total


def finish(total, net):
    """Multiply a sum of partial contractions by the network's scalar and its factors 2."""
    K, q = net.K, net.q
    total = series_mul(total, net.scalar.reshape((1,) * len(net.output) + (K + 1,)), K, q)
    return np.fmod(total * pow(2, net.factor2, q), q)
