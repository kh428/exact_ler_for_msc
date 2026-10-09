"""The paper's growth tree (569 leaves, eight Fourier characters) with polynomial tables.

Every table is an array (K+1, 2^d) of residues: the coefficients of x^0 .. x^K of each entry. A join transforms each
coefficient of both children (coordinate permutation and Walsh transform on the shared coordinates), forms the
Cauchy products of coefficient pairs with the repository's restricted-product kernel, and transforms back. The tree,
its join shapes and the leaf maps are the repository's data/plan files; the kernels are its native modular code.

  run(prime, leaves, degree, build_dir)  ->  {'A': [...], 'B': [...]}, the average of the eight characters
"""
import json
import time

import numpy as np

from .. import REPO

PLAN = REPO / 'data' / 'plan'


class Tree:
    def __init__(self, prime, degree, build_dir):
        from calculation.native_modular_binary import BinaryField
        from calculation.native_polynomial_growth import PolynomialGrowthField
        self.prime, self.K = prime, degree
        self.candidate = json.loads((PLAN / 'candidate.json').read_text())
        self.nodes, self.root = self.candidate['plan']['nodes'], self.candidate['plan']['root']
        self.characters = self.candidate['all_slices']
        self.field = BinaryField(prime, build_dir)
        self.native = PolynomialGrowthField(self.field, build_dir)
        with np.load(PLAN / 'maps.npz', allow_pickle=False) as data:
            self.maps = {f: (data[f'd{f}'], data[f't{f}'], self.candidate['leaves'][str(f)]['output_bits'])
                         for f in range(569)}
        self.has_final = {}
        self._final(self.root)

    def _final(self, i):
        node = self.nodes[i]
        if not node['children']:
            self.has_final[i] = node['factor'] == 568
        else:
            a, b = node['children']
            self.has_final[i] = self._final(a) | self._final(b)
        return self.has_final[i]

    def leaf(self, values, f, character):
        """values: (K+1, n) residues in the leaf's own index; returns (K+1, 2^bits) Montgomery residues."""
        destinations, tags, bits = self.maps[f]
        projected = self.native.project(np.ascontiguousarray(values.T), destinations, tags, bits, character)
        out = np.ascontiguousarray(projected.T)
        for k in range(self.K + 1):
            self.field.convert(out[k], encode=True)
        return out

    def join(self, left, right, shape):
        """Exact join of two polynomial tables (both consumed)."""
        g, d = shape['g'], shape['d']
        K = self.K
        for k in range(K + 1):
            self.native.permute(left[k], shape['up'])
            self.native.walsh(left[k], g)
            self.native.permute(right[k], shape['vp'])
            self.native.walsh(right[k], g)
        lz = [bool(left[k].any()) for k in range(K + 1)]
        rz = [bool(right[k].any()) for k in range(K + 1)]
        out = np.zeros((K + 1, 1 << d), dtype=np.uint64)
        for k in range(K + 1):
            first = True
            for i in range(k + 1):
                if lz[i] and rz[k - i]:
                    self.native.products(left[i], right[k - i], out[k], shape, reset=first)
                    first = False
        for k in range(K + 1):
            self.native.walsh(out[k], g, inverse=True)
            self.native.permute(out[k], shape['cp'], inverse=True)
        return out

    def evaluate(self, leaves, character, log=None):
        """leaves: {f: (K+1, n) array} for every f, with leaves[568] = (F_A, F_B). Returns residues of A and B."""
        t0 = time.time()

        def visit(i):
            node = self.nodes[i]
            if not node['children']:
                f = node['factor']
                if f == 568:
                    return tuple(self.leaf(v, f, character) for v in leaves[568])
                return self.leaf(leaves[f], f, character)
            a, b = node['children']
            left, right = visit(a), visit(b)
            if self.has_final[i]:
                if self.has_final[a]:
                    return (self.join(left[0], right.copy(), node['shape']), self.join(left[1], right, node['shape']))
                return (self.join(left.copy(), right[0], node['shape']), self.join(left, right[1], node['shape']))
            return self.join(left, right, node['shape'])

        result = visit(self.root)
        out = {}
        for name, table in zip('AB', result):
            assert table.shape == (self.K + 1, 1)
            col = np.ascontiguousarray(table[:, 0])
            self.field.convert(col, encode=False)
            out[name] = [int(v) for v in col]
        if log:
            log(f'    character {character}: {time.time() - t0:.0f} s')
        return out


def run(prime, leaves, degree, build_dir, characters=None, log=None):
    tree = Tree(prime, degree, build_dir)
    chars = range(tree.characters) if characters is None else characters
    total = {'A': [0] * (degree + 1), 'B': [0] * (degree + 1)}
    for ch in chars:
        r = tree.evaluate(leaves, ch, log)
        for name in 'AB':
            total[name] = [(a + b) % prime for a, b in zip(total[name], r[name])]
    if characters is None:
        inv = pow(tree.characters, -1, prime)
        total = {name: [v * inv % prime for v in vals] for name, vals in total.items()}
    return total
