"""Slow checks that contract networks again (minutes each; about 25 minutes in all on one core).

  python -B -m unittest tests.test_slow

- d=3, whole circuit: the circuit-level noise of the source file against the repository's d=3 series, and a
  Pauli on every edge against results/d3_edges_uniform.json, through x^4 with six primes.
- d=5, three stages, one prime through x^3: the circuit-level noise against the repository's d=5 series; a Pauli on
  every edge against the saved residues; and the same with the data wires that cross a stage cut moved into the
  other stage.
"""
import json
import unittest
from fractions import Fraction

from zxedge import INPUTS, REPO, FOLDER
from zxedge import circuit as C, noise as N, series as S
from zxedge.whole import exact
from zxedge.staged import run as R, endpoints as EP

K3 = 3


def saved(name):
    return json.loads((FOLDER / 'results' / name).read_text())


class WholeD3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit = C.load(INPUTS / 'clifft_d3_p001.stim')

    def test_source_noise_gives_the_repository_series(self):
        locs, _ = N.locations(self.circuit, {'model': 'source'})
        _, _, L, _ = exact(self.circuit, locs, 4, 6)
        want = json.loads((REPO / 'data/series/d3.json').read_text())['conditional_LER_coefficients'][:5]
        self.assertEqual([str(v) for v in L], want)

    def test_edges(self):
        locs, _ = N.locations(self.circuit, {'model': 'edges'})
        A, B, L, _ = exact(self.circuit, locs, 4, 6)
        result = saved('d3_edges_uniform.json')
        self.assertEqual([str(v) for v in A], result['A'][:5])
        self.assertEqual([str(v) for v in B], result['B'][:5])
        self.assertEqual([str(v) for v in L], result['P_L'][:5])


class StagedD5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit = C.load(R.SOURCE)
        cls.prime, cls.root2 = S.primes(1)[0]

    def one_prime(self, locs):
        return R.one_prime(self.circuit, locs, self.prime, self.root2, K3, workers=1, log=lambda s: None)

    def test_source_noise_gives_the_repository_series(self):
        locs, _ = N.locations(self.circuit, {'model': 'source'})
        got = self.one_prime(locs)
        paper = json.loads((REPO / 'data/series/d5.json').read_text())
        shift = len(locs) - paper['retained_physical_locations']        # identity locations it leaves out
        for name in 'AB':
            coeffs = S.times_binomial([Fraction(v) for v in paper['raw_coefficients'][name][:K3 + 1]], shift)
            self.assertEqual(got[name], [S.residue(v, self.prime) for v in coeffs])

    def test_edges_against_saved_residues(self):
        locs, _ = N.locations(self.circuit, {'model': 'edges'})
        got = self.one_prime(locs)
        ref = next(r for r in saved('d5_edges_uniform.json')['residues'] if r['prime'] == self.prime)
        self.assertEqual(got, {name: ref[name][:K3 + 1] for name in 'AB'})

    def test_moving_the_cut_changes_nothing(self):
        locs, _ = N.locations(self.circuit, {'model': 'edges'})
        lines = [g['line'] for g in self.circuit['gates']]
        before = {e['index']: e['before'] for e in N.E.edges(self.circuit)}
        moved = 0
        for loc in locs:
            a, b = lines[loc['after']], lines[before[loc['edge']]]
            if a < EP.FIRST_STOP <= b:
                loc['line'], moved = EP.FIRST_STOP, moved + 1
            elif a < EP.FINAL_START <= b:
                loc['line'], moved = EP.FINAL_START, moved + 1
        self.assertEqual(moved, 26)
        got = self.one_prime(locs)
        ref = next(r for r in saved('d5_edges_uniform.json')['residues'] if r['prime'] == self.prime)
        self.assertEqual(got, {name: ref[name][:K3 + 1] for name in 'AB'})


if __name__ == '__main__':
    unittest.main()
