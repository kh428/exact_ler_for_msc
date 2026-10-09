"""Fast checks (a few seconds each): edges, growth labels, leaf tables, the stage cut and the saved results.

  python -B -m unittest tests.test_fast
"""
import hashlib
import json
import unittest
from fractions import Fraction

from zxedge import INPUTS, REPO, FOLDER
from zxedge import circuit as C, edges as E, noise as N, series as S
from zxedge.staged import endpoints as EP, growth as G

D3 = INPUTS / 'clifft_d3_p001.stim'
D5 = INPUTS / 'soft_cultivation_d5_p0005.stim'


def direct_table(items, K):
    """{label: [c_0..c_K]} by multiplying out each channel's law 1 + x (sum_f w_f [l_f] + w_0 [0])."""
    zero = [Fraction(0)] * (K + 1)
    table = {0: [Fraction(1)] + zero[1:]}
    for item in items:
        law = {0: 1 - sum(item['weights'].values())}
        for code, w in item['weights'].items():
            law[item['labels'][code]] = law.get(item['labels'][code], 0) + w
        new = {}
        for v, coeffs in table.items():
            t = new.setdefault(v, list(zero))
            for k in range(K + 1):
                t[k] += coeffs[k]
            for u, w in law.items():
                t = new.setdefault(v ^ u, list(zero))
                for k in range(1, K + 1):
                    t[k] += coeffs[k - 1] * w
        table = new
    return table


class Edges(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(E.edges(C.load(D3))), 285)
        self.assertEqual(len(E.edges(C.load(D5))), 1602)

    def test_source_locations(self):
        self.assertEqual(len(E.source_locations(C.load(D3))), 518)
        self.assertEqual(len(E.source_locations(C.load(D5))), 3564)

    def test_outcomes(self):
        cnot = next(e for e in E.edges(C.load(D3)) if e['kind'] == 'cnot')
        c, t = cnot['qubits']
        self.assertEqual(E.outcome(cnot, 1), {t: 'X'})
        self.assertEqual(E.outcome(cnot, 2), {c: 'Z'})
        self.assertEqual(E.outcome(cnot, 3), {t: 'X', c: 'Z'})

    def test_noise_rules(self):
        circuit = C.load(D3)
        locs, _ = N.locations(circuit, {'model': 'edges', 'rules': [{}, {'select': {'kinds': ['cnot']},
                                                                          'noiseless': True}]})
        self.assertEqual(len(locs), 204)
        locs, discard = N.locations(circuit, json.loads((FOLDER / 'specs/edges_erasure_discard.json').read_text()))
        self.assertEqual((len(locs), len(discard)), (285, 285))
        locs, _ = N.locations(circuit, json.loads((FOLDER / 'specs/edges_erasure_ignore.json').read_text()))
        self.assertEqual(len(locs), 570)


class Growth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit = C.load(D5)

    def test_source_labels_match_the_repository_model(self):
        locs, _ = N.locations(self.circuit, {'model': 'source'})
        channels, model = G.compile_growth(self.circuit, locs)
        self.assertEqual((model['constraint_bits'], model['output_bits'], model['random_measurements']), (49, 26, 10))
        saved = json.loads((REPO / 'data/models/growth_frame_model.json').read_text())['model']
        want = {ch['source_location']: ch['generator_labels'] for ch in saved['channels'] + saved['ignored_channels']}
        got = {ch['location']: ch['generator_labels'] for ch in channels}
        self.assertEqual(got, want)

    def test_edge_channels_fit_the_tree(self):
        locs, _ = N.locations(self.circuit, {'model': 'edges'})
        channels, _ = G.compile_growth(self.circuit, locs)
        placed, homeless, identity = G.assign(channels, G.leaf_groups())
        self.assertEqual((len(channels), len(homeless), len(identity)), (1037, 0, 0))

    def test_leaf_tables(self):
        locs, _ = N.locations(self.circuit, {'model': 'edges'})
        channels, _ = G.compile_growth(self.circuit, locs)
        groups = G.leaf_groups()
        placed, _, _ = G.assign(channels, groups)
        prime, K = S.primes(1)[0][0], 4
        for f in sorted(placed, key=lambda f: -len(placed[f]))[:8]:
            basis = groups[f]['basis']
            table = G.leaf_table(basis, placed[f], K, prime)
            want = direct_table(placed[f], K)
            for label, coeffs in want.items():
                i = G.coordinates(basis, label)
                self.assertEqual([int(v) for v in table[:, i]], [S.residue(x, prime) for x in coeffs])
            self.assertEqual(int((table != 0).any(axis=0).sum()),
                             sum(1 for c in want.values() if any(c)))

    def test_stage_cut_is_exact(self):
        locs, _ = N.locations(self.circuit, {'model': 'source'})
        _, model = G.compile_growth(self.circuit, locs)
        audit = EP.audit_cut(self.circuit, model['audit'], G.FREE_RECORDS)
        self.assertEqual(audit, dict(declared_rank=107, stage_rank=107, stages_implied_by_detectors=True,
                                     detectors_implied_by_stages=True, free_records_in_detectors=[]))


class IgnoredErasure(unittest.TestCase):
    def test_saved_erasure_series_are_the_shifted_pauli_series(self):
        pauli = {}
        for path in sorted((FOLDER / 'results').glob('d*_edges_uniform.json')):
            pauli[path.name[:2]] = [Fraction(v) for v in json.loads(path.read_text())['P_L']]
        checked = 0
        for path in sorted((FOLDER / 'results').glob('d*_edges_erasure_ignore*.json')):
            result = json.loads(path.read_text())
            rate = Fraction(result['noise']['rules'][0]['erasure']['rate'])
            got = [Fraction(v) for v in result['P_L']]
            with self.subTest(path.name):
                self.assertEqual(S.ignored_erasure(pauli[path.name[:2]], rate, len(got) - 1), got)
            checked += 1
        self.assertGreaterEqual(checked, 4)


class SavedResults(unittest.TestCase):
    def test_every_result_file(self):
        for path in sorted((FOLDER / 'results').glob('*.json')):
            result = json.loads(path.read_text())
            if 'residues' not in result:
                continue
            with self.subTest(path.name):
                digest = hashlib.sha256((INPUTS / result['circuit']).read_bytes()).hexdigest()
                self.assertEqual(digest, result['circuit_sha256'])
                S.check_record(result)


if __name__ == '__main__':
    unittest.main()
