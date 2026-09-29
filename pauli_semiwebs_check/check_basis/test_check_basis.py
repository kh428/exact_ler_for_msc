"""Toy circuits with known answers for the whole-circuit web checks.

Run from pauli_semiwebs_check:  python -B -m unittest check_basis.test_check_basis -v
"""

import unittest
from itertools import product

from src.generate import local_scalar
from src.semiweb import BITS
from check_basis import circuit, graph, oracle, webs


def prepared(text):
    parsed = circuit.parse_text(text)
    parsed['text'] = text
    return parsed


def web_space(text):
    parsed = prepared(text)
    discard = oracle.discard_states(parsed)
    result = webs.web_checks(graph.build(parsed, discard))
    return parsed, result['equations']


def oracle_space(parsed):
    return oracle.deterministic_space(parsed, random_count=40, shots=128)


class RecordSpiders(unittest.TestCase):
    def test_record_phase_flips_scalar_exactly_with_opposite_support(self):
        for kind, degree in product('XZ', (1, 2, 3, 4)):
            for labels in product('IXYZ', repeat=degree):
                opposite = [BITS[p][0 if kind == 'Z' else 1] for p in labels]
                if len(set(opposite)) != 1:
                    continue
                shift = (local_scalar(kind, 4, labels) - local_scalar(kind, 0, labels)) % 8
                self.assertEqual(shift, 4 * opposite[0])


class CliffordCircuits(unittest.TestCase):
    """For stabilizer circuits closed webs must give exactly Stim's deterministic parities."""

    def assert_equal_spaces(self, text, expected_rank=None):
        parsed, eqs = web_space(text)
        space = oracle_space(parsed)
        nrec = len(parsed['records'])
        self.assertEqual(webs.affine_rank(eqs, nrec), space['dimension'])
        self.assertEqual(webs.affine_rank(eqs + space['equations'], nrec), space['dimension'])
        if expected_rank is not None:
            self.assertEqual(space['dimension'], expected_rank)

    def test_repetition_code_with_products(self):
        self.assert_equal_spaces('\n'.join([
            'R 0 1 2 3 4', 'CX 0 3 1 3', 'CX 1 4 2 4', 'M 3 4',
            'R 3 4', 'CX 0 3 1 3', 'CX 1 4 2 4', 'M 3 4', 'MPP Z0*Z1 Z1*Z2']), expected_rank=6)

    def test_ghz_x_products_feedforward_and_remeasurement(self):
        self.assert_equal_spaces('\n'.join([
            'RX 0', 'R 1 2', 'CX 0 1 0 2', 'MPP X0*X1*X2 Z0*Z1',
            'M 0', 'CX rec[-1] 1', 'M 1', 'M 2', 'M 2']), expected_rank=5)

    def test_y_products_and_known_state_discard(self):
        self.assert_equal_spaces('\n'.join([
            'RX 5', 'R 6', 'CX 5 6', 'CX 5 6', 'RX 5',     # q5 is |+> again when reset
            'RX 0 1', 'CX 0 2 1 2', 'S 0 1', 'MPP Y0*Y1', 'MX 0', 'MX 1', 'M 2', 'MX 5', 'M 6']))


class NonCliffordToy(unittest.TestCase):
    def test_phase_independent_found_and_phase_dependent_excluded(self):
        # rec0 is 0 for any phase; rec1 is 0 only because T_DAG undoes T.
        parsed, eqs = web_space('\n'.join(['R 0', 'T 0', 'M 0', 'RX 1', 'T 1', 'T_DAG 1', 'MX 1']))
        nrec = len(parsed['records'])
        self.assertEqual(webs.affine_rank(eqs, nrec), 1)
        self.assertTrue(oracle.contains(eqs, nrec, 0b01, 0))
        self.assertFalse(oracle.contains(eqs, nrec, 0b10, 0))
        space = oracle_space(parsed)
        self.assertEqual(space['dimension'], 1)
        self.assertTrue(oracle.contains(space['equations'], nrec, 0b01, 0))


if __name__ == '__main__':
    unittest.main()
