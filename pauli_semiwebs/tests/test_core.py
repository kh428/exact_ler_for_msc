"""Compare binary constraints and scalar formulae with explicit exact tensors."""
from itertools import product
import unittest
from src.generate import local_scalar
from src.linear import kernel, rank
from src.semiweb import BITS, constraints
from verification.check_diagrams import local

class SemiwebTests(unittest.TestCase):
    def test_local_phase_and_scalar_against_tensors(self):
        for kind, degree, phase in product(('X', 'Z'), (1, 2, 3), range(8)):
            for labels in product('IXYZ', repeat=degree):
                opposite = [BITS[p][0 if kind == 'Z' else 1] for p in labels]
                if len(set(opposite)) != 1:
                    continue
                _, scalar = local(kind, phase, labels)
                self.assertEqual(local_scalar(kind, phase, labels), scalar)

    def test_kernel_against_exhaustive_solutions(self):
        rows = [0b10101, 0b01110, 0b11011]
        basis = kernel(rows, 5)
        span = {0}
        for v in basis:
            span |= {x ^ v for x in span}
        exact = {x for x in range(32) if all((x&r).bit_count()%2 == 0 for r in rows)}
        self.assertEqual(span, exact)
        self.assertEqual(len(basis), 5-rank(rows))

    def test_shared_nonclifford_defect_cancels(self):
        first, s1 = local('Z', 1, ('X', 'X'))
        second, s2 = local('Z', (1+first)%8, ('X', 'X'))
        self.assertEqual((first+second)%8, 0)
        self.assertEqual((s1+s2)%8, 0)
        self.assertNotEqual((first+first)%8, 0)

    def test_y_scalar_is_not_discarded(self):
        self.assertEqual(local('Z', 1, ('Y',)), (2, 7))

    def test_hadamard_swaps_pauli_bits(self):
        graph = {'nodes': {'a': {'kind': 'boundary'}, 'h': {'kind': 'H'},
                           'b': {'kind': 'boundary'}},
                 'edges': [{'u': 'a', 'v': 'h'}, {'u': 'h', 'v': 'b'}]}
        rows = constraints(graph)
        permitted = []
        for a, b in product('IXYZ', repeat=2):
            x, z = BITS[a]; u, v = BITS[b]
            word = x + 2*z + 4*u + 8*v
            if all((word&r).bit_count()%2 == 0 for r in rows):
                permitted.append((a,b))
        self.assertEqual(set(permitted), {('I','I'),('X','Z'),('Z','X'),('Y','Y')})

if __name__ == '__main__':
    unittest.main()
