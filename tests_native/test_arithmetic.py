"""Native integer arithmetic compared with small Python-integer references."""
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calculation.native_modular_binary import BinaryField
from calculation.native_modular_polynomial import PolynomialField
from calculation.truncated_polynomial import PolynomialRing
from calculation.walsh import walsh_low

Q = 2305940865748566017


class ArithmeticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="msc-native-test-")
        cls.field = BinaryField(Q, Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_montgomery_multiply_matches_python(self):
        left = np.array([0, 1, Q-1, 123456789123456789], dtype=np.uint64)
        original = left.copy()
        self.field.convert(left, encode=True)
        got = np.array([self.field.scalar_multiply(a, b) for a, b in zip(left, left[::-1])], dtype=np.uint64)
        self.field.convert(got, encode=False)
        expected = np.array([int(a)*int(b)%Q for a, b in zip(original, original[::-1])], dtype=np.uint64)
        np.testing.assert_array_equal(got, expected)

    def test_walsh_matches_character_sum(self):
        values = np.array([0, 1, Q-1, 43, 91, 101, Q-2, 0], dtype=np.uint64)
        original = values.copy()
        expected = [sum((-1)**((i&j).bit_count())*int(v) for j, v in enumerate(original))%Q for i in range(8)]
        walsh_low(values, 3, Q)
        np.testing.assert_array_equal(values, expected)
        walsh_low(values, 3, Q, inverse=True)
        np.testing.assert_array_equal(values, original)

    def test_degree_ten_native_cauchy_product(self):
        ring = PolynomialRing(Q, 10)
        field = PolynomialField(ring, Path(self.folder.name))
        a = ring.coefficients(ring(Fraction((-1)**k, 3**k)).coefficients[0] for k in range(11))
        b = ring.coefficients(range(1, 12))
        left = field.encode_polynomial(a, degree=10)
        right = field.encode_polynomial(b, degree=10)
        field.broadcast(left, right, (), (), 10)
        self.assertEqual(field.decode_polynomial(left), a*b)


if __name__ == "__main__":
    unittest.main()
