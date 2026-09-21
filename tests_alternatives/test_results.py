"""Regressions for scientifically distinct coefficient and comparison conventions."""
from copy import deepcopy
from fractions import Fraction as F
import unittest

from alternatives.results import DATA, evaluate, latex_series, load_case, quotient, read_json
from alternatives.verification import projector_coefficients, verify_case, verify_comparisons
from verify_results import prime


class AlternativeResultsTest(unittest.TestCase):
    def test_partial_acceptance_is_sufficient_only_after_leading_zeros(self):
        record = load_case("chan-four-round-d5")
        self.assertEqual(quotient(record["a"], record["e"]), list(map(F, record["L"])))
        # An added lower-order error makes an absent acceptance coefficient necessary.
        altered = record["e"][:]
        altered[0] = "1"
        with self.assertRaisesRegex(ValueError, "Missing acceptance coefficient"):
            quotient(record["a"], altered)

    def test_noise_strength_is_converted_to_x(self):
        record = load_case("rp2-d3")
        expected = F(131, 450) * F(1, 999)**2
        self.assertEqual(evaluate(record, "1/1000", 2), expected)
        self.assertNotEqual(expected, F(131, 450) * F(1, 1000)**2)

    def test_uncomputed_orders_are_not_silently_padded(self):
        with self.assertRaises(ValueError):
            evaluate(load_case("fold-d5"), "1/1000", 7)

    def test_invalid_noise_strength_is_rejected(self):
        for p in ("-0.001", "1", "2"):
            with self.subTest(p=p), self.assertRaises(ValueError):
                evaluate(load_case("rp2-d5"), p)

    def test_repair_changes_leading_order(self):
        old, new = verify_case("chan-v1-d5"), verify_case("chan-four-round-d5")
        self.assertEqual((old["fault_distance"], new["fault_distance"]), (4, 5))
        ratio = evaluate(new, "1/1000", 6) / evaluate(old, "1/1000", 6)
        self.assertAlmostEqual(float(ratio), .9943146426093127, places=14)

    def test_latex_remainder_starts_after_last_kept_degree(self):
        text = latex_series(load_case("fold-d5"))
        self.assertIn("O(x^{7})", text)
        self.assertNotIn("O(x^{6})", text)

    def test_projector_keeps_signed_cancellation(self):
        # A pure correct magic state has X=Y=1/sqrt(2).
        self.assertEqual(projector_coefficients([(F(1), F(0)), (F(0), F(1, 2)), (F(0), F(1, 2))]), (1, 0))
        # The opposite logical outcome has X=Y=-1/sqrt(2).
        self.assertEqual(projector_coefficients([(F(1), F(0)), (F(0), F(-1, 2)), (F(0), F(-1, 2))]), (1, 1))
        with self.assertRaises(ValueError):
            projector_coefficients([(F(1), F(0)), (F(1), F(0)), (F(0), F(0))])

    def test_changed_prime_witness_fails(self):
        record = read_json(DATA / "certificates/rp2-d3/q0_plus.json")
        certificate = deepcopy(record["prime_certificate"])
        certificate["witness"] = 1
        with self.assertRaises(ValueError):
            prime(certificate)

    def test_fold_diagnostics_do_not_claim_four_native_matches(self):
        result = verify_comparisons()
        self.assertEqual(result["proxy_B_inside_published_bars"], 4)
        self.assertEqual(result["native_B_inside_published_bars"], 3)


if __name__ == "__main__":
    unittest.main()
