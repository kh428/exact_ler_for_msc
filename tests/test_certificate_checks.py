"""Small negative controls for the readable certificate verifier; no simulations."""
import copy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_results as verifier
from exact_data import load_json, quotient_coefficients


class CertificateChecks(unittest.TestCase):
    def test_known_crt_example(self):
        self.assertEqual(verifier.crt([2, 3, 2], [3, 5, 7]), (23, 105))

    def test_quotient_has_acceptance_corrections(self):
        # x/(1+x) = x - x^2 + x^3 - ... . L_k need not be positive.
        result = quotient_coefficients([Fraction(1), Fraction(1), Fraction(0), Fraction(0)],
                                       [Fraction(0), Fraction(1), Fraction(0), Fraction(0)])
        self.assertEqual(result, [0, 1, -1, 1])

    def test_bad_primality_witness_is_rejected(self):
        data = load_json("data/exact_points/p001.json")
        certificate = dict(data["prime_records"][0]["certificate"], witness=1)
        with self.assertRaisesRegex(ValueError, "Primality witness failed"):
            verifier.prime(certificate)

    def assert_broken_point_rejected(self, change, error):
        entry = {"p": "1/1000", "file": "data/exact_points/p001.json"}
        point = copy.deepcopy(load_json(entry["file"]))
        change(point)

        def load_changed(relative):
            return point if relative == entry["file"] else load_json(relative)

        with patch.object(verifier, "load_json", side_effect=load_changed):
            with self.assertRaisesRegex(ValueError, error):
                verifier.verify_point(entry)

    def test_duplicate_character_is_rejected(self):
        def change(point):
            point["prime_records"][0]["characters"][1]["character"] = 0
        self.assert_broken_point_rejected(change, "Character coverage mismatch")

    def test_consistent_but_wrong_held_out_record_is_rejected(self):
        def change(point):
            row = point["prime_records"][-1]
            q = int(row["certificate"]["prime"])
            # Increase one character by eight, and its average by one.
            # Coverage and within-prime assembly still pass; the held-out check must fail.
            char = row["characters"][0]["residues"]
            char["A"] = (char["A"] + 8) % q
            row["residues"]["A"] = (row["residues"]["A"] + 1) % q
        self.assert_broken_point_rejected(change, "Held-out prime disagrees")

    def test_insufficient_reconstruction_bound_is_rejected(self):
        def change(point):
            point["reconstruction_primes"] = 1
            point["verification_primes"] = len(point["prime_records"]) - 1
        self.assert_broken_point_rejected(change, "CRT modulus does not exceed")


if __name__ == "__main__":
    unittest.main()
