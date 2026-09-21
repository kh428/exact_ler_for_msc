"""Check saved exact-result certificates without importing or running simulation code.

This verifies hashes, prime witnesses, character coverage, reconstruction bounds,
CRT, quotient coefficients, and positive-tail inequalities. It does not rederive
the physical tensor network or rerun the ZX calculations.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from exact_data import ROOT, decimal, fraction, load_json, positive_partial_sums, quotient_coefficients


def require(condition, message):
    """Validation must remain active even if Python is run with optimisation flags."""
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=None)
def certified_prime(q, odd_k, power_of_two, witness):
    """Proth's theorem: k odd, 0<k<2^m, q=k 2^m+1, a^((q-1)/2)=-1 (mod q)."""
    require(odd_k > 0 and odd_k % 2 == 1, "Proth k must be positive and odd")
    require(odd_k < 2 ** power_of_two, "Proth k must be smaller than 2^m")
    require(q == odd_k * 2 ** power_of_two + 1, "Incorrect Proth representation")
    require(pow(witness, (q - 1) // 2, q) == q - 1, "Primality witness failed")
    require(2 ** 61 < q < 2 ** 62, "Prime outside the recorded 62-bit arithmetic domain")
    return q


def prime(certificate):
    return certified_prime(int(certificate["prime"]), int(certificate["odd_k"]),
                           int(certificate["power_of_two"]), int(certificate["witness"]))


def crt(residues, primes):
    """Return the unique integer in [0, product(primes)) having these residues."""
    require(len(residues) == len(primes), "CRT length mismatch")
    integer, modulus = 0, 1
    for residue, q in zip(residues, primes):
        correction = (residue - integer) * pow(modulus, -1, q) % q
        integer += modulus * correction
        modulus *= q
    return integer, modulus


def fraction_mod(value, q):
    return value.numerator * pow(value.denominator, -1, q) % q


def flattened(values):
    for value in values:
        if isinstance(value, list):
            yield from flattened(value)
        else:
            yield int(value)


def verify_integrity():
    """Check every distributed file against the package SHA256 manifest."""
    manifest = load_json("manifest.json")
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        require(path.is_relative_to(ROOT), "Manifest path escapes package")
        require(path.is_file(), "Missing file: " + item["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                "File hash mismatch: " + item["path"])
    return {"manifest_files": len(manifest["files"])}


def verify_denominator(point_name, point):
    """Check the recorded denominator accounting; tensor compilation is not rerun."""
    config = load_json(f"data/proofs/{point_name}_config.json")
    denominator = int(point["denominator_hex"], 16)
    p = fraction(point["p"])
    growth = ((1 - 2 * p).denominator ** 130
              * (1 - Fraction(4, 3) * p).denominator ** 456
              * (1 - Fraction(16, 15) * p).denominator ** 322)
    first = load_json(f"data/endpoints/{point_name}_first_stage.json")
    require(fraction(first["p"]) == p and first["complete"], "First-stage p or completion mismatch")
    common = int(first["denominator"])
    common //= math.gcd(common, *flattened(first["numerators"]))
    if "denominator_hex" in config:
        final = math.lcm(*(int(v["denominator_hex"], 16) for v in config["final_factor_bounds"])) * 2 ** 11
        require(final == int(config["F_denominator_hex"], 16), "Final-response denominator mismatch")
        require(common == int(config["W_denominator_hex"], 16), "First-stage denominator mismatch")
        require(growth == int(config["growth_denominator_hex"], 16), "Growth denominator mismatch")
    else:
        proof = load_json("data/proofs/p001_factor_denominator_bound.json")
        final = math.prod(int(base) ** exponent for base, exponent in proof["F_sufficient_denominator"].items())
        require(common == math.prod(int(base) ** exponent for base, exponent in proof["W_exact_common_denominator"].items()),
                "Anchor first-stage denominator mismatch")
        require(denominator == math.prod(int(base) ** exponent for base, exponent in config["denominator_factors"].items()),
                "Anchor factored denominator mismatch")
    require(denominator == common * final * growth * 2 ** 75, "Combined denominator mismatch")
    require(denominator.bit_length() == point["denominator_bits"], "Denominator bit count mismatch")
    return denominator


def verify_point(entry):
    point = load_json(entry["file"])
    name = Path(entry["file"]).stem
    require(fraction(entry["p"]) == fraction(point["p"]), "Grid index p mismatch")
    source = ROOT / "data/inputs/soft_cultivation_d5_p0005.stim"
    require(hashlib.sha256(source.read_bytes()).hexdigest() == point["source_circuit_sha256"],
            "Circuit digest mismatch")
    denominator = verify_denominator(name, point)
    rows = point["prime_records"]
    qs = [prime(row["certificate"]) for row in rows]
    require(len(set(qs)) == len(qs), "Duplicate reconstruction prime")
    count = point["reconstruction_primes"]
    require(len(qs) == count + point["verification_primes"], "Prime coverage mismatch")
    require(math.prod(qs[:count]) > denominator, "CRT modulus does not exceed the positive height bound")
    modular = {key: [] for key in ("A", "B")}
    for index, (row, q) in enumerate(zip(rows, qs)):
        require(row["index"] == index, "Prime index mismatch")
        require(math.gcd(denominator, q) == 1, "Denominator is not invertible")
        chars = row["characters"]
        require(sorted(c["character"] for c in chars) == list(range(8)), "Character coverage mismatch")
        for key in ("A", "B"):
            values = [int(c["residues"][key]) for c in chars]
            require(all(0 <= v < q for v in values), "Residue outside the field")
            average = sum(values) * pow(8, -1, q) % q
            require(average == int(row["residues"][key]), "Eight-character assembly mismatch")
            modular[key].append(denominator * average % q)
    numerators = {}
    for key in ("A", "B"):
        residues = modular[key]
        integer, modulus = crt(residues[:count], qs[:count])
        reverse = crt(residues[:count][::-1], qs[:count][::-1])
        require((integer, modulus) == reverse, "Reverse CRT disagrees")
        require(0 <= integer <= denominator, "Reconstructed probability is out of range")
        require(all(integer % q == residue for q, residue in zip(qs[count:], residues[count:])),
                "Held-out prime disagrees")
        require(Fraction(integer, denominator) == fraction(point["exact"][key]), "Exact fraction disagrees")
        numerators[key] = integer
    require(0 <= numerators["B"] <= numerators["A"] <= denominator and numerators["A"] > 0,
            "A/B event inclusion or positive acceptance fails")
    ler = Fraction(numerators["B"], numerators["A"])
    require(ler == fraction(point["exact"]["LER"]), "B/A differs from saved LER")
    return {"p": point["p"], "LER": decimal(ler), "reconstruction_primes": count,
            "held_out_primes": len(qs) - count, "characters_checked": len(qs) * 8,
            "denominator_bits": denominator.bit_length()}


def verify_series(distance):
    series = load_json(f"data/series/d{distance}.json")
    degree = series["degree"]
    records = series["modular_records"]
    qs = [prime(row["certificate"]) for row in records]
    count = series["reconstruction_primes"]
    require(len(qs) == count + series["verification_primes"] and len(set(qs)) == len(qs),
            "Series prime coverage mismatch")
    require(series["raw_source_locations"] == series["retained_physical_locations"]
            + series["removed_identity_response_locations"], "Location counts disagree")
    modular = []
    for row, q in zip(records, qs):
        if distance == 5:
            chars = row["characters"]
            require(sorted(c["character"] for c in chars) == list(range(8)), "Polynomial character coverage mismatch")
            for char in chars:
                for key in ("A", "B"):
                    require(len(char["coefficients"][key]) == degree + 1, "Polynomial degree mismatch")
                    require(all(0 <= int(v) < q for v in char["coefficients"][key]), "Invalid coefficient residue")
            coefficients = {key: [sum(c["coefficients"][key][k] for c in chars) * pow(8, -1, q) % q
                                  for k in range(degree + 1)] for key in ("A", "B")}
        else:
            coefficients = row["coefficients"]
            require(all(len(v) == degree + 1 for v in coefficients.values()), "d3 polynomial degree mismatch")
            require(all(0 <= int(value) < q for values in coefficients.values() for value in values),
                    "Invalid d3 residue")
        modular.append(coefficients)
    raw = {key: [] for key in ("A", "B")}
    max_height_bits = 0
    for k in range(degree + 1):
        # The conservative proof allows the additional terminal projector denominator.
        scale = 2 ** series["conservative_dyadic_power"] * 15 ** k
        height = scale * math.comb(series["retained_physical_locations"], k)
        require(math.prod(qs[:count]) > height, "Insufficient coefficient CRT modulus")
        max_height_bits = max(max_height_bits, height.bit_length())
        for key in ("A", "B"):
            residues = [scale * row[key][k] % q for row, q in zip(modular, qs)]
            integer, modulus = crt(residues[:count], qs[:count])
            require((integer, modulus) == crt(residues[:count][::-1], qs[:count][::-1]), "Coefficient reverse CRT disagrees")
            require(0 <= integer <= height, "Coefficient exceeds its positive height bound")
            require(all(integer % q == residue for q, residue in zip(qs[count:], residues[count:])),
                    "Held-out coefficient prime disagrees")
            coefficient = Fraction(integer, scale)
            require(coefficient == fraction(series["raw_coefficients"][key][k]), "Raw coefficient disagrees")
            raw[key].append(coefficient)
        require(0 <= raw["B"][k] <= raw["A"][k] <= math.comb(series["retained_physical_locations"], k),
                "Coefficient event-inclusion bound fails")
    require(raw["A"][0] == 1, "a_0 must be one in this convention")
    ler = quotient_coefficients(raw["A"], raw["B"])
    require(ler == [fraction(v) for v in series["conditional_LER_coefficients"]], "LER quotient recursion disagrees")
    expected_leading = Fraction(32, 75) if distance == 3 else Fraction(574, 375)
    leading_order = 2 if distance == 3 else 3
    require(all(v == 0 for v in ler[:leading_order]) and ler[leading_order] == expected_leading,
            "Leading coefficient mismatch")
    return {"distance": distance, "raw_coefficients_checked": 2 * (degree + 1),
            "LER_coefficients_checked": len(ler), "through_order": degree,
            "reconstruction_primes": count, "held_out_primes": len(qs) - count,
            "maximum_conservative_height_bits": max_height_bits}


def verify_pointwise_tails():
    """Check positive-tail inequalities against known exact points, not convergence claims."""
    count = 0
    for distance in (3, 5):
        series = load_json(f"data/series/d{distance}.json")
        if distance == 5:
            points = []
            for entry in load_json("data/exact_points/index.json"):
                row = load_json(entry["file"])
                points.append((fraction(row["p"]), {key: fraction(v) for key, v in row["exact"].items()}))
        else:
            points = [(fraction(row["p"]), {key: fraction(row[key + "_exact"]) for key in ("A", "B", "LER")})
                      for row in load_json("data/exact_points/d3_saved_grid.json")["rows"]]
        for p, exact in points:
            require(0 <= exact["B"] <= exact["A"] <= 1 and exact["A"] > 0, "Invalid saved d3/d5 probability")
            require(exact["LER"] == exact["B"] / exact["A"], "Saved ratio disagrees")
            accepted, bad = positive_partial_sums(series, p, series["degree"])
            tail = exact["A"] - accepted
            require(0 <= exact["B"] - bad <= tail, "Positive physical-fault tail inequality fails")
            require(bad / exact["A"] <= exact["LER"] <= (bad + tail) / exact["A"], "LER enclosure fails")
            count += 1
    return count


def verify_saved_zx():
    """Read saved numerical witnesses; do not call tsim or pyzx_param."""
    count = 0
    for path in sorted((ROOT / "data/zx/records").rglob("result*.json")):
        row = json.loads(path.read_text())
        if "p_accept_bad" in row:
            good, bad = row["p_accept_good"], row["p_accept_bad"]
            reference = row["faultfree_flip0"]
        elif "good" in row and "bad" in row:
            good, bad = row["good"]["p"], row["bad"]["p"]
            reference = row["faultfree_flip0"]["p"]
        else:
            raise ValueError("Unrecognised ZX record " + path.name)
        expected = 0.25 if row["case"] == "d3" else 0.0625
        require(abs(good) < 1e-12 and abs(bad - expected) < 1e-12 and abs(reference - 1) < 1e-12,
                "Saved numerical ZX witness mismatch")
        count += 1
    return {"saved_numerical_records_checked": count, "fresh_ZX_runs": 0,
            "qualification": "Graph transformations may be exact; these stored scalar probabilities were evaluated in floating point."}


def verify_saved_growth_membership():
    """Check the recorded Choi membership results, without rebuilding the Choi state."""
    saved = load_json("data/validation/20260905_growth_choi_membership.json")
    records = saved["records"]
    require(len(records) == 16 and len({r["record_mask"] for r in records}) == 16,
            "Growth Choi record coverage mismatch")
    for row in records:
        require(row["input_CSS_projector_matches"] and row["reference_only_rank"] == 6,
                "Recorded input-code membership failed")
        require(row["output_positive_CSS_matches"] and row["output_data_only_rank"] == 18,
                "Recorded output-code membership failed")
        require(row["choi_probability_power_of_two"] == 10,
                "Recorded Choi normalisation differs")
    return {"saved_records_checked": 16, "fresh_stabiliser_calculations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optionally write this verification report to a new path")
    args = parser.parse_args()
    report = {}
    report["integrity"] = verify_integrity()
    report["exact_d5_points"] = [verify_point(entry) for entry in load_json("data/exact_points/index.json")]
    report["series"] = [verify_series(distance) for distance in (3, 5)]
    report["pointwise_tail_checks"] = verify_pointwise_tails()
    report["ZX_saved_records"] = verify_saved_zx()
    report["growth_saved_membership"] = verify_saved_growth_membership()
    report["complete"] = True
    report["scope"] = "Reconstruction from the supplied character residues, coefficient residues and saved ZX probabilities."
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
