"""Small, readable helpers for the accompanying exact results (standard library only)."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import sys

# The stored, trusted fractions have more digits than Python's default limit.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(40000)

ROOT = Path(__file__).resolve().parent


def load_json(relative_path):
    """Read a data file relative to this package, independently of the current directory."""
    return json.loads((ROOT / relative_path).read_text())


def fraction(value):
    """Decode either 'numerator/denominator' or the two-field JSON representation."""
    if isinstance(value, dict):
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    return Fraction(value)


def decimal(value, digits=30):
    """Display an exact fraction as a rounded decimal; leave the fraction unchanged."""
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def quotient_coefficients(acceptance, accepted_error):
    """Return coefficients of e(x)/a(x), using exact rational arithmetic.

    At order k, a_0 L_k = e_k - sum_{j=1}^k a_j L_{k-j}.
    Both input lists contain coefficients starting at order zero.
    """
    if len(acceptance) != len(accepted_error) or not acceptance or acceptance[0] == 0:
        raise ValueError("Need equal nonempty lists and a nonzero a_0")
    result = []
    for k, error_coefficient in enumerate(accepted_error):
        earlier = sum(acceptance[j] * result[k - j] for j in range(1, k + 1))
        result.append((error_coefficient - earlier) / acceptance[0])
    return result


def polynomial(coefficients, x):
    """Horner evaluation of a finite coefficient list, lowest power first."""
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def positive_partial_sums(series, p, order):
    """A_K and E_K count up to K relevant faulty locations; these are not S_K."""
    if not 0 <= p < 1:
        raise ValueError("The parameter x=p/(1-p) requires 0 <= p < 1")
    x = p / (1 - p)
    prefactor = (1 - p) ** series["retained_physical_locations"]
    return tuple(
        prefactor * polynomial([fraction(v) for v in series["raw_coefficients"][key][:order + 1]], x)
        for key in ("A", "B")
    )


def stored_exact_point(distance, p):
    """Find a completed exact point. Absence does not imply a new point was evaluated."""
    if distance == 5:
        for entry in load_json("data/exact_points/index.json"):
            if fraction(entry["p"]) == p:
                result = load_json(entry["file"])
                return {key: fraction(value) for key, value in result["exact"].items()}
    else:
        for row in load_json("data/exact_points/d3_saved_grid.json")["rows"]:
            if fraction(row["p"]) == p:
                return {key: fraction(row[key + "_exact"]) for key in ("A", "B", "LER")}
    return None
