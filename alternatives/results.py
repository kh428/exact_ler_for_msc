"""Read and evaluate the alternative-circuit results using exact fractions."""
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "alternatives"


def read_json(path):
    return json.loads(Path(path).read_text())


def cases():
    return [row["case"] for row in read_json(DATA / "index.json")["cases"]]


def load_case(name):
    if name not in cases():
        raise ValueError(f"Unknown case {name!r}; choose from {', '.join(cases())}")
    return read_json(DATA / "series" / f"{name}.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def batches(case):
    """Stream the lossless batch archive; no files are extracted."""
    with gzip.open(DATA / "certificates" / case / "batches.jsonl.gz", "rt") as stream:
        for line in stream:
            yield json.loads(line)


def quotient(a, e):
    """Divide raw fault-weight series, allowing only provably unused a_k to be absent.

    If e starts at degree d, L through degree K needs a only through K-d.
    Missing acceptance coefficients are never silently set to zero.
    """
    a, e = list(map(Fraction, a)), list(map(Fraction, e))
    require(a and a[0] != 0, "Acceptance needs a nonzero constant term")
    result = []
    for k, numerator in enumerate(e):
        correction = Fraction(0)
        for j in range(1, k + 1):
            if result[k - j] == 0:
                continue
            require(j < len(a), f"Missing acceptance coefficient a_{j} for L_{k}")
            correction += a[j] * result[k - j]
        result.append((numerator - correction) / a[0])
    return result


def evaluate(record, p, order=None):
    """Return an exact value of the finite polynomial, not an all-orders LER."""
    p = Fraction(p)
    require(0 <= p < 1, "p must satisfy 0 <= p < 1")
    order = record["degree"] if order is None else order
    require(isinstance(order, int) and 0 <= order <= record["degree"],
            f"Available orders are 0 through {record['degree']}")
    x = p / (1 - p)
    value = Fraction(0)
    for coefficient in reversed(record["L"][:order + 1]):
        value = value * x + Fraction(coefficient)
    return value


def latex_series(record, order=None):
    order = record["degree"] if order is None else order
    require(0 <= order <= record["degree"], "Requested coefficient is not available")
    terms = []
    for k, value in enumerate(record["L"][:order + 1]):
        c = Fraction(value)
        if not c:
            continue
        sign = "-" if c < 0 else ("+" if terms else "")
        c = abs(c)
        coefficient = str(c.numerator) if c.denominator == 1 else rf"\frac{{{c.numerator}}}{{{c.denominator}}}"
        terms.append(sign + coefficient + (rf"x^{{{k}}}" if k else ""))
    return " ".join(terms or ["0"]) + rf" + O(x^{{{order + 1}}})"
