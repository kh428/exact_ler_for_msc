"""Evaluate the stored d=3 or d=5 finite LER series. This does not run a simulation."""
import argparse
import json
from fractions import Fraction

from exact_data import decimal, fraction, load_json, polynomial, positive_partial_sums, stored_exact_point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5), default=5)
    parser.add_argument("--p", required=True, help="Exact fraction or decimal, e.g. 1/1000 or 0.001")
    parser.add_argument("--order", type=int, default=10, choices=range(11))
    args = parser.parse_args()
    p = Fraction(args.p)
    if not 0 <= p < 1:
        parser.error("Use 0 <= p < 1; p=1 makes x undefined.")
    series = load_json(f"data/series/d{args.distance}.json")
    coefficients = [fraction(v) for v in series["conditional_LER_coefficients"]]
    x = p / (1 - p)
    partial = polynomial(coefficients[:args.order + 1], x)
    report = {
        "distance": args.distance, "p": str(p), "x": str(x), "order": args.order,
        "finite_LER_series": decimal(partial),
        "meaning": "S_K=sum_{k=0}^K L_k x^k. Its coefficients are exact; it omits higher powers.",
    }
    exact = stored_exact_point(args.distance, p)
    if exact is not None:
        accepted, bad = positive_partial_sums(series, p, args.order)
        tail = exact["A"] - accepted
        report.update({
            "stored_all_orders_LER": decimal(exact["LER"]),
            "exact_minus_finite_series": decimal(exact["LER"] - partial),
            "absolute_relative_series_error": decimal(abs(exact["LER"] - partial) / exact["LER"]),
            "acceptance_tail_interval": [decimal(bad / exact["A"]),
                                         decimal((bad + tail) / exact["A"])],
            "interval_meaning": "Uses the exact A and positive raw fault-weight sums; it is not an error bar obtained from successive L_k.",
            "display": "All printed decimals are rounded displays; interval endpoints here are not outward-rounded certificates.",
        })
    else:
        report["limitation"] = "No stored exact point at this p. This command supplies neither a new all-orders value nor a certified truncation error."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
