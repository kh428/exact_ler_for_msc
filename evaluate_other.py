"""Print exact finite series for RP2, Fold or repaired Chan cultivation."""
import argparse
from fractions import Fraction
from exact_data import decimal
from alternatives.results import cases, evaluate, latex_series, load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=cases())
    parser.add_argument("--p", default="1/1000", help="Decimal or fraction, e.g. 0.001 or 1/1000")
    parser.add_argument("--order", type=int, help="Highest retained degree (default: all stored coefficients)")
    parser.add_argument("--fraction", action="store_true", help="Also print the exact finite-polynomial fraction")
    args = parser.parse_args()
    record = load_case(args.case)
    order = record["degree"] if args.order is None else args.order
    try:
        p = Fraction(args.p)
        value = evaluate(record, p, order)
    except (ValueError, ZeroDivisionError) as error:
        parser.error(str(error))
    print(record["family"], f"d={record['distance']}")
    print(record["scope"])
    print("P_L(x) =", latex_series(record, order))
    print(f"At p={p}, x={p/(1-p)}: S_{order} = {decimal(value)}")
    if args.fraction:
        print("Exact value of this finite polynomial:", value)
    print("The coefficients are exact. The omitted higher orders have no supplied error bound.")


if __name__ == "__main__":
    main()
