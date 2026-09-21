"""Explicitly repeat a modular circuit calculation, an exact batch or the new repair growth sum."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    poly = sub.add_parser("polynomial", help="Full d=3 circuit, one prime and one sqrt(2) embedding")
    poly.add_argument("case", choices=("rp2-d3", "fold-d3", "chan-v1-d3"))
    poly.add_argument("--degree", type=int, default=5)
    poly.add_argument("--prime-index", type=int, choices=range(3), default=0)
    poly.add_argument("--embedding", choices=("plus", "minus"), default="plus")
    batch = sub.add_parser("batch", help="One exact d=5 sixth-order batch")
    batch.add_argument("case", choices=("rp2-d5", "fold-d5"))
    batch.add_argument("--batch", default="slice-00001")
    growth = sub.add_parser("chan-growth", help="Full four-round G4/G5 sum with saved endpoint functions")
    growth.add_argument("--threads", type=int, choices=range(1, 15), default=1)
    for command in (poly, batch, growth):
        command.add_argument("--output", required=True, type=Path, help="New directory for compilation and results")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    # Optional native dependencies are not imported by the help or verification commands.
    from alternatives.recompute import polynomial, phase_batch, chan_growth
    output.mkdir(parents=True)
    started = time.perf_counter()
    if args.command == "polynomial":
        result = polynomial(args.case, output, args.degree, args.prime_index, args.embedding)
    elif args.command == "batch":
        result = phase_batch(args.case, output, args.batch)
    else:
        result = chan_growth(output, args.threads)
    result["seconds"] = time.perf_counter() - started
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
