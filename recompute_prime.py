"""Rebuild final responses and all growth characters for one saved grid prime."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", choices=("p0005", "p00075", "p001", "p0015", "p002"), default="p001")
    parser.add_argument("--prime-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    from exact_data import load_json
    from verify_results import prime as certified_prime
    from calculation.point import final_response, first_weights, PointExecutor
    point = load_json(f"data/exact_points/{args.point}.json")
    if not 0 <= args.prime_index < len(point["prime_records"]):
        parser.error("prime-index is outside the saved prime pool")
    record = point["prime_records"][args.prime_index]
    prime = certified_prime(record["certificate"])
    p = Fraction(point["p"])
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    response, checks = final_response(p, prime, args.output/"native", lambda s: print(s, flush=True))
    endpoint_seconds = time.perf_counter()-began
    executor = PointExecutor(p, prime, first_weights(args.point), response, args.output/"native")
    characters = []
    for character in range(executor.characters):
        row = executor.evaluate(character)
        expected = next(c for c in record["characters"] if c["character"] == character)
        if row["residues"] != expected["residues"]:
            raise ValueError(f"Character {character} disagrees with supplied residues")
        characters.append(row)
        print(f"Growth character {character+1}/8 agrees for A and B", flush=True)
    residues = {key: sum(c["residues"][key] for c in characters)*pow(8,-1,prime)%prime for key in ("A","B")}
    if residues != record["residues"]:
        raise ValueError("Eight-character assembly disagrees")
    report = {"complete": True, "p": str(p), "certificate": record["certificate"],
              "prime_index": args.prime_index, "characters": characters, "residues": residues,
              "endpoint_scalar_checks": checks, "endpoint_seconds": endpoint_seconds,
              "total_seconds": time.perf_counter()-began,
              "first_stage_input": f"data/endpoints/{args.point}_first_stage.json",
              "scope": "Final responses rebuilt; complete growth sum; supplied exact first-stage weights."}
    (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Complete: {report['total_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
