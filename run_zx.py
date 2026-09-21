"""Recompute fixed-fault ZX witnesses and optionally export every cutting stage."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

# Set these before importing numerical libraries. Each invocation uses one worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("d3", "d5"))
    parser.add_argument("--method", choices=("ladder", "cut", "both"), default="ladder")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for circuits, graph JSON and probabilities")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import pyzx_param as zx
    from zx.amplitude import amplitude_graph, cat_ladder, decompose_paper, parse
    from zx.circuits import CASES, build
    from zx.graph_io import read_terms, write_terms

    report = {"case": args.case, "method": args.method,
              "versions": {"pyzx-param_distribution": version("pyzx-param"),
                           "pyzx_param_internal": zx.__version__, "stim": version("stim")},
              "scalar_arithmetic": "complex floating point", "absolute_tolerance": 1e-12,
              "records": {}}
    for label, faults, last, expected in (
        ("reference", False, 0, 1.), ("good", True, 0, CASES[args.case]["expected"][0]),
        ("bad", True, 1, CASES[args.case]["expected"][1])):
        text = build(args.case, with_faults=faults)
        (args.output / f"{label}.stim").write_text(text+"\n")
        count = sum(len(targets) for op, targets in parse(text) if op in ("M", "MX"))
        outcomes = [0]*count
        outcomes[-1] = last
        graph, wires, _ = amplitude_graph(text, outcomes)

        def export(stage, terms):
            path = args.output / f"{label}_{stage}.json"
            write_terms(path, terms, case=args.case, record=label, outcomes=outcomes, stage=stage)
            # Reading immediately exercises the public interchange format.
            restored = read_terms(path)
            from zx.amplitude import signature
            if len(restored) != len(terms):
                raise ValueError("Graph round-trip term count differs")
            for (original, weight), (copy, other_weight) in zip(terms, restored):
                if signature(original) != signature(copy) or weight != other_weight:
                    raise ValueError("Graph round-trip changed topology, phases or weight")
                if vars(original.scalar) != vars(copy.scalar):
                    from zx.graph_io import encode
                    if encode(vars(original.scalar)) != encode(vars(copy.scalar)):
                        raise ValueError("Graph round-trip changed the scalar")

        export("raw", [(graph, 1+0j)])
        reduced = deepcopy(graph)
        zx.full_reduce(reduced)
        export("full_reduce", [(reduced, 1+0j)])
        results = {}
        for method in (("ladder", "cut") if args.method == "both" else (args.method,)):
            start = time.perf_counter()
            if method == "cut":
                amplitude, stats = decompose_paper(graph, trace=export,
                    log=lambda message: print(f"{args.case}/{label}: {message}", flush=True))
            else:
                amplitude, leaves = cat_ladder(graph)
                stats = {"clifford_leaves": leaves}
            probability = float(abs(amplitude)**2)
            if abs(probability-expected) > 1e-12:
                raise ValueError(f"{label}: {probability} differs from {expected}")
            results[method] = {"amplitude": [float(amplitude.real), float(amplitude.imag)],
                               "probability": probability, "seconds": time.perf_counter()-start,
                               "statistics": stats}
            print(f"{args.case}/{label}/{method}: P={probability:.16g}", flush=True)
        report["records"][label] = {"expected_probability": expected, "wires": wires,
                                   "measurement_bits": count, "methods": results}
    report["complete"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
