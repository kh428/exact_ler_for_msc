"""Repeat a gate-level ZX record of the four-fault pattern in either Chan repair."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("chan-v1-d5", "chan-four-round-d5"))
    parser.add_argument("--record", type=int, choices=range(16), default=0)
    parser.add_argument("--outcome", choices=("reference", "good", "bad"), default="bad")
    parser.add_argument("--method", choices=("ladder", "cut", "both"), default="both")
    parser.add_argument("--save-graphs", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Output directory already exists; choose a new directory")
    import pyzx_param as zx
    from alternatives.chan_zx import build
    from alternatives.results import DATA, read_json, require, sha256
    from zx.amplitude import amplitude_graph, cat_ladder, decompose_paper
    from zx.graph_io import write_terms
    text, metadata = build(args.variant, args.record, args.outcome != "reference", int(args.outcome == "bad"))
    destination.mkdir(parents=True)
    circuit = destination / "circuit.stim"
    circuit.write_text(text)
    expected = read_json(DATA / "certificates" / args.variant / "zx" / f"d5_zx_record{args.record:02d}_{args.outcome}.json")
    require(sha256(circuit) == expected["circuit_sha256"], "Gate-level circuit differs from the saved witness")
    graph, wires, count = amplitude_graph(text, metadata["measurement_outcomes"])

    def trace(stage, terms):
        if args.save_graphs:
            write_terms(destination / f"{stage}.json", terms, stage=stage,
                        case=args.outcome, record=args.record, outcomes=metadata["measurement_outcomes"])

    trace("raw", [(graph, 1+0j)])
    if args.save_graphs:
        reduced = deepcopy(graph)
        zx.full_reduce(reduced)
        trace("full_reduce", [(reduced, 1+0j)])
    methods = ("ladder", "cut") if args.method == "both" else (args.method,)
    results = {}
    for method in methods:
        if method == "ladder":
            amplitude, leaves = cat_ladder(graph)
        else:
            amplitude, _ = decompose_paper(graph, trace=trace, log=lambda _: None)
        probability = float(abs(amplitude)**2)
        saved = next(iter(expected["methods"].values()))["probability"]
        require(abs(probability-saved) < 1e-12, "ZX probability differs from the saved record")
        results[method] = {"amplitude": [float(amplitude.real), float(amplitude.imag)], "probability": probability}
    report = {"variant": args.variant, "record": args.record, "outcome": args.outcome,
              "methods": results, "metadata": metadata, "saved_record_matches": True,
              "pyzx_param_distribution": version("pyzx-param"), "pyzx_param_internal": zx.__version__,
              "arithmetic": "Rational ZX phases; complex floating scalar sums; absolute tolerance 1e-12."}
    (destination / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
