"""Recompute the d=3 fixed-fault witness using a doubled ZX probability diagram."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("cat5", "bss", "cutting"), default="cat5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import tsim
    from tsim.sampler import CompiledStateProbs
    from zx.circuits import build
    results = {}
    for name, faults, last, expected in (("reference", False, 0, 1), ("good", True, 0, 0), ("bad", True, 1, .25)):
        circuit = tsim.Circuit(build("d3", with_faults=faults))
        record = np.zeros(circuit.num_measurements, dtype=np.uint8)
        record[-1] = last
        evaluator = CompiledStateProbs(circuit, strategy=args.strategy, seed=1)
        probability = float(evaluator.probability_of(record, batch_size=1)[0])
        if abs(probability-expected) > 1e-12:
            raise ValueError(f"{name} probability {probability} differs from {expected}")
        results[name] = probability
        print(f"d3/{name}: {probability:.16g}", flush=True)
    report = {"case": "d3", "strategy": args.strategy, "probabilities": results,
              "bloqade-tsim": version("bloqade-tsim"), "absolute_tolerance": 1e-12,
              "scalar_arithmetic": "floating point", "complete": True}
    (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
