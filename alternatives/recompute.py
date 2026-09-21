"""Small, explicit recomputations using the supplied circuit models and kernels."""
import ctypes as ct
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import subprocess

import numpy as np

from verify_results import fraction_mod, prime
from .results import DATA, batches, load_case, read_json, require
from .calculation.build import build


def polynomial(case, output, degree, prime_index, sign):
    """Recompile one d=3 circuit and contract all slices for one prime/embedding."""
    from .calculation.pauli_polynomial import PolynomialRing, compile_puri, eliminate_linear_constraints, evaluate
    from .calculation.rp2 import compile_rp2
    from .calculation.chan import compile_chan

    record = load_case(case)
    require(0 <= degree <= record["degree"], "Degree exceeds the saved comparison")
    reference = read_json(DATA / "certificates" / case / f"q{prime_index}_{sign}.json")
    q = prime(reference["prime_certificate"])
    root2 = reference["sqrt2_embedding"]
    model = read_json(DATA / record["model"])
    ring = PolynomialRing(q, degree)
    compiler = {"rp2-d3": compile_rp2, "fold-d3": compile_puri, "chan-v1-d3": compile_chan}[case]
    observables = ("A", "B") if case == "chan-v1-d3" else ("A", "X", "Y")
    outputs = {}
    for observable in observables:
        net, stats = compiler(model, ring, observable, root2=root2, basis="xz")
        # Match the archived exact coordinate reduction before reusing its order.
        try:
            net, stats["linear_reduction"] = eliminate_linear_constraints(net)
        except RuntimeError:
            if case != "chan-v1-d3":
                raise
        ref = reference["outputs"][observable]
        slices = ref.get("slices", [])
        plan = ref.get("plan", ref.get("evidence", {}).get("plan"))
        require(plan is not None, "Saved contraction plan missing")
        total = [0] * (degree + 1)
        for assignment in product((0, 1), repeat=len(slices)):
            child = net.condition(dict(zip(slices, assignment))) if slices else net
            current_plan = plan
            if set(plan["order"]) != child.variables:
                require(case == "chan-v1-d3", "Compiled network does not match the saved order")
                # The Chan compiler now chains detector XORs explicitly, changing
                # auxiliary labels. Choose a fresh bounded order of this network;
                # do not guess a correspondence between old and new indices.
                current_plan = None
            values, evidence = evaluate(child, build_dir=output / "build", plan=current_plan,
                                        max_work=70_000_000_000, max_bytes=700*2**20)
            require(values is not None, "Contraction exceeds the memory/work guard")
            total = [(a + b) % q for a, b in zip(total, values)]
        require(total == ref["residues"][:degree+1], f"{observable} differs from archived contraction")
        outputs[observable] = total
        print(f"{case}: {observable}, all {2**len(slices)} slices match", flush=True)
    if "B" not in outputs:
        h = root2 * pow(2, -1, q) % q
        outputs["B"] = [(a-h*(x+y))*pow(2, -1, q) % q
                        for a, x, y in zip(outputs["A"], outputs["X"], outputs["Y"])]
        require(outputs["B"] == reference["B_residues"][:degree+1], "Wrong-outcome projector mismatch")
    return {"case": case, "degree": degree, "prime": str(q), "embedding": sign,
            "all_saved_residues_match": True, "outputs": outputs,
            "scope": "One modular calculation, rebuilt from the frozen d=3 gate/noise model."}


def phase_batch(case, output, batch_id):
    """Repeat one complete saved sixth-order batch with freshly compiled kernels."""
    from .calculation.phase_batches import SixthContext, histogram_sum, fraction
    from .verification import verify_phase_input

    verify_phase_input(case)
    reference = next((r for r in batches(case) if r["id"] == batch_id), None)
    require(reference is not None, "Unknown batch identifier")
    directory = DATA / "inputs" / case / "phase_source"
    source = "phase_sixth_direct.cpp" if case == "rp2-d5" else "fold_sixth_radical.cpp"
    library = build(source, output / "build")
    context = SixthContext(read_json(directory / "plan.json"),
                           list(map(int, read_json(directory / "labels.json"))),
                           list(map(int, read_json(directory / "signatures.json"))),
                           np.load(directory / "powers.npy", allow_pickle=False), library)
    try:
        task = {k: reference[k] for k in ("kind", "index", "begin", "end")}
        if case == "rp2-d5":
            require(context.prepare_zero() == [1884175, 18105], "RP2 zero-signature coverage")
            actual = context.run(**task)
            for key in ("sums", "lower_sums", "stats"):
                require(actual[key] == reference[key], "Recomputed RP2 batch differs: " + key)
        else:
            lib = context.kernel.lib
            pointer = lambda a: ct.cast(a.ctypes.data, ct.POINTER(ct.c_uint64))
            u = ct.POINTER(ct.c_uint64)
            lib.fold_fast_zero.argtypes = [ct.c_void_p, ct.c_uint64, u]
            counts = np.zeros(2, dtype=np.uint64)
            require(lib.fold_fast_zero(context.ptr, 25_000_000, pointer(counts)) == 0, "Fold zero-source allocation")
            require(list(map(int, counts)) == [14657825, 77280], "Fold zero-signature coverage")
            lib.fold_accelerated_six.argtypes = [ct.c_void_p, ct.c_int, ct.c_uint32,
                                                ct.c_uint64, ct.c_uint64, ct.c_uint64, ct.c_int, ct.c_int, u, u]
            limbs, stats = np.zeros(8192, dtype=np.uint64), np.zeros(15, dtype=np.uint64)
            code = lib.fold_accelerated_six(context.ptr, task["kind"], task["index"], task["begin"], task["end"],
                                            reference["triples"], 0, 32, pointer(limbs), pointer(stats))
            require(code == 0, f"Fold batch guard {code}")
            value = str(fraction(histogram_sum(limbs)))
            require(value == reference["sum"], "Recomputed Fold weighted sum differs")
            # Internal optimisation counters may differ by platform. The census cannot.
            require(int(stats[3]) == reference["stats"][3], "Recomputed Fold census differs")
            actual = {"sum": value, "stats": list(map(int, stats))}
    finally:
        context.close()
    return {"case": case, "batch": batch_id, "exact_sum_matches": True, "result": actual,
            "scope": "One complete batch, using the archived compiled phase source. Not a new full coefficient."}


def chan_growth(output, threads):
    """Recompute G4/G5; the certified endpoint functions and G0..G3 sum are reused."""
    from .calculation.chan_powers import powers
    model = read_json(DATA / "inputs/chan-four-round-d5/growth.json")
    terms = powers(model)
    path = output / "channel_powers.txt"
    with path.open("w") as stream:
        stream.write(str(len(terms)) + "\n")
        for label, coefficients in sorted(terms.items()):
            stream.write(f"{label & (2**64-1)} {label >> 64} " + " ".join(map(str, coefficients)) + "\n")
    executable = build("chan_growth.cpp", output / "build", shared=False)
    source = DATA / "inputs/chan-four-round-d5"
    destination = output / "growth.json"
    subprocess.run([str(executable), str(path), str(source / "h1.txt"), str(source / "h2.txt"),
                    str(model["constraint_bits"]), str(threads), "5", "0", str(destination)],
                   check=True, timeout=600)
    result = read_json(destination)
    saved = read_json(DATA / "certificates/chan-four-round-d5/new_growth_mode5_limit0.json")
    for key in ("G4_h1", "G4_h2", "G5_h1", "C_groups_used", "C_groups_all"):
        require(result[key] == saved[key], "Chan growth mismatch: " + key)
    require(result["complete"] and result["growth_five_computed"], "Incomplete growth run")
    return {"case": "chan-four-round-d5", "growth_sums_match": True,
            "scope": "G4/G5 rebuilt from growth channels; saved exact h1/h2 and certified G0..G3 endpoint sum reused."}
