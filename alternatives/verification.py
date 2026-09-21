"""Arithmetic checks of saved coefficients and complete batch records.

These checks do not rerun a circuit simulation. Native recomputation is a
separate command, so ordinary verification needs only the Python standard library.
"""
from collections import Counter
from fractions import Fraction as F
import math
import csv
import gzip
import json

from verify_results import crt, fraction_mod, prime
from .results import DATA, batches, digest, load_case, quotient, read_json, require, sha256


def verify_case(name):
    row = load_case(name)
    require(sha256(DATA / row["model"]) == row["model_sha256"], name + ": model hash")
    require(len(row["L"]) == len(row["e"]) == row["degree"] + 1, name + ": degree")
    require(quotient(row["a"], row["e"]) == list(map(F, row["L"])), name + ": quotient recursion")
    require(next(k for k, v in enumerate(row["L"]) if F(v)) == row["fault_distance"], name + ": leading order")
    require(all(0 <= F(v) <= math.comb(row["locations_in_coefficient_convention"], k)
                for key in ("a", "e") for k, v in enumerate(row[key])), name + ": raw coefficient bound")
    return row


def signed_crt(residues, qs):
    integer, modulus = crt(residues, qs)
    return integer - modulus if 2 * integer > modulus else integer


def raw_residues(record, observable):
    if observable == "B" and "B_residues" in record:
        q = int(record["prime_certificate"]["prime"])
        a, x, y = [record["outputs"][key]["residues"] for key in ("A", "X", "Y")]
        h = record["sqrt2_embedding"] * pow(2, -1, q) % q
        derived = [(av - h * (xv + yv)) * pow(2, -1, q) % q for av, xv, yv in zip(a, x, y)]
        require(derived == record["B_residues"], "Wrong-outcome projector assembly")
        return derived
    return record["outputs"][observable]["residues"]


def verify_d3(name):
    row = verify_case(name)
    model = read_json(DATA / row["model"])
    directory = DATA / "certificates" / name
    result = read_json(directory / "result.json")
    counts = Counter(e["name"] for e in model["events"])
    if name == "fold-d3":
        power = (3 * counts["CCZ"] + 2 * (counts["CSX"] + counts["CS_DAG_X"])
                 + 3 * counts["T_DAG"] + 5 * counts["INJECT_HXY"] + counts["POST_Z"] + 2)
        noise_denominator = 315
        expected_source = row["model_sha256"]
    else:
        power = counts["T"] + counts["T_DAG"] + counts["MEASURE"] + counts["MEASURE_PRODUCT"] + 2
        noise_denominator = 15
        expected_source = model["source_sha256"]
    require(power == result["denominator_power_of_two"], name + ": dyadic denominator accounting")
    require(all(noise_denominator % c["denominator"] == 0 for c in model["channels"]), name + ": noise denominator")
    n = len(model["channels"])
    require(n == row["locations_in_coefficient_convention"], name + ": location count")
    records, qs = [], []
    count = 6 if name == "chan-v1-d3" else 3
    for i in range(count):
        plus, minus = [read_json(directory / f"q{i}_{s}.json") for s in ("plus", "minus")]
        q = prime(plus["prime_certificate"])
        require(minus["prime_certificate"] == plus["prime_certificate"], "Embedding primes differ")
        r = plus["sqrt2_embedding"]
        require(r * r % q == 2 and r + minus["sqrt2_embedding"] == q, "sqrt(2) embeddings")
        for record in (plus, minus):
            require(record["degree"] == row["degree"] and record["source_sha256"] == expected_source,
                    name + ": residue model/degree")
            require(not record.get("proxy", False) and not record.get("ideal", False), "Unexpected ideal/proxy run")
            for component in record["outputs"].values():
                if "slice_results" not in component:
                    continue
                assignments = [tuple(s["assignment"]) for s in component["slice_results"]]
                size = len(component["slices"])
                require(len(assignments) == len(set(assignments)) == 2 ** size, "Slice coverage")
                require(all(len(a) == size and set(a) <= {0, 1} for a in assignments), "Nonbinary slice")
                summed = [sum(s["residues"][k] for s in component["slice_results"]) % q
                          for k in range(row["degree"] + 1)]
                require(summed == component["residues"], "Slice assembly")
        records.append((plus, minus))
        qs.append(q)
    require(len(set(qs)) == count, "Duplicate prime")
    reconstruction_qs = qs[:3]
    modulus = math.prod(reconstruction_qs)
    for observable, field in (("A", "a"), ("B", "e")):
        values = [(raw_residues(p, observable), raw_residues(m, observable)) for p, m in records]
        for k in range(row["degree"] + 1):
            denominator = (1 << power) * noise_denominator ** k
            bound = denominator * math.comb(n, k)
            require(modulus > 2 * bound, "Insufficient reconstruction height")
            rational, radical = [], []
            for q, pair, (v, w) in zip(qs[:3], records[:3], values[:3]):
                r = pair[0]["sqrt2_embedding"]
                rational.append((v[k] + w[k]) * pow(2, -1, q) * denominator % q)
                radical.append((v[k] - w[k]) * pow(2 * r, -1, q) * denominator % q)
            u, v = signed_crt(rational, reconstruction_qs), signed_crt(radical, reconstruction_qs)
            require(abs(u) <= bound and v == 0, "Height or nonzero irrational part")
            require(F(u, denominator) == F(row[field][k]), name + ": reconstructed coefficient")
            for q, pair, (a, b) in zip(qs, records, values):
                require(a[k] == b[k] == u * pow(denominator, -1, q) % q, "Independent embedding/prime check")
    return {"orders": row["degree"], "prime_pairs": count, "irrational_parts": "proved zero"}


def expected_tasks(bits, zero_count, block, digits=7, lower=True):
    result = [{"id": "lower-zero", "kind": 2, "index": 0, "begin": 0, "end": 0}] if lower else []
    for start in range(0, zero_count, block):
        result.append(dict(id=f"zero-{start:0{digits}d}", kind=1, index=0, begin=start, end=min(zero_count, start + block)))
    result.extend(dict(id=f"slice-{i:05d}", kind=0, index=i, begin=0, end=0) for i in range(2 ** bits))
    return result


def checked_batches(name, tasks):
    directory = DATA / "certificates" / name
    manifest_hash = sha256(directory / "manifest.json")
    expected = {task["id"]: task for task in tasks}
    seen = set()
    for row in batches(name):
        key = row["id"]
        require(key in expected and key not in seen, "Unexpected or repeated batch: " + key)
        require(all(row[k] == v for k, v in expected[key].items()), "Batch identity: " + key)
        require(row["manifest_sha256"] == manifest_hash, "Batch manifest: " + key)
        require(row["sha256"] == digest({k: v for k, v in row.items() if k != "sha256"}), "Batch checksum: " + key)
        seen.add(key)
        yield row
    require(seen == set(expected), "Incomplete batch coverage")


def verify_phase_input(name):
    row = verify_case(name)
    directory = DATA / "inputs" / name / "phase_source"
    meta = read_json(directory / "manifest.json")
    require(meta.get("model_sha256", meta.get("source_sha256")) == row["model_sha256"], "Phase model hash")
    for name, checksum in meta["hashes"].items():
        require(sha256(directory / name) == checksum, "Phase input checksum: " + name)
    require(int(meta["signed_histogram_absolute_bound"]) < 2 ** 127, "Histogram integer bound")
    return meta


def dyadic_pair(value):
    """Return (r,s) for r+s sqrt(2), retaining both parts exactly."""
    if "real" in value:
        require(value["imag"]["a"] == value["imag"]["b"] == "0", "Non-real phase sum")
        value = value["real"]
    denominator = 2 ** value["denominator_power_of_two"]
    return F(int(value["a"]), denominator), F(int(value["b"]), denominator)


def projector_coefficients(values):
    """B=(A-(X+Y)/sqrt(2))/2 in the target-state convention."""
    (a, ar), (x, xr), (y, yr) = values
    require(ar == 0, "Acceptance contains an irrational part")
    require(x + y == 0, "Accepted-error coefficient contains an irrational part")
    return a, (a - xr - yr)/2


def verify_lower_orders(name):
    row = load_case(name)
    directory = DATA / "certificates" / name
    if name == "rp2-d5":
        first = read_json(directory / "exact_single_rp201_20260916.json")
        require(first["source_sha256"] == row["model_sha256"], "RP2 first-order source")
        values = [(F(first["outputs"][o]["rational"]), F(first["outputs"][o]["sqrt2"])) for o in ("A", "X", "Y")]
        require(projector_coefficients(values) == (F(row["a"][1]), F(row["e"][1])), "RP2 first order")
        a2, e2, covered = F(0), F(0), 0
        for i in range(7):
            pair = read_json(directory / f"pairs_{i}.json")
            require(pair["offset"] == i and pair["stride"] == 7 and pair["done"] == pair["term_count"], "RP2 pair coverage")
            require(pair["model_sha256"] == row["model_sha256"], "RP2 pair source")
            a, e = projector_coefficients([dyadic_pair(pair["sums"][o]) for o in ("A", "X", "Y")])
            a2 += a / pair["pair_weight_denominator"]
            e2 += e / pair["pair_weight_denominator"]
            covered += pair["done"]
        require(covered == 39567 and (a2, e2) == (F(row["a"][2]), F(row["e"][2])), "RP2 second order")
        third = read_json(directory / "third_rp2_complete01_20260916.json")
        require(third["total_tasks"] == third["selected_tasks"] and third["source_sha256"] == row["model_sha256"], "RP2 third-order coverage")
        values = []
        for observable in ("A", "X", "Y"):
            part = third["outputs"][observable]
            require(part["done"] == third["total_tasks"], "RP2 third-order task count")
            s111, s12, s3 = [dyadic_pair(part[k]) for k in ("S1_cubed", "S1_S2", "S3")]
            values.append(tuple((s111[i]-3*s12[i]+2*s3[i])/(6*15**3) for i in (0, 1)))
        require(projector_coefficients(values) == (F(row["a"][3]), F(row["e"][3])), "RP2 third order")
    else:
        with gzip.open(directory / "lower_orders.json.gz", "rt") as stream:
            low = json.load(stream)
        require(low["model_sha256"] == row["model_sha256"], "Fold lower-order source")
        moments = {key: [dyadic_pair(v) for v in values] for key, values in low["moments"].items()}
        for k in (1, 2, 3):
            values = []
            for j in range(3):
                if k == 1:
                    values.append(tuple(v/315 for v in moments["S1"][j]))
                elif k == 2:
                    values.append(tuple((moments["S1_squared"][j][i]-moments["S2"][j][i])/(2*315**2) for i in (0, 1)))
                else:
                    values.append(tuple((moments["S1_cubed"][j][i]-3*moments["S1_S2"][j][i]+2*moments["S3"][j][i])/(6*315**3) for i in (0, 1)))
            require(projector_coefficients(values) == (F(row["a"][k]), F(row["e"][k])), "Fold lower-order Newton identity")
        indices = [i for batch in low["batches"] for i in batch["indices"]]
        require(len(indices) == len(set(indices)) == low["tasks"] and set(indices) == set(range(low["tasks"])), "Fold triple task coverage")
        require(sum(batch["candidates"] for batch in low["batches"]) == low["candidate_triples"], "Fold triple census")
        sums = [[F(0), F(0)] for _ in range(3)]
        for batch in low["batches"]:
            for j, value in enumerate(batch["sums"]):
                sums[j] = [a+b for a, b in zip(sums[j], dyadic_pair(value))]
        require([tuple(v) for v in sums] == moments["S1_cubed"], "Fold third-order batch sum")
    return True


def verify_rp2_sixth():
    name = "rp2-d5"
    row, meta = verify_case(name), verify_phase_input(name)
    verify_lower_orders(name)
    directory = DATA / "certificates" / name
    manifest, summary = [read_json(directory / f"{s}.json") for s in ("manifest", "summary")]
    require(manifest["source_sha256"] == row["model_sha256"], "RP2 source")
    tasks = expected_tasks(manifest["bits"], manifest["zero_counts"][0], manifest["block"])
    sums, lower, counts = [F(0)] * 4, [F(0)] * 3, [0] * 10
    for batch in checked_batches(name, tasks):
        sums = [a + F(b) for a, b in zip(sums, batch["sums"])]
        lower = [a + F(b) for a, b in zip(lower, batch["lower_sums"])]
        counts = [a + b for a, b in zip(counts, batch["stats"])]
    require(len(tasks) == manifest["total_tasks"] == summary["done"], "RP2 completion")
    require(list(map(str, sums)) == summary["U6_EGF_parts"], "RP2 batch sums")
    require(list(map(str, lower)) == summary["lower_U_EGF"], "RP2 lower-order sums")
    require(counts[:4] == summary["compatible_counts"], "RP2 count assembly")
    require(counts[3] == int(manifest["expected_distinct_six_class_count"]), "RP2 independent census")
    d = meta["denominator"]
    c2, c3 = F(int(meta["S1_squared"]), 2), -F(int(meta["S1_S2"]), 2)
    u3, u4, u5 = [value / math.factorial(k) for k, value in zip((3, 4, 5), lower)]
    reconstructed = [u3 / d**3, u4 / d**4, (u5 + c2*u3) / d**5,
                     (sum(sums)/720 + c2*u4 + c3*u3) / d**6]
    require(reconstructed == list(map(F, row["e"][3:])), "RP2 physical-location corrections")
    return {"batches": len(tasks), "compatible_six_class_sets": counts[3], "L6": row["L"][6]}


def verify_fold_sixth():
    name = "fold-d5"
    row, meta = verify_case(name), verify_phase_input(name)
    verify_lower_orders(name)
    directory = DATA / "certificates" / name
    manifest, summary = [read_json(directory / f"{s}.json") for s in ("manifest", "summary")]
    proof = manifest["proof"]
    old_tasks = expected_tasks(14, 14657825, 2048, digits=8)
    lower, parts, task_rows, old_hashes = [F(0)] * 3, [F(0)] * 4, {}, {}
    for batch in checked_batches("fold-d5-fifth", old_tasks):
        lower = [a + F(b) for a, b in zip(lower, batch["lower_sums"])]
        parts = [a + F(b) for a, b in zip(parts, batch["sums"])]
        old_hashes[batch["id"]] = batch["sha256"]
        if batch["kind"] == 2:
            continue
        task = {k: batch[k] for k in ("id", "kind", "index", "begin", "end")}
        triples = batch["stats"][8] - (14657825 if task["kind"] == 0 and task["index"] == 0 else 0)
        require(triples >= 0, "Negative triple count")
        task["triples"] = triples if task["kind"] == 0 else 0
        task_rows[task["id"]] = task
    require(list(map(str, lower)) == proof["lower_EGF"], "Fold lower-order sums")
    require(list(map(str, parts)) == proof["parts"], "Fold correction sums")
    require(digest([old_hashes[t["id"]] for t in old_tasks]) == proof["previous_shard_digest"], "Fold previous batch digest")
    tasks = [task_rows[t["id"]] for t in old_tasks if t["kind"] != 2]
    require(digest(tasks) == manifest["all_tasks_digest"], "Fold complete task digest")
    total, count, hashes = F(0), 0, {}
    for batch in checked_batches(name, tasks):
        total += F(batch["sum"])
        count += batch["stats"][3]
        hashes[batch["id"]] = batch["sha256"]
    require(len(tasks) == manifest["total_tasks"] == summary["done"], "Fold completion")
    require(digest([hashes[t["id"]] for t in tasks]) == summary["saved_shards_digest"], "Fold final batch digest")
    census = read_json(directory / "census.json")
    require(count == proof["expected_six"] == summary["candidates"] == int(census["compatible_six_class_sets"]), "Fold census")
    require(total == F(summary["six_class_sum"]) and parts[3] == 0, "Fold sixth sum")
    parts[3] = total
    d, c2 = meta["denominator"], F(int(meta["S1_squared"]), 2)
    require(lower[0] == 0 and lower[1]/(24*d**4) == F(row["e"][4]), "Fold fourth order")
    require(lower[2]/(120*d**5) == F(row["e"][5]), "Fold fifth order")
    require((sum(parts)/720 + c2*lower[1]/24)/d**6 == F(row["e"][6]), "Fold physical-location corrections")
    return {"fifth_order_batches": len(old_tasks), "sixth_order_batches": len(tasks), "L6": row["L"][6]}


def verify_chan_v1():
    row = verify_case("chan-v1-d5")
    result = read_json(DATA / row["certificate"])
    cert = result["certificate"]
    qs = [prime(c) for c in cert["certificates"]]
    require(len(set(qs)) == len(qs), "Chan repeated prime")
    require(len(result["reconstruction"]) == 2 * (row["degree"] + 1), "Chan coefficient coverage")
    seen = set()
    for entry in result["reconstruction"]:
        k, observable = entry["order"], entry["observable"]
        require((k, observable) not in seen, "Repeated Chan coefficient")
        seen.add((k, observable))
        d = 2**38 * 15**k
        bound = d * math.comb(2308, k)
        integer, modulus = crt(list(map(int, entry["scaled_residues"])), qs)
        require(modulus > bound and 0 <= integer <= bound, "Chan reconstruction bound")
        require(integer == int(entry["numerator"]) and d == int(entry["denominator"]), "Chan CRT")
        require(bound == int(entry["upper_bound"]), "Chan height accounting")
        require(F(integer, d) == F(row["a" if observable == "A" else "e"][k]), "Chan raw coefficient")
    require(seen == {(k, o) for k in range(8) for o in ("A", "B")}, "Chan missing coefficient")
    return {"orders": 7, "primes": len(qs), "coefficients_reconstructed": len(seen)}


def verify_chan_four_round():
    row = verify_case("chan-four-round-d5")
    directory = DATA / "certificates" / row["case"]
    partial = read_json(directory / "new_partial_endpoints.json")
    functions = read_json(directory / "endpoint_functions.json")
    growth = read_json(directory / "new_growth_mode5_limit0.json")
    check = read_json(directory / "verification.json")
    qs = [prime(r["certificate"]) for r in partial["residues"]]
    require(len(qs) == len(set(qs)) == 2, "Partial reconstruction primes")
    for key, destination in (("e", "e_partial"), ("a", "a")):
        for k, coefficient in enumerate(partial[destination]):
            d = 2**38 * 15**k
            bound = d * math.comb(2576, k)
            integer, modulus = crt([r[key][k]*d % q for r, q in zip(partial["residues"], qs)], qs)
            require(modulus > bound and 0 <= integer <= bound, "Four-round height bound")
            require(F(integer, d) == F(coefficient), "Four-round partial reconstruction")
    for proof in functions["proof"]:
        k = proof["order"]
        require(int(proof["clearing_denominator"]) == 2**30*15**k, "Endpoint denominator")
        require(int(proof["integer_upper"]) == 2**30*15**k*math.comb(926, k), "Endpoint height")
        require(int(proof["reconstruction_prime"]) > int(proof["integer_upper"]), "Endpoint reconstruction modulus")
    require(growth["complete"] and growth["growth_five_computed"], "Four-round growth completion")
    require(growth["C_groups_used"] == growth["C_groups_all"] == 2308, "Absorbed response coverage")
    e = list(map(F, partial["e_partial"]))
    e[5] += F(int(growth["G4_h1"]), 15**4*functions["h1_denominator"])
    e[6] += F(int(growth["G4_h2"]), 15**4*functions["h2_denominator"])
    e[6] += F(int(growth["G5_h1"]), 15**5*functions["h1_denominator"])
    require(e == list(map(F, row["e"])) and row["a"] == partial["a"], "Complete growth assembly")
    q = prime(check["third_prime_certificate"])
    require(q not in qs and [fraction_mod(v, q) for v in e] == check["third_prime_raw_e_residues"], "Held-out growth prime")
    max_h = 0
    for name in ("h1", "h2"):
        lines = (DATA / "inputs" / row["case"] / f"{name}.txt").read_text().splitlines()
        require(int(lines[0]) == functions[name + "_entries"] == len(lines)-1, "Response support count")
        max_h = max(max_h, max(int(line.split()[1]) for line in lines[1:]))
    model = read_json(DATA / "inputs" / row["case"] / "growth.json")
    require(474 + 926 + len(model["channels"]) == 2576, "Four-round location count")
    bound = 120 * max_h * (15*len(model["channels"]))**5
    require(bound == int(check["signed_accumulator_absolute_bound"]) < 2**127, "Signed accumulator bound")
    require(len(model["channels"])**2*15**5 == int(check["pair_uint64_bound"]) < 2**64, "Pair integer bound")
    old = read_json(directory / "old_series6.json")
    control = load_case("chan-v1-d5")
    require(old["e"] == control["e"][:7] and old["a"] == control["a"][:2], "Old-source regression")
    zx = read_json(directory / "zx_old_fault_summary.json")
    require(zx["complete"] and zx["old_fault_pattern_acceptance"] == 0 and zx["old_fault_pattern_accepted_error"] == 0,
            "Saved ZX rejection check")
    return {"orders": 6, "partial_reconstruction_primes": 2, "held_out_primes": 1, "growth_coverage": "complete"}


def verify_comparisons():
    rp2 = read_json(DATA / "comparisons/rp2.json")
    for d in (3, 5):
        row = rp2[f"d{d}"]
        path = DATA / "comparisons" / row["source"]
        require(sha256(path) == row["sha256"], "RP2 raw-count source hash")
        records = [r for r in csv.reader(path.read_text().splitlines(), skipinitialspace=True)
                   if r and r[0].strip().isdigit()]
        shots, errors, discards = [sum(int(r[k]) for r in records) for k in range(3)]
        require((shots, errors, discards) == (row["shots"], row["errors"], row["discards"]), "RP2 count aggregation")
        require(errors/(shots-discards) == row["LER"], "RP2 conditional sampling rate")
    fold = read_json(DATA / "comparisons/fold.json")
    result = {}
    for key in ("proxy", "native"):
        for row in fold[key]:
            require(math.isclose(row["B"]/row["A"], row["PL"], rel_tol=1e-12), "Fold B/A normalisation")
            if key == "proxy":
                require(row["B"] == row["errors"]/row["shots"] and row["A"] == row["accepted"]/row["shots"], "Fold sampled counts")
        result[key + "_B_inside_published_bars"] = sum(r["paper_lower"] <= r["B"] <= r["paper_upper"] for r in fold[key])
    return {"rp2_raw_counts": "matched", **result}


def verify_chan_zx_records():
    report = {}
    for name in ("chan-v1-d5", "chan-four-round-d5"):
        directory = DATA / "certificates" / name / "zx"
        totals = {"reference": 0.0, "good": 0.0, "bad": 0.0}
        for case in totals:
            for record in range(16):
                row = read_json(directory / f"d5_zx_record{record:02d}_{case}.json")
                require(row["complete"] and row["case"] == case and row["record_index"] == record, "ZX record coverage")
                for value in row["methods"].values():
                    real, imag = value["amplitude"]
                    require(abs(real*real+imag*imag-value["probability"]) < 1e-12, "ZX amplitude/probability")
                values = [v["probability"] for v in row["methods"].values()]
                require(max(values)-min(values) < 1e-12, "ZX reduction routes disagree")
                totals[case] += values[0]
        expected = {"reference": 1.0, "good": 0.0, "bad": .25 if name == "chan-v1-d5" else 0.0}
        require(all(abs(totals[k]-v) < 1e-12 for k, v in expected.items()), "ZX totals")
        report[name] = {"records": 48, "totals": totals, "arithmetic": "complex floating scalars, tolerance 1e-12"}
    return report
