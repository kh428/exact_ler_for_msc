"""Fault response of the web checks, and the fault pairs they cannot see.

The web checks are the reduced basis of whole-circuit record equations from
closed, defect-free Pauli webs (check_basis.webs). They need not be single
declared detectors.

(a) Pauli determinism. Declare the web checks as Stim detectors. For every Clifford
    substitution of the T gates, Stim explains each physical error location's flips.
    A web check's response to a Pauli fault is fixed by the web's label at that
    location, so the location -> flip map must be identical for every substitution,
    and therefore also for the T circuit, by the same trigonometric-polynomial argument
    as in check_basis.oracle.
(b) Pairs invisible to the web checks. In the S proxy, declare the web checks and all
    declared detectors, keep the logical observable, and list pairs of single faults
    that flip the observable but no web check. Each must be caught by declared detectors
    outside the web span, since the proxy has fault distance three. These pairs are the
    only weight-two patterns that can be accepted as logical errors in the T circuit,
    because there every web check responds to Pauli faults deterministically.
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import stim

from . import circuit, graph, oracle, webs
from .circuit import substituted_text
from .oracle import CLIFFORD, proxy_phases


def with_detectors(text, tsites, phases, detector_sets, nrec, keep_observables):
    body = [l for l in substituted_text(text, tsites, phases, tagged=True).split('\n')
            if not l.strip().startswith('DETECTOR') and
            (keep_observables or not l.strip().startswith('OBSERVABLE_INCLUDE'))]
    body += ['DETECTOR ' + ' '.join(f'rec[{r - nrec}]' for r in recs) for recs in detector_sets]
    return stim.Circuit('\n'.join(body))


def location_key(loc):
    """Physical identity of an error: noise instruction, its targets and the Pauli.

    Substituted gates are tagged, so Stim numbers instructions identically in every
    substitution and the instruction index identifies the noise channel instance.
    """
    it = loc.instruction_targets
    return (loc.stack_frames[0].instruction_offset, it.gate, tuple(it.args), it.target_range_start,
            it.target_range_end, str(it.targets_in_range), str(loc.flipped_pauli_product),
            str(loc.flipped_measurement))


def location_map(stim_circuit):
    """Physical error location -> (flipped detector indices, observable flip)."""
    out = {}
    for err in stim_circuit.explain_detector_error_model_errors():
        dets = frozenset(t.dem_target.val for t in err.dem_error_terms if t.dem_target.is_relative_detector_id())
        obs = any(t.dem_target.is_logical_observable_id() for t in err.dem_error_terms)
        for loc in err.circuit_error_locations:
            key = location_key(loc)
            assert key not in out, 'one location explained twice'
            out[key] = (dets, obs)
    return out


def web_check_sets(parsed):
    """Record sets of the reduced whole-circuit web-check basis."""
    eqs = webs.web_checks(graph.build(parsed, oracle.discard_states(parsed)))['equations']
    nrec = len(parsed['records'])
    return [[i for i in range(nrec) if (v >> i) & 1] for v, _ in webs.reduced_basis(eqs, nrec)]


def pauli_determinism(parsed, web_sets, *, seed=11, random_count=20):
    rng = random.Random(seed)
    tsites, nrec = parsed['tsites'], len(parsed['records'])
    proxy = proxy_phases(tsites)
    trials = [proxy, [0] * len(tsites)] + [[rng.choice(CLIFFORD) for _ in tsites] for _ in range(random_count)]
    reference = location_map(with_detectors(parsed['text'], tsites, proxy, web_sets, nrec, False))
    identical = 0
    for phases in trials:
        m = location_map(with_detectors(parsed['text'], tsites, phases, web_sets, nrec, False))
        assert m.keys() == reference.keys(), 'fault locations differ between substitutions'
        identical += m == reference
    return dict(substitutions=len(trials), identical_maps=identical,
                fault_events_flipping_some_web_check=len(reference))


def proxy_readout_phases(parsed):
    """S proxy, with the noiseless logical readout measuring Y_L directly.

    The native-T readout projects onto the magic state with T gates around MPP Y_L
    (after the last single-qubit measurement). In the proxy the state is already a
    Y_L eigenstate, so those sites become I.
    """
    last = max(op['line'] for op in parsed['ops'] if op['op'] in ('M', 'MX'))
    return [0 if s['line'] > last else p for s, p in zip(parsed['tsites'], proxy_phases(parsed['tsites']))]


def invisible_pairs(parsed, web_sets):
    """Proxy fault pairs that flip the logical observable and no web check.

    A fault is one Pauli at one physical position. Two Paulis at the same position
    are mutually exclusive outcomes of one channel, so such pairs are excluded.
    """
    nrec, nweb = len(parsed['records']), len(web_sets)
    declared = [d['records'] for d in parsed['detectors']]
    loc = location_map(with_detectors(parsed['text'], parsed['tsites'], proxy_readout_phases(parsed),
                                      web_sets + declared, nrec, True))
    by_web = defaultdict(lambda: ([], []))
    for key, (dets, obs) in loc.items():
        web_part = frozenset(d for d in dets if d < nweb)
        by_web[web_part][int(obs)].append((key[:6], frozenset(d - nweb for d in dets if d >= nweb)))
    pairs, undetected, catchers = 0, 0, defaultdict(int)
    for zero, one in by_web.values():
        for p0, d0 in zero:
            for p1, d1 in one:
                if p0 == p1:
                    continue
                pairs += 1
                caught = d0 ^ d1
                undetected += not caught
                catchers[tuple(sorted(caught))] += 1
    singles = sum(len(one) for w, (_, one) in by_web.items() if not w)
    return dict(proxy_fault_events=len(loc), web_checks=nweb,
                single_faults_flipping_logical_but_no_web_check=singles,
                invisible_logical_pairs=pairs, undetected_by_all_detectors=undetected,
                distinct_catching_sets=len(catchers),
                detectors_used_to_catch=sorted({d for k in catchers for d in k}))


def analyse(key, *, random_count=20):
    began = time.process_time()
    parsed = circuit.load(key)
    web_sets = web_check_sets(parsed)
    nrec = len(parsed['records'])
    eqs = [(sum(1 << r for r in s), 0) for s in web_sets]
    outside = [d['index'] for d in parsed['detectors']
               if not oracle.contains(eqs, nrec, sum(1 << r for r in d['records']), 0)]
    a = pauli_determinism(parsed, web_sets, random_count=random_count)
    b = invisible_pairs(parsed, web_sets)
    b['catchers_are_all_outside_web_span'] = set(b['detectors_used_to_catch']) <= set(outside)
    return dict(web_checks=len(web_sets), detectors_outside_web_span=outside,
                pauli_determinism=a, invisible_pairs=b,
                cpu_seconds=round(time.process_time() - began, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--circuit', choices=('3', '5'), action='append')
    p.add_argument('--random', type=int, default=20)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    results = {}
    for key in args.circuit or ['3', '5']:
        results[key] = analyse(int(key), random_count=args.random)
        print(key, json.dumps(results[key]), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1) + '\n')
        print(args.output)


if __name__ == '__main__':
    main()
