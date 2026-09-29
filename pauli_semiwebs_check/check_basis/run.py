"""Whole-circuit check basis from Pauli webs, with independent cross-checks.

Run from pauli_semiwebs_check:  python -B -m check_basis.run [--circuit 3|5] [--output PATH]
Nothing is written unless --output is given.
"""

import argparse
import json
import time
from pathlib import Path

import stim

from src.double_check_stage import boundary_constraints, graph_for, read_stage
from src.linear import kernel
from src.semiweb import constraints as stage_constraints
from . import SEMIWEBS, circuit, graph, oracle, webs


def stim_indexing(parsed):
    """Record count and detector record lists as Stim itself resolves them."""
    c = stim.Circuit(circuit.substituted_text(parsed['text'], parsed['tsites'],
                                              oracle.proxy_phases(parsed['tsites'])))
    dets, m = [], 0
    for inst in c.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            m += len(inst.target_groups())
        elif inst.name == 'DETECTOR':
            dets.append(sorted(m + t.value for t in inst.targets_copy()))
    return m, dets


def package_line_map(parsed):
    """Line in the package's own circuit file -> line in the analysed file, for the
    operations the two files share from the start."""
    name = {3: 'd3_native_t.txt', 5: 'd5_native_t.stim'}[parsed['distance']]
    own = circuit.parse_text((SEMIWEBS / 'circuits' / name).read_text())['ops']
    out = {}
    for a, b in zip(own, parsed['ops']):
        if {k: v for k, v in a.items() if k != 'line'} != {k: v for k, v in b.items() if k != 'line'}:
            break
        out[a['line']] = b['line']
    return out


def stage_equations_in_full_records(parsed):
    """The package's isolated-stage defect-free webs, re-expressed on full-circuit records."""
    stage = read_stage(parsed['distance'])
    line_map = package_line_map(parsed)
    sg = graph_for(stage)
    rows = stage_constraints(sg, require_web=True) + boundary_constraints(sg, {'input', 'output'})
    basis = kernel(rows, 2 * len(sg['edges']))
    inc = webs.incidence(sg)
    by_key = {(r['line'], int(r['target'])): r['index'] for r in parsed['records'] if r['kind'] == 'MX'}
    measure_nodes = [k for k, n in sg['nodes'].items() if n['role'] == 'measure']
    eqs = []
    for w in basis:
        support = 0
        for k in measure_nodes:
            n = sg['nodes'][k]
            line = line_map[stage['groups'][int(n['x'])]['source_lines'][0]]   # node x is its group index
            (j,) = inc[k]
            if (w >> (2 * j)) & 1:            # X on a Z-spider effect <+| is opposite colour
                support ^= 1 << by_key[(line, n['qubit'])]
        if support:
            eqs.append(support)
    return eqs


def describe(parsed, v, b):
    recs = [i for i in range(len(parsed['records'])) if (v >> i) & 1]
    return dict(records=recs, value=b,
                text=' + '.join(f"rec{i}:{parsed['records'][i]['kind']} {parsed['records'][i]['target']}"
                                f"@{parsed['records'][i]['line']}" for i in recs) + f' = {b}')


def analyse(distance, *, random_count, shots):
    began = time.process_time()
    parsed = circuit.load(distance)
    nrec = len(parsed['records'])

    # 1. Parser against Stim's own indexing.
    m, stim_dets = stim_indexing(parsed)
    assert m == nrec, (m, nrec)
    assert stim_dets == [d['records'] for d in parsed['detectors']], 'detector indexing differs from Stim'

    # 2. Mid-circuit discards and the full graph.
    discard = oracle.discard_states(parsed)
    g = graph.build(parsed, discard)
    web = webs.web_checks(g)
    web_eqs = web['equations']
    web_rank = webs.affine_rank(web_eqs, nrec)

    # 3. Independent oracle: phase-independent parities.
    space = oracle.deterministic_space(parsed, random_count=random_count, shots=shots)
    oracle_eqs = space['equations']
    union_rank = webs.affine_rank(web_eqs + oracle_eqs, nrec)
    sound = all(oracle.contains(oracle_eqs, nrec, v, b) for v, b in web_eqs)
    complete = union_rank == web_rank

    # 4. Declared detectors.
    web_space = dict(equations=web_eqs, records=nrec, reference=space['reference'])
    in_web = oracle.classify_detectors(parsed, web_space)
    in_oracle = oracle.classify_detectors(parsed, space)
    assert all(a['inside'] == b['inside'] for a, b in zip(in_web, in_oracle)), 'web and oracle disagree on a detector'
    outside = [d['index'] for d in in_web if not d['inside']]

    # 5. The package's isolated-stage webs lie in the full-circuit span.
    if distance in (3, 5):
        stage_eqs = stage_equations_in_full_records(parsed)
        stage_ok = all(webs.affine_rank(web_eqs + [(v, 0)], nrec) == web_rank for v in stage_eqs)
    else:
        stage_eqs, stage_ok = [], None

    # 6. S-proxy control: same graph with T -> S phases; compare with the proxy's full space.
    proxy = oracle.proxy_phases(parsed['tsites'])
    pg = graph.build(parsed, discard, phase_override=proxy)
    pweb = webs.web_checks(pg)
    pspace = oracle.deterministic_space(parsed, phase_sets=[('proxy', proxy)], shots=4 * shots)
    p_rank = webs.affine_rank(pweb['equations'], nrec)
    p_union = webs.affine_rank(pweb['equations'] + pspace['equations'], nrec)
    proxy_detectors_in_web = sum(d['inside'] for d in oracle.classify_detectors(
        parsed, dict(equations=pweb['equations'], records=nrec, reference=pspace['reference'])))

    basis = webs.reduced_basis(web_eqs, nrec)
    return dict(
        distance=distance, source=parsed['source'], source_sha256=parsed['source_sha256'],
        records=nrec, declared_detectors=len(parsed['detectors']), t_sites=len(parsed['tsites']),
        discarded_known_states={f"op{i}:q{parsed['ops'][i]['q']}@line{parsed['ops'][i]['line']}": f'{s}{v:+d}'
                                for i, (s, v) in discard.items()},
        graph=dict(nodes=web['nodes'], edges=web['edges'], constraint_rows=web['constraint_rows'],
                   closed_defect_free_webs=web['web_dimension'], record_free_webs=web['record_free_webs']),
        web_check_rank=web_rank,
        oracle=dict(rank=space['dimension'], assignments=space['assignments'],
                    shots_per_assignment=space['shots_per_assignment']),
        web_checks_sound=sound, web_checks_complete=complete,
        detectors_in_web_span=len(parsed['detectors']) - len(outside),
        detectors_outside=[dict(index=i, **describe(parsed, sum(1 << r for r in parsed['detectors'][i]['records']), 0))
                           for i in outside],
        stage_webs=dict(equations=len(stage_eqs), all_in_full_span=stage_ok),
        proxy=dict(web_check_rank=p_rank, deterministic_rank=pspace['dimension'],
                   equal=(p_rank == p_union == pspace['dimension']),
                   declared_detectors_in_web_span=proxy_detectors_in_web),
        check_basis=[describe(parsed, v, b) for v, b in basis],
        cpu_seconds=round(time.process_time() - began, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--circuit', choices=('3', '5'), action='append',
                   help='d=3 or d=5; default both')
    p.add_argument('--random', type=int, default=300)
    p.add_argument('--shots', type=int, default=64)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    keys = [int(k) for k in (args.circuit or ['3', '5'])]
    results = [analyse(k, random_count=args.random, shots=args.shots) for k in keys]
    for r in results:
        summary = {k: r[k] for k in ('distance', 'records', 'declared_detectors', 'web_check_rank',
                                     'web_checks_sound', 'web_checks_complete', 'detectors_in_web_span',
                                     'stage_webs', 'proxy', 'cpu_seconds')}
        summary['oracle_rank'] = r['oracle']['rank']
        summary['detectors_outside'] = [d['index'] for d in r['detectors_outside']]
        print(json.dumps(summary))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1) + '\n')
        print(args.output)


if __name__ == '__main__':
    main()
