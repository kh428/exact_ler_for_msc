"""Drawing coordinates only: separate overlapping CNOT spans within a layer."""

from collections import defaultdict


def diagram_layout(stage, graph):
    original = graph['nodes']
    nodes = {key: dict(node) for key, node in original.items()}
    center = next(i for i, g in enumerate(stage['groups']) if g['name'] == 'MX')
    last = len(stage['groups']) - 1
    if stage['distance'] == 3:
        headers = [(1, 'Entry'), ((2 + center - 1) / 2, 'Fold'),
                   (center + .5, 'Measure / reset'),
                   ((center + 2 + last - 2) / 2, 'Unfold'), (last - 1, 'Exit')]
        return dict(nodes=nodes, last=last, headers=headers, layers=[],
                    header_y=1.25, star_dx=.13, input_x=-.25, output_x=last+.22)

    rows = {q: i for i, q in enumerate(stage['wires'])}
    cursor = 0.0
    group_x, layers = {}, []
    counts = dict(F=0, U=0)
    for gi, group in enumerate(stage['groups']):
        if group['name'] != 'CX':
            group_x[gi] = cursor
            for key, node in nodes.items():
                if original[key]['x'] == gi:
                    node['x'] = cursor
            cursor += 1.35
            continue

        support = group['support']
        # Gates within a source group are disjoint. Separating their drawings
        # cannot reorder two operations that share a physical wire.
        assert len(support) == len(set(support))
        pairs = list(zip(support[::2], support[1::2]))
        spans = sorted((min(rows[c], rows[t]), max(rows[c], rows[t]), c, t)
                       for c, t in pairs)
        lanes = []
        assignment = {}
        for lo, hi, c, t in spans:
            for lane, occupied in enumerate(lanes):
                if all(hi < a or b < lo for a, b in occupied):
                    occupied.append((lo, hi))
                    break
            else:
                lane = len(lanes)
                lanes.append([(lo, hi)])
            assignment[c] = assignment[t] = lane
        if gi > center:
            assignment = {q: len(lanes) - 1 - lane for q, lane in assignment.items()}
        for key, node in nodes.items():
            if original[key]['x'] == gi:
                node['x'] = cursor + assignment[node['qubit']]
        prefix = 'F' if gi < center else 'U'
        counts[prefix] += 1
        end = cursor + len(lanes) - 1
        layers.append(dict(group=gi, label=f'{prefix}{counts[prefix]}',
                           left=cursor-.35, right=end+.35,
                           columns=len(lanes), cnots=len(pairs)))
        group_x[gi] = (cursor + end) / 2
        cursor = end + 1.50

    last = group_x[len(stage['groups']) - 1]
    sx = 33.0 / last
    headers = [(group_x[1], 'Entry'),
               ((layers[0]['left'] + layers[6]['right']) / 2, 'Fold'),
               ((group_x[center] + group_x[center+1]) / 2, 'Measure / reset'),
               ((layers[7]['left'] + layers[-1]['right']) / 2, 'Unfold'),
               (group_x[len(stage['groups']) - 2], 'Exit')]

    # Drawing invariants, independently of the interval-packing assignment.
    vertical = defaultdict(list)
    for edge in graph['edges']:
        a, b = (nodes[edge[k]] for k in ('u', 'v'))
        if a['qubit'] == b['qubit']:
            assert a['y'] == b['y'] and a['x'] < b['x']
        else:
            assert a['x'] == b['x']
            lo, hi = sorted((a['y'], b['y']))
            assert all(hi < old_lo or old_hi < lo for old_lo, old_hi in vertical[a['x']])
            vertical[a['x']].append((lo, hi))
    assert sum(len(v) for v in vertical.values()) == 74
    return dict(nodes=nodes, last=last, headers=headers, layers=layers,
                header_y=2.25, star_dx=.23/sx, input_x=-.40/sx,
                output_x=last+.35/sx)
