"""The factory's ZX diagram, read from a TikZiT drawing.

Five wires run from left to right and are numbered 0 to 4 from the top. Wire 0 starts in |+>
and is the output. Wires 1 to 4 each start with a T state and end in an X measurement, drawn
as the effect of outcome 0; its record is r_1 to r_4. Eleven more T states sit on phase
gadgets: a phase spider joined, through an X spider called its hub, to the wires it acts on.
Phases are in units of pi/4, so a T state has phase 1 and its S-gate proxy has phase 2.
"""

import copy
import re

NODE = re.compile(r'\\node \[style=([^\]]+)\] \((\d+)\) at \(([-\d.]+), ([-\d.]+)\) \{(.*)\};')
EDGE = re.compile(r'\\draw \((\d+)\) to \((\d+)\);')
OVERLAY = re.compile(r'\\draw\[(red|green|blue), line width=2.5pt, opacity=0.4\] '
                     r'\((\d+)\) -- \(\$\(\d+\)!0.5!\((\d+)\)\$\);')
PHASES = {r'$\frac{\pi}{4}$': 1, r'$\frac{\pi}{2}$': 2}
COLOUR = {'red': 'X', 'green': 'Z', 'blue': 'Y'}


def read(path):
    """Graph of one drawing, the role of each node and any web drawn on it by hand."""
    text = path.read_text()
    nodes = {}
    for style, key, x, y, label in NODE.findall(text):
        if (style == 'Z phase dot') != (label in PHASES):
            raise ValueError(f'{path.name}: unexpected label at node {key}')
        kind = {'none': 'boundary', 'X dot': 'X'}.get(style, 'Z')
        nodes[key] = dict(kind=kind, phase=PHASES.get(label, 0), x=float(x), y=float(y))
    edges = [dict(u=u, v=v) for u, v in EDGE.findall(text)]
    degree = {k: 0 for k in nodes}
    for e in edges:
        degree[e['u']] += 1
        degree[e['v']] += 1
    left = min(n['x'] for n in nodes.values())
    rows = sorted({n['y'] for n in nodes.values() if n['x'] == left}, reverse=True)
    for k, n in nodes.items():
        n['wire'] = rows.index(n['y']) if n['kind'] != 'X' and n['y'] in rows else None
        if n['kind'] == 'boundary':
            n['role'] = 'output'
        elif n['kind'] == 'X':
            n['role'] = 'hub'
        elif n['phase']:
            n['role'] = 'T'
        elif degree[k] == 1:
            n['role'] = 'prepare' if n['x'] == left else 'measure'
        else:
            n['role'] = 'wire'
        if n['role'] == 'measure':
            n['record'] = n['wire'] - 1
    # T states are numbered from 1: first those on wires 1 to 4, then the gadgets from the left.
    order = sorted((k for k, n in nodes.items() if n['role'] == 'T'),
                   key=lambda k: (nodes[k]['wire'] is None, nodes[k]['wire'] or 0, nodes[k]['x']))
    for i, k in enumerate(order):
        nodes[k]['site'] = i
    halves = {}
    for colour, u, v in OVERLAY.findall(text):
        halves.setdefault(frozenset((u, v)), set()).add(colour)
    overlay = {}
    for pair, colours in halves.items():
        if len(colours) != 1:
            raise ValueError(f'{path.name}: two colours on one edge')
        overlay[pair] = COLOUR[colours.pop()]
    return dict(nodes=nodes, edges=edges, overlay=overlay, sites=order)


def with_phase(g, phase):
    """The same diagram with every T state's spider set to one phase."""
    out = copy.deepcopy(g)
    for k in out['sites']:
        out['nodes'][k]['phase'] = phase
    return out


def incidence(g):
    inc = {k: [] for k in g['nodes']}
    for j, e in enumerate(g['edges']):
        inc[e['u']].append(j)
        inc[e['v']].append(j)
    return inc


def supports(g):
    """The wires each T state acts on: the columns of the factory's 5 x 15 matrix."""
    near = {k: [] for k in g['nodes']}
    for e in g['edges']:
        near[e['u']].append(e['v'])
        near[e['v']].append(e['u'])
    out = []
    for k in g['sites']:
        n = g['nodes'][k]
        if n['wire'] is not None:
            out.append(frozenset([n['wire']]))
        else:
            (hub,) = near[k]
            if g['nodes'][hub]['role'] != 'hub':
                raise ValueError('A gadget phase spider must hang from a hub')
            out.append(frozenset(g['nodes'][w]['wire'] for w in near[hub] if w != k))
    return out
