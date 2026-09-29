"""TikZ rendering with named defect markers and exact phase metadata."""

import json
from .layout import diagram_layout

def angle_tex(value):
    return {0: '0', 1: r'\frac{\pi}{4}', -1: r'-\frac{\pi}{4}',
            2: r'\frac{\pi}{2}', -2: r'-\frac{\pi}{2}',
            3: r'\frac{3\pi}{4}', -3: r'-\frac{3\pi}{4}', 4: r'\pi'}[value]


def scalar_tex(exponent):
    return {0: '1', 1: r'e^{i\pi/4}', 2: 'i', 3: r'e^{3i\pi/4}',
            4: '-1', 5: r'e^{5i\pi/4}', 6: '-i', 7: r'e^{-i\pi/4}'}[exponent % 8]


def draw(stage, graph, record, path):
    layout = diagram_layout(stage, graph)
    nodes, labels = layout['nodes'], record['edge_labels']
    change_ids = {node: i + 1 for i, node in enumerate(record['defects'])}
    # Both stages have the same horizontal physical width; the d=5 page uses
    # the full A3 height so its 38 wires and phase labels remain legible.
    sx = 33.0 / layout['last']
    sy = .58 if stage['distance'] == 5 else .73
    lines = ['% Semiweb: ' + record['id'],
             '% All phase metadata use units of pi/4; unchanged nodes have delta=0.',
             rf'\begin{{tikzpicture}}[x={sx:.5f}cm,y={sy}cm]',
             r'\tikzset{Z dot/.append style={minimum size=2.6mm},X dot/.append style={minimum size=2.6mm},',
             r'Z phase dot/.append style={shape=circle,rounded corners=0pt,scale=1,font={\fontsize{7}{8}\selectfont},minimum size=6.5mm,text width=4.7mm,align=center,inner sep=.25mm,outer sep=0pt}}']
    if layout['layers']:
        lines.append(r'\begin{pgfonlayer}{background}')
        for i, layer in enumerate(layout['layers']):
            lo, hi = layer['left'], layer['right']
            shade = 3 if i % 2 == 0 else 1
            lines.append(rf'\fill[black!{shade}] ({lo},.50) rectangle ({hi},{-len(stage["wires"])+.50});')
            lines.append(rf'\draw[black!35,line width=.25pt] ({lo},.58) -- ({lo},.82) -- ({hi},.82) -- ({hi},.58);')
        lines.append(r'\end{pgfonlayer}')
    lines.append(r'\begin{pgfonlayer}{nodelayer}')
    for key, node in nodes.items():
        style = 'none' if node['kind'] == 'boundary' else (
            node['kind'] + ' phase dot' if node['phase'] else node['kind'] + ' dot')
        text = '$' + angle_tex(node['phase']) + '$' if node['phase'] else ''
        lines.append(rf'\node [style={style}] ({key}) at ({node["x"]},{node["y"]}) {{{text}}};')
        if key in change_ids:
            delta = record['defects'][key]
            metadata = dict(marker=change_ids[key], node=key, qubit=node['qubit'],
                            role=node['role'], phase_pi_over_4=node['phase'],
                            delta_pi_over_4=delta,
                            changed_phase_pi_over_4=node['phase'] + delta)
            lines.append('% defect: ' + json.dumps(metadata, separators=(',', ':')))
            dx = format(layout['star_dx'], '.12g')
            lines.append(rf'\node [style=none,text=violet!80!black,font=\tiny,anchor=west] (defect-{key}) at ($({key}.center)+({dx},0.3)$) {{$\ast_{{{change_ids[key]}}}$}};')
    incoming, outgoing = record['input'], record['output']
    last = layout['last']
    input_x = '-.25' if stage['distance'] == 3 else str(layout['input_x'])
    for i, q in enumerate(stage['wires']):
        label = rf'$q_{{{q}}}$'
        if q in incoming:
            label += rf' $\mathbf{{{incoming[q]}}}$'
        color = 'black' if q in stage['data'] else 'black!50'
        lines.append(rf'\node [style=none,anchor=east,text={color},font=\scriptsize] at ({input_x},{-i}) {{{label}}};')
        label = f'${outgoing.get(q, "I")}$' if q in stage['data'] else r'$\langle+|$'
        lines.append(rf'\node [style=none,anchor=west,text={color},font=\scriptsize] at ({layout["output_x"]},{-i}) {{{label}}};')
    for x, label in layout['headers']:
        lines.append(rf'\node [style=none,font=\small] at ({x},{layout["header_y"]}) {{{label}}};')
    for layer in layout['layers']:
        mid = (layer['left'] + layer['right']) / 2
        lines.append(rf'\node [style=none,font=\scriptsize,text=black!65] at ({mid},1.30) {{{layer["label"]}}};')
    lines += [r'\end{pgfonlayer}', r'\begin{pgfonlayer}{edgelayer}']
    edges = sorted(enumerate(graph['edges']), key=lambda ie: nodes[ie[1]['u']]['x'] == nodes[ie[1]['v']]['x'])
    for j, edge in edges:
        style = '' if labels[j] == 'I' else f' [style={labels[j]} Web]'
        lines.append(rf'\draw{style} ({edge["u"]}.center) to ({edge["v"]}.center);')
    lines += [r'\end{pgfonlayer}', r'\end{tikzpicture}']
    path.write_text('\n'.join(lines) + '\n')

