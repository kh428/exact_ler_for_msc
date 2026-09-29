"""TikZ for one detector web on the whole-circuit ZX graph, in the package's styles.

A page shows the columns between x0 and x1 of the layout. Wires that enter or
leave that window are cut at its edges. Every other connection lies inside one
operation, so it is either wholly inside the window or wholly outside it.
"""

import json

from src.render import angle_tex

DATA_TEXT, OTHER_TEXT = 'black', 'black!50'


def signed(phase):
    return phase if phase <= 4 else phase - 8


def tikz(g, lay, web, *, window, sx, sy, bands, data, title_records):
    pos, col = lay['pos'], lay['node_column']
    c0, c1 = window
    x0, x1 = lay['left'][c0], lay['right'][c1]
    labels = ['IXZY'[(web['vector'] >> (2 * j)) & 3] for j in range(len(g['edges']))]
    inside = {k for k in pos if c0 <= col[k] <= c1}
    order = sorted(web['defects'], key=lambda k: (pos[k][0], -pos[k][1]))
    marker = {k: i + 1 for i, k in enumerate(order)}
    nrows = len(lay['qubits'])

    def X(x):
        return f'{x - x0:.3f}'

    out = [f"% Detector {web['index']}: records {web['records']}; phases in units of pi/4",
           rf'\begin{{tikzpicture}}[x={sx:.4f}cm,y={sy:.4f}cm]',
           r'\tikzset{Z dot/.append style={minimum size=2.1mm},X dot/.append style={minimum size=2.1mm},',
           r'Z phase dot/.append style={shape=circle,rounded corners=0pt,scale=1,font={\fontsize{5.5}{6}\selectfont},'
           r'minimum size=5.2mm,text width=4.4mm,align=center,inner sep=.1mm,outer sep=0pt},'
           r'hadamard/.append style={inner sep=.45mm}}']

    out.append(r'\begin{pgfonlayer}{background}')
    for i, (label, lo, hi) in enumerate(bands):
        lo, hi = max(lo, x0), min(hi, x1)
        if hi <= lo:
            continue
        out.append(rf'\fill[black!{4 if i % 2 == 0 else 1.5}] ({X(lo)},.55) rectangle ({X(hi)},{-nrows + .45});')
        out.append(rf'\draw[black!35,line width=.25pt] ({X(lo)},.62) -- ({X(lo)},.85) -- ({X(hi)},.85) -- ({X(hi)},.62);')
        out.append(rf'\node[font=\scriptsize,text=black!70,anchor=south] at ({(float(X(lo)) + float(X(hi))) / 2:.3f},.9) {{{label}}};')
    out.append(r'\end{pgfonlayer}')

    out.append(r'\begin{pgfonlayer}{nodelayer}')
    for k in sorted(inside, key=lambda k: pos[k]):
        n = g['nodes'][k]
        x, y = pos[k]
        if n['kind'] == 'boundary':
            out.append(rf'\node[style=none] ({k}) at ({X(x)},{y}) {{}};')
            continue
        if n['kind'] == 'H':
            out.append(rf'\node[style=hadamard] ({k}) at ({X(x)},{y:.2f}) {{}};')
            continue
        phase = signed(n['phase'])
        style = f"{n['kind']} phase dot" if phase else f"{n['kind']} dot"
        text = f'${angle_tex(phase)}$' if phase else ''
        out.append(rf'\node[style={style}] ({k}) at ({X(x)},{y:.2f}) {{{text}}};')
        if 'record' in n:
            r = n['record']
            mine = r in title_records
            colour = 'blue!75!black' if mine else 'black!55'
            font = r'\small\boldmath' if mine else r'\scriptsize'
            place = 'anchor=west' if n['role'] == 'product' else 'anchor=south west'
            dy = 0 if n['role'] == 'product' else .12
            out.append(rf'\node[style=none,{place},text={colour},font={font}] at ($({k}.center)+(.12,{dy})$) {{$r_{{{r}}}$}};')
        if k in marker:
            delta = web['defects'][k]
            meta = dict(marker=marker[k], node=k, qubit=n['qubit'], line=n['line'], gate=n.get('gate'),
                        t_site=n.get('site'), phase_pi_over_4=phase, delta_pi_over_4=delta,
                        changed_phase_pi_over_4=signed((n['phase'] + delta) % 8))
            out.append('% defect: ' + json.dumps(meta, separators=(',', ':')))
            out.append(rf'\node[style=none,text=violet!80!black,font=\small,anchor=south west] '
                       rf'at ($({k}.center)+(.18,.22)$) {{$\ast_{{{marker[k]}}}$}};')
    for q in lay['qubits']:
        colour = DATA_TEXT if q in data else OTHER_TEXT
        y = -lay['row'][q]
        out.append(rf'\node[style=none,anchor=east,text={colour},font=\scriptsize] at (-0.25,{y}) {{$q_{{{q}}}$}};')
        out.append(rf'\node[style=none,anchor=west,text={colour},font=\scriptsize] at ({x1 - x0 + .25:.3f},{y}) {{$q_{{{q}}}$}};')
    out.append(r'\end{pgfonlayer}')

    out.append(r'\begin{pgfonlayer}{edgelayer}')
    drawn = []
    for j, e in enumerate(g['edges']):
        u, v = e['u'], e['v']
        (xu, yu), (xv, yv) = pos[u], pos[v]
        style = '' if labels[j] == 'I' else f'[style={labels[j]} Web]'
        if u in inside and v in inside:
            path = rf'({u}.center) to ({v}.center)'
        elif u in inside or v in inside:
            assert yu == yv, 'a connection leaves the window off a wire'
            a, b = (u, v) if u in inside else (v, u)
            edge = x0 if col[b] < c0 else x1
            path = rf'({a}.center) to ({X(edge)},{yu})'
        elif yu == yv and min(col[u], col[v]) < c0 and max(col[u], col[v]) > c1:
            path = rf'({X(x0)},{yu}) to ({X(x1)},{yu})'
        else:
            continue
        drawn.append((xu == xv, labels[j] != 'I', rf'\draw{style} {path};'))
    # Plain wires first, then web edges, vertical connections last, as in src/render.py.
    out += [d for _, _, d in sorted(drawn, key=lambda t: (t[0], t[1]))]
    out += [r'\end{pgfonlayer}', r'\end{tikzpicture}']
    return '\n'.join(out) + '\n'
