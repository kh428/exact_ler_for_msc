"""TikZ drawings of the factory with a web, in the style of ../pauli_semiwebs/drawings.

Node positions are those of the layout drawing. Web edges use the X Web, Y Web and Z Web
styles. Each defect has a violet ring around its spider, a numbered star and a comment with
its phase data.
"""

import json

from src.render import angle_tex

SCALE = r'x=0.62cm,y=0.5cm'
# One bounding box for every drawing, so that a proxy web and its semiweb have the same size.
BOX = r'\useasboundingbox (0.25,-10.95) rectangle (26.00,0.50);'
STYLE = (r'\tikzset{Z dot/.append style={minimum size=2.1mm},X dot/.append style={minimum size=2.1mm},'
         '\n'
         r'Z phase dot/.append style={shape=circle,rounded corners=0pt,scale=1,'
         r'font={\fontsize{5.5}{6}\selectfont},minimum size=5.2mm,text width=4.4mm,align=center,'
         r'inner sep=.1mm,outer sep=0pt}}')


def tikz(g, vector, *, phase, heading, records=0, marks=None, output=None):
    """One drawing: the diagram at one T phase, a web, the records it reads and its defects."""
    nodes = g['nodes']
    marks = marks or {}
    number = {k: i for i, k in enumerate(sorted(marks, key=lambda k: nodes[k]['site']), 1)}
    lines = [f'% {heading}; phases in units of pi/4',
             rf'\begin{{tikzpicture}}[{SCALE}]', BOX, STYLE, r'\begin{pgfonlayer}{nodelayer}']
    for k, n in nodes.items():
        if n['role'] == 'T':
            style, text = 'Z phase dot', f'${angle_tex(phase)}$'
        else:
            style, text = {'boundary': 'none', 'X': 'X dot', 'Z': 'Z dot'}[n['kind']], ''
        lines.append(rf"\node[style={style}] (n{k}) at ({n['x'] + 0.0:.2f},{n['y'] + 0.0:.2f}) {{{text}}};")
        if k in marks:
            meta = dict(marker=number[k], node=f'n{k}', t_state=n['site'] + 1, phase_pi_over_4=phase,
                        delta_pi_over_4=marks[k], changed_phase_pi_over_4=phase + marks[k])
            lines.append('% defect: ' + json.dumps(meta, separators=(',', ':')))
            lines.append(r'\node[style=none,draw=violet!80!black,line width=.8pt,shape=circle,minimum size=7mm] '
                         rf'at (n{k}.center) {{}};')
            lines.append(r'\node[style=none,text=violet!80!black,font=\small,anchor=south west] '
                         rf'at ($(n{k}.center)+(.42,.52)$) {{$\ast_{{{number[k]}}}$}};')
        if 'record' in n:
            read = (records >> n['record']) & 1
            ink = r'text=blue!75!black,font=\small\boldmath' if read else r'text=black!55,font=\scriptsize'
            lines.append(rf'\node[style=none,anchor=west,{ink}] at ($(n{k}.center)+(.3,0)$) '
                         rf"{{$r_{{{n['record'] + 1}}}$}};")
        if n['role'] == 'output' and output:
            lines.append(r'\node[style=none,anchor=west,font=\small\boldmath] '
                         rf'at ($(n{k}.center)+(.3,0)$) {{${output}$}};')
    lines += [r'\end{pgfonlayer}', r'\begin{pgfonlayer}{edgelayer}']
    for j, e in enumerate(g['edges']):
        label = 'IXZY'[(vector >> (2 * j)) & 3]
        style = '' if label == 'I' else f'[style={label} Web]'
        lines.append(rf"\draw{style} (n{e['u']}.center) to (n{e['v']}.center);")
    lines += [r'\end{pgfonlayer}', r'\end{tikzpicture}']
    return '\n'.join(lines) + '\n'
