"""Build the detector-web guide: one A3 page per declared detector.

Run from pauli_semiwebs_check:
    python -B -m check_basis.guide.build [--circuit 3|5] [--no-pdf] [--paper DIR]
Writes detector_webs/{detector_webs.tex, pages.tex, diagrams/*.tikz} and two standalone
figures in detector_webs/figures: the paper's Appendix J figure and the repository README one. Unless --no-pdf, it compiles
detector_webs/detector_webs.pdf with pdflatex (three passes) and writes
detector_webs/index.tsv with the page of each detector. The drawing styles are read from
../pauli_semiwebs/drawings through TEXINPUTS. With --paper, the guide is also copied into
a paper source tree and compiled there.
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

from src.render import angle_tex
from .. import HERE, SEMIWEBS
from .layout import layout
from .render import signed, tikz
from .select import detector_webs

ROOT = HERE
OUT = HERE / 'detector_webs'

# Stage bands: (label, first source line, last source line).
BANDS = {
    3: [('Injection', 1, 130), ('Double check', 131, 191), ('Logical readout', 192, 195),
        ('Code readout', 196, 10 ** 6)],
    5: [('Injection', 1, 137), ('$d=3$ double check', 138, 186), ('Grow to $d=5$', 187, 211),
        ('Round 1', 212, 268), ('Corrections', 269, 277), ('Round 2', 278, 331),
        ('Round 3', 332, 385), ('$d=5$ double check', 386, 469), ('Code readout', 470, 490),
        ('Logical readout', 491, 10 ** 6)],
}
GATE_TEX = {'T': 'T', 'T_DAG': r'T^\dagger'}
# Names in the companion code of the paper. Both are byte-identical to the files analysed.
PAPER_SOURCE = {3: 'data/inputs/clifft_d3_p001.stim', 5: 'data/inputs/soft_cultivation_d5_p0005.stim'}
# Where the guide finds its styles and diagrams: in this repository (styles from
# ../pauli_semiwebs/drawings through TEXINPUTS), or in the paper's source tree.
PATHS = {'repo': dict(preamble='drawings/preamble.tex', legend='drawings/legend.tex',
                      reading='drawings/examples/reading_legend.tikz',
                      diagrams='detector_webs/diagrams', pages='detector_webs/pages.tex'),
         'paper': dict(preamble='semiweb_preamble.tex', legend='semiweb_caption_legend.tex',
                       reading='figures/semiwebs/reading_legend.tikz',
                       diagrams='detector_webs/diagrams', pages='detector_webs/pages.tex')}


def bands_for(key, parsed, lay):
    out = []
    for label, lo, hi in BANDS[key]:
        xs = [x for line, (a, b) in lay['line_x'].items() if lo <= line <= hi for x in (a, b)]
        if xs:
            out.append((label, min(xs), max(xs)))
    return out


def record_tex(parsed, r):
    rec = parsed['records'][r]
    if rec['kind'] == 'MPP':
        body = ''.join(f'{t[0]}_{{{t[1:]}}}' for t in rec['target'].split('*'))
        return rf'$r_{{{r}}}$: MPP ${body}$, line {rec["line"]}'
    return rf'$r_{{{r}}}$: {rec["kind"]} $q_{{{rec["target"]}}}$, line {rec["line"]}'


def parity_tex(records):
    return r'\oplus '.join(f'r_{{{r}}}' for r in records)


def window_for(g, lay, web, key):
    """First and last layout column shown: all of d=3, the web's columns plus one at d=5."""
    last = lay['columns'] + 1
    if key == 3:
        return (1, last)
    cs = [lay['node_column'][e[k]] for j, e in enumerate(g['edges'])
          if (web['vector'] >> (2 * j)) & 3 for k in ('u', 'v')]
    return (max(min(cs) - 1, 1), min(max(cs) + 1, last))


def page(key, parsed, g, web, name):
    recs = web['records']
    count = len(web['defects'])
    lines = [rf"\pdfbookmark[1]{{d={key}, detector {web['index']}}}{{{name}}}\label{{{name}}}",
             rf"{{\Large $d={key}$: detector {web['index']}}}\qquad"
             rf"{{\large ${parity_tex(recs)}={web['proxy_value']}$}}\par"]
    if count == 0:
        lines.append(r'\textbf{Phase-independent.} Closed Pauli web with no defects, so the check holds '
                     r'for any angle at every $T$ spider.')
    else:
        lines.append(rf'\textbf{{Phase-dependent.}} Closed semiweb with {count} defects, every one at a $T$ or '
                     r'$T^\dagger$ spider. No closed Pauli web without defects has these records. With every '
                     r'$T$ replaced by $S$ (the $S$ proxy), the same labels are a Pauli web.')
    lines += [r'\begin{center}', rf"\semiwebinput{{{PATHS['repo']['diagrams']}/{name}.tikz}}", r'\end{center}',
              r'\vspace{-1mm}']
    exponent = '+'.join(f'r_{{{r}}}' for r in recs)
    lam0 = web['scalar_pi_over_4']
    lam0_tex = {0: '', 4: '-'}.get(lam0, None)
    assert lam0_tex is not None, 'unexpected scalar'
    lhs = r'D_{\boldsymbol r}=\lambda(\boldsymbol r)\,D_{\boldsymbol r' + (r',\boldsymbol\delta}' if count else '}')
    lines.append(rf'${lhs}$, \quad $\lambda(\boldsymbol r)={lam0_tex}(-1)^{{{exponent}}}$. \qquad '
                 + ('So $D_{\\boldsymbol r}=0$ unless $' + parity_tex(recs) + f"={web['proxy_value']}$."
                    if not count else
                    r'$D_{\boldsymbol r,\boldsymbol\delta}$ has the marked phases changed, so this identity alone '
                    r'does not fix the parity.'))
    lines.append(r'\par\smallskip\begin{minipage}{.98\linewidth}\small Records: '
                 + r';\quad '.join(record_tex(parsed, r) for r in recs) + r'.\end{minipage}')
    if count:
        pos_order = sorted(web['defects'], key=lambda k: (lay_cache[key]['pos'][k][0], -lay_cache[key]['pos'][k][1]))
        items = []
        for i, k in enumerate(pos_order, 1):
            n = g['nodes'][k]
            items.append(rf"$\ast_{{{i}}}$: $\delta={angle_tex(web['defects'][k])}$ at "
                         rf"${GATE_TEX[n['gate']]}$ on $q_{{{n['qubit']}}}$, line {n['line']}")
        lines.append(r'\par\smallskip\begin{minipage}{.98\linewidth}\small Phase changes: '
                     + r';\quad '.join(items) + r'.\end{minipage}')
    lines += [r'\vfill', rf'{{\footnotesize\texttt{{\detokenize{{{name}.tikz}}}}, declared at line {web["line"]} of '
              rf'\texttt{{\detokenize{{{PAPER_SOURCE[key]}}}}}}}', r'\newpage', '']
    return '\n'.join(lines)


lay_cache = {}


def short_record(parsed, r):
    rec = parsed['records'][r]
    if rec['kind'] == 'MPP':
        return 'MPP $' + ''.join(f'{t[0]}_{{{t[1:]}}}' for t in rec['target'].split('*')) + '$'
    return f"{rec['kind']} $q_{{{rec['target']}}}$"


def detector_table(key, parsed, dets):
    rows = []
    for w in dets:
        kind = 'Pauli web' if not w['defects'] else rf"\textbf{{semiweb, {len(w['defects'])} $T$ defects}}"
        recs = '; '.join(f'$r_{{{r}}}$ ' + short_record(parsed, r) + f", l.\\,{parsed['records'][r]['line']}"
                         for r in w['records'])
        name = f"d{key}_det{w['index']:03d}"
        rows.append(rf"{w['index']} & ${parity_tex(w['records'])}$ & {recs} & {kind} & "
                    rf"\hyperref[{name}]{{\pageref*{{{name}}}}}\\")
    return rows


# Standalone d=3 figures: (file, detector, first source line or None for the web's own columns, sy).
FIGURES = [('d3_det014_check.tikz', 14, 131, .62),   # Appendix J of the paper: double check to the end
           ('d3_det007_web.tikz', 7, None, 1.0)]     # README of the repository: the columns the web uses


def figures(r):
    g, p = r['graph'], r['parsed']
    lay = layout(p, g, barriers=[lo for _, lo, _ in BANDS[3][1:]])
    data = {q for op in p['ops'] if op['op'] == 'MPP' and op['factors'][0][0] == 'Y' for _, q in op['factors']}
    paths = []
    for name, index, first, sy in FIGURES:
        web = next(d for d in r['detectors'] if d['index'] == index)
        if first is None:
            window = window_for(g, lay, web, 5)            # the web's columns plus one, as on the d=5 pages
        else:
            window = (min(lay['column_of_op'][i] for i, op in enumerate(p['ops']) if op['line'] >= first),
                      lay['columns'] + 1)
        text = tikz(g, lay, web, window=window, sx=.62, sy=sy, bands=bands_for(3, p, lay),
                    data=data, title_records=set(web['records']))
        path = OUT / 'figures' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths.append(path)
    return paths


def write(keys, compile_pdf):
    (OUT / 'diagrams').mkdir(parents=True, exist_ok=True)
    results = {}
    for key in keys:
        r = detector_webs(key)
        lay_cache[key] = layout(r['parsed'], r['graph'], barriers=[lo for _, lo, _ in BANDS[key][1:]])
        results[key] = r


    pages, index = [], []
    for key in keys:
        r = results[key]
        parsed, g = r['parsed'], r['graph']
        lay = lay_cache[key]
        data = {q for op in parsed['ops'] if op['op'] == 'MPP' and op['factors'][0][0] == 'Y' for _, q in op['factors']}
        bands = bands_for(key, parsed, lay)
        for web in r['detectors']:
            name = f"d{key}_det{web['index']:03d}"
            window = window_for(g, lay, web, key)
            width = lay['right'][window[1]] - lay['left'][window[0]]
            sx = min(.62, 36.0 / (width + 1.2))
            sy = 1.3 if key == 3 else .5
            text = tikz(g, lay, web, window=window, sx=sx, sy=sy, bands=bands, data=data,
                        title_records=set(web['records']))
            (OUT / 'diagrams' / f'{name}.tikz').write_text(text)
            pages.append(page(key, parsed, g, web, name))
            assert web['scalar_pi_over_4'] == 0, 'the guide text states lambda_0 = 1 for every web'
            index.append((name, f"{key}\t{web['index']}\t{{page}}\t{' '.join(map(str, web['records']))}\t"
                                f"{int(not web['defects'])}\t{len(web['defects'])}\t{web['labelled_edges']}\t{name}.tikz"))
    (OUT / 'pages.tex').write_text('\n'.join(pages))
    (OUT / 'detector_webs.tex').write_text(guide_tex(keys, results))
    if 3 in results:
        figures(results[3])
    if not compile_pdf:
        return
    env = dict(os.environ, TEXINPUTS=f'.:{SEMIWEBS}:')
    for _ in range(3):
        subprocess.run(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error',
                        '-output-directory', str(OUT.relative_to(ROOT)),
                        str((OUT / 'detector_webs.tex').relative_to(ROOT))],
                       cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    aux = (OUT / 'detector_webs.aux').read_text()
    for ext in ('aux', 'log', 'out'):
        (OUT / f'detector_webs.{ext}').unlink(missing_ok=True)
    page_of = dict(re.findall(r'\\newlabel\{(d\d_det\d+)\}\{\{[^}]*\}\{(\d+)\}', aux))
    header = 'circuit\tdetector\tpage\trecords\tphase_independent\tdefects\tlabelled_edges\tfile'
    (OUT / 'index.tsv').write_text('\n'.join([header] + [row.format(page=page_of[name]) for name, row in index]) + '\n')
    print(OUT / 'detector_webs.pdf')


def guide_tex(keys, results):
    from .text import guide  # the prose lives in one place
    return guide(keys, results, detector_table)


def export_paper(target):
    """Copy the guide into a paper source tree as detector_webs/, next to semiwebs/,
    and compile detector_webs.pdf in its root, as for pauli_semiwebs.pdf."""
    target = Path(target)
    out = target / 'detector_webs'
    (out / 'diagrams').mkdir(parents=True, exist_ok=True)
    for f in sorted((OUT / 'diagrams').glob('*.tikz')):
        shutil.copy2(f, out / 'diagrams' / f.name)
    figure = target / 'figures' / 'detector_webs'
    figure.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / 'figures' / 'd3_det014_check.tikz', figure / 'd3_det014_check.tikz')
    a, b = PATHS['repo'], PATHS['paper']

    def swap(text):
        for k in a:
            text = text.replace(a[k], b[k])
        return text
    (out / 'pages.tex').write_text(swap((OUT / 'pages.tex').read_text()))
    (out / 'detector_webs.tex').write_text(swap((OUT / 'detector_webs.tex').read_text()))
    shutil.copy2(OUT / 'index.tsv', out / 'index.tsv')
    for _ in range(3):
        subprocess.run(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', 'detector_webs/detector_webs.tex'],
                       cwd=target, check=True, stdout=subprocess.DEVNULL)
    for ext in ('aux', 'log', 'out'):
        (target / f'detector_webs.{ext}').unlink(missing_ok=True)
    print(target / 'detector_webs.pdf')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--circuit', choices=('3', '5'), action='append')
    p.add_argument('--no-pdf', action='store_true')
    p.add_argument('--paper', type=Path, help='also export into this paper source tree')
    args = p.parse_args()
    write([int(k) for k in (args.circuit or ['3', '5'])], not args.no_pdf)
    if args.paper:
        export_paper(args.paper)


if __name__ == '__main__':
    main()
