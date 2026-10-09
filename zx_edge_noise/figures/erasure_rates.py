"""P_L against p with a Pauli and an ignored erasure on every ZX edge, e = 0, p, 5p, 10p, at d=3 and d=5.

No new calculation: each curve is the saved Pauli-only series of ../results evaluated at the shifted error rate
p' = p + 3e/4 - p e (zxedge.series.ignored_erasure; tests/test_fast.py checks this against every saved erasure
series). A curve stops where its last term reaches 1% of the sum, beyond which the truncated series is not trusted.

Run from the folder above with matplotlib (../requirements-plots.txt):  python3 -B figures/erasure_rates.py
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
LAST_TERM = 0.01


def pauli_series(name):
    return [float(Fraction(v)) for v in json.loads((RESULTS / name).read_text())['P_L']]


def curve(L, ps, rate):
    peff = ps + 0.75 * rate * ps - rate * ps ** 2
    x = peff / (1 - peff)
    terms = np.array([c * x ** k for k, c in enumerate(L)])
    total = terms.sum(axis=0)
    ok = np.abs(terms[-1]) <= LAST_TERM * total
    stop = np.argmin(ok) if not ok.all() else len(ps)
    return ps[:stop], total[:stop]


def main():
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
    ps = np.logspace(-4, -2, 600)
    colours = {3: '#1f6fb2', 5: '#c2452d'}
    styles = {0: '-', 1: (0, (5, 2)), 5: (0, (5, 1.5, 1.2, 1.5)), 10: (0, (1.2, 1.4))}
    labels = {0: r'$e=0$', 1: r'$e=p$', 5: r'$e=5p$', 10: r'$e=10p$'}
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for d, name in ((3, 'd3_edges_uniform.json'), (5, 'd5_edges_uniform.json')):
        L = pauli_series(name)
        for rate, style in styles.items():
            x, y = curve(L, ps, rate)
            ax.loglog(x, y, color=colours[d], ls=style, lw=1.3)
            if len(x) < len(ps):
                ax.plot(x[-1], y[-1], 'o', ms=2.4, color=colours[d], mec='none')
    ax.set_xlabel(r'physical error rate $p$')
    ax.set_ylabel(r'logical error rate $P_\mathrm{L}$')
    ax.set_xlim(1e-4, 1e-2)
    ax.grid(True, which='major', lw=0.3, color='0.85')
    keys = [Line2D([], [], color=colours[3], lw=1.3, label=r'$d=3$'),
            Line2D([], [], color=colours[5], lw=1.3, label=r'$d=5$')]
    keys += [Line2D([], [], color='0.25', lw=1.3, ls=styles[r], label=labels[r]) for r in styles]
    ax.legend(handles=keys, fontsize=7, frameon=False, loc='lower right', ncol=2, handlelength=2.6,
              columnspacing=1.2, title='Pauli and erasure\n(flag ignored) on every edge', title_fontsize=7)
    fig.tight_layout()
    for ext, kw in (('png', dict(dpi=600, metadata={'Software': None})),
                    ('pdf', dict(metadata={'Creator': None, 'Producer': None, 'CreationDate': None}))):
        fig.savefig(HERE / f'ler_erasure_rates.{ext}', **kw)
    print('saved', HERE / 'ler_erasure_rates.png')


if __name__ == '__main__':
    main()
