"""Check that every TikZ file draws exactly the computed web and defects.

For each detector page and each standalone figure:
  - every edge drawn in a web style has that label in the computed web,
  - every labelled edge inside the drawn window is drawn in its style,
  - the defect comments list exactly the computed defects inside the window,
    with the right phase, shift and changed phase, and one star each.
With --paper DIR, the copies in that paper source tree must be byte-identical.
Run from pauli_semiwebs_check:  python -B -m check_basis.guide.check_tikz [--paper DIR]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .build import FIGURES, OUT
from .select import detector_webs

EDGE = re.compile(r'^\\draw(?:\[style=([XYZ]) Web\])? \((n\d+)\.center\) to \((n\d+)\.center\);$')
CUT = re.compile(r'^\\draw(?:\[style=([XYZ]) Web\])? \(')
STAR = re.compile(r'\\ast_\{(\d+)\}')


def check_file(path, g, web):
    text = path.read_text().split('\n')
    labels = {}
    for j, e in enumerate(g['edges']):
        lab = 'IXZY'[(web['vector'] >> (2 * j)) & 3]
        labels.setdefault(frozenset((e['u'], e['v'])), []).append(lab)
    drawn, cut_web = Counter(), 0
    for line in text:
        m = EDGE.match(line)
        if m:
            style, u, v = m.groups()
            drawn[(frozenset((u, v)), style or 'I')] += 1
        elif CUT.match(line) and '[style=' in line:
            cut_web += 1
    named = {k for line in text for k in re.findall(r'\((n\d+)\) at', line)}
    problems = []
    for (pair, style), count in drawn.items():
        want = Counter(labels.get(pair, []))
        if want[style] < count:
            problems.append(f'edge {sorted(pair)} drawn as {style}, web has {dict(want)}')
    expected = Counter()
    for pair, labs in labels.items():
        if pair <= named:
            for lab in labs:
                expected[(pair, lab)] += 1
    for key, count in expected.items():
        if key[1] != 'I' and drawn[key] != count:
            problems.append(f'labelled edge {sorted(key[0])} {key[1]} drawn {drawn[key]} of {count} times')
    metas = [json.loads(line[len('% defect: '):]) for line in text if line.startswith('% defect: ')]
    got = {m['node']: m for m in metas}
    want = {k: dlt for k, dlt in web['defects'].items() if k in named}
    if set(got) != set(want):
        problems.append(f'defect nodes differ: drawn {sorted(got)} computed {sorted(want)}')
    for k, m in got.items():
        n = g['nodes'][k]
        phase = n['phase'] if n['phase'] <= 4 else n['phase'] - 8
        changed = (n['phase'] + want.get(k, 0)) % 8
        changed = changed if changed <= 4 else changed - 8
        if (m['delta_pi_over_4'], m['phase_pi_over_4'], m['changed_phase_pi_over_4'], m['qubit'], m['line']) != \
                (want.get(k), phase, changed, n['qubit'], n['line']):
            problems.append(f'defect metadata wrong at {k}: {m}')
    stars = sorted(int(s) for line in text if 'text=violet' in line for s in STAR.findall(line))
    if stars != list(range(1, len(got) + 1)):
        problems.append(f'stars {stars} for {len(got)} defects')
    web_drawn = sum(c for (pair, style), c in drawn.items() if style != 'I')
    return problems, web_drawn, cut_web, len(got)


def main(paper=None):
    total, bad = 0, 0
    for key in (3, 5):
        r = detector_webs(key)
        g = r['graph']
        for web in r['detectors']:
            name = f"d{key}_det{web['index']:03d}.tikz"
            path = OUT / 'diagrams' / name
            problems, n_web, n_cut, n_def = check_file(path, g, web)
            if n_web != web['labelled_edges']:
                problems.append(f'{n_web} web edges drawn node to node, web has {web["labelled_edges"]}')
            if paper is not None:
                copy = Path(paper) / 'detector_webs' / 'diagrams' / name
                if not copy.exists() or copy.read_bytes() != path.read_bytes():
                    problems.append('paper copy missing or different')
            total += 1
            if problems:
                bad += 1
                print('FAIL', name, problems[:3])
        if key == 3:
            for name, index, _, _ in FIGURES:
                figure = OUT / 'figures' / name
                web = next(d for d in r['detectors'] if d['index'] == index)
                problems, n_web, n_cut, n_def = check_file(figure, g, web)
                if index == 14 and paper is not None:      # the paper holds the Appendix J figure
                    copy = Path(paper) / 'figures' / 'detector_webs' / name
                    if not copy.exists() or copy.read_bytes() != figure.read_bytes():
                        problems.append('paper copy missing or different')
                total += 1
                bad += bool(problems)
                print('FAIL' if problems else 'PASS', name, f'{n_web} web edges inside, {n_cut} cut web wires, '
                      f'{n_def} defects', problems[:3])
    print(f'{total - bad} of {total} TikZ files draw exactly their computed web and defects')
    return total - bad, total


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper', type=Path, help='also require byte-identical copies in this paper source tree')
    ok, total = main(p.parse_args().paper)
    raise SystemExit(0 if ok == total else 1)
