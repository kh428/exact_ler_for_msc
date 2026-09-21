"""Build the two fixed-fault double-checking examples from frozen circuits."""
import re
from pathlib import Path
import stim

INPUTS = Path(__file__).resolve().parents[1] / "data/inputs"

CASES = {
    'd3': dict(
        file=str(INPUTS / 'clifft_d3_p001.stim'),
        sites=(0, 3, 7, 9, 10, 12, 13),
        faces=((0, 3, 7, 10), (3, 7, 9, 12), (7, 10, 12, 13)),
        window=(131, 180),                       # RX ancillas ... final MX of ancillas
        faults={142: ['X 3'], 173: ['Y 9']},     # after these source lines
        expected=(0.0, 0.25),
    ),
    'd5': dict(
        file=str(INPUTS / 'soft_cultivation_d5_p0005.stim'),
        sites=(0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40),
        faces=((0, 3, 7, 11), (9, 13, 15, 19, 21, 27), (17, 25, 32, 36), (3, 7, 9, 13), (15, 21, 23, 29),
               (19, 25, 27, 32, 34, 38), (7, 11, 13, 17, 19, 25), (21, 27, 29, 34), (32, 36, 38, 40)),
        window=(388, 444),
        faults={397: ['X 3', 'X 14'], 438: ['Y 23']},
        expected=(0.0, 1 / 16),
    ),
}

NOISE = re.compile(r'^(DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL|TICK|SHIFT_COORDS|DETECTOR|OBSERVABLE|QUBIT_COORDS|#)')

def window_lines(path, lo, hi, faults):
    out = []
    lines = Path(path).read_text().split('\n')
    for n in range(lo, hi + 1):
        line = lines[n - 1].strip()
        if not line or NOISE.match(line):
            continue
        line = re.sub(r'\(([^)]*)\)', '', line, count=1) if line.startswith(('M', 'R')) else line
        out.append(line)
        for g in faults.get(n, []):
            out.append(g)
    return out

def encoder(sites, faces, z_sign=+1):
    """Clifford circuit text: |psi> on sites[-1], |0> elsewhere  ->  |psi_L> (Z_{last} -> Z_L)."""
    n = len(sites)
    idx = {q: i for i, q in enumerate(sites)}
    stabs = []
    for P in 'XZ':
        for f in faces:
            s = ['_'] * n
            for q in f:
                s[idx[q]] = P
            stabs.append(stim.PauliString(''.join(s)))
    stabs.append(stim.PauliString(('+' if z_sign > 0 else '-') + 'Z' * n))     # Z_L = Z on every site
    tab = stim.Tableau.from_stabilizers(stabs)
    circ = tab.to_circuit(method='elimination')
    def relabel(c):
        text = []
        for inst in c.flattened():
            targets = [sites[t.value] for t in inst.targets_copy()]
            text.append(inst.name + ' ' + ' '.join(str(t) for t in targets))
        return text
    return relabel(circ), relabel(circ.inverse()), tab

def build(case, flip_input=False, with_faults=True):
    c = CASES[case]
    sites, faces = c['sites'], c['faces']
    enc, dec, tab = encoder(sites, faces)
    lin = sites[-1]                                   # logical input qubit
    text = ['R ' + ' '.join(str(q) for q in sites), f'H {lin}', f'T_DAG {lin}']
    if flip_input:
        text.append(f'Z {lin}')
    text += enc
    text += window_lines(c['file'], *c['window'], c['faults'] if with_faults else {})
    text += dec
    text.append('M ' + ' '.join(str(q) for q in sites[:-1]))   # syndrome qubits must read 0
    text.append(f'T {lin}')
    text.append(f'MX {lin}')                                    # 0 = good (M_L=+1), 1 = bad
    return '\n'.join(text)
