"""Noiseless statevector run of a small source, with the exact probability of every record.

One trajectory per seed: each outcome is drawn with the Born rule, and its probability
is kept. A record is deterministic when its probability is 1 in every trajectory.
Resets are measure-and-prepare, an exact unravelling of the reset channel.
Only practical up to about 20 qubits, so it is used for d=3.
"""

import numpy as np

from .circuit import parse_text

PHASE = {0: 1, 1: np.exp(1j * np.pi / 4), 2: 1j, 4: -1, 6: -1j, 7: np.exp(-1j * np.pi / 4)}


def run(text, seed):
    p = parse_text(text)
    qs = sorted({o[k] for o in p['ops'] for k in ('q', 'c', 't') if k in o}
                | {q for o in p['ops'] if o['op'] == 'MPP' for _, q in o['factors']})
    axis = {q: i for i, q in enumerate(qs)}
    n = len(qs)
    psi = np.zeros((2,) * n, complex)
    psi[(0,) * n] = 1
    rng = np.random.default_rng(seed)
    prob, bit = {}, {}

    def at(q, b):
        s = [slice(None)] * n
        s[axis[q]] = b
        return tuple(s)

    def pauli(state, P, q):
        out = state.copy()
        if P in 'ZY':
            out[at(q, 1)] *= -1
        if P in 'XY':
            out = np.flip(out, axis[q])
        return 1j * out if P == 'Y' else out      # Y = i X Z

    def measure(factors):
        nonlocal psi
        image = psi
        for P, q in factors:
            image = pauli(image, P, q)
        p0 = (1 + np.vdot(psi, image).real) / 2
        b = int(rng.random() >= p0)
        psi = (psi + (-1) ** b * image) / 2
        psi /= np.linalg.norm(psi)
        return b, p0 if b == 0 else 1 - p0

    for o in p['ops']:
        k = o['op']
        if k == 'PHASE':
            psi[at(o['q'], 1)] *= PHASE[o['phase']]
        elif k == 'CX':
            c, t = axis[o['c']], axis[o['t']]
            view = psi[at(o['c'], 1)]
            psi[at(o['c'], 1)] = np.flip(view, t - (t > c))
        elif k in ('R', 'RX'):
            b, _ = measure([('Z' if k == 'R' else 'X', o['q'])])
            if b:
                psi = pauli(psi, 'X' if k == 'R' else 'Z', o['q'])
        elif k in ('M', 'MX', 'MPP'):
            factors = o['factors'] if k == 'MPP' else [('Z' if k == 'M' else 'X', o['q'])]
            bit[o['record']], prob[o['record']] = measure(factors)
        else:
            raise ValueError(f'statevector run does not support {k}')
    return p, prob, bit
