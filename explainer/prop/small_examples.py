"""The small worked circuits of the explainer: propagated step by step, and checked against a density-matrix
simulation at several noise strengths.

  python -B small_examples.py      # prints the checks
"""
from fractions import Fraction

from engine import (Poly, Operator, ONE, parse, records, step_ops, apply_backward, expect_zero, density_check,
                    string_tex)

# Each example: a Stim text, the parity of records it reads, which qubits to draw and a note per operation.
EXAMPLES = {
    'one_t': dict(
        title='One \\(T\\) gate',
        text='RX 0\nT 0\nMX 0\n',
        parity=[1], qubits=[0],
        lesson='One fork, then the input decides: \\(\\langle X\\rangle=1\\) and \\(\\langle Y\\rangle=0\\) on \\(\\ket+\\).',
    ),
    'two_t': dict(
        title='Two \\(T\\) gates make \\(S\\)',
        text='RX 0\nT 0\nT 0\nMX 0\n',
        parity=[1], qubits=[0],
        lesson='Four paths, but the two \\(X\\) paths cancel and the two \\(Y\\) paths add: \\(S^\\dagger XS=-Y\\).',
    ),
    't_tdag': dict(
        title='\\(T\\) then \\(T^\\dagger\\) undo each other',
        text='RX 0\nT 0\nT_DAG 0\nMX 0\n',
        parity=[1], qubits=[0],
        lesson='The fork closes again: the two \\(Y\\) paths cancel and \\(X\\) comes back with coefficient 1.',
    ),
    'detector': dict(
        title='A detector through a CNOT, with noise',
        text='RX 0\nR 1\nDEPOLARIZE1(p) 0 1\nCX 0 1\nDEPOLARIZE2(p) 0 1\nMX(p) 0 1\n',
        parity=[1, 2], qubits=[0, 1],
        lesson='One string all the way: a Clifford circuit never forks it. Noise only counts where the string has support.',
    ),
    'mini_check': dict(
        title='A double check on one qubit',
        text='RX 0\nT_DAG 0\nRX 1\nT 0\nDEPOLARIZE1(p) 0\nCX 1 0\nDEPOLARIZE2(p) 1 0\nT_DAG 0\nDEPOLARIZE1(p) 0\nMX(p) 1\n',
        parity=[1], qubits=[0, 1],
        lesson='The readout measures \\(T^\\dagger XT=(X-Y)/\\sqrt2\\) on the data, whose \\(+1\\) state is \\(\\ket{T^\\dagger}\\).',
    ),
}


def run(name, noisy=True):
    ex = EXAMPLES[name]
    text = ex['text'].replace('(p)', '(0.001)')
    ops = parse(text)
    rec_of, _ = records(ops)
    steps = step_ops(ops)
    O = Operator.identity()
    hist = []
    for st in reversed(steps):
        apply_backward(O, st, rec_of, set(ex['parity']), noisy)
        hist.append((st, O.copy()))
    return O, hist


def check_all():
    for name, ex in EXAMPLES.items():
        O, hist = run(name)
        mean = expect_zero(O)
        for p in (0.0, 0.01, 0.137):
            text = ex['text'].replace('(p)', f'({p})')
            ref = density_check(text, ex['parity'], p)
            got = mean.value(p)
            assert abs(ref - got) < 1e-12, (name, p, ref, got)
        print(f'{name:12s} mean = {mean.tex():40s} terms at the start: {len(O)}; density matrix agrees at p=0, 0.01, 0.137')


if __name__ == '__main__':
    check_all()
