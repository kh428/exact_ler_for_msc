"""A circuit, or a stage of it, with its noise locations as a Pauli-label network.

The tensors are the repository's (alternatives/calculation/chan.py, RecordedPauliTensors, in the X^a Z^b label
basis): a label network whose value is the series of A (all detectors pass) or B (all pass and the observable is
flipped), every location contributing 1 + x (its law), x = p/(1-p). Only the events are built here.

A stage network can start from an open input on chosen qubits and end with open output labels on chosen qubits; the
open indices are returned with the network and kept by every simplification.

  build(circuit, locations, ring, root2, observable='B')       the whole circuit
"""
from fractions import Fraction

from . import REPO  # noqa: F401  (puts the repository on the path)
from .circuit import touched
from .noise import law_value

from alternatives.calculation.chan import RecordedPauliTensors          # noqa: E402
from calculation.truncated_polynomial import PolynomialRing             # noqa: E402,F401

PRODUCT_ANCILLA = 10000


def residue(value, prime):
    value = Fraction(value)
    return value.numerator % prime * pow(value.denominator % prime, -1, prime) % prime


class Tensors(RecordedPauliTensors):
    """The repository's tensors, with the general edge law and classical record flips."""

    def law(self, loc):
        bits = []
        for gen in loc['generators']:
            (q, s), = gen.items()
            x, z = self.wire(q)
            bits.append(z if s == 'X' else x if s == 'Z' else self.xor(x, z, 'y-part'))
        prime = self.ring.prime
        table = {(b0, b1): self.ring.coefficients((1, residue(law_value(loc['weights'], (b0, b1)), prime)))
                 for b0 in (0, 1) for b1 in (0, 1)}
        self.channels += 1
        self.factor(bits, lambda b: table[b])

    def record_flip(self, record):
        """A classical flip of a recorded outcome: a new record bit equal to the old one, or flipped with weight x."""
        new = self.variable('flipped-record')
        same, flip = self.ring.coefficients((1, 0)), self.ring.coefficients((0, 1))
        self.factor((record, new), lambda b: same if b[0] == b[1] else flip)
        self.channels += 1
        return new


def build(circuit, locations, ring, root2, *, observable=None, reference=0, gates=None, detectors=None,
          start=None, finish=None):
    """The network and its open variables.

    gates       (first, stop) gate indices of the stage, default the whole circuit
    detectors   detectors to impose (dicts with records and after), default every detector of the circuit
    observable  'A', 'B' or None (no observable condition)
    start       a function of the compiler that sets the initial wires of some qubits and returns open variables
    finish      a function of the compiler called after the last gate, before every remaining wire is traced out;
                it removes the wires it handles and returns open variables
    """
    first, stop = gates or (0, len(circuit['gates']))
    compiler = Tensors(ring, root2=root2, basis='xz')
    open_vars = list(start(compiler)) if start else []
    after, flips = {}, {}
    for loc in locations:
        if loc['kind'] == 'record':
            flips[loc['record']] = loc
        else:
            after.setdefault(max(loc['after'], first - 1), []).append(loc)
    by_gate = {}
    for d in (circuit['detectors'] if detectors is None else detectors):
        by_gate.setdefault(max(d['after'], first - 1), []).append(d)
    records = {}

    def close(g):
        for loc in after.get(g, ()):
            if loc['kind'] == 'law':
                compiler.law(loc)
            elif loc['kind'] == 'flip':
                compiler.noise(dict(kind='flip', axis=loc['axis'], support=loc['qubits'], denominator=1))
            elif loc['kind'] == 'depolarizing':
                compiler.noise(dict(kind='depolarizing', support=loc['qubits'],
                                    denominator=3 if len(loc['qubits']) == 1 else 15))
            else:
                raise ValueError(loc['kind'])
        for d in by_gate.get(g, ()):
            compiler.parity_constraint([records[i] for i in d['records']], 0, 'detector')

    close(first - 1)
    for g in range(first, stop):
        gate = circuit['gates'][g]
        op = gate['op']
        if op in ('M', 'MX'):
            records[gate['record']] = compiler.measure(gate['q'], 'Z' if op == 'M' else 'X', False)
        elif op == 'MPP':
            pauli, qubits = gate['pauli'], gate['qubits']
            if pauli == 'Y':                     # Y = S X S_dag: measure X between S_dag and S
                for q in qubits:
                    compiler.gate('S_DAG', [q])
            x_axis = pauli != 'Z'
            compiler.gate('RX' if x_axis else 'R', [PRODUCT_ANCILLA])
            for q in qubits:
                compiler.gate('CX', [PRODUCT_ANCILLA, q] if x_axis else [q, PRODUCT_ANCILLA])
            records[gate['record']] = compiler.measure(PRODUCT_ANCILLA, 'X' if x_axis else 'Z')
            compiler.gate('R', [PRODUCT_ANCILLA])
            if pauli == 'Y':
                for q in qubits:
                    compiler.gate('S', [q])
        elif op == 'CZ':                         # CZ = H on the target, CX, H on the target
            compiler.gate('H', [gate['t']])
            compiler.gate('CX', [gate['c'], gate['t']])
            compiler.gate('H', [gate['t']])
        elif op == 'FEEDBACK':
            compiler.feedback(records[gate['record']], gate['q'], gate['axis'])
        else:
            compiler.gate(op, touched(gate))
        if 'record' in gate and gate['record'] in flips and op != 'FEEDBACK':
            records[gate['record']] = compiler.record_flip(records[gate['record']])
        close(g)
    if observable == 'B':
        compiler.parity_constraint([records[i] for i in circuit['observable']], reference ^ 1, 'observable')
    elif observable not in ('A', None):
        raise ValueError(observable)
    if finish:
        open_vars += list(finish(compiler))
    net = compiler.trace()
    net.simplify(protected=open_vars)
    return net, open_vars, compiler.channels


def to_engine(net, ring, outputs=()):
    """The same network as float64 series tensors for engine.Net, with the given open variables."""
    import numpy as np
    from .engine import Net
    K, q = ring.degree, ring.prime
    tensors = []
    for f in net.factors:
        vals = np.zeros((len(f.values.flat), K + 1))
        for i, v in enumerate(f.values.flat):
            co = v.coefficients if hasattr(v, 'coefficients') else ring(v).coefficients
            vals[i, :len(co)] = co
        tensors.append((tuple(f'v{v}' for v in f.scope), vals.reshape((2,) * len(f.scope) + (K + 1,))))
    s = net.scalar
    scalar = list(s.coefficients) if hasattr(s, 'coefficients') else list(ring(s).coefficients)
    return Net(tensors, [f'v{v}' for v in outputs], scalar + [0] * (K + 1 - len(scalar)), K, q)
