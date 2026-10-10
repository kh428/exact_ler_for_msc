"""Exact Pauli propagation in the Heisenberg picture, for the worked examples of the explainer.

An operator is a sum of Pauli strings with exact coefficients. A string is a pair of bit masks (x, z) over
qubits: X sets x, Z sets z, Y sets both, so a string is a tensor product of the Hermitian I, X, Y, Z and every
coefficient is real. A coefficient is a polynomial in

    s = 1/sqrt(2)  (s^2 = 1/2),   l1 = 1 - 4p/3,   l2 = 1 - 16p/15,   f = 1 - 2p,

stored as {(s, e1, e2, ef): Fraction} with s in {0, 1}.

Circuits are Stim text with T and T_DAG allowed. The observable is a parity of measurement records; it is
carried from the end of the circuit to the start:

  unitary U            each string P -> U^dag P U (looked up in tables computed from the matrices)
  Pauli channel        each string P -> lambda_P P, lambda_P = 1 - 2 Pr(fault anticommutes with P)
  measurement of Q     strings anticommuting with Q are dropped; if the record is in the parity the rest
                       are multiplied by Q, and by f if the readout flips with probability p
  reset to |0> (|+>)   Z (X) on that qubit -> I, X or Y (Z or Y) -> 0
"""
import re
from fractions import Fraction
from itertools import product

import numpy as np

# ------------------------------------------------------------------ coefficients

class Poly:
    __slots__ = ('t',)

    def __init__(self, terms=None):
        self.t = {k: v for k, v in (terms or {}).items() if v}

    @staticmethod
    def const(c):
        return Poly({(0, 0, 0, 0): Fraction(c)})

    @staticmethod
    def sym(name, power=1):
        k = {'s': (1, 0, 0, 0), 'l1': (0, 1, 0, 0), 'l2': (0, 0, 1, 0), 'f': (0, 0, 0, 1)}[name]
        out = Poly.const(1)
        for _ in range(power):
            out = out * Poly({k: Fraction(1)})
        return out

    def __add__(self, o):
        t = dict(self.t)
        for k, v in o.t.items():
            t[k] = t.get(k, 0) + v
        return Poly(t)

    def __neg__(self):
        return Poly({k: -v for k, v in self.t.items()})

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if not isinstance(o, Poly):
            o = Poly.const(o)
        t = {}
        for (s1, a1, b1, c1), v1 in self.t.items():
            for (s2, a2, b2, c2), v2 in o.t.items():
                s, v = s1 + s2, v1 * v2
                if s == 2:
                    s, v = 0, v / 2
                k = (s, a1 + a2, b1 + b2, c1 + c2)
                t[k] = t.get(k, 0) + v
        return Poly(t)

    __rmul__ = __mul__

    def is_zero(self):
        return not self.t

    def __eq__(self, o):
        return isinstance(o, Poly) and self.t == o.t

    def __hash__(self):
        return hash(tuple(sorted(self.t.items())))

    def value(self, p):
        """Float value at noise strength p."""
        s, l1, l2, f = 2 ** -0.5, 1 - 4 * p / 3, 1 - 16 * p / 15, 1 - 2 * p
        return sum(float(v) * s ** a * l1 ** b * l2 ** c * f ** d for (a, b, c, d), v in self.t.items())

    def in_p(self):
        """Exact polynomial in p as a list of Fractions, when no s remains."""
        assert all(k[0] == 0 for k in self.t), 'an odd power of 1/sqrt(2) remains'
        out = [Fraction(0)]

        def mul(a, b):
            r = [Fraction(0)] * (len(a) + len(b) - 1)
            for i, x in enumerate(a):
                for j, y in enumerate(b):
                    r[i + j] += x * y
            return r

        base = {1: [Fraction(1), Fraction(-4, 3)], 2: [Fraction(1), Fraction(-16, 15)], 3: [Fraction(1), Fraction(-2)]}
        for (_, b, c, d), v in self.t.items():
            term = [v]
            for which, e in ((1, b), (2, c), (3, d)):
                for _ in range(e):
                    term = mul(term, base[which])
            if len(term) > len(out):
                out += [Fraction(0)] * (len(term) - len(out))
            for i, x in enumerate(term):
                out[i] += x
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    def tex(self):
        """TeX for the coefficient, monomials in a fixed order."""
        if not self.t:
            return '0'
        parts = []
        for (a, b, c, d), v in sorted(self.t.items(), key=lambda kv: (-kv[0][0], kv[0][1:])):
            sym = ''
            for name, e in (('\\lambda_1', b), ('\\lambda_2', c), ('f', d)):
                if e == 1:
                    sym += name
                elif e > 1:
                    sym += f'{name}^{{{e}}}'
            num, den = abs(v.numerator), v.denominator
            sign = '-' if v < 0 else '+'
            if a:
                den_tex = f'{den}\\sqrt2' if den != 1 else '\\sqrt2'
            else:
                den_tex = str(den) if den != 1 else ''
            if den_tex:
                top = (str(num) if num != 1 else '') + sym or '1'
                body = f'\\frac{{{top}}}{{{den_tex}}}'
            else:
                body = (str(num) if (num != 1 or not sym) else '') + sym
            parts.append((sign, body))
        out = ''.join((s if i or s == '-' else '') + b for i, (s, b) in enumerate(parts))
        return out


ONE = Poly.const(1)
S = Poly.sym('s')

# ------------------------------------------------------------------ Pauli tables from matrices

PM = {'I': np.eye(2, dtype=complex), 'X': np.array([[0, 1], [1, 0]], dtype=complex),
      'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
W8 = np.exp(1j * np.pi / 4)
UNITARY = {
    'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    'S': np.diag([1, 1j]), 'S_DAG': np.diag([1, -1j]),
    'T': np.diag([1, W8]), 'T_DAG': np.diag([1, np.conj(W8)]),
    'X': PM['X'], 'Y': PM['Y'], 'Z': PM['Z'],
    'CX': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    'CZ': np.diag([1, 1, 1, -1]).astype(complex),
}
LETTERS = 'IXYZ'


def _recognise(c):
    """An exact coefficient for 0, +-1, +-1/sqrt2."""
    for val, poly in ((1, ONE), (-1, -ONE), (2 ** -0.5, S), (-2 ** -0.5, -S)):
        if abs(c - val) < 1e-12:
            return poly
    assert abs(c) < 1e-12, c
    return None


def _table(U):
    n = int(round(np.log2(U.shape[0])))
    out = {}
    for P in product(LETTERS, repeat=n):
        M = PM[P[0]]
        for l in P[1:]:
            M = np.kron(M, PM[l])
        back = U.conj().T @ M @ U
        terms = []
        for Q in product(LETTERS, repeat=n):
            N = PM[Q[0]]
            for l in Q[1:]:
                N = np.kron(N, PM[l])
            c = np.trace(N @ back) / 2 ** n
            assert abs(c.imag) < 1e-12
            poly = _recognise(c.real)
            if poly is not None:
                terms.append((''.join(Q), poly))
        out[''.join(P)] = terms
    return out


TABLES = {name: _table(U) for name, U in UNITARY.items()}


def backward_rule(gate, letters):
    """U^dag P U for a local Pauli word, as [(word, Poly)]."""
    return TABLES[gate][letters]

# ------------------------------------------------------------------ strings


def letter(x, z, q):
    return 'IXZY'[(x >> q & 1) + 2 * (z >> q & 1)]


def set_letter(x, z, q, l):
    x &= ~(1 << q)
    z &= ~(1 << q)
    if l in 'XY':
        x |= 1 << q
    if l in 'ZY':
        z |= 1 << q
    return x, z


def anticommutes(a, b):
    (x1, z1), (x2, z2) = a, b
    return (bin(x1 & z2).count('1') + bin(z1 & x2).count('1')) & 1


def string_tex(key, order=None):
    x, z = key
    qs = sorted(set(q for q in range((x | z).bit_length()) if (x | z) >> q & 1))
    if order:
        qs = sorted(qs, key=order)
    if not qs:
        return 'I'
    return ''.join(f'{letter(x, z, q)}_{{{q}}}' for q in qs)


def string_word(key, qubits):
    return ''.join(letter(*key, q) for q in qubits)


def mul_strings(a, b):
    """Product of two Hermitian strings: (phase as a power of i, string)."""
    (x1, z1), (x2, z2) = a, b
    # P = i^{|x z|} X^x Z^z for a Hermitian string; multiply in X^x Z^z form and track i's.
    k1 = bin(x1 & z1).count('1')
    k2 = bin(x2 & z2).count('1')
    # X^x1 Z^z1 X^x2 Z^z2 = (-1)^{|z1 & x2|} X^{x1^x2} Z^{z1^z2}
    sign = bin(z1 & x2).count('1') & 1
    x, z = x1 ^ x2, z1 ^ z2
    k = bin(x & z).count('1')
    power = (k1 + k2 + 2 * sign - k) % 4       # i^{k1+k2} (-1)^sign X Z = i^{power} * (i^k X^x Z^z)
    return power, (x, z)


class Operator:
    def __init__(self, terms=None):
        self.terms = dict(terms or {})

    @staticmethod
    def identity():
        return Operator({(0, 0): ONE})

    def copy(self):
        return Operator(self.terms)

    def add(self, key, coef):
        c = self.terms.get(key)
        c = coef if c is None else c + coef
        if c.is_zero():
            self.terms.pop(key, None)
        else:
            self.terms[key] = c

    def unitary(self, gate, qubits):
        out = Operator()
        for key, coef in self.terms.items():
            x, z = key
            word = ''.join(letter(x, z, q) for q in qubits)
            for new, c in backward_rule(gate, word):
                nx, nz = x, z
                for q, l in zip(qubits, new):
                    nx, nz = set_letter(nx, nz, q, l)
                out.add((nx, nz), coef * c)
        self.terms = out.terms

    def channel(self, lam_of_word, qubits):
        out = Operator()
        for key, coef in self.terms.items():
            word = ''.join(letter(*key, q) for q in qubits)
            lam = lam_of_word(word)
            if lam is not None:
                out.add(key, coef * lam)
        self.terms = out.terms

    def measure(self, pauli_key, in_parity, flip):
        out = Operator()
        for key, coef in self.terms.items():
            if anticommutes(key, pauli_key):
                continue
            if in_parity:
                power, new = mul_strings(pauli_key, key)
                # Q P for commuting Hermitian Q, P is Hermitian, so the phase is real
                assert power in (0, 2), power
                c = coef if power == 0 else -coef
                if flip:
                    c = c * Poly.sym('f')
                out.add(new, c)
            else:
                out.add(key, coef)
        self.terms = out.terms

    def reset(self, q, basis):
        out = Operator()
        keep = 'Z' if basis == 'Z' else 'X'
        for key, coef in self.terms.items():
            l = letter(*key, q)
            if l == 'I':
                out.add(key, coef)
            elif l == keep:
                out.add(set_letter(*key, q, 'I'), coef)
        self.terms = out.terms

    def __len__(self):
        return len(self.terms)


# Pauli channels as eigenvalues on local words
def lam_depolarize1(word):
    return ONE if word == 'I' else Poly.sym('l1')


def lam_depolarize2(word):
    return ONE if word == 'II' else Poly.sym('l2')


def lam_flip(axis):
    """X_ERROR, Y_ERROR or Z_ERROR: 1 on I and the axis, f on the two anticommuting letters."""
    def lam(word):
        return ONE if word in ('I', axis) else Poly.sym('f')
    return lam

# ------------------------------------------------------------------ circuits

OP_RE = re.compile(r'^([A-Z_]+[0-9]?)(?:\(([^)]*)\))?\s*(.*)$')


def parse(text, first_line=1):
    """[(line, op, arg, targets)] for the operations of a Stim text; comments and blank lines are skipped."""
    ops = []
    for i, raw in enumerate(text.splitlines(), start=first_line):
        line = raw.split('#')[0].strip()
        if not line:
            continue
        m = OP_RE.match(line)
        if not m:
            raise ValueError(raw)
        name, arg, rest = m.groups()
        targets = rest.split()
        ops.append((i, name, arg, targets))
    return ops


def records(ops):
    """Number every measurement record in order (1-based), as {(op index, target index): record}."""
    out, k = {}, 0
    for i, (line, name, arg, targets) in enumerate(ops):
        if name in ('M', 'MX', 'MY', 'MPP', 'MR', 'MRX'):
            for j, _ in enumerate(targets):
                k += 1
                out[(i, j)] = k
    return out, k


def mpp_key(target):
    x = z = 0
    for part in target.split('*'):
        l, q = part[0], int(part[1:])
        x, z = set_letter(x, z, q, l)
    return x, z


def single_strength(ops):
    """The engine gives every channel and every readout flip the same symbols, so a circuit must use one noise
    strength throughout. Returns that strength (None if noiseless); raises otherwise."""
    args = {arg for (_, name, arg, _) in ops
            if arg is not None and name in ('DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'M', 'MX', 'MPP')}
    if len(args) > 1:
        raise ValueError(f'more than one noise strength: {sorted(args)}')
    return next(iter(args), None)


def step_ops(ops):
    """Split every operation into elementary steps with the qubits each acts on."""
    steps = []
    for i, (line, name, arg, targets) in enumerate(ops):
        if name in ('TICK', 'QUBIT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS'):
            continue
        if name in ('CX', 'CZ') and any(t.startswith('rec[') for t in targets):
            # a Pauli applied when a record is 1 (feedback); not used by the examples
            steps.append(dict(op=i, line=line, kind='feedback', gate=name, targets=targets))
        elif name in ('CX', 'CZ'):
            pairs = [(int(targets[k]), int(targets[k + 1])) for k in range(0, len(targets), 2)]
            steps.append(dict(op=i, line=line, kind='unitary', gate=name, groups=pairs))
        elif name in UNITARY:
            steps.append(dict(op=i, line=line, kind='unitary', gate=name, groups=[(int(t),) for t in targets]))
        elif name == 'DEPOLARIZE1':
            steps.append(dict(op=i, line=line, kind='noise', gate=name, p=arg, groups=[(int(t),) for t in targets]))
        elif name == 'DEPOLARIZE2':
            steps.append(dict(op=i, line=line, kind='noise', gate=name, p=arg,
                              groups=[(int(targets[k]), int(targets[k + 1])) for k in range(0, len(targets), 2)]))
        elif name in ('X_ERROR', 'Y_ERROR', 'Z_ERROR'):
            steps.append(dict(op=i, line=line, kind='noise', gate=name, p=arg, groups=[(int(t),) for t in targets]))
        elif name in ('M', 'MX', 'MPP'):
            steps.append(dict(op=i, line=line, kind='measure', gate=name, p=arg, targets=targets))
        elif name in ('R', 'RX'):
            steps.append(dict(op=i, line=line, kind='reset', gate=name, groups=[(int(t),) for t in targets]))
        else:
            raise ValueError(name)
    return steps


def apply_backward(O, step, rec_of, parity, noisy=True):
    """Apply one elementary step to the operator, backwards."""
    kind = step['kind']
    if kind == 'feedback':
        raise NotImplementedError('feedback inside a propagated region')
    if kind == 'unitary':
        for g in reversed(step['groups']):
            O.unitary(step['gate'], g)
    elif kind == 'noise':
        if not noisy:
            return
        lam = {'DEPOLARIZE1': lam_depolarize1, 'DEPOLARIZE2': lam_depolarize2,
               'X_ERROR': lam_flip('X'), 'Y_ERROR': lam_flip('Y'), 'Z_ERROR': lam_flip('Z')}[step['gate']]
        for g in step['groups']:
            O.channel(lam, g)
    elif kind == 'measure':
        for j in reversed(range(len(step['targets']))):
            t = step['targets'][j]
            if step['gate'] == 'MPP':
                key = mpp_key(t)
            else:
                q = int(t)
                key = set_letter(0, 0, q, 'X' if step['gate'] == 'MX' else 'Z')
            r = rec_of[(step['op'], j)]
            O.measure(key, r in parity, flip=noisy and step['p'] is not None)
    elif kind == 'reset':
        for (q,) in step['groups']:
            O.reset(q, 'X' if step['gate'] == 'RX' else 'Z')


def propagate(text, parity, noisy=True, first_line=1, keep=None):
    """Carry a parity of records backwards through the whole text. Returns the operator at the start and the
    history [(step, operator copy)] when keep is true."""
    ops = parse(text, first_line)
    rec_of, _ = records(ops)
    steps = step_ops(ops)
    O = Operator.identity()
    history = []
    for st in reversed(steps):
        apply_backward(O, st, rec_of, parity, noisy)
        if keep:
            history.append((st, O.copy()))
    return O, history

# ------------------------------------------------------------------ density matrices, for checks


def density_check(text, parity, p, input_state=None):
    """Forward density-matrix simulation of a small circuit at noise strength p: E[(-1)^parity]."""
    ops = parse(text)
    rec_of, nrec = records(ops)
    def qubits_of(name, ts):
        if name == 'MPP':
            return [int(part[1:]) for t in ts for part in t.split('*')]
        return [int(q) for q in ts if q.isdigit()]
    n = 1 + max((q for (_, name, _, ts) in ops for q in qubits_of(name, ts)), default=0)
    dim = 2 ** n

    def op_on(M, qs):
        """Embed a k-qubit matrix on qubits qs (qubit 0 is the most significant tensor factor)."""
        k = len(qs)
        rest = [q for q in range(n) if q not in qs]
        perm = list(qs) + rest
        full = np.kron(M, np.eye(2 ** (n - k)))
        # reorder: full acts on (qs, rest); permute axes back to 0..n-1
        full = full.reshape([2] * (2 * n))
        inv = [perm.index(q) for q in range(n)]
        full = full.transpose(inv + [n + i for i in inv])
        return full.reshape(dim, dim)

    if input_state is None:
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1
        rho = np.outer(psi, psi.conj())
    else:
        rho = input_state
    # branches over records: a dict {record bits tuple: rho}
    branches = {(): rho}
    rec_index = 0

    def pauli_on(word, qs):
        M = PM[word[0]]
        for l in word[1:]:
            M = np.kron(M, PM[l])
        return op_on(M, qs)

    for i, (line, name, arg, targets) in enumerate(ops):
        if name in ('TICK', 'QUBIT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS'):
            continue
        new = {}
        for bits, r in branches.items():
            if name in ('CX', 'CZ'):
                for k in range(0, len(targets), 2):
                    U = op_on(UNITARY[name], [int(targets[k]), int(targets[k + 1])])
                    r = U @ r @ U.conj().T
                new[bits] = r
            elif name in UNITARY:
                for t in targets:
                    U = op_on(UNITARY[name], [int(t)])
                    r = U @ r @ U.conj().T
                new[bits] = r
            elif name in ('DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR'):
                pp = p if arg is not None else 0
                if name == 'DEPOLARIZE1':
                    groups, words = [(int(t),) for t in targets], ['X', 'Y', 'Z']
                    w = pp / 3
                elif name == 'DEPOLARIZE2':
                    groups = [(int(targets[k]), int(targets[k + 1])) for k in range(0, len(targets), 2)]
                    words = [a + b for a in LETTERS for b in LETTERS if a + b != 'II']
                    w = pp / 15
                else:
                    groups, words, w = [(int(t),) for t in targets], [name[0]], pp
                for g in groups:
                    acc = (1 - (w * len(words))) * r
                    for wd in words:
                        P = pauli_on(wd, list(g))
                        acc = acc + w * P @ r @ P
                    r = acc
                new[bits] = r
            elif name in ('M', 'MX', 'MPP'):
                outs = [(bits, r)]
                for t in targets:
                    nxt = []
                    for b, rr in outs:
                        if name == 'MPP':
                            parts = t.split('*')
                            word = ''.join(s[0] for s in parts)
                            qs = [int(s[1:]) for s in parts]
                            P = pauli_on(word, qs)
                        else:
                            P = pauli_on('X' if name == 'MX' else 'Z', [int(t)])
                        for val in (0, 1):
                            proj = (np.eye(dim) + (1 - 2 * val) * P) / 2
                            nxt.append((b + (val,), proj @ rr @ proj))
                    outs = nxt
                    # readout flips
                    if arg is not None:
                        flipped = {}
                        for b, rr in outs:
                            flipped[b] = flipped.get(b, 0) + (1 - p) * rr
                            fb = b[:-1] + (1 - b[-1],)
                            flipped[fb] = flipped.get(fb, 0) + p * rr
                        outs = list(flipped.items())
                for b, rr in outs:
                    new[b] = new.get(b, 0) + rr
            elif name in ('R', 'RX'):
                for t in targets:
                    q = int(t)
                    if name == 'R':
                        kraus = [np.array([[1, 0], [0, 0]], dtype=complex), np.array([[0, 1], [0, 0]], dtype=complex)]
                    else:
                        plus = np.array([1, 1]) / np.sqrt(2)
                        minus = np.array([1, -1]) / np.sqrt(2)
                        kraus = [np.outer(plus, plus).astype(complex), np.outer(plus, minus).astype(complex)]
                    acc = 0
                    for K in kraus:
                        KK = op_on(K, [q])
                        acc = acc + KK @ r @ KK.conj().T
                    r = acc
                new[bits] = r
            else:
                raise ValueError(name)
        branches = new
    total = 0.0
    for bits, r in branches.items():
        par = sum(bits[k - 1] for k in parity) % 2
        total += (1 - 2 * par) * np.trace(r).real
    return total


def expect_zero(O):
    """Expectation on |0...0>: strings of I and Z count with their coefficient, any X or Y gives zero."""
    total = Poly()
    for (x, z), c in O.terms.items():
        if x == 0:
            total = total + c
    return total
