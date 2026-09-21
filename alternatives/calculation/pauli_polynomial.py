"""Pauli-transfer circuit tensors over F_q[x]/(x^(K+1)).

The Pauli basis is Hermitian: P(a,b)=i^(a.b) X^a Z^b.  State coefficients
are Tr(P rho); consequently a final trace selects the identity coefficient.
Each physical channel contributes 1+x times its fault law, i.e. the common
no-fault factor (1-p)^N has been removed. No fault patterns are enumerated.

This compiler reuses the paper's polynomial tensor contractor, but constructs
new circuit tensors. sqrt(2) is embedded in the prime field explicitly. A
modular result alone is not a reconstructed exact rational coefficient.
"""
from functools import lru_cache
from itertools import product
import numpy as np

from calculation.truncated_polynomial import PolynomialNetwork, PolynomialRing
from calculation.native_modular_polynomial import PolynomialField
from calculation.polynomial_binary_contraction import (
    CompiledPolynomialNetwork, memory_plan, contract_polynomial)


def sqrt_mod(a, p):
    """Tonelli--Shanks, with the smaller root chosen deterministically."""
    a %= p
    if not a:
        return 0
    if pow(a, (p-1)//2, p) != 1:
        raise ValueError('The chosen prime does not split this coefficient field')
    q, s = p-1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p-1)//2, p) != p-1:
        z += 1
    c, t, r, m = pow(z, q, p), pow(a, q, p), pow(a, (q+1)//2, p), s
    while t != 1:
        v, j = t, 0
        while v != 1:
            v = v*v % p
            j += 1
        b = pow(c, 1 << (m-j-1), p)
        r, t, c, m = r*b % p, t*b*b % p, b*b % p, j
    assert r*r % p == a
    return min(r, p-r)


def unitary(name, q, root2):
    """Small gate matrices, in little-endian local qubit order."""
    imag = sqrt_mod(q-1, q)
    h = root2*pow(2, -1, q) % q
    w = (1+imag)*h % q
    if name == 'H':
        u = [[h, h], [h, -h]]
    elif name in ('S', 'S_DAG', 'T', 'T_DAG'):
        power = {'S': 2, 'S_DAG': -2, 'T': 1, 'T_DAG': -1}[name]
        u = [[1, 0], [0, pow(w, power, q)]]
    elif name == 'INJECT_HXY':
        u = [[h, h*pow(w, 3, q)], [h*w, h]]
    elif name in ('X', 'Y', 'Z'):
        u = {'X': [[0, 1], [1, 0]], 'Y': [[0, -imag], [imag, 0]],
             'Z': [[1, 0], [0, -1]]}[name]
    elif name in ('CX', 'CSX', 'CS_DAG_X', 'CCZ'):
        n = 3 if name == 'CCZ' else 2
        u = [[0]*(1 << n) for _ in range(1 << n)]
        for b in range(1 << n):
            out, phase = b, 1
            if name == 'CCZ':
                phase = -1 if b == 7 else 1
            elif b & 1:
                out ^= 2
                if not b & 2 and name != 'CX':
                    phase = imag if name == 'CSX' else -imag
            u[out][b] = phase
    else:
        raise ValueError('Unsupported gate: '+name)
    return np.array(u, dtype=object) % q


def pauli_matrix(bits, q):
    n = len(bits)//2
    a = sum(bits[2*j] << j for j in range(n))
    b = sum(bits[2*j+1] << j for j in range(n))
    imag = sqrt_mod(q-1, q)
    phase = pow(imag, (a & b).bit_count(), q)
    result = np.zeros((1 << n, 1 << n), dtype=object)
    for k in range(1 << n):
        result[k ^ a, k] = phase*(-1 if (b & k).bit_count() % 2 else 1) % q
    return result


@lru_cache(maxsize=100)
def transfer(name, q, root2, basis='hermitian'):
    """U P U^dagger = sum_Q transfer[Q,P] Q, entirely modulo q."""
    u = unitary(name, q, root2)
    # Inverse is used rather than complex conjugation of field elements.
    # Gate columns are orthonormal in the cyclotomic field; their field inverse
    # is obtained by ordinary Gaussian elimination, avoiding root conventions.
    n = len(u)
    aug = np.concatenate((u.copy(), np.eye(n, dtype=object)), axis=1)
    for j in range(n):
        pivot = next(i for i in range(j, n) if aug[i, j] % q)
        aug[[j, pivot]] = aug[[pivot, j]]
        aug[j] = aug[j]*pow(int(aug[j, j]), -1, q) % q
        for i in range(n):
            if i != j:
                aug[i] = (aug[i]-aug[i, j]*aug[j]) % q
    inv = aug[:, n:]
    assert np.array_equal(u @ inv % q, np.eye(n, dtype=object))
    qubits = n.bit_length()-1
    bits = list(product((0, 1), repeat=2*qubits))
    matrices = [pauli_matrix(b, q) for b in bits]
    table = np.empty((len(bits), len(bits)), dtype=object)
    inv_n = pow(n, -1, q)
    for j, P in enumerate(matrices):
        evolved = u @ P @ inv % q
        for i, Q in enumerate(matrices):
            table[i, j] = sum(Q[a, b]*evolved[b, a]
                              for a in range(n) for b in range(n))*inv_n % q
    if basis == 'xz':
        imag = sqrt_mod(q-1, q)
        degrees = [sum(b[2*j]*b[2*j+1] for j in range(qubits)) for b in bits]
        for i in range(len(bits)):
            for j in range(len(bits)):
                table[i, j] = table[i, j]*pow(imag, degrees[i]-degrees[j], q) % q
    elif basis != 'hermitian':
        raise ValueError('Unknown operator basis')
    return table.reshape((2,)*(4*qubits))


class CircuitTensors:
    """Local binary Pauli indices, with Clifford wire equalities kept explicit."""
    def __init__(self, ring, root2=None, basis='hermitian'):
        self.ring = ring
        self.root2 = sqrt_mod(2, ring.prime) if root2 is None else root2
        if self.root2*self.root2 % ring.prime != 2:
            raise ValueError('Invalid sqrt(2) embedding')
        self.net = PolynomialNetwork(ring)
        self.basis = basis
        self.wires = {}
        self.channels = 0
        self.event = 0

    def variable(self, label):
        return self.net.variable(f'{self.event}:{label}')

    def wire(self, qubit):
        if qubit not in self.wires:
            # Unmentioned input qubits in the exact state reference start |0>.
            self.wires[qubit] = (None, self.variable(f'z{qubit}:initial'))
        return self.wires[qubit]

    def factor(self, bits, function):
        """None is a fixed zero; repeated variables denote the same index."""
        scope = tuple(sorted({v for v in bits if v is not None}))
        if len(scope) > 12:
            raise RuntimeError('Local source tensor exceeds its small-table guard')
        positions = {v: i for i, v in enumerate(scope)}
        table = [function(tuple(0 if v is None else row[positions[v]] for v in bits))
                 for row in product((0, 1), repeat=len(scope))]
        self.net.add(scope, table)

    def zero(self, bit):
        if bit is not None:
            self.net.fixed(bit)

    def xor(self, a, b, label):
        if a is None:
            return b
        if b is None:
            return a
        if a == b:
            return None
        c = self.variable(label)
        self.net.xor((a, b, c))
        return c

    def gate(self, name, sites):
        self.event += 1
        incoming = [bit for q in sites for bit in self.wire(q)]
        if name in ('R', 'RX'):
            for bit in incoming:
                self.zero(bit)
            v = self.variable(f'reset{sites[0]}')
            self.wires[sites[0]] = (v, None) if name == 'RX' else (None, v)
        elif name == 'POST_Z':
            self.zero(incoming[0])
            # r'_I=r'_Z=(r_I+r_Z)/2. The old z is summed, not fixed.
            self.net.multiply_scalar(self.ring(1)/2)
            self.wires[sites[0]] = (None, self.variable(f'post{sites[0]}'))
        elif name == 'H':
            x, z = incoming
            self.factor((x, z), lambda b: -1 if b[0] & b[1] else 1)
            self.wires[sites[0]] = (z, x)
        elif name in ('S', 'S_DAG'):
            x, z = incoming
            if self.basis == 'xz':
                imag = sqrt_mod(self.ring.prime-1, self.ring.prime)
                phase = imag if name == 'S' else -imag
                self.factor((x,), lambda b: phase if b[0] else 1)
            else:
                self.factor((x, z), lambda b: (-1)**(b[0]*(b[1] ^ (name == 'S_DAG'))))
            self.wires[sites[0]] = (x, self.xor(z, x, 'phase-z'))
        elif name in ('X', 'Y', 'Z'):
            self.factor(incoming, lambda b: (-1)**((b[1] if name in ('X', 'Y') else 0)
                                                   ^ (b[0] if name in ('Z', 'Y') else 0)))
        elif name == 'CX':
            xc, zc, xt, zt = incoming
            if self.basis == 'hermitian':
                self.factor(incoming, lambda b: (-1)**(b[0]*b[3]*(b[2] ^ b[1] ^ 1)))
            self.wires[sites[0]] = (xc, self.xor(zc, zt, 'cx-z'))
            self.wires[sites[1]] = (self.xor(xt, xc, 'cx-x'), zt)
        else:
            outgoing = []
            for j, q in enumerate(sites):
                if name in ('CCZ', 'T', 'T_DAG'):
                    x = incoming[2*j]
                elif name in ('CSX', 'CS_DAG_X'):
                    x = incoming[0] if j == 0 else self.xor(incoming[0], incoming[2], 'csx-x')
                else:
                    x = self.variable(f'{name}-x{q}')
                z = self.variable(f'{name}-z{q}')
                outgoing += [x, z]
                self.wires[q] = (x, z)
            table = transfer(name, self.ring.prime, self.root2, self.basis)
            self.factor(outgoing+incoming, lambda b: int(table[b]))

    def noise(self, channel, *, selected=True, fixed=None, scale=1):
        bits = [b for q in channel['support'] for b in self.wire(q)]
        den = channel['denominator']
        self.channels += bool(selected)
        if fixed is not None:
            if channel['kind'] == 'flip':
                axis = channel.get('axis', 'X')
                error = [int(axis in 'XY'), int(axis in 'YZ')]
            else:
                error = [(fixed >> j) & 1 for j in range(len(bits))]
            self.factor(bits, lambda b: (-1)**(sum(b[j]*error[j ^ 1] for j in range(len(b))) % 2))
        elif selected:
            if channel['kind'] == 'flip':
                axis = channel.get('axis', 'X')
                slope = lambda b: (-1)**((b[0] if axis in 'ZY' else 0) ^ (b[1] if axis in 'XY' else 0))
            else:
                slope = lambda b: -pow(den, -1, self.ring.prime) if any(b) else 1
            self.factor(bits, lambda b: self.ring.coefficients((1, scale*slope(b))))

    def finish(self, logical, observable):
        for q, bits in self.wires.items():
            if q != logical:
                for b in bits:
                    self.zero(b)
        x, z = self.wire(logical)
        if observable == 'A':
            self.zero(x)
            self.zero(z)
        elif observable == 'B':
            h = self.root2*pow(2, -1, self.ring.prime) % self.ring.prime
            # (I - (X+Y)/sqrt(2))/2 on the final decoded logical qubit.
            weights = {(0, 0): self.ring(1)/2, (0, 1): self.ring(0),
                       (1, 0): self.ring(-h)/2, (1, 1): self.ring(-h)/2}
            if self.basis == 'xz':
                imag = sqrt_mod(self.ring.prime-1, self.ring.prime)
                weights[1, 1] = self.ring(imag*h)/2
            self.factor((x, z), lambda b: weights[b])
        elif observable in ('X', 'Y', 'Z'):
            for bit, value in [(x, int(observable in 'XY')), (z, int(observable in 'ZY'))]:
                if bit is None:
                    if value:
                        self.net.multiply_scalar(0)
                else:
                    self.net.fixed(bit, value)
            if self.basis == 'xz' and observable == 'Y':
                self.net.multiply_scalar(-sqrt_mod(self.ring.prime-1, self.ring.prime))
        else:
            raise ValueError('Expected A, B, X, Y or Z')
        return self.net


def compile_puri(model, ring, observable, *, selected=None, fixed=None, root2=None, basis='hermitian'):
    compiler = CircuitTensors(ring, root2, basis)
    for ev in model['events']:
        if ev['name'] == 'NOISE':
            loc = ev['location']
            compiler.noise(model['channels'][loc], selected=selected is None or loc in selected,
                           fixed=None if fixed is None else fixed.get(loc))
        else:
            compiler.gate(ev['name'], ev['support'])
    net = compiler.finish(model['logical'], observable)
    stats = net.simplify()
    stats['noise_locations'] = compiler.channels
    return net, stats


def eliminate_linear_constraints(net, *, max_factor_bits=18, max_source_entries=1_000_000):
    """Solve exact XOR constraints before materialising the remaining tensors.

    A removed variable is substituted as an affine XOR, with no approximation
    and no summation factor. Pivot costs use the supports of the actual remaining
    factors. This is deliberately separate from numerical contraction ordering.
    """
    constant = 1 << len(net.labels)
    variable_mask = constant-1
    equations, factors = [], []
    original_variables = len(net.variables)
    for f in net.factors:
        nonzero = [i for i, value in enumerate(f.values.flat) if value]
        c = f.values.flat[nonzero[0]] if nonzero else net.ring(0)
        parity = nonzero[0].bit_count() % 2 if nonzero else 0
        if (len(nonzero)*2 == f.values.size and all(f.values.flat[i] == c for i in nonzero)
                and all((i.bit_count() % 2 == parity) == bool(v) for i, v in enumerate(f.values.flat))):
            equations.append(sum(1 << v for v in f.scope) ^ (constant if parity else 0))
            net.scalar *= c
        else:
            factors.append(([1 << v for v in f.scope], f.values))
    variables = set(net.variables)
    eliminated = 0
    while equations:
        if constant in equations:
            raise ValueError('Inconsistent affine conditions')
        equations = list(dict.fromkeys(e for e in equations if e))
        if not equations:
            break
        supports = []
        for expressions, _ in factors:
            support = 0
            for e in expressions:
                support |= e & variable_mask
            supports.append(support)
        occurrence = {v: [i for i, s in enumerate(supports) if s >> v & 1] for v in variables}
        best = None
        for eq in equations:
            bits = eq & variable_mask
            while bits:
                bit = bits & -bits
                bits ^= bit
                v = bit.bit_length()-1
                delta, largest = 0, 0
                for i in occurrence[v]:
                    support = 0
                    for e in factors[i][0]:
                        support |= (e ^ eq if e & bit else e) & variable_mask
                    width = support.bit_count()
                    largest = max(largest, width)
                    delta += (1 << width)-(1 << supports[i].bit_count())
                score = (largest, delta, (eq & variable_mask).bit_count(), v, eq)
                if best is None or score < best:
                    best = score
        largest, _, _, v, eq = best
        if largest > max_factor_bits:
            break
        bit = 1 << v
        factors = [([e ^ eq if e & bit else e for e in expressions], table)
                   for expressions, table in factors]
        equations = [e ^ eq if e & bit else e for e in equations]
        variables.remove(v)
        eliminated += 1
    result = PolynomialNetwork(net.ring)
    result.labels = list(net.labels)
    result.variables = variables
    result.scalar = net.scalar
    source_entries = 0
    for expressions, table in factors:
        support = 0
        for e in expressions:
            support |= e & variable_mask
        scope = tuple(i for i in range(len(net.labels)) if support >> i & 1)
        source_entries += 1 << len(scope)
        if len(scope) > max_factor_bits or source_entries > max_source_entries:
            raise RuntimeError('Affine substitution exceeds the source-array guard')
        vals = np.arange(1 << len(scope), dtype=np.uint64)
        indices = []
        for e in expressions:
            mask = sum(1 << (len(scope)-1-j) for j, v in enumerate(scope) if e >> v & 1)
            indices.append((np.bitwise_count(vals & np.uint64(mask)) % 2).astype(np.intp) ^ int(bool(e & constant)))
        result.add(scope, table[tuple(indices)] if expressions else table)
    for eq in equations:
        scope = tuple(v for v in sorted(variables) if eq >> v & 1)
        if len(scope) > max_factor_bits:
            raise RuntimeError('Remaining parity equation exceeds source guard')
        result.xor(scope, int(bool(eq & constant)))
    cleanup = result.simplify()
    return result, dict(original_variables=original_variables, eliminated=eliminated,
                        residual_equations=len(equations), expanded_source_entries=source_entries,
                        cleanup=cleanup)


def evaluate(net, *, build_dir, max_bytes=1024*2**20, max_work=300_000_000, plan=None):
    plans = [plan] if plan is not None else [net.plan(strategy=s) for s in ('min_fill', 'min_degree')]
    scored = [(memory_plan(net, p), p) for p in plans]
    sizing, plan = min(scored, key=lambda item: item[0]['elimination_coefficient_multiplications'])
    if (plan['maximum_product_scope_bits'] > 26
            or sizing['maximum_estimated_live_array_bytes'] > max_bytes
            or sizing['elimination_coefficient_multiplications'] > max_work):
        return None, dict(status='guarded', memory=sizing, plan=plan)
    field = PolynomialField(net.ring, build_dir)
    value, profile = contract_polynomial(CompiledPolynomialNetwork(net, field), plan,
                                         max_live_bytes=max_bytes)
    return list(map(int, value)), dict(status='computed_modulo_one_prime', profile=profile, plan=plan)


class DensityTensors(CircuitTensors):
    """The same polynomial circuit in the computational operator basis.

    Wires label |ket><bra|. This exact basis change removes Pauli branching
    from diagonal CCZ gates, at the expense of branching in depolarisation.
    A selector decomposes a depolarising channel into identity and trace/replace
    terms; its signed weights are polynomial coefficients, not probabilities.
    """
    def wire(self, qubit):
        if qubit not in self.wires:
            self.wires[qubit] = (None, None)
        return self.wires[qubit]

    def equal(self, a, b):
        if a == b:
            return
        if a is None:
            self.zero(b)
        elif b is None:
            self.zero(a)
        else:
            self.net.xor((a, b))

    def gate(self, name, sites):
        self.event += 1
        old = [self.wire(q) for q in sites]
        if name in ('R', 'RX'):
            self.equal(*old[0])
            if name == 'R':
                self.wires[sites[0]] = (None, None)
            else:
                self.wires[sites[0]] = tuple(self.variable(f'RX-{s}') for s in ('ket', 'bra'))
                self.net.multiply_scalar(self.ring(1)/2)
        elif name == 'POST_Z':
            for b in old[0]:
                self.zero(b)
            self.wires[sites[0]] = (None, None)
        elif name == 'CX':
            self.wires[sites[1]] = tuple(self.xor(old[0][j], old[1][j], f'CX-{j}') for j in range(2))
        elif name == 'CCZ':
            for j in range(2):
                self.factor([w[j] for w in old], lambda b: -1 if all(b) else 1)
        elif name in ('CSX', 'CS_DAG_X'):
            imag = sqrt_mod(self.ring.prime-1, self.ring.prime)
            for j in range(2):
                phase = imag * (-1 if (name == 'CS_DAG_X') ^ bool(j) else 1)
                self.factor([w[j] for w in old], lambda b, phase=phase: phase if b[0] and not b[1] else 1)
            self.wires[sites[1]] = tuple(self.xor(old[0][j], old[1][j], f'CSX-{j}') for j in range(2))
        elif name in ('S', 'S_DAG', 'T', 'T_DAG', 'Z'):
            q = self.ring.prime
            imag = sqrt_mod(q-1, q)
            omega = (1+imag)*self.root2*pow(2, -1, q) % q
            power = {'S': 2, 'S_DAG': -2, 'T': 1, 'T_DAG': -1, 'Z': 4}[name]
            self.factor(old[0], lambda b: pow(omega, power*(b[0]-b[1]), q))
        elif name in ('X', 'Y'):
            bit = self.variable('one')
            self.net.fixed(bit, 1)
            if name == 'Y':
                self.factor(old[0], lambda b: (-1)**(b[0] ^ b[1]))
            self.wires[sites[0]] = tuple(self.xor(b, bit, name) for b in old[0])
        elif name in ('H', 'INJECT_HXY'):
            q = self.ring.prime
            u = unitary(name, q, self.root2)
            # Complex conjugation is i -> -i, sqrt(2) fixed.
            if name == 'H':
                conjugate = u
            else:
                imag = sqrt_mod(q-1, q)
                h = self.root2*pow(2, -1, q) % q
                w = (1-imag)*h % q
                conjugate = np.array([[h, h*pow(w, 3, q)], [h*w, h]], dtype=object) % q
            new = []
            for j in range(2):
                out = self.variable(f'{name}-{j}')
                matrix = u if j == 0 else conjugate
                self.factor((out, old[0][j]), lambda b, matrix=matrix: int(matrix[b]))
                new.append(out)
            self.wires[sites[0]] = tuple(new)
        else:
            raise ValueError('Unsupported density gate: '+name)

    def noise(self, channel, *, selected=True, fixed=None):
        if not selected and fixed is None:
            return
        sites = channel['support']
        old = [self.wire(q) for q in sites]
        self.channels += bool(selected)
        if fixed is not None or channel['kind'] == 'flip':
            bit = self.variable('pauli-fault-selector')
            if fixed is not None:
                self.net.fixed(bit, 1)
            else:
                self.net.add((bit,), [1, self.ring.x])
            for j, q in enumerate(sites):
                if channel['kind'] == 'flip':
                    axis = channel.get('axis', 'X')
                    x, z = int(axis in 'XY'), int(axis in 'YZ')
                else:
                    x, z = (fixed >> (2*j)) & 1, (fixed >> (2*j+1)) & 1
                if z:
                    self.factor((bit, *old[j]), lambda b: (-1)**(b[0]*(b[1] ^ b[2])))
                if x:
                    self.wires[q] = tuple(self.xor(v, bit, 'noise-X') for v in old[j])
        else:
            den, n = channel['denominator'], len(sites)
            selector = self.variable('trace-replace-selector')
            self.net.add((selector,), [1-self.ring.x/den, self.ring.x*(1 << n)/den])
            for q, (ki, bi) in zip(sites, old):
                ko, bo = self.variable('noise-ket'), self.variable('noise-bra')
                self.factor((selector, ko, bo, ki, bi),
                            lambda b: int((b[1] == b[3] and b[2] == b[4]) if b[0] == 0
                                          else (b[1] == b[2] and b[3] == b[4])))
                self.wires[q] = (ko, bo)

    def finish(self, logical, observable):
        for q, bits in self.wires.items():
            if q != logical:
                self.equal(*bits)
        k, b = self.wire(logical)
        if observable == 'A':
            self.equal(k, b)
        elif observable == 'B':
            q = self.ring.prime
            imag = sqrt_mod(q-1, q)
            w = (1+imag)*self.root2*pow(2, -1, q) % q
            self.factor((k, b), lambda v: self.ring(1 if v[0] == v[1]
                                                   else -pow(w, v[1]-v[0], q))/2)
        else:
            raise ValueError('Expected A or B')
        return self.net


class MixedDensityTensors(DensityTensors):
    """Use |k><k XOR d|: diagonal gates and Pauli noise both preserve d.

    This is the partial Fourier/mixed representation relevant to the earlier
    paper. Summing physical Z errors locally gives exact dephasing factors;
    physical X errors move k. No signed coefficient is interpreted as a
    stochastic branch probability.
    """
    def gate(self, name, sites):
        self.event += 1
        old = [self.wire(q) for q in sites]
        q = self.ring.prime
        imag = sqrt_mod(q-1, q)
        w = (1+imag)*self.root2*pow(2, -1, q) % q
        if name in ('R', 'RX'):
            self.zero(old[0][1])  # trace input: d=0, k summed
            if name == 'R':
                self.wires[sites[0]] = (None, None)
            else:
                self.wires[sites[0]] = tuple(self.variable(f'RX-{s}') for s in ('k', 'd'))
                self.net.multiply_scalar(self.ring(1)/2)
        elif name == 'POST_Z':
            for b in old[0]:
                self.zero(b)
            self.wires[sites[0]] = (None, None)
        elif name == 'CX':
            self.wires[sites[1]] = tuple(self.xor(old[0][j], old[1][j], f'CX-{j}') for j in range(2))
        elif name == 'CCZ':
            self.factor([b for pair in old for b in pair],
                        lambda b: (-1)**((b[0]*b[2]*b[4]) ^ ((b[0]^b[1])*(b[2]^b[3])*(b[4]^b[5]))))
        elif name in ('CSX', 'CS_DAG_X'):
            phase = imag if name == 'CSX' else -imag
            self.factor([*old[0], *old[1]],
                        lambda b: pow(phase, b[0]*(1-b[2])-(b[0]^b[1])*(1-(b[2]^b[3])), q))
            self.wires[sites[1]] = tuple(self.xor(old[0][j], old[1][j], f'CSX-{j}') for j in range(2))
        elif name in ('S', 'S_DAG', 'T', 'T_DAG', 'Z'):
            power = {'S': 2, 'S_DAG': -2, 'T': 1, 'T_DAG': -1, 'Z': 4}[name]
            self.factor(old[0], lambda b: pow(w, power*(b[0]-(b[0]^b[1])), q))
        elif name in ('X', 'Y'):
            if name == 'Y':
                self.factor((old[0][1],), lambda b: (-1)**b[0])
            bit = self.variable('one')
            self.net.fixed(bit, 1)
            self.wires[sites[0]] = (self.xor(old[0][0], bit, name), old[0][1])
        elif name in ('H', 'INJECT_HXY'):
            new = tuple(self.variable(f'{name}-{s}') for s in ('k', 'd'))
            if name == 'H':
                self.factor((*new, *old[0]), lambda b: self.ring((-1)**((b[0]*b[3]) ^ (b[1]*b[2]) ^ (b[1]*b[3])))/2)
            else:
                u = unitary(name, q, self.root2)
                h = self.root2*pow(2, -1, q) % q
                wc = (1-imag)*h % q
                uc = np.array([[h, h*pow(wc, 3, q)], [h*wc, h]], dtype=object) % q
                self.factor((*new, *old[0]), lambda b: int(u[b[0], b[2]]*uc[b[0]^b[1], b[2]^b[3]] % q))
            self.wires[sites[0]] = new
        else:
            raise ValueError('Unsupported mixed gate: '+name)

    def noise(self, channel, *, selected=True, fixed=None):
        if not selected and fixed is None:
            return
        sites = channel['support']
        old = [self.wire(q) for q in sites]
        self.channels += bool(selected)
        if fixed is not None or channel['kind'] == 'flip':
            bit = self.variable('pauli-fault-selector')
            if fixed is not None:
                self.net.fixed(bit, 1)
            else:
                self.net.add((bit,), [1, self.ring.x])
            for j, site in enumerate(sites):
                if channel['kind'] == 'flip':
                    axis = channel.get('axis', 'X')
                    x, z = int(axis in 'XY'), int(axis in 'YZ')
                else:
                    x, z = (fixed >> (2*j)) & 1, (fixed >> (2*j+1)) & 1
                if z:
                    self.factor((bit, old[j][1]), lambda b: (-1)**(b[0]*b[1]))
                if x:
                    self.wires[site] = (self.xor(old[j][0], bit, 'noise-X'), old[j][1])
        else:
            n, den = len(sites), channel['denominator']
            outputs = [self.variable(f'noise-k{site}') for site in sites]
            # For translation a=k_out XOR k_in, summing all Z labels gives
            # sum_nonidentity (-1)^(z.d) = 2^n [d=0] - [a=0].
            self.factor([p[1] for p in old]+[p[0] for p in old]+outputs,
                        lambda b: self.ring(int(b[n:2*n] == b[2*n:]))
                        + self.ring.x*((1 << n)*int(not any(b[:n]))
                                       - int(b[n:2*n] == b[2*n:]))/den)
            for site, old_pair, ko in zip(sites, old, outputs):
                self.wires[site] = (ko, old_pair[1])

    def finish(self, logical, observable):
        for site, (k, d) in self.wires.items():
            if site != logical:
                self.zero(d)
        k, d = self.wire(logical)
        if observable == 'A':
            self.zero(d)
        elif observable == 'B':
            q = self.ring.prime
            w = (1+sqrt_mod(q-1, q))*self.root2*pow(2, -1, q) % q
            self.factor((k, d), lambda b: self.ring(1 if b[1] == 0
                                                   else -pow(w, (b[0]^b[1])-b[0], q))/2)
        else:
            raise ValueError('Expected A or B')
        return self.net


def compile_density(model, ring, observable, *, selected=None, fixed=None, root2=None, mixed=False):
    compiler = (MixedDensityTensors if mixed else DensityTensors)(ring, root2)
    for ev in model['events']:
        if ev['name'] == 'NOISE':
            loc = ev['location']
            compiler.noise(model['channels'][loc], selected=selected is None or loc in selected,
                           fixed=None if fixed is None else fixed.get(loc))
        else:
            compiler.gate(ev['name'], ev['support'])
    net = compiler.finish(model['logical'], observable)
    stats = net.simplify()
    stats['noise_locations'] = compiler.channels
    return net, stats
