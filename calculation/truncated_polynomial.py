"""Exact F_q[x]/(x^(K+1)) source coefficients; never accept floating inputs."""
from fractions import Fraction
from numbers import Integral
import numpy as np
from .rational_binary_network import RationalNetwork, Factor, residue


class PolynomialRing:
    def __init__(self, prime, degree):
        if not (2 < prime < 2**62 and prime % 2 and 0 <= degree <= 10):
            raise ValueError('Expected an odd field modulus and degree at most ten')
        self.prime, self.degree = int(prime), int(degree)

    def __call__(self, value):
        if isinstance(value, Polynomial):
            if value.ring is not self:
                raise ValueError('Different polynomial rings')
            return value
        if not isinstance(value, (Integral, Fraction)):
            raise TypeError('Polynomial scalars accept integers/Fractions, never floats')
        return self.coefficients((residue(value, self.prime),))

    def coefficients(self, values):
        values = tuple(int(v) % self.prime for v in values)[:self.degree+1]
        while len(values) > 1 and not values[-1]:
            values = values[:-1]
        return Polynomial(self, values or (0,))

    @property
    def x(self):
        return self.coefficients((0, 1))


class Polynomial:
    __slots__ = ('ring', 'coefficients')

    def __init__(self, ring, coefficients):
        self.ring, self.coefficients = ring, coefficients

    @property
    def degree(self):
        return len(self.coefficients)-1

    def __bool__(self):
        return any(self.coefficients)

    def __eq__(self, other):
        if isinstance(other, (Integral, Fraction)):
            other = self.ring(other)
        return (isinstance(other, Polynomial) and self.ring is other.ring
                and self.coefficients == other.coefficients)

    def __hash__(self):
        return hash((id(self.ring), self.coefficients))

    def __add__(self, other):
        other = self.ring(other)
        a, b = self.coefficients, other.coefficients
        return self.ring.coefficients((a[i] if i < len(a) else 0)+(b[i] if i < len(b) else 0)
                                      for i in range(max(len(a), len(b))))

    __radd__ = __add__

    def __neg__(self):
        return self.ring.coefficients(-v for v in self.coefficients)

    def __sub__(self, other):
        return self + -self.ring(other)

    def __rsub__(self, other):
        return self.ring(other) + -self

    def __mul__(self, other):
        other = self.ring(other)
        a, b = self.coefficients, other.coefficients
        output = [0]*(min(self.ring.degree, self.degree+other.degree)+1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b[:len(output)-i]):
                    output[i+j] += ai*bj
        return self.ring.coefficients(output)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.ring(other)
        if other.degree or not other:
            raise ValueError('Only nonzero constant division is used in source factors')
        return self*pow(other.coefficients[0], -1, self.ring.prime)

    def __repr__(self):
        return 'Poly'+repr(self.coefficients)


class PolynomialNetwork(RationalNetwork):
    """Reuse scope planning and exact structural simplification, with ring scalars.

    Equality/zero tests inspect every retained coefficient. In particular a
    factor that vanishes at x=0 is not removed if its higher coefficients live.
    """
    def __init__(self, ring):
        super().__init__()
        self.ring = ring
        self.scalar = ring(1)

    def multiply_scalar(self, value):
        self.scalar *= self.ring(value)

    def add(self, scope, values):
        scope = tuple(int(v) for v in scope)
        if len(set(scope)) != len(scope) or not set(scope) <= self.variables:
            raise ValueError('Invalid polynomial factor scope')
        source = np.asarray(values, dtype=object)
        array = np.array([self.ring(v) for v in source.flat], dtype=object).reshape((2,)*len(scope))
        if scope:
            order = np.argsort(scope)
            array = array.transpose(order)
            scope = tuple(scope[i] for i in order)
        self.factors.append(Factor(scope, array))

    def condition(self, assignment):
        if not set(assignment) <= self.variables or any(v not in (0, 1) for v in assignment.values()):
            raise ValueError('Invalid polynomial binary conditioning')
        result = PolynomialNetwork(self.ring)
        result.labels = list(self.labels)
        result.variables = self.variables-set(assignment)
        result.scalar = self.scalar
        for f in self.factors:
            scope = tuple(v for v in f.scope if v not in assignment)
            values = f.values[tuple(assignment.get(v, slice(None)) for v in f.scope)]
            if scope:
                result.factors.append(Factor(scope, values))
            else:
                result.scalar *= values.item() if isinstance(values, np.ndarray) else values
        return result


def contract_polynomial_reference(network, plan, *, output_variables=(), max_entries=2**16):
    """Small independent Python-object elimination; no native scalar kernel."""
    outputs = tuple(sorted(output_variables))
    if list(outputs) != plan['output_variables'] or set(plan['order']) != network.variables-set(outputs):
        raise ValueError('Polynomial reference plan mismatch')
    if plan['maximum_product_entries'] > max_entries:
        raise ValueError('Polynomial reference exceeds its small array cap')
    ring = network.ring
    factors, scalar = list(network.factors), network.scalar
    for v in plan['order']:
        bucket = [f for f in factors if v in f.scope]
        factors = [f for f in factors if v not in f.scope]
        if not bucket:
            scalar *= 2
            continue
        joined = tuple(sorted(set().union(*(set(f.scope) for f in bucket))))
        product = np.full((2,)*len(joined), ring(1), dtype=object)
        for f in bucket:
            product *= f.values.reshape(tuple(2 if x in f.scope else 1 for x in joined))
        value = product.sum(axis=joined.index(v))
        scope = tuple(x for x in joined if x != v)
        if scope:
            factors.append(Factor(scope, value))
        else:
            scalar *= value.item() if isinstance(value, np.ndarray) else value
    result = np.full((2,)*len(outputs), scalar, dtype=object)
    for f in factors:
        result *= f.values.reshape(tuple(2 if v in f.scope else 1 for v in outputs))
    return result
