"""Rational source factors; bounded exact elimination after reduction modulo q.

The planner sees scopes only. No binary64 value is accepted as a coefficient.
This reference contractor deliberately uses Python integers and small arrays;
larger endpoint pilots must select a separately checked native contractor.
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import time
import numpy as np
from .binary_factors import Network as ScopePlanner


def rational(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    raise TypeError('Exact source factors accept integers/Fractions, never floats')


def residue(value, prime):
    value = rational(value)
    return value.numerator % prime * pow(value.denominator, -1, prime) % prime


@dataclass
class Factor:
    scope: tuple
    values: np.ndarray

    def __post_init__(self):
        # NumPy object-array indexing can return a bare Fraction at rank zero.
        self.values = np.asarray(self.values, dtype=object).reshape((2,)*len(self.scope))


class RationalNetwork:
    plan = ScopePlanner.plan  # inspected scope-only graph planner

    def __init__(self):
        self.labels, self.factors, self.variables = [], [], set()
        self.scalar = Fraction(1)

    def variable(self, label):
        result = len(self.labels)
        self.labels.append(label)
        self.variables.add(result)
        return result

    def multiply_scalar(self, value):
        self.scalar *= rational(value)

    def add(self, scope, values):
        scope = tuple(int(v) for v in scope)
        if len(set(scope)) != len(scope) or not set(scope) <= self.variables:
            raise ValueError('Invalid factor scope')
        source = np.asarray(values, dtype=object)
        array = np.array([rational(v) for v in source.flat], dtype=object).reshape((2,)*len(scope))
        order = np.argsort(scope)
        if scope:
            array = np.transpose(array, order)
            scope = tuple(scope[i] for i in order)
        self.factors.append(Factor(scope, array))

    def fixed(self, variable, value=0):
        if value not in (0, 1):
            raise ValueError('Nonbinary fixed value')
        self.add((variable,), (1, 0) if value == 0 else (0, 1))

    def xor(self, scope, parity=0):
        if parity not in (0, 1):
            raise ValueError('Nonbinary parity')
        shape = (2,)*len(scope)
        values = (np.sum(np.indices(shape), axis=0) % 2 == parity).astype(np.int8)
        self.add(scope, values)

    def condition(self, assignment):
        if not set(assignment) <= self.variables or any(v not in (0, 1) for v in assignment.values()):
            raise ValueError('Invalid binary conditioning')
        result = RationalNetwork()
        result.labels = list(self.labels)
        result.variables = self.variables - set(assignment)
        result.scalar = self.scalar
        for factor in self.factors:
            scope = tuple(v for v in factor.scope if v not in assignment)
            values = factor.values[tuple(assignment.get(v, slice(None)) for v in factor.scope)]
            if scope:
                result.factors.append(Factor(scope, values))
            else:
                result.scalar *= values.item() if isinstance(values, np.ndarray) else values
        return result

    def simplify(self, protected=()):
        """Exact constant/fixed/equality removal with all outputs protected."""
        protected = set(protected)
        if not protected <= self.variables:
            raise ValueError('Unknown protected index')
        original = (len(self.variables), len(self.factors))
        counts = dict(fixed_variables=0, merged_equalities=0, removed_constants=0)
        identity = np.array([[1, 0], [0, 1]], dtype=object)
        while True:
            changed = False
            for index, factor in enumerate(self.factors):
                scope, values = factor.scope, factor.values
                if np.all(values == values.flat[0]):
                    self.scalar *= values.flat[0]
                    self.factors.pop(index)
                    counts['removed_constants'] += 1
                    changed = True
                    break
                if len(scope) == 1 and np.count_nonzero(values) == 1 and scope[0] not in protected:
                    bit = int(np.flatnonzero(values)[0])
                    variable = scope[0]
                    self.scalar *= values[bit]
                    self.factors.pop(index)
                    self.factors = [Factor(tuple(v for v in f.scope if v != variable),
                                           np.take(f.values, bit, axis=f.scope.index(variable)))
                                    if variable in f.scope else f for f in self.factors]
                    self.variables.remove(variable)
                    counts['fixed_variables'] += 1
                    changed = True
                    break
                if len(scope) == 2 and (np.array_equal(values, identity) or np.array_equal(values, identity[:, ::-1])):
                    if set(scope) <= protected:
                        continue
                    keep, remove = scope
                    if remove in protected:
                        keep, remove = remove, keep
                    offset = int(values[0, 0] == 0)
                    self.factors.pop(index)
                    updated = []
                    for other in self.factors:
                        if remove not in other.scope:
                            updated.append(other)
                            continue
                        replacement = tuple(keep if v == remove else v for v in other.scope)
                        unique = tuple(sorted(set(replacement)))
                        grids = [np.arange(2).reshape((1,)*i+(2,)+(1,)*(len(unique)-i-1)) for i in range(len(unique))]
                        array = other.values[tuple(grids[unique.index(v)] ^ (offset if old == remove else 0)
                                                   for old, v in zip(other.scope, replacement))]
                        updated.append(Factor(unique, array))
                    self.factors = updated
                    self.variables.remove(remove)
                    counts['merged_equalities'] += 1
                    changed = True
                    break
            if not changed:
                break
        groups = {}
        for f in self.factors:
            if f.scope in groups:
                groups[f.scope] *= f.values
            else:
                groups[f.scope] = f.values.copy()
        self.factors = [Factor(scope, values) for scope, values in groups.items()]
        return {'original_variables': original[0], 'original_factors': original[1],
                'remaining_variables': len(self.variables), 'remaining_factors': len(self.factors), **counts}


def contract_reference(network, plan, prime, *, output_variables=(), max_entries=1 << 20):
    """Small Python-integer reference, with modular reduction after each op."""
    outputs = tuple(sorted(set(output_variables)))
    if list(outputs) != plan['output_variables'] or not set(outputs) <= network.variables:
        raise ValueError('Plan/output mismatch')
    if set(plan['order']) != network.variables-set(outputs) or len(set(plan['order'])) != len(plan['order']):
        raise ValueError('Plan does not cover every internal index')
    if not 0 < max_entries <= 1 << 20 or plan['maximum_product_entries'] > max_entries:
        raise ValueError('Small exact reference entry budget exceeded')
    began = time.perf_counter()
    factors = [Factor(f.scope, np.array([residue(v, prime) for v in f.values.flat], dtype=object).reshape(f.values.shape))
               for f in network.factors]
    scalar = residue(network.scalar, prime)
    for variable in plan['order']:
        bucket = [f for f in factors if variable in f.scope]
        factors = [f for f in factors if variable not in f.scope]
        if not bucket:
            scalar = 2*scalar % prime
            continue
        joined = tuple(sorted(set().union(*(set(f.scope) for f in bucket))))
        if 1 << len(joined) > max_entries:
            raise ValueError('Runtime reference allocation cap exceeded')
        product = np.ones((2,)*len(joined), dtype=object)
        for f in bucket:
            product = product*f.values.reshape(tuple(2 if v in f.scope else 1 for v in joined)) % prime
        output = np.sum(product, axis=joined.index(variable)) % prime
        scope = tuple(v for v in joined if v != variable)
        if scope:
            factors.append(Factor(scope, output))
        else:
            scalar = scalar*int(output) % prime
        del bucket, product, output
    result = np.full((2,)*len(outputs), scalar, dtype=object)
    for f in factors:
        if not set(f.scope) <= set(outputs):
            raise ValueError('Internal factor remains')
        result = result*f.values.reshape(tuple(2 if v in f.scope else 1 for v in outputs)) % prime
    return result, {'seconds': time.perf_counter()-began, 'arithmetic': 'Python integers modulo q',
                    'maximum_product_entries': plan['maximum_product_entries']}
