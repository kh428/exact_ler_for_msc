"""Degree-aware exact binary elimination, with explicit polynomial memory planning."""
from dataclasses import dataclass
import time
import numpy as np
from .modular_binary_contraction import ModularWorkspace


@dataclass
class PolynomialFactor:
    scope: tuple
    values: np.ndarray
    buffer: object = None

    @property
    def degree(self):
        return self.values.shape[-1]-1


class CompiledPolynomialNetwork:
    def __init__(self, network, field):
        if network.ring is not field.ring:
            raise ValueError('Polynomial source and native field disagree')
        self.field, self.ring = field, field.ring
        self.labels, self.variables = list(network.labels), set(network.variables)
        self.scalar, self.factors = network.scalar, []
        for f in network.factors:
            degree = max(v.degree for v in f.values.flat)
            values = np.zeros((1 << len(f.scope), degree+1), dtype=np.uint64)
            for row, value in zip(values, f.values.flat):
                row[:len(value.coefficients)] = [v*field.one % field.prime for v in value.coefficients]
            self.factors.append(PolynomialFactor(f.scope, values.reshape((2,)*len(f.scope)+(degree+1,))))

    def condition(self, assignment):
        if not set(assignment) <= self.variables or any(v not in (0, 1) for v in assignment.values()):
            raise ValueError('Invalid compiled polynomial conditioning')
        result = object.__new__(CompiledPolynomialNetwork)
        result.field, result.ring = self.field, self.ring
        result.labels, result.variables = self.labels, self.variables-set(assignment)
        result.scalar, result.factors = self.scalar, []
        for f in self.factors:
            scope = tuple(v for v in f.scope if v not in assignment)
            values = f.values[tuple(assignment.get(v, slice(None)) for v in f.scope)]
            if scope:
                result.factors.append(PolynomialFactor(scope, np.ascontiguousarray(values)))
            else:
                result.scalar *= self.field.decode_polynomial(values)
        return result


def memory_plan(network, plan, output_variables=()):
    """Mirror the actual scratch/pool schedule without allocating large arrays."""
    outputs = tuple(sorted(output_variables))
    if list(outputs) != plan['output_variables'] or set(plan['order']) != network.variables-set(outputs):
        raise ValueError('Polynomial memory plan mismatch')
    K = network.ring.degree
    # A source network or a compiled network can both be inspected here.
    def degree(f):
        return f.degree if isinstance(f, PolynomialFactor) else max(v.degree for v in f.values.flat)
    source = [(f.scope, degree(f), None) for f in network.factors]
    resident = sum((1 << len(scope))*(d+1)*8 for scope, d, _ in source)
    scratch_entries = max([1 << s['output_scope_bits'] for s in plan['steps']]+[1 << len(outputs)])
    allocated = scratch_entries*(K+1)*8
    free, multiplications, additions = {}, 0, 0
    for v in plan['order']:
        bucket = [f for f in source if v in f[0]]
        source = [f for f in source if v not in f[0]]
        if not bucket:
            continue
        scope = tuple(sorted(set().union(*(set(f[0]) for f in bucket))-{v}))
        degree_out = min(K, sum(d for _, d, _ in bucket))
        size = (1 << len(scope))*(degree_out+1)
        if free.get(size, 0):
            free[size] -= 1
        else:
            allocated += size*8
        active = 0
        for _, d, _ in sorted(bucket, key=lambda f: f[1]):
            new = min(K, active+d)
            terms = sum(min(d, k)-max(0, k-active)+1 for k in range(new+1))
            multiplications += 2*(1 << len(scope))*terms
            active = new
        additions += size
        for _, _, buffer in bucket:
            if buffer is not None:
                free[buffer] = free.get(buffer, 0)+1
        if scope:
            source.append((scope, degree_out, size))
        else:
            free[size] = free.get(size, 0)+1
    output_bytes = (1 << len(outputs))*(K+1)*8
    return {'degree': K, 'scratch_entries': scratch_entries, 'scratch_bytes': scratch_entries*(K+1)*8,
            'pooled_array_bytes': allocated, 'source_array_bytes': resident,
            'maximum_estimated_live_array_bytes': allocated+2*resident+output_bytes,
            'elimination_coefficient_multiplications': multiplications,
            'elimination_branch_additions': additions,
            'maximum_binary_product_bits': plan['maximum_product_scope_bits'],
            'assumptions': 'Exact pool schedule; two source copies and result included. Python/source-object overhead and allocator RSS measured separately.'}


def contract_polynomial(network, plan, *, output_variables=(), max_live_bytes=3400*2**20,
                        workspace=None, progress=None):
    outputs = tuple(sorted(output_variables))
    forecast = memory_plan(network, plan, outputs)
    if forecast['maximum_estimated_live_array_bytes'] > max_live_bytes or plan['maximum_product_scope_bits'] > 26:
        raise ValueError('Polynomial contraction rejected before allocation by its memory/index plan')
    if len(plan['order']) != len(set(plan['order'])):
        raise ValueError('Repeated eliminated variable')
    field, ring, K = network.field, network.ring, network.ring.degree
    scalar = network.scalar
    factors = list(network.factors)
    for f in list(factors):
        if not f.scope:
            scalar *= field.decode_polynomial(f.values)
            factors.remove(f)
    budget = max_live_bytes-2*forecast['source_array_bytes']-(1 << len(outputs))*(K+1)*8
    workspace = ModularWorkspace() if workspace is None else workspace
    workspace.prepare(forecast['scratch_entries']*(K+1), budget)
    began = last = time.perf_counter()
    for index, variable in enumerate(plan['order']):
        bucket = sorted((f for f in factors if variable in f.scope), key=lambda f: f.degree)
        factors = [f for f in factors if variable not in f.scope]
        if not bucket:
            scalar *= 2
            continue
        joined = tuple(sorted(set().union(*(set(f.scope) for f in bucket))))
        scope = tuple(v for v in joined if v != variable)
        degree = min(K, sum(f.degree for f in bucket))
        entries = (1 << len(scope))*(degree+1)
        buffer = workspace.acquire(entries, budget)
        output = buffer.reshape((2,)*len(scope)+(degree+1,))
        temporary = workspace.scratch[:entries].reshape(output.shape)
        for bit in (0, 1):
            product = output if bit == 0 else temporary
            product.fill(0)
            product[..., 0] = field.one
            active = 0
            for f in bucket:
                active = field.broadcast(product, f.values, f.scope, scope, active, {variable: bit})
        field.add(output, temporary)
        for f in bucket:
            if f.buffer is not None:
                workspace.release(f.buffer)
        if scope:
            factors.append(PolynomialFactor(scope, output, buffer))
        else:
            scalar *= field.decode_polynomial(output)
            workspace.release(buffer)
        if progress and time.perf_counter()-last > 4:
            progress({'event': 'polynomial_elimination', 'step': index, 'steps': len(plan['order']),
                      'seconds': time.perf_counter()-began, 'pooled_bytes': workspace.allocated_bytes})
            last = time.perf_counter()
        del bucket, output, temporary, product
    result = workspace.scratch[:(1 << len(outputs))*(K+1)].reshape((2,)*len(outputs)+(K+1,))
    result[...] = field.encode_polynomial(scalar, K)
    active = scalar.degree
    for f in sorted(factors, key=lambda f: f.degree):
        if not set(f.scope) <= set(outputs):
            raise ValueError('Polynomial internal factor remains')
        active = field.broadcast(result, f.values, f.scope, outputs, active)
        if f.buffer is not None:
            workspace.release(f.buffer)
    result = result.copy()
    field.convert(result, encode=False)
    if workspace.allocated_bytes > forecast['pooled_array_bytes']:
        raise ValueError('Polynomial runtime allocation exceeded the pool forecast')
    return result, {'seconds': time.perf_counter()-began, 'pooled_array_bytes': workspace.allocated_bytes,
                    'memory_plan': forecast, 'kernel': field.evidence,
                    'arithmetic': 'All retained coefficients exact modulo q; Cauchy products truncated only above K'}
