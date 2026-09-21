"""Exact modular elimination, retaining both branches of every binary index."""
from dataclasses import dataclass
import time
import numpy as np
from .rational_binary_network import residue


@dataclass
class ModularFactor:
    scope: tuple
    values: np.ndarray
    buffer: object = None


class ModularNetwork:
    def __init__(self, rational_network, field):
        self.field = field
        self.labels = list(rational_network.labels)
        self.variables = set(rational_network.variables)
        self.scalar = residue(rational_network.scalar, field.prime)*field.one % field.prime
        cache = {}
        self.factors = []
        for f in rational_network.factors:
            for value in f.values.flat:
                if value not in cache:
                    cache[value] = residue(value, field.prime)*field.one % field.prime
            values = np.array([cache[v] for v in f.values.flat], dtype=np.uint64).reshape(f.values.shape)
            self.factors.append(ModularFactor(f.scope, values))

    def condition(self, assignment):
        if not set(assignment) <= self.variables or any(bit not in (0, 1) for bit in assignment.values()):
            raise ValueError('Invalid modular-network conditioning')
        result = object.__new__(ModularNetwork)
        result.field, result.labels = self.field, list(self.labels)
        result.variables = self.variables - set(assignment)
        result.scalar = self.scalar
        result.factors = []
        for f in self.factors:
            scope = tuple(v for v in f.scope if v not in assignment)
            value = f.values[tuple(assignment.get(v, slice(None)) for v in f.scope)]
            if scope:
                result.factors.append(ModularFactor(scope, np.ascontiguousarray(value)))
            else:
                result.scalar = self.field.scalar_multiply(result.scalar, int(value))
        return result


class ModularWorkspace:
    def __init__(self):
        self.scratch = None
        self.free = {}
        self.allocated_bytes = 0

    def prepare(self, entries, budget):
        if self.scratch is None or self.scratch.size < entries:
            old = 0 if self.scratch is None else self.scratch.nbytes
            if self.allocated_bytes+8*entries > budget:
                raise ValueError('Modular scratch exceeds the declared live-array budget')
            self.scratch = np.empty(entries, dtype=np.uint64)
            self.allocated_bytes += self.scratch.nbytes-old

    def acquire(self, entries, budget):
        if self.free.get(entries):
            return self.free[entries].pop()
        if self.allocated_bytes+8*entries > budget:
            raise ValueError('Modular output exceeds the declared live-array budget')
        array = np.empty(entries, dtype=np.uint64)
        self.allocated_bytes += array.nbytes
        return array

    def release(self, buffer):
        self.free.setdefault(buffer.size, []).append(buffer)


def contract_modular(network, plan, *, output_variables=(), workspace=None,
                     max_entries=1 << 26, max_live_bytes=1700*2**20, progress=None):
    outputs = tuple(sorted(set(output_variables)))
    if list(outputs) != plan['output_variables'] or not set(outputs) <= network.variables:
        raise ValueError('Requested outputs disagree with plan')
    if (set(plan['order']) != network.variables-set(outputs) or
        len(plan['order']) != len(set(plan['order']))):
        raise ValueError('Plan must eliminate every internal variable exactly once')
    if not 0 < max_entries <= 1 << 26 or plan['maximum_product_entries'] > max_entries:
        raise ValueError('Plan exceeds the declared native array ceiling')
    field = network.field
    factors, scalar = [], network.scalar
    for f in network.factors:
        if f.scope:
            factors.append(f)
        else:
            scalar = field.scalar_multiply(scalar, f.values.item())
    resident = sum(f.values.nbytes for f in network.factors)
    output_bytes = 8*(1 << len(outputs))
    budget = max_live_bytes-2*resident-output_bytes
    scratch = max([1 << step['output_scope_bits'] for step in plan['steps']]+[1 << len(outputs)])
    workspace = ModularWorkspace() if workspace is None else workspace
    workspace.prepare(scratch, budget)
    began = last = time.perf_counter()
    for step, variable in enumerate(plan['order']):
        bucket = [f for f in factors if variable in f.scope]
        factors = [f for f in factors if variable not in f.scope]
        if not bucket:
            scalar = 2*scalar % field.prime
            continue
        joined = tuple(sorted(set().union(*(set(f.scope) for f in bucket))))
        scope = tuple(v for v in joined if v != variable)
        entries = 1 << len(scope)
        if 2*entries > max_entries or entries > workspace.scratch.size:
            raise ValueError('Actual binary bucket exceeds its declared cap')
        buffer = workspace.acquire(entries, budget)
        output = buffer.reshape((2,)*len(scope))
        temporary = workspace.scratch[:entries].reshape(output.shape)
        for bit in (0, 1):
            product = output if bit == 0 else temporary
            product.fill(field.one)
            for f in bucket:
                field.broadcast(product, f.values, f.scope, scope, {variable: bit})
        field.add(output, temporary)
        for f in bucket:
            if f.buffer is not None:
                workspace.release(f.buffer)
        if scope:
            factors.append(ModularFactor(scope, output, buffer))
        else:
            scalar = field.scalar_multiply(scalar, output.item())
            workspace.release(buffer)
        if progress and time.perf_counter()-last > 4:
            progress({'step': step, 'steps': len(plan['order']), 'seconds': time.perf_counter()-began,
                      'pooled_bytes': workspace.allocated_bytes})
            last = time.perf_counter()
        del bucket, f, product, output, temporary
    result = workspace.scratch[:1 << len(outputs)].reshape((2,)*len(outputs))
    result.fill(scalar)
    for f in factors:
        if not set(f.scope) <= set(outputs):
            raise ValueError('Internal index remains after elimination')
        field.broadcast(result, f.values, f.scope, outputs)
        if f.buffer is not None:
            workspace.release(f.buffer)
    result = result.copy()
    field.convert(result, encode=False)
    return result, {'seconds': time.perf_counter()-began,
                    'pooled_array_bytes': workspace.allocated_bytes,
                    'maximum_estimated_live_array_bytes': 2*resident+workspace.allocated_bytes+output_bytes,
                    'arithmetic': 'Exact Montgomery field operations; both branches retained; decoded ordinary residues',
                    'backend': field.evidence, 'output_variables': list(outputs)}
