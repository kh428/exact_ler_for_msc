"""Bounded binary factor networks with structural planning before allocation.

The simplifier uses only exact zeros, equality constraints and constants.
The planner allocates no large tensors. Numerical elimination checks its
entry and live-array budgets before forming each bucket product.
"""

from dataclasses import dataclass
import time
import numpy as np


@dataclass
class Factor:
    scope: tuple
    values: np.ndarray
    buffer: object = None


class Workspace:
    """Reuse large allocations across buckets and conditioned slices.

    Released output buffers remain explicitly counted in the memory budget.
    This avoids relying on the platform allocator to return pages promptly.
    """

    def __init__(self):
        self.scratch = None
        self.free = {}
        self.allocated_bytes = 0

    def prepare(self, entries, budget):
        if self.scratch is None or self.scratch.size < entries:
            old = 0 if self.scratch is None else self.scratch.nbytes
            if self.allocated_bytes+16*entries > budget:
                raise RuntimeError("Scratch allocation would exceed the tensor budget")
            self.scratch = np.empty(entries, dtype=np.complex128)
            self.allocated_bytes += self.scratch.nbytes-old

    def acquire(self, entries, budget):
        available = self.free.get(entries)
        if available:
            return available.pop()
        if self.allocated_bytes+16*entries > budget:
            raise RuntimeError("Output-buffer allocation would exceed the tensor budget")
        result = np.empty(entries, dtype=np.complex128)
        self.allocated_bytes += result.nbytes
        return result

    def release(self, buffer):
        self.free.setdefault(buffer.size, []).append(buffer)


class Network:
    def __init__(self):
        self.labels = []
        self.factors = []
        self.variables = set()
        self.scalar = np.complex128(1)

    def variable(self, label):
        value = len(self.labels)
        self.labels.append(label)
        self.variables.add(value)
        return value

    def multiply_scalar(self, value):
        self.scalar *= value

    def add(self, scope, values):
        scope = tuple(int(v) for v in scope)
        if len(set(scope)) != len(scope):
            raise ValueError("Factor scope has duplicate variables")
        array = np.asarray(values, dtype=np.complex128).reshape((2,)*len(scope))
        order = np.argsort(scope)
        if len(scope):
            array = np.transpose(array, order)
            scope = tuple(scope[i] for i in order)
        self.factors.append(Factor(scope, array))

    def fixed(self, variable, value=0):
        self.add((variable,), (1, 0) if value == 0 else (0, 1))

    def xor(self, scope, parity=0):
        shape = (2,)*len(scope)
        values = (np.sum(np.indices(shape), axis=0) % 2 == parity).astype(float)
        self.add(scope, values)

    def condition(self, assignment):
        """Fix selected indices without summing them or changing other indices.

        Scopes are value independent, so one structural plan covers every
        assignment. Summing all assignments recovers the original contraction.
        """
        if not set(assignment) <= self.variables or any(bit not in (0, 1) for bit in assignment.values()):
            raise ValueError("Invalid binary conditioning assignment")
        result = Network()
        result.labels = list(self.labels)
        result.variables = self.variables-set(assignment)
        result.scalar = self.scalar
        for factor in self.factors:
            indices = tuple(assignment.get(v, slice(None)) for v in factor.scope)
            scope = tuple(v for v in factor.scope if v not in assignment)
            values = factor.values[indices]
            if scope:
                result.factors.append(Factor(scope, values))
            else:
                result.scalar *= complex(values)
        return result

    def simplify(self, protected=()):
        """Preserve every named output index, including fixed/equal indices."""
        protected = set(protected)
        if not protected <= self.variables:
            raise ValueError("Protected indices must belong to this network")
        original = (len(self.variables), len(self.factors))
        fixed_count = equality_count = constants = 0
        while True:
            changed = False
            for index, factor in enumerate(self.factors):
                values, scope = factor.values, factor.scope
                if not values.size or np.all(values == values.flat[0]):
                    self.scalar *= complex(values.flat[0])
                    self.factors.pop(index)
                    constants += 1
                    changed = True
                    break
                if len(scope) == 1 and np.count_nonzero(values) == 1:
                    if scope[0] in protected:
                        continue
                    bit = int(np.flatnonzero(values)[0])
                    variable = scope[0]
                    self.scalar *= values[bit]
                    self.factors.pop(index)
                    updated = []
                    for other in self.factors:
                        if variable in other.scope:
                            axis = other.scope.index(variable)
                            updated.append(Factor(tuple(v for v in other.scope if v != variable), np.take(other.values, bit, axis=axis)))
                        else:
                            updated.append(other)
                    self.factors = updated
                    self.variables.remove(variable)
                    fixed_count += 1
                    changed = True
                    break
                if len(scope) == 2 and (np.array_equal(values, np.eye(2)) or np.array_equal(values, np.fliplr(np.eye(2)))):
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
                    equality_count += 1
                    changed = True
                    break
            if not changed:
                break
        # Factors with the same scope can be multiplied without increasing width.
        groups = {}
        for factor in self.factors:
            if factor.scope in groups:
                groups[factor.scope] *= factor.values
            else:
                groups[factor.scope] = factor.values.copy()
        self.factors = [Factor(scope, values) for scope, values in groups.items()]
        return {"original_variables": original[0], "original_factors": original[1],
                "remaining_variables": len(self.variables), "remaining_factors": len(self.factors),
                "fixed_variables": fixed_count, "merged_equalities": equality_count, "removed_constants": constants}

    def plan(self, strategy="min_fill", keep=()):
        if strategy not in ("min_fill", "min_degree"):
            raise ValueError("Unknown elimination strategy")
        keep = set(keep)
        if not keep <= self.variables:
            raise ValueError("Open indices must belong to this network")
        began = time.perf_counter()
        graph = {v: 0 for v in self.variables}
        for factor in self.factors:
            mask = sum(1 << v for v in factor.scope)
            for v in factor.scope:
                graph[v] |= mask ^ (1 << v)
        scopes = [set(f.scope) for f in self.factors]
        order, rows = [], []
        max_scope = peak_entries = max_live_entries = multiplications = 0
        while set(graph)-keep:
            def score(v):
                neighbors = graph[v]
                degree = neighbors.bit_count()
                if strategy == "min_degree":
                    return degree, v
                edges = 0
                remaining = neighbors
                while remaining:
                    bit = remaining & -remaining
                    u = bit.bit_length()-1
                    edges += (graph[u] & neighbors).bit_count()
                    remaining ^= bit
                return degree*(degree-1)//2-edges//2, degree, v
            variable = min(set(graph)-keep, key=score)
            neighbors = graph.pop(variable)
            remaining = neighbors
            while remaining:
                bit = remaining & -remaining
                other = bit.bit_length()-1
                graph[other] = (graph[other] | (neighbors ^ bit)) & ~(1 << variable)
                remaining ^= bit
            bucket = [scope for scope in scopes if variable in scope]
            scopes = [scope for scope in scopes if variable not in scope]
            joined = set().union(*bucket) if bucket else {variable}
            count = 1 << len(joined)
            output = joined-{variable}
            live = sum(1 << len(scope) for scope in scopes) + sum(1 << len(scope) for scope in bucket)
            max_live_entries = max(max_live_entries, live+count+(1 << len(output)))
            peak_entries = max(peak_entries, count)
            max_scope = max(max_scope, len(joined))
            multiplications += count*max(1, len(bucket))
            if output:
                scopes.append(output)
            order.append(variable)
            rows.append({"variable": variable, "label": self.labels[variable], "product_scope_bits": len(joined),
                         "input_factors": len(bucket), "output_scope_bits": len(output), "product_scope": sorted(joined)})
        if keep:
            # The final tensor includes fixed or isolated output indices too.
            entries = 1 << len(keep)
            max_scope = max(max_scope, len(keep))
            peak_entries = max(peak_entries, entries)
            max_live_entries = max(max_live_entries, sum(1 << len(scope) for scope in scopes)+2*entries)
            multiplications += entries*(len(scopes)+1)
        return {"strategy": strategy, "order": order, "maximum_product_scope_bits": max_scope,
                "output_variables": sorted(keep),
                "maximum_product_entries": peak_entries, "estimated_peak_live_entries": max_live_entries,
                "estimated_peak_complex_bytes": 16*max_live_entries, "estimated_element_multiplications": multiplications,
                "seconds": time.perf_counter()-began, "steps": rows}

    def contract(self, plan, max_entries=2**22, max_live_bytes=512*1024**2, progress=None, workspace=None,
                 output_variables=()):
        """Sum internal indices; optional output axes follow sorted variable IDs."""
        output_variables = tuple(sorted(set(output_variables)))
        if not set(output_variables) <= self.variables or list(output_variables) != plan.get("output_variables", []):
            raise ValueError("Plan and requested open indices disagree")
        factors, scalar = [], np.complex128(self.scalar)
        for factor in self.factors:
            if factor.scope:
                factors.append(factor)
            else:
                scalar *= complex(factor.values)
        resident_bytes = sum(f.values.nbytes for f in self.factors)
        eliminated = self.variables-set(output_variables)
        if set(plan["order"]) != eliminated or len(plan["order"]) != len(eliminated):
            raise ValueError("Plan does not eliminate exactly this network's variables")
        began = time.perf_counter()
        if plan["maximum_product_entries"] > max_entries:
            raise RuntimeError("Plan exceeds the declared entry budget before allocation")
        workspace = Workspace() if workspace is None else workspace
        # Count both the source network and conditioned views conservatively.
        output_bytes = 16*(1 << len(output_variables)) if output_variables else 0
        workspace_budget = max_live_bytes-2*resident_bytes-output_bytes
        workspace.prepare(plan["maximum_product_entries"], workspace_budget)
        peak = 0
        for index, variable in enumerate(plan["order"]):
            bucket = [factor for factor in factors if variable in factor.scope]
            factors = [factor for factor in factors if variable not in factor.scope]
            if not bucket:
                scalar *= 2
                continue
            joined = tuple(sorted(set().union(*(set(f.scope) for f in bucket))))
            entries = 1 << len(joined)
            if entries > max_entries or entries > workspace.scratch.size:
                raise RuntimeError(f"Tensor budget exceeded before allocation: {len(joined)} bits")
            product = workspace.scratch[:entries].reshape((2,)*len(joined))
            product.fill(1)
            for factor in bucket:
                shape = tuple(2 if v in factor.scope else 1 for v in joined)
                product *= factor.values.reshape(shape)
            # Inputs have now been consumed into scratch. Their storage can be
            # reused for the sum's output, which never aliases scratch.
            for factor in bucket:
                if factor.buffer is not None:
                    workspace.release(factor.buffer)
            buffer = workspace.acquire(entries//2, workspace_budget)
            output = buffer.reshape((2,)*(len(joined)-1))
            np.sum(product, axis=joined.index(variable), out=output)
            del product
            scope = tuple(v for v in joined if v != variable)
            if scope:
                factors.append(Factor(scope, output, buffer))
            else:
                scalar *= complex(output)
                workspace.release(buffer)
            live_bytes = 2*resident_bytes+workspace.allocated_bytes
            peak = max(peak, live_bytes)
            if progress and (index % 20 == 0 or entries >= 2**20):
                progress({"step": index, "variable": variable, "scope_bits": len(joined),
                          "estimated_live_bytes": live_bytes, "seconds": time.perf_counter()-began})
            # The consumed bucket is no longer part of the network. Release
            # these references before allocating the next bucket product.
            del bucket, factor, output
        if output_variables:
            entries = 1 << len(output_variables)
            if entries > max_entries or entries > workspace.scratch.size:
                raise RuntimeError("Open-output tensor exceeds its declared entry budget")
            product = workspace.scratch[:entries].reshape((2,)*len(output_variables))
            product.fill(scalar)
            for factor in factors:
                if not set(factor.scope) <= set(output_variables):
                    raise ValueError("Contraction left an internal index")
                shape = tuple(2 if v in factor.scope else 1 for v in output_variables)
                product *= factor.values.reshape(shape)
            result = product.copy()
            for factor in factors:
                if factor.buffer is not None:
                    workspace.release(factor.buffer)
            peak = max(peak, 2*resident_bytes+workspace.allocated_bytes+output_bytes)
        elif factors:
            raise ValueError("Contraction left uneliminated factors")
        else:
            result = complex(scalar)
        return result, {"seconds": time.perf_counter()-began, "maximum_estimated_live_bytes": peak,
                        "pooled_array_bytes": workspace.allocated_bytes,
                        "output_variables": list(output_variables),
                        "arithmetic": "Complex128, no coefficient truncation"}
