"""Exact sparse Heisenberg maps, apart from floating-point coefficient arithmetic.

Hermitian P(x,z) = i**popcount(x & z) X**x Z**z; qubit 0 is least significant.
Terms with identical (x,z) are added coherently. No coefficient cutoff is used.
Operations are tuples documented by adjoint(). Measurement maps are selective
and unnormalised. A probability requires summing all desired classical records.
"""

import math


def multiply(a, b):
    x, z = a
    u, v = b
    key = (x ^ u, z ^ v)
    exponent = ((x & z).bit_count() + (u & v).bit_count()
                - (key[0] & key[1]).bit_count() + 2 * (z & u).bit_count()) % 4
    return key, (1, 1j, -1, -1j)[exponent]


def anticommutes(a, b):
    return ((a[0] & b[1]).bit_count() + (a[1] & b[0]).bit_count()) % 2


def add(out, key, value):
    if value:
        out[key] = out.get(key, 0) + value


def clean(terms):
    return {key: value for key, value in terms.items() if value != 0}


def rotation(terms, generator, cosine, sine):
    out = {}
    for key, coefficient in terms.items():
        if not anticommutes(generator, key):
            add(out, key, coefficient)
        else:
            other, phase = multiply(generator, key)
            add(out, key, coefficient * cosine)
            add(out, other, coefficient * (1j * phase) * sine)
    return clean(out)


def project(terms, measurement, sign):
    if sign not in (-1, 1):
        raise ValueError("Projector sign must be +1 or -1")
    out = {}
    for key, coefficient in terms.items():
        if not anticommutes(measurement, key):
            other, phase = multiply(measurement, key)
            add(out, key, coefficient / 2)
            add(out, other, coefficient * sign * phase / 2)
    return clean(out)


def adjoint(terms, operation, sqrt_half=1 / math.sqrt(2)):
    """Apply an unnormalised adjoint channel to an observable.

    H/S/S_DAG/T/T_DAG/X/Y/Z(q), CX(c,t), ROT(x,z,theta),
    DEP(qubits,p) with total nonidentity Pauli probability p, FLIP(x,z,p),
    POST(x,z,sign) for (I+sign*P)/2, RESET(q,axis,sign), PAULI(x,z).
    """
    name, *args = operation
    if name in ("T", "T_DAG", "S", "S_DAG"):
        q, = args
        cosine, sine = ((sqrt_half, sqrt_half)
                        if name.startswith("T") else (0, 1))
        if name.endswith("DAG"):
            sine = -sine
        return rotation(terms, (0, 1 << q), cosine, sine)
    if name == "ROT":
        x, z, theta = args
        return rotation(terms, (x, z), math.cos(theta), math.sin(theta))
    if name == "POST":
        x, z, sign = args
        return project(terms, (x, z), sign)
    if name == "FLIP":
        x, z, probability = args
        if not 0 <= probability <= 1:
            raise ValueError("Invalid Pauli-flip probability")
        return clean({key: coefficient * (1 - 2 * probability if anticommutes((x, z), key) else 1)
                      for key, coefficient in terms.items()})
    if name in ("PAULI", "X", "Y", "Z"):
        if name == "PAULI":
            fault = tuple(args)
        else:
            q, = args
            fault = ((1 << q) if name in "XY" else 0,
                     (1 << q) if name in "YZ" else 0)
        return {key: coefficient * (-1 if anticommutes(fault, key) else 1)
                for key, coefficient in terms.items()}
    out = {}
    for (x, z), coefficient in terms.items():
        if name == "H":
            q, = args
            a, b = (x >> q) & 1, (z >> q) & 1
            key = (x ^ ((a ^ b) << q), z ^ ((a ^ b) << q))
            coefficient *= -1 if a and b else 1
        elif name == "CX":
            c, t = args
            xc, xt, zc, zt = (x >> c) & 1, (x >> t) & 1, (z >> c) & 1, (z >> t) & 1
            coefficient *= -1 if xc and zt and (xt ^ zc ^ 1) else 1
            key = (x ^ (xc << t), z ^ (zt << c))
        elif name == "DEP":
            qubits, probability = args
            if not qubits or len(set(qubits)) != len(qubits) or not 0 <= probability <= 1:
                raise ValueError("Invalid depolarising channel")
            mask = sum(1 << q for q in qubits)
            if (x | z) & mask:
                coefficient *= 1 - 4**len(qubits) / (4**len(qubits) - 1) * probability
            key = (x, z)
        elif name == "RESET":
            q, axis, sign = args
            a, b = (x >> q) & 1, (z >> q) & 1
            if (a, b) == (0, 0):
                key = (x, z)
            elif (a, b) == {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}[axis]:
                key = (x ^ (a << q), z ^ (b << q))
                coefficient *= sign
            else:
                continue
        else:
            raise ValueError("Unsupported operation: " + name)
        add(out, key, coefficient)
    return clean(out)


def backwards(effect, operations, max_terms=1_000_000):
    terms = dict(effect)
    profile = []
    for index in range(len(operations) - 1, -1, -1):
        terms = adjoint(terms, operations[index])
        profile.append({"index": index, "operation": operations[index][0], "terms": len(terms)})
        if len(terms) > max_terms:
            raise RuntimeError(f"Pauli term budget exceeded at operation {index}: {len(terms)}")
    return terms, profile


def zero_expectation(terms):
    """Expectation in |0...0>; requires no dense matrix or list of 2**n strings."""
    return sum(coefficient for (x, z), coefficient in terms.items() if x == 0)
