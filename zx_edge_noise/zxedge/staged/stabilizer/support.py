"""Certified moving stabilizer supports for exact, state-dependent Pauli merging.

At cut j, signed commuting Paulis S satisfy S rho_j = rho_j. If a Pauli
anticommutes with any S, its expectation vanishes. Otherwise multiply by S
to choose a canonical coset representative and add coefficients coherently.

Supports are conservative certificates, not a complete description of rho.
There is no assumed bound of one logical qubit. General Pauli noise removes
support constraints unless EVERY possible error preserves them. The certificate
can become empty, in which case this implementation offers no compression.
"""

from pauli import add, adjoint, anticommutes, clean, multiply, zero_expectation


def product(a, b):
    key, phase = multiply(a[:2], b[:2])
    sign = a[2] * b[2] * phase
    if sign not in (-1, 1):
        raise ValueError("Support generators must be commuting Hermitian Paulis")
    return (*key, int(complex(sign).real))


def basis(generators, n):
    rows = {}
    for generator in generators:
        row = generator
        while row[0] or row[1]:
            code = row[0] | (row[1] << n)
            pivot = code.bit_length() - 1
            if pivot in rows:
                row = product(row, rows[pivot])
            else:
                rows[pivot] = row
                break
        if not row[0] and not row[1] and row[2] == -1:
            return {"rows": [], "zero": True}
    values = [rows[p] for p in sorted(rows, reverse=True)]
    if any(anticommutes(a[:2], b[:2]) for a in values for b in values):
        raise ValueError("Noncommuting support certificate")
    return {"rows": values, "zero": False}


def kernel(rows, functional):
    pivot = next((row for row in rows if functional(row)), None)
    if pivot is None:
        return list(rows)
    return [product(row, pivot) if functional(row) else row for row in rows if row != pivot]


def advance(certificate, operation, n):
    if certificate["zero"]:
        return certificate
    rows = certificate["rows"]
    name, *args = operation
    if name in ("T", "T_DAG", "ROT"):
        generator = (0, 1 << args[0]) if name != "ROT" else tuple(args[:2])
        rows = kernel(rows, lambda row: anticommutes(row[:2], generator))
    elif name == "POST":
        x, z, sign = args
        rows = kernel(rows, lambda row: anticommutes(row[:2], (x, z)))
        rows.append((x, z, sign))
    elif name == "RESET":
        q, axis, sign = args
        rows = kernel(rows, lambda row: (row[0] >> q) & 1)
        rows = kernel(rows, lambda row: (row[1] >> q) & 1)
        rows.append(((1 << q) if axis in "XY" else 0,
                     (1 << q) if axis in "YZ" else 0, sign))
    elif name == "DEP":
        qubits, probability = args
        if probability:
            for q in qubits:
                rows = kernel(rows, lambda row: (row[0] >> q) & 1)
                rows = kernel(rows, lambda row: (row[1] >> q) & 1)
    elif name == "FLIP":
        x, z, probability = args
        if probability:
            rows = kernel(rows, lambda row: anticommutes(row[:2], (x, z)))
    elif name in ("H", "S", "S_DAG", "CX", "PAULI", "X", "Y", "Z"):
        inverse = ({"S": "S_DAG", "S_DAG": "S"}.get(name, name), *args)
        updated = []
        for x, z, sign in rows:
            image = adjoint({(x, z): sign}, inverse)
            if len(image) != 1:
                raise ValueError("Expected Clifford signed permutation")
            (key, coefficient), = image.items()
            updated.append((*key, int(complex(coefficient).real)))
        rows = updated
    else:
        raise ValueError(name)
    return basis(rows, n)


def trajectory(n, operations, certified_extensions=None):
    """Optional extensions require a separate proof, e.g. clean_check.certify()."""
    certificates = [basis([(0, 1 << q, 1) for q in range(n)], n)]
    for index,operation in enumerate(operations):
        following=advance(certificates[-1], operation, n)
        extra=(certified_extensions or {}).get(index+1)
        if extra and not following["zero"]:
            following=basis(following["rows"]+extra,n)
        certificates.append(following)
    return certificates


def compress(terms, certificate, n):
    if certificate["zero"]:
        return {}
    rows = certificate["rows"]
    out = {}
    for key, coefficient in terms.items():
        if any(anticommutes(key, row[:2]) for row in rows):
            continue
        x, z = key
        for a, b, sign in rows:
            pivot = (a | (b << n)).bit_length() - 1
            if ((x | (z << n)) >> pivot) & 1:
                (x, z), phase = multiply((x, z), (a, b))
                coefficient *= phase * sign
        add(out, (x, z), coefficient)
    return clean(out)


def backwards(n, operations, effect, max_terms=150_000, exact_coefficients=False, certified_extensions=None):
    certificates = trajectory(n, operations, certified_extensions)
    if exact_coefficients:
        from dyadic import convert, make
        effect = {key: convert(value) for key,value in effect.items()}
    terms = compress(effect, certificates[-1], n)
    profile = []
    peak_before_merge = len(terms)
    for index in range(len(operations) - 1, -1, -1):
        if exact_coefficients:
            terms = adjoint(terms, operations[index], sqrt_half=make(0,1,1))
        else:
            terms = adjoint(terms, operations[index])
        raw_count = len(terms)
        peak_before_merge = max(peak_before_merge, raw_count)
        terms = compress(terms, certificates[index], n)
        profile.append({"index": index, "operation": operations[index][0],
                        "before_merge": raw_count, "after_merge": len(terms),
                        "certified_rank": len(certificates[index]["rows"])})
        if raw_count > max_terms:
            raise RuntimeError(f"Term budget exceeded at {index}: {raw_count}")
    raw_value = zero_expectation(terms)
    if exact_coefficients:
        raw_value = convert(raw_value)
    value = complex(raw_value)
    if abs(value.imag) > 1e-13:
        raise ValueError(value)
    return value.real, {"peak_terms": peak_before_merge,
                        "exact_value": raw_value.exact() if hasattr(raw_value,"exact") else None,
                        "peak_terms_after_merge": max([0] + [p["after_merge"] for p in profile]),
                        "minimum_certified_rank": min(len(c["rows"]) for c in certificates),
                        "profile": profile}
