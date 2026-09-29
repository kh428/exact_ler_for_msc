"""Small GF(2) routines using integers as packed bit vectors."""


def echelon(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p not in pivots:
                pivots[p] = row
                break
            row ^= pivots[p]
    return pivots


def rank(rows):
    return len(echelon(rows))


def kernel(rows, columns):
    pivots = echelon(rows)
    basis = []
    for f in range(columns):
        if f in pivots:
            continue
        vector = 1 << f
        for p, row in sorted(pivots.items()):
            if (row & vector).bit_count() & 1:
                vector ^= 1 << p
        basis.append(vector)
    return basis


def xor_sum(values):
    result = 0
    for value in values:
        result ^= value
    return result

