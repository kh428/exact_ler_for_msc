"""Exact low-rank splits of series tensors over F_q (quimb's split simplification, without rounding).

A tensor T over indices L u R, with coefficient matrices M_o (2^|L| x 2^|R|, o = 0..K), splits exactly as
    M_o = A B_o   (A constant: a basis of the column space of [M_0 | ... | M_K]; B_o from its reduced row echelon form)
or, with L and R swapped, M_o = A_o B (B constant). The bond of rank r is written as b = ceil(log2 r) binary indices
(zero padding), and a split is useful when b < min(|L|, |R|).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _inv(a, q):
    r, x, y = q, a % q, 1
    s0, s1 = 0, 1
    r0, r1 = q, a % q
    while r1:
        k = r0 // r1
        r0, r1 = r1, r0 - k * r1
        s0, s1 = s1, s0 - k * s1
    return s0 % q


@njit(cache=True)
def rref(M, q, max_rank):
    """Row-reduce M (int64, modified in place) mod q. Returns (rank, pivot columns); rank = max_rank + 1 means more."""
    rows, cols = M.shape
    piv = np.empty(min(rows, cols) + 1, np.int64)
    r = 0
    for c in range(cols):
        if r == rows:
            break
        p = -1
        for i in range(r, rows):
            if M[i, c] != 0:
                p = i
                break
        if p < 0:
            continue
        if r >= max_rank:
            return max_rank + 1, piv[:0]
        if p != r:
            for j in range(cols):
                t = M[r, j]
                M[r, j] = M[p, j]
                M[p, j] = t
        inv = _inv(M[r, c], q)
        for j in range(c, cols):
            M[r, j] = M[r, j] * inv % q
        for i in range(rows):
            if i != r and M[i, c] != 0:
                f = M[i, c]
                for j in range(c, cols):
                    M[i, j] = (M[i, j] - f * M[r, j]) % q
        piv[r] = c
        r += 1
    return r, piv[:r]


def _matrix(arr, inds, L, R, K):
    """[M_0 | M_1 | ... | M_K] with rows over L and columns over R (for each order)."""
    perm = [inds.index(v) for v in L] + [inds.index(v) for v in R] + [len(inds)]
    t = np.transpose(arr, perm).reshape(1 << len(L), 1 << len(R), K + 1)
    return np.ascontiguousarray(np.transpose(t, (0, 2, 1)).reshape(1 << len(L), (K + 1) << len(R))).astype(np.int64)


def best_split(inds, arr, K, q, max_n=12, max_bits=2, protect=()):
    """The bipartition and orientation with the smallest max(|L| + b, |R| + b), if any beats len(inds)."""
    n = len(inds)
    if n < 3 or n > max_n:
        return None
    best = None
    for k in range(1, 2 ** (n - 1)):
        L = [inds[i] for i in range(n) if not (k >> i) & 1]
        R = [inds[i] for i in range(n) if (k >> i) & 1]
        if not L or not R:
            continue
        for side in (0, 1):                           # 0: A constant over L; 1: B constant over R
            P, Q = (L, R) if side == 0 else (R, L)
            mmin = min(len(P), len(Q))
            cap = 1 << min(mmin - 1, max_bits)          # b < min(|L|, |R|), bond at most 2^max_bits
            M = _matrix(arr, inds, P, Q, K)
            r, _ = rref(M, q, cap)
            if r > cap or r == 0:
                continue
            b = (r - 1).bit_length()
            score = (max(len(L), len(R)) + b, (1 << (len(L) + b)) + (1 << (len(R) + b)))
            if best is None or score < best[0]:
                best = (score, L, R, side, r, b)
    return best


def apply_split(inds, arr, K, q, L, R, side, r, b, bond_names):
    """Exact factors: returns [(inds, array), (inds, array)]."""
    P, Q = (L, R) if side == 0 else (R, L)
    M = _matrix(arr, inds, P, Q, K)
    C = M.copy()
    rank, piv = rref(M, q, 1 << 30)
    assert rank == r
    A = C[:, piv]                                       # (2^|P|, r) constant
    Bf = M[:r]                                          # (r, (K+1) 2^|Q|)
    Bo = Bf.reshape(r, K + 1, 1 << len(Q)).transpose(0, 2, 1)       # (r, 2^|Q|, K+1)
    pad = 1 << b
    Ap = np.zeros((1 << len(P), pad, K + 1))
    Ap[:, :r, 0] = A
    Bp = np.zeros((pad, 1 << len(Q), K + 1))
    Bp[:r] = Bo
    bond = list(bond_names[:b])
    ta = (tuple(P) + tuple(bond), Ap.reshape((2,) * (len(P) + b) + (K + 1,)))
    tb = (tuple(bond) + tuple(Q), Bp.reshape((2,) * (b + len(Q)) + (K + 1,)))
    return [ta, tb]
