"""Primes, the exact rational series from modular results, and series arithmetic.

The float64 engine is exact for primes below 2^21. Each prime is 1 modulo 8, so that i and sqrt(2) are field elements.
A coefficient is reconstructed from several primes by the Chinese remainder theorem and rational reconstruction
(the fraction with numerator and denominator below sqrt(M/2)); a held-out prime must then agree.
"""
from fractions import Fraction
from math import comb, isqrt

from . import REPO  # noqa: F401
from alternatives.calculation.pauli_polynomial import sqrt_mod          # noqa: E402


def is_prime(n):
    if n < 2:
        return False
    for d in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % d == 0:
            return n == d
    s, r = n - 1, 0
    while s % 2 == 0:
        s //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, s, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes(count, below=1 << 21):
    """The largest primes below 2^21 that are 1 modulo 8, each with its smaller square root of 2."""
    out, q = [], below - 1
    while len(out) < count:
        if q % 8 == 1 and is_prime(q):
            out.append((q, sqrt_mod(2, q)))
        q -= 1
    return out


def crt(residues, moduli):
    value, modulus = 0, 1
    for r, m in zip(residues, moduli):
        value += modulus * ((r - value) * pow(modulus, -1, m) % m)
        modulus *= m
    return value % modulus, modulus


def rational(value, modulus):
    """The fraction n/d with |n|, d below sqrt(modulus/2) that reduces to value, or None."""
    bound = isqrt(modulus // 2)
    r0, r1, t0, t1 = modulus, value, 0, 1
    while r1 > bound:
        k = r0 // r1
        r0, r1, t0, t1 = r1, r0 - k * r1, t1, t0 - k * t1
    if t1 == 0 or abs(t1) > bound:
        return None
    return Fraction(r1, t1)


def residue(value, prime):
    value = Fraction(value)
    return value.numerator % prime * pow(value.denominator % prime, -1, prime) % prime


def reconstruct(by_prime, held_out=1):
    """Exact coefficients from {prime: [c_0, ..., c_K]}; the last held_out primes check the result.

    Returns the list of Fractions, or None for a coefficient that needs more primes."""
    ps = list(by_prime)
    used, spare = ps[:len(ps) - held_out], ps[len(ps) - held_out:]
    out = []
    for k in range(len(by_prime[ps[0]])):
        value, modulus = crt([by_prime[p][k] for p in used], used)
        f = rational(value, modulus)
        if f is not None and any(residue(f, p) != by_prime[p][k] for p in spare):
            f = None
        out.append(f)
    return out


def ratio(b, a):
    """The series of B/A."""
    out = []
    for k in range(len(b)):
        out.append((b[k] - sum(out[j] * a[k - j] for j in range(k))) / a[0])
    return out


def times_binomial(coefficients, count):
    """Multiply a series by (1+x)^count (count may be negative)."""
    K = len(coefficients) - 1
    if count >= 0:
        factor = [Fraction(comb(count, k)) for k in range(K + 1)]
    else:
        factor = [Fraction((-1) ** k * comb(-count + k - 1, k)) for k in range(K + 1)]
    return [sum(coefficients[j] * factor[k - j] for j in range(k + 1)) for k in range(K + 1)]


NORMALISATION = ('A(p) = (1-p)^N sum_k A[k] x^k and B(p) = (1-p)^N sum_k B[k] x^k with x = p/(1-p) and N the number '
                 'of locations; P_L = B/A = sum_k P_L[k] x^k')


def record(circuit_path, spec, route, locations, discarded, degree, A, B, L, runs, law_norm=1, **extra):
    """A result file: the exact series and every residue it was rebuilt from (the last prime held out)."""
    import hashlib
    from pathlib import Path
    path = Path(circuit_path)
    return dict(circuit=path.name, circuit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), route=route,
                noise=spec, locations=locations, discarded_erasure_edges=discarded, law_norm=str(law_norm),
                degree=degree,
                normalisation=NORMALISATION, A=[str(v) for v in A], B=[str(v) for v in B], P_L=[str(v) for v in L],
                held_out_primes=1,
                residues=[dict(prime=p, sqrt2=r['sqrt2'], A=r['A'], B=r['B']) for p, r in runs.items()], **extra)


def check_record(result):
    """Rebuild the series of a result file from its residues and check it; returns a short report."""
    runs = {r['prime']: r for r in result['residues']}
    for p, r in runs.items():
        if not (is_prime(p) and p % 8 == 1 and p < 1 << 21 and r['sqrt2'] ** 2 % p == 2):
            raise ValueError(f'invalid prime or square root of 2: {p}')
    A = reconstruct({p: r['A'] for p, r in runs.items()}, result['held_out_primes'])
    B = reconstruct({p: r['B'] for p, r in runs.items()}, result['held_out_primes'])
    if None in A or None in B:
        raise ValueError('a coefficient is not determined by the saved residues')
    if [str(v) for v in A] != result['A'] or [str(v) for v in B] != result['B']:
        raise ValueError('the saved fractions differ from the reconstruction')
    if [str(v) for v in ratio(B, A)] != result['P_L']:
        raise ValueError('P_L is not B/A')
    n, norm = result['locations'], Fraction(result.get('law_norm', 1))
    if norm == 1:
        if not all(0 <= b <= a <= comb(n, k) for k, (a, b) in enumerate(zip(A, B))):
            raise ValueError('a coefficient is outside 0 <= B[k] <= A[k] <= C(N, k)')
    elif not all(abs(a) <= comb(n, k) * norm ** k and abs(b) <= comb(n, k) * norm ** k
                 for k, (a, b) in enumerate(zip(A, B))):
        raise ValueError('a coefficient is outside |A[k]|, |B[k]| <= C(N, k) m^k')
    return dict(degree=result['degree'], primes=len(runs), first_nonzero_order=next(
        (k for k, v in enumerate(B) if v), None))


def compose(L, z, K):
    """sum_k L_k z(x)^k as a series in x, for a series z with z(0) = 0; exact Fractions, truncated at x^K."""
    def mul(a, b):
        out = [Fraction(0)] * (K + 1)
        for i, u in enumerate(a):
            if u:
                for j, v in enumerate(b[:K + 1 - i]):
                    out[i + j] += u * v
        return out
    out, power = [Fraction(0)] * (K + 1), [Fraction(1)] + [Fraction(0)] * K
    for c in L[:K + 1]:
        out = [o + Fraction(c) * w for o, w in zip(out, power)]
        power = mul(power, z)
    return out


def ignored_erasure(L, rate, K):
    """P_L with an ignored erasure of rate e = rate * p on every edge, from the Pauli-only series L.

    An ignored erasure depolarises its edge completely; on every edge both channels only damp nontrivial labels,
    by (1 - 4p/3) and (1 - e), so with the same channels on every edge P_L(p, e) = P_L^Pauli(p + 3e/4 - p e).
    Returns the series in x = p/(1-p) of L at x' = p'/(1-p'), p' = p + 3e/4 - p e."""
    rate = Fraction(rate)
    p = [Fraction(0)] + [Fraction((-1) ** (k - 1)) for k in range(1, K + 1)]          # x/(1+x)
    p2 = compose([0, 0, 1], p, K)
    peff = [(1 + Fraction(3, 4) * rate) * a - rate * b for a, b in zip(p, p2)]
    xeff = compose([0] + [1] * K, peff, K)                                              # p'/(1-p')
    return compose(L, xeff, K)
