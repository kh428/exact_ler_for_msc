"""Positive binary-auxiliary response for a corrected check's Pauli input."""

from .dyadic import make


def verify_code(boundary):
    """The corrected signed-weight identity is needed on the whole normalizer."""
    for word in boundary.words:
        for value in (word,word ^ boundary.all):
            if (boundary.signed_weight(value)-boundary.target_phase*(value.bit_count()%2))%4:
                raise ValueError('The signed-weight code identity is not satisfied')


def linear_system(boundary,u,z,a=0,b=0,sign=1):
    if sign not in (-1,1) or any(v < 0 or v >> boundary.n for v in (u,z,a,b)):
        raise ValueError('Invalid frame or check sign')
    if any(((u ^ a)&g).bit_count()%2 for g in boundary.checks):
        return None
    generators = boundary.checks+(boundary.all,)
    last = len(generators)-1
    k = (u ^ a).bit_count()%2
    e = int(sign == -1) ^ (z.bit_count()%2)
    linear = b ^ z ^ (u & a)
    rows = [sum(((u & gi & gj).bit_count()%2 ^ (k if i == j == last else 0)) << j
                for j,gj in enumerate(generators)) for i,gi in enumerate(generators)]
    rhs = sum(((linear & gi).bit_count()%2 ^ (e if i == last else 0)) << i
              for i,gi in enumerate(generators))
    return rows,rhs,e


def explicit_response(boundary,u,z,a=0,b=0,sign=1):
    """Count every auxiliary assignment, independently of Gaussian elimination."""
    system = linear_system(boundary,u,z,a,b,sign)
    zero = make(0,0,0)
    if system is None:
        return zero,zero
    rows,rhs,e = system
    count = sum(all(((row & h).bit_count()%2) == (rhs >> i & 1)
                    for i,row in enumerate(rows)) for h in range(1 << len(rows)))
    acceptance = make(count,0,len(rows))
    return acceptance,acceptance if e else zero
