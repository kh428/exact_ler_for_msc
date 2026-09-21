"""Exact real dyadic Q(sqrt(2)) coefficients for noiseless Clifford+T maps.

Value = (a + b sqrt(2)) / 2**k, with integer a,b,k. Normalisation and a small
cache identify exact cancellations without a numerical cutoff. Floats are
accepted only as their exact binary rational values; sqrt(1/2) is supplied
explicitly as Dyadic(0,1,1), never guessed from a decimal approximation.
"""

from functools import lru_cache
import math


class Dyadic:
    __slots__ = ("a", "b", "k")

    def __init__(self, a=0, b=0, k=0):
        self.a, self.b, self.k = a, b, k

    def __bool__(self):
        return self.a != 0 or self.b != 0

    def __hash__(self):
        return hash((self.a,self.b,self.k))

    def __eq__(self, other):
        if not isinstance(other,Dyadic):
            other = convert(other)
        return (self.a,self.b,self.k)==(other.a,other.b,other.k)

    def __add__(self, other):
        other=convert(other)
        k=max(self.k,other.k)
        return make((self.a << (k-self.k))+(other.a << (k-other.k)),
                    (self.b << (k-self.k))+(other.b << (k-other.k)),k)

    __radd__=__add__

    def __neg__(self):
        return make(-self.a,-self.b,self.k)

    def __sub__(self, other):
        return self + -convert(other)

    def __rsub__(self, other):
        return convert(other) + -self

    def __mul__(self, other):
        other=convert(other)
        return make(self.a*other.a+2*self.b*other.b,
                    self.a*other.b+self.b*other.a,self.k+other.k)

    __rmul__=__mul__

    def __truediv__(self, other):
        if isinstance(other,int) and other > 0 and other & (other-1) == 0:
            return make(self.a,self.b,self.k+other.bit_length()-1)
        raise ValueError("Only division by a positive power of two is supported")

    def __complex__(self):
        return complex(float(self))

    def __float__(self):
        return math.ldexp(float(self.a)+float(self.b)*math.sqrt(2),-self.k)

    def exact(self):
        return {"a":str(self.a),"b":str(self.b),"denominator_power_of_two":self.k}


@lru_cache(maxsize=8192)
def make(a,b,k):
    if a==0 and b==0:
        return Dyadic()
    while k and not (a&1 or b&1):
        a//=2; b//=2; k-=1
    return Dyadic(a,b,k)


def convert(value):
    if isinstance(value,Dyadic):
        return value
    if isinstance(value,complex):
        if value.imag:
            raise ValueError("A Hermitian coefficient acquired an imaginary part")
        value=value.real
    if isinstance(value,int):
        return make(value,0,0)
    if isinstance(value,float):
        numerator,denominator=value.as_integer_ratio()
        return make(numerator,0,denominator.bit_length()-1)
    raise TypeError(type(value))
