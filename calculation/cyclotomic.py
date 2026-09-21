"""Exact complex Q(sqrt(2)) arithmetic for Clifford+T boundary amplitudes."""

from .dyadic import convert, make


class C8:
    __slots__ = ("real", "imag")

    def __init__(self, real=0, imag=0):
        self.real, self.imag = convert(real), convert(imag)

    def __bool__(self):
        return bool(self.real) or bool(self.imag)

    def __add__(self, other):
        other = as_c8(other)
        return C8(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    def __neg__(self):
        return C8(-self.real, -self.imag)

    def __sub__(self, other):
        return self + -as_c8(other)

    def __rsub__(self, other):
        return as_c8(other) + -self

    def __mul__(self, other):
        other = as_c8(other)
        return C8(self.real * other.real - self.imag * other.imag,
                  self.real * other.imag + self.imag * other.real)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return C8(self.real / other, self.imag / other)

    def conjugate(self):
        return C8(self.real, -self.imag)

    def abs2(self):
        return self.real * self.real + self.imag * self.imag

    def __complex__(self):
        return complex(float(self.real), float(self.imag))

    def exact(self):
        return {"real": self.real.exact(), "imag": self.imag.exact()}


def as_c8(value):
    return value if isinstance(value, C8) else C8(value)


_h = make(0, 1, 1)
ROOTS = (C8(1), C8(_h, _h), C8(0, 1), C8(-_h, _h),
         C8(-1), C8(-_h, -_h), C8(0, -1), C8(_h, -_h))


def omega(exponent):
    return ROOTS[exponent % 8]


def matvec(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), C8()) for row in matrix)


def norm2(vector):
    return sum((value.abs2() for value in vector), make(0, 0, 0))
