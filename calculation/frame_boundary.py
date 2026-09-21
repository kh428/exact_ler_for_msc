"""Exact CSS code restriction of D Q D†, retaining all interfering phases.

This evaluates a *code-boundary* matrix of a rotated Pauli frame. The sum is
over the X-stabilizer codewords, not over physical fault sets. The only early
rejection is an exact Z-syndrome test. The resulting 2x2 matrix is not assumed
unitary: its lost norm is leakage rejected by the ideal code projection.
"""

from functools import lru_cache
from .css_boundary import span, sign_violations
from .cyclotomic import C8, matvec, norm2, omega
from .dyadic import make


class CodeBoundary:
    def __init__(self, n, checks, flips=0):
        self.n, self.checks, self.flips = n, tuple(checks), flips
        self.all = (1 << n) - 1
        if not n % 2 or len(checks) * 2 != n - 1:
            raise ValueError("Expected a one-logical-qubit self-dual CSS code")
        if any((a & b).bit_count() % 2 for a in checks for b in checks):
            raise ValueError("CSS generators do not commute")
        if any(row >> n for row in checks) or flips >> n:
            raise ValueError("Mask outside the code")
        self.words = tuple(span(checks))
        if sign_violations(checks, flips):
            raise ValueError("The requested check pattern does not preserve the code")
        self.target_phase = (n - 2 * flips.bit_count()) % 8

    def signed_weight(self, mask):
        return mask.bit_count() - 2 * (mask & self.flips).bit_count()

    @lru_cache(maxsize=128)
    def phase_rows(self, x):
        constant = self.signed_weight(x)
        return tuple((word, (constant - 2 * self.signed_weight(x & word)) % 8)
                     for word in self.words)

    @lru_cache(maxsize=32768)
    def matrix(self, x, z):
        if (x | z) >> self.n or x < 0 or z < 0:
            raise ValueError("Frame mask outside the code")
        zero = C8()
        if any((x & row).bit_count() % 2 for row in self.checks):
            return ((zero, zero), (zero, zero))
        counts = [[0] * 8, [0] * 8]
        pauli_phase = 2 * (x & z).bit_count()
        logical_z_phase = 4 * (z.bit_count() % 2)
        for word, diagonal_phase in self.phase_rows(x):
            exponent = pauli_phase + 4 * ((z & word).bit_count() % 2)
            counts[0][(exponent + diagonal_phase) % 8] += 1
            counts[1][(exponent + logical_z_phase - diagonal_phase) % 8] += 1
        entries = [sum((count * omega(k) for k, count in enumerate(row)), C8()) / len(self.words)
                   for row in counts]
        if x.bit_count() % 2:
            return ((zero, entries[1]), (entries[0], zero))
        return ((entries[0], zero), (zero, entries[1]))

    def initial(self, phase):
        h = make(0, 1, 1)
        return (C8(h), omega(phase) * h)

    def check(self, vector, sign=1):
        if sign not in (-1, 1):
            raise ValueError("Check sign must be +/-1")
        phase = self.target_phase
        return ((vector[0] + sign * omega(-phase) * vector[1]) / 2,
                (vector[1] + sign * omega(phase) * vector[0]) / 2)

    def instrument(self, x, z, sign=1, initial_phase=None):
        """Unnormalised acceptance and bad probability for one fixed frame.

        The ideal code projection is included. Ancilla records must be
        checked by the caller; x,z refer to data only. The input is the
        equatorial logical state with the given eighth-root relative phase.
        """
        if initial_phase is None:
            initial_phase = self.target_phase
        checked = self.check(self.initial(initial_phase), sign)
        output = matvec(self.matrix(x, z), checked)
        return norm2(output), norm2(self.check(output, -1))
