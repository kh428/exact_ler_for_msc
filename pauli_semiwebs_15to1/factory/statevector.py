"""The factory as five qubits, without the ZX diagram.

Each wire starts in |+>. A T state that acts on the set S of wires multiplies a computational
basis state by w^(phase * parity of its bits on S), with w = exp(i pi/4). The X measurement
of wire i with record r_i projects onto <0| + (-1)^(r_i) <1|. Only the supports of the fifteen
T states are taken from the diagram. Entries are exact, in the form of tensor.py.
"""


def output(supports, phases, records):
    """The unnormalised state of wire 0 for one record pattern."""
    counts = [[0] * 8, [0] * 8]
    for x in range(32):
        exponent = 0
        for support, phase in zip(supports, phases):
            exponent += phase * (sum((x >> w) & 1 for w in support) % 2)
        for wire in range(1, 5):
            exponent += 4 * ((x >> wire) & 1) * ((records >> (wire - 1)) & 1)
        counts[x & 1][exponent % 8] += 1
    return [[c[j] - c[j + 4] for j in range(4)] for c in counts]


def live(supports, phases):
    """The record patterns that occur."""
    return [r for r in range(16) if any(any(entry) for entry in output(supports, phases, r))]
