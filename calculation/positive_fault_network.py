"""Nonnegative Pauli-fault network with a proved auxiliary CSS boundary.

This applies only to the frozen SOFT source-noise window and its check's
logical + eigenstate. The T-layer interference is encoded in the validated
boundary identity; arbitrary non-Clifford circuits are outside this domain.
"""

import itertools
import numpy as np
from .binary_factors import Network


def convolve(left, right):
    result = np.zeros(4)
    for a in range(4):
        for b in range(4):
            result[a ^ b] += left[a]*right[b]
    return result


class FaultNetwork:
    def __init__(self, qubits, network=None):
        self.network = Network() if network is None else network
        self.wires = {q: [self.network.variable(f"initial_error_x_{q}"),
                          self.network.variable(f"initial_error_z_{q}")] for q in qubits}
        self.pending = {}
        self.events = {"one_qubit": 0, "two_qubit": 0}
        self.index = 0
        for pair in self.wires.values():
            for variable in pair:
                self.network.fixed(variable)

    def local_noise(self, q, probabilities):
        self.pending[q] = convolve(self.pending.get(q, (1, 0, 0, 0)), probabilities)

    def source_channel(self, item, probability):
        if item["kind"] == "depol2":
            self.pair_noise(item["support"], probability)
            return
        q, = item["support"]
        if item["kind"] == "depol1":
            values = np.array([1-probability, probability/3, probability/3, probability/3])
        else:
            axis = "Z" if item["source_kind"] == "X_READOUT_FLIP" else item["source_kind"][0]
            values = np.zeros(4)
            values[0], values[{"X": 1, "Z": 2, "Y": 3}[axis]] = 1-probability, probability
        self.local_noise(q, values)

    def flush(self, qubits):
        for q in qubits:
            if q in self.pending:
                self.noise((q,), self.pending.pop(q))

    def noise(self, qubits, probabilities):
        probabilities = np.asarray(probabilities)
        if np.any(probabilities < 0) or abs(float(np.sum(probabilities))-1) > 2e-13:
            raise ValueError("Invalid nonnegative Pauli channel")
        if probabilities.flat[0] == 1 and np.count_nonzero(probabilities) == 1:
            return
        self.index += 1
        old = [v for q in qubits for v in self.wires[q]]
        new = []
        for q in qubits:
            pair = [self.network.variable(f"error_{axis}_{q}_after_noise_{self.index}") for axis in ("x", "z")]
            self.wires[q] = pair
            new += pair
        count = len(qubits)
        values = np.zeros((2,)*(4*count))
        for bits in itertools.product((0, 1), repeat=4*count):
            labels = tuple((bits[2*j] ^ bits[2*count+2*j]) +
                           2*(bits[2*j+1] ^ bits[2*count+2*j+1]) for j in range(count))
            values[bits] = probabilities[labels]
        self.network.add(old+new, values)
        self.events["one_qubit" if count == 1 else "two_qubit"] += 1

    def pair_noise(self, qubits, probability):
        c, t = qubits
        probabilities = np.full((4, 4), probability/15)
        probabilities[0, 0] = 1-probability
        for axis, q in enumerate((c, t)):
            if q not in self.pending:
                continue
            local = self.pending.pop(q)
            updated = np.zeros((4, 4))
            for a, b, e in itertools.product(range(4), repeat=3):
                index = (a ^ e, b) if axis == 0 else (a, b ^ e)
                updated[index] += probabilities[a, b]*local[e]
            probabilities = updated
        self.noise(tuple(qubits), probabilities)

    def cnot(self, c, t):
        self.flush((c, t))
        self.index += 1
        xc, zc = self.wires[c]
        xt, zt = self.wires[t]
        ox = self.network.variable(f"error_x_{t}_after_CX_{self.index}")
        oz = self.network.variable(f"error_z_{c}_after_CX_{self.index}")
        self.network.xor((ox, xt, xc))
        self.network.xor((oz, zc, zt))
        self.wires[t][0], self.wires[c][1] = ox, oz


def source_fault_network(model, probability, builder_factory=FaultNetwork, *, retain_check_flip=False):
    """Propagate the common source-noise frames before the code boundary."""
    if not 0 <= probability <= 1:
        raise ValueError("Invalid probability")
    data, ancillas = model["data"], model["specification"]["ancillas"]
    builder = builder_factory(sorted(data+ancillas))
    net = builder.network
    middle = next(i for i, op in enumerate(model["block"]) if op[0] == "POST")
    reset = next(i for i, op in enumerate(model["block"]) if op[0] == "RESET")
    insertions = {}
    for item in model["channels"]:
        cut = item["mapped_block_cut"]
        if middle < cut <= reset and model["specification"]["hub"] in item["support"]:
            raise ValueError("Retained hub noise between measurement and reset requires an explicit erasure rewrite")
        insertions.setdefault(cut, []).append(item)
    for cut in range(len(model["block"])+1):
        for item in insertions.get(cut, []):
            builder.source_channel(item, probability)
        if cut == len(model["block"]):
            break
        name, *args = model["block"][cut]
        if name == "CX":
            builder.cnot(*args)
        elif name == "POST":
            x, z, sign = args
            if z or sign != 1 or x != 1 << model["specification"]["hub"]:
                raise ValueError("Unexpected selected check")
            q = x.bit_length()-1
            builder.flush((q,))
            builder.check_flip = builder.wires[q][1]
            if not retain_check_flip:
                net.fixed(builder.check_flip)
        elif name not in ("T", "T_DAG", "RESET"):
            raise ValueError("Unexpected source block gate")
    builder.flush(tuple(builder.pending))
    return builder


def positive_network(model, probability, quantity="A", builder_factory=FaultNetwork):
    if quantity not in ("A", "B"):
        raise ValueError("Invalid quantity")
    target = sum(1 if name == "T" else -1 for name, q in model["specification"]["exit"]) % 8
    if target != 7:
        raise ValueError("This source-window builder assumes the verified SOFT logical phase 7")
    builder = source_fault_network(model, probability, builder_factory)
    net = builder.network
    data, ancillas = model["data"], model["specification"]["ancillas"]
    faces = model["faces"]
    a = [net.variable(f"output_X_face_{i}") for i in range(len(faces))]
    b = [net.variable(f"output_shifted_Z_face_{i}") for i in range(len(faces))]
    v = [net.variable(f"uniform_auxiliary_face_{i}") for i in range(len(faces))]
    lx = net.variable("output_logical_x")
    if quantity == "A":
        lz = net.variable("output_logical_z")
    and_values = np.zeros((2, 2, 2))
    for x, h in itertools.product((0, 1), repeat=2):
        and_values[x & h, x, h] = 1
    for q in data:
        x, z = builder.wires[q]
        incident = [i for i, face in enumerate(faces) if q in face]
        h = net.variable(f"auxiliary_codeword_at_{q}")
        product = net.variable(f"error_x_AND_auxiliary_at_{q}")
        net.xor([x]+[a[i] for i in incident]+[lx])
        net.xor([h]+[v[i] for i in incident])
        net.add((product, x, h), and_values)
        net.xor([z, product]+[b[i] for i in incident]+([lz] if quantity == "A" else []),
                parity=0 if quantity == "A" else 1)
    for q in ancillas:
        net.fixed(builder.wires[q][1])
    net.multiply_scalar(2.0**(-len(faces)))
    net.positive_metadata = {"convolved_noise_events": builder.events,
                             "source_channels": model["relevant_channels"], "auxiliary_bits": len(faces)}
    return net
