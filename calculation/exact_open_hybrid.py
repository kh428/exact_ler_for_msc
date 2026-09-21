"""The open Pauli-check response built directly from rational probabilities.

This duplicates the inspected open_hybrid_boundary geometry, preserving its
variable names/order so saved scope plans can be checked and reused. Channel
kernels, pending-channel convolutions and all normalisations are rational
before any modulus is chosen. No rounded boundary table is an input.
"""
from fractions import Fraction
import itertools
import numpy as np
from .character_surrogate import parity_preimages
from .frame_boundary import CodeBoundary
from .incoming_positive import verify_code
from .hybrid_pauli_network import HybridFaultNetwork
from .rational_binary_network import RationalNetwork, rational


class ExactSourceHybrid(HybridFaultNetwork):
    # Only the inspected combinatorial CNOT and character-insertion methods
    # are inherited. Every method that constructs coefficients is replaced.
    def __init__(self, qubits):
        self.network = RationalNetwork()
        self.wires, self.pending = {}, {}
        self.events = {'one_qubit': 0, 'two_qubit': 0}
        self.index = 0
        for q in qubits:
            pair = [self.network.variable(f'initial_hybrid_{axis}_{q}') for axis in ('x', 'z_character')]
            self.wires[q] = pair
            self.network.fixed(pair[0])

    def mixed_noise(self, qubits, mixed):
        self.index += 1
        count = len(qubits)
        old = [self.wires[q][0] for q in qubits]
        characters = [self.wires[q][1] for q in qubits]
        new = [self.network.variable(f'hybrid_x_{q}_after_noise_{self.index}') for q in qubits]
        table = np.zeros((2,)*(3*count), dtype=object)
        for bits in itertools.product((0, 1), repeat=3*count):
            ex = tuple(bits[j] ^ bits[count+j] for j in range(count))
            table[bits] = mixed[ex+bits[2*count:]]
        self.network.add(old+new+characters, table)
        for q, v in zip(qubits, new):
            self.wires[q][0] = v
        self.events['one_qubit' if count == 1 else 'two_qubit'] += 1

    def source_channel(self, item, p):
        p = rational(p)
        if item['kind'] == 'depol2':
            c, t = item['support']
            mixed = np.zeros((2, 2, 2, 2), dtype=object)
            mixed[0, 0, :, :] = 1-16*p/15
            mixed[:, :, 0, 0] = 4*p/15
            mixed[0, 0, 0, 0] = 1-4*p/5
            for axis, q in enumerate((c, t)):
                if q not in self.pending:
                    continue
                local = self.pending.pop(q)
                updated = np.zeros_like(mixed)
                for x0, x1, v0, v1 in itertools.product((0, 1), repeat=4):
                    for e in (0, 1):
                        source = (x0 ^ e, x1, v0, v1) if axis == 0 else (x0, x1 ^ e, v0, v1)
                        updated[x0, x1, v0, v1] += mixed[source]*local[e, (v0, v1)[axis]]
                mixed = updated
            self.mixed_noise((c, t), mixed)
            return
        if item['kind'] == 'depol1':
            mixed = np.array([[1-2*p/3, 1-4*p/3], [2*p/3, 0]], dtype=object)
        else:
            axis = 'Z' if item['source_kind'] == 'X_READOUT_FLIP' else item['source_kind'][0]
            if axis == 'X':
                mixed = np.array([[1-p, 1-p], [p, p]], dtype=object)
            elif axis == 'Z':
                mixed = np.array([[1, 1-2*p], [0, 0]], dtype=object)
            elif axis == 'Y':
                mixed = np.array([[1-p, 1-p], [p, -p]], dtype=object)
            else:
                raise ValueError('Unknown source Pauli axis')
        q, = item['support']
        previous = self.pending.get(q, np.array([[1, 1], [0, 0]], dtype=object))
        self.pending[q] = np.array([previous[0]*mixed[0]+previous[1]*mixed[1],
                                    previous[0]*mixed[1]+previous[1]*mixed[0]], dtype=object)

    def flush(self, qubits):
        for q in qubits:
            if q in self.pending:
                self.mixed_noise((q,), self.pending.pop(q))


def exact_open_hybrid(model, probability, component=0):
    p = rational(probability)
    if component not in (0, 1) or not 0 <= p < Fraction(1, 2):
        raise ValueError('Invalid exact probability or component')
    data, ancillas = model['data'], model['specification']['ancillas']
    n, r = len(data), len(model['checks'])
    verify_code(CodeBoundary(n, model['checks'], model['flips']))
    preimages = parity_preimages(tuple(model['checks'])+((1 << n)-1,), n)
    generators = [preimages[1 << i] for i in range(r+1)]
    assert all(preimages[word] == _xor(g for i, g in enumerate(generators) if word >> i & 1)
               for word in range(1 << (r+1)))
    builder = ExactSourceHybrid(sorted(data+ancillas))
    net = builder.network
    ys = [net.variable(f'incoming_Z_character_{i}') for i in range(r+1)]
    incoming = [net.variable(f'incoming_X_label_{i}') for i in range(r+1)]
    insertions = {}
    for item in model['channels']:
        insertions.setdefault(item['mapped_block_cut'], []).append(item)
    for cut in range(len(model['block'])+1):
        for item in insertions.get(cut, []):
            builder.source_channel(item, p)
        if cut == len(model['block']):
            break
        name, *args = model['block'][cut]
        if name == 'CX':
            builder.cnot(*args)
        elif name == 'POST':
            x, z, sign = args
            if z != 0 or sign != 1 or x != 1 << model['specification']['hub']:
                raise ValueError('Unexpected selected check')
            builder.character_insertion(model['specification']['hub'], ys[-1], component)
        elif name not in ('T', 'T_DAG', 'RESET'):
            raise ValueError('Unexpected source operation')
    builder.flush(tuple(builder.pending))
    xs = [net.variable(f'hybrid_X_face_{i}') for i in range(r)]
    ws = [net.variable(f'hybrid_kernel_face_{i}') for i in range(r)]
    lx = net.variable('hybrid_X_logical_coset')
    and_table = np.zeros((2, 2, 2), dtype=object)
    for left, right in itertools.product((0, 1), repeat=2):
        and_table[left & right, left, right] = 1
    for i, q in enumerate(data):
        x, vz = builder.wires[q]
        incident = [j for j, face in enumerate(model['faces']) if q in face]
        s = net.variable(f'hybrid_code_character_at_{q}')
        product = net.variable(f'hybrid_kernel_product_at_{q}')
        a = net.variable(f'incoming_X_representative_at_{q}')
        net.xor([a]+[incoming[j] for j, g in enumerate(generators) if g >> i & 1])
        net.xor([s]+[ys[j] for j in incident])
        net.xor((vz, s), parity=component)
        net.xor([x, lx, a]+[xs[j] for j in incident])
        net.add((product, x, s), and_table)
        central_product = net.variable(f'incoming_X_times_central_at_{q}')
        net.add((central_product, a, ys[-1]), and_table)
        net.xor([product, central_product]+[ws[j] for j in incident])
        phase = np.empty((2, 2, 2, 2), dtype=object)
        for bx, bs, bc, ba in itertools.product((0, 1), repeat=4):
            phase[bx, bs, bc, ba] = (-1)**(ba & bx & (bs ^ bc))
        net.add((x, s, ys[-1], a), phase)
    net.multiply_scalar(Fraction(1, 1 << len(ancillas)))
    net.boundary_y, net.boundary_a = ys, incoming
    net.source_metadata = {'probability': str(p), 'component': component,
                           'source_channels': len(model['channels']), 'events': builder.events,
                           'ancilla_normalization': str(Fraction(1, 1 << len(ancillas))),
                           'coefficient_arithmetic': 'rational before any reduction modulo q'}
    return net


def _xor(values):
    value = 0
    for item in values:
        value ^= item
    return value
