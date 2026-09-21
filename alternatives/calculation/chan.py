"""Recorded-measurement tensors for the flagged Chan d=3 source."""
from .pauli_polynomial import CircuitTensors

class RecordedPauliTensors(CircuitTensors):
    """Quantum Pauli indices and classical measurement bits in one network (as in rp2_polynomial)."""
    def measure(self, site, axis, inverted=False):
        if axis == 'X':
            self.gate('H', [site])
        elif axis != 'Z':
            raise ValueError(axis)
        x, z = self.wire(site)
        self.zero(x)
        result = self.variable('measurement-record')
        zout = self.variable('measured-z')
        self.factor((z, zout, result),
                    lambda b: self.ring((-1) ** ((b[0] ^ b[1]) * (b[2] ^ int(inverted)))) / 2)
        self.wires[site] = (None, zout)
        if axis == 'X':
            self.gate('H', [site])
        return result

    def feedback(self, record, site, axis):
        x, z = self.wire(site)
        bit = z if axis == 'X' else x
        self.factor((record, bit), lambda b: (-1) ** (b[0] * b[1]))

    def parity_constraint(self, bits, parity, label):
        """Constrain XOR(bits) = parity, chaining long parities through auxiliary bits."""
        acc = None
        for bit in bits:
            acc = self.xor(acc, bit, label) if acc is not None else bit
        if acc is None:
            assert parity == 0, 'Empty parity constraint with odd parity'
            return
        self.net.fixed(acc, parity)

    def trace(self):
        for site, (x, z) in self.wires.items():
            for bit in (x, z):
                if bit is not None:
                    self.net.fixed(bit, 0)
        return self.net


def compile_chan(model, ring, observable, *, selected=None, root2=None, basis='xz'):
    compiler = RecordedPauliTensors(ring, root2=root2, basis=basis)
    records = {}
    ancilla = 10000
    for event in model['events']:
        name = event['name']
        if name == 'NOISE':
            loc = event['location']
            compiler.noise(model['channels'][loc], selected=selected is None or loc in selected)
        elif name == 'MEASURE':
            records[event['record']] = compiler.measure(event['support'][0], event['axis'], event['inverted'])
        elif name == 'MEASURE_PRODUCT':
            x_axis = event['axis'] == 'X'
            compiler.gate('RX' if x_axis else 'R', [ancilla])
            for q in event['support']:
                compiler.gate('CX', [ancilla, q] if x_axis else [q, ancilla])
            records[event['record']] = compiler.measure(ancilla, event['axis'])
            compiler.gate('R', [ancilla])
        elif name == 'DETECTOR':
            compiler.parity_constraint([records[i] for i in event['records']],
                                       event['reference_parity'], 'detector')
        elif name.startswith('FEEDBACK_'):
            compiler.feedback(records[event['record']], event['support'][0], name[-1])
        else:
            compiler.gate(name, event['support'])
    if observable == 'B':
        compiler.parity_constraint([records[i] for i in model['observable_records']],
                                   model['observable_reference_parity'] ^ 1, 'observable')
    elif observable != 'A':
        raise ValueError(observable)
    net = compiler.trace()
    stats = net.simplify()
    stats['noise_locations'] = compiler.channels
    stats['records_retained'] = len(records)
    return net, stats

