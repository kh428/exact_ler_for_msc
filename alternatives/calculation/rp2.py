"""RP2 measured-record tensors; frozen inputs are read by the runner."""
from .pauli_polynomial import CircuitTensors, sqrt_mod

class RecordedPauliTensors(CircuitTensors):
    """Quantum Pauli indices and classical measurement bits in one network."""
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
                    lambda b: self.ring((-1)**((b[0]^b[1])*(b[2]^int(inverted))))/2)
        self.wires[site] = (None, zout)
        if axis == 'X':
            self.gate('H', [site])
        return result

    def feedback(self, record, site, axis):
        x, z = self.wire(site)
        bit = z if axis == 'X' else x
        self.factor((record, bit), lambda b: (-1)**(b[0]*b[1]))

    def expectation(self, pauli):
        for site in pauli:
            self.wire(site)
        overlap = 0
        for site, (x, z) in self.wires.items():
            axis = pauli.get(site, 'I')
            for bit, value in [(x, int(axis in 'XY')), (z, int(axis in 'ZY'))]:
                if bit is None:
                    if value:
                        self.net.multiply_scalar(0)
                else:
                    self.net.fixed(bit, value)
            overlap += axis == 'Y'
        if self.basis == 'xz':
            self.net.multiply_scalar(pow(sqrt_mod(self.ring.prime-1, self.ring.prime),
                                          -overlap, self.ring.prime))
        return self.net


def compile_rp2(model, ring, observable, *, selected=None, fixed=None, proxy=False, basis='xz', root2=None, noise_scales=None):
    compiler = RecordedPauliTensors(ring, root2=root2, basis=basis)
    records = {}
    # A fresh ancilla is traced/reset after each nondemolition MPP measurement.
    ancilla = 10000
    for event in model['events']:
        name = event['name']
        if name == 'NOISE':
            loc = event['location']
            compiler.noise(model['channels'][loc], selected=selected is None or loc in selected,
                           fixed=None if fixed is None else fixed.get(loc),
                           scale=1 if noise_scales is None else noise_scales.get(loc, 1))
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
            compiler.net.xor([records[i] for i in event['records']], event['reference_parity'])
        elif name.startswith('FEEDBACK_'):
            compiler.feedback(records[event['record']], event['support'][0], name[-1])
        else:
            compiler.gate(event.get('proxy_gate', name) if proxy else name, event['support'])
    pauli = {} if observable == 'A' else model['logical_'+observable.lower()]
    net = compiler.expectation({int(q): a for q, a in pauli.items()})
    stats = net.simplify()
    stats['noise_locations'] = compiler.channels
    stats['records_retained'] = len(records)
    return net, stats

