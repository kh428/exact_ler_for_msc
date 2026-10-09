"""The edges of a circuit's ZX diagram, each a place where a fault can happen, and the source file's own noise.

A wire edge joins two consecutive operations on one qubit. A measurement ends its wire and a reset starts a new one,
so nothing joins a measurement to the reset after it; a qubit reset while still in use has an edge from its last
operation to the reset ("discard"). A CNOT has one more edge, between its two spiders: X on it is X on the target, Z on
it is Z on the control, Y both. A CZ is two green spiders joined by a Hadamard edge: Z on either qubit, or both.

Edges end at the start of the noiseless tail (by default the first MPP, with the T layer just before it): the edges
that enter the tail ("into readout") are noisy, the edges inside it are not listed.

Every edge has a stable name, two generator Paulis and three outcomes (codes 1, 2, 3 = first, second, both).
"""
from .circuit import touched

PAULIS = ('X', 'Y', 'Z')


def tail_start(circuit):
    """The first gate of the noiseless tail: the first MPP, moved back over a T layer just before it."""
    gates = circuit['gates']
    product = next((i for i, g in enumerate(gates) if g['op'] == 'MPP'), len(gates))
    start = product
    while start and gates[start - 1]['op'] in ('T', 'T_DAG') and gates[start - 1]['line'] == gates[product - 1]['line']:
        start -= 1
    return start


def name(gates, g):
    gate = gates[g]
    return f"{gate['op']}({','.join(map(str, touched(gate)))})@{gate['line']}"


def edges(circuit, start=None):
    gates = circuit['gates']
    start = tail_start(circuit) if start is None else start
    out, open_end = [], {}
    for g, gate in enumerate(gates):
        if gate['op'] == 'FEEDBACK':                 # classical frame updates are not spiders of the circuit
            continue
        for q in touched(gate):
            prev = open_end.get(q)
            if prev is not None and prev < start:
                role = 'discard' if gate['op'] in ('R', 'RX') else 'into readout' if g >= start else 'wire'
                out.append(dict(kind='wire', role=role, qubits=[q], after=prev, before=g, line=gates[prev]['line'],
                                name=f"q{q}:{name(gates, prev)}->{name(gates, g)}", generators=[{q: 'X'}, {q: 'Z'}]))
            open_end[q] = None if gate['op'] in ('M', 'MX') else g
        if gate['op'] in ('CX', 'CZ') and g < start:
            c, t = gate['c'], gate['t']
            out.append(dict(kind='cnot' if gate['op'] == 'CX' else 'cz', role='inside', qubits=[c, t], after=g,
                            before=g, line=gate['line'], name=f"inside {name(gates, g)}",
                            generators=[{t: 'X'}, {c: 'Z'}] if gate['op'] == 'CX' else [{t: 'Z'}, {c: 'Z'}]))
    for i, e in enumerate(out):
        e['index'] = i
    return out


def outcome(edge, code):
    """The Pauli of outcome code 1, 2 or 3 of an edge, as {qubit: 'X' | 'Y' | 'Z'}."""
    pauli = {}
    for j, gen in enumerate(edge['generators']):
        if code >> j & 1:
            for q, s in gen.items():
                pauli[q] = multiply(pauli.get(q, 'I'), s)
    return {q: s for q, s in pauli.items() if s != 'I'}


def multiply(a, b):
    """The product of two one-qubit Paulis, up to phase."""
    bits = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    x, z = bits[a][0] ^ bits[b][0], bits[a][1] ^ bits[b][1]
    return {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[(x, z)]


def source_locations(circuit):
    """The noise instructions of the source file, the circuit-level model of the paper, as locations."""
    out = []
    for n in circuit['source_noise']:
        if n['kind'] == 'record':
            out.append(dict(kind='record', record=n['record'], after=n['after'], line=n['line'], qubits=[],
                            name=f"record {n['record']} flip @{n['line']}"))
        elif n['kind'] == 'flip':
            out.append(dict(kind='flip', axis=n['axis'], qubits=n['support'], after=n['after'], line=n['line'],
                            name=f"{n['axis']}_ERROR({n['support'][0]})@{n['line']}"))
        else:
            out.append(dict(kind='depolarizing', qubits=n['support'], after=n['after'], line=n['line'],
                            name=f"DEPOLARIZE{len(n['support'])}({','.join(map(str, n['support']))})@{n['line']}"))
    for i, loc in enumerate(out):
        loc['index'] = i
    return out


def main():
    """List the edges of a circuit: index, kind, role, source line, name.

      python -B -m zxedge.edges clifft_d3_p001.stim
    """
    import sys
    from pathlib import Path
    from . import INPUTS
    from .circuit import load
    name_ = sys.argv[1] if len(sys.argv) > 1 else 'clifft_d3_p001.stim'
    path = Path(name_) if Path(name_).exists() else INPUTS / name_
    circuit = load(path)
    out = edges(circuit)
    start = tail_start(circuit)
    print(f'{path.name}: {len(out)} edges; noiseless tail from gate {start} (line {circuit["gates"][start]["line"]})')
    for e in out:
        print(f"{e['index']:5d}  {e['kind']:5s}  {e['role']:12s}  line {e['line']:4d}  {e['name']}")


if __name__ == '__main__':
    main()
