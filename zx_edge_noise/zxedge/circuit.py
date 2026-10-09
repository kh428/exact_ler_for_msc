"""Read a stim or clifft circuit into gates, its own noise instructions, detectors and the observable.

Supported: R, RX, M, MX (with an optional flip probability), MPP of X, Y or Z products, CX and CZ (a recorded control
is a frame update, FEEDBACK), H, S, S_DAG, T, T_DAG, X, Y, Z, TICK, DETECTOR, OBSERVABLE_INCLUDE. Drawing and
coordinate instructions (QUBIT_COORDS, SHIFT_COORDS, POLYGON) are skipped. The file's own noise instructions
(DEPOLARIZE1/2, X_ERROR, Y_ERROR, Z_ERROR and measurement flips) are kept as the "source" noise model, the
circuit-level model of the paper; the edge models ignore them.

Every gate carries the moment it belongs to (the number of TICKs before it) and its source line.
"""
import re

LINE = re.compile(r'^([A-Z_0-9]+)(\(([^)]*)\))?\s*(.*)$')
SKIPPED = {'QUBIT_COORDS', 'SHIFT_COORDS', 'POLYGON'}
ONE = ('R', 'RX', 'H', 'S', 'S_DAG', 'T', 'T_DAG', 'X', 'Y', 'Z')
ALIASES = {'RZ': 'R', 'MZ': 'M', 'SQRT_Z': 'S', 'SQRT_Z_DAG': 'S_DAG'}


def parse(text):
    gates, noise, detectors, observable = [], [], [], []
    records, moment = 0, 0
    for number, raw in enumerate(text.split('\n'), 1):
        line = raw.split('#')[0].strip()
        if not line:
            continue
        name, _, parens, rest = LINE.match(line).groups()
        name = ALIASES.get(name, name)
        args = rest.split()
        if name in SKIPPED:
            continue
        if name == 'TICK':
            moment += 1
            continue
        after = len(gates) - 1
        base = dict(line=number, moment=moment)
        if name in ONE:
            gates += [dict(base, op=name, q=int(q)) for q in args]
        elif name in ('CX', 'CNOT', 'CZ'):
            name = 'CX' if name == 'CNOT' else name
            for a, b in zip(args[::2], args[1::2]):
                if a.startswith('rec['):            # a Pauli applied when a recorded outcome is 1: a frame update
                    gates.append(dict(base, op='FEEDBACK', axis='X' if name == 'CX' else 'Z', q=int(b),
                                      record=records + int(a[4:-1])))
                else:
                    gates.append(dict(base, op=name, c=int(a), t=int(b)))
        elif name in ('M', 'MX'):
            for q in args:
                if parens:                           # the source's measurement flip: a classical flip of the record,
                    noise.append(dict(kind='record', record=records, after=len(gates), support=[], line=number,
                                      moment=moment))           # placed after the measurement gate itself
                gates.append(dict(base, op=name, q=int(q), record=records))
                records += 1
        elif name == 'MPP':
            for group in args:
                factors = [(f[0], int(f[1:])) for f in group.split('*')]
                assert len({p for p, _ in factors}) == 1, 'only pure X, Y or Z products'
                gates.append(dict(base, op='MPP', pauli=factors[0][0], qubits=[q for _, q in factors], record=records))
                records += 1
        elif name in ('X_ERROR', 'Y_ERROR', 'Z_ERROR'):
            noise += [dict(kind='flip', axis=name[0], support=[int(q)], after=after, line=number, moment=moment)
                      for q in args]
        elif name == 'DEPOLARIZE1':
            noise += [dict(kind='depolarizing', support=[int(q)], after=after, line=number, moment=moment) for q in args]
        elif name == 'DEPOLARIZE2':
            noise += [dict(kind='depolarizing', support=[int(a), int(b)], after=after, line=number, moment=moment)
                      for a, b in zip(args[::2], args[1::2])]
        elif name == 'DETECTOR':
            detectors.append(dict(records=sorted(records + int(a[4:-1]) for a in args), after=after, line=number))
        elif name == 'OBSERVABLE_INCLUDE':
            observable += [records + int(a[4:-1]) for a in args]
        else:
            raise ValueError(f'unsupported instruction {name} at line {number}')
    obs = sorted(set(r for r in observable if observable.count(r) % 2))
    return dict(gates=gates, source_noise=noise, detectors=detectors, observable=obs, records=records, moments=moment + 1)


def touched(gate):
    if gate['op'] in ('CX', 'CZ'):
        return [gate['c'], gate['t']]
    if gate['op'] == 'MPP':
        return gate['qubits']
    if gate['op'] in ('MPAD', 'INPUT', 'OUTPUT', 'COINS'):
        return []
    return [gate['q']]


def load(path):
    from pathlib import Path
    c = parse(Path(path).read_text())
    c['source'] = str(path)
    return c
