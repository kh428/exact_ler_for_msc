"""Parse the frozen Stim-like sources into a flat, noiseless operation list.

Only the ideal structure is kept: preparations, phase gates (Clifford and T),
CNOTs, record-controlled Paulis, measurements (M, MX, MPP), detectors and
observables. Noise channels, measurement flip probabilities, coordinates and
TICKs are dropped, because a check is a property of the ideal circuit.

Record indices are absolute and chronological. A qubit number never names a
record: several qubits are measured more than once.
"""

import hashlib
import re

from . import REPO

# The paper's inputs, SOFT's d=3 and d=5 cultivation circuits. The d=3 file has the same
# operations as ../pauli_semiwebs/circuits/d3_native_t.txt up to its noiseless readout,
# which projects onto the magic state. The d=5 file equals ../pauli_semiwebs/circuits/d5_native_t.stim.
SOURCES = {3: REPO / 'data' / 'inputs' / 'clifft_d3_p001.stim',
           5: REPO / 'data' / 'inputs' / 'soft_cultivation_d5_p0005.stim'}

# Diagonal phase gates, in units of pi/4.
PHASE = {'I': 0, 'T': 1, 'S': 2, 'Z': 4, 'S_DAG': 6, 'T_DAG': 7}
IGNORED = {'TICK', 'QUBIT_COORDS', 'SHIFT_COORDS', 'DEPOLARIZE1', 'DEPOLARIZE2',
           'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'PAULI_CHANNEL_1', 'PAULI_CHANNEL_2'}
LINE = re.compile(r'^([A-Z_0-9]+)(\([^)]*\))?\s*(.*)$')


def _record(token, count):
    if not (token.startswith('rec[-') and token.endswith(']')):
        raise ValueError(f'Expected a rec[-k] target, got {token}')
    index = count + int(token[4:-1])
    if not 0 <= index < count:
        raise ValueError(f'{token} points outside the {count} earlier records')
    return index


def parse_text(text):
    ops, records, detectors, observables, tsites = [], [], [], {}, []
    for number, raw in enumerate(text.split('\n'), 1):
        line = raw.split('#')[0].strip()
        if not line:
            continue
        name, parens, rest = LINE.match(line).groups()
        args = rest.split()
        if name in IGNORED:
            continue
        if name in ('R', 'RX'):
            ops += [dict(op=name, q=int(q), line=number) for q in args]
        elif name in PHASE:
            for q in args:
                site = None
                if name in ('T', 'T_DAG'):
                    site = len(tsites)
                    tsites.append(dict(line=number, q=int(q), phase=PHASE[name]))
                ops.append(dict(op='PHASE', gate=name, q=int(q), phase=PHASE[name],
                                site=site, line=number))
        elif name in ('CX', 'CZ'):
            if len(args) % 2:
                raise ValueError(f'Odd target count at line {number}')
            for a, b in zip(args[::2], args[1::2]):
                if a.startswith('rec['):
                    ops.append(dict(op='FEED' + name[1], record=_record(a, len(records)),
                                    q=int(b), line=number))
                elif b.startswith('rec['):
                    raise ValueError(f'Unsupported record target order at line {number}')
                else:
                    ops.append(dict(op=name, c=int(a), t=int(b), line=number))
        elif name in ('M', 'MX'):
            for q in args:
                if q.startswith('!'):
                    raise ValueError(f'Inverted measurement at line {number}')
                index = len(records)
                records.append(dict(index=index, kind=name, target=q, line=number))
                ops.append(dict(op=name, q=int(q), record=index, line=number))
        elif name == 'MPP':
            for group in args:
                factors = [(p[0], int(p[1:])) for p in group.split('*')]
                if any(p not in 'XYZ' for p, _ in factors) or len({p for p, _ in factors}) != 1:
                    raise ValueError(f'Only pure X, Y or Z products are supported: {group}')
                index = len(records)
                records.append(dict(index=index, kind='MPP', target=group, line=number))
                ops.append(dict(op='MPP', factors=factors, record=index, line=number))
        elif name == 'DETECTOR':
            detectors.append(dict(index=len(detectors), line=number,
                                  records=sorted(_record(a, len(records)) for a in args)))
        elif name == 'OBSERVABLE_INCLUDE':
            key = int(parens.strip('()'))
            observables.setdefault(key, []).extend(_record(a, len(records)) for a in args)
        else:
            raise ValueError(f'Unsupported operation {name} at line {number}')
    return dict(ops=ops, records=records, detectors=detectors,
                observables={k: sorted(v) for k, v in observables.items()}, tsites=tsites)


def load(distance):
    path = SOURCES[distance]
    data = path.read_bytes()
    parsed = parse_text(data.decode())
    parsed.update(distance=distance, source=str(path.relative_to(REPO)),
                  source_sha256=hashlib.sha256(data).hexdigest(), text=data.decode())
    return parsed


def substituted_text(text, tsites, phases, *, tagged=False):
    """Return the source with T site k replaced by the diagonal gate of phase phases[k].

    Phases are in units of pi/4 and must be Clifford (even) for Stim. With tagged=True
    each site gets its own tag, so Stim never fuses neighbouring sites into one
    instruction and instruction numbering is the same for every substitution.
    """
    names = {0: 'I', 2: 'S', 4: 'Z', 6: 'S_DAG'}
    by_line = {}
    for k, (site, phase) in enumerate(zip(tsites, phases)):
        tag = f'[t{k}]' if tagged else ''
        by_line.setdefault(site['line'], []).append(f"{names[phase % 8]}{tag} {site['q']}")
    out = []
    for number, line in enumerate(text.split('\n'), 1):
        out.extend(by_line[number]) if number in by_line else out.append(line)
    return '\n'.join(out)
