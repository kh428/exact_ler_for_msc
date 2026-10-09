"""Noise specifications: which edges are noisy, with which Pauli rates, and what happens to erasures.

A specification is a dict (or JSON file) with a model and a list of rules applied in order:

    {"model": "edges",
     "rules": [
        {"pauli": {"X": "1/3", "Y": "1/3", "Z": "1/3"}},                  every edge, uniform (the default)
        {"select": {"kinds": ["cnot"]}, "pauli": {"X": "1/2", "Z": "1/2"}},
        {"select": {"lines": [400, 470]}, "noiseless": true},
        {"select": {"roles": ["wire"]}, "erasure": {"rate": "1/2", "rule": "ignore"}}
     ]}

Rates are rational multiples of p. On an edge the Paulis are named as on the edge itself: for the edge inside a CNOT,
X is X on the target, Z is Z on the control and Y both; for the edge inside a CZ, X is Z on the target, Z is Z on the
control. A selection may use kinds (wire, cnot, cz), roles (wire, discard, into readout, inside), lines [first, last)
of the edge's first operation, qubits (any qubit of the edge), names, indices. A later rule overrides an earlier one.

Erasure of an edge at rate e = rate * p, with its flag
    ignore     the edge is replaced by a random qubit: X, Y, Z each with e/4, an independent location;
    discard    the shot is rejected: A and B gain the factor (1 - e) and P_L does not change.

The model "source" keeps the file's own noise instructions (the circuit-level model of the paper); its rules can only
make locations noiseless.

Every location is one factor 1 + x (sum_f w_f [f] + (1 - sum_f w_f) [I]) with x = p/(1-p), so that
A(p) = (1-p)^N * A-hat(x) and B(p) = (1-p)^N * B-hat(x) with N the number of locations.
"""
import json
from fractions import Fraction
from pathlib import Path

from . import edges as E

CODES = {'X': 1, 'Z': 2, 'Y': 3}
UNIFORM = {'X': Fraction(1, 3), 'Y': Fraction(1, 3), 'Z': Fraction(1, 3)}


def read(spec):
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    return dict(model=spec.get('model', 'edges'), rules=spec.get('rules', [{}]))


def selected(item, select):
    if not select:
        return True
    if 'kinds' in select and item['kind'] not in select['kinds']:
        return False
    if 'roles' in select and item.get('role') not in select['roles']:
        return False
    if 'lines' in select:
        lo, hi = select['lines']
        if not (lo <= item['line'] and (hi is None or item['line'] < hi)):
            return False
    if 'qubits' in select and not set(item['qubits']) & set(select['qubits']):
        return False
    if 'names' in select and item['name'] not in select['names']:
        return False
    if 'indices' in select and item['index'] not in select['indices']:
        return False
    return True


def locations(circuit, spec=None):
    """The noisy locations and the edges whose erasures are discarded.

    A location is a dict with kind 'law' (generators and weights of codes 1, 2, 3) or one of the source kinds
    'depolarizing', 'flip', 'record', and the index of the gate it follows.
    """
    spec = read(spec or {})
    if spec['model'] == 'source':
        out = []
        for loc in E.source_locations(circuit):
            noisy = True
            for rule in spec['rules']:
                if selected(loc, rule.get('select')) and 'noiseless' in rule:
                    noisy = not rule['noiseless']
            if noisy:
                out.append(loc)
        return out, []
    if spec['model'] != 'edges':
        raise ValueError(spec['model'])
    out, discard = [], []
    for edge in E.edges(circuit):
        pauli, erasure, noisy = dict(UNIFORM), None, True
        for rule in spec['rules']:
            if not selected(edge, rule.get('select')):
                continue
            if 'noiseless' in rule:
                noisy = not rule['noiseless']
            if 'pauli' in rule:
                pauli = {k: Fraction(v) for k, v in rule['pauli'].items()}
            if 'erasure' in rule:
                erasure = rule['erasure']
        if not noisy:
            continue
        weights = {CODES[k]: Fraction(v) for k, v in pauli.items() if Fraction(v)}
        if weights:
            out.append(dict(kind='law', edge=edge['index'], name=edge['name'], after=edge['after'], line=edge['line'],
                            qubits=edge['qubits'], generators=edge['generators'], weights=weights, role=edge['role']))
        if erasure and Fraction(erasure['rate']):
            rate = Fraction(erasure['rate'])
            if erasure['rule'] == 'ignore':
                out.append(dict(kind='law', edge=edge['index'], name=edge['name'] + ' (erasure)', after=edge['after'],
                                line=edge['line'], qubits=edge['qubits'], generators=edge['generators'],
                                weights={c: rate / 4 for c in (1, 2, 3)}, role=edge['role'], erasure=True))
            elif erasure['rule'] == 'discard':
                discard.append(dict(edge=edge['index'], name=edge['name'], rate=rate))
            else:
                raise ValueError(erasure['rule'])
    return out, discard


def law_value(weights, bits):
    """sum_f w_f chi_f + w_0 at a label whose anticommutation with the two generators is bits = (b0, b1)."""
    mask = bits[0] | (bits[1] << 1)
    total = sum(weights.values())
    return (1 - total) + sum(w * (-1) ** bin(code & mask).count('1') for code, w in weights.items())


def law_norm(locations):
    """The largest sum_f |w_f| + |1 - sum_f w_f| over the locations: 1 when every law is a sub-probability.

    Each coefficient of x^k then obeys |A[k]|, |B[k]| <= C(N, k) norm^k."""
    norm = 1
    for loc in locations:
        if loc['kind'] == 'law':
            total = sum(loc['weights'].values())
            norm = max(norm, sum(abs(w) for w in loc['weights'].values()) + abs(1 - total))
    return norm
