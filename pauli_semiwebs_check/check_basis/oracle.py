"""Independent Stim oracle for phase-independent record parities.

Replace every T or T-dagger by one of the diagonal Clifford gates I, S, Z, S_dag
(phases 0, 2, 4, 6 in units of pi/4). Each substituted circuit is a stabilizer
circuit, so Stim samples it exactly: every record parity is either constant or
uniformly random.

Soundness. Each T gate enters the amplitude of a fixed record sequence with at
most one factor exp(i*theta), so every record-parity probability is a trigonometric
polynomial with frequencies -1, 0, 1 in each theta. If a parity is constant, with
the same value, at the four Clifford angles of every T gate, it is constant at every
angle. The space returned here is the set of checks that hold whatever phase sits at
each T gate. A finite sample of substitutions and shots can only make the reported
space too large, never too small.
"""

import random

import numpy as np
import stim

from src.linear import kernel, rank
from .circuit import substituted_text

CLIFFORD = (0, 2, 4, 6)


def proxy_phases(tsites):
    """S proxy: T -> S and T_dag -> S_dag."""
    return [2 if s['phase'] == 1 else 6 for s in tsites]


def assignments(tsites, seed, random_count):
    """The proxy, all-identity, every single-site change and random assignments."""
    rng = random.Random(seed)
    proxy = proxy_phases(tsites)
    out = [('proxy', proxy), ('identity', [0] * len(tsites))]
    for k in range(len(tsites)):
        for v in CLIFFORD:
            if v != proxy[k]:
                a = list(proxy)
                a[k] = v
                out.append((f'site{k}={v}', a))
    out += [(f'random{r}', [rng.choice(CLIFFORD) for _ in tsites]) for r in range(random_count)]
    return out


def sample_records(text, tsites, phases, shots):
    """Noiseless record samples, each packed as an int with bit j = record j."""
    circuit = stim.Circuit(substituted_text(text, tsites, phases)).without_noise()
    samples = circuit.compile_sampler().sample(shots).astype(np.uint8)
    packed = [int(''.join(map(str, row[::-1])), 2) if len(row) else 0 for row in samples]
    return packed, samples.shape[1]


def deterministic_space(parsed, *, phase_sets=None, seed=20260929, random_count=300, shots=64):
    """Affine record constraints v.m = b holding in every sampled substitution."""
    text, tsites = parsed['text'], parsed['tsites']
    phase_sets = phase_sets or assignments(tsites, seed, random_count)
    reference, nrec, rows = None, None, []
    for _, phases in phase_sets:
        packed, nrec = sample_records(text, tsites, phases, shots)
        if reference is None:
            reference = packed[0]
        rows += [s ^ reference for s in packed]
    basis = kernel(rows, nrec)
    return dict(equations=[(v, (v & reference).bit_count() & 1) for v in basis],
                dimension=len(basis), records=nrec, assignments=len(phase_sets),
                shots_per_assignment=shots, reference=reference)


def contains(equations, nrec, v, b):
    aug = [e | (c << nrec) for e, c in equations]
    return rank(aug + [v | (b << nrec)]) == rank(aug)


def classify_detectors(parsed, space):
    """Which declared detectors lie in the given affine constraint space."""
    out = []
    for det in parsed['detectors']:
        v = sum(1 << r for r in det['records'])
        b = (v & space['reference']).bit_count() & 1
        out.append(dict(index=det['index'], records=det['records'],
                        inside=contains(space['equations'], space['records'], v, b)))
    return out


def discard_states(parsed, *, seed=7, random_trials=64):
    """Known pure state of every qubit reset while still in use.

    Returns {op_index: ('X' or 'Z', eigenvalue)}. The state must be the same for
    every Clifford substitution of the T gates that precede the reset, and must
    be an eigenstate of X or Z; otherwise a ValueError is raised.
    """
    rng = random.Random(seed)
    text, tsites, ops = parsed['text'], parsed['tsites'], parsed['ops']
    lines = text.split('\n')
    in_use, result = set(), {}
    for i, op in enumerate(ops):
        name = op['op']
        if name in ('R', 'RX'):
            if op['q'] in in_use:
                earlier = [s for s in tsites if s['line'] < op['line']]
                trials = [[0] * len(earlier)]
                for k in range(len(earlier)):
                    for v in CLIFFORD:
                        a = [0] * len(earlier)
                        a[k] = v
                        trials.append(a)
                trials += [[rng.choice(CLIFFORD) for _ in earlier] for _ in range(random_trials)]
                cut = '\n'.join(lines[:op['line'] - 1])
                seen = set()
                for phases in trials:
                    sim = stim.TableauSimulator()
                    sim.do(stim.Circuit(substituted_text(cut, earlier, phases)).without_noise())
                    x, z = sim.peek_x(op['q']), sim.peek_z(op['q'])
                    seen.add(('X', x) if x else ('Z', z) if z else None)
                if len(seen) != 1 or None in seen:
                    raise ValueError(f"Qubit {op['q']} has no single known state before line {op['line']}: {seen}")
                result[i] = next(iter(seen))
            in_use.discard(op['q'])
        elif name in ('PHASE', 'FEEDX', 'FEEDZ'):
            in_use.add(op['q'])
        elif name in ('CX', 'CZ'):
            in_use.update((op['c'], op['t']))
        elif name in ('M', 'MX'):
            in_use.discard(op['q'])
        elif name == 'MPP':
            in_use.update(q for _, q in op['factors'])
    return result
