"""ZX amplitude evaluation by cutting, collection and stabiliser decomposition.

Phases are rational multiples of pi. Scalar amplitudes use complex floating
point; probabilities are checked to absolute tolerance 1e-12.
"""
import os, sys, time, json, re, argparse
from fractions import Fraction
from copy import deepcopy
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pyzx_param as zx
from pyzx_param.graph.scalar import Scalar
from pyzx_param.circuit.gates import ZPhase, XPhase, HAD, CNOT
from .cutting import cut_spider, can_cut, tcount as tcount_g

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- circuit -> ZX amplitude
def parse(text):
    ops = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, *targets = line.split()
        name = re.sub(r'\(.*\)', '', name)
        ops.append((name, [int(t) for t in targets]))
    return ops


def amplitude_graph(text, outcomes):
    """ZX scalar diagram of <outcomes| circuit |resets>.

    Each reset opens a fresh wire (state |0> for R, |+> for RX); each measurement closes the
    wire with the effect for its outcome bit (M: 0->|0>, 1->|1>; MX: 0->|+>, 1->|->), taken in
    measurement order from `outcomes`. Every wire must be measured exactly once.
    """
    ops = parse(text)
    wire_of = {}            # qubit -> open wire index
    states, effects = [], {}
    gates = []              # (gate constructor, wire args)
    n_meas = 0
    for name, tg in ops:
        if name in ('R', 'RX'):
            for q in tg:
                if q in wire_of:
                    raise ValueError(f'reset of unmeasured qubit {q}: a partial trace is not an amplitude')
                wire_of[q] = len(states)
                states.append('0' if name == 'R' else '+')
        elif name in ('M', 'MX'):
            for q in tg:
                w = wire_of.pop(q)
                b = int(outcomes[n_meas]); n_meas += 1
                effects[w] = ('0' if b == 0 else '1') if name == 'M' else ('+' if b == 0 else '-')
        elif name == 'CX':
            for c, t in zip(tg[::2], tg[1::2]):
                gates.append(CNOT(wire_of[c], wire_of[t]))
        else:
            for q in tg:
                w = wire_of[q]
                if name == 'H':
                    gates.append(HAD(w))
                elif name == 'T':
                    gates.append(ZPhase(w, Fraction(1, 4)))
                elif name == 'T_DAG':
                    gates.append(ZPhase(w, Fraction(7, 4)))
                elif name == 'S':
                    gates.append(ZPhase(w, Fraction(1, 2)))
                elif name == 'S_DAG':
                    gates.append(ZPhase(w, Fraction(3, 2)))
                elif name == 'Z':
                    gates.append(ZPhase(w, Fraction(1)))
                elif name == 'X':
                    gates.append(XPhase(w, Fraction(1)))
                elif name == 'Y':                       # Y = i X Z, the global phase is irrelevant
                    gates.append(ZPhase(w, Fraction(1)))
                    gates.append(XPhase(w, Fraction(1)))
                else:
                    raise ValueError(f'unsupported instruction {name}')
    if wire_of:
        raise ValueError(f'unmeasured wires at the end: {wire_of}')
    if n_meas != len(outcomes):
        raise ValueError(f'{n_meas} measurements but {len(outcomes)} outcome bits')
    W = len(states)
    circ = zx.Circuit(W)
    for gt in gates:
        circ.add_gate(gt)
    g = circ.to_graph()
    # apply_state / apply_effect address inputs and outputs in the order g.inputs() / g.outputs()
    state_str = ''.join(states[g.qubit(v)] for v in g.inputs())
    effect_str = ''.join(effects[g.qubit(v)] for v in g.outputs())
    g.apply_state(state_str)
    g.apply_effect(effect_str)
    return g, W, n_meas


# ----------------------------------------------------------------------------- decomposition
def fold_scalar(g):
    """Return g.scalar as a complex number and reset g.scalar to one."""
    val = g.scalar.to_number()
    g.scalar = Scalar()
    return val


def signature(g):
    """Structure-and-phase key of a diagram (scalar excluded): identical keys mean identical
    tensors, so the terms can be summed. Same idea as _graph_signature in the accel repo."""
    verts = sorted(g.vertices())
    idx = {v: i for i, v in enumerate(verts)}
    vs = tuple((idx[v], g.type(v), g.phase(v)) for v in verts)
    es = tuple(sorted((min(idx[a], idx[b]), max(idx[a], idx[b]), g.edge_type(e))
                      for e in g.edges() for a, b in [g.edge_st(e)]))
    return (vs, es)


def reduce_term(g, w):
    """full_reduce, fold the scalar into the weight. Returns (g, w) or None if the term vanishes."""
    zx.full_reduce(g)
    if g.scalar.is_zero:
        return None
    w = w * fold_scalar(g)
    if abs(w) == 0.0:
        return None
    return g, w


def clifford_value(g):
    """Value of a T-count-0 diagram (after full_reduce it must be empty)."""
    if g.num_vertices() == 0:
        return 1.0 + 0j
    zx.full_reduce(g)
    if g.num_vertices() == 0:
        return fold_scalar(g)
    if g.num_vertices() <= 12:                 # small leftover: contract it directly
        return complex(zx.tensorfy(g)) * fold_scalar(g)
    raise RuntimeError(f'Clifford term did not reduce to a scalar: {g.num_vertices()} vertices')


def best_cut(g, max_candidates=None):
    """One-step lookahead: the spider whose cut gives the smallest total T-count of the two
    reduced branches. Ties: fewer surviving vertices."""
    best = None
    cands = [v for v in g.vertices() if g.type(v) in (1, 2) and can_cut(g, v)]
    for v in cands:
        gl, gr = cut_spider(g, v)
        zx.full_reduce(gl); zx.full_reduce(gr)
        tl = 0 if gl.scalar.is_zero else tcount_g(gl)
        tr = 0 if gr.scalar.is_zero else tcount_g(gr)
        score = (tl + tr, max(tl, tr), gl.num_vertices() + gr.num_vertices())
        if best is None or score < best[0]:
            best = (score, v)
    return best


def collect(terms):
    """Sum the weights of identical diagrams."""
    groups = {}
    for g, w in terms:
        key = signature(g)
        if key in groups:
            groups[key][1] += w
        else:
            groups[key] = [g, w]
    out = [(g, w) for g, w in groups.values() if abs(w) > 0.0]
    return out


def decompose_reduced(g, bss_tcount=6, log=print, weight=1.0 + 0j):
    """Cut with lookahead + collect like terms until every term has T-count <= bss_tcount,
    then finish each with pyzx's stabiliser decomposition. Returns (amplitude, stats)."""
    stats = dict(rounds=[], clifford_terms=0, bss_leaves=0)
    t = reduce_term(g, weight)
    if t is None:
        return 0j, stats
    terms = [t]
    amp = 0j
    rnd = 0
    while terms:
        rnd += 1
        # 1. finished terms
        done = [(h, w) for h, w in terms if tcount_g(h) == 0]
        for h, w in done:
            amp += w * clifford_value(h)
            stats['clifford_terms'] += 1
        terms = [(h, w) for h, w in terms if tcount_g(h) > 0]
        # 2. small residual T-count: BSS-type ladder (cat5 / two-T / one-T) from pyzx
        small = [(h, w) for h, w in terms if tcount_g(h) <= bss_tcount]
        for h, w in small:
            leaves = zx.simulate.find_stabilizer_decomp(h)
            for leaf in leaves:
                if leaf.scalar.is_zero:
                    continue
                wl = w * fold_scalar(leaf)
                amp += wl * clifford_value(leaf)
                stats['bss_leaves'] += 1
                stats['clifford_terms'] += 1
        terms = [(h, w) for h, w in terms if tcount_g(h) > bss_tcount]
        if not terms:
            break
        # 3. cut every remaining term at its best spider, reduce, collect
        new_terms = []
        tcs = sorted(tcount_g(h) for h, _ in terms)
        t0 = time.time()
        for h, w in terms:
            b = best_cut(h)
            if b is None:
                raise RuntimeError('no cuttable spider left')
            (score, v) = b
            gl, gr = cut_spider(h, v)
            for br in (gl, gr):
                r = reduce_term(br, w)
                if r is not None:
                    new_terms.append(r)
        n_before = len(new_terms)
        terms = collect(new_terms)
        info = dict(round=rnd, terms_in=len(tcs), tcounts_in=tcs, branches=n_before, terms_after_collect=len(terms),
                    tcounts_after=sorted(tcount_g(h) for h, _ in terms), seconds=round(time.time() - t0, 1))
        stats['rounds'].append(info)
        log(f"  round {rnd}: {len(tcs)} terms with T-counts {tcs} -> {n_before} branches -> {len(terms)} after collection, "
            f"T-counts {info['tcounts_after']}  ({info['seconds']}s)")
    return amp, stats


def fuse(g):
    """Spider fusion and identity removal only (the paper's pre-simplification, no full_reduce)."""
    zx.spider_simp(g, quiet=True)
    zx.id_simp(g, quiet=True)
    zx.spider_simp(g, quiet=True)
    return g


def reduced_tcount(g):
    h = deepcopy(g)
    zx.full_reduce(h)
    return 0 if h.scalar.is_zero else tcount_g(h)


def best_cut_fused(g, min_degree=3):
    """Lookahead on a fused diagram: cut candidate v, full_reduce both branches (on copies), score
    by the total T-count that survives. Returns (score, v, t_left, t_right)."""
    best = None
    cands = [v for v in g.vertices() if g.type(v) in (1, 2) and can_cut(g, v) and g.vertex_degree(v) >= min_degree]
    if not cands:
        cands = [v for v in g.vertices() if g.type(v) in (1, 2) and can_cut(g, v)]
    for v in cands:
        gl, gr = cut_spider(g, v)
        zx.full_reduce(gl); zx.full_reduce(gr)
        tl = 0 if gl.scalar.is_zero else tcount_g(gl)
        tr = 0 if gr.scalar.is_zero else tcount_g(gr)
        score = (tl + tr, max(tl, tr), gl.num_vertices() + gr.num_vertices())
        if best is None or score < best[0]:
            best = (score, v, tl, tr)
    return best


def decompose_paper(g, bss_tcount=6, max_cut_rounds=12, log=print, trace=None):
    """The paper's order of operations: fuse, choose cuts by lookahead on the fused diagram,
    collect identical diagrams after every cut, full_reduce the terms at the end, then finish
    the residual T-count with cutting on the reduced diagrams / pyzx's ladder (decompose)."""
    stats = dict(fused_rounds=[], reduced=None)
    g = fuse(deepcopy(g))
    w0 = fold_scalar(g)
    terms = [(g, w0)]
    if trace: trace("fused", terms)
    for rnd in range(1, max_cut_rounds + 1):
        t0 = time.time()
        new_terms, kept = [], []
        before = [reduced_tcount(h) for h, _ in terms]
        for (h, w), tb in zip(terms, before):
            if tb <= bss_tcount:
                kept.append((h, w)); continue
            b = best_cut_fused(h)
            if b is None or b[0][1] >= tb or b[0][0] >= 2 * tb - 2:   # no branch loses T-count, or no better than cutting one T spider
                kept.append((h, w)); continue
            (score, v, tl, tr) = b
            gl, gr = cut_spider(h, v)
            for br in (gl, gr):
                fuse(br)
                if br.scalar.is_zero:
                    continue
                wb = w * fold_scalar(br)
                if wb != 0:
                    new_terms.append((br, wb))
        n_branches = len(new_terms)
        terms = kept + collect(new_terms)
        if trace: trace(f"cut_{rnd}", terms)
        after = sorted(reduced_tcount(h) for h, _ in terms)
        info = dict(round=rnd, terms_in=len(before), reduced_tcounts_in=sorted(before), branches=n_branches,
                    terms_out=len(terms), reduced_tcounts_out=after, seconds=round(time.time() - t0, 1))
        stats['fused_rounds'].append(info)
        log(f"  fused round {rnd}: {len(before)} terms, reduced T-counts {sorted(before)} -> {n_branches} new branches -> "
            f"{len(terms)} terms, reduced T-counts {after}  ({info['seconds']}s)")
        if n_branches == 0:
            break
    # finish: full_reduce, collect, then the reduced-diagram loop for whatever T-count remains
    amp = 0j
    reduced = []
    for h, w in terms:
        r = reduce_term(h, w)
        if r is not None:
            reduced.append(r)
    reduced = collect(reduced)
    if trace: trace("reduced_terms", reduced)
    log(f"  after full_reduce and collection: {len(reduced)} terms, T-counts {sorted(tcount_g(h) for h, _ in reduced)}")
    stats['reduced'] = dict(terms=len(reduced), tcounts=sorted(tcount_g(h) for h, _ in reduced))
    total = dict(clifford_terms=0, bss_leaves=0, ladder_per_term=[])
    for h, w in reduced:
        if tcount_g(h) == 0:
            amp += w * clifford_value(h); total['clifford_terms'] += 1; total['ladder_per_term'].append(1); continue
        leaves = zx.simulate.find_stabilizer_decomp(h)
        n = 0
        for leaf in leaves:
            if leaf.scalar.is_zero:
                continue
            amp += w * fold_scalar(leaf) * clifford_value(leaf); n += 1
        total['clifford_terms'] += n; total['bss_leaves'] += n; total['ladder_per_term'].append(n)
    log(f"  ladder finish: {total['ladder_per_term']} Clifford terms per reduced term, {total['clifford_terms']} in total")
    stats.update(total)
    return amp, stats


def cat_ladder(g):
    """Cross-check: pyzx's own decomposition ladder on the whole reduced diagram, no cutting."""
    g = deepcopy(g)
    t = reduce_term(g, 1.0 + 0j)
    if t is None:
        return 0j, 0
    g, w = t
    leaves = zx.simulate.find_stabilizer_decomp(g)
    amp = 0j
    n = 0
    for leaf in leaves:
        if leaf.scalar.is_zero:
            continue
        wl = w * fold_scalar(leaf)
        amp += wl * clifford_value(leaf)
        n += 1
    return amp, n


