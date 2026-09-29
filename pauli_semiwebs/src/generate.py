"""Compute the basis, phase defects and boundary identities from the circuits."""

from .double_check_stage import read_stage, graph_for, full_basis
from .semiweb import BITS, defects, edge_paulis, incidence


def local_scalar(kind, phase, labels):
    """Return the scalar exponent in units of pi/4 for a valid semiweb.

    Z spiders have all X bits equal; X spiders have all Z bits equal.
    Conjugating Y by H contributes a minus sign for an X spider.
    """
    bits = [BITS[p] for p in labels]
    if not bits or kind not in ('X', 'Z'):
        raise ValueError('Expected a nonempty X or Z spider')
    opposite = 0 if kind == 'Z' else 1
    flip = bits[0][opposite]
    if any(b[opposite] != flip for b in bits):
        raise ValueError('The labels violate the semiweb constraint')
    parity = sum(b[1-opposite] for b in bits) % 2
    ys = sum(x*z for x, z in bits)
    return (2*ys + (4*ys if kind == 'X' else 0)
            + flip*(phase + 4*parity)) % 8


def compute():
    stages = []
    for distance in (3, 5):
        stage = read_stage(distance)
        graph = graph_for(stage)
        basis, counts = full_basis(stage, graph)
        incident = incidence(graph)
        records = []
        for number, item in enumerate(basis, 1):
            vector = item['vector']
            labels = edge_paulis(graph, vector)
            changes = defects(graph, vector)
            exponent = sum(
                local_scalar(n['kind'], n['phase'],
                             tuple(labels[j] for j in incident[k]))
                for k, n in graph['nodes'].items() if n['kind'] != 'boundary'
            )
            # A contracted Y transposes to -Y. Include input cups as well.
            for edge, label in zip(graph['edges'], labels):
                roles = {graph['nodes'][edge[k]]['role'] for k in ('u', 'v')}
                if label == 'Y' and 'output' not in roles:
                    exponent += 4

            def boundary(role):
                return {n['qubit']: labels[incident[k][0]]
                        for k, n in graph['nodes'].items()
                        if n['role'] == role and labels[incident[k][0]] != 'I'}

            records.append(dict(kind=item['kind'], label=item['label'],
                                id=f'd{distance}_{number:03d}', number=number,
                                vector_hex=hex(vector), edge_labels=labels,
                                defects=changes, input=boundary('input'),
                                output=boundary('output'),
                                scalar_pi_over_4=exponent % 8,
                                is_pauli_web=not changes))
        stages.append(dict(stage=stage, graph=graph, counts=counts,
                           generators=records))
    return {'stages': stages}
