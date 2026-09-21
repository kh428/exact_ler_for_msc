"""Portable JSON for fixed-phase ZX graphs, including their complete scalar.

Vertex phases are rational multiples of pi. Scalar fields retain exact dyadic
integers where the backend supplies them; external collected-term weights are
complex floating-point numbers. This format contains no executable objects.
"""
from fractions import Fraction
import json
from pathlib import Path

import pyzx_param as zx
from pyzx_param.graph.scalar import DyadicNumber, Scalar, SpiderPair


def encode(value):
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, set):
        return {"set": [encode(v) for v in sorted(value)]}
    if isinstance(value, (DyadicNumber, SpiderPair)):
        return {type(value).__name__: encode(vars(value))}
    if isinstance(value, dict):
        return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported scalar field: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ValueError("Expected one tagged scalar field")
    kind, payload = next(iter(value.items()))
    if kind == "fraction":
        return Fraction(*payload)
    if kind == "complex":
        return complex(*payload)
    if kind == "set":
        return set(map(decode, payload))
    if kind == "mapping":
        return {decode(k): decode(v) for k, v in payload}
    if kind in ("DyadicNumber", "SpiderPair"):
        cls = DyadicNumber if kind == "DyadicNumber" else SpiderPair
        obj = object.__new__(cls)
        obj.__dict__.update(decode(payload))
        return obj
    raise ValueError(f"Unknown scalar field: {kind}")


def graph_data(graph):
    vertices = sorted(graph.vertices())
    return {
        "vertices": [{"id": int(v), "type": int(graph.type(v)),
                      "phase_pi": str(graph.phase(v)),
                      "qubit": graph.qubit(v), "row": graph.row(v)} for v in vertices],
        "edges": [[int(a), int(b), int(graph.edge_type(e))]
                  for e in graph.edges() for a, b in [graph.edge_st(e)]],
        "inputs": list(graph.inputs()), "outputs": list(graph.outputs()),
        "scalar": encode(vars(graph.scalar)),
    }


def restore_graph(data):
    graph = zx.Graph()
    remap = {}
    for vertex in data["vertices"]:
        remap[vertex["id"]] = graph.add_vertex(
            ty=vertex["type"], phase=Fraction(vertex["phase_pi"]),
            qubit=vertex["qubit"], row=vertex["row"])
    for a, b, edge_type in data["edges"]:
        graph.add_edge((remap[a], remap[b]), edgetype=edge_type)
    graph.set_inputs(tuple(remap[v] for v in data["inputs"]))
    graph.set_outputs(tuple(remap[v] for v in data["outputs"]))
    graph.scalar = Scalar()
    fields = decode(data["scalar"])
    if set(fields) != set(vars(graph.scalar)):
        raise ValueError("Scalar fields differ from the supported pyzx_param version")
    graph.scalar.__dict__.update(fields)
    return graph


def write_terms(path, terms, *, case, record, outcomes, stage):
    """Write a scalar amplitude as sum(weight * graph), retaining every factor."""
    payload = {"schema": "msc-zx-terms-1", "case": case, "record": record,
               "outcomes": list(outcomes), "stage": stage,
               "terms": [{"weight": encode(complex(weight)), "graph": graph_data(g)}
                         for g, weight in terms]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def read_terms(path):
    payload = json.loads(Path(path).read_text())
    if payload["schema"] != "msc-zx-terms-1":
        raise ValueError("Unrecognised graph schema")
    return [(restore_graph(t["graph"]), decode(t["weight"])) for t in payload["terms"]]
