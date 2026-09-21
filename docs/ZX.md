# Fixed-fault ZX graphs

The graphs represent fixed measurement-record amplitudes for ideal encoded input,
the double-checking sub-circuit with specified Pauli faults, and noiseless final
code/logical readout. They are scalar graphs, since all preparations and measured
outcomes are inserted. These witnesses are distinct from the complete noisy
probability calculation in `calculation/`.

| Example | Faults inserted after source lines | Accepted-correct | Accepted-error |
|---|---|---:|---:|
| d=3 | `X 3` after 142; `Y 9` after 173 | 0 | 1/4 |
| d=5 | `X 3`, `X 14` after 397; `Y 23` after 438 | 0 | 1/16 |

Line numbers refer to the frozen circuits in `data/inputs/`. The builder in
`zx/circuits.py` specifies the data sites, faces and double-checking windows.
It strips stochastic noise and inserts only the named faults. The fault-free
all-zero measurement record fixes the logical reference convention.

## Graph files

`data/zx/graphs/d3/` and `d5/` contain three records: `reference`, `good`, `bad`.
For each record, the snapshots are:

- `*_raw.json`: the amplitude graph built from the circuit.
- `*_full_reduce.json`: the same scalar after direct full reduction.
- `*_fused.json`: spider fusion and identity removal, before cutting.
- `*_cut_N.json`: all surviving weighted terms after cutting round N.
- `*_reduced_terms.json`: the collected reduced terms before final decomposition.

Each file represents `sum_t weight_t * graph_t`. The graph scalar is included
separately from the external term weight and must be multiplied by it. A list
of no terms represents zero. Vertex types, edges, rational phases in units of
pi, drawing coordinates, boundaries and every scalar field are retained.

Use the supplied reader:

```python
from zx.graph_io import read_terms
from zx.amplitude import cat_ladder

terms = read_terms("data/zx/graphs/d5/bad_reduced_terms.json")
amplitude = sum(weight * cat_ladder(graph)[0] for graph, weight in terms)
probability = abs(amplitude)**2
print(probability)  # approximately 0.0625
```

For plotting, a loaded graph can be passed to `pyzx_param.draw(graph)` or
`graph.to_tikz()`. A topology-only drawing does not display its full scalar;
the JSON is the numerical record.

## Arithmetic and packages

Graph phases use exact rational multiples of pi. The backend keeps dyadic
scalar fields where available; amplitude summation and probabilities use complex
floating point. The check tolerance is `1e-12`, separate from the prime-field
certification of the complete noisy LER.

The amplitude routes use `pyzx-param` distribution `0.9.3`, whose internal
version string is `0.9.0`, and Stim `1.16.0`. The cutting primitive is adapted
from the companion code of arXiv:2509.08658. The additional `d=3` probability
diagram records use `bloqade-tsim 0.1.5` with cat5, BSS and cutting strategies.
The completed `d=5` results use the amplitude routes.
