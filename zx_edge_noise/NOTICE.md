# Sources and attribution

This is an addition to the companion code for *Exact logical error rates for magic state cultivation*,
[arXiv:2609.18922](https://arxiv.org/abs/2609.18922). The results in this folder are not in that paper. The original
code in this folder uses the repository's Apache 2.0 licence.

The circuits are the SOFT d=3 and d=5 files of [`../data/inputs`](../data/inputs); their sources are listed in
[`../docs/SOURCES.md`](../docs/SOURCES.md). The repository's own code and data are used in place and not copied:
the Pauli-label tensors (`../alternatives/calculation/chan.py`, `pauli_polynomial.py`), the exact polynomial ring and
native kernels (`../calculation`), the d=3 and d=5 stage definitions (`../data/inputs/*_source_noise_20260905.json`),
the growth model (`../data/models/growth_frame_model.json`) and the growth tree with its leaf maps (`../data/plan`).
`zxedge/elimination.py` extends the repository's `eliminate_linear_constraints`.

`zxedge/staged/stabilizer/pauli.py` and `support.py` are the stabilizer-support helpers of the compiler that wrote
`growth_frame_model.json`, copied unchanged. `zxedge/staged/growth.py` follows that compiler, with each channel given
by explicit generator Paulis.

Faults on the edges of a ZX diagram follow Hector Bombin, Daniel Litinski, Naomi Nickerson, Fernando Pastawski and
Sam Roberts, *Unifying flavors of fault tolerance with the ZX calculus*,
[arXiv:2303.08829](https://arxiv.org/abs/2303.08829).

Contraction plans are found with [cotengra](https://github.com/jcmgray/cotengra): Johnnie Gray and Stefanos Kourtis,
*Hyper-optimized tensor network contraction*, Quantum 5, 410 (2021),
[arXiv:2002.01935](https://arxiv.org/abs/2002.01935),
with the [KaHyPar](https://kahypar.org) hypergraph partitioner. The exact low-rank splits of `zxedge/splitter.py`
follow the split simplification of [quimb](https://github.com/jcmgray/quimb) (Johnnie Gray, *quimb: a python library
for quantum information and many-body calculations*, [J. Open Source Softw. 3, 819
(2018)](https://doi.org/10.21105/joss.00819)), done exactly over a prime field. Elimination over a prime field uses
[numba](https://numba.pydata.org).
