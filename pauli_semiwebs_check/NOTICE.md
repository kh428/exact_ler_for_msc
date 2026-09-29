# Sources and attribution

This is an addition to the companion code for *Exact logical error rates for
magic state cultivation*, [arXiv:2609.18922](https://arxiv.org/abs/2609.18922),
Appendix J. The original code in this folder uses the repository's Apache 2.0
licence.

Pauli web and semiweb definitions follow Aleks Kissinger and John van de
Wetering, *ZX-Flow: A Flexible Criterion for Deterministic Computation with
ZX-Diagrams*, [arXiv:2603.09580](https://arxiv.org/abs/2603.09580). The drawings
use the TikZ styles of `../pauli_semiwebs/drawings`, adapted from that paper.
They are read in place and not copied here.

The circuits are the paper's inputs in `../data/inputs`: the d=3 and d=5 magic
state cultivation circuits distributed with SOFT (Riling Li et al., *SOFT: a
high-performance simulator for universal fault-tolerant quantum circuits*,
[arXiv:2512.23037](https://arxiv.org/abs/2512.23037), Apache 2.0). They follow
the construction of Craig Gidney, Noah Shutty and Cody Jones, *Magic state
cultivation: growing T states as cheap as CNOT gates*,
[arXiv:2409.17595](https://arxiv.org/abs/2409.17595). They are read by path and
checksum and not copied here. The drawings and results in this folder are
derived from them and retain their attribution.
