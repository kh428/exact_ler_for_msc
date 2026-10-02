# Sources and attribution

This is an addition to the companion code for *Exact logical error rates for
magic state cultivation*, [arXiv:2609.18922](https://arxiv.org/abs/2609.18922).
The results in this folder are not in that paper. The original code in this
folder uses the repository's Apache 2.0 licence.

Pauli web and semiweb definitions follow Aleks Kissinger and John van de
Wetering, *ZX-Flow: A Flexible Criterion for Deterministic Computation with
ZX-Diagrams*, [arXiv:2603.09580](https://arxiv.org/abs/2603.09580). The drawings
use the TikZ styles of `../pauli_semiwebs/drawings`, adapted from that paper.
They are read in place and not copied here.

The 15-to-1 protocol is from Sergey Bravyi and Alexei Kitaev, *Universal
quantum computation with ideal Clifford gates and noisy ancillas*,
[Phys. Rev. A 71, 022316 (2005)](https://doi.org/10.1103/PhysRevA.71.022316).
The factory analysed here is the 15-to-1 distillation circuit of Daniel
Litinski, *Magic State Distillation: Not as Costly as You Think*,
[Quantum 3, 205 (2019)](https://doi.org/10.22331/q-2019-12-02-205): five qubits
in |+>, fifteen Z-type pi/8 rotations and four X measurements. A unit test
compares the fifteen rotations of the diagram with those of Fig. 3 of
[arXiv:1905.06903v3](https://arxiv.org/abs/1905.06903v3).

The ZX diagrams in `diagrams/` are original drawings of that circuit, made
with [TikZiT](https://tikzit.github.io). They are read by path and checksum,
and the drawings and results in this folder are derived from them.
