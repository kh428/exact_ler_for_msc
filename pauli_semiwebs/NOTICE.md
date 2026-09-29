# Sources and attribution

This is an addition to the companion code for *Exact logical error rates for
magic state cultivation*, [arXiv:2609.18922](https://arxiv.org/abs/2609.18922).
The original code in this folder uses the repository's Apache 2.0 licence.

Pauli semiweb definitions and the diagram notation follow Aleks Kissinger
and John van de Wetering, *ZX-Flow: A Flexible Criterion for Deterministic
Computation with ZX-Diagrams*, [arXiv:2603.09580](https://arxiv.org/abs/2603.09580).
The TikZ styles are adapted from that paper and retain their source comments.

The circuit construction is from Craig Gidney, Noah Shutty and Cody Jones,
*Magic state cultivation: growing T states as cheap as CNOT gates*,
[arXiv:2409.17595](https://arxiv.org/abs/2409.17595).
The d=3 input is the generated unitary-injection member of this family with
the documented conversion to actual T gates. The d=5 input is the corrected
native-T SOFT circuit distributed with the main exact-LER calculation.
The included circuit bytes and checksums are fixed; no online lookup is
needed to reproduce the semiweb calculation.

The introductory teleportation drawings use the example attributed in the
paper to [TQEC](https://github.com/tqec/tqec), with the explicit S-dagger
correction shown on the right-hand diagram. Copied circuit and diagram
material retains its upstream attribution; the licence for original code
does not replace any applicable upstream terms.
