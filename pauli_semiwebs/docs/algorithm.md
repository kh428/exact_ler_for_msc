# Computing a Pauli semiweb

Assign two binary variables to each wire:

\[
I=(0,0),\qquad X=(1,0),\qquad Z=(0,1),\qquad Y=(1,1).
\]

At a Z spider, all incident X bits must agree. At an X spider, all incident
Z bits must agree. Across a Hadamard, the X and Z bits are exchanged. These
are linear constraints over F2. Row reduction gives their kernel and hence
a complete semiweb basis, modulo overall Pauli phase.

For a Z spider of phase theta, let a be the common X bit and b the parity
of its incident Z bits. For an X spider, exchange X and Z in this definition.
The changed phase is

\[
\theta'=(-1)^a\theta+\pi b,\qquad
\delta=\theta'-\theta\pmod{2\pi}.
\]

The code stores phases in units of pi/4. A nonzero delta is a defect. A
semiweb with no defects is a Pauli web. Products multiply the operators
on each wire, retaining their scalar phases. The phase maps must be
composed: applying X on both legs of a pi/4 Z spider twice restores its
original phase, although each operation on the original spider would
have a defect of -pi/2.

Scalar phases are separate from the binary kernel. The generator calculates
them analytically, including signs from transposing Y on contracted edges.
The verifier constructs each local tensor explicitly in Z[exp(i*pi/4)] and
checks both the phase change and the scalar. Common normalisation factors
cancel between the two sides of these identities.

## The two circuit graphs

The parser extracts the final double-checking stage from each frozen input.
It preserves gate signs, the CNOT order, selected X measurements and fresh
preparations. It maps CNOTs to connected Z/X spiders and T or T-dagger gates
to phases +pi/4 or -pi/4. The d=5 source has a mixed sign pattern.

The basis first includes the incoming X_q and Z_q semiwebs, then extends it
with output-only elements and elements with no boundary support. Independent
rank checks establish that this spans the entire kernel.

| Graph | Semiweb dimension | No-boundary dimension | Pauli-web dimension |
|---|---:|---:|---:|
| d=3 | 90 | 62 | 12 |
| d=5 | 264 | 188 | 37 |

For each element the boundary identity is

\[
K P_{\mathrm{in}}=\lambda P_{\mathrm{out}}K_{\boldsymbol\delta}.
\]

K denotes the noiseless selected double-checking stage, before the terminal
code projection. A defect can change a selected measurement or preparation.
The identity alone does not determine acceptance of a physical fault.

The independent d=3 check evaluates both sides on all computational input
basis states. At d=5 the local tensor identities, wiring and scalar accounting
establish the corresponding diagram identity without constructing a dense
whole-stage matrix.

The definitions follow Kissinger and van de Wetering, *ZX-Flow*,
[arXiv:2603.09580](https://arxiv.org/abs/2603.09580).
