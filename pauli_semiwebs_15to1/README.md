# Pauli webs and semiwebs of a 15-to-1 factory

This folder applies the semiweb code of [`../pauli_semiwebs`](../pauli_semiwebs) to a
15-to-1 magic state distillation factory. It is an addition to the companion code of
[Exact logical error rates for magic state cultivation](https://arxiv.org/abs/2609.18922).
The results here are not in that paper.

The factory is the 15-to-1 circuit of Litinski,
[Quantum 3, 205 (2019)](https://doi.org/10.22331/q-2019-12-02-205), for the protocol of
Bravyi and Kitaev,
[Phys. Rev. A 71, 022316 (2005)](https://doi.org/10.1103/PhysRevA.71.022316). Five qubits
start in |+>, fifteen T states act on them and four qubits are measured in X. The fifth
qubit is the output. The S-gate proxy is the same diagram with every T state replaced by
an S state, so that every pi/4 becomes pi/2.

## Result

| | S-gate proxy | T states |
|---|---|---|
| Closed Pauli webs | 4, one for each X measurement, none without a record | 0 |
| Web of each X measurement | no defect | 8 defects of +pi/2, all at T states |
| Web of each of the 15 parities of the four records | no defect | 8 defects of +pi/2, all at T states |
| Web of the output, with Y on the output leg | no defect | 7 defects of +pi/2, all at T states |
| Output, fully contracted | X spider of phase +pi/2: \|0> - i\|1> | Z spider of phase -pi/4: \|0> + e^{-i pi/4}\|1> |
| Operator with the output as its +1 eigenstate | -Y, a Pauli | (X - Y)/sqrt2, not a Pauli |

In both diagrams every X measurement gives 0 with certainty. In the proxy each of them is
a closed Pauli web. With T states the same labels are a closed semiweb, and its defects
sit on exactly the eight T states that act on the measured wire. The diagram with T states
has no closed Pauli web at all, and none of its checks holds for every phase of the T
states, as shown below.

## Why the checks still hold

For a parity c of the four records, the semiweb gives
`D_r = lambda_0 (-1)^(c.r) D_{r,delta}`. Here `D_r` is the diagram with record pattern r
and `D_{r,delta}` the same diagram with the eight marked phases moved by +pi/2. For all 15
parities `lambda_0 = 1`, and the shifted diagram is the original one,
`D_{r,delta} = D_r`. So `D_r` vanishes unless c.r = 0, and the parity is 0 with certainty.

The eight shifts cancel because of the factory's 5 x 15 matrix, whose entry is 1 where a
T state acts on a wire. A shift of +pi/2 on a T state multiplies a basis state by i when
the parity of its bits on that T state's wires is odd. Writing each parity as a polynomial
in the bits, the product of the eight shifts is 1 when the row of the measured wire has
weight 0 mod 4, overlaps every other row in 0 mod 4 columns, and every triple of rows
overlaps in an even number of columns. The rows of the four measured wires have weight 8,
all pairs of rows overlap in 4 columns and all triples in 2.

The output row has weight 7, so its seven shifts do not cancel. The next section works
out what they do.

## The output

Contracted fully, each diagram is one spider on the output wire. The proxy gives the X
spider of phase +pi/2, which is |0> - i|1> up to a scalar. The same state is a Z spider of
phase +pi/2 followed by a Hadamard and, up to a scalar, a Z spider of phase -pi/2. With T
states the output is the Z spider of phase -pi/4, |0> + e^{-i pi/4}|1>, proportional to
T_dag|+>.

The output has an open web with Y on the output leg and no record. An open web with a
Pauli P on the output and defects delta gives `D = lambda P D_delta`, with P acting on the
output.

- Proxy: no defect and `lambda = -1`, so `D = -Y D`. The output is the -1 eigenstate of Y.
- T states: the same labels leave seven defects of +pi/2, on the seven T states that act
  on the output wire, and `lambda = e^{5 i pi/4}`. By the count of the last section the
  seven shifts multiply a basis state by i^(7 x_0), where x_0 is the bit of the output
  wire. That is S_dag on the output, so `D_delta = S_dag D` and `D = lambda Y S_dag D`:
  the output is an eigenstate of Y S_dag with eigenvalue e^{3 i pi/4}. Since
  `Y S_dag = e^{3 i pi/4} (X - Y)/sqrt2`, the phases cancel and the output is the +1
  eigenstate of (X - Y)/sqrt2 = T_dag X T. That state is T_dag|+>, the spider found by
  contraction. It is not an eigenstate of X, Y or Z: its expectation values are 1/sqrt2,
  -1/sqrt2 and 0.

For the checks only the closed webs matter. The open web names the observable of the
output: the Pauli Y in the proxy, and (X - Y)/sqrt2, which is not a Pauli, with T states.

## The checks depend on the phases

One T state at a time is moved through the three other Clifford angles. With columns
written as record patterns r_1 r_2 r_3 r_4:

| Phase of one T state | Records that occur | Effect |
|---|---|---|
| +pi/2 or +3pi/2, that is T S or T_dag | 0000 and the T state's column | the checks on its wires become random |
| +pi, a Z error on the T state | the T state's column | its checks flip and stay determined |

No parity of the four records is fixed for every Clifford angle. This agrees with the
empty web space: a closed Pauli web gives a check that holds at every phase of the T
spiders, Lemma 1 of [`../pauli_semiwebs_check`](../pauli_semiwebs_check).

## Method

`factory/diagram.py` reads the ZX diagram from the TikZiT drawings in `diagrams/`: 67
nodes and 86 edges, with four T states on the wires and eleven on phase gadgets. The
wires are numbered 0 to 4 from the top. Wire 0 is the output, and r_i is the X measurement
of wire i, drawn as the effect of outcome 0. T states are numbered from 1: first those on
wires 1 to 4, then the gadgets from the left. `results/factory.json` lists the wires of
each.

`factory/webs.py` finds the closed webs as the kernel of the binary constraints of
`../pauli_semiwebs/src/semiweb.py`, with no label on the output leg. The defects and the
scalar `lambda_0` come from that package's local rules.

`factory/tensor.py` evaluates the tensor of the diagram in exact arithmetic. Every phase
is a multiple of pi/4, so each entry is an integer combination of 1, w, w^2 and w^3 with
w = exp(i pi/4). All identities above are equalities of integers.

These cross-checks pass:

- the fifteen rotations of the diagram are those of the 15-to-1 circuit in Fig. 3 of
  [arXiv:1905.06903v3](https://arxiv.org/abs/1905.06903v3),
- the tensor of the diagram equals a direct five-qubit calculation,
  `factory/statevector.py`, for all sixteen record patterns, with T states, with the proxy
  and with 53 other choices of the fifteen phases,
- a third calculation with 32 x 32 matrices in floating point agrees with both,
- every one of the 15 parity webs satisfies the semiweb identity for all sixteen record
  patterns,
- the seven shifts of the output web act as S_dag on the output wire on all 32 basis
  states, and the eight shifts of each check act as the identity,
- the statements about the output are repeated with ordinary complex matrices,
- the copy, fuse and unfuse rules and the examples of the local rule on the opening
  slides agree with the package's rule and with explicit matrices,
- each check web has X on its own measurement and ends as Y on exactly its eight T states,
- the four proxy webs drawn by hand in `diagrams/` are the computed webs,
- each drawing in `webs/`, parsed on its own, gives back the stated records and defects.

## Drawings

`webs/` has twelve TikZ drawings in the style of `../pauli_semiwebs/drawings`:
`proxy_check1.tikz` to `proxy_check4.tikz`, the same four webs on the diagram with T
states as `t_check1.tikz` to `t_check4.tikz`, the output web as `proxy_output.tikz` and
`t_output.tikz`, and the two diagrams with no web as `proxy_diagram.tikz` and
`t_diagram.tikz`. Red dashed edges carry X, green edges Z and blue dashed edges Y. A
violet ring and a numbered star mark each defect, and a comment records its T state,
original phase and phase change. The record a web reads is printed in bold.

`slides/slides.pdf` has eleven slides. Four introduce the semiweb: its definition with
labels drawn as pi spiders, three examples at one spider drawn as web labels, as pi
spiders and after the spider has absorbed them, how a Pauli is pushed through a spider by
the copy and fuse rules of the ZX calculus, which is Pauli propagation, and where the
defects come from.
Seven are about the factory: the four checks as Pauli webs of the proxy, as semiwebs with
T states, why each check still holds, what the factory outputs, the web of the output, and
the two steps that lead from its seven defects to (X - Y)/sqrt2. The small pictures of the
first four slides are in `slides/illustrations.tex`.

## Reproduce

Use Python 3.11 with `numpy`. The slides also need pdflatex with beamer, TikZ, mathtools
and adjustbox. Run from this folder:

```sh
python -B -m unittest factory.test_factory -v
python -B -m factory.run
python -B -m factory.slides
python -B verify.py
```

`factory.run` recomputes the results with all their checks and writes nothing. Add
`--output` to rewrite `results/factory.json` and `webs/`. `verify.py` checks the manifest
and the input checksums, runs the tests, and compares a fresh calculation with the saved
results and drawings byte for byte.

## Limitations

- The diagram is the ideal factory: no noise, exact T states and every X measurement at
  outcome 0. Nothing here is a statement about the output error rate of a noisy factory.
- The webs and the positions of their defects are properties of this ZX diagram of the
  circuit, not of the circuit alone.
- The phase dependence is tested for one T state at a time at Clifford angles, together
  with the empty space of closed Pauli webs.

## Files

| Location | Contents |
|---|---|
| `factory/diagram.py` | Reader for the TikZiT drawings, node roles and the wires of each T state |
| `factory/webs.py` | Closed webs, the records they read and their scalars |
| `factory/tensor.py` | Exact tensor of the diagram and its arithmetic |
| `factory/statevector.py` | The factory as five qubits, without the diagram |
| `factory/run.py` | The analysis with all its checks |
| `factory/draw.py` | TikZ drawing of a web on the diagram |
| `factory/slides.py` | Builds `slides/slides.pdf` |
| `factory/test_factory.py` | Unit tests, including the cross-checks listed above |
| `diagrams/` | Input drawings: the factory with T states, and four proxy webs drawn by hand |
| `results/factory.json` | Matrix, webs, defects, scalars, the outputs and the single T state sweep |
| `webs/` | The twelve generated drawings |
| `slides/` | The eleven slides and their sources |
| `provenance.json` | Input drawing checksums and public sources |
| `manifest.json` | SHA-256 checksums of the files in this folder |

See [NOTICE.md](NOTICE.md) for sources and attribution.
