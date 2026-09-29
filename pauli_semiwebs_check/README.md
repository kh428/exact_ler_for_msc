# Pauli webs of the whole cultivation circuits

This folder extends [`../pauli_semiwebs`](../pauli_semiwebs) from the double-checking
stages to the whole noiseless d=3 and d=5 circuits of
[Exact logical error rates for magic state cultivation](https://arxiv.org/abs/2609.18922).
It finds which declared detectors are Pauli webs of the circuits with actual T gates,
and draws a web for every declared detector. This is Appendix J of the paper and its
accompanying `detector_webs.pdf`.

## Result

| Circuit | Records | Declared detectors | Pauli webs | Semiwebs with T defects |
|---|---:|---:|---:|---|
| d=3, `../data/inputs/clifft_d3_p001.stim` | 21 | 20 | 16 | 4: detectors 7, 14, 16, 18 |
| d=5, `../data/inputs/soft_cultivation_d5_p0005.stim` | 112 | 107 | 93 | 14: detectors 7, 19, 20, 23, 69, 89, 91, 93, 95, 97, 99, 101, 103, 105 |

Detectors are numbered from 0 in source order. Every declared detector is deterministic
in the noiseless circuit with its T gates, since the exact series in `../data/series`
start with a_0 = 1 at both distances. A Pauli-web detector holds whatever the phase of
each T gate. Each of the others is a Pauli web of the S proxy, the circuit with every T
replaced by S, but in the T circuit only a closed semiweb with defects on T spiders. It
depends on the phases of individual T gates, and each one becomes random under some
Clifford choice of those phases.

The Pauli webs give exactly the phase-independent checks. A Stim oracle over a few
hundred Clifford choices of the T phases finds the same 16 and 93 independent checks.
Sampling can only overcount, and the webs give the lower bound.

## Why the webs stop there

**Lemma 1, phase independence.** In `../pauli_semiwebs/src/semiweb.py` a defect-free
web at an odd-phase spider has no opposite-colour support and even same-colour parity.
With the defect rule delta = -2 o theta + pi s, a T spider with o = 1 always has
delta = +-pi/2, so o = 0 is forced whatever theta is. The local identity there is then
trivial, so the web, its scalar and its record check do not depend on the phase of any
T spider.

**Lemma 2, the oracle is exact in the limit.** In the probability of a record sequence,
each T gate contributes a factor 1 or exp(i theta) to the ket and its conjugate to the
bra. Every record-parity probability is therefore a trigonometric polynomial with
frequencies -1, 0 and 1 in each theta. A parity that is constant, with one value, at the
four Clifford angles of every T gate is constant at every angle.
`check_basis/oracle.py` intersects Stim's exact deterministic parities over Clifford
substitutions.

**Corollary, Pauli determinism.** A Pauli fault on a web edge multiplies the web's
scalar by +-1. A web check therefore responds to every Pauli fault deterministically,
and identically in the T circuit and the S proxy. A fault pattern that the T circuit
accepts flips no web check, so the proxy can reject it only through the other
detectors. An X or Y fault next to a T gate acts as T -> T^dagger, since
X T = exp(i pi/4) T^dagger X.

## Circuits

The inputs are the paper's SOFT circuits in `../data/inputs`, read by path and checked
against the checksums in `provenance.json`. Both end with a noiseless readout. The code
readout measures the stabilizers as Pauli products. The logical readout measures MPP Y_L
after a noiseless layer of T gates (T and T^dagger at d=5). Because
T^dagger Y T = T X T^dagger = (X+Y)/sqrt2, it measures the same transversal operator as
the last double check, a noiseless projection onto the magic state.

`../pauli_semiwebs/circuits/d3_native_t.txt` has the same operations as the d=3 input up
to its readout, which is a bare MPP Y_L. That readout is random in the noiseless T circuit
and does not affect the stage-only basis computed there. The d=5 input is byte-identical
to `../pauli_semiwebs/circuits/d5_native_t.stim`.

## Method

`check_basis/graph.py` builds the ZX graph of the whole noiseless circuit with the
conventions of `../pauli_semiwebs`. A CNOT is a joined Z/X spider pair and a T gate a
pi/4 Z spider. Measurements, Pauli-product measurements and record-controlled Paulis
become spiders of phase r pi tagged with their record. Six qubits are reset
mid-injection without being measured. `check_basis/oracle.py` shows that each is in |+>
for every phase at the preceding T gate, so each reset is a post-selected effect onto
|+>. The six record-free webs certify these known states.

`check_basis/webs.py` solves for the closed, defect-free webs and turns each into a
record check, with its sign from the package's local scalars. These cross-checks pass:

- the parser's record and detector indexing equals Stim's,
- the web span equals the Stim oracle's space, in both directions,
- the package's stage-level webs (6 at d=3, 19 at d=5) lie in the whole-circuit span,
- with every T replaced by S, the webs recover every deterministic parity of the proxy,
- toy circuits with feedforward, re-measurement, Y products and a known-state discard
  give exactly Stim's parities (`check_basis/test_check_basis.py`).

`check_basis/verify_claims.py` makes 13 further checks of the statements in the guide
and in Appendix J, including the readout algebra above and a d=3 statevector run.

## Fault response

`check_basis/faults.py` declares the web checks as Stim detectors and compares Stim's
per-location error explanations over Clifford substitutions of the T gates. It then
lists S-proxy fault pairs that flip the logical observable and no web check.

| Circuit | Fault events that flip a web check | Substitutions with an identical fault map | Proxy pairs flipping the logical but no web check | Of these, invisible to every detector |
|---|---:|---:|---:|---:|
| d=3 | 1,848 | 22 of 22 | 4,515 | 0 |
| d=5 | 10,465 | 22 of 22 | 33,797 | 0 |

In the proxy each such pair is caught only by detectors that are not Pauli webs of the T
circuit. A pair is two Paulis at different physical positions.

## Drawings

`detector_webs/detector_webs.pdf` (132 A3 pages) draws one web for every declared
detector on the whole-circuit ZX diagram. The 20 d=3 pages show the whole circuit, and
the 107 d=5 pages show the columns each web uses. The TikZ sources are in
`detector_webs/diagrams/`, and `detector_webs/index.tsv` lists the page of each detector.
`detector_webs/figures/d3_det014_check.tikz` is the figure of Appendix J, and
`detector_webs/figures/d3_det007_web.tikz` the one in the repository README.

Each drawn web is a closed Pauli web of the S proxy with exactly the detector's records.
It is chosen among the 64 webs of its coset with the fewest T defects, then the fewest
labelled edges. A comment in each TikZ file records every defect: its spider, qubit,
source line, original phase and phase change.

## Reproduce

Use Python 3.11 with `stim==1.16.0` and `numpy`, as pinned in `../requirements-zx.txt`.
The guide also needs pdflatex with TikZ, adjustbox, booktabs, caption, longtable and
hyperref. Run from this folder:

```sh
python -B -m unittest check_basis.test_check_basis -v
python -B -m check_basis.run --output results/check_basis.json
python -B -m check_basis.faults --output results/fault_response.json
python -B -m check_basis.verify_claims
python -B -m check_basis.guide.build
python -B -m check_basis.guide.check_tikz
python -B verify.py
```

The analyses write nothing without `--output`. The guide build rewrites `detector_webs/`
and runs pdflatex three times, which takes a few minutes. `verify.py` checks the manifest
and the input checksums, reruns the tests and both analyses against `results/`, and runs
the claim and TikZ checks. Add `--quick` to stop after the tests.

## Limitations

- The drawn semiweb of a phase-dependent detector has the fewest T defects among the 64
  proxy webs with its records, not necessarily among all semiwebs.
- Weight-three fault patterns at d=5 are not enumerated here.
- The acceptance of each candidate fault pair in the T circuit needs the noisy
  tensor-contraction calculation of this repository, not Stim.

## Files

| Location | Contents |
|---|---|
| `check_basis/circuit.py` | Parser for the Stim-style sources, and T-to-Clifford substitution |
| `check_basis/graph.py` | ZX graph of a whole circuit with every record exposed |
| `check_basis/webs.py` | Closed, defect-free webs and their record checks |
| `check_basis/oracle.py` | Stim oracle for phase-independent parities, and the discarded-state check |
| `check_basis/run.py` | Check basis with all cross-checks |
| `check_basis/faults.py` | Pauli determinism, and the fault pairs the web checks cannot see |
| `check_basis/statevector.py` | Noiseless statevector run, used at d=3 |
| `check_basis/verify_claims.py` | Checks of the statements in the guide and Appendix J |
| `check_basis/test_check_basis.py` | Toy circuits with known answers |
| `check_basis/guide/` | Web choice, layout, TikZ rendering, the A3 guide and its TikZ check |
| `results/` | JSON results, including every check as records with source lines |
| `detector_webs/` | The guide PDF, its TikZ sources, page index and two standalone figures |
| `provenance.json` | Input circuit paths, checksums and public sources |
| `manifest.json` | SHA-256 checksums of the files in this folder |
