# Pauli semiwebs for double-checking circuits

This folder computes and verifies the Pauli semiwebs used in Appendix I of
[Exact logical error rates for magic state cultivation](https://arxiv.org/abs/2609.18922).
It includes the two frozen circuit inputs, the complete semiweb bases, the
editable TikZ drawings and exact checks of the phase identities.

The bases contain 90 elements for d=3 and 264 for d=5. Products generate every
semiweb of each specified graph, up to overall Pauli phase. They include the
semiwebs with no support on the input or output wires. This folder studies
the noiseless double-checking stages; the repository's existing calculation
code performs the physical noise average for the logical error rate.

## Reproduce the results

Use Python 3.10 or newer. No third-party Python packages are required. Run
these commands from this folder:

```sh
python -B compute_basis.py --check
python -B draw_diagrams.py --check
python -B -m unittest discover -s tests -v
python -B verify.py --full
```

The first command recomputes both bases from the circuits and compares every
record with `data/basis.json`. The second regenerates all 354 basis drawings
and compares them byte for byte with the supplied TikZ. The tests compare the
phase rules with explicit exact tensors, including Y signs and cancellation.

`verify.py` checks the file manifest, circuit-to-graph correspondence, basis
ranks, local tensor identities, diagram labels and defect markers. Add
`--full` to evaluate all 90 d=3 basis identities and seven displayed examples
on all 128 input basis states. This takes roughly three minutes on one core;
no dense d=5 state is constructed. The d=5 checks use exact local identities,
scalar factors, connectivity and binary ranks.

To save regenerated data, drawings or a verification report:

```sh
python -B compute_basis.py --output output/basis.json
python -B draw_diagrams.py --data output/basis.json --output output/diagrams
python -B verify.py --full --output output/verification.json
```

## Drawings

`drawings/basis/` contains the 354 basis elements. The seven full-circuit
examples from the main text and appendix are in `drawings/examples/`, alongside
the legend and introductory teleportation drawings. Each defect marker names
its spider. A comment records the qubit, original phase and phase change;
these are mathematical data needed to interpret the drawing.

With a LaTeX installation providing TikZ, adjustbox, caption and hyperref,
build the 355-page guide from this folder:

```sh
mkdir -p output
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -jobname=pauli_semiwebs -output-directory=output drawings/guide.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -jobname=pauli_semiwebs -output-directory=output drawings/guide.tex
```

The PDF is `output/pauli_semiwebs.pdf`. Two passes populate its bookmarks.

## Files

| Location | Contents |
|---|---|
| `src/` | Binary constraints, stage extraction, phase scalars and TikZ rendering |
| `circuits/` | Frozen input circuits; only the double-checking stages are extracted |
| `data/basis.json` | Graphs, basis vectors, defects and boundary identities |
| `drawings/` | Reviewed TikZ sources and LaTeX guide |
| `verification/` | Exact tensor and whole-stage checks, independent of the basis solver |
| `tests/` | Small algebra regressions |
| `docs/algorithm.md` | How the basis and phase data are computed |
| `provenance.json` | Circuit checksums and public sources |
| `manifest.json` | SHA-256 checksums of the distributed files |

The checks cover these fixed diagrams and conventions. They do not assert a
new fault distance or a speed-up over the noisy tensor-contraction engine.
See [NOTICE.md](NOTICE.md) for sources and attribution.
