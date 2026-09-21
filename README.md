# Exact logical error rates for magic state cultivation

Companion code and data for [arXiv:2609.18922](https://arxiv.org/abs/2609.18922).
The calculations give the acceptance probability `A`, the probability `B` of
accepting an output with a logical error, and the conditional logical error
rate `P_L = B/A`.

This package includes the original SOFT results and the additional RP²,
fold-transversal and repaired colour-code results. All commands run locally.
Checking the saved results does not start a simulation.

## Start here

With Python 3.11 or later, run:

```sh
python3 -B verify_results.py
python3 -B verify_other.py
python3 -B evaluate_other.py chan-four-round-d5 --p 1/1000
```

The first command checks the original results. The second checks the seven
additional circuit variants, including every saved sixth-order batch. The third
prints the exact coefficients and evaluates their finite series at the chosen
noise strength. These commands need no third-party Python packages.

## Additional cultivation circuits

All series use `x = p/(1-p)`. The coefficients are exact; evaluating a finite
series does not include the omitted higher orders. No truncation-error bound
is supplied for these additional circuits.

| Case name | Circuit | Available through | Leading power |
|---|---|---:|---:|
| `rp2-d3` | RP², d=3 | x⁵ | x² |
| `rp2-d5` | RP², d=5 | x⁶ | x³ |
| `fold-d3` | Fold-transversal, terminated d=3 core | x⁵ | x³ |
| `fold-d5` | Updated native-T d=5 reconstruction | x⁶ | x⁴ |
| `chan-v1-d3` | Chan et al., arXiv v1 repair, d=3 | x¹⁰ | x³ |
| `chan-v1-d5` | Chan et al., arXiv v1 repair, d=5 | x⁷ | x⁴ |
| `chan-four-round-d5` | Chan et al., supplied four-round repair | x⁶ | x⁵ |

The leading power identifies the fault distance of the specified circuit and
noise model. It is not the distance of the underlying code. These calculations
end with noiseless code/logical readout and exclude escape and decoding.
The Fold d=3 result excludes the noisy pre-escape syndrome round; the updated
Fold d=5 result uses a declared native-T reconstruction. It has not been matched
to an author-supplied noisy T circuit. See the [model descriptions](docs/OTHER_CULTIVATION.md)
before comparing results from different protocols.

To change the noise strength or retained order:

```sh
python3 -B evaluate_other.py rp2-d5 --p 1/2000 --order 6
python3 -B evaluate_other.py fold-d5 --p 0.001 --order 5 --fraction
```

`--fraction` prints the exact value of the **finite polynomial**, not an
all-orders logical error rate. Requesting an unavailable order gives an error.

To regenerate the comparison figures, install `requirements-plots.txt` in your
Python environment, then run:

```sh
python3 -B plot_other.py --output outputs/comparisons
```

This writes four vector PDFs, 500-dpi PNGs, LaTeX equations and a numerical
check file. RP² points are supplied S-proxy samples. The Fold plots compare
normalisation conventions using sampled or floating-point diagnostics. Chan's
sampled points belong to arXiv v1; none are new four-round samples.

## Repeat an additional calculation

These optional commands need `requirements-contraction.txt` and a C++17
compiler. They compile the kernels into the requested output directory.

```sh
# Rebuild and contract the full d=3 gate/noise model, for one prime.
python3 -B recompute_other.py polynomial rp2-d3 --degree 5 --output outputs/rp2-d3
python3 -B recompute_other.py polynomial fold-d3 --degree 5 --output outputs/fold-d3
python3 -B recompute_other.py polynomial chan-v1-d3 --degree 10 --output outputs/chan-d3

# Repeat a complete sixth-order batch from each saved d=5 phase source.
python3 -B recompute_other.py batch rp2-d5 --batch slice-03689 --output outputs/rp2-batch
python3 -B recompute_other.py batch fold-d5 --batch slice-03689 --output outputs/fold-batch

# Repeat the four-round repair's G4/G5 growth sum, using saved exact endpoints.
python3 -B recompute_other.py chan-growth --threads 1 --output outputs/chan-growth
```

Each command checks its result against the saved record. The d=3 command
recompiles the frozen model and contracts every slice for one prime and one
embedding of √2. It does not by itself reconstruct the rational coefficients.
The d=5 batch commands reuse compiled phase inputs; they do not rerun an entire
sixth-order campaign. The Chan growth command reuses the exact endpoint
functions and the certified contribution from growth orders zero to three.
Those boundaries are also recorded in the output JSON.

The [validation record](docs/VALIDATION.md) lists the calculations repeated
after refactoring, their measured times and the relocation check.

The refactor keeps the arithmetic kernels and saved plans, with small Python
entry points around them. It does not introduce a second implementation of the
physics. The [calculation notes](docs/OTHER_CALCULATIONS.md) explain what is
rebuilt and what is reused.

To repeat the direct ZX check of the old four-fault pattern, use
`requirements-zx.txt` and select a source:

```sh
python3 -B run_other_zx.py chan-v1-d5 --record 0 --method both --save-graphs --output outputs/chan-v1-zx
python3 -B run_other_zx.py chan-four-round-d5 --record 0 --method both --save-graphs --output outputs/chan-four-round-zx
```

Each command checks one of 16 accepted measurement records. `--outcome` selects
`reference`, `good` or `bad` (the default). The saved complete record sets give
`A=B=1/4` for the old pattern in arXiv v1 and zero in the four-round source.
`--save-graphs` writes the actual graph JSON, including intermediate cutting
terms. Scalar comparisons use a `1e-12` tolerance, separate from the exact
coefficient arithmetic.

## Original SOFT results

Acceptance probabilities, conditional logical error rates and series coefficients
for the standalone `d=3` and complete `d=5` SOFT cultivation circuits distributed
with Clifft and SymFT. The `d=5` calculation includes injection, `d=3` double
checking, noisy code growth, `d=5` double checking and noiseless final readout.

The package contains:

- Five exact `d=5` fractions at `p = 0.0005, 0.00075, 0.001, 0.0015, 0.002`.
- Exact `d=3` and `d=5` coefficients through `x^10`, where `x=p/(1-p)`.
- Prime certificates and character residues for reconstructing those results.
- Frozen circuits, binary tensor models, contraction plans and numerical comparison data.
- Fixed-fault ZX examples, including graph JSON before reduction, after each cutting
  round and after reduction. Graph files retain phases, scalars and term weights.

At `p=1/1000`, the complete `d=5` logical error rate is

```text
3.32842537141804345097245631814... × 10^-9
```

The numerator and denominator, rather than this decimal display, are stored in
[`data/exact_points/p001.json`](data/exact_points/p001.json).

### Verify and evaluate

Run these commands from this directory. They require Python 3.11 or later and
use only the standard library:

```sh
python3 -B verify_results.py
python3 -B -m unittest discover -s tests -v
python3 -B evaluate_series.py --distance 5 --p 1/1000 --order 10
python3 -B evaluate_series.py --distance 3 --p 1/1000 --order 10
```

The verifier reconstructs `A` and `B` from the supplied residues, checks the prime
witnesses and reconstruction bounds, then forms `P_L=B/A`. It also checks the raw
coefficients and their quotient recursion. The series command gives the finite
series at any specified `p`; at stored exact points it also reports the residual
and an acceptance-tail interval.

### Recompute a modular contraction

Use the packages pinned in `requirements-contraction.txt` and a C++17 compiler.
The command compiles the supplied kernels locally and uses one worker:

```sh
python3 -B recompute_prime.py --point p001 --prime-index 0 --output outputs/prime-p001-0
```

This rebuilds the final responses and contracts all eight growth characters,
checking both `A` and `B` against the stored residues. The exact first-stage
distribution is read from `data/endpoints/`. Each grid point and every prime in
its saved pool can be selected. One modulus supplies an exact modular result;
the complete fraction uses the full pool and the reconstruction bound.

### Recompute and inspect the ZX examples

Use `requirements-zx.txt`. This requires **pyzx-param**, the parametrised PyZX
fork. Its distribution version is `0.9.3`; its internal version string is `0.9.0`.

```sh
python3 -B run_zx.py d3 --method both --output outputs/zx-d3
python3 -B run_zx.py d5 --method both --output outputs/zx-d5
python3 -B -m unittest discover -s tests_zx -v
```

`both` evaluates the same measurement records by cut-and-collect followed by
stabiliser decomposition, and by direct reduction followed by stabiliser
decomposition. The outputs include the circuit texts, graph JSON and numerical
probabilities. These fixed-pattern examples give accepted-correct probability
zero and accepted-logical-error probability `1/4` at `d=3`, or `1/16` at `d=5`.
Their fault-free reference has probability one. The scalar probabilities use
complex floating point and are checked to absolute tolerance `1e-12`.

For the additional `d=3` doubled-diagram route, use `requirements-tsim.txt`:

```sh
python3 -B run_tsim.py --strategy cat5 --output outputs/tsim-d3
```

All output directories must be new. The distributed data remain unchanged.

## Files and conventions

| Location | Contents |
|---|---|
| `data/exact_points/` | Exact fractions and complete modular certificates |
| `data/series/` | Raw acceptance/error coefficients and conditional LER coefficients |
| `data/inputs/` | Frozen circuits and double-checking models |
| `data/models/`, `data/plan/` | Growth model, response plans and binary index maps |
| `data/endpoints/`, `data/proofs/` | Exact first-stage weights and reconstruction bounds |
| `data/comparisons/` | Clifft, SOFT and Gidney et al. numerical comparisons |
| `data/zx/graphs/` | Complete serialized graphs and cutting-stage terms |
| `data/zx/records/` | Additional numerical ZX records |
| `calculation/` | Pauli response, binary contraction and polynomial arithmetic |
| `zx/` | Circuit builder, ZX cutting, amplitude evaluation and graph interchange |
| `data/alternatives/series/` | One consistent `a`, `e`, `L` coefficient record per additional circuit |
| `data/alternatives/inputs/` | Frozen circuit models, source circuits and compiled phase inputs |
| `data/alternatives/certificates/` | Modular records, height bounds and compressed complete batch ledgers |
| `data/alternatives/comparisons/` | Supplied counts, digitised points and labelled diagnostic estimates |
| `alternatives/` | Fraction arithmetic, certificate checks and calculation adapters |

See [numerical conventions](docs/NUMERICS.md), [ZX graph format](docs/ZX.md),
[calculation guide](docs/CALCULATION.md) and [sources](docs/SOURCES.md).
`manifest.json` supplies SHA256 checksums for the distributed files.

The dependency files record the versions used with Python 3.11.13. Result
verification, native contraction and ZX evaluation have separate dependencies;
tensor-network order-search packages are not required to execute the saved plan.

Run the small regression tests with:

```sh
python3 -B -m unittest discover -s tests -v
python3 -B -m unittest discover -s tests_alternatives -v
```

Generated files go into `outputs/`, which is excluded from version control.
No command uploads results or modifies a manuscript.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{wan_zapirain_exact_ler_2026,
      title={Exact logical error rates for magic state cultivation}, 
      author={Kwok Ho Wan and Ainhoa Zapirain},
      year={2026},
      eprint={2609.18922},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2609.18922}, 
}
```

## Licence

Apache 2.0


## Acknowledgements

Parts of this codebase were developed with the assistance of LLMs.
