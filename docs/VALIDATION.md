# Package validation

Checked on 21 September 2026, on macOS arm64 with Python 3.11.13.
Native checks used NumPy 2.4.6 and a local C++17 compiler. Plotting used
Matplotlib 3.11.1. ZX checks used `pyzx-param` distribution 0.9.3, whose
internal version string is 0.9.0, and Stim 1.16.0.

## Saved results

Both verification commands pass. They check the original five exact SOFT
points and both original series, plus all seven additional circuit variants.
The alternative checks include modular reconstruction, denominator and height
bounds, quotient recursion, and every stored batch checksum and exact sum.
They cover all 17,306 RP² sixth-order batches, 23,542 Fold sixth-order batches
and 23,543 Fold fifth-order batches. The complete saved Chan ZX record sets
are checked separately from the coefficient arithmetic.

The six original regression tests and nine additional regression tests pass.
The tests include rejection of a changed modular record, unavailable series
orders, and incomplete acceptance coefficients that would affect the quotient.

## Fresh calculations after the refactor

Each row below matched its archived reference. Times are wall-clock seconds,
including local kernel preparation, with one worker. They are measurements
of these checks, not estimates for a complete campaign.

| Calculation | Fresh work | Time (s) |
|---|---|---:|
| RP² d=3 | All slices through degree 5, one prime and one √2 embedding | 20.5 |
| Fold d=3 | All slices through degree 5, one prime and one √2 embedding | 75.1 |
| Chan arXiv v1 d=3 | Through degree 10, one prime and one √2 embedding | 8.2 |
| RP² d=5 | Complete sixth-order batch `slice-03689` | 3.1 |
| Fold d=5 | Complete sixth-order batch `slice-03689` | 5.4 |
| Chan four-round d=5 | Complete G4/G5 growth sum, with saved exact endpoints | 306.4 |

The d=3 checks rebuild tensors from frozen gate/noise models. The d=5 batch
checks reuse the compiled phase inputs. The Chan growth check rebuilds the
physical-channel power sums but reuses h1/h2 and the certified contribution
from growth orders zero to three. See [the calculation notes](OTHER_CALCULATIONS.md).

For each Chan d=5 source, measurement record 0 of the old four-fault pattern
was evaluated afresh by both ZX routes: cut-and-collect and direct reduction,
each followed by stabiliser decomposition. Both routes give accepted-wrong
probability 1/64 for this record in arXiv v1 and zero in the four-round source.
The full saved v1 sum over all 16 records is 1/4. These checks use complex
floating scalar arithmetic with absolute tolerance 1e-12.

## Relocation and figures

A copy of the package in a separate temporary directory passed both verifiers
and the series evaluator in a new environment without third-party packages.
It also compiled and repeated the RP² d=5 test batch using the native
dependencies, with no research folder on its import path.

All four comparison figures were regenerated. Their plotted values were
checked and the PNGs inspected. Sampling, proxy data and floating diagnostics
remain labelled; none are presented as additional exact fractions.

These checks establish that the packaged commands reproduce the stated
saved calculations. They do not constitute an independent implementation of
every circuit or a new full d=5 coefficient campaign. Linux and Windows have
not been tested for this package.
