# An explainer and a film

Two pages that explain the calculation of
[Exact logical error rates for magic state cultivation](https://arxiv.org/abs/2609.18922), from Pauli propagation to
the exact series. They follow the paper and use its figures; they add no new results.

- [`exact_ler_atlas.html`](exact_ler_atlas.html), the Exact LER Atlas: a zoomable map of the calculation in 24
  frames, each with the paper's figures, its equations and small interactive widgets. Six frames work through Pauli
  propagation in detail: the rules, small circuits by hand, the real d=3 double check moment by moment, propagation
  next to Pauli semiwebs, the response table, and propagation as one way of contracting a tensor network.
- [`exact_ler_movie.html`](exact_ler_movie.html), the calculation as a film: thirteen scenes, about four minutes,
  from one Pauli string carried backwards through a gate to residues modulo primes, the Chinese remainder theorem
  and the exact series, with chapters and a short version of each stage.

## Viewing

GitHub shows these files as source. Clone or download the repository and open either page in a browser. The
figures load from [`svg`](svg) and [`img`](img) next to the pages, so keep the folder together. The equations are
typeset by MathJax, loaded from jsDelivr, so the maths needs a network connection; without it the film shows plain
Unicode maths and the Atlas the TeX source.

## Contents

| Path | What it holds |
|---|---|
| `exact_ler_atlas.html`, `exact_ler_movie.html` | the two pages; data and the Latin Modern fonts are embedded |
| `svg/` | the paper's figures and tables as SVG, and a few drawn for these pages, captioned "Drawn for this page" |
| `img/` | the paper's plots as PNG |
| `prop/` | an exact Pauli-propagation engine and the script behind the numbers on the propagation pages and in the film |

## Checking the numbers

The strings, noise factors, detector counts and residues on the propagation pages and in the film come from
[`prop/build_prop_data.py`](prop/build_prop_data.py). With `numpy` and `stim` from
[`../requirements-zx.txt`](../requirements-zx.txt) installed, run from the repository root:

```sh
python3 -B explainer/prop/build_prop_data.py
```

It recomputes everything from the circuit files in [`../data/inputs`](../data/inputs) and the stored residues in
[`../data/series`](../data/series), checks each piece before using it, writes
`outputs/explainer/prop_data.json`, and confirms that the result is identical to the copy embedded in the Atlas.
The checks include:

- the propagated mean of each small circuit against a density-matrix simulation at three noise strengths;
- the noise factor of the central d=3 readout, lambda_1^20 lambda_2^12 f^5, against Stim's detector error model of
  the same lines with S in place of T;
- the d=3 central readout reaching the data as 128 strings, of which 16 survive on the code space as M_L, and at d=5
  as 2^19 strings, of which 1024 survive;
- all twenty declared d=3 detectors carried back through the whole circuit: each is deterministic, and the counts of
  defects match the drawn webs of [`../pauli_semiwebs_check`](../pauli_semiwebs_check);
- a one-qubit double check from propagation to its exact series, with residues modulo small primes from a separate
  contraction of its transfer-matrix network;
- the stored d=3 and d=5 residues rebuilt by the Chinese remainder theorem into 32/75, 574/375 and the tenth-order
  coefficient a_10 at d=5.

## Notes

Records are numbered from 1 as in Fig. 19 of the paper, and detectors from 0 as in Stim and Table 9. The pages embed
the Latin Modern fonts, distributed under the GUST Font License
([`../licenses/LatinModern-GUST-Font-License.txt`](../licenses/LatinModern-GUST-Font-License.txt)).
