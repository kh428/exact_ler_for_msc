# Scientific sources and attribution

- **Clifft:** circuit distribution and sampling comparisons,
  [arXiv:2604.27058](https://arxiv.org/abs/2604.27058).
- **SymFT:** equivalent circuit distribution,
  [arXiv:2607.28600](https://arxiv.org/abs/2607.28600).
- **SOFT:** the circuit family and numerical comparison,
  [arXiv:2512.23037](https://arxiv.org/abs/2512.23037).
- **Gidney et al.:** cultivation comparison data,
  [archived dataset](https://zenodo.org/records/13777072).
- **Pauli propagation:** [arXiv:2505.21606](https://arxiv.org/abs/2505.21606).
- **ZX cutting:** [arXiv:2509.08658](https://arxiv.org/abs/2509.08658) and
  [companion code](https://github.com/kh428/accel-cutting-magic-state).

`zx/cutting.py` retains the cutting implementation from that companion code.
The package uses the parametrised PyZX fork through an installed dependency;
the dependency's own source is not vendored. Existing Apache-2.0 notices for
the incorporated Clifft, SymFT and cutting sources are retained in `licenses/`.

The circuit SHA256 values in the point files identify the exact source used.
`manifest.json` identifies the files distributed here. Workstation paths and
private launch records are not part of either scientific identifier.

## Numerical comparison conventions

The comparison JSON files retain source links and extraction information for
published intervals. In particular, the Clifft point at `p=0.0007` remains at
that value; it is distinct from the additional exact point at `0.00075`.
No separate SymFT sampling curve is supplied.

The Gidney et al. proxy and statevector curves retain their own circuit and
injection conventions. The `T/S` comparison data include the zero-observed-error
upper limit at `p=0.0005`. Its ratio is a bound, not a measured finite ratio.
The interval conventions are those recorded in the comparison files.
