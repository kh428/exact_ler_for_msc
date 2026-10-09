"""Exact logical error rates of Clifford+T circuits with noise on the edges of their ZX diagram.

A circuit (stim or clifft text) is read into gates, its ZX edges are listed, every edge gets its own Pauli channel
and erasure rule from a noise specification, and the repository's Pauli-label tensor network gives A (acceptance),
B (accepted with the wrong logical outcome) and P_L = B/A as exact series with rational coefficients.

The tensors, the native kernels, the growth tree and the circuits are those of this repository, read in place.
Plans and compiled kernels are written to outputs/zx_edge_noise/ at the repository root.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
REPO = FOLDER.parent
INPUTS = REPO / 'data' / 'inputs'
OUTPUTS = REPO / 'outputs' / 'zx_edge_noise'

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
