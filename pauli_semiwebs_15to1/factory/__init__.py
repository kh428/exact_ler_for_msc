"""Pauli webs and semiwebs of a 15-to-1 magic state factory.

This folder reuses the semiweb code and drawing styles of ../pauli_semiwebs, which it
leaves unchanged, and reads the factory's ZX diagrams from diagrams/.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]      # this folder
REPO = HERE.parent                              # repository root
SEMIWEBS = REPO / 'pauli_semiwebs'
if str(SEMIWEBS) not in sys.path:
    sys.path.insert(0, str(SEMIWEBS))           # provides the package's src/
