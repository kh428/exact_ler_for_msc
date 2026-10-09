"""The d=5 SOFT circuit in three stages, as in Appendix E of the paper.

The first stage (injection and the d=3 double check) becomes the table W over the d=3 input syndrome and logical tag,
the Clifford growth becomes XOR-convolved label distributions in the leaves of the repository's growth tree, and the
final stage (d=5 double check and noiseless readout) becomes the response table F over the d=5 frame.

stabilizer/ holds the stabilizer-support helpers of the compiler that wrote data/models/growth_frame_model.json,
unchanged; growth.py uses them to track the Choi reference of the growth instrument.
"""
import sys
from pathlib import Path

STABILIZER = Path(__file__).resolve().parent / 'stabilizer'
if str(STABILIZER) not in sys.path:
    sys.path.insert(0, str(STABILIZER))
