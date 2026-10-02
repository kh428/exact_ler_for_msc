"""Build slides/slides.pdf: the proxy webs, the semiwebs with T states and the output web.

Run from this folder:  python -B -m factory.slides
The drawing styles are read from ../pauli_semiwebs/drawings through TEXINPUTS. pdflatex runs
twice in a temporary directory, so no auxiliary file is left in this folder.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import HERE, SEMIWEBS


def main():
    env = dict(os.environ, TEXINPUTS=f'.:{SEMIWEBS}:')
    with tempfile.TemporaryDirectory() as build:
        for _ in range(2):
            subprocess.run(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error',
                            '-output-directory', build, 'slides/slides.tex'],
                           cwd=HERE, env=env, check=True, stdout=subprocess.DEVNULL)
        shutil.copyfile(Path(build) / 'slides.pdf', HERE / 'slides' / 'slides.pdf')
    print(HERE / 'slides' / 'slides.pdf')


if __name__ == '__main__':
    main()
