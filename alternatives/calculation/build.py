"""Compile the supplied C++17 source into a caller-selected output directory."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent


def build(name, directory, *, shared=True):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    suffix = (".dylib" if sys.platform == "darwin" else ".so") if shared else ""
    output = directory / (Path(name).stem + suffix)
    command = shlex.split(os.environ.get("CXX", "c++"))
    command += ["-O3", "-std=c++17", "-pthread"]
    if shared:
        command += ["-dynamiclib", "-fPIC"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    command += [str(SOURCE / name), "-o", str(output)]
    subprocess.run(command, check=True)
    return output
