"""Compile the supplied C++ kernels with a locally installed compiler."""
from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def build_library(name, build_dir=None):
    if name not in {"modular_binary.cpp", "modular_growth.cpp",
                    "modular_polynomial_endpoints.cpp", "modular_polynomial_growth.cpp"}:
        raise ValueError("Unknown native kernel")
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("A C++17 compiler is required for contraction; result verification needs only Python")
    build_dir = Path(build_dir) if build_dir else SOURCE.parent / "outputs/native"
    build_dir.mkdir(parents=True, exist_ok=True)
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.glob("*.cpp"))}
    key = hashlib.sha256((name+str(sorted(digests.items()))).encode()).hexdigest()[:20]
    output = build_dir / (Path(name).stem+"_"+key+".so")
    if not output.exists():
        subprocess.run([compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(SOURCE/name), "-o", str(output)], check=True)
    return output, {"sources": digests,
                    "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
