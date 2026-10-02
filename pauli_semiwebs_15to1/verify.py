"""Check the manifest, the input drawings, the unit tests and the saved results.

Run from this folder:  python -B verify.py
"""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest():
    records = json.loads((ROOT / 'manifest.json').read_text())['files']
    for name, expected in records.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Invalid or missing manifest file: {name}')
        if sha256(path) != expected:
            raise ValueError(f'File checksum differs: {name}')
    return len(records)


def check_inputs():
    diagrams = json.loads((ROOT / 'provenance.json').read_text())['diagrams']
    for d in diagrams:
        if sha256(ROOT / d['file']) != d['sha256']:
            raise ValueError(f"Input drawing checksum differs: {d['file']}")
    return len(diagrams)


def check_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName('factory.test_factory')
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('unit tests failed')
    return result.testsRun


def check_results():
    from factory.run import analyse
    results, drawings = analyse()
    saved = json.loads((ROOT / 'results' / 'factory.json').read_text())
    if json.loads(json.dumps(results)) != saved:
        raise AssertionError('recomputed results differ from results/factory.json')
    for name, text in drawings.items():
        if (ROOT / 'webs' / name).read_text() != text:
            raise AssertionError(f'recomputed drawing differs: webs/{name}')
    return len(drawings)


def main():
    report = {'manifest_files': check_manifest(), 'input_drawings': check_inputs(),
              'unit_tests': check_tests(), 'recomputed_drawings': check_results()}
    print(json.dumps({'status': 'pass', **report}, indent=2))


if __name__ == '__main__':
    main()
