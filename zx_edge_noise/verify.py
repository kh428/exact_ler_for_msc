"""Check the manifest and the repository inputs, rebuild every saved series from its residues, run the tests.

Run from this folder:  python -B verify.py            (seconds)
                       python -B verify.py --slow     (also contracts networks again, about 25 minutes)
"""

import hashlib
import json
import sys
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
    """The repository files this folder reads in place must be the ones it was computed with."""
    inputs = json.loads((ROOT / 'provenance.json').read_text())['inputs']
    for item in inputs:
        path = (ROOT.parent / item['file']).resolve()
        if not path.is_relative_to(ROOT.parent) or not path.is_file():
            raise ValueError(f"Invalid or missing input file: {item['file']}")
        if sha256(path) != item['sha256']:
            raise ValueError(f"Input checksum differs: {item['file']}")
    return len(inputs)


def check_results():
    from zxedge import series as S
    report = {}
    for path in sorted((ROOT / 'results').glob('*.json')):
        result = json.loads(path.read_text())
        report[path.name] = S.check_record(result)
    return report


def check_tests(slow):
    names = ['tests.test_fast'] + (['tests.test_slow'] if slow else [])
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('tests failed')
    return result.testsRun


def main():
    slow = '--slow' in sys.argv[1:]
    report = {'manifest_files': check_manifest(), 'repository_inputs': check_inputs(), 'results': check_results(),
              'tests': check_tests(slow)}
    print(json.dumps({'status': 'pass', **report}, indent=2))


if __name__ == '__main__':
    main()
