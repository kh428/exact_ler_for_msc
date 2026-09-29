"""Check the manifest, the input circuits, the analyses, the claims and every TikZ file.

Run from this folder:  python -B verify.py [--quick]
--quick stops after the manifest, the input checksums and the unit tests.
"""

import argparse
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
    circuits = json.loads((ROOT / 'provenance.json').read_text())['circuits']
    for c in circuits:
        if sha256(ROOT / c['file']) != c['sha256']:
            raise ValueError(f"Input circuit checksum differs: {c['file']}")
    return len(circuits)


def check_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName('check_basis.test_check_basis')
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('unit tests failed')
    return result.testsRun


def check_results():
    from check_basis import faults, run
    saved = {str(r['distance']): r for r in json.loads((ROOT / 'results' / 'check_basis.json').read_text())}
    saved_faults = json.loads((ROOT / 'results' / 'fault_response.json').read_text())
    for key in (3, 5):
        for new, old in ((run.analyse(key, random_count=300, shots=64), saved[str(key)]),
                         (faults.analyse(key), saved_faults[str(key)])):
            differ = [k for k in new if k != 'cpu_seconds' and new[k] != old[k]]
            if differ:
                raise AssertionError(f'd={key}: recomputed results differ in {differ}')
    return 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='stop after the unit tests')
    args = parser.parse_args()
    report = {'manifest_files': check_manifest(), 'input_circuits': check_inputs(),
              'unit_tests': check_tests()}
    if not args.quick:
        from check_basis.guide import check_tikz
        from check_basis import verify_claims
        report['recomputed_results'] = check_results()
        passed, total = verify_claims.main()
        if passed != total:
            raise AssertionError(f'{total - passed} claim checks failed')
        report['claim_checks'] = total
        ok, files = check_tikz.main()
        if ok != files:
            raise AssertionError(f'{files - ok} TikZ files differ from their computed webs')
        report['tikz_files'] = files
    print(json.dumps({'status': 'pass', **report}, indent=2))


if __name__ == '__main__':
    main()
