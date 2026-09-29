"""Check local tensors, complete bases, diagrams and optional d=3 operators."""

import argparse
import hashlib
import json
from pathlib import Path
from verification import check_diagrams, check_operators

ROOT = Path(__file__).resolve().parent


def check_manifest():
    manifest = ROOT/'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('manifest.json is missing')
    records = json.loads(manifest.read_text())['files']
    for name, expected in records.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Invalid or missing manifest file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'File checksum differs: {name}')
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='Also evaluate every d=3 basis/example identity on all 128 inputs')
    parser.add_argument('--output', type=Path,
                        help='Optional JSON report; no file is written by default')
    args = parser.parse_args()
    result = {'manifest_files': check_manifest(), 'diagrams': check_diagrams.main()}
    if args.full:
        result['operators'] = check_operators.main()
    if args.output:
        destination = args.output.resolve()
        listed = json.loads((ROOT/'manifest.json').read_text())['files']
        if destination == (ROOT/'manifest.json') or any(destination == ROOT/p for p in listed):
            raise ValueError('Choose a report path separate from the distributed files')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({
        'status': 'pass', 'manifest_files': result['manifest_files'],
        'local_tensor_checks': result['diagrams']['exact_local_tensor_checks'],
        'defect_markers': result['diagrams']['defect_markers_checked'],
        'basis_dimensions': [s['semiweb_dimension'] for s in result['diagrams']['stages']],
        'd3_matrix_columns': result.get('operators', {}).get('exact_matrix_columns_checked', 0),
    }, indent=2))


if __name__ == '__main__':
    main()
