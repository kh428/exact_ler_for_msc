"""Recompute both semiweb bases and their exact phase records."""

import argparse
import json
from pathlib import Path
from src.generate import compute

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='Compare with data/basis.json without writing')
    parser.add_argument('--output', type=Path, default=ROOT/'output/basis.json')
    args = parser.parse_args()
    result = compute()
    text = json.dumps(result, indent=2) + '\n'
    if args.check:
        expected = json.loads((ROOT/'data/basis.json').read_text())
        if json.loads(text) != expected:
            raise RuntimeError('Recomputed basis differs from the saved records')
        print('Both bases, all defects and all scalars agree with data/basis.json.')
    else:
        if args.output.resolve() == (ROOT/'data/basis.json').resolve():
            raise ValueError('Choose an output separate from the reference data')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    for item in result['stages']:
        print(f"d={item['stage']['distance']}: {item['counts']['semiweb_dimension']} basis elements")


if __name__ == '__main__':
    main()
