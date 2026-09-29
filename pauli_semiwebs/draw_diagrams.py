"""Render the full basis using the manuscript's final diagram layout."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from src.render import draw

ROOT = Path(__file__).resolve().parent


def render(data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for item in data['stages']:
        for record in item['generators']:
            record = dict(record)
            for field in ('input', 'output'):
                record[field] = {int(q): p for q, p in record[field].items()}
            name = record['id'] + '.tikz'
            draw(item['stage'], item['graph'], record, folder/name)
            names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/basis.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/diagrams')
    parser.add_argument('--check', action='store_true',
                        help='Compare regenerated drawings with the supplied TikZ')
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    if args.check:
        with TemporaryDirectory(prefix='semiweb_drawings_') as tmp:
            folder = Path(tmp)
            names = render(data, folder)
            changed = [name for name in names
                       if (folder/name).read_bytes() != (ROOT/'drawings/basis'/name).read_bytes()]
            if changed:
                raise RuntimeError(f'{len(changed)} drawings differ: {changed[:6]}')
        print(f'All {len(names)} regenerated TikZ files match byte for byte.')
    else:
        if args.output.resolve() == (ROOT/'drawings/basis').resolve():
            raise ValueError('Choose an output separate from the supplied drawings')
        names = render(data, args.output)
        print(f'Wrote {len(names)} drawings to {args.output}')


if __name__ == '__main__':
    main()
