"""Print saved exact series: coefficients, checks (0 <= b_k <= a_k <= C(N, k)) and P_L at a few p.

  python -B -m zxedge.summary results/*.json
"""
import json
import sys
from fractions import Fraction
from math import comb


def main():
    for path in sys.argv[1:]:
        d = json.loads(open(path).read())
        A = [Fraction(v) for v in d['A']]
        B = [Fraction(v) for v in d['B']]
        L = [Fraction(v) for v in d['P_L']]
        N = d['locations']
        print(f'{path}: {N} locations, degree {len(A) - 1}')
        ok = all(0 <= b <= a <= comb(N, k) for k, (a, b) in enumerate(zip(A, B)))
        print(f'  0 <= b_k <= a_k <= C(N,k) for every k: {ok}')
        for k, v in enumerate(L):
            print(f'  x^{k}: {str(v):>48}   {float(v):.6g}')
        for p in (Fraction(1, 10000), Fraction(1, 1000), Fraction(1, 500)):
            x = p / (1 - p)
            terms = [float(v * x ** k) for k, v in enumerate(L)]
            print(f'  p = {float(p):g}: P_L = {sum(terms):.6e} (last term {terms[-1]:.1e})')


if __name__ == '__main__':
    main()
