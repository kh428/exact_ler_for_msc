"""Independent checks of the factory calculation.

Run from this folder:  python -B -m unittest factory.test_factory -v
"""

import cmath
import json
import re
import unittest
from functools import reduce

import numpy as np

from src.semiweb import BITS, constraints, defects

from . import HERE, statevector
from .diagram import incidence, read, supports, with_phase
from .run import FACTORY, LAYOUT, analyse
from .tensor import Tensor, product, times_unit

W = cmath.exp(1j * cmath.pi / 4)
# The fifteen Z-type pi/8 rotations of the 15-to-1 circuit, Fig. 3 of arXiv:1905.06903v3, read
# from that figure. Its qubits 1 to 5 are wires 0 to 4 here.
LITINSKI = [{1}, {2}, {3}, {4}, {1, 2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {0, 3, 4}, {0, 1, 4},
            {0, 2, 4}, {0, 1, 2, 3, 4}, {2, 3, 4}, {1, 3, 4}, {1, 2, 4}]


def number(entry):
    """An exact entry as a complex number."""
    return sum(int(a) * W ** j for j, a in enumerate(entry))


def dense_output(columns, phases, records):
    """The five-qubit calculation again, with 32 x 32 matrices in floating point."""
    z, one = np.diag([1.0, -1.0]), np.eye(2)
    state = np.ones(32, dtype=complex)
    for support, phase in zip(columns, phases):
        # wire 0 is the lowest bit, so it is the last factor of the Kronecker product
        z_s = reduce(np.kron, [z if w in support else one for w in (4, 3, 2, 1, 0)])
        state = np.exp(1j * np.pi / 8 * phase * (1 - np.diag(z_s))) * state
    out = np.zeros(2, dtype=complex)
    for x in range(32):
        sign = sum(((x >> w) & 1) * ((records >> (w - 1)) & 1) for w in (1, 2, 3, 4))
        out[x & 1] += (-1) ** sign * state[x]
    return out


class FactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = read(LAYOUT)
        cls.columns = supports(cls.base)
        cls.results, cls.drawings = analyse()

    def test_the_diagram_is_a_15_to_1_factory(self):
        self.assertEqual(supports(read(FACTORY)), self.columns)
        self.assertEqual(len(self.columns), 15)
        measured = sorted(sum(1 << (w - 1) for w in s if w) for s in self.columns)
        self.assertEqual(measured, list(range(1, 16)))
        self.assertTrue(all(len(s) % 2 == 1 for s in self.columns))
        self.assertEqual([sum(w in s for s in self.columns) for w in range(5)], [7, 8, 8, 8, 8])

    def test_the_rotations_are_those_of_the_published_circuit(self):
        self.assertEqual(sorted(sorted(s) for s in self.columns), sorted(sorted(s) for s in LITINSKI))

    def test_exact_arithmetic_against_complex_numbers(self):
        a, b = [3, -1, 4, 2], [-2, 5, 0, 7]
        self.assertAlmostEqual(number(product(a, b)), number(a) * number(b))
        for e in range(-8, 9):
            self.assertAlmostEqual(number(times_unit(a, e)), number(a) * W ** e)
        self.assertEqual(times_unit(a, 8).tolist(), a)
        self.assertEqual(times_unit(a, 4).tolist(), [-x for x in a])

    def test_diagram_tensor_against_five_qubits(self):
        choices = [[k] * 15 for k in range(8)] + [[(3 * i * i + i + 1) % 8 for i in range(15)],
                                                  [(5 * i + 2) % 8 for i in range(15)]]
        for site in range(15):
            for shift in (2, 4, 6):
                choices.append([1 + shift * (i == site) for i in range(15)])
        tensor = Tensor(with_phase(self.base, 1))
        for phases in choices:
            for r in range(16):
                exact = statevector.output(self.columns, phases, r)
                self.assertEqual(tensor(r, phases).tolist(), exact)
                dense = dense_output(self.columns, phases, r)
                for entry, value in zip(exact, dense):
                    self.assertAlmostEqual(number(entry), value, places=9)

    def test_local_rules_on_the_first_slide(self):
        """The examples of slides/illustrations.tex, by the library's rule and by matrices."""
        pauli = {'X': np.array([[0, 1], [1, 0]], dtype=complex), 'Y': np.array([[0, -1j], [1j, 0]]),
                 'Z': np.diag([1.0 + 0j, -1.0])}

        def graph(phase, legs):
            nodes = {'v': dict(kind='Z', phase=phase)}
            nodes.update({f'b{i}': dict(kind='boundary', phase=0) for i in range(legs)})
            return dict(nodes=nodes, edges=[dict(u='v', v=f'b{i}') for i in range(legs)])

        def spider(angle, legs):                            # |0...0> + e^{i angle}|1...1>
            out = np.zeros(1 << legs, dtype=complex)
            out[0], out[-1] = 1, np.exp(1j * angle)
            return out

        # phase in units of pi/4, labels on the legs, defect in units of pi/4
        cases = [(2, 'Y', 0), (1, 'Y', 2), (1, 'XX', -2), (1, 'ZZ', 0), (3, 'ZZ', 0), (2, 'XY', 0), (1, 'XY', 2)]
        # a Y label is drawn as a red and a green pi spider: X Z is Y up to a scalar
        self.assertTrue(np.allclose(pauli['X'] @ pauli['Z'], -1j * pauli['Y']))
        for phase, labels, delta in cases:
            vector = sum((BITS[p][0] << (2 * j)) | (BITS[p][1] << (2 * j + 1)) for j, p in enumerate(labels))
            self.assertEqual(defects(graph(phase, len(labels)), vector), {'v': delta} if delta else {})
            o = int(labels[0] in 'XY')
            s = sum(p in 'ZY' for p in labels) % 2
            self.assertEqual((-2 * o * phase + 4 * s - delta) % 8, 0)                # delta = -2 o theta + pi s
            w = reduce(np.kron, [pauli[p] for p in labels])
            lhs = w @ spider(phase * np.pi / 4, len(labels))
            rhs = spider((phase + delta) * np.pi / 4, len(labels))
            scale = lhs[np.argmax(abs(rhs))] / rhs[np.argmax(abs(rhs))]
            self.assertTrue(abs(abs(scale) - 1) < 1e-12 and np.allclose(lhs, scale * rhs), (phase, labels))

    def test_copy_and_fuse_rules(self):
        """The rules of slides/illustrations.tex: copy, fuse and unfuse, and copy at a T gate."""
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        z = np.diag([1.0 + 0j, -1.0])
        one = np.eye(2)

        def spider(angle):                                  # |000> + e^{i angle}|111>
            out = np.zeros(8, dtype=complex)
            out[0], out[7] = 1, np.exp(1j * angle)
            return out

        for k in range(8):
            theta = k * np.pi / 4
            copied = reduce(np.kron, [one, x, x]) @ spider(-theta)           # red pi on the two other legs
            self.assertTrue(np.allclose(reduce(np.kron, [x, one, one]) @ spider(theta), np.exp(1j * theta) * copied))
            self.assertTrue(np.allclose(reduce(np.kron, [z, one, one]) @ spider(theta), spider(theta + np.pi)))
            # unfuse: the green pi leaves again on another leg, and the spider is as it was
            self.assertTrue(np.allclose(reduce(np.kron, [z, one, one]) @ spider(theta),
                                        reduce(np.kron, [one, z, one]) @ spider(theta)))
        t_gate = np.diag([1, W])
        self.assertTrue(np.allclose(t_gate @ x, W * x @ t_gate.conj().T))    # T X = e^{i pi/4} X T_dag

    def test_output_statements_with_matrices(self):
        """The statements about the output, again, with ordinary complex matrices."""
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]])
        s_dag, t_dag = np.diag([1, -1j]), np.diag([1, W.conjugate()])
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        magic = (x - y) / np.sqrt(2)
        self.assertTrue(np.allclose(t_dag @ x @ t_dag.conj().T, magic))
        self.assertTrue(np.allclose(y @ s_dag, W ** 3 * magic))
        z_spider = lambda a: np.array([1, np.exp(1j * a)])
        x_spider = lambda a: h @ z_spider(a)                # |+> + e^{ia}|->, up to the same factor
        proportional = lambda u, v: abs(u[0] * v[1] - u[1] * v[0]) < 1e-12
        t_out = dense_output(self.columns, [1] * 15, 0)
        proxy_out = dense_output(self.columns, [2] * 15, 0)
        self.assertTrue(proportional(t_out, z_spider(-np.pi / 4)))
        self.assertTrue(np.allclose(magic @ t_out, t_out))
        z = np.diag([1.0, -1.0])
        values = [(t_out.conj() @ p @ t_out / (t_out.conj() @ t_out)).real for p in (x, y, z)]
        self.assertTrue(np.allclose(values, [1 / np.sqrt(2), -1 / np.sqrt(2), 0]))     # no Pauli eigenstate
        self.assertTrue(proportional(proxy_out, x_spider(np.pi / 2)))
        self.assertTrue(proportional(proxy_out, z_spider(-np.pi / 2)))
        self.assertTrue(np.allclose(y @ proxy_out, -proxy_out))
        # the +pi/2 shifts of the T states of one wire, as 32 x 32 diagonal operators
        bits = np.arange(32)
        for wire in range(5):
            total = sum(sum((bits >> w) & 1 for w in s) % 2 for s in self.columns if wire in s)
            operator = 1j ** total
            expected = (-1j) ** (bits & 1) if wire == 0 else np.ones(32)
            self.assertTrue(np.allclose(operator, expected), wire)

    def test_saved_results_and_drawings(self):
        saved = json.loads((HERE / 'results' / 'factory.json').read_text())
        self.assertEqual(json.loads(json.dumps(self.results)), saved)
        names = sorted(p.name for p in (HERE / 'webs').glob('*.tikz'))
        self.assertEqual(names, sorted(self.drawings))
        for name, text in self.drawings.items():
            self.assertEqual((HERE / 'webs' / name).read_text(), text, name)

    def test_drawings_read_back_as_the_stated_webs(self):
        """Parse each TikZ file on its own and recompute its records and defects."""
        node = re.compile(r'\\node\[style=(Z dot|X dot|Z phase dot|none)\] \(n(\d+)\) '
                          r'at \(([-\d.]+),([-\d.]+)\) \{(.*)\};')
        edge = re.compile(r'\\draw(?:\[style=([XYZ]) Web\])? \(n(\d+)\.center\) to \(n(\d+)\.center\);')
        phase_of = {r'$\frac{\pi}{4}$': 1, r'$\frac{\pi}{2}$': 2, '': 0}
        for name, text in sorted(self.drawings.items()):
            phase = 1 if name.startswith('t_') else 2
            g = with_phase(self.base, phase)
            found = {k: (style, float(x), float(y), label) for style, k, x, y, label in node.findall(text)}
            self.assertEqual(set(found), set(g['nodes']), name)
            for k, n in g['nodes'].items():
                style, x, y, label = found[k]
                self.assertEqual((x, y, phase_of[label]), (n['x'], n['y'], n['phase']), name)
                expected = {'boundary': 'none', 'X': 'X dot'}.get(n['kind'], 'Z phase dot' if n['phase'] else 'Z dot')
                self.assertEqual(style, expected, name)
            drawn = edge.findall(text)
            self.assertEqual([(u, v) for _, u, v in drawn], [(e['u'], e['v']) for e in g['edges']], name)
            vector = sum((BITS[label or 'I'][0] << (2 * j)) | (BITS[label or 'I'][1] << (2 * j + 1))
                         for j, (label, _, _) in enumerate(drawn))
            inc = incidence(g)
            read_records = sorted(n['record'] + 1 for k, n in g['nodes'].items()
                                  if 'record' in n and BITS['IXZY'[(vector >> (2 * inc[k][0])) & 3]][0])
            (out,) = [k for k, n in g['nodes'].items() if n['kind'] == 'boundary']
            out_label = 'IXZY'[(vector >> (2 * inc[out][0])) & 3]
            marks = defects(g, vector)                      # raises if the labels are not a semiweb
            comments = [json.loads(c) for c in re.findall(r'% defect: (\{.*\})', text)]
            self.assertEqual({f"n{k}": d for k, d in marks.items()},
                             {c['node']: c['delta_pi_over_4'] for c in comments}, name)
            self.assertEqual(len(re.findall(r'\\ast_', text)), len(marks), name)
            rings = re.findall(r'shape=circle,minimum size=7mm\] at \(n(\d+)\.center\)', text)
            self.assertEqual(sorted(rings), sorted(marks), name)
            bold = sorted(int(i) for i in re.findall(
                r'\\boldmath\] at \(\$\(n\d+\.center\)\+\(\.3,0\)\$\) \{\$r_\{(\d)\}\$\}', text))
            self.assertEqual(bold, read_records, name)
            if 'check' in name:
                self.assertEqual((out_label, read_records, len(marks)),
                                 ('I', [int(name[-6])], 8 if phase == 1 else 0), name)
            elif 'output' in name:
                self.assertEqual((out_label, read_records, len(marks)), ('Y', [], 7 if phase == 1 else 0), name)
            else:                                           # the diagram with no web on it
                self.assertEqual((vector, read_records, len(marks)), (0, [], 0), name)
            if phase == 2:                                  # a proxy drawing is a Pauli web
                self.assertFalse(any((row & vector).bit_count() & 1 for row in constraints(g, require_web=True)))


if __name__ == '__main__':
    unittest.main()
