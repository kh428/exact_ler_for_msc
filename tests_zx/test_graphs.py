"""Check graph interchange and probabilities directly from the shipped graphs."""
from pathlib import Path
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zx.graph_io import read_terms, graph_data, restore_graph
from zx.amplitude import cat_ladder, signature
from zx.circuits import build


class GraphChecks(unittest.TestCase):
    def test_built_circuits_match_saved_text(self):
        for case in ("d3", "d5"):
            folder = ROOT / "data/zx/graphs" / case
            self.assertEqual(build(case, with_faults=True)+"\n", (folder/"bad.stim").read_text())
            self.assertEqual(build(case, with_faults=False)+"\n", (folder/"reference.stim").read_text())

    def test_every_snapshot_roundtrips(self):
        for path in (ROOT/"data/zx/graphs").rglob("*.json"):
            if path.name == "summary.json":
                continue
            for graph, _ in read_terms(path):
                restored = restore_graph(graph_data(graph))
                self.assertEqual(signature(graph), signature(restored), str(path))
                # Compare fields, rather than a rounded scalar evaluation.
                self.assertEqual(graph_data(graph)["scalar"], graph_data(restored)["scalar"])

    def test_raw_graphs_give_expected_probabilities(self):
        for case, bad in (("d3", .25), ("d5", .0625)):
            for record, expected in (("reference", 1.), ("good", 0.), ("bad", bad)):
                terms = read_terms(ROOT/f"data/zx/graphs/{case}/{record}_raw.json")
                amplitude = sum(weight*cat_ladder(graph)[0] for graph, weight in terms)
                self.assertAlmostEqual(abs(amplitude)**2, expected, delta=1e-12)

    def test_reduced_collected_terms_give_expected_probabilities(self):
        for case, bad in (("d3", .25), ("d5", .0625)):
            for record, expected in (("reference", 1.), ("good", 0.), ("bad", bad)):
                terms = read_terms(ROOT/f"data/zx/graphs/{case}/{record}_reduced_terms.json")
                amplitude = sum(weight*cat_ladder(graph)[0] for graph, weight in terms)
                self.assertAlmostEqual(abs(amplitude)**2, expected, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
