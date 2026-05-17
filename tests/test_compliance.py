import re
import unittest
from pathlib import Path

import networkx as nx

from src.validation import STRICT_NP_N, validate_candidate, validate_graph_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DummyConjecture:
    def __init__(self, graph_class, x_key="n", y_key="m", sign="<=", intercept=0.0, coeffs=None):
        self.graph_class = list(graph_class)
        self.x_key = x_key
        self.y_key = y_key
        self.sign = sign
        self.intercept = float(intercept)
        self.coeffs = list(coeffs or [])

    def rhs(self, x_val: float) -> float:
        out = self.intercept
        for k, c in enumerate(self.coeffs, start=1):
            out += c * (x_val ** k)
        return out

    def violation(self, invariants: dict) -> float:
        x_val = float(invariants.get(self.x_key, 0.0))
        y_val = float(invariants.get(self.y_key, 0.0))
        rhs = self.rhs(x_val)
        if self.sign == "<=":
            return y_val - rhs
        return rhs - y_val


class ComplianceTests(unittest.TestCase):
    def test_no_conjecture_id_shortcuts(self):
        code_files = [PROJECT_ROOT / "main.py"] + list((PROJECT_ROOT / "src").glob("*.py"))
        forbidden = [
            re.compile(r"conjecture\.id\s*=="),
            re.compile(r"if\s+.*\bconj(?:ecture)?\.id\b"),
            re.compile(r"\bif\s+id\s*=="),
        ]
        for path in code_files:
            text = path.read_text(encoding="utf-8")
            for pat in forbidden:
                self.assertIsNone(
                    pat.search(text),
                    msg=f"Forbidden conjecture-ID conditional found in {path}: {pat.pattern}",
                )

    def test_graph_class_validation_connected_tree_bipartite_planar_claw_free(self):
        conj_connected = DummyConjecture(["connected"])
        self.assertTrue(validate_graph_class(nx.path_graph(4), conj_connected.graph_class)[0])
        self.assertFalse(validate_graph_class(nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)), conj_connected.graph_class)[0])

        conj_tree = DummyConjecture(["tree"])
        self.assertTrue(validate_graph_class(nx.path_graph(6), conj_tree.graph_class)[0])
        self.assertFalse(validate_graph_class(nx.cycle_graph(6), conj_tree.graph_class)[0])

        conj_bip = DummyConjecture(["bipartite"])
        self.assertTrue(validate_graph_class(nx.complete_bipartite_graph(3, 4), conj_bip.graph_class)[0])
        self.assertFalse(validate_graph_class(nx.cycle_graph(5), conj_bip.graph_class)[0])

        conj_planar = DummyConjecture(["planar"])
        self.assertTrue(validate_graph_class(nx.cycle_graph(8), conj_planar.graph_class)[0])
        self.assertFalse(validate_graph_class(nx.complete_graph(5), conj_planar.graph_class)[0])

        conj_claw = DummyConjecture(["claw_free"])
        self.assertTrue(validate_graph_class(nx.cycle_graph(6), conj_claw.graph_class)[0])
        self.assertFalse(validate_graph_class(nx.star_graph(3), conj_claw.graph_class)[0])  # K1,3

    def test_violation_must_be_strict(self):
        # For n=2 path graph: m=1. Conjecture m <= n-1 gives equality -> not strict.
        eq_conj = DummyConjecture(["connected"], x_key="n", y_key="m", sign="<=", intercept=-1.0, coeffs=[1.0])
        eq_res = validate_candidate(nx.path_graph(2), eq_conj)
        self.assertFalse(eq_res["valid"])
        self.assertIn("non-strict violation", eq_res["reason"])

        # Strictly violated: m <= n-2 on path_graph(3) (m=2, rhs=1).
        strict_conj = DummyConjecture(["connected"], x_key="n", y_key="m", sign="<=", intercept=-2.0, coeffs=[1.0])
        strict_res = validate_candidate(nx.path_graph(3), strict_conj)
        self.assertTrue(strict_res["valid"])
        self.assertGreater(strict_res["violation"], 0.0)

    def test_required_np_hard_invariant_must_be_exact(self):
        # alpha exact is disabled past STRICT_NP_N in strict validation -> must reject.
        n = STRICT_NP_N + 6
        G = nx.path_graph(n)
        conj = DummyConjecture(["connected"], x_key="alpha", y_key="alpha", sign="<=", intercept=-1.0, coeffs=[1.0])
        out = validate_candidate(G, conj)
        self.assertFalse(out["valid"])
        self.assertIn("required invariant", out["reason"])


if __name__ == "__main__":
    unittest.main()

