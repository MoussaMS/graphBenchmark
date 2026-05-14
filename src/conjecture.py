"""
conjecture.py
─────────────
Parse et représente une conjecture depuis le benchmark Excel.
Une conjecture est de la forme :
    Y(G)  <=  intercept + c1*X + c2*X² + ... + ck*X^k
    Y(G)  >=  intercept + c1*X + c2*X² + ... + ck*X^k
"""

from __future__ import annotations
import ast
from fractions import Fraction
from typing import List


# ─── mapping nom_colonne → clé invariant ────────────────────────────────────
INVARIANT_MAP = {
    "order":                               "n",
    "size":                                "m",
    "diameter":                            "diam",
    "radius":                              "rad",
    "density":                             "density",
    "minimum_degree":                      "delta",
    "maximum_degree":                      "Delta",
    "average_degree":                      "avg",
    "clique_number":                       "omega",
    "triangle_number":                     "triangles",
    "domination_number":                   "gamma",
    "total_domination_number":             "gamma_t",
    "independence_number":                 "alpha",
    "vertex_cover_number":                 "tau",
    "independent_domination_number":       "gamma_i",
    "matching_number":                     "mu",
    "vertex_connectivity":                 "kappa",
    "edge_connectivity":                   "kappa_edge",
    "connectivity":                        "kappa",
    "randic_index":                        "randic",
    "harmonic_index":                      "harmonic",
    "first_zagreb_index":                  "z1",
    "second_zagreb_index":                 "z2",
    "proximity":                           "proximity",
    "remoteness":                          "remoteness",
    "largest_eigenvalue":                  "lambda1",
    "largest_distance_eigenvalue":         "lambda_d",
    "second_smallest_laplace_eigenvalue":  "lambda_l2",
}


def _parse_fraction(s: str) -> float:
    """Convertit '1/6', '-3/5', '0', '106807/693000' → float."""
    s = str(s).strip()
    if not s or s in ("nan", "0"):
        return 0.0
    try:
        return float(Fraction(s))
    except Exception:
        return float(s)


def _parse_coeffs(raw) -> List[float]:
    """Parse la colonne Coefficients qui peut être une string repr de liste."""
    if isinstance(raw, list):
        return [_parse_fraction(c) for c in raw]
    s = str(raw).strip()
    try:
        lst = ast.literal_eval(s)
        return [_parse_fraction(c) for c in lst]
    except Exception:
        return [_parse_fraction(s)]


def _parse_subgroup(raw) -> List[str]:
    s = str(raw).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return [s]


class Conjecture:
    """
    Représente une conjecture :
        Y(G)  sign  intercept + c1*X + c2*X² + … + ck*X^k
    """

    def __init__(self, row):
        self.id        = int(row["Conjecture ID"])
        self.text      = str(row["Conjecture"])
        self.subgroup  = _parse_subgroup(row["Subgroup"])
        self.x_name    = str(row["X"]).strip()
        self.y_name    = str(row["Y"]).strip()
        self.sign      = str(row["Sign"]).strip()       # "<=" ou ">="
        self.coeffs    = _parse_coeffs(row["Coefficients"])
        self.intercept = _parse_fraction(row["Intercept"])
        self.degree    = int(row["Degree"])

        # clés utilisées dans le dict invariants
        self.x_key = INVARIANT_MAP.get(self.x_name, self.x_name)
        self.y_key = INVARIANT_MAP.get(self.y_name, self.y_name)

    # ── propriétés utiles ────────────────────────────────────────────────────

    @property
    def graph_class(self) -> List[str]:
        return self.subgroup

    @property
    def is_connected(self) -> bool:
        return "connected" in self.subgroup

    @property
    def is_tree(self) -> bool:
        return "tree" in self.subgroup

    @property
    def is_claw_free(self) -> bool:
        return "claw_free" in self.subgroup

    # ── évaluation ───────────────────────────────────────────────────────────

    def rhs(self, x_val: float) -> float:
        """Évalue le membre droit : intercept + c1*x + c2*x² + …"""
        result = self.intercept
        for k, c in enumerate(self.coeffs, start=1):
            result += c * (x_val ** k)
        return result

    def violation(self, invariants: dict) -> float:
        """
        violation > 0  ⟺  contre-exemple trouvé.

        Pour sign "<=":  violation = Y - rhs(X)      on veut Y > rhs(X)
        Pour sign ">=":  violation = rhs(X) - Y      on veut Y < rhs(X)
        """
        x_val = invariants.get(self.x_key, 0.0)
        y_val = invariants.get(self.y_key, 0.0)
        rhs   = self.rhs(x_val)

        if self.sign == "<=":
            return y_val - rhs
        else:   # ">="
            return rhs - y_val

    def __repr__(self):
        return (f"Conjecture(id={self.id}, "
                f"class={self.subgroup}, "
                f"Y={self.y_name} {self.sign} f({self.x_name}))")
