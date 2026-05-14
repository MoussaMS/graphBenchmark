"""
score.py
--------
Adaptive generic scoring for conjecture search.
"""

from __future__ import annotations

from typing import Dict


def _sc(inv: Dict[str, float], conjecture) -> float:
    viol = conjecture.violation(inv)

    if viol > 1e-9:
        return 1000.0 + viol

    n = max(inv.get("n", 1.0), 1.0)
    x_val = inv.get(conjecture.x_key, 0.0)
    y_val = inv.get(conjecture.y_key, 0.0)

    if conjecture.sign == "<=":
        direction = 0.3 * (y_val / max(n, 1)) - 0.1 * (x_val / max(n, 1))
    else:
        direction = -0.3 * (y_val / max(n, 1)) + 0.1 * (x_val / max(n, 1))

    structure = 0.05 * inv.get("diam", 0.0) + 0.03 * inv.get("Delta", 0.0)
    size_pen = 0.02 * max(0.0, n - 25)

    return 10.0 * viol + direction + structure - size_pen


def heuristic_score(G, invariants: Dict[str, float], conjecture) -> float:
    return _sc(invariants, conjecture)
