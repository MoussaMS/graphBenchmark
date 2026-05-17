"""
score.py
--------
Adaptive generic scoring for conjecture search.
"""

from __future__ import annotations

from typing import Dict

from .profiles import invariant_family


VIOL_EPS = 1e-9
SUCCESS_BASE = 5000.0


def _rhs_derivative(conjecture, x_val: float) -> float:
    """
    Derivative of:
        intercept + c1*x + c2*x^2 + ... + ck*x^k
    with respect to x.
    """
    deriv = 0.0
    for k, coeff in enumerate(conjecture.coeffs, start=1):
        deriv += k * coeff * (x_val ** (k - 1))
    return deriv


def _sc(inv: Dict[str, float], conjecture) -> float:
    raw_viol = float(conjecture.violation(inv))
    n = max(float(inv.get("n", 1.0)), 1.0)
    x_family = invariant_family(conjecture.x_key)
    y_family = invariant_family(conjecture.y_key)

    x_val = float(inv.get(conjecture.x_key, 0.0))
    y_val = float(inv.get(conjecture.y_key, 0.0))
    rhs_val = float(conjecture.rhs(x_val))
    scale = 1.0 + abs(rhs_val) + abs(y_val)
    norm_viol = raw_viol / scale
    if "distance" in {x_family, y_family}:
        size_pen = 0.015 * max(0.0, n - 46.0)
    elif "domination_cover" in {x_family, y_family}:
        size_pen = 0.022 * max(0.0, n - 42.0)
    else:
        size_pen = 0.035 * max(0.0, n - 35.0)

    # Strict violation is overwhelmingly preferred.
    if raw_viol > VIOL_EPS:
        cert = min(250.0, 80.0 * abs(norm_viol) + min(raw_viol, 120.0))
        return SUCCESS_BASE + 1200.0 * norm_viol + raw_viol + cert - size_pen

    # Pre-violation guidance:
    # - maximize normalized violation (closest to crossing),
    # - steer x with local derivative information,
    # - use lightweight structural hints.
    closeness = 1500.0 * norm_viol

    d_rhs = _rhs_derivative(conjecture, x_val)
    d_viol_dx = -d_rhs if conjecture.sign == "<=" else d_rhs

    x_norm = x_val / n
    y_norm = y_val / n

    y_push = y_norm if conjecture.sign == "<=" else -y_norm
    x_push_sign = 1.0 if d_viol_dx >= 0.0 else -1.0
    x_push_scale = min(1.0, abs(d_viol_dx))
    x_push = x_push_sign * x_norm * x_push_scale

    density = float(inv.get("density", 0.0))
    diam_norm = float(inv.get("diam", 0.0)) / n
    delta_norm = float(inv.get("Delta", 0.0)) / max(1.0, n - 1.0)

    structure = 0.12 * density + 0.08 * diam_norm + 0.06 * delta_norm

    # Extra pressure for distance-sensitive relations.
    if "distance" in {x_family, y_family} or conjecture.x_key == "lambda_d" or conjecture.y_key == "lambda_d":
        structure += 0.07 * diam_norm

    # Domination-cover <-> distance inequalities are the current hard family:
    # reinforce joint guidance on x and y in the same direction as violation.
    if {x_family, y_family} == {"domination_cover", "distance"}:
        closeness *= 1.18
        if conjecture.sign == ">=":
            structure += 0.16 * x_norm - 0.22 * y_norm
        else:
            structure += -0.16 * x_norm + 0.22 * y_norm

    if "spectral" in {x_family, y_family}:
        structure += 0.06 * min(2.0, abs(float(inv.get("lambda1", 0.0))) / max(1.0, n))

    return closeness + 3.2 * y_push + 2.4 * x_push + structure - size_pen


def heuristic_score(G, invariants: Dict[str, float], conjecture) -> float:
    return _sc(invariants, conjecture)
