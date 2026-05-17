"""
profiles.py
-----------
Generic invariant-family helpers used to adapt search strategy.
"""

from __future__ import annotations

from typing import Dict, Set


SIZE_KEYS = {"n", "m"}
DEGREE_DENSITY_KEYS = {
    "delta",
    "Delta",
    "avg",
    "density",
    "omega",
    "triangles",
    "randic",
    "harmonic",
    "z1",
    "z2",
}
DISTANCE_KEYS = {"diam", "rad", "proximity", "remoteness", "lambda_d"}
DOMINATION_COVER_KEYS = {"gamma", "gamma_t", "gamma_i", "alpha", "tau", "mu"}
CONNECTIVITY_KEYS = {"kappa", "kappa_edge"}
SPECTRAL_KEYS = {"lambda1", "lambda_l2"}


def invariant_family(key: str) -> str:
    """
    Return a broad family label for an invariant key.
    """
    k = str(key or "").strip()
    if k in SIZE_KEYS:
        return "size"
    if k in DEGREE_DENSITY_KEYS:
        return "degree_density"
    if k in DISTANCE_KEYS:
        return "distance"
    if k in DOMINATION_COVER_KEYS:
        return "domination_cover"
    if k in CONNECTIVITY_KEYS:
        return "connectivity"
    if k in SPECTRAL_KEYS:
        return "spectral"
    return "other"


def conjecture_profile(conjecture) -> Dict[str, object]:
    x_key = str(conjecture.x_key)
    y_key = str(conjecture.y_key)
    x_family = invariant_family(x_key)
    y_family = invariant_family(y_key)
    families: Set[str] = {x_family, y_family}

    return {
        "x_key": x_key,
        "y_key": y_key,
        "x_family": x_family,
        "y_family": y_family,
        "families": families,
        "is_tree": "tree" in conjecture.graph_class,
        "is_claw_free": "claw_free" in conjecture.graph_class,
        "distance_sensitive": "distance" in families,
        "spectral_sensitive": "spectral" in families or ("lambda_d" in {x_key, y_key}),
        "domination_distance_pair": families == {"domination_cover", "distance"},
        "domination_heavy": x_family == "domination_cover" or y_family == "domination_cover",
    }
