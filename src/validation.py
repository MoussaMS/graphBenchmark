"""
validation.py
-------------
Strict counterexample validation utilities.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

import networkx as nx

from .invariants import compute_invariants


STRICT_EPS = 1e-9
STRICT_NP_N = 28


def normalize_graph(G: nx.Graph) -> nx.Graph:
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return nx.convert_node_labels_to_integers(H)


def find_induced_claw(G: nx.Graph):
    """Return (center, a, b, c) witness of an induced K1,3, or None if claw-free."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    a, b, c = neighbors[i], neighbors[j], neighbors[k]
                    if not G.has_edge(a, b) and not G.has_edge(b, c) and not G.has_edge(a, c):
                        return (v, a, b, c)
    return None


def is_claw_free(G: nx.Graph) -> bool:
    """Return True iff G contains no induced K1,3 subgraph."""
    return find_induced_claw(G) is None


def has_claw(G: nx.Graph) -> bool:
    return not is_claw_free(G)


def validate_graph_class(G: nx.Graph, graph_class) -> Tuple[bool, str]:
    H = normalize_graph(G)
    if isinstance(graph_class, str):
        classes = {graph_class}
    else:
        classes = set(graph_class)

    if H.number_of_nodes() < 2:
        return False, "order < 2"
    if any(u == v for u, v in H.edges()):
        return False, "self-loop detected"

    if "connected" in classes:
        if H.number_of_nodes() == 0 or not nx.is_connected(H):
            return False, "not connected"
    if "tree" in classes and not nx.is_tree(H):
        return False, "not a tree"
    if "bipartite" in classes and not nx.is_bipartite(H):
        return False, "not bipartite"
    if "planar" in classes:
        ok, _ = nx.check_planarity(H)
        if not ok:
            return False, "not planar"
    if "claw_free" in classes and has_claw(H):
        return False, "claw found"

    return True, "ok"


def satisfies_class(G: nx.Graph, conjecture) -> Tuple[bool, str]:
    return validate_graph_class(G, conjecture.graph_class)


def graph6_roundtrip(G: nx.Graph) -> Tuple[str, nx.Graph]:
    H = normalize_graph(G)
    g6 = nx.to_graph6_bytes(H, header=False).decode().strip()
    H2 = nx.from_graph6_bytes(g6.encode())
    H2 = normalize_graph(H2)
    return g6, H2


def true_violation(invariants: Dict[str, float], conjecture) -> float:
    return float(conjecture.violation(invariants))


def validate_counterexample(G: nx.Graph, conjecture, eps: float = STRICT_EPS) -> Dict[str, object]:
    try:
        H = normalize_graph(G)
        ok, reason = satisfies_class(H, conjecture)
        if not ok:
            return {"valid": False, "reason": reason}

        g6, H2 = graph6_roundtrip(H)
        ok2, reason2 = satisfies_class(H2, conjecture)
        if not ok2:
            return {"valid": False, "reason": f"roundtrip class failure: {reason2}"}

        strict_inv = _compute_required_invariants_strict(H2, conjecture)
        if strict_inv is None:
            return {"valid": False, "reason": "required invariant(s) not exactly computable"}

        # Required keys come from strict computation only. Extra keys are for reporting.
        full_inv = dict(strict_inv)
        try:
            approx_inv = compute_invariants(H2)
        except Exception:
            approx_inv = {}
        for k, v in approx_inv.items():
            if k not in full_inv:
                full_inv[k] = v

        violation = true_violation(full_inv, conjecture)
        if violation <= eps:
            return {"valid": False, "reason": f"non-strict violation ({violation:.6g})"}

        return {
            "valid": True,
            "reason": "ok",
            "graph": H2,
            "graph6": g6,
            "invariants": full_inv,
            "violation": violation,
        }
    except Exception as e:
        return {"valid": False, "reason": f"exception: {e}"}


def validate_candidate(G: nx.Graph, conjecture, eps: float = STRICT_EPS) -> Dict[str, object]:
    return validate_counterexample(G, conjecture, eps=eps)


def _compute_required_invariants_strict(G: nx.Graph, conjecture) -> Dict[str, float] | None:
    keys = {str(conjecture.x_key), str(conjecture.y_key)}
    if "tau" in keys:
        keys.add("alpha")
    if "avg" in keys or "density" in keys:
        keys.update({"n", "m", "delta", "Delta"})

    base = _base_invariants(G)
    cache: Dict[str, float] = dict(base)

    for key in keys:
        value = _strict_invariant_value(key, G, cache)
        if value is None:
            return None
        cache[key] = float(value)
    return {k: float(v) for k, v in cache.items() if k in keys or k in {conjecture.x_key, conjecture.y_key, "n", "m"}}


def _base_invariants(G: nx.Graph) -> Dict[str, float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = [d for _, d in G.degree()]
    delta = float(min(deg)) if deg else 0.0
    Delta = float(max(deg)) if deg else 0.0
    avg = float(sum(deg) / n) if n > 0 else 0.0
    density = float(nx.density(G)) if n > 0 else 0.0
    return {"n": float(n), "m": float(m), "delta": delta, "Delta": Delta, "avg": avg, "density": density}


def _strict_invariant_value(key: str, G: nx.Graph, cache: Dict[str, float]) -> float | None:
    if key in cache:
        return float(cache[key])
    n = int(cache.get("n", G.number_of_nodes()))

    if key == "diam":
        if n <= 1:
            return 0.0
        if not nx.is_connected(G):
            return None
        return float(nx.diameter(G))

    if key == "rad":
        if n <= 1:
            return 0.0
        if not nx.is_connected(G):
            return None
        return float(nx.radius(G))

    if key in {"proximity", "remoteness"}:
        if n <= 1 or not nx.is_connected(G):
            return None
        prox, remote = _distance_profile_exact(G)
        return prox if key == "proximity" else remote

    if key == "omega":
        return float(len(max(nx.find_cliques(G), key=len, default=[])))

    if key == "triangles":
        return float(sum(nx.triangles(G).values()) // 3)

    if key == "mu":
        return float(len(nx.max_weight_matching(G, maxcardinality=True)))

    if key == "kappa":
        return float(nx.node_connectivity(G))

    if key == "kappa_edge":
        return float(nx.edge_connectivity(G))

    if key == "randic":
        return float(
            sum(
                1.0 / math.sqrt(G.degree(u) * G.degree(v))
                for u, v in G.edges()
                if G.degree(u) > 0 and G.degree(v) > 0
            )
        )

    if key == "harmonic":
        return float(
            sum(
                2.0 / (G.degree(u) + G.degree(v))
                for u, v in G.edges()
                if (G.degree(u) + G.degree(v)) > 0
            )
        )

    if key == "z1":
        return float(sum(G.degree(u) + G.degree(v) for u, v in G.edges()))

    if key == "z2":
        return float(sum(G.degree(u) * G.degree(v) for u, v in G.edges()))

    if key in {"lambda1", "lambda_l2", "lambda_d"}:
        try:
            import numpy as np
        except Exception:
            return None
        if key == "lambda1":
            A = nx.to_numpy_array(G, dtype=float)
            evals = sorted(np.linalg.eigvalsh(A), reverse=True)
            return float(evals[0]) if evals else 0.0
        if key == "lambda_l2":
            A = nx.to_numpy_array(G, dtype=float)
            L = np.diag([d for _, d in G.degree()]).astype(float) - A
            levals = sorted(np.linalg.eigvalsh(L))
            return float(levals[1]) if len(levals) > 1 else 0.0
        if not nx.is_connected(G):
            return None
        D = nx.floyd_warshall_numpy(G).astype(float)
        devals = sorted(np.linalg.eigvalsh(D), reverse=True)
        return float(devals[0]) if devals else 0.0

    if key == "alpha":
        if n > STRICT_NP_N:
            return None
        C = nx.complement(G)
        return float(len(max(nx.find_cliques(C), key=len, default=[])))

    if key == "tau":
        alpha = _strict_invariant_value("alpha", G, cache)
        if alpha is None:
            return None
        return float(n) - float(alpha)

    if key == "gamma":
        if n > STRICT_NP_N:
            return None
        masks = _closed_neighborhood_masks(G)
        return float(_exact_set_cover_size(masks, n))

    if key == "gamma_t":
        if n > STRICT_NP_N:
            return None
        if any(G.degree(v) == 0 for v in G.nodes()):
            return None
        masks = _open_neighborhood_masks(G)
        return float(_exact_set_cover_size(masks, n))

    if key == "gamma_i":
        if n > STRICT_NP_N:
            return None
        return float(_exact_independent_domination_number(G))

    return None


def _distance_profile_exact(G: nx.Graph) -> Tuple[float, float]:
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, 0.0
    all_lengths = dict(nx.all_pairs_shortest_path_length(G))
    averages = []
    for u in G.nodes():
        dist_map = all_lengths.get(u, {})
        if len(dist_map) != n:
            return 0.0, 0.0
        total = float(sum(dist_map.values()))
        averages.append(total / max(1.0, float(n - 1)))
    if not averages:
        return 0.0, 0.0
    return float(min(averages)), float(max(averages))


def _closed_neighborhood_masks(G: nx.Graph) -> Tuple[int, ...]:
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    masks = []
    for v in nodes:
        mask = 1 << idx[v]
        for u in G.neighbors(v):
            mask |= 1 << idx[u]
        masks.append(mask)
    return tuple(masks)


def _open_neighborhood_masks(G: nx.Graph) -> Tuple[int, ...]:
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    masks = []
    for v in nodes:
        mask = 0
        for u in G.neighbors(v):
            mask |= 1 << idx[u]
        masks.append(mask)
    return tuple(masks)


def _exact_set_cover_size(masks: Tuple[int, ...], n: int) -> int:
    if n <= 0:
        return 0
    full = (1 << n) - 1
    uniq_masks = tuple(sorted(set(masks), key=lambda x: x.bit_count(), reverse=True))
    if not uniq_masks:
        return n

    union_all = 0
    for m in uniq_masks:
        union_all |= m
    if union_all != full:
        return n

    by_vertex = [[] for _ in range(n)]
    for i, mask in enumerate(uniq_masks):
        bits = mask
        while bits:
            b = bits & -bits
            bit = b.bit_length() - 1
            by_vertex[bit].append(i)
            bits ^= b

    @lru_cache(maxsize=None)
    def dp(covered: int) -> int:
        if covered == full:
            return 0
        uncovered = full ^ covered
        pivot = (uncovered & -uncovered).bit_length() - 1
        best = n
        for i in by_vertex[pivot]:
            new_covered = covered | uniq_masks[i]
            if new_covered == covered:
                continue
            cand = 1 + dp(new_covered)
            if cand < best:
                best = cand
                if best == 1:
                    break
        return best

    return int(dp(0))


def _exact_independent_domination_number(G: nx.Graph) -> int:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0
    if n == 1:
        return 1
    idx = {v: i for i, v in enumerate(nodes)}
    closed = []
    adj = []
    for v in nodes:
        c = 1 << idx[v]
        a = 0
        for u in G.neighbors(v):
            c |= 1 << idx[u]
            a |= 1 << idx[u]
        closed.append(c)
        adj.append(a)

    full = (1 << n) - 1
    best = [n + 1]

    def dfs(dominated: int, forbidden: int, chosen_count: int) -> None:
        if chosen_count >= best[0]:
            return
        if dominated == full:
            best[0] = min(best[0], chosen_count)
            return

        uncovered = full ^ dominated
        pivot = (uncovered & -uncovered).bit_length() - 1
        candidate_bits = closed[pivot] & (~forbidden)
        if candidate_bits == 0:
            return
        while candidate_bits:
            b = candidate_bits & -candidate_bits
            v = b.bit_length() - 1
            new_forbidden = forbidden | (1 << v) | adj[v]
            new_dominated = dominated | closed[v]
            dfs(new_dominated, new_forbidden, chosen_count + 1)
            candidate_bits ^= b

    dfs(0, 0, 0)
    return int(best[0] if best[0] <= n else n)
