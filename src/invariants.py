"""
invariants.py
-------------
Robust invariant computation for GraphBench.

Design goals:
- exact or reliable values for key invariants;
- bounded runtime on larger graphs;
- defensive defaults on any failure;
- LRU caching for expensive computations.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import networkx as nx


EXACT_N = 20
MAX_INVARIANT_N = 60
MAX_CONNECTIVITY_N = 36
MAX_SPECTRAL_N = 40
MAX_DISTANCE_SPECTRAL_N = 36
MAX_CACHE = 4096


KNOWN_INVARIANTS = (
    "n",
    "m",
    "delta",
    "Delta",
    "avg",
    "density",
    "diam",
    "rad",
    "proximity",
    "remoteness",
    "omega",
    "triangles",
    "gamma",
    "gamma_t",
    "gamma_i",
    "alpha",
    "tau",
    "mu",
    "kappa",
    "kappa_edge",
    "randic",
    "harmonic",
    "z1",
    "z2",
    "lambda1",
    "lambda_l2",
    "lambda_d",
)

_EXPENSIVE_KEYS = tuple(k for k in KNOWN_INVARIANTS if k not in {"n", "m", "delta", "Delta", "avg", "density"})


def compute_invariants(G: nx.Graph) -> Dict[str, float]:
    """
    Return a complete invariant dictionary for G.

    Policy:
    - basic quantities are always computed;
    - expensive quantities use a cached bundle keyed by graph6;
    - exact domination/independence for n <= 20;
    - safe fallbacks (0.0) on any failure.
    """
    inv: Dict[str, float] = {k: 0.0 for k in KNOWN_INVARIANTS}

    try:
        H = nx.Graph(G)
        H.remove_edges_from(nx.selfloop_edges(H))
        H = nx.convert_node_labels_to_integers(H)
    except Exception:
        return inv

    n = H.number_of_nodes()
    m = H.number_of_edges()
    inv["n"] = float(n)
    inv["m"] = float(m)

    if n == 0:
        return inv

    try:
        degrees = [d for _, d in H.degree()]
        inv["delta"] = float(min(degrees)) if degrees else 0.0
        inv["Delta"] = float(max(degrees)) if degrees else 0.0
        inv["avg"] = float(sum(degrees) / n)
        inv["density"] = float(nx.density(H))
    except Exception:
        pass

    # Hard cap for expensive computations.
    if n > MAX_INVARIANT_N:
        try:
            inv["gamma"] = float(_domination_number_greedy(H))
        except Exception:
            pass
        try:
            inv["alpha"] = float(_independence_number_approx(H))
            inv["tau"] = float(n) - inv["alpha"]
        except Exception:
            pass
        return inv

    try:
        key = _graph_key(H)
        inv.update(dict(_expensive_bundle_cached(key)))
    except Exception:
        # Keep base values and safely degrade per invariant.
        pass

    return inv


def _graph_key(G: nx.Graph) -> bytes:
    H = nx.convert_node_labels_to_integers(nx.Graph(G))
    H.remove_edges_from(nx.selfloop_edges(H))
    return nx.to_graph6_bytes(H, header=False)


def _graph_from_key(key: bytes) -> nx.Graph:
    return nx.from_graph6_bytes(key.strip())


@lru_cache(maxsize=MAX_CACHE)
def _expensive_bundle_cached(key: bytes) -> Tuple[Tuple[str, float], ...]:
    G = _graph_from_key(key)
    n = G.number_of_nodes()
    inv = {k: 0.0 for k in _EXPENSIVE_KEYS}

    connected = False
    try:
        connected = n > 0 and nx.is_connected(G)
    except Exception:
        connected = False

    if connected and n > 1:
        try:
            inv["diam"] = float(nx.diameter(G))
            inv["rad"] = float(nx.radius(G))
        except Exception:
            pass
        try:
            inv["proximity"], inv["remoteness"] = _distance_profile_exact(G)
        except Exception:
            pass

    try:
        inv["omega"] = float(len(max(nx.find_cliques(G), key=len, default=[])))
    except Exception:
        pass

    try:
        inv["triangles"] = float(sum(nx.triangles(G).values()) // 3)
    except Exception:
        pass

    try:
        inv["gamma"] = float(_domination_number(G))
    except Exception:
        pass

    try:
        inv["gamma_t"] = float(_total_domination_number(G))
    except Exception:
        pass

    try:
        inv["gamma_i"] = float(_independent_domination_number(G))
    except Exception:
        pass

    try:
        inv["alpha"] = float(_independence_number(G))
    except Exception:
        pass

    try:
        inv["tau"] = float(n) - inv["alpha"]
    except Exception:
        pass

    try:
        # Exact maximum-cardinality matching (Edmonds blossom under NetworkX).
        inv["mu"] = float(len(nx.max_weight_matching(G, maxcardinality=True)))
    except Exception:
        pass

    if n <= MAX_CONNECTIVITY_N:
        try:
            inv["kappa"] = float(nx.node_connectivity(G))
        except Exception:
            pass
        try:
            inv["kappa_edge"] = float(nx.edge_connectivity(G))
        except Exception:
            pass

    try:
        inv["randic"] = float(
            sum(
                1.0 / math.sqrt(G.degree(u) * G.degree(v))
                for u, v in G.edges()
                if G.degree(u) > 0 and G.degree(v) > 0
            )
        )
    except Exception:
        pass

    try:
        inv["harmonic"] = float(
            sum(
                2.0 / (G.degree(u) + G.degree(v))
                for u, v in G.edges()
                if (G.degree(u) + G.degree(v)) > 0
            )
        )
    except Exception:
        pass

    try:
        inv["z1"] = float(sum(G.degree(u) + G.degree(v) for u, v in G.edges()))
    except Exception:
        pass

    try:
        inv["z2"] = float(sum(G.degree(u) * G.degree(v) for u, v in G.edges()))
    except Exception:
        pass

    if 1 < n <= MAX_SPECTRAL_N:
        try:
            import numpy as np

            A = nx.to_numpy_array(G, dtype=float)
            evals = sorted(np.linalg.eigvalsh(A), reverse=True)
            inv["lambda1"] = float(evals[0]) if evals else 0.0
        except Exception:
            pass

        try:
            import numpy as np

            A = nx.to_numpy_array(G, dtype=float)
            L = np.diag([d for _, d in G.degree()]).astype(float) - A
            levals = sorted(np.linalg.eigvalsh(L))
            inv["lambda_l2"] = float(levals[1]) if len(levals) > 1 else 0.0
        except Exception:
            pass

    if connected and 1 < n <= MAX_DISTANCE_SPECTRAL_N:
        try:
            import numpy as np

            D = nx.floyd_warshall_numpy(G).astype(float)
            devals = sorted(np.linalg.eigvalsh(D), reverse=True)
            inv["lambda_d"] = float(devals[0]) if devals else 0.0
        except Exception:
            pass

    return tuple(inv.items())


def _closed_neighborhood_masks(G: nx.Graph) -> Tuple[int, ...]:
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    masks = []
    for v in nodes:
        mask = 1 << index[v]
        for u in G.neighbors(v):
            mask |= 1 << index[u]
        masks.append(mask)
    return tuple(masks)


def _open_neighborhood_masks(G: nx.Graph) -> Tuple[int, ...]:
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    masks = []
    for v in nodes:
        mask = 0
        for u in G.neighbors(v):
            mask |= 1 << index[u]
        masks.append(mask)
    return tuple(masks)


def _exact_cover_size(masks: Iterable[int], n: int) -> int:
    """
    Exact minimum set cover cardinality via DP on bitmasks.

    This is fast enough for n <= 20 and avoids combinatorial blowups from
    naive subset enumeration.
    """
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
    for idx, mask in enumerate(uniq_masks):
        bits = mask
        while bits:
            b = bits & -bits
            bit = b.bit_length() - 1
            by_vertex[bit].append(idx)
            bits ^= b

    @lru_cache(maxsize=None)
    def dp(covered: int) -> int:
        if covered == full:
            return 0
        uncovered = full ^ covered
        pivot_bit = (uncovered & -uncovered).bit_length() - 1
        best = n
        for idx in by_vertex[pivot_bit]:
            new_covered = covered | uniq_masks[idx]
            if new_covered == covered:
                continue
            candidate = 1 + dp(new_covered)
            if candidate < best:
                best = candidate
                if best == 1:
                    break
        return best

    return int(dp(0))


def _solve_set_cover_ilp(masks: Tuple[int, ...], n: int) -> int | None:
    """
    Optional ILP exact solver for set cover.

    Uses pulp when available; returns None when unavailable or unsolved.
    """
    try:
        import pulp
    except Exception:
        return None

    try:
        prob = pulp.LpProblem("domination_cover", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(len(masks))]
        prob += pulp.lpSum(x)
        for v in range(n):
            bit = 1 << v
            prob += pulp.lpSum(x[i] for i, mask in enumerate(masks) if (mask & bit) != 0) >= 1
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=2)
        prob.solve(solver)
        if pulp.LpStatus[prob.status] in {"Optimal", "Integer Feasible"}:
            return int(round(sum(var.value() or 0.0 for var in x)))
    except Exception:
        return None
    return None


def _domination_number(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    if n <= EXACT_N:
        return _domination_number_exact_cached(_graph_key(G))
    return _domination_number_greedy(G)


@lru_cache(maxsize=MAX_CACHE)
def _domination_number_exact_cached(key: bytes) -> int:
    G = _graph_from_key(key)
    masks = _closed_neighborhood_masks(G)
    ilp_value = _solve_set_cover_ilp(masks, G.number_of_nodes())
    if ilp_value is not None:
        return ilp_value
    return _exact_cover_size(masks, G.number_of_nodes())


def _domination_number_greedy(G: nx.Graph) -> int:
    dominated = set()
    domset = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if v not in dominated:
            domset.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(domset)


def _total_domination_number(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    if n == 0:
        return 0
    if n <= EXACT_N and all(G.degree(v) > 0 for v in G.nodes()):
        return _total_domination_number_exact_cached(_graph_key(G))
    dominated = set()
    domset = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        neighbors = set(G.neighbors(v))
        if neighbors - dominated:
            domset.add(v)
            dominated.update(neighbors)
        if len(dominated) == n:
            break
    return len(domset)


@lru_cache(maxsize=MAX_CACHE)
def _total_domination_number_exact_cached(key: bytes) -> int:
    G = _graph_from_key(key)
    return _exact_cover_size(_open_neighborhood_masks(G), G.number_of_nodes())


def _independent_domination_number(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    if n <= EXACT_N:
        return _independent_domination_number_exact_cached(_graph_key(G))

    dominated = set()
    domset = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if v not in dominated and all(u not in domset for u in G.neighbors(v)):
            domset.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(domset)


@lru_cache(maxsize=MAX_CACHE)
def _independent_domination_number_exact_cached(key: bytes) -> int:
    G = _graph_from_key(key)
    n = G.number_of_nodes()
    if n == 0:
        return 0

    # Independent dominating sets are exactly maximal independent sets.
    # Using maximal cliques of the complement gives an exact minimum size.
    C = nx.complement(G)
    best = n
    for clique in nx.find_cliques(C):
        size = len(clique)
        if size < best:
            best = size
            if best == 1:
                break
    return best


def _independence_number(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    if n <= EXACT_N:
        return _independence_number_exact_cached(_graph_key(G))
    return _independence_number_approx(G)


@lru_cache(maxsize=MAX_CACHE)
def _independence_number_exact_cached(key: bytes) -> int:
    G = _graph_from_key(key)
    C = nx.complement(G)
    return len(max(nx.find_cliques(C), key=len, default=[]))


def _independence_number_approx(G: nx.Graph) -> int:
    try:
        return len(nx.algorithms.approximation.maximum_independent_set(G))
    except Exception:
        try:
            return len(nx.maximal_independent_set(G))
        except Exception:
            return 0


def _distance_profile_exact(G: nx.Graph) -> Tuple[float, float]:
    """
    Return (proximity, remoteness) as exact min/max average distance.
    """
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
