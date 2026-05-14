"""
invariants.py
-------------
Robust invariant computation for NetworkX graphs.

Expensive exact invariants are cached by graph6 encoding.  Every public
calculation is guarded: unavailable or oversized computations return 0.0
instead of breaking a search thread.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Dict, Iterable, Optional, Set, Tuple

import networkx as nx


MAX_EXPENSIVE_N = 30
MAX_OTHER_N = 50
EXACT_ILP_N = 20


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


def _need(requested: Set[str], *keys: str) -> bool:
    return any(k in requested for k in keys)


def compute_invariants(G: nx.Graph, required_keys: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """
    Return a complete invariant dictionary for G.

    Expensive invariants are bounded by graph size:
    - exact gamma/alpha up to n <= 20, greedy/approximation afterwards;
    - eigenvalues up to n <= 30;
    - most other structural invariants up to n <= 50.
    """
    requested = set(required_keys or KNOWN_INVARIANTS)
    if "tau" in requested:
        requested.add("alpha")
    inv: Dict[str, float] = {k: 0.0 for k in KNOWN_INVARIANTS}

    try:
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.convert_node_labels_to_integers(G)
    except Exception:
        return inv

    n = G.number_of_nodes()
    m = G.number_of_edges()
    inv["n"] = float(n)
    if "m" in requested:
        inv["m"] = float(m)

    if n == 0:
        return inv

    degrees = None
    if _need(requested, "delta", "Delta", "avg"):
        try:
            degrees = [d for _, d in G.degree()]
            if "delta" in requested:
                inv["delta"] = float(min(degrees)) if degrees else 0.0
            if "Delta" in requested:
                inv["Delta"] = float(max(degrees)) if degrees else 0.0
            if "avg" in requested:
                inv["avg"] = float(sum(degrees) / n) if n else 0.0
        except Exception:
            pass

    if "density" in requested:
        try:
            inv["density"] = float(nx.density(G))
        except Exception:
            pass

    if n > MAX_OTHER_N:
        return inv

    connected = False
    if _need(requested, "diam", "rad", "proximity", "remoteness", "lambda_d"):
        try:
            connected = nx.is_connected(G) if n > 0 else False
        except Exception:
            connected = False

    if connected and n > 1:
        ecc = None
        if _need(requested, "diam", "rad", "remoteness"):
            try:
                ecc = nx.eccentricity(G)
                if "diam" in requested and ecc:
                    inv["diam"] = float(max(ecc.values()))
                if "rad" in requested and ecc:
                    inv["rad"] = float(min(ecc.values()))
                if "remoteness" in requested and ecc:
                    inv["remoteness"] = float(sum(ecc.values()) / n)
            except Exception:
                pass

        if "proximity" in requested:
            try:
                closeness_vals = [nx.closeness_centrality(G, u) for u in G.nodes()]
                inv["proximity"] = float(sum(closeness_vals) / n)
            except Exception:
                pass

    if "omega" in requested:
        try:
            inv["omega"] = float(len(max(nx.find_cliques(G), key=len, default=[])))
        except Exception:
            pass

    if "triangles" in requested:
        try:
            inv["triangles"] = float(sum(nx.triangles(G).values()) // 3)
        except Exception:
            pass

    if "gamma" in requested:
        try:
            inv["gamma"] = float(_domination_number(G))
        except Exception:
            pass

    if "gamma_t" in requested:
        try:
            inv["gamma_t"] = float(_total_domination_number(G))
        except Exception:
            pass

    if "gamma_i" in requested:
        try:
            inv["gamma_i"] = float(_independent_domination_number(G))
        except Exception:
            pass

    if "alpha" in requested:
        try:
            inv["alpha"] = float(_independence_number(G))
        except Exception:
            pass

    if "tau" in requested:
        try:
            inv["tau"] = float(n) - inv["alpha"]
        except Exception:
            pass

    if "mu" in requested:
        try:
            # Edmonds' blossom implementation in NetworkX is exact for general graphs.
            inv["mu"] = float(len(nx.max_weight_matching(G, maxcardinality=True)))
        except Exception:
            pass

    if n <= MAX_EXPENSIVE_N:
        if "kappa" in requested:
            try:
                inv["kappa"] = float(nx.node_connectivity(G))
            except Exception:
                pass
        if "kappa_edge" in requested:
            try:
                inv["kappa_edge"] = float(nx.edge_connectivity(G))
            except Exception:
                pass

    if "randic" in requested:
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

    if "harmonic" in requested:
        try:
            inv["harmonic"] = float(
                sum(
                    2.0 / (G.degree(u) + G.degree(v))
                    for u, v in G.edges()
                    if G.degree(u) + G.degree(v) > 0
                )
            )
        except Exception:
            pass

    if "z1" in requested:
        try:
            inv["z1"] = float(sum(G.degree(u) + G.degree(v) for u, v in G.edges()))
        except Exception:
            pass

    if "z2" in requested:
        try:
            inv["z2"] = float(sum(G.degree(u) * G.degree(v) for u, v in G.edges()))
        except Exception:
            pass

    if 1 < n <= MAX_EXPENSIVE_N and _need(requested, "lambda1", "lambda_l2", "lambda_d"):
        try:
            import numpy as np

            A = nx.to_numpy_array(G)
        except Exception:
            A = None

        if A is not None and "lambda1" in requested:
            try:
                evals = sorted(np.linalg.eigvalsh(A), reverse=True)
                inv["lambda1"] = float(evals[0]) if evals else 0.0
            except Exception:
                pass

        if A is not None and "lambda_l2" in requested:
            try:
                deg_diag = [d for _, d in G.degree()]
                L = np.diag(deg_diag).astype(float) - A
                levals = sorted(np.linalg.eigvalsh(L))
                inv["lambda_l2"] = float(levals[1]) if len(levals) > 1 else 0.0
            except Exception:
                pass

        if "lambda_d" in requested and connected:
            try:
                import numpy as np

                D = nx.floyd_warshall_numpy(G)
                devals = sorted(np.linalg.eigvalsh(D), reverse=True)
                inv["lambda_d"] = float(devals[0]) if devals else 0.0
            except Exception:
                pass

    return inv


def _graph_key(G: nx.Graph) -> bytes:
    H = nx.convert_node_labels_to_integers(nx.Graph(G))
    H.remove_edges_from(nx.selfloop_edges(H))
    return nx.to_graph6_bytes(H, header=False)


def _graph_from_key(key: bytes) -> nx.Graph:
    return nx.from_graph6_bytes(key.strip())


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
    full = (1 << n) - 1
    masks = sorted(set(masks), key=lambda x: x.bit_count(), reverse=True)
    if full == 0:
        return 0
    for r in range(1, n + 1):
        for combo in itertools.combinations(masks, r):
            cover = 0
            for mask in combo:
                cover |= mask
                if cover == full:
                    return r
    return n


def _domination_number(G: nx.Graph) -> int:
    key = _graph_key(G)
    n = G.number_of_nodes()
    if n <= EXACT_ILP_N:
        return _domination_number_exact_cached(key)
    return _domination_number_greedy(G)


@lru_cache(maxsize=4096)
def _domination_number_exact_cached(key: bytes) -> int:
    G = _graph_from_key(key)
    return _exact_cover_size(_closed_neighborhood_masks(G), G.number_of_nodes())


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
    if G.number_of_nodes() == 0:
        return 0
    if G.number_of_nodes() <= EXACT_ILP_N and all(G.degree(v) > 0 for v in G.nodes()):
        return _total_domination_number_exact_cached(_graph_key(G))
    dominated = set()
    domset = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        neighbors = set(G.neighbors(v))
        if neighbors - dominated:
            domset.add(v)
            dominated.update(neighbors)
        if dominated == set(G.nodes()):
            break
    return len(domset)


@lru_cache(maxsize=4096)
def _total_domination_number_exact_cached(key: bytes) -> int:
    G = _graph_from_key(key)
    return _exact_cover_size(_open_neighborhood_masks(G), G.number_of_nodes())


def _independent_domination_number(G: nx.Graph) -> int:
    dominated = set()
    domset = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if v not in dominated and all(u not in domset for u in G.neighbors(v)):
            domset.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(domset)


def _independence_number(G: nx.Graph) -> int:
    key = _graph_key(G)
    if G.number_of_nodes() <= EXACT_ILP_N:
        return _independence_number_exact_cached(key)
    return _independence_number_approx(G)


@lru_cache(maxsize=4096)
def _independence_number_exact_cached(key: bytes) -> int:
    G = _graph_from_key(key)
    C = nx.complement(G)
    return len(max(nx.find_cliques(C), key=len, default=[]))


def _independence_number_approx(G: nx.Graph) -> int:
    try:
        return len(nx.algorithms.approximation.maximum_independent_set(G))
    except Exception:
        return len(nx.maximal_independent_set(G))
