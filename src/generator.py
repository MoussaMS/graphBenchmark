"""
generator.py
------------
Specialized initial graph generation by conjecture class.
"""

from __future__ import annotations

import random
from typing import List

import networkx as nx

from .profiles import invariant_family


CONTINUOUS_KEYS = {
    "proximity",
    "remoteness",
    "lambda1",
    "lambda_l2",
    "lambda_d",
    "randic",
    "harmonic",
    "z1",
    "z2",
}

DISTANCE_SENSITIVE_KEYS = {"proximity", "remoteness", "lambda_d", "diam", "rad"}


def generate_initial_population(conjecture, pop_size: int = 12, seeds: List[nx.Graph] | None = None) -> List[nx.Graph]:
    graphs: List[nx.Graph] = []
    classes = conjecture.graph_class
    x_family = invariant_family(conjecture.x_key)
    y_family = invariant_family(conjecture.y_key)
    families = {x_family, y_family}

    if seeds:
        graphs.extend(nx.Graph(G).copy() for G in seeds[: max(1, pop_size // 3)])

    generators = []

    if "tree" in classes:
        sizes = [6, 10, 14, 20, 28, 36, 46]
        generators = [
            lambda: _path(random.choice(sizes)),
            lambda: _star(random.randint(4, 24)),
            lambda: _random_caterpillar(random.randint(5, 18)),
            lambda: _random_tree(random.choice(sizes)),
            lambda: _broom_graph(random.randint(10, 32), random.randint(6, 22)),
            lambda: _double_star(random.randint(4, 10), random.randint(4, 10), random.randint(4, 12)),
        ]
    elif "bipartite" in classes:
        generators = [
            lambda: _random_connected_bipartite(random.randint(6, 30), random.choice([0.15, 0.25, 0.35, 0.45])),
            lambda: _complete_bipartite(random.randint(2, 8), random.randint(2, 10)),
            lambda: _grid(random.randint(2, 6), random.randint(2, 7)),
            lambda: _path(random.randint(8, 36)),
        ]
    elif "planar" in classes:
        generators = [
            lambda: _path(random.randint(8, 36)),
            lambda: _cycle(random.randint(6, 30)),
            lambda: _grid(random.randint(2, 6), random.randint(2, 7)),
            lambda: _wheel(random.randint(5, 20)),
            lambda: _random_planar_like(random.randint(8, 30)),
        ]
    elif "claw_free" in classes:
        generators = [
            lambda: _line_graph_from_base(random.randint(5, 18), random.choice([0.25, 0.4, 0.55, 0.7])),
            lambda: _line_graph_from_base(random.randint(8, 24), random.choice([0.2, 0.35, 0.5])),
            lambda: _line_graph_cycle(random.randint(5, 20)),
            lambda: _line_graph_complete(random.randint(4, 9)),
            # barbell(3,k) and lollipop(3,k) are always claw-free: clique neighbors are pairwise adjacent
            lambda: nx.barbell_graph(3, random.randint(1, 14)),
            lambda: nx.barbell_graph(3, random.randint(4, 20)),
            lambda: nx.lollipop_graph(3, random.randint(2, 16)),
            lambda: nx.path_graph(random.randint(5, 30)),
        ]
        if conjecture.x_key in DISTANCE_SENSITIVE_KEYS or conjecture.y_key in DISTANCE_SENSITIVE_KEYS:
            generators.extend(
                [
                    lambda: _line_graph_distance_base(random.choice(["path", "broom", "lollipop", "barbell"]), random.randint(10, 30)),
                    lambda: _line_graph_distance_base(random.choice(["path", "broom", "lollipop"]), random.randint(16, 34)),
                    lambda: nx.barbell_graph(3, random.randint(2, 18)),
                    lambda: nx.barbell_graph(4, random.randint(2, 14)),
                    lambda: nx.lollipop_graph(3, random.randint(4, 20)),
                    lambda: nx.path_graph(random.randint(8, 32)),
                ]
            )
    elif "connected" in classes:
        densities = [0.1, 0.3, 0.5, 0.7]
        generators = [
            lambda: _random_connected(random.randint(5, 24), random.choice(densities)),
            lambda: _random_connected(random.randint(20, 40), random.choice([0.08, 0.15, 0.25, 0.35])),
            lambda: _cycle(random.randint(4, 24)),
            lambda: _complete_graph(random.randint(3, 12)),
            lambda: _random_tree(random.randint(4, 24)),
            lambda: _wheel(random.randint(5, 18)),
            lambda: _grid(random.randint(2, 5), random.randint(2, 6)),
            lambda: _barbell(random.randint(4, 10), random.randint(2, 12)),
            lambda: _lollipop(random.randint(4, 10), random.randint(2, 14)),
        ]
    else:
        generators = [lambda: _random_connected(random.randint(5, 24), random.uniform(0.2, 0.6))]

    if conjecture.x_key in CONTINUOUS_KEYS or conjecture.y_key in CONTINUOUS_KEYS:
        generators.extend(_regular_generators(conjecture))
    if conjecture.x_key in DISTANCE_SENSITIVE_KEYS or conjecture.y_key in DISTANCE_SENSITIVE_KEYS:
        generators.extend(
            [
                lambda: _path(random.randint(12, 36)),
                lambda: _barbell(random.randint(4, 9), random.randint(6, 16)),
                lambda: _lollipop(random.randint(4, 8), random.randint(8, 20)),
                lambda: _turnip_graph(random.randint(5, 12), random.randint(8, 30)),
                lambda: _broom_graph(random.randint(14, 34), random.randint(8, 22)),
            ]
        )

    if families == {"domination_cover", "distance"}:
        if "claw_free" in classes:
            generators.extend(
                [
                    lambda: _line_graph_cycle(random.randint(10, 28)),
                    lambda: _line_graph_from_base(random.randint(10, 22), random.choice([0.2, 0.3, 0.4])),
                ]
            )
        else:
            generators.extend(
                [
                    lambda: _path(random.randint(16, 42)),
                    lambda: _broom_graph(random.randint(10, 22), random.randint(5, 14)),
                    lambda: _double_star(random.randint(4, 10), random.randint(4, 10), random.randint(2, 8)),
                    lambda: _barbell(random.randint(3, 7), random.randint(10, 20)),
                    lambda: _lollipop(random.randint(3, 7), random.randint(10, 20)),
                    lambda: _turnip_graph(random.randint(4, 10), random.randint(10, 28)),
                ]
            )

    if "spectral" in families:
        generators.extend(
            [
                lambda: _complete_graph(random.randint(4, 12)),
                lambda: _wheel(random.randint(6, 18)),
                lambda: _connected_random_regular(random.choice([12, 16, 20, 24]), random.choice([2, 3, 4])),
            ]
        )

    while len(graphs) < pop_size:
        try:
            graphs.append(random.choice(generators)())
        except Exception:
            pass

    if not graphs:
        graphs.append(_random_connected(8, 0.4))

    return [nx.convert_node_labels_to_integers(nx.Graph(G)) for G in graphs[:pop_size]]


def _regular_generators(conjecture):
    gens = []
    for n in (8, 12, 16, 20, 24, 30):
        for d in (2, 3, 4, max(2, n // 3)):
            if 0 < d < n and (n * d) % 2 == 0:
                gens.append(lambda n=n, d=d: _connected_random_regular(n, d))
    if "claw_free" in conjecture.graph_class:
        return [lambda gen=gen: _bounded_line_graph(gen()) for gen in gens]
    if "tree" in conjecture.graph_class:
        return []
    return gens


def _bounded_line_graph(gen) -> nx.Graph:
    try:
        g = gen() if callable(gen) else gen
        L = nx.line_graph(g)
        if 0 < L.number_of_nodes() <= 50:
            return nx.convert_node_labels_to_integers(L)
    except Exception:
        pass
    return _line_graph_from_base(random.randint(6, 14), random.choice([0.2, 0.35, 0.5]))


def _random_connected(n: int, p: float) -> nx.Graph:
    G = nx.gnp_random_graph(n, p)
    if n <= 1:
        return G
    if not nx.is_connected(G):
        nodes = list(G.nodes())
        random.shuffle(nodes)
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i + 1])
    return G


def _connected_random_regular(n: int, d: int) -> nx.Graph:
    for _ in range(20):
        G = nx.random_regular_graph(d, n)
        if nx.is_connected(G):
            return G
    return _random_connected(n, min(0.8, max(0.1, d / max(1, n - 1))))


def _complete_bipartite(a: int, b: int) -> nx.Graph:
    return nx.complete_bipartite_graph(max(1, a), max(1, b))


def _random_connected_bipartite(n: int, p: float) -> nx.Graph:
    n = max(4, n)
    left = random.randint(1, n - 1)
    right = n - left
    G = nx.Graph()
    A = list(range(left))
    B = list(range(left, left + right))
    G.add_nodes_from(A + B)

    # Backbone for connectivity.
    order = A + B
    random.shuffle(order)
    for i in range(len(order) - 1):
        u, v = order[i], order[i + 1]
        if (u in A and v in A) or (u in B and v in B):
            if u in A:
                v = random.choice(B)
            else:
                v = random.choice(A)
        G.add_edge(u, v)

    for u in A:
        for v in B:
            if random.random() < p:
                G.add_edge(u, v)
    return nx.convert_node_labels_to_integers(G)


def _random_planar_like(n: int) -> nx.Graph:
    # Start from a tree and add a few edges while preserving planarity.
    G = _random_tree(max(4, n))
    nodes = list(G.nodes())
    attempts = max(20, 4 * len(nodes))
    for _ in range(attempts):
        u, v = random.sample(nodes, 2)
        if G.has_edge(u, v):
            continue
        G.add_edge(u, v)
        ok, _ = nx.check_planarity(G)
        if not ok:
            G.remove_edge(u, v)
    return nx.convert_node_labels_to_integers(G)


def _random_tree(n: int) -> nx.Graph:
    try:
        return nx.random_labeled_tree(n)
    except Exception:
        return nx.random_powerlaw_tree(n, tries=100)


def _random_caterpillar(spine: int) -> nx.Graph:
    G = nx.path_graph(spine)
    node_id = spine
    for v in range(spine):
        for _ in range(random.randint(0, 3)):
            G.add_edge(v, node_id)
            node_id += 1
    return G


def _star(k: int) -> nx.Graph:
    return nx.star_graph(k)


def _path(n: int) -> nx.Graph:
    return nx.path_graph(n)


def _cycle(n: int) -> nx.Graph:
    return nx.cycle_graph(n)


def _complete_graph(n: int) -> nx.Graph:
    return nx.complete_graph(n)


def _wheel(n: int) -> nx.Graph:
    return nx.wheel_graph(n)


def _grid(r: int, c: int) -> nx.Graph:
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(r, c))


def _line_graph_from_base(n: int, p: float) -> nx.Graph:
    for _ in range(20):
        base = _random_connected(n, p)
        L = nx.line_graph(base)
        if 0 < L.number_of_nodes() <= 50:
            return nx.convert_node_labels_to_integers(L)
        n = max(5, n - 2)
        p = max(0.15, p * 0.8)
    return nx.convert_node_labels_to_integers(nx.line_graph(nx.cycle_graph(max(5, min(n, 20)))))


def _line_graph_cycle(n: int) -> nx.Graph:
    return nx.convert_node_labels_to_integers(nx.line_graph(nx.cycle_graph(n)))


def _line_graph_complete(n: int) -> nx.Graph:
    return nx.convert_node_labels_to_integers(nx.line_graph(nx.complete_graph(n)))


def _line_graph_distance_base(style: str, n: int) -> nx.Graph:
    n = max(6, min(40, n))
    if style == "path":
        base = nx.path_graph(max(4, n))
    elif style == "broom":
        path_len = max(3, min(n - 1, int(0.6 * n)))
        leaves = max(1, n - path_len)
        base = nx.path_graph(path_len)
        nxt = path_len
        for _ in range(leaves):
            base.add_edge(0, nxt)
            nxt += 1
    elif style == "lollipop":
        clique = max(3, min(10, n // 3))
        base = nx.lollipop_graph(clique, max(1, n - clique))
    else:
        clique = max(3, min(10, n // 4))
        base = nx.barbell_graph(clique, max(0, n - 2 * clique))
    return nx.convert_node_labels_to_integers(nx.line_graph(nx.convert_node_labels_to_integers(base)))


def _barbell(c1: int, path_len: int) -> nx.Graph:
    c1 = max(3, c1)
    path_len = max(0, path_len)
    return nx.barbell_graph(c1, path_len)


def _lollipop(clique_size: int, path_len: int) -> nx.Graph:
    clique_size = max(3, clique_size)
    path_len = max(1, path_len)
    return nx.lollipop_graph(clique_size, path_len)


def _broom_graph(path_len: int, leaf_count: int) -> nx.Graph:
    path_len = max(3, path_len)
    leaf_count = max(1, leaf_count)
    G = nx.path_graph(path_len)
    hub = 0
    new_v = path_len
    for _ in range(leaf_count):
        G.add_edge(hub, new_v)
        new_v += 1
    return G


def _turnip_graph(cycle_len: int, leaves: int) -> nx.Graph:
    cycle_len = max(3, cycle_len)
    leaves = max(1, leaves)
    G = nx.cycle_graph(cycle_len)
    hub = 0
    nxt = cycle_len
    for _ in range(leaves):
        G.add_edge(hub, nxt)
        nxt += 1
    return G


def _double_star(left: int, right: int, bridge_len: int) -> nx.Graph:
    left = max(2, left)
    right = max(2, right)
    bridge_len = max(1, bridge_len)

    A = nx.star_graph(left)
    offset = A.number_of_nodes()
    B = nx.relabel_nodes(nx.star_graph(right), lambda v: v + offset)
    G = nx.compose(A, B)

    center_a = 0
    center_b = offset
    prev = center_a
    next_id = offset + right + 1
    for _ in range(bridge_len):
        G.add_edge(prev, next_id)
        prev = next_id
        next_id += 1
    G.add_edge(prev, center_b)
    return G
