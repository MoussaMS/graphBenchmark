"""
generator.py
------------
Specialized initial graph generation by conjecture class.
"""

from __future__ import annotations

import random
from typing import List

import networkx as nx


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


def generate_initial_population(conjecture, pop_size: int = 12, seeds: List[nx.Graph] | None = None) -> List[nx.Graph]:
    graphs: List[nx.Graph] = []
    classes = conjecture.graph_class

    if seeds:
        graphs.extend(nx.Graph(G).copy() for G in seeds[: max(1, pop_size // 3)])

    generators = []

    if "tree" in classes:
        sizes = [6, 10, 14, 20, 28]
        generators = [
            lambda: _path(random.choice(sizes)),
            lambda: _star(random.randint(4, 24)),
            lambda: _random_caterpillar(random.randint(5, 18)),
            lambda: _random_tree(random.choice(sizes)),
        ]
    elif "claw_free" in classes:
        generators = [
            lambda: _line_graph_from_base(random.randint(5, 18), random.choice([0.25, 0.4, 0.55, 0.7])),
            lambda: _line_graph_from_base(random.randint(8, 24), random.choice([0.2, 0.35, 0.5])),
            lambda: _line_graph_cycle(random.randint(5, 20)),
            lambda: _line_graph_complete(random.randint(4, 9)),
        ]
    elif "connected" in classes:
        densities = [0.1, 0.3, 0.5, 0.7]
        generators = [
            lambda: _random_connected(random.randint(5, 24), random.choice(densities)),
            lambda: _cycle(random.randint(4, 24)),
            lambda: _complete_graph(random.randint(3, 12)),
            lambda: _random_tree(random.randint(4, 24)),
            lambda: _wheel(random.randint(5, 18)),
            lambda: _grid(random.randint(2, 5), random.randint(2, 6)),
        ]
    else:
        generators = [lambda: _random_connected(random.randint(5, 24), random.uniform(0.2, 0.6))]

    if conjecture.x_key in CONTINUOUS_KEYS or conjecture.y_key in CONTINUOUS_KEYS:
        generators.extend(_regular_generators(conjecture))

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
        L = nx.line_graph(gen())
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
