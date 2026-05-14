"""
mutations.py
------------
Local and targeted graph mutations.  Each mutation returns a new graph.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

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

UP_BY_DENSIFY = {
    "m",
    "density",
    "avg",
    "Delta",
    "omega",
    "triangles",
    "tau",
    "mu",
    "kappa",
    "kappa_edge",
    "lambda1",
    "lambda_l2",
    "randic",
    "harmonic",
    "z1",
    "z2",
}

UP_BY_STRETCH = {"n", "diam", "rad", "remoteness", "lambda_d", "gamma", "alpha", "gamma_i", "gamma_t"}


def mutation_names(conjecture) -> List[str]:
    if "tree" in conjecture.graph_class:
        return [
            "target_x",
            "target_y",
            "add_leaf",
            "remove_leaf",
            "subdivide_edge",
            "add_path_tree",
            "edge_contraction",
        ]
    if "claw_free" in conjecture.graph_class:
        return [
            "target_x",
            "target_y",
            "line_graph_refresh",
            "clique_expansion",
            "edge_contraction",
            "graph_product",
        ]
    names = [
        "target_x",
        "target_y",
        "add_edge",
        "remove_edge",
        "add_vertex",
        "remove_vertex",
        "subdivide_edge",
        "add_leaf",
        "add_twin",
        "clique_expansion",
        "edge_contraction",
        "graph_product",
    ]
    if conjecture.x_key in CONTINUOUS_KEYS or conjecture.y_key in CONTINUOUS_KEYS:
        names.append("numeric_gradient")
    return names


def mutate(G: nx.Graph, conjecture) -> nx.Graph:
    names = mutation_names(conjecture)
    weights = [2.0 if name in ("target_x", "target_y") else 1.0 for name in names]
    return mutate_named(G, conjecture, random.choices(names, weights=weights, k=1)[0])


def mutate_named(G: nx.Graph, conjecture, name: str) -> nx.Graph:
    H = nx.Graph(G).copy()
    try:
        if "claw_free" in conjecture.graph_class:
            return _mutate_claw_free(H, conjecture, name)
        if name == "target_x":
            return targeted_mutation(H, conjecture, conjecture.x_key, _target_direction(conjecture, "x"))
        if name == "target_y":
            return targeted_mutation(H, conjecture, conjecture.y_key, _target_direction(conjecture, "y"))
        fn = _OPS.get(name)
        if fn is None:
            return mutate(H, conjecture)
        return fn(H)
    except Exception:
        return nx.Graph(G).copy()


def _target_direction(conjecture, side: str) -> int:
    if conjecture.sign == "<=":
        return -1 if side == "x" else 1
    return 1 if side == "x" else -1


def targeted_mutation(G: nx.Graph, conjecture, invariant_key: str, direction: int) -> nx.Graph:
    if direction >= 0:
        if invariant_key in UP_BY_STRETCH:
            return random.choice([add_leaf, subdivide_edge, add_path_tree])(G)
        if invariant_key == "proximity":
            return random.choice([add_edge, clique_expansion])(G)
        return random.choice([add_edge, clique_expansion, add_twin])(G)

    if invariant_key in UP_BY_STRETCH:
        return random.choice([add_edge, edge_contraction, remove_leaf])(G)
    if invariant_key == "proximity":
        return random.choice([subdivide_edge, remove_edge, add_leaf])(G)
    return random.choice([remove_edge, remove_vertex, edge_contraction])(G)


def _mutate_claw_free(G: nx.Graph, conjecture, name: str) -> nx.Graph:
    if name == "target_x":
        H = targeted_mutation(G, conjecture, conjecture.x_key, _target_direction(conjecture, "x"))
    elif name == "target_y":
        H = targeted_mutation(G, conjecture, conjecture.y_key, _target_direction(conjecture, "y"))
    elif name == "line_graph_refresh":
        H = line_graph_refresh(G)
    elif name == "clique_expansion":
        H = clique_expansion(G)
    elif name == "edge_contraction":
        H = edge_contraction(G)
    elif name == "graph_product":
        H = graph_product_k2(G)
    else:
        H = line_graph_refresh(G)
    if _has_claw(H):
        H = line_graph_refresh(H)
    return H


def add_edge(G: nx.Graph) -> nx.Graph:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    for _ in range(30):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            return G
    return G


def remove_edge(G: nx.Graph) -> nx.Graph:
    edges = list(G.edges())
    if not edges:
        return G
    G.remove_edge(*random.choice(edges))
    return G


def add_vertex(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() >= 50:
        return G
    new_v = max(G.nodes(), default=-1) + 1
    G.add_node(new_v)
    nodes = [v for v in G.nodes() if v != new_v]
    if nodes:
        k = random.randint(1, max(1, min(4, len(nodes))))
        for v in random.sample(nodes, k):
            G.add_edge(new_v, v)
    return G


def remove_vertex(G: nx.Graph) -> nx.Graph:
    nodes = list(G.nodes())
    if len(nodes) <= 2:
        return G
    G.remove_node(random.choice(nodes))
    return G


def add_leaf(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or G.number_of_nodes() >= 50:
        return G
    new_v = max(G.nodes(), default=-1) + 1
    anchor = random.choice(list(G.nodes()))
    G.add_edge(new_v, anchor)
    return G


def remove_leaf(G: nx.Graph) -> nx.Graph:
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves:
        return G
    G.remove_node(random.choice(leaves))
    return G


def subdivide_edge(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() >= 50:
        return G
    edges = list(G.edges())
    if not edges:
        return G
    u, v = random.choice(edges)
    G.remove_edge(u, v)
    w = max(G.nodes(), default=-1) + 1
    G.add_edge(u, w)
    G.add_edge(w, v)
    return G


def add_twin(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or G.number_of_nodes() >= 50:
        return G
    v = random.choice(list(G.nodes()))
    neighbors = list(G.neighbors(v))
    new_v = max(G.nodes(), default=-1) + 1
    G.add_node(new_v)
    for u in neighbors:
        G.add_edge(new_v, u)
    return G


def remove_pendant(G: nx.Graph) -> nx.Graph:
    return remove_leaf(G)


def add_path_tree(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or G.number_of_nodes() >= 48:
        return G
    anchor = random.choice(list(G.nodes()))
    length = random.randint(2, 3)
    prev = anchor
    new_v = max(G.nodes(), default=-1) + 1
    for _ in range(length):
        G.add_edge(prev, new_v)
        prev = new_v
        new_v += 1
    return G


def clique_expansion(G: nx.Graph, k: Optional[int] = None) -> nx.Graph:
    if k is None:
        k = random.randint(3, 6)
    if G.number_of_nodes() + k > 50:
        return G
    start = max(G.nodes(), default=-1) + 1
    clique = list(range(start, start + k))
    G.add_nodes_from(clique)
    for i, u in enumerate(clique):
        for v in clique[i + 1 :]:
            G.add_edge(u, v)
    if G.number_of_nodes() > k:
        anchors = random.sample(list(set(G.nodes()) - set(clique)), min(2, G.number_of_nodes() - k))
        for u in anchors:
            G.add_edge(u, random.choice(clique))
    return G


def edge_contraction(G: nx.Graph) -> nx.Graph:
    edges = list(G.edges())
    if not edges or G.number_of_nodes() <= 2:
        return G
    u, v = random.choice(edges)
    H = nx.contracted_nodes(G, u, v, self_loops=False)
    return nx.convert_node_labels_to_integers(H)


def graph_product_k2(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or G.number_of_nodes() * 2 > 50:
        return G
    H = nx.cartesian_product(G, nx.complete_graph(2))
    return nx.convert_node_labels_to_integers(H)


def numeric_gradient(G: nx.Graph) -> nx.Graph:
    return random.choice([add_edge, remove_edge, subdivide_edge, edge_contraction, clique_expansion])(G)


def line_graph_refresh(G: nx.Graph) -> nx.Graph:
    n = max(4, min(18, G.number_of_nodes() + random.randint(-2, 3)))
    p = random.choice([0.25, 0.4, 0.55, 0.7])
    base = nx.gnp_random_graph(n, p)
    if base.number_of_edges() == 0:
        base = nx.path_graph(n)
    if not nx.is_connected(base):
        nodes = list(base.nodes())
        random.shuffle(nodes)
        for i in range(len(nodes) - 1):
            base.add_edge(nodes[i], nodes[i + 1])
    L = nx.line_graph(base)
    return nx.convert_node_labels_to_integers(L)


def add_edge_claw_safe(G: nx.Graph) -> nx.Graph:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    candidates = [(u, v) for u in nodes for v in nodes if u < v and not G.has_edge(u, v)]
    random.shuffle(candidates)
    for u, v in candidates[:30]:
        G.add_edge(u, v)
        if not _has_claw(G):
            return G
        G.remove_edge(u, v)
    return G


def add_vertex_claw_safe(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() < 2 or G.number_of_nodes() >= 50:
        return G
    new_v = max(G.nodes(), default=-1) + 1
    edges = list(G.edges())
    if not edges:
        return add_leaf(G)
    u, v = random.choice(edges)
    G.add_edge(new_v, u)
    G.add_edge(new_v, v)
    if _has_claw(G):
        G.remove_node(new_v)
    return G


def _has_claw(G: nx.Graph) -> bool:
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    a, b, c = neighbors[i], neighbors[j], neighbors[k]
                    if not G.has_edge(a, b) and not G.has_edge(b, c) and not G.has_edge(a, c):
                        return True
    return False


_OPS: Dict[str, Callable[[nx.Graph], nx.Graph]] = {
    "add_edge": add_edge,
    "remove_edge": remove_edge,
    "add_vertex": add_vertex,
    "remove_vertex": remove_vertex,
    "add_leaf": add_leaf,
    "remove_leaf": remove_leaf,
    "subdivide_edge": subdivide_edge,
    "add_path_tree": add_path_tree,
    "add_twin": add_twin,
    "clique_expansion": clique_expansion,
    "edge_contraction": edge_contraction,
    "graph_product": graph_product_k2,
    "numeric_gradient": numeric_gradient,
}
