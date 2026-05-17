"""
mutations.py
------------
Local and targeted graph mutations.

Each operator returns a new graph candidate and is designed to stay generic
(no conjecture-id shortcuts).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import networkx as nx

from .profiles import invariant_family


MAX_NODES = 60
MAX_TRIES = 40

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
DISTANCE_KEYS = {"diam", "rad", "proximity", "remoteness", "lambda_d"}

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

UP_BY_STRETCH = {
    "n",
    "diam",
    "rad",
    "remoteness",
    "lambda_d",
    "gamma",
    "alpha",
    "gamma_i",
    "gamma_t",
}

UP_BY_COMPRESSION = {"proximity", "lambda_l2", "kappa", "kappa_edge"}


def mutation_names(conjecture) -> List[str]:
    distance_sensitive = _is_distance_conjecture(conjecture)
    if "tree" in conjecture.graph_class:
        names = [
            "paired_target",
            "target_x",
            "target_y",
            "add_leaf",
            "remove_leaf",
            "subdivide_edge",
            "add_path_tree",
            "edge_contraction",
            "tree_rewire",
            "local_perturbation",
            "strong_perturbation",
        ]
        if distance_sensitive:
            names.extend(["tree_pathify", "broom_refresh"])
        return names
    if "claw_free" in conjecture.graph_class:
        names = [
            "paired_target",
            "target_x",
            "target_y",
            "line_graph_local",
            "line_graph_strong",
            "line_graph_refresh",
            "clique_expansion",
            "edge_contraction",
            "graph_product",
            "local_perturbation",
            "barbell_refresh",
            "lollipop_refresh",
        ]
        if distance_sensitive:
            names.extend(["line_graph_distance_archetype", "distance_expand", "distance_compact"])
        return names
    names = [
        "paired_target",
        "target_x",
        "target_y",
        "add_edge_controlled",
        "remove_edge_controlled",
        "add_vertex_controlled",
        "remove_vertex_controlled",
        "subdivide_edge",
        "add_leaf",
        "add_path",
        "add_twin",
        "clique_expansion",
        "edge_contraction",
        "graph_product",
        "local_perturbation",
        "strong_perturbation",
    ]
    if conjecture.x_key in CONTINUOUS_KEYS or conjecture.y_key in CONTINUOUS_KEYS:
        names.append("numeric_gradient")
    if distance_sensitive:
        names.extend(["barbell_refresh", "lollipop_refresh", "broom_refresh", "turnip_refresh"])
    if "connected" in conjecture.graph_class:
        names.extend(["remove_edge_connected", "remove_vertex_connected"])
    return names


def _is_distance_conjecture(conjecture) -> bool:
    return conjecture.x_key in DISTANCE_KEYS or conjecture.y_key in DISTANCE_KEYS


def mutate(G: nx.Graph, conjecture) -> nx.Graph:
    names = mutation_names(conjecture)
    weights = []
    for name in names:
        if name == "paired_target":
            weights.append(3.0)
        elif name in ("target_x", "target_y"):
            weights.append(2.5)
        elif name in ("local_perturbation", "line_graph_local"):
            weights.append(1.5)
        elif name in ("strong_perturbation", "line_graph_strong"):
            weights.append(0.7)
        else:
            weights.append(1.0)
    return mutate_named(G, conjecture, random.choices(names, weights=weights, k=1)[0])


def mutate_named(G: nx.Graph, conjecture, name: str) -> nx.Graph:
    H = nx.Graph(G).copy()
    try:
        if "claw_free" in conjecture.graph_class:
            return _mutate_claw_free(H, conjecture, name)
        if "bipartite" in conjecture.graph_class:
            return _mutate_bipartite(H, conjecture, name)
        if "planar" in conjecture.graph_class:
            return _mutate_planar(H, conjecture, name)
        if name == "paired_target":
            return paired_target_mutation(H, conjecture)
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
    if invariant_key in {"diam", "rad", "remoteness", "lambda_d"}:
        if direction >= 0:
            return random.choice([distance_expand, barbell_refresh, lollipop_refresh, broom_refresh, turnip_refresh])(G)
        return random.choice([distance_compact, clique_expansion, edge_contraction, add_edge_controlled])(G)
    if invariant_key == "proximity":
        if direction >= 0:
            return random.choice([distance_compact, add_edge_controlled, clique_expansion, edge_contraction])(G)
        return random.choice([distance_expand, barbell_refresh, lollipop_refresh, broom_refresh, remove_edge_controlled])(G)

    if direction >= 0:
        if invariant_key in UP_BY_STRETCH:
            return random.choice([add_leaf, subdivide_edge, add_path, add_path_tree])(G)
        if invariant_key in UP_BY_DENSIFY:
            return random.choice([add_edge_controlled, clique_expansion, add_twin])(G)
        if invariant_key in UP_BY_COMPRESSION:
            return random.choice([add_edge_controlled, edge_contraction, clique_expansion])(G)
        if invariant_key == "remoteness":
            return random.choice([distance_expand, add_path, subdivide_edge, broom_refresh, barbell_refresh])(G)
        return random.choice([add_edge_controlled, add_twin, local_perturbation])(G)

    if invariant_key in UP_BY_STRETCH:
        return random.choice([remove_edge_controlled, edge_contraction, remove_leaf])(G)
    if invariant_key in UP_BY_DENSIFY:
        return random.choice([remove_edge_controlled, remove_vertex_controlled, subdivide_edge])(G)
    if invariant_key in UP_BY_COMPRESSION:
        return random.choice([subdivide_edge, add_path, remove_edge_controlled])(G)
    if invariant_key == "proximity":
        return random.choice([distance_expand, subdivide_edge, remove_edge_controlled, add_path, lollipop_refresh])(G)
    if invariant_key == "remoteness":
        return random.choice([distance_compact, add_edge_controlled, edge_contraction, clique_expansion])(G)
    return random.choice([remove_edge_controlled, remove_vertex_controlled, edge_contraction])(G)


def paired_target_mutation(G: nx.Graph, conjecture) -> nx.Graph:
    """
    Apply a coupled X/Y push to move both sides of the inequality together.
    """
    x_dir = _target_direction(conjecture, "x")
    y_dir = _target_direction(conjecture, "y")
    x_family = invariant_family(conjecture.x_key)
    y_family = invariant_family(conjecture.y_key)

    if {x_family, y_family} == {"domination_cover", "distance"}:
        if y_family == "distance":
            H = targeted_mutation(G, conjecture, conjecture.y_key, y_dir)
            H = targeted_mutation(H, conjecture, conjecture.x_key, x_dir)
        else:
            H = targeted_mutation(G, conjecture, conjecture.x_key, x_dir)
            H = targeted_mutation(H, conjecture, conjecture.y_key, y_dir)
        if random.random() < 0.6:
            H = random.choice([subdivide_edge, add_path, remove_edge_controlled, local_perturbation])(H)
        return H

    if "spectral" in {x_family, y_family} and "distance" in {x_family, y_family}:
        H = targeted_mutation(G, conjecture, conjecture.x_key, x_dir)
        H = targeted_mutation(H, conjecture, conjecture.y_key, y_dir)
        if random.random() < 0.4:
            H = random.choice([distance_expand, distance_compact, graph_product_k2])(H)
        return H

    if random.random() < 0.5:
        H = targeted_mutation(G, conjecture, conjecture.x_key, x_dir)
        return targeted_mutation(H, conjecture, conjecture.y_key, y_dir)
    H = targeted_mutation(G, conjecture, conjecture.y_key, y_dir)
    return targeted_mutation(H, conjecture, conjecture.x_key, x_dir)


def _mutate_claw_free(G: nx.Graph, conjecture, name: str) -> nx.Graph:
    if name in {"line_graph_local", "target_x", "target_y", "paired_target"}:
        if name == "target_x":
            H = _line_graph_targeted(G, conjecture.x_key, _target_direction(conjecture, "x"))
        elif name == "target_y":
            H = _line_graph_targeted(G, conjecture.y_key, _target_direction(conjecture, "y"))
        elif name == "paired_target":
            H = _line_graph_targeted(G, conjecture.x_key, _target_direction(conjecture, "x"))
            H = _line_graph_targeted(H, conjecture.y_key, _target_direction(conjecture, "y"))
        else:
            H = line_graph_local(G)
        return _force_claw_free(H)
    if name == "line_graph_strong":
        return _force_claw_free(line_graph_strong(G))
    if name == "line_graph_refresh":
        return line_graph_refresh(G)
    if name == "line_graph_distance_archetype":
        return _force_claw_free(line_graph_distance_archetype(G))
    if name == "distance_expand":
        return _force_claw_free(line_graph_distance_archetype(G))
    if name == "distance_compact":
        return _force_claw_free(line_graph_local(G))
    if name == "barbell_refresh":
        # barbell(3, k) is always claw-free; vary path length around current graph size
        n = max(4, G.number_of_nodes())
        k = random.randint(max(0, n // 3 - 4), max(1, n - 6))
        return _force_claw_free(nx.barbell_graph(3, k))
    if name == "lollipop_refresh":
        n = max(4, G.number_of_nodes())
        k = random.randint(max(1, n // 3), max(2, n - 3))
        return _force_claw_free(nx.lollipop_graph(3, k))
    if name == "clique_expansion":
        return _force_claw_free(clique_expansion(G))
    if name == "edge_contraction":
        return _force_claw_free(edge_contraction(G))
    if name == "graph_product":
        return _force_claw_free(graph_product_k2(G))
    if name == "local_perturbation":
        return _force_claw_free(local_perturbation(G))
    return line_graph_local(G)


def _mutate_bipartite(G: nx.Graph, conjecture, name: str) -> nx.Graph:
    if name in {"target_x", "target_y", "paired_target"}:
        if name == "target_x":
            H = targeted_mutation(G, conjecture, conjecture.x_key, _target_direction(conjecture, "x"))
        elif name == "target_y":
            H = targeted_mutation(G, conjecture, conjecture.y_key, _target_direction(conjecture, "y"))
        else:
            H = paired_target_mutation(G, conjecture)
        return _force_bipartite(H)
    if name in {"add_edge_controlled", "add_edge"}:
        return add_edge_bipartite_safe(G)
    if name in {"remove_edge_connected", "remove_edge_controlled", "remove_edge"}:
        return _force_bipartite(remove_edge_controlled(G))
    fn = _OPS.get(name)
    if fn is None:
        return _force_bipartite(local_perturbation(G))
    return _force_bipartite(fn(G))


def _mutate_planar(G: nx.Graph, conjecture, name: str) -> nx.Graph:
    if name in {"target_x", "target_y", "paired_target"}:
        if name == "target_x":
            H = targeted_mutation(G, conjecture, conjecture.x_key, _target_direction(conjecture, "x"))
        elif name == "target_y":
            H = targeted_mutation(G, conjecture, conjecture.y_key, _target_direction(conjecture, "y"))
        else:
            H = paired_target_mutation(G, conjecture)
        return _force_planar(H)
    if name in {"add_edge_controlled", "add_edge"}:
        return add_edge_planar_safe(G)
    fn = _OPS.get(name)
    if fn is None:
        return _force_planar(local_perturbation(G))
    return _force_planar(fn(G))


def add_edge_controlled(G: nx.Graph) -> nx.Graph:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    best = None
    best_score = float("-inf")
    for _ in range(MAX_TRIES):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            tri_bonus = len(set(G.neighbors(u)).intersection(G.neighbors(v)))
            score = G.degree(u) + G.degree(v) + 2.5 * tri_bonus
            if score > best_score:
                best_score = score
                best = (u, v)
    if best is not None:
        G.add_edge(*best)
    return G


def remove_edge_controlled(G: nx.Graph) -> nx.Graph:
    edges = list(G.edges())
    if not edges:
        return G
    random.shuffle(edges)
    for u, v in edges[:MAX_TRIES]:
        H = G.copy()
        H.remove_edge(u, v)
        if H.number_of_nodes() <= 1 or H.number_of_edges() == 0 or nx.is_connected(H):
            return H
    G.remove_edge(*random.choice(edges))
    return G


def remove_edge_connected(G: nx.Graph) -> nx.Graph:
    if not nx.is_connected(G):
        return remove_edge_controlled(G)
    bridge_set = {tuple(sorted(edge)) for edge in nx.bridges(G)}
    candidates = [(u, v) for (u, v) in G.edges() if tuple(sorted((u, v))) not in bridge_set]
    # Fallback to explicit connectivity check when all edges are bridges.
    if not candidates:
        for u, v in list(G.edges()):
            H = G.copy()
            H.remove_edge(u, v)
            if H.number_of_edges() > 0 and nx.is_connected(H):
                candidates.append((u, v))
    if candidates:
        G.remove_edge(*random.choice(candidates))
    return G


def add_vertex_controlled(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() >= MAX_NODES:
        return G
    new_v = max(G.nodes(), default=-1) + 1
    G.add_node(new_v)
    nodes = [v for v in G.nodes() if v != new_v]
    if nodes:
        nodes.sort(key=lambda v: G.degree(v), reverse=True)
        k = random.randint(1, max(1, min(4, len(nodes))))
        anchors = nodes[: max(k, min(len(nodes), 8))]
        for v in random.sample(anchors, min(k, len(anchors))):
            G.add_edge(new_v, v)
    return G


def remove_vertex_controlled(G: nx.Graph) -> nx.Graph:
    nodes = list(G.nodes())
    if len(nodes) <= 2:
        return G
    nodes.sort(key=lambda v: G.degree(v))
    for v in nodes[: min(len(nodes), 12)]:
        H = G.copy()
        H.remove_node(v)
        if H.number_of_nodes() >= 2:
            return H
    G.remove_node(random.choice(nodes))
    return G


def remove_vertex_connected(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() <= 3:
        return G
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for v in nodes[: min(len(nodes), MAX_TRIES)]:
        H = G.copy()
        H.remove_node(v)
        if H.number_of_nodes() >= 2 and nx.is_connected(H):
            return H
    return G


def add_leaf(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or G.number_of_nodes() >= MAX_NODES:
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
    if G.number_of_nodes() >= MAX_NODES:
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
    if G.number_of_nodes() == 0 or G.number_of_nodes() >= MAX_NODES:
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


def add_path(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or G.number_of_nodes() >= MAX_NODES - 2:
        return G
    anchor = random.choice(list(G.nodes()))
    length = random.randint(2, 4)
    prev = anchor
    nxt = max(G.nodes(), default=-1) + 1
    for _ in range(length):
        if G.number_of_nodes() >= MAX_NODES:
            break
        G.add_edge(prev, nxt)
        prev = nxt
        nxt += 1
    return G


def add_path_tree(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or G.number_of_nodes() >= MAX_NODES - 2:
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


def tree_rewire(G: nx.Graph) -> nx.Graph:
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves:
        return G
    leaf = random.choice(leaves)
    parent = next(iter(G.neighbors(leaf)))
    candidates = [v for v in G.nodes() if v not in (leaf, parent) and not G.has_edge(leaf, v)]
    if not candidates:
        return G
    G.remove_edge(leaf, parent)
    G.add_edge(leaf, random.choice(candidates))
    return G


def clique_expansion(G: nx.Graph, k: Optional[int] = None) -> nx.Graph:
    if k is None:
        k = random.randint(3, 6)
    if G.number_of_nodes() + k > MAX_NODES:
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
    if G.number_of_nodes() == 0 or G.number_of_nodes() * 2 > MAX_NODES:
        return G
    H = nx.cartesian_product(G, nx.complete_graph(2))
    return nx.convert_node_labels_to_integers(H)


def numeric_gradient(G: nx.Graph) -> nx.Graph:
    return random.choice(
        [
            add_edge_controlled,
            remove_edge_controlled,
            subdivide_edge,
            edge_contraction,
            clique_expansion,
            add_path,
        ]
    )(G)


def local_perturbation(G: nx.Graph) -> nx.Graph:
    return random.choice(
        [
            add_edge_controlled,
            remove_edge_controlled,
            add_leaf,
            remove_leaf,
            subdivide_edge,
            edge_contraction,
        ]
    )(G)


def strong_perturbation(G: nx.Graph) -> nx.Graph:
    k = random.randint(2, 4)
    ops = [
        add_edge_controlled,
        remove_edge_controlled,
        add_vertex_controlled,
        remove_vertex_controlled,
        subdivide_edge,
        edge_contraction,
        add_path,
        clique_expansion,
        graph_product_k2,
        distance_expand,
        distance_compact,
    ]
    H = G
    for _ in range(k):
        H = random.choice(ops)(H)
    return H


def _recover_base_graph_from_line_graph(G: nx.Graph) -> nx.Graph:
    try:
        base = nx.inverse_line_graph(nx.convert_node_labels_to_integers(nx.Graph(G)))
        if base.number_of_nodes() > 1:
            return nx.convert_node_labels_to_integers(base)
    except Exception:
        pass
    n = max(5, min(16, G.number_of_nodes() // 2 + 3))
    return _random_connected_base(n, random.choice([0.25, 0.4, 0.55]))


def _random_connected_base(n: int, p: float) -> nx.Graph:
    base = nx.gnp_random_graph(n, p)
    if n <= 1:
        return base
    if not nx.is_connected(base):
        nodes = list(base.nodes())
        random.shuffle(nodes)
        for i in range(len(nodes) - 1):
            base.add_edge(nodes[i], nodes[i + 1])
    return base


def _line_graph_from_base(base: nx.Graph) -> nx.Graph:
    if base.number_of_edges() == 0:
        base = _random_connected_base(max(3, base.number_of_nodes()), 0.4)
    L = nx.line_graph(base)
    L = nx.convert_node_labels_to_integers(nx.Graph(L))
    if L.number_of_nodes() == 0 or L.number_of_nodes() > MAX_NODES:
        return line_graph_refresh(L)
    return L


def line_graph_local(G: nx.Graph) -> nx.Graph:
    base = _recover_base_graph_from_line_graph(G)
    op = random.choice([add_edge_controlled, remove_edge_controlled, subdivide_edge, edge_contraction])
    base = op(base)
    if base.number_of_nodes() > 1 and not nx.is_connected(base):
        base = _random_connected_base(base.number_of_nodes(), 0.4)
    return _line_graph_from_base(base)


def line_graph_strong(G: nx.Graph) -> nx.Graph:
    base = _recover_base_graph_from_line_graph(G)
    for _ in range(random.randint(2, 4)):
        base = random.choice(
            [
                add_edge_controlled,
                remove_edge_controlled,
                add_vertex_controlled,
                remove_vertex_controlled,
                subdivide_edge,
                edge_contraction,
                add_path,
            ]
        )(base)
    if base.number_of_nodes() > 1 and not nx.is_connected(base):
        base = _random_connected_base(max(4, base.number_of_nodes()), 0.35)
    return _line_graph_from_base(base)


def line_graph_refresh(G: nx.Graph) -> nx.Graph:
    n_hint = max(4, min(18, G.number_of_nodes() + random.randint(-2, 3)))
    for _ in range(20):
        n = max(4, min(18, n_hint + random.randint(-2, 2)))
        p = random.choice([0.2, 0.3, 0.4, 0.5, 0.6])
        base = _random_connected_base(n, p)
        L = nx.convert_node_labels_to_integers(nx.line_graph(base))
        if 1 <= L.number_of_nodes() <= MAX_NODES:
            return L
        n_hint = max(4, n_hint - 1)
    fallback = nx.line_graph(nx.cycle_graph(max(5, min(20, n_hint + 2))))
    return nx.convert_node_labels_to_integers(fallback)


def tree_pathify(G: nx.Graph) -> nx.Graph:
    n = max(3, min(MAX_NODES, G.number_of_nodes() + random.randint(-2, 3)))
    return nx.path_graph(n)


def broom_refresh(G: nx.Graph) -> nx.Graph:
    n = max(6, min(MAX_NODES, G.number_of_nodes() + random.randint(-3, 5)))
    path_len = max(3, min(n - 1, int(0.55 * n) + random.randint(-2, 2)))
    leaves = max(1, n - path_len)
    H = nx.path_graph(path_len)
    hub = 0
    nxt = path_len
    for _ in range(leaves):
        if nxt >= MAX_NODES + 5:
            break
        H.add_edge(hub, nxt)
        nxt += 1
    return nx.convert_node_labels_to_integers(H)


def lollipop_refresh(G: nx.Graph) -> nx.Graph:
    n = max(6, min(MAX_NODES, G.number_of_nodes() + random.randint(-3, 6)))
    clique = max(3, min(12, int(0.35 * n) + random.randint(-2, 2)))
    path_len = max(1, n - clique)
    H = nx.lollipop_graph(clique, path_len)
    return nx.convert_node_labels_to_integers(H)


def barbell_refresh(G: nx.Graph) -> nx.Graph:
    n = max(8, min(MAX_NODES, G.number_of_nodes() + random.randint(-3, 6)))
    clique = max(3, min(10, int(0.28 * n) + random.randint(-2, 2)))
    path_len = max(0, n - 2 * clique)
    H = nx.barbell_graph(clique, path_len)
    return nx.convert_node_labels_to_integers(H)


def turnip_refresh(G: nx.Graph) -> nx.Graph:
    n = max(7, min(MAX_NODES, G.number_of_nodes() + random.randint(-4, 5)))
    cycle_len = max(3, min(12, int(0.35 * n) + random.randint(-2, 2)))
    leaves = max(1, n - cycle_len)
    H = nx.cycle_graph(cycle_len)
    hub = 0
    nxt = cycle_len
    for _ in range(leaves):
        H.add_edge(hub, nxt)
        nxt += 1
    return nx.convert_node_labels_to_integers(H)


def line_graph_distance_archetype(G: nx.Graph) -> nx.Graph:
    n = max(6, min(28, G.number_of_nodes() // 2 + random.randint(2, 8)))
    style = random.choice(["path", "broom", "lollipop", "barbell"])
    if style == "path":
        base = nx.path_graph(max(4, n))
    elif style == "broom":
        path_len = max(3, int(0.65 * n))
        leaves = max(1, n - path_len)
        base = nx.path_graph(path_len)
        nxt = path_len
        for _ in range(leaves):
            base.add_edge(0, nxt)
            nxt += 1
    elif style == "lollipop":
        clique = max(3, min(8, n // 3))
        base = nx.lollipop_graph(clique, max(1, n - clique))
    else:
        clique = max(3, min(8, n // 4))
        base = nx.barbell_graph(clique, max(0, n - 2 * clique))
    return _line_graph_from_base(nx.convert_node_labels_to_integers(base))


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
    if G.number_of_nodes() < 2 or G.number_of_nodes() >= MAX_NODES:
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


def add_edge_bipartite_safe(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    if H.number_of_nodes() < 2:
        return H
    if not nx.is_bipartite(H):
        return _force_bipartite(H)
    color = nx.bipartite.color(H)
    left = [u for u in H.nodes() if color.get(u, 0) == 0]
    right = [u for u in H.nodes() if color.get(u, 0) == 1]
    if not left or not right:
        return H
    candidates = [(u, v) for u in left for v in right if not H.has_edge(u, v)]
    if not candidates:
        return H
    u, v = random.choice(candidates)
    H.add_edge(u, v)
    return H


def add_edge_planar_safe(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    nodes = list(H.nodes())
    if len(nodes) < 2:
        return H
    for _ in range(MAX_TRIES):
        u, v = random.sample(nodes, 2)
        if H.has_edge(u, v):
            continue
        H.add_edge(u, v)
        ok, _ = nx.check_planarity(H)
        if ok:
            return H
        H.remove_edge(u, v)
    return H


def add_edge(G: nx.Graph) -> nx.Graph:
    return add_edge_controlled(G)


def remove_edge(G: nx.Graph) -> nx.Graph:
    return remove_edge_controlled(G)


def add_vertex(G: nx.Graph) -> nx.Graph:
    return add_vertex_controlled(G)


def remove_vertex(G: nx.Graph) -> nx.Graph:
    return remove_vertex_controlled(G)


def _line_graph_targeted(G: nx.Graph, invariant_key: str, direction: int) -> nx.Graph:
    if invariant_key in DISTANCE_KEYS:
        if direction >= 0:
            return line_graph_distance_archetype(G)
        return line_graph_local(G)

    base = _recover_base_graph_from_line_graph(G)
    if direction >= 0:
        if invariant_key in UP_BY_STRETCH:
            base = random.choice([subdivide_edge, add_path])(base)
        elif invariant_key in UP_BY_DENSIFY or invariant_key in UP_BY_COMPRESSION:
            base = random.choice([add_edge_controlled, edge_contraction])(base)
        else:
            base = local_perturbation(base)
    else:
        if invariant_key in UP_BY_STRETCH:
            base = random.choice([add_edge_controlled, edge_contraction])(base)
        elif invariant_key in UP_BY_DENSIFY or invariant_key in UP_BY_COMPRESSION:
            base = random.choice([remove_edge_controlled, subdivide_edge])(base)
        else:
            base = random.choice([remove_edge_controlled, add_path])(base)
    if base.number_of_nodes() > 1 and not nx.is_connected(base):
        base = _random_connected_base(max(4, base.number_of_nodes()), 0.4)
    return _line_graph_from_base(base)


def _force_claw_free(G: nx.Graph) -> nx.Graph:
    if _has_claw(G):
        return line_graph_refresh(G)
    return G


def _force_bipartite(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    if H.number_of_nodes() < 3:
        return H
    for _ in range(max(24, 2 * H.number_of_edges() + 1)):
        if nx.is_bipartite(H):
            return H
        try:
            colors = nx.coloring.greedy_color(H, strategy="largest_first")
        except Exception:
            return H
        conflict = None
        for u, v in H.edges():
            if colors.get(u, 0) == colors.get(v, 0):
                conflict = (u, v)
                break
        if conflict is None:
            return H
        H.remove_edge(*conflict)
    return H


def _force_planar(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    for _ in range(max(32, 2 * H.number_of_edges() + 1)):
        ok, cert = nx.check_planarity(H, counterexample=True)
        if ok:
            return H
        removed = False
        try:
            cert_edges = list(cert.edges()) if cert is not None else []
            if cert_edges:
                H.remove_edge(*random.choice(cert_edges))
                removed = True
        except Exception:
            removed = False
        if not removed:
            if H.number_of_edges() == 0:
                return H
            H.remove_edge(*random.choice(list(H.edges())))
    return H


def distance_expand(G: nx.Graph) -> nx.Graph:
    n0 = max(4, G.number_of_nodes())
    n = max(4, min(MAX_NODES, n0 + random.randint(-2, 6)))
    style = random.choice(["path", "barbell", "lollipop"])
    if style == "path":
        H = nx.path_graph(n)
    elif style == "barbell":
        clique = max(3, min(10, n // 3))
        path_len = max(0, n - 2 * clique)
        H = nx.barbell_graph(clique, path_len)
    else:
        clique = max(3, min(10, n // 3))
        path_len = max(1, n - clique)
        H = nx.lollipop_graph(clique, path_len)
    return nx.convert_node_labels_to_integers(H)


def distance_compact(G: nx.Graph) -> nx.Graph:
    n0 = max(4, G.number_of_nodes())
    n = max(4, min(MAX_NODES, n0 + random.randint(-4, 2)))
    style = random.choice(["complete", "wheel", "dense"])
    if style == "complete":
        H = nx.complete_graph(n)
    elif style == "wheel":
        H = nx.wheel_graph(max(4, n))
    else:
        p = random.choice([0.45, 0.6, 0.75, 0.9])
        H = nx.gnp_random_graph(n, p)
        if n > 1 and not nx.is_connected(H):
            nodes = list(H.nodes())
            random.shuffle(nodes)
            for i in range(len(nodes) - 1):
                H.add_edge(nodes[i], nodes[i + 1])
    return nx.convert_node_labels_to_integers(H)


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
    "add_edge_controlled": add_edge_controlled,
    "remove_edge": remove_edge,
    "remove_edge_controlled": remove_edge_controlled,
    "remove_edge_connected": remove_edge_connected,
    "add_vertex": add_vertex,
    "add_vertex_controlled": add_vertex_controlled,
    "remove_vertex": remove_vertex,
    "remove_vertex_controlled": remove_vertex_controlled,
    "remove_vertex_connected": remove_vertex_connected,
    "add_leaf": add_leaf,
    "remove_leaf": remove_leaf,
    "subdivide_edge": subdivide_edge,
    "add_path": add_path,
    "add_path_tree": add_path_tree,
    "tree_rewire": tree_rewire,
    "add_twin": add_twin,
    "clique_expansion": clique_expansion,
    "edge_contraction": edge_contraction,
    "graph_product": graph_product_k2,
    "numeric_gradient": numeric_gradient,
    "local_perturbation": local_perturbation,
    "strong_perturbation": strong_perturbation,
    "line_graph_local": line_graph_local,
    "line_graph_strong": line_graph_strong,
    "line_graph_refresh": line_graph_refresh,
    "line_graph_distance_archetype": line_graph_distance_archetype,
    "distance_expand": distance_expand,
    "distance_compact": distance_compact,
    "tree_pathify": tree_pathify,
    "broom_refresh": broom_refresh,
    "lollipop_refresh": lollipop_refresh,
    "barbell_refresh": barbell_refresh,
    "turnip_refresh": turnip_refresh,
}
