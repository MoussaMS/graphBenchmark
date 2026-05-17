"""
repair.py
---------
Repair graph candidates so they satisfy the conjecture graph class.
"""

from __future__ import annotations

import random

import networkx as nx


def repair(G: nx.Graph, conjecture) -> nx.Graph:
    classes = set(conjecture.graph_class)
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))

    if H.number_of_nodes() < 2:
        H.add_edge(0, 1)

    if "tree" in classes:
        H = _ensure_tree(H)
        if "claw_free" in classes:
            # Trees that are claw-free are exactly paths.
            H = nx.path_graph(max(2, H.number_of_nodes()))
        return nx.convert_node_labels_to_integers(H)

    if "bipartite" in classes:
        H = _ensure_bipartite(H)
    if "planar" in classes:
        H = _ensure_planar(H)

    if "connected" in classes or "claw_free" in classes:
        if "bipartite" in classes:
            H = _ensure_connected_bipartite(H)
        else:
            H = _ensure_connected(H)

    if "claw_free" in classes:
        H = _ensure_claw_free(H)
        if "connected" in classes and not nx.is_connected(H):
            H = _ensure_connected(H)
            H = _ensure_claw_free(H)

    if H.number_of_nodes() < 2:
        H.add_edge(0, 1)

    return nx.convert_node_labels_to_integers(H)


def _ensure_connected(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        G.add_edge(0, 1)
        return G
    if nx.is_connected(G):
        return G
    comps = [list(c) for c in nx.connected_components(G)]
    for i in range(len(comps) - 1):
        G.add_edge(random.choice(comps[i]), random.choice(comps[i + 1]))
    return G


def _ensure_connected_bipartite(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        G.add_edge(0, 1)
        return G
    if nx.is_connected(G):
        return G

    H = G
    for _ in range(8):
        if not nx.is_bipartite(H):
            H = _ensure_bipartite(H)
        color = nx.bipartite.color(H)
        comps = [list(c) for c in nx.connected_components(H)]
        if len(comps) <= 1:
            return H
        for i in range(len(comps) - 1):
            c1 = comps[i]
            c2 = comps[i + 1]
            left1 = [u for u in c1 if color.get(u, 0) == 0]
            right1 = [u for u in c1 if color.get(u, 0) == 1]
            left2 = [u for u in c2 if color.get(u, 0) == 0]
            right2 = [u for u in c2 if color.get(u, 0) == 1]
            if left1 and right2:
                H.add_edge(random.choice(left1), random.choice(right2))
            elif right1 and left2:
                H.add_edge(random.choice(right1), random.choice(left2))
            else:
                H.add_edge(random.choice(c1), random.choice(c2))
        if nx.is_connected(H) and nx.is_bipartite(H):
            return H
    return H


def _ensure_tree(G: nx.Graph) -> nx.Graph:
    H = _ensure_connected(G)
    return nx.minimum_spanning_tree(H)


def _ensure_claw_free(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() < 4:
        return G
    if max((d for _, d in G.degree()), default=0) < 3:
        return G

    H = G
    for _ in range(64):
        claw = _find_claw(H)
        if claw is None:
            break
        _, a, b, c = claw
        pairs = [(a, b), (b, c), (a, c)]
        pair = max(pairs, key=lambda uv: len(set(H.neighbors(uv[0])).intersection(H.neighbors(uv[1]))))
        H.add_edge(*pair)
    return H


def _ensure_bipartite(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() < 3:
        return G
    H = G.copy()
    max_iter = max(32, 3 * H.number_of_edges() + 1)
    for _ in range(max_iter):
        if nx.is_bipartite(H):
            return H
        try:
            colors = nx.coloring.greedy_color(H, strategy="largest_first")
        except Exception:
            break
        removed = False
        for u, v in list(H.edges()):
            if colors.get(u, 0) == colors.get(v, 0):
                H.remove_edge(u, v)
                removed = True
                break
        if not removed:
            if H.number_of_edges() == 0:
                break
            H.remove_edge(*random.choice(list(H.edges())))
    return H


def _ensure_planar(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    if H.number_of_nodes() < 3:
        return H
    max_iter = max(40, 3 * H.number_of_edges() + 1)
    for _ in range(max_iter):
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
                break
            H.remove_edge(*random.choice(list(H.edges())))
    return H


def _find_claw(G: nx.Graph):
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
