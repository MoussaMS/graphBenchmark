"""
repair.py
─────────
Répare un graphe pour qu'il respecte la classe imposée par la conjecture.
"""

from __future__ import annotations
import random
import networkx as nx


def repair(G: nx.Graph, conjecture) -> nx.Graph:
    """
    S'assure que G respecte la classe de la conjecture.
    Retourne G réparé (potentiellement modifié en place + retourné).
    """
    classes = conjecture.graph_class

    # ── connexité ────────────────────────────────────────────────────────────
    if ("connected" in classes or "tree" in classes or "claw_free" in classes):
        G = _ensure_connected(G)

    # ── arbre ────────────────────────────────────────────────────────────────
    if "tree" in classes:
        G = _ensure_tree(G)

    # ── sans griffe ──────────────────────────────────────────────────────────
    if "claw_free" in classes:
        G = _ensure_claw_free(G)

    # Garantir au moins 2 sommets
    if G.number_of_nodes() < 2:
        G.add_edge(0, 1)

    return G


# ─── réparations individuelles ────────────────────────────────────────────────

def _ensure_connected(G: nx.Graph) -> nx.Graph:
    """Relie les composantes connexes jusqu'à obtenir un graphe connexe."""
    if G.number_of_nodes() == 0:
        G.add_edge(0, 1)
        return G
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        # Relier la première composante à une autre au hasard
        c1 = random.choice(list(comps[0]))
        c2 = random.choice(list(comps[1]))
        G.add_edge(c1, c2)
    return G


def _ensure_tree(G: nx.Graph) -> nx.Graph:
    """
    Transforme G en arbre couvrant :
    1. Assurer la connexité
    2. Supprimer les arêtes supplémentaires (garder un spanning tree)
    """
    G = _ensure_connected(G)
    # Extraire le spanning tree
    T = nx.minimum_spanning_tree(G)
    return T


def _ensure_claw_free(G: nx.Graph) -> nx.Graph:
    """
    Supprime des arêtes ou ajoute des arêtes pour éliminer les griffes.
    Stratégie : pour chaque griffe trouvée, ajouter une arête entre deux
    des trois voisins indépendants (rend le triplet non indépendant).
    """
    max_iter = 50
    for _ in range(max_iter):
        claw = _find_claw(G)
        if claw is None:
            break
        center, a, b, c = claw
        # Ajouter une arête entre deux des voisins (stratégie aléatoire)
        pair = random.choice([(a, b), (b, c), (a, c)])
        G.add_edge(*pair)
    return G


def _find_claw(G: nx.Graph):
    """Retourne (center, a, b, c) si une griffe est trouvée, sinon None."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    a, b, c = neighbors[i], neighbors[j], neighbors[k]
                    if (not G.has_edge(a, b) and
                            not G.has_edge(b, c) and
                            not G.has_edge(a, c)):
                        return (v, a, b, c)
    return None
