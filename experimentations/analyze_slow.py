import sys; sys.path.insert(0, '.')
import networkx as nx
from src.benchmark import load_benchmark
from src.invariants import compute_invariants

conjs = load_benchmark('benchmark/benchmark.xlsx')
c1880 = next(c for c in conjs if c.id == 1880)
c2882 = next(c for c in conjs if c.id == 2882)

# Analyze the known counterexamples
graphs = [
    (1880, 'Ks`CC@?OA?G?', c1880),
    (2882, 'JLCGGC@?O?_', c2882),
]

for cid, g6, c in graphs:
    G = nx.from_graph6_bytes(g6.encode())
    G = nx.convert_node_labels_to_integers(G)
    print(f"ID={cid} n={G.number_of_nodes()} m={G.number_of_edges()}")
    print(f"  Degrees: {sorted([d for _,d in G.degree()], reverse=True)}")
    print(f"  Edges: {list(G.edges())}")
    # Is it a specific family?
    print(f"  Is tree: {nx.is_tree(G)}")
    print(f"  Is bipartite: {nx.is_bipartite(G)}")
    inv = compute_invariants(G)
    print(f"  gamma_i={inv.get('gamma_i')} tau={inv.get('tau')} z2={inv.get('z2')}")
    print()
