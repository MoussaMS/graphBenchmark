import sys; sys.path.insert(0, '.')
import networkx as nx
from src.benchmark import load_benchmark
from src.invariants import compute_invariants
from src.validation import validate_candidate
import random

conjs = load_benchmark('benchmark/benchmark.xlsx')
c2882 = next(c for c in conjs if c.id == 2882)

best_viol = 0
best_g6 = None

# Generate caterpillar trees
def caterpillar(spine_len, leaves_per_node):
    """Caterpillar: spine of spine_len nodes, each with leaves_per_node extra leaves."""
    G = nx.path_graph(spine_len)
    nxt = spine_len
    for i in range(spine_len):
        for _ in range(leaves_per_node):
            G.add_edge(i, nxt)
            nxt += 1
    return G

# Try various caterpillars
for spine in range(3, 12):
    for lpn in range(1, 4):
        G = caterpillar(spine, lpn)
        inv = compute_invariants(G)
        v = c2882.violation(inv)
        if v > best_viol:
            res = validate_candidate(G, c2882)
            if res['valid']:
                best_viol = res['violation']
                best_g6 = res['graph6']
                print(f"Caterpillar(spine={spine},lpn={lpn}): n={G.number_of_nodes()} viol={best_viol:.4f} g6={best_g6}")

# Generate spider trees (central vertex + paths)
def spider(arms, arm_len):
    """Spider: central vertex with 'arms' paths of length arm_len."""
    G = nx.Graph()
    G.add_node(0)
    nxt = 1
    for _ in range(arms):
        prev = 0
        for _ in range(arm_len):
            G.add_edge(prev, nxt)
            prev = nxt
            nxt += 1
    return G

for arms in range(3, 8):
    for arm_len in range(2, 6):
        G = spider(arms, arm_len)
        inv = compute_invariants(G)
        v = c2882.violation(inv)
        if v > best_viol:
            res = validate_candidate(G, c2882)
            if res['valid']:
                best_viol = res['violation']
                best_g6 = res['graph6']
                print(f"Spider(arms={arms},len={arm_len}): n={G.number_of_nodes()} viol={best_viol:.4f} g6={best_g6}")

print(f"\nBest: viol={best_viol:.4f} g6={best_g6}")
