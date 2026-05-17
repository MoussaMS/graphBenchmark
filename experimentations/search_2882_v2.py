import sys; sys.path.insert(0, '.')
import networkx as nx
from src.benchmark import load_benchmark
from src.invariants import compute_invariants
from src.validation import validate_candidate
import random

conjs = load_benchmark('benchmark/benchmark.xlsx')
c2882 = next(c for c in conjs if c.id == 2882)
print(f"ID=2882: gamma_i <= z2/12 + 17/12")
print(f"Need: 12*gamma_i > z2 + 17")
print()

# Known counterexample
known = 'JLCGGC@?O?_'
G0 = nx.from_graph6_bytes(known.encode())
inv0 = compute_invariants(G0)
print(f"Known: n={G0.number_of_nodes()} z2={inv0['z2']} gamma_i={inv0['gamma_i']} viol={c2882.violation(inv0):.4f}")
print(f"  12*gamma_i={12*inv0['gamma_i']:.0f} z2+17={inv0['z2']+17:.0f}")
print()

# Try enumeration of random trees
random.seed(42)
best_viol = 0
best_g6 = None

for trial in range(5000):
    n = random.randint(8, 20)
    # Generate random Prüfer sequence to get random tree
    if n == 2:
        G = nx.path_graph(2)
    else:
        prufer = [random.randint(0, n-1) for _ in range(n-2)]
        G = nx.from_prufer_sequence(prufer)
    G = nx.convert_node_labels_to_integers(G)

    inv = compute_invariants(G)
    v = c2882.violation(inv)
    if v > best_viol:
        res = validate_candidate(G, c2882)
        if res['valid']:
            best_viol = res['violation']
            best_g6 = res['graph6']
            G2 = nx.from_graph6_bytes(best_g6.encode())
            print(f"Trial {trial}: NEW BEST n={G2.number_of_nodes()} viol={best_viol:.4f} g6={best_g6}")

print(f"\nBest: viol={best_viol:.4f} g6={best_g6}")
