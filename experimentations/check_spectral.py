import sys; sys.path.insert(0, '.')
import networkx as nx
import numpy as np
from src.benchmark import load_benchmark
from src.invariants import compute_invariants
from src.validation import validate_candidate, validate_graph_class

conjs = load_benchmark('benchmark/benchmark.xlsx')
# Focus on spectral + distance claw-free and remoteness/proximity conjectures
hard = [c for c in conjs if c.id in [7703, 7813, 7711, 7712, 7713, 7718, 7719, 5421, 3368, 3384, 713, 688]]

print("=== Testing candidate graphs on hard conjectures ===\n")

test_graphs = []
for n in range(4, 40):
    test_graphs.append((nx.path_graph(n), f'P{n}'))
for n in range(4, 20):
    test_graphs.append((nx.cycle_graph(n), f'C{n}'))
for n in range(3, 10):
    test_graphs.append((nx.complete_graph(n), f'K{n}'))
for n in range(4, 16):
    test_graphs.append((nx.barbell_graph(3, n), f'barbell3_{n}'))
for n in range(3, 10):
    test_graphs.append((nx.lollipop_graph(3, n), f'lollipop3_{n}'))

for c in hard:
    best_viol = -999
    best_name = ''
    for G, name in test_graphs:
        inv = compute_invariants(G)
        viol = c.violation(inv)
        if viol > best_viol:
            best_viol = viol
            best_name = name
        if viol > 1e-9:
            ok, reason = validate_graph_class(G, c.graph_class)
            if ok:
                res = validate_candidate(G, c)
                if res['valid']:
                    print(f"FOUND conjecture {c.id} ({c.y_name} {c.sign} f({c.x_name}))")
                    print(f"  Graph: {name} n={G.number_of_nodes()}")
                    print(f"  violation={res['violation']:.6f}")
                    print(f"  inv: x={c.x_key}={inv.get(c.x_key,0):.4f} y={c.y_key}={inv.get(c.y_key,0):.4f}")
                    break
    else:
        print(f"NOT FOUND {c.id} ({c.y_name} {c.sign} f({c.x_name})) | best_viol={best_viol:.6f} at {best_name}")
