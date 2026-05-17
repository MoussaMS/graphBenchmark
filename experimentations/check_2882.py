import sys; sys.path.insert(0, '.')
import networkx as nx
from src.benchmark import load_benchmark
from src.invariants import compute_invariants
from src.validation import validate_candidate

conjs = load_benchmark('benchmark/benchmark.xlsx')
c2882 = next(c for c in conjs if c.id == 2882)
print(f"ID=2882: {c2882.y_name} {c2882.sign} f({c2882.x_name})")
print(f"  y_key={c2882.y_key} x_key={c2882.x_key} coeffs={c2882.coeffs} intercept={c2882.intercept}")
print(f"  violation = gamma_i - (z2/12 + 17/12) > 0")
print(f"  Need: gamma_i * 12 > z2 + 17")
print()

# Search tree graphs systematically
# For trees: z2 = sum d(u)*d(v) over edges
# For caterpillar/spider trees: can have high gamma_i and moderate z2

# Try all trees up to n=15 that are available via atlas
import networkx.algorithms.isomorphism as iso
best_viol = 0
best_g6 = None

for G in nx.graph_atlas_g():
    if G.number_of_nodes() < 3: continue
    if not nx.is_connected(G): continue
    if not nx.is_tree(G): continue
    inv = compute_invariants(G)
    v = c2882.violation(inv)
    if v > best_viol:
        best_viol = v
        best_g6 = nx.to_graph6_bytes(nx.convert_node_labels_to_integers(G), header=False).decode().strip()
        print(f"Atlas new best: n={G.number_of_nodes()} viol={v:.4f} gamma_i={inv.get('gamma_i')} z2={inv.get('z2')} g6={best_g6}")

print(f"\nBest atlas viol={best_viol:.4f} g6={best_g6}")

if best_g6:
    G = nx.from_graph6_bytes(best_g6.encode())
    res = validate_candidate(G, c2882)
    print(f"Valid={res['valid']} strict_viol={res.get('violation',0):.4f}")
