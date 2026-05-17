import sys; sys.path.insert(0, '.')
import networkx as nx
from src.benchmark import load_benchmark
from src.invariants import compute_invariants
from src.validation import validate_candidate

conjs = load_benchmark('benchmark/benchmark.xlsx')
c = next(x for x in conjs if x.id == 1587)
print(f"Conjecture {c.id}: {c.y_name} {c.sign} f({c.x_name})")
print(f"  violation = -1/3 + 2/3*gamma_t - mu")
print(f"  Need: mu < 2*gamma_t/3 - 1/3")

best = None
best_viol = -999

# Try small graphs - atlas
try:
    for G in nx.graph_atlas_g():
        if G.number_of_nodes() < 2: continue
        if not nx.is_connected(G): continue
        inv = compute_invariants(G)
        viol = c.violation(inv)
        if viol > best_viol:
            best_viol = viol
            best = (G.copy(), dict(inv))
        if viol > 1e-9:
            res = validate_candidate(G, c)
            if res['valid']:
                print(f"ATLAS HIT: n={G.number_of_nodes()} m={G.number_of_edges()} viol={res['violation']:.4f}")
                print(f"  gamma_t={inv.get('gamma_t',0):.0f} mu={inv.get('mu',0):.0f}")
                break
except Exception as e:
    print(f"Atlas error: {e}")

print(f"Best atlas violation: {best_viol:.4f}")
if best:
    g, inv = best
    print(f"  n={g.number_of_nodes()} m={g.number_of_edges()} gamma_t={inv.get('gamma_t',0):.0f} mu={inv.get('mu',0):.0f}")

# Try random connected graphs with specific structure
import random
random.seed(42)
for trial in range(2000):
    n = random.randint(6, 20)
    p = random.choice([0.15, 0.2, 0.25, 0.3])
    G = nx.gnp_random_graph(n, p)
    # Ensure connected
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps)-1):
            u = list(comps[i])[0]; v = list(comps[i+1])[0]
            G.add_edge(u, v)
    if G.number_of_nodes() < 2: continue
    inv = compute_invariants(G)
    viol = c.violation(inv)
    if viol > best_viol:
        best_viol = viol
        best = (G.copy(), dict(inv))
    if viol > 1e-9:
        res = validate_candidate(G, c)
        if res['valid']:
            print(f"RANDOM HIT trial={trial}: n={G.number_of_nodes()} viol={res['violation']:.4f}")
            print(f"  gamma_t={inv.get('gamma_t',0):.0f} mu={inv.get('mu',0):.0f}")
            import networkx as nx2
            print(f"  graph6={nx.to_graph6_bytes(nx.convert_node_labels_to_integers(G), header=False).decode().strip()}")
            break

print(f"Best random violation: {best_viol:.4f}")
if best:
    g, inv = best
    print(f"  n={g.number_of_nodes()} gamma_t={inv.get('gamma_t',0):.0f} mu={inv.get('mu',0):.0f}")
