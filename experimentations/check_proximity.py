import sys, json
sys.path.insert(0, '.')
import networkx as nx
from src.invariants import compute_invariants
from src.benchmark import load_benchmark

# Test proximity on known graphs
for G, name in [
    (nx.path_graph(4), 'P4'),
    (nx.path_graph(10), 'P10'),
    (nx.complete_graph(4), 'K4'),
    (nx.cycle_graph(6), 'C6'),
    (nx.star_graph(4), 'star5'),
    (nx.barbell_graph(3, 5), 'barbell'),
]:
    inv = compute_invariants(G)
    print(f'{name}: n={inv["n"]:.0f} density={inv["density"]:.3f} proximity={inv["proximity"]:.4f} remoteness={inv["remoteness"]:.4f}')

# Load benchmark and check proximity conjectures
conjs = load_benchmark('benchmark/benchmark.xlsx')
prox_conjs = [c for c in conjs if 'proximity' in (c.x_name, c.y_name)]
print(f'\nProximity conjectures: {len(prox_conjs)}')
for c in prox_conjs[:5]:
    print(f'  ID={c.id} X={c.x_name} Y={c.y_name} sign={c.sign} rhs_at_1={c.rhs(1.0):.4f}')
    # Test K4
    K4 = nx.complete_graph(4)
    inv = compute_invariants(K4)
    viol = c.violation(inv)
    print(f'    K4 violation={viol:.4f} (proximity={inv["proximity"]:.4f} density={inv["density"]:.4f})')
