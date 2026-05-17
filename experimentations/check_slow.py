import sys; sys.path.insert(0, '.')
import networkx as nx
from src.benchmark import load_benchmark
from src.validation import validate_candidate
from src.invariants import compute_invariants

conjs = load_benchmark('benchmark/benchmark.xlsx')
c1880 = next(c for c in conjs if c.id == 1880)
c2882 = next(c for c in conjs if c.id == 2882)

test_cases = [
    (1880, 'Ks`CC@?OA?G?'),
    (1880, 'Q{aAAA?_@?C?G?o?G?@??O?@???'),
    (2882, 'JLCGGC@?O?_'),
]
for cid, g6 in test_cases:
    G = nx.from_graph6_bytes(g6.encode())
    c = c1880 if cid == 1880 else c2882
    inv = compute_invariants(G)
    approx_viol = c.violation(inv)
    res = validate_candidate(G, c)
    print(f'ID={cid} n={G.number_of_nodes()} approx={approx_viol:.4f} valid={res["valid"]} strict_viol={res.get("violation",0):.4f}')
    if not res['valid']:
        print(f'  reason={res.get("reason")}')
    else:
        xi = res['invariants']
        if cid == 1880:
            print(f'  tau={xi.get("tau",0):.0f} gamma_i={xi.get("gamma_i",0):.0f}  bound={4*xi.get("tau",0)-3:.1f}')
        elif cid == 2882:
            print(f'  z2={xi.get("z2",0):.2f} gamma_i={xi.get("gamma_i",0):.0f}  bound={xi.get("z2",0)/12+17/12:.3f}')
