import sys; sys.path.insert(0, '.')
from src.benchmark import load_benchmark
from src.searcher import search
import time

conjs = load_benchmark('benchmark/benchmark.xlsx')
# Previously not-found conjectures
prev_not_found = [1880, 688, 713, 1591, 1729, 1917, 1919, 2129, 2131, 2882,
                  3144, 3157, 3191, 3193, 3368, 3384, 5421, 5968, 6011, 6013,
                  7694, 7695, 7703, 7813]
targets = [c for c in conjs if c.id in prev_not_found]

found_count = 0
for c in targets:
    t0 = time.perf_counter()
    res = search(c, time_limit=20.0)
    elapsed = time.perf_counter() - t0
    if res.found:
        found_count += 1
        print(f"FOUND  ID={c.id} {c.y_name} {c.sign} f({c.x_name}) in {res.elapsed:.2f}s viol={res.violation:.6f}")
    else:
        print(f"MISSED ID={c.id} {c.y_name} {c.sign} f({c.x_name}) ({elapsed:.1f}s)")
print(f"\nTotal found: {found_count}/{len(targets)}")
