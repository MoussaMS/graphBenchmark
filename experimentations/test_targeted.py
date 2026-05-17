import sys; sys.path.insert(0, '.')
from src.benchmark import load_benchmark
from src.searcher import search
import time

conjs = load_benchmark('benchmark/benchmark.xlsx')
target_ids = [7703, 7813, 7711, 7718, 7719, 7712, 7713]
targets = [c for c in conjs if c.id in target_ids]

for c in targets:
    t0 = time.perf_counter()
    res = search(c, time_limit=15.0)
    elapsed = time.perf_counter() - t0
    status = f"FOUND in {res.elapsed:.2f}s viol={res.violation:.6f}" if res.found else f"NOT FOUND ({elapsed:.1f}s)"
    print(f"ID={c.id} {c.y_name} {c.sign} f({c.x_name}): {status}")
