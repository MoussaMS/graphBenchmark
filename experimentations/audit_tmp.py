import sys; sys.path.insert(0,'.')
from src.benchmark import load_benchmark
conjs = load_benchmark('benchmark/benchmark.xlsx')
print(f'{len(conjs)} conjectures')
for c in conjs:
    print(f'ID={c.id} class={c.subgroup} X={c.x_name} Y={c.y_name} sign={c.sign} degree={c.degree} intercept={c.intercept} coeffs={c.coeffs}')
