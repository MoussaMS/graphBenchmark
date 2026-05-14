from src.benchmark import load_benchmark
from src.funsearch import run_funsearch


def main():

    benchmark = load_benchmark(
        "benchmark/benchmark.xlsx"
    )

    # petit sous-ensemble pour tests rapides
    subset = benchmark[:5]

    best = run_funsearch(
        subset,
        generations=10,
        population_size=10
    )

    print("\n=== BEST HEURISTIC ===")
    print(best.weights)
    print(best.fitness)


if __name__ == "__main__":
    main()