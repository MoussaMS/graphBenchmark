import unittest
import os
from src.benchmark import load_benchmark
from src.funsearch import run_funsearch

class TestFunSearchQuick(unittest.TestCase):
    def test_run_short(self):
        bench = load_benchmark('benchmark/benchmark.xlsx')
        subset = bench[:3]
        archive = run_funsearch(subset, generations=1, population_size=6, time_per_conjecture=1.0, pop_size_search=6, sample_size=2, full_eval_interval=2, top_k_full_eval=1, adaptive_sample=False)
        self.assertIsInstance(archive, list)
        # archive may be empty if errors but function should return list

if __name__ == '__main__':
    unittest.main()
