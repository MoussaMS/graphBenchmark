#!/usr/bin/env python3
"""
validate_funsearch.py
─────────────────────
Compare FunSearch amélioré avec la baseline (heuristique simple).

Usage:
    python validate_funsearch.py --generations 30 --time_per_conj 10 --subset 20
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.benchmark import load_benchmark
from src.funsearch import run_funsearch, HeuristicCandidate
from src.searcher import search
from src.score import heuristic_score


def test_baseline_heuristic(conjectures, time_limit_per_conj=5.0):
    """
    Évalue la baseline: heuristique simple (score générique par défaut).
    """
    print("\n" + "="*70)
    print("BASELINE: Heuristique simple (score générique)")
    print("="*70)
    
    found = 0
    total_time = 0.0
    results = []
    
    for i, conj in enumerate(conjectures, 1):
        result = search(
            conj,
            time_limit=time_limit_per_conj,
            pop_size=12,
            score_fn=heuristic_score
        )
        
        total_time += result.elapsed
        
        if result.found:
            found += 1
            status = "✓ FOUND"
        else:
            status = "✗ NOT FOUND"
        
        print(f"  [{i:3d}] {status:15s} | {result.elapsed:6.2f}s | {conj.id}")
        results.append(result)
    
    success_rate = 100.0 * found / len(conjectures)
    avg_time = total_time / len(conjectures)
    
    print(f"\nRésultats baseline:")
    print(f"  Réussite: {found}/{len(conjectures)} ({success_rate:.1f}%)")
    print(f"  Temps total: {total_time:.1f}s")
    print(f"  Temps moyen par conjecture: {avg_time:.2f}s")
    
    return found, success_rate, total_time, results


def test_optimized_heuristic(best_candidate, conjectures, time_limit_per_conj=10.0):
    """
    Évalue l'heuristique optimisée par FunSearch.
    """
    print("\n" + "="*70)
    print("OPTIMISÉE: Heuristique trouvée par FunSearch")
    print("="*70)
    
    found = 0
    total_time = 0.0
    results = []
    
    fn = best_candidate.score_function()
    
    for i, conj in enumerate(conjectures, 1):
        result = search(
            conj,
            time_limit=time_limit_per_conj,
            pop_size=12,
            score_fn=fn
        )
        
        total_time += result.elapsed
        
        if result.found:
            found += 1
            status = "✓ FOUND"
        else:
            status = "✗ NOT FOUND"
        
        print(f"  [{i:3d}] {status:15s} | {result.elapsed:6.2f}s | {conj.id}")
        results.append(result)
    
    success_rate = 100.0 * found / len(conjectures)
    avg_time = total_time / len(conjectures)
    
    print(f"\nRésultats optimisés:")
    print(f"  Réussite: {found}/{len(conjectures)} ({success_rate:.1f}%)")
    print(f"  Temps total: {total_time:.1f}s")
    print(f"  Temps moyen par conjecture: {avg_time:.2f}s")
    print(f"\nPoids optimisés:")
    for feature, weight in sorted(best_candidate.weights.items()):
        print(f"    {feature:12s}: {weight:7.3f}")
    
    return found, success_rate, total_time, results


def test_optimized_ensemble(candidates, conjectures, time_limit_per_conj=10.0):
    """
    Évalue un ensemble de candidats: pour chaque conjecture, on teste séquentiellement
    chaque heuristique pendant time_limit_per_conj / len(candidates) secondes
    et on marque trouvé si une heuristique trouve un contre-exemple.
    """
    print("\n" + "="*70)
    print("OPTIMISÉE (ENSEMBLE): Combinaison des meilleurs heuristiques")
    print("="*70)

    found = 0
    total_time = 0.0
    results = []

    k = max(1, len(candidates))
    per_heur_time = max(0.5, time_limit_per_conj / k)

    for i, conj in enumerate(conjectures, 1):
        conj_found = False
        conj_time = 0.0
        for cand in candidates:
            fn = cand.score_function()
            result = search(conj, time_limit=per_heur_time, pop_size=12, score_fn=fn)
            conj_time += result.elapsed
            if result.found:
                conj_found = True
                break
        total_time += conj_time
        if conj_found:
            found += 1
            status = "✓ FOUND"
        else:
            status = "✗ NOT FOUND"
        print(f"  [{i:3d}] {status:15s} | {conj_time:6.2f}s | {conj.id}")
        results.append(None)

    success_rate = 100.0 * found / len(conjectures)
    avg_time = total_time / len(conjectures)

    print(f"\nRésultats optimisés (ensemble):")
    print(f"  Réussite: {found}/{len(conjectures)} ({success_rate:.1f}%)")
    print(f"  Temps total: {total_time:.1f}s")
    print(f"  Temps moyen par conjecture: {avg_time:.2f}s")

    print(f"\nPoids des heuristiques de l'ensemble:")
    for j, cand in enumerate(candidates, 1):
        print(f"  Heuristique {j}: {cand.weights}")

    return found, success_rate, total_time, results


def compare_results(baseline_found, optimized_found, baseline_rate, optimized_rate):
    """
    Affiche une comparaison claire.
    """
    improvement_count = optimized_found - baseline_found
    improvement_rate = optimized_rate - baseline_rate
    
    print("\n" + "="*70)
    print("COMPARAISON")
    print("="*70)
    
    print(f"  Baseline:  {baseline_found} trouvés ({baseline_rate:.1f}%)")
    print(f"  Optimisé:  {optimized_found} trouvés ({optimized_rate:.1f}%)")
    
    if improvement_count > 0:
        print(f"\n  ✓ AMÉLIORATION: +{improvement_count} contre-exemples (+{improvement_rate:.1f}%)")
    elif improvement_count == 0:
        print(f"\n  → Pas d'amélioration")
    else:
        print(f"\n  ✗ DÉGRADATION: {improvement_count} contre-exemples ({improvement_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Valide et compare FunSearch avec la baseline"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=20,
        help="Nombre de conjectures à tester (défaut: 20)"
    )
    parser.add_argument(
        "--time_per_conj",
        type=float,
        default=10.0,
        help="Temps limite par conjecture pour l'évaluation (défaut: 10s)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=15,
        help="Nombre de générations FunSearch (défaut: 15)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Taille d'un échantillon de conjectures pour l'évaluation rapide (défaut: automatique)"
    )
    parser.add_argument(
        "--full_eval_interval",
        type=int,
        default=5,
        help="Nombre de générations entre évaluations complètes (défaut: 5)"
    )
    parser.add_argument(
        "--top_k_full_eval",
        type=int,
        default=3,
        help="Top-K candidats évalués sur l'ensemble chaque génération (défaut: 3)"
    )
    parser.add_argument(
        "--adaptive_sample",
        action="store_true",
        help="Activer l'échantillonnage adaptatif (croissant sur les générations)"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=3,
        help="Taille de l'ensemble final (combiner top-K heuristiques) (défaut: 3)"
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Sauter le test de la baseline"
    )
    args = parser.parse_args()

    # Charger le benchmark
    print("Chargement du benchmark...")
    benchmark = load_benchmark("benchmark/benchmark.xlsx")
    
    # Utiliser un sous-ensemble pour tester rapidement
    test_conjectures = benchmark[:args.subset]
    print(f"✓ {len(test_conjectures)} conjectures chargées\n")

    # Test baseline
    if not args.skip_baseline:
        baseline_found, baseline_rate, _, _ = test_baseline_heuristic(
            test_conjectures,
            time_limit_per_conj=5.0
        )
    else:
        baseline_found, baseline_rate = 0, 0.0
        print("Baseline skippée")

    # FunSearch
    print("\n" + "="*70)
    print(f"FunSearch en cours... ({args.generations} générations)")
    print("="*70)
    
    start = time.perf_counter()
    archive = run_funsearch(
        test_conjectures,
        generations=args.generations,
        population_size=16,
        time_per_conjecture=args.time_per_conj,
        pop_size_search=12,
        sample_size=args.sample_size,
        full_eval_interval=args.full_eval_interval,
        top_k_full_eval=args.top_k_full_eval,
        adaptive_sample=args.adaptive_sample
    )
    elapsed = time.perf_counter() - start
    print(f"FunSearch complété en {elapsed:.1f}s\n")

    # archive est une liste triée (meilleurs candidats); construire un ensemble
    if isinstance(archive, list) and archive:
        ensemble_candidates = archive[:args.ensemble_size]
    else:
        ensemble_candidates = []

    # Test de l'heuristique optimisée ou de l'ensemble
    if len(ensemble_candidates) == 0:
        optimized_found, optimized_rate, _, _ = 0, 0.0, None, None
    elif len(ensemble_candidates) == 1:
        optimized_found, optimized_rate, _, _ = test_optimized_heuristic(
            ensemble_candidates[0],
            test_conjectures,
            time_limit_per_conj=args.time_per_conj
        )
    else:
        optimized_found, optimized_rate, _, _ = test_optimized_ensemble(
            ensemble_candidates,
            test_conjectures,
            time_limit_per_conj=args.time_per_conj
        )

    # Comparaison
    compare_results(baseline_found, optimized_found, baseline_rate, optimized_rate)

    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    print(f"Conjectures testées: {len(test_conjectures)}")
    print(f"Baseline: {baseline_found}/{len(test_conjectures)} ({baseline_rate:.1f}%)")
    print(f"Optimisé: {optimized_found}/{len(test_conjectures)} ({optimized_rate:.1f}%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
