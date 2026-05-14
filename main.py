#!/usr/bin/env python3
"""
main.py
───────
Point d'entrée du projet GraphBench — Partie 1.

Usage :
    python main.py --benchmark benchmark/benchmark.xlsx --time_limit 60 --output results/results.xlsx
    python main.py --benchmark benchmark/benchmark.xlsx --time_limit 60 --ids 886 980 981
"""

import argparse
import sys
import os
import time
import json
import pandas as pd
import networkx as nx

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.benchmark import load_benchmark
from src.searcher  import search


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphBench — Réfutation automatique de conjectures"
    )
    parser.add_argument(
        "--benchmark", "-b",
        default="benchmark/benchmark.xlsx",
        help="Chemin vers le fichier Excel du benchmark"
    )
    parser.add_argument(
        "--time_limit", "-t",
        type=float,
        default=60.0,
        help="Limite de temps par conjecture en secondes (défaut: 60)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/results.xlsx",
        help="Fichier Excel de sortie"
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        default=None,
        help="IDs de conjectures spécifiques à tester (défaut: toutes)"
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=12,
        help="Taille de la population initiale (défaut: 12)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Affichage détaillé"
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Active le warm start expérimental depuis results/best_graphs.json (désactivé par défaut)"
    )
    return parser.parse_args()


def run(args):
    # ── chargement du benchmark ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GraphBench Challenge — Partie 1")
    print(f"{'='*60}")
    print(f"  Benchmark : {args.benchmark}")
    print(f"  Limite    : {args.time_limit}s par conjecture")
    print(f"{'='*60}\n")

    conjectures = load_benchmark(args.benchmark)
    print(f"[INFO] {len(conjectures)} conjectures chargées.\n")

    # Filtrer par IDs si spécifié
    if args.ids:
        conjectures = [c for c in conjectures if c.id in args.ids]
        print(f"[INFO] Filtrage : {len(conjectures)} conjectures sélectionnées.\n")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # ── boucle principale ─────────────────────────────────────────────────────
    results      = []
    total_cost   = 0.0
    n_found      = 0
    global_start = time.perf_counter()

    for i, conj in enumerate(conjectures):
        print(f"[{i+1:3d}/{len(conjectures)}] Conjecture {conj.id:5d} "
              f"| {conj.y_name} {conj.sign} f({conj.x_name}) "
              f"| class={conj.subgroup}", end="", flush=True)

        result = search(conj,
                        time_limit=args.time_limit,
                        pop_size=args.pop_size,
                        warm_start=args.warm_start)

        total_cost += result.cost
        if result.found:
            n_found += 1
            status = f"FOUND  in {result.elapsed:6.2f}s"
        else:
            status = f"NOT FOUND ({result.elapsed:.1f}s)"

        print(f"  ->  {status}  | cost={result.cost:.1f}", flush=True)

        if args.verbose and result.found:
            print(f"        graph6 : {result.graph6}")
            print(f"        violation = {result.violation:.4f}")

        # Enregistrer le résultat
        row = {
            "Conjecture ID":       conj.id,
            "Conjecture":          conj.text,
            "Class":               str(conj.subgroup),
            "X":                   conj.x_name,
            "Y":                   conj.y_name,
            "Sign":                conj.sign,
            "Found":               result.found,
            "Elapsed (s)":         round(result.elapsed, 3),
            "Cost":                round(result.cost, 3),
            "Violation":           round(result.violation, 6),
            "Counter example (g6)": result.graph6 or "",
            "N_nodes":             result.invariants.get("n", ""),
            "N_edges":             result.invariants.get("m", ""),
        }
        # Ajouter tous les invariants
        for k, v in result.invariants.items():
            row[f"inv_{k}"] = round(v, 4) if isinstance(v, float) else v

        results.append(row)

    # ── résumé ────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - global_start
    print(f"\n{'='*60}")
    print(f"  Résultats finaux")
    print(f"{'='*60}")
    print(f"  Conjectures réfutées : {n_found} / {len(conjectures)}")
    print(f"  Score total          : {total_cost:.1f}")
    print(f"  Temps total          : {total_elapsed:.1f}s")
    print(f"{'='*60}\n")

    # ── export Excel ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_excel(args.output, index=False)
    print(f"[INFO] Résultats exportés -> {args.output}")

    # ── export JSON résumé ────────────────────────────────────────────────────
    summary = {
        "n_conjectures": len(conjectures),
        "n_found":       n_found,
        "total_cost":    round(total_cost, 2),
        "total_elapsed": round(total_elapsed, 2),
        "details": [
            {
                "id":       r["Conjecture ID"],
                "found":    r["Found"],
                "cost":     r["Cost"],
                "graph6":   r["Counter example (g6)"],
                "violation": r["Violation"],
            }
            for r in results
        ]
    }
    json_path = args.output.replace(".xlsx", ".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Résumé JSON      -> {json_path}\n")

    return total_cost, n_found


if __name__ == "__main__":
    args = parse_args()
    run(args)
