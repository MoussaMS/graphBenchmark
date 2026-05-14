from __future__ import annotations

import random
from copy import deepcopy
import multiprocessing
from multiprocessing import cpu_count
from .invariants import compute_invariants


def _search_worker(args):
    # args: (conj, weights, time_limit, pop_size)
    conj, wts, tlim, psize = args

    # reconstruire la fonction de score côté worker
    def score_fn(G, invariants, conjecture):
        # Ensure we have all features available (searcher may compute a small set)
        if not all(f in invariants for f in FEATURES):
            invariants = compute_invariants(G, required_keys=FEATURES)

        violation = conjecture.violation(invariants)
        score = 10.0 * violation
        for feature, wt in wts.items():
            val = invariants.get(feature, 0.0)
            score += wt * val
        n = invariants.get("n", 0.0)
        score -= 0.03 * max(0, n - 25)
        return score

    try:
        res = search(conj, time_limit=tlim, pop_size=psize, score_fn=score_fn)
        return (res.found, res.cost)
    except Exception:
        return (False, 120.0)


from .searcher import search


FEATURES = [
    "diam",
    "Delta",
    "delta",
    "triangles",
    "omega",
    "alpha",
    "gamma",
    "mu",
    "density",
    "avg",
]


class HeuristicCandidate:

    def __init__(self, weights=None):

        if weights is None:
            weights = {
                f: random.uniform(-1.0, 1.0)
                for f in FEATURES
            }

        self.weights = weights
        self.fitness = None

    def score_function(self):

        weights = self.weights

        def heuristic_score(G, invariants, conjecture):

            # Ensure invariants include required features (searcher may pass a
            # reduced set for speed). Recompute only when necessary.
            if not all(f in invariants for f in FEATURES):
                invariants = compute_invariants(G, required_keys=FEATURES)

            violation = conjecture.violation(invariants)

            score = 10.0 * violation

            for feature, w in weights.items():
                val = invariants.get(feature, 0.0)
                score += w * val

            n = invariants.get("n", 0.0)
            score -= 0.03 * max(0, n - 25)

            return score

        return heuristic_score

    def mutate(self, sigma=0.5):

        child = deepcopy(self)

        for f in FEATURES:

            # petite mutation normale avec sigma adaptatif
            if random.random() < 0.7:
                child.weights[f] += random.gauss(0, sigma)

            # parfois reset aléatoire (moins fréquent maintenant)
            if random.random() < 0.1:
                child.weights[f] = random.uniform(-1.0, 1.0)

        # Clip les poids pour éviter les extrêmes
        for f in FEATURES:
            child.weights[f] = max(-5.0, min(5.0, child.weights[f]))

        # IMPORTANT :
        # l'enfant doit être réévalué
        child.fitness = None

        return child


# ─────────────────────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_candidate(candidate,
                       conjectures,
                       time_limit=10.0,
                       pop_size=12,
                       sample_size: int | None = None,
                       pool: multiprocessing.Pool | None = None): # type: ignore
    """
    Évalue la qualité d'un candidat.
    
    Fitness = reward pour trouver + pénalité pour temps
    - Récompense: +1000 par conjecture trouvée
    - Coût: temps réel si trouvé, 120s sinon (pénalité de non-solution)
    - Ajustement: divisé par nombre de conjectures pour normalisation
    """
    total_cost = 0.0
    found = 0
    num_conj = len(conjectures)

    # Échantillonnage optionnel pour accélérer l'évaluation
    if sample_size is not None and 0 < sample_size < num_conj:
        subset = random.sample(conjectures, sample_size)
    else:
        subset = conjectures

    # Préparer les arguments pour le pool de workers
    weights = candidate.weights

    def _worker_args():
        for conj in subset:
            yield (conj, weights, time_limit, pop_size)

    # Exécuter en parallèle par conjecture (réutiliser pool si fourni)
    args_list = list(_worker_args())
    if pool is None:
        workers = min(max(1, cpu_count()), num_conj)
        with multiprocessing.Pool(processes=workers) as _local_pool:
            results = _local_pool.map(_search_worker, args_list)
    else:
        results = pool.map(_search_worker, args_list)

    for res_found, res_cost in results:
        total_cost += res_cost
        if res_found:
            found += 1

    # Si on a évalué un sous-ensemble, normaliser les compteurs
    evaluated = len(subset)
    if evaluated != num_conj:
        # Estimer found et coût sur l'ensemble par proportion
        found = int(round(found * (num_conj / evaluated)))
        total_cost = total_cost * (num_conj / evaluated)

    # Fitness normalisée (moyenne par conjecture)
    # - Trouver toutes les 100 conjectures → ~1000
    # - Trouver 50 et rapide → ~500
    # - Trouver aucune → ~-12000
    avg_cost = total_cost / max(1, num_conj)
    
    fitness = (
        1000.0 * (found / max(1, num_conj))  # Pourcentage de réussite
        - avg_cost / 12.0                     # Coût normalisé
    )

    candidate.fitness = fitness

    return fitness


# ─────────────────────────────────────────────────────────────
# FUNSEARCH
# ─────────────────────────────────────────────────────────────

def run_funsearch(conjectures,
                  generations=30,
                  population_size=16,
                  time_per_conjecture=10.0,
                  pop_size_search=12,
                  sample_size: int | None = None,
                  full_eval_interval: int = 5,
                  top_k_full_eval: int = 3,
                  adaptive_sample: bool = True):
    """
    Évolution génétique de fonctions de score heuristiques.
    
    Params:
    - generations: nombre de générations (30 par défaut)
    - population_size: taille de la population (16 par défaut)
    - time_per_conjecture: temps limite par conjecture lors de l'évaluation (10s)
    - pop_size_search: taille de la population intra-search (12 par défaut)
    
    Stratégie:
    1. Sélection par tournoi (2-way) → exploration meilleure
    2. Élitisme modéré: top-6 survivent (37.5%) → meilleur équilibre
    3. Mutations adaptatives: sigma décroît avec les générations
    4. Diversité maintenue: keepe les meilleures mais aussi des "mutants"
    """
    
    population = [
        HeuristicCandidate()
        for _ in range(population_size)
    ]

    best_overall = None
    best_fitness_history = []
    archive: list[HeuristicCandidate] = []

    # définir sample_size par défaut si non précisé (accélérer les générations)
    num_conj = len(conjectures)
    if sample_size is None:
        sample_size = max(1, num_conj // 4)

    # Créer un pool stable pour réutilisation (évite cout de création répétée)
    pool_workers = min(max(1, cpu_count()), len(conjectures))
    pool = multiprocessing.Pool(processes=pool_workers)

    for gen in range(generations):

        print(f"\n{'='*60}")
        print(f"Generation {gen + 1}/{generations}")
        print(f"{'='*60}")

        # Évaluation de tous les candidats (échantillonnage adaptatif possible)
        use_full_eval = ((gen + 1) % full_eval_interval) == 0

        # sample_size adaptatif (augmente linéairement vers num_conj)
        if adaptive_sample:
            frac = gen / max(1, generations - 1)
            eval_sample = int(min(num_conj, max(1, sample_size + (num_conj - sample_size) * frac)))
            if use_full_eval:
                eval_sample = None
        else:
            eval_sample = None if use_full_eval else sample_size

        for i, cand in enumerate(population):

            if cand.fitness is None:
                fitness = evaluate_candidate(
                    cand,
                    conjectures,
                    time_limit=time_per_conjecture,
                    pop_size=pop_size_search,
                    sample_size=eval_sample,
                    pool=pool
                )
                print(f"  [{i+1:2d}/{population_size}] fitness = {fitness:8.2f}")

        # Tri par fitness (meilleur en premier)
        population.sort(
            key=lambda c: c.fitness,
            reverse=True
        )

        best = population[0]
        
        if best_overall is None or best.fitness > best_overall.fitness:
            best_overall = deepcopy(best)

        # Statistiques
        avg_fitness = sum(c.fitness for c in population) / len(population)
        max_fitness = population[0].fitness
        min_fitness = population[-1].fitness

        print(f"\nStatistiques:")
        print(f"  Best:      {max_fitness:8.2f}")
        print(f"  Average:   {avg_fitness:8.2f}")
        print(f"  Worst:     {min_fitness:8.2f}")
        print(f"  Best weights: {best.weights}")

        best_fitness_history.append(max_fitness)

        # Élitisme modéré: garder top-6 (37.5% de la population)
        num_elite = max(6, population_size // 3)
        survivors = population[:num_elite]

        # Validation complète top-K: évaluer les meilleurs candidats sur l'ensemble
        try:
            k = min(top_k_full_eval, len(population))
            if k > 0:
                for idx in range(k):
                    cand = population[idx]
                    full_fit = evaluate_candidate(cand, conjectures, time_limit=time_per_conjecture, pop_size=pop_size_search, sample_size=None, pool=pool)
                    # Mettre à jour la fitness si la validation complète est meilleure
                    cand.fitness = full_fit
                # resort après validation
                population.sort(key=lambda c: c.fitness, reverse=True)
                best = population[0]
                if best_overall is None or best.fitness > best_overall.fitness:
                    best_overall = deepcopy(best)
            # conserver dans l'archive les candidats validés sur l'ensemble
            for c in population[:max(1, top_k_full_eval)]:
                # ajouter une copie si nouveaux poids
                exists = any(all(abs(c.weights[f] - a.weights.get(f, 0)) < 1e-9 for f in FEATURES) for a in archive)
                if not exists:
                    archive.append(deepcopy(c))
        except Exception:
            pass

        # Reproduction pour compléter la population
        children = []

        sigma_current = 0.5 * (1.0 - gen / generations)  # Refroidissement
        
        while len(children) < population_size - len(survivors):

            # Sélection par tournoi 2-way (meilleur que aléatoire)
            candidate1 = random.choice(survivors)
            candidate2 = random.choice(survivors)
            parent = candidate1 if candidate1.fitness > candidate2.fitness else candidate2

            # Mutation avec sigma adaptatif
            child = parent.mutate(sigma=max(0.1, sigma_current))

            children.append(child)

        population = survivors + children

        # Détection de convergence prématurée
        if len(best_fitness_history) > 5:
            recent_improvement = (
                best_fitness_history[-1] - best_fitness_history[-6]
            )
            if recent_improvement < 5.0:
                print(f"  ⚠️  Stagnation détectée (amélioration: {recent_improvement:.1f})")
                # Injection de diversité : remplacer les pires candidats
                num_new = population_size // 4
                for i in range(num_new):
                    population[-(i+1)] = HeuristicCandidate()

    # Fermer le pool de workers avant de retourner
    pool.close()
    pool.join()

    print(f"\n{'='*60}")
    print(f"Recherche terminée après {generations} générations")
    print(f"Meilleur heuristique trouvé:")
    if best_overall is not None:
        print(f"  Fitness: {best_overall.fitness:.2f}")
        print(f"  Weights: {best_overall.weights}")
    print(f"{'='*60}\n")

    # Retourner l'archive triée par fitness décroissante
    archive.sort(key=lambda c: c.fitness or -1.0, reverse=True)
    return archive