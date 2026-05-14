# GraphBench Challenge — Partie 1

Réfutation automatique de conjectures en théorie des graphes.

## Structure

```
graphbench/
├── src/
│   ├── __init__.py       # exports publics
│   ├── conjecture.py     # parsing & représentation des conjectures
│   ├── benchmark.py      # chargement du fichier Excel
│   ├── invariants.py     # calcul de tous les invariants
│   ├── generator.py      # génération de graphes initiaux
│   ├── mutations.py      # mutations locales
│   ├── repair.py         # réparation selon la classe
│   ├── score.py          # fonction de score générique
│   └── searcher.py       # boucle de recherche principale
├── benchmark/
│   └── benchmark.xlsx    # fichier de conjectures
├── results/              # résultats générés
├── main.py               # point d'entrée
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Toutes les conjectures, 60s par conjecture
python main.py --benchmark benchmark/benchmark.xlsx --time_limit 60

# Conjectures spécifiques
python main.py --benchmark benchmark/benchmark.xlsx --ids 886 980 981 --time_limit 30

# Avec affichage détaillé
python main.py -v

# Autre benchmark (le programme s'adapte automatiquement)
python main.py --benchmark mon_autre_benchmark.xlsx
```

## Architecture

### Partie 1 — Heuristique simple

1. **Représentation** : graphes NetworkX
2. **Générateur** : population diversifiée selon la classe (connexe, arbre, claw-free)
3. **Score** : violation + bonus directionnels génériques
4. **Mutations** : ajout/suppression arêtes/sommets, subdivision, feuilles, jumeaux
5. **Boucle** : sélection par tournoi + exploration aléatoire + restart si stagnation
6. **Réparation** : reconnexion, spanning tree pour arbres, élimination des griffes

### Classes supportées

- `connected` : graphes connexes
- `tree` : arbres (connexes sans cycle)
- `claw_free + connected` : graphes connexes sans griffe (via line graphs)

### Score générique

```python
score = 10 * violation
      + bonus_directionnel(X, Y, sign)
      + bonus_structure(diam, Delta, density)
      - penalite_taille(n)
```

Aucun identifiant de conjecture n'est utilisé dans le score. La fonction s'adapte
automatiquement à n'importe quelle conjecture.

### Partie 2 — FunSearch (optimisation de l'heuristique)

FunSearch est un module d'optimisation par algorithme évolutionnaire qui cherche
à améliorer la fonction de score utilisée par le moteur de recherche. Le but est
d'augmenter le nombre de contre-exemples trouvés et de réduire le temps moyen par
conjecture. Une version détaillée à inclure dans le rapport se trouve dans
`FUNSEARCH_PART2.txt`.

Points clés :
- Population de `HeuristicCandidate` (poids sur invariants)
- Évaluation par fitness (récompense pour les trouvailles, pénalité pour le temps)
- Sélection, mutation, élitisme, et validation complète périodique
- À la fin on crée un ensemble (`--ensemble_size`) des meilleures heuristiques

Exemple d'utilisation rapide (test court) :
```
python3 validate_funsearch.py --subset 7 --generations 2 --time_per_conj 4 --sample_size 7 --full_eval_interval 2 --ensemble_size 3
```

Voir `FUNSEARCH_PART2.txt` pour une explication complète prête à intégrer
dans votre rapport.
