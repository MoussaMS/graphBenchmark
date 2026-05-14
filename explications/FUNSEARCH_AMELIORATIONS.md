# FunSearch Optimisé - Améliorations Implémentées

## Résumé des Problèmes & Solutions

### 1. ❌ Fitness mal équilibrée → ✅ Fitness normalisée

**Problème:**
```python
# Avant: ignores les conjectures non trouvées
fitness = 1000 * found - total_cost
```

**Solution:**
```python
# Après: pénalité pour non-réussite + normalisation
avg_cost = total_cost / num_conj
fitness = (
    1000.0 * (found / num_conj)    # Pourcentage: 0-1000
    - avg_cost / 12.0              # Coût normalisé: ~0-100
)
```

**Impact:** Évalue vraiment la qualité et la rapidité globales.

---

### 2. ❌ Paramètres trop courts → ✅ Paramètres adaptatifs

**Problème:**
```python
# Avant: toujours 5s par conjecture, pop_size=8
time_limit=5.0
pop_size=8
```

**Solution:**
```python
# Après: paramètres configurables et optimisés
time_per_conjecture=10.0    # +100% de temps
pop_size_search=12          # +50% de population
```

**Impact:** Plus de temps pour explorer, meilleure chance de trouver des contre-exemples.

---

### 3. ❌ Élitisme agressif → ✅ Élitisme modéré

**Problème:**
```python
# Avant: seulement 4 survivants sur 12 (33%)
survivors = population[:4]
```

**Solution:**
```python
# Après: 6 survivants sur 16 (37.5%) - meilleur équilibre
num_elite = max(6, population_size // 3)
survivors = population[:num_elite]
```

**Impact:** Meilleure rétention de diversité.

---

### 4. ❌ Sélection aléatoire → ✅ Sélection par tournoi

**Problème:**
```python
# Avant: choix aléatoire des parents
parent = random.choice(survivors)
```

**Solution:**
```python
# Après: tournoi 2-way (meilleur vs aléatoire)
candidate1 = random.choice(survivors)
candidate2 = random.choice(survivors)
parent = candidate1 if candidate1.fitness > candidate2.fitness else candidate2
```

**Impact:** Favorise les meilleurs gènes, meilleure convergence.

---

### 5. ❌ Mutations fixes → ✅ Mutations adaptatives

**Problème:**
```python
# Avant: sigma constant + reset trop fréquent
mutate(sigma=0.5)  # 15% reset
```

**Solution:**
```python
# Après: sigma décroissant + reset moins fréquent
sigma_current = 0.5 * (1.0 - gen / generations)  # Rechauffement simulé
child = parent.mutate(sigma=max(0.1, sigma_current))
```

**Impact:** Exploration large au début, exploitation fine à la fin.

---

### 6. ❌ Pas de détection stagnation → ✅ Détection stagnation + injection diversité

**Problème:**
```python
# Avant: aucune vérification de convergence prématurée
```

**Solution:**
```python
# Après: si pas d'amélioration récente, injecter de nouveaux candidats
if recent_improvement < 5.0:
    print(f"Stagnation détectée")
    # Remplacer les pires individus
    num_new = population_size // 4
    for i in range(num_new):
        population[-(i+1)] = HeuristicCandidate()
```

**Impact:** Échappe aux minima locaux.

---

### 7. ❌ Aucun suivi des progrès → ✅ Métriques détaillées

**Avant:** Affichage minimaliste (une ligne par génération)

**Après:** Pour chaque génération:
```
================================================================
Generation 15/30
================================================================
  [ 1/16] fitness =   567.23
  [ 2/16] fitness =   521.45
  ...
  [16/16] fitness =   412.89

Statistiques:
  Best:        567.23
  Average:     489.45
  Worst:       412.89
  Best weights: {diam: 0.234, Delta: -0.156, ...}
```

**Impact:** Suivi clair de la progression réelle.

---

### 8. ❌ Pas d'écrêtage des poids → ✅ Clipping des poids

**Solution:**
```python
for f in FEATURES:
    child.weights[f] = max(-5.0, min(5.0, child.weights[f]))
```

**Impact:** Évite les poids extrêmes qui rendent le score inutilisable.

---

## Résultats Attendus

**Avant (Baseline Part 1):** ~79/100 contre-exemples (79%)

**Après (FunSearch Optimisé):** 
- Cible: **85-90%** sur le même ensemble
- Avec plus de temps: **90%+**

---

## Comment Utiliser

### Test rapide (20 conjectures, 15 générations):
```bash
python validate_funsearch.py --subset 20 --generations 15 --time_per_conj 10
```

### Test complet (100 conjectures, 30 générations):
```bash
python validate_funsearch.py --subset 100 --generations 30 --time_per_conj 10
```

### Utiliser le meilleur heuristique trouvé:
```bash
python run_funsearch.py  # génère le meilleur candidat
python main.py --benchmark benchmark/benchmark.xlsx  # utilise le score custom
```

---

## Architecture Générique

Le code FunSearch est **100% générique**:
- ✅ Ne dépend d'aucun identifiant de conjecture
- ✅ Adapte les poids features automatiquement
- ✅ Fonctionne sur n'importe quel ensemble de conjectures
- ✅ Scalable: 10, 100, 1000 conjectures = même algorithme
- ✅ Paramétrisable: changez `generations`, `population_size`, etc.

---

## Paramètres de Tuning

Explorez ces paramètres pour optimiser davantage:

```python
run_funsearch(
    conjectures,
    generations=30,           # +30% temps? +5% amélioration?
    population_size=20,       # +25% population? Meilleure diversité?
    time_per_conjecture=15,   # +50% temps/conj? +10% taux réussite?
    pop_size_search=16        # +33% intra-search?
)
```

---

## Prochaines Étapes

1. **Validator**: Exécuter `validate_funsearch.py` pour comparer
2. **Tuning**: Ajuster `generations`, `population_size`, `time_per_conjecture`
3. **Analyse**: Regarder les poids optimisés - quelles features sont plus importantes?
4. **Intégration**: Utiliser le meilleur candidat dans `main.py` en production

---

## Questions d'Optimisation

- Quels poids doivent être positifs vs négatifs?
- Quelles features ont le plus d'impact?
- Faut-il des features supplémentaires? (connexité, degré minimal, etc.)
- Time limit adaptive par conjecture?
