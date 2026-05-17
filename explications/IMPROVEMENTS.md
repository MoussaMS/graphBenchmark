# Improvements

## Step 1 - Result analysis

I analyzed the available result file found at:

`C:\Users\Mekkiou\Desktop\M1\OC\td projet noté OC\graphbench\results\results.xlsx`

The requested `results/results.xlsx` did not exist in this workspace before the new run.

### 21 non-refuted conjectures

| ID | Class | X | Y | Sign |
| --- | --- | --- | --- | --- |
| 1466 | connected | domination_number | independent_domination_number | <= |
| 1880 | connected | vertex_cover_number | independent_domination_number | <= |
| 2041 | connected | independent_domination_number | matching_number | >= |
| 2051 | connected | independent_domination_number | vertex_cover_number | >= |
| 688 | connected | density | proximity | >= |
| 713 | connected | density | remoteness | >= |
| 1591 | connected | total_domination_number | proximity | >= |
| 1917 | connected | vertex_cover_number | proximity | >= |
| 1919 | connected | vertex_cover_number | proximity | >= |
| 2129 | connected | matching_number | proximity | >= |
| 2131 | connected | matching_number | proximity | >= |
| 3144 | connected | proximity | total_domination_number | >= |
| 3191 | connected | proximity | vertex_cover_number | >= |
| 3368 | connected | remoteness | matching_number | >= |
| 3384 | connected | remoteness | vertex_cover_number | >= |
| 3550 | connected | largest_eigenvalue | triangle_number | <= |
| 5968 | claw_free, connected | diameter | proximity | >= |
| 6011 | claw_free, connected | radius | remoteness | >= |
| 6013 | claw_free, connected | radius | remoteness | >= |
| 7694 | claw_free, connected | largest_distance_eigenvalue | proximity | >= |
| 7695 | claw_free, connected | largest_distance_eigenvalue | proximity | >= |

### Slowly refuted conjectures

Only conjecture `1178` was found with `elapsed > 30s` in the analyzed result file.

### Common hard patterns

The hard conjectures are dominated by continuous or distance-sensitive invariants:
`proximity`, `remoteness`, `largest_distance_eigenvalue`, and `largest_eigenvalue`.
The discrete hard cases mostly combine domination, independent domination, matching,
and vertex cover.  This motivated targeted X/Y mutations, regular-graph starts,
line-graph-only starts for claw-free cases, and a numeric-gradient search for
continuous invariants.

## Step 2 - Exact and cached invariants

`src/invariants.py` was rewritten around robust guarded computations.

- `gamma` is exact for `n <= 20` through an exact set-cover search over closed neighborhoods, then greedy.
- `alpha` is exact for `n <= 20` through maximum clique in the complement, then approximation.
- `mu` still uses `networkx.max_weight_matching(maxcardinality=True)`, which is exact for general graphs.
- `kappa` and `kappa_edge` were added through `networkx.node_connectivity` and `networkx.edge_connectivity`.
- Expensive exact computations are cached with `functools.lru_cache`.
- Eigenvalues are limited to `n <= 30`; other non-basic computations are limited to `n <= 50`.
- All invariant blocks are wrapped in `try/except` with `0.0` fallback.

## Step 3 - Mutations

`src/mutations.py` now exposes both `mutate()` and `mutate_named()`.

Added mutations:

- targeted X mutation
- targeted Y mutation
- clique expansion
- edge contraction
- Cartesian product with `K2`
- numeric-gradient perturbation operator

For `claw_free` conjectures, generation starts from line graphs and mutations fall
back to line-graph refresh when a claw appears.

## Step 4 - Adaptive score

`src/score.py` now uses the requested adaptive score:

- counterexamples get `1000 + violation`;
- X/Y are normalized by `n`;
- the directional term follows the conjecture sign;
- structure bonus uses diameter and maximum degree;
- large graphs receive a size penalty.

## Step 5 - ALNS

`src/searcher.py` now includes `alns()`.

It keeps one weight per mutation, samples proportionally, rewards improving
mutations by `+0.1`, penalizes degrading mutations by `-0.05` with minimum `0.1`,
normalizes every 50 iterations, and checks `_stop_event` at the top of loops.

## Step 6 - Specialized class starts

`src/generator.py` now specializes the initial population:

- connected: Erdos-Renyi graphs at densities `0.1`, `0.3`, `0.5`, `0.7`;
- tree: paths, stars, caterpillars, and random labeled trees;
- claw-free: only line graphs.

## Step 7 - Continuous invariant strategy

For conjectures involving `proximity`, `remoteness`, `lambda1`, `lambda_l2`,
`lambda_d`, `randic`, `harmonic`, `z1`, or `z2`:

- initial generation includes connected random regular graphs;
- claw-free continuous starts use line graphs of regular graphs when bounded;
- `numeric_gradient_search()` probes several small perturbations and keeps the
  one with the best score change.

## Step 8 - Time strategy

For the official project rules, `main.py` now defaults to `--time_limit 60`.
If no counterexample is found within that time, the reported cost remains `120`.

`auto_select()` runs all heuristics for the first 30 seconds.  If no
counterexample is found, it relaunches only the two best heuristics for the
remaining time.

## Step 9 - Warm start

Warm start is now disabled by default to keep official runs fully compliant with
the rule forbidding pre-recorded solutions by conjecture ID.

For experiments only, it can be enabled explicitly with:

```powershell
py main.py --benchmark benchmark/benchmark.xlsx --time_limit 60 --warm_start
```

When enabled, `src/searcher.py` saves generic warm seeds in
`results/warm_seed_pool.json` and loads them as seeds on later runs.  Official
runs should omit `--warm_start`.

## Verification

Command used for official-style smoke run:

```powershell
py main.py --time_limit 30 -v
```

## Compliance Hardening (May 2026)

This pass focused on strict rule compliance and removal of any borderline logic.

### 1) Removed selective conjecture filtering in official runner

- `main.py` no longer exposes `--ids`.
- All benchmark rows are processed by default in standard runs.

### 2) Replaced warm start keyed by conjecture ID

- Warm start storage is now `results/warm_seed_pool.json`.
- Seeds are grouped by generic profile buckets (class + invariant families + sign), not by conjecture ID.
- No `if conjecture.id == ...` behavior is used for returning a pre-registered answer.
- Warm seeds are decoded and rechecked against the conjecture class before use.

### 3) Added strict final counterexample validation before export

- New module: `src/validation.py`.
- For each found result, before writing outputs:
  - class constraints are checked (`connected`, `tree`, `bipartite`, `planar`, `claw_free`);
  - graph6 encode/decode round-trip is performed;
  - required invariants (X/Y) are recomputed in strict mode on the decoded graph;
  - strict violation (`violation > 1e-9`) is enforced.
- Invalid candidates are downgraded to `NOT FOUND` (cost `120`).

Strict mode is conservative:
- if a required NP-hard invariant cannot be guaranteed exact beyond a safe size limit, the candidate is rejected.

### 4) Improved invariant reliability

- `proximity` and `remoteness` now use exact distance-profile definitions (min/max average distance), not a closeness/eccentricity proxy.
- `independent_domination_number` is exact for small graphs (`n <= 20`) via cached maximal-independent-set characterization.

### 5) Extended repair support for additional graph classes

- `src/repair.py` now includes repair paths for:
  - `bipartite`,
  - `planar`,
  - `claw_free`,
  - combinations with `connected` and `tree`.
- Repairs preserve simple-graph constraints (no self-loop) and relabel nodes consistently.

### 6) Benchmark loading is now strict

- `src/benchmark.py` no longer silently ignores malformed rows.
- Any malformed conjecture row raises an explicit error.

## Optimization Pass (May 2026 - Recovery)

This pass targets the regression where many false positives were invalidated and
the final score/time dropped.

### A) Positive-hit validation moved into search phase

- `src/searcher.py` now validates any candidate with `violation > 0` before
  declaring `FOUND`.
- Validation uses `validate_counterexample(...)` and only accepted candidates
  stop the heuristic.
- Result: fewer "FOUND then invalidated" loops and better use of search budget.

### B) Session warm-start (generic, no IDs)

- Added in-memory warm buckets per generic profile key
  (class + family + sign), independent of conjecture ID.
- Found counterexamples from earlier conjectures become seeds for later
  conjectures of similar type in the same run.
- This is generic and still revalidated before acceptance.

### C) Recovery pass for hard unresolved conjectures

- If fast+hard pass still fails and there is enough time left, a short recovery
  portfolio is launched with restart-heavy heuristics.
- This improves hard-case coverage without rewriting the full solver.

### D) Retry overhead capped in runner

- In `main.py`, if a found candidate is invalid, the second attempt uses a
  capped budget instead of consuming almost all remaining time.
- This keeps total runtime under better control.

## Optimization Pass (May 2026 - Distance/Claw-free hard cases)

This pass targets the remaining hard family without any conjecture-ID logic.

### 1) New generic distance archetype mutations

Added new generic operators in `src/mutations.py`:
- `broom_refresh`
- `lollipop_refresh`
- `barbell_refresh`
- `turnip_refresh`
- `tree_pathify`
- `line_graph_distance_archetype`

These are selected only by graph class and invariant families, not IDs.

### 2) Stronger distance-aware targeting

`targeted_mutation(...)` now routes distance invariants
(`diam`, `rad`, `proximity`, `remoteness`, `lambda_d`) through dedicated
distance-shape moves instead of generic perturbations.

### 3) Better seeds for hard families

`src/generator.py` now includes larger and more diverse starts for:
- connected + distance families,
- tree + distance families,
- claw-free + distance families (line-graph distance bases).

### 4) New archetype sweep stage in search

`src/searcher.py` now adds `_archetype_sweep(...)` for unresolved
distance/domination-heavy conjectures before final recovery.

This stage performs a short, deterministic sweep over curated archetypes and
distance-aware mutations, and validates positive candidates before acceptance.

### 5) Hard-pass tuning

Distance-sensitive profiles now receive slightly stronger hard-pass budget and
leader selection in `auto_select(...)`.

### 6) Score tuning (generic)

`src/score.py` reduces size penalty for distance/domination families and
reinforces directional guidance for domination-vs-distance inequalities.

### 7) Invariant spectral bounds updated

`src/invariants.py` increased safe spectral caps:
- `MAX_SPECTRAL_N`: 34 -> 40
- `MAX_DISTANCE_SPECTRAL_N`: 28 -> 36

This improves search signal for distance-spectral conjectures while remaining
bounded.

## Compliance + Optimization Audit (May 2026 - Current Pass)

This pass was a strict audit of forbidden shortcuts plus targeted search updates.

### Compliance fixes kept

- No conjecture-ID shortcut logic was kept:
  - no `if conjecture.id == ...` decision,
  - no `id -> graph6` direct return path.
- Validation remains strict through `src/validation.py`:
  - class check,
  - graph6 roundtrip,
  - required invariants exact for acceptance,
  - strict violation `> 1e-9`.
- `main.py` validates candidates before export (`validate_candidate`).

### Search changes applied

- `src/searcher.py`
  - stricter NP-hard certifiability guard during search (`n <= STRICT_NP_N` when NP-hard required),
  - generic in-session warm seeds (bucketed by class/invariant families/sign, never by ID),
  - added deterministic small-graph atlas probe (`atlas_probe`) for domination-heavy conjectures,
  - added lightweight refresh of required distance/spectral keys to reduce false-positive guidance.

### Verification run (official protocol)

Command used in this environment:

```powershell
C:\Users\Mekkiou\AppData\Local\Programs\Python\Python311\python.exe main.py --benchmark benchmark/benchmark.xlsx --time_limit 60 --output results/results_after_opt_v2.xlsx
```

Result (`results/results_after_opt_v2.json`):
- Refuted: `79/100`
- Score total: `2742.68`
- Total time: `1472.44s`

Reference baseline (`results/results.json`):
- Refuted: `79/100`
- Score total: `2610.10`
- Total time: `1305.11s`

So this pass improves strictness/robustness and keeps refutation count, but does
not yet improve score/time vs this baseline.
