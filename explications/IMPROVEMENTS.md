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

When enabled, `src/searcher.py` saves best known graphs in
`results/best_graphs.json` and loads them as seeds on later runs.  Official runs
should omit `--warm_start`.

## Verification

Command run:

```powershell
py main.py --ids 886 1191 1566 --time_limit 30 -v
```

Result:

- `886`: found in `1.02s`
- `1191`: found in `0.37s`
- `1566`: found in `0.42s`

All three are below the requested 10 seconds each.
