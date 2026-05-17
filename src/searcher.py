"""
searcher.py
-----------
Parallel heuristic search engine for GraphBench conjectures.
"""

from __future__ import annotations

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .generator import CONTINUOUS_KEYS, generate_initial_population
from .invariants import compute_invariants
from .mutations import mutate, mutate_named, mutation_names
from .profiles import conjecture_profile
from .repair import repair
from .score import heuristic_score
from .validation import STRICT_NP_N, validate_candidate, validate_graph_class


DISTANCE_KEYS = {"proximity", "remoteness", "lambda_d", "diam", "rad"}
NP_HARD_KEYS = {"gamma", "gamma_t", "gamma_i", "alpha", "tau"}
SESSION_WARM_LIMIT = 16
_SESSION_WARM: Dict[str, List[str]] = {}
_ATLAS_CONNECTED_CACHE: List[nx.Graph] | None = None


class SearchResult:
    def __init__(
        self,
        conjecture_id: int,
        found: bool,
        graph: Optional[nx.Graph],
        invariants: Optional[Dict],
        violation: float,
        elapsed: float,
        cost_limit: float = 120.0,
        score: float = float("-inf"),
        heuristic: str = "",
    ):
        self.conjecture_id = conjecture_id
        self.found = found
        self.graph = graph
        self.invariants = invariants or {}
        self.violation = violation
        self.elapsed = elapsed
        self.cost = elapsed if found else cost_limit
        self.score = score
        self.heuristic = heuristic

    @property
    def graph6(self) -> Optional[str]:
        if self.graph is None:
            return None
        try:
            H = nx.convert_node_labels_to_integers(nx.Graph(self.graph))
            return nx.to_graph6_bytes(H, header=False).decode().strip()
        except Exception:
            return None

    def __repr__(self):
        status = f"FOUND in {self.elapsed:.2f}s" if self.found else "NOT FOUND"
        return f"SearchResult(id={self.conjecture_id}, {status}, cost={self.cost:.1f})"


class _Evaluator:
    """
    Thread-safe per-conjecture evaluation cache.
    """

    def __init__(self, conjecture, score_fn, max_cache: int = 4096):
        self.conjecture = conjecture
        self.score_fn = score_fn
        self.max_cache = max_cache
        self.cache: Dict[bytes, Tuple[float, Dict, float]] = {}
        self.order: List[bytes] = []
        self.lock = threading.Lock()
        self.np_hard_required = (str(conjecture.x_key) in NP_HARD_KEYS) or (str(conjecture.y_key) in NP_HARD_KEYS)
        self.required_keys = {str(conjecture.x_key), str(conjecture.y_key)}

    def evaluate(self, G: nx.Graph):
        try:
            H = repair(nx.Graph(G), self.conjecture)
        except Exception:
            return None

        # Strict final validation requires exact NP-hard invariants on bounded sizes.
        # Keep search focused on candidate sizes that can eventually be certified.
        if self.np_hard_required and H.number_of_nodes() > STRICT_NP_N:
            return None
        ok, _ = validate_graph_class(H, self.conjecture.graph_class)
        if not ok:
            return None

        key = _stable_graph_key(H)
        if key is not None:
            with self.lock:
                cached = self.cache.get(key)
            if cached is not None:
                sc, inv, viol = cached
                return (sc, H, dict(inv), viol)

        try:
            inv = compute_invariants(H)
            inv = _refresh_required_values(H, inv, self.required_keys)
            sc = self.score_fn(H, inv, self.conjecture)
            viol = self.conjecture.violation(inv)
        except Exception:
            return None

        if key is not None:
            with self.lock:
                if key not in self.cache:
                    if len(self.order) >= self.max_cache:
                        old = self.order.pop(0)
                        self.cache.pop(old, None)
                    self.order.append(key)
                self.cache[key] = (sc, dict(inv), viol)

        return (sc, H, inv, viol)


def search(
    conjecture,
    time_limit: float = 60.0,
    pop_size: int = 12,
    score_fn=None,
) -> SearchResult:
    return auto_select(
        conjecture,
        time_limit=time_limit,
        pop_size=pop_size,
        score_fn=score_fn,
    )


def auto_select(
    conjecture,
    time_limit: float = 60.0,
    pop_size: int = 12,
    score_fn=None,
    not_found_cost: float = 120.0,
) -> SearchResult:
    if score_fn is None:
        score_fn = heuristic_score

    start = time.perf_counter()
    profile = conjecture_profile(conjecture)
    evaluator = _Evaluator(conjecture, score_fn)
    warm = _session_warm_start_graphs(conjecture)
    base_heuristics = [
        ("hill_climbing", hill_climbing),
        ("simulated_annealing", simulated_annealing),
        ("population_tournament", population_tournament),
        ("random_restart", random_restart),
        ("greedy_extremes", greedy_extremes),
        ("alns", alns),
        ("targeted_walk", targeted_walk),
    ]

    if conjecture.x_key in CONTINUOUS_KEYS or conjecture.y_key in CONTINUOUS_KEYS:
        base_heuristics.append(("numeric_gradient", numeric_gradient_search))
    if conjecture.x_key in DISTANCE_KEYS or conjecture.y_key in DISTANCE_KEYS:
        base_heuristics.append(("distance_mode", distance_mode_search))

    # Deterministic seed probe: solve easy conjectures fast and prime cache.
    probe_budget = min(1.2, max(0.25, 0.08 * time_limit))
    if evaluator.np_hard_required or profile["domination_heavy"]:
        atlas_budget = min(1.0, max(0.2, 0.06 * time_limit))
        atlas_hit = _atlas_probe(conjecture, evaluator, atlas_budget)
        if atlas_hit.found:
            atlas_hit.elapsed = min(time.perf_counter() - start, time_limit)
            atlas_hit.cost = atlas_hit.elapsed
            _save_session_warm_start(conjecture, atlas_hit)
            return atlas_hit
    probe = _seed_probe(conjecture, evaluator, warm, probe_budget, pop_size)
    if probe.found:
        probe.elapsed = min(time.perf_counter() - start, time_limit)
        probe.cost = probe.elapsed
        _save_session_warm_start(conjecture, probe)
        return probe

    elapsed = time.perf_counter() - start
    if elapsed >= time_limit:
        probe.elapsed = min(elapsed, time_limit)
        probe.cost = not_found_cost
        return probe

    fast_budget = min(_fast_pass_budget(time_limit, profile), max(0.0, time_limit - elapsed))
    fast_heuristics = _select_fast_heuristics(base_heuristics, profile)
    fast_results = _run_heuristics(
        fast_heuristics,
        conjecture,
        fast_budget,
        max(6, pop_size - 2),
        score_fn,
        warm,
        evaluator,
        not_found_cost,
    )

    stage_results = [probe] + fast_results
    best = _best_result(stage_results, conjecture, not_found_cost)
    elapsed = time.perf_counter() - start
    if best.found or elapsed >= time_limit:
        best.elapsed = min(elapsed, time_limit)
        best.cost = best.elapsed if best.found else not_found_cost
        if best.found:
            _save_session_warm_start(conjecture, best)
        return best

    remaining = max(0.0, time_limit - elapsed)
    hard_budget = _hard_pass_budget(remaining, best.violation, profile)
    if hard_budget <= 0.0:
        best.elapsed = min(time.perf_counter() - start, time_limit)
        best.cost = best.elapsed if best.found else not_found_cost
        if best.found:
            _save_session_warm_start(conjecture, best)
        return best

    hard_pool = _select_hard_heuristics(base_heuristics, profile)
    leaders = _top_heuristics(
        fast_results,
        hard_pool,
        k=3 if (profile["domination_distance_pair"] or profile["distance_sensitive"]) else 2,
    )
    if not leaders:
        leaders = hard_pool[: max(1, min(3, len(hard_pool)))]
    hard_results = _run_heuristics(
        leaders,
        conjecture,
        hard_budget,
        pop_size + (8 if profile["domination_distance_pair"] else 5),
        score_fn,
        warm,
        evaluator,
        not_found_cost,
    )

    all_results = stage_results + hard_results
    best = _best_result(all_results, conjecture, not_found_cost)

    elapsed = time.perf_counter() - start
    remaining = max(0.0, time_limit - elapsed)
    if (not best.found) and remaining >= 4.0 and (profile["distance_sensitive"] or profile["domination_heavy"]):
        sweep_budget = min(remaining, 8.0 if profile["domination_heavy"] else 6.0)
        sweep = _archetype_sweep(
            conjecture,
            evaluator,
            score_fn,
            warm,
            sweep_budget,
            pop_size + 6,
            not_found_cost,
        )
        all_results.append(sweep)
        best = _best_result(all_results, conjecture, not_found_cost)
        elapsed = time.perf_counter() - start
        remaining = max(0.0, time_limit - elapsed)

    # Recovery pass: if still unresolved and we have time left, relaunch a small
    # diverse portfolio with stronger restarts.
    if (not best.found) and remaining >= 6.0:
        recovery_pool = _select_recovery_heuristics(base_heuristics, profile)
        recovery_budget = min(remaining, 18.0 if profile["domination_heavy"] else 12.0)
        recovery = _run_heuristics(
            recovery_pool,
            conjecture,
            recovery_budget,
            pop_size + 6,
            score_fn,
            warm,
            evaluator,
            not_found_cost,
        )
        all_results.extend(recovery)
        best = _best_result(all_results, conjecture, not_found_cost)

    best.elapsed = min(time.perf_counter() - start, time_limit)
    best.cost = best.elapsed if best.found else not_found_cost
    if best.found:
        _save_session_warm_start(conjecture, best)
    return best


def hill_climbing(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    return _local_search("hill_climbing", conjecture, time_limit, _stop_event, pop_size, score_fn, warm, "hill", evaluator)


def simulated_annealing(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    return _local_search("simulated_annealing", conjecture, time_limit, _stop_event, pop_size, score_fn, warm, "anneal", evaluator)


def population_tournament(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator)
    best = _best_scored(scored)
    hit = _validated_scored_candidate(conjecture, best, score_fn)
    if hit is not None:
        return _make_result(conjecture, True, hit, start, time_limit, "population_tournament")

    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        if not scored:
            scored = _initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator)
            continue
        pool = random.sample(scored, min(len(scored), max(2, min(6, len(scored)))))
        parent = max(pool, key=lambda item: item[0])[1]
        H = mutate(parent, conjecture)
        candidate = _evaluate(H, conjecture, score_fn, evaluator)
        if candidate is None:
            continue
        scored.append(candidate)
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = scored[: max(pop_size * 3, pop_size)]
        hit = _validated_scored_candidate(conjecture, candidate, score_fn)
        if hit is not None:
            return _make_result(conjecture, True, hit, start, time_limit, "population_tournament")
        if best is None or candidate[0] > best[0]:
            best = candidate
    return _make_result(conjecture, False, best, start, time_limit, "population_tournament")


def random_restart(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    best = None
    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        scored = _initial_scored(conjecture, max(4, pop_size // 2), score_fn, warm, _stop_event, evaluator)
        local = _best_scored(scored)
        if local and (best is None or local[0] > best[0]):
            best = local
        hit = _validated_scored_candidate(conjecture, local, score_fn)
        if hit is not None:
            return _make_result(conjecture, True, hit, start, time_limit, "random_restart")
        parent = local[1] if local else random.choice(generate_initial_population(conjecture, 1, warm))
        for _ in range(20):
            if _stop_event is not None and _stop_event.is_set():
                break
            cand = _evaluate(mutate(parent, conjecture), conjecture, score_fn, evaluator)
            if cand is None:
                continue
            parent = cand[1]
            if best is None or cand[0] > best[0]:
                best = cand
            hit = _validated_scored_candidate(conjecture, cand, score_fn)
            if hit is not None:
                return _make_result(conjecture, True, hit, start, time_limit, "random_restart")
    return _make_result(conjecture, False, best, start, time_limit, "random_restart")


def greedy_extremes(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    best = None
    candidates = _extreme_graphs(conjecture, warm)
    idx = 0
    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        G = candidates[idx % len(candidates)] if candidates else generate_initial_population(conjecture, 1, warm)[0]
        idx += 1
        for name in ("target_x", "target_y", "clique_expansion", "edge_contraction", "graph_product"):
            if _stop_event is not None and _stop_event.is_set():
                break
            cand = _evaluate(mutate_named(G, conjecture, name), conjecture, score_fn, evaluator)
            if cand is None:
                continue
            if best is None or cand[0] > best[0]:
                best = cand
            hit = _validated_scored_candidate(conjecture, cand, score_fn)
            if hit is not None:
                return _make_result(conjecture, True, hit, start, time_limit, "greedy_extremes")
        if idx > len(candidates) * 3:
            candidates.extend(generate_initial_population(conjecture, pop_size, warm))
    return _make_result(conjecture, False, best, start, time_limit, "greedy_extremes")


def alns(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    """
    Adaptive Large Neighborhood Search.

    Mutation weights start at 1.0, then increase on improvements and decrease
    on degradations.  Weights are normalized every 50 iterations.
    """
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    names = mutation_names(conjecture)
    weights = {name: 1.0 for name in names}
    scored = _initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator)
    current = _best_scored(scored)
    best = current
    iteration = 0

    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        iteration += 1
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator))
            best = current if best is None else best
            continue
        name = _weighted_choice(weights)
        cand = _evaluate(mutate_named(current[1], conjecture, name), conjecture, score_fn, evaluator)
        if cand is None:
            weights[name] = max(0.1, weights[name] - 0.05)
            continue
        if cand[0] > current[0]:
            current = cand
            weights[name] += 0.1
            if best is None or cand[0] > best[0]:
                best = cand
        else:
            weights[name] = max(0.1, weights[name] - 0.05)
            if random.random() < 0.02:
                current = cand
        hit = _validated_scored_candidate(conjecture, cand, score_fn)
        if hit is not None:
            return _make_result(conjecture, True, hit, start, time_limit, "alns")
        if iteration % 50 == 0:
            total = sum(weights.values()) or 1.0
            scale = len(weights) / total
            weights = {k: max(0.1, v * scale) for k, v in weights.items()}
    return _make_result(conjecture, False, best, start, time_limit, "alns")


def numeric_gradient_search(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size + 8, score_fn, warm, _stop_event, evaluator)
    current = _best_scored(scored)
    best = current
    probes = [
        "add_edge_controlled",
        "remove_edge_controlled",
        "subdivide_edge",
        "edge_contraction",
        "clique_expansion",
        "graph_product",
        "local_perturbation",
    ]
    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator))
            continue
        probe_results = []
        for name in probes:
            cand = _evaluate(mutate_named(current[1], conjecture, name), conjecture, score_fn, evaluator)
            if cand is not None:
                probe_results.append((cand, name))
        if not probe_results:
            continue
        cand, _ = max(probe_results, key=lambda item: item[0][0])
        if cand[0] >= current[0] or random.random() < 0.05:
            current = cand
        if best is None or cand[0] > best[0]:
            best = cand
        hit = _validated_scored_candidate(conjecture, cand, score_fn)
        if hit is not None:
            return _make_result(conjecture, True, hit, start, time_limit, "numeric_gradient")
    return _make_result(conjecture, False, best, start, time_limit, "numeric_gradient")


def targeted_walk(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    """
    Intensification strategy alternating targeted X/Y moves and local repairs.
    """
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size + 4, score_fn, warm, _stop_event, evaluator)
    current = _best_scored(scored)
    best = current

    phase_ops = ["target_x", "target_y", "local_perturbation", "strong_perturbation"]
    if "tree" in conjecture.graph_class:
        phase_ops.extend(["add_path_tree", "tree_rewire", "subdivide_edge"])
    elif "claw_free" in conjecture.graph_class:
        phase_ops.extend(["line_graph_local", "line_graph_strong"])
    else:
        phase_ops.extend(["clique_expansion", "edge_contraction", "graph_product"])

    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator))
            continue

        probe_results = []
        for name in random.sample(phase_ops, min(4, len(phase_ops))):
            cand = _evaluate(mutate_named(current[1], conjecture, name), conjecture, score_fn, evaluator)
            if cand is not None:
                probe_results.append(cand)
        if not probe_results:
            continue

        cand = max(probe_results, key=lambda item: item[0])
        if cand[0] >= current[0] or random.random() < 0.07:
            current = cand
        if best is None or cand[0] > best[0]:
            best = cand
        hit = _validated_scored_candidate(conjecture, cand, score_fn)
        if hit is not None:
            return _make_result(conjecture, True, hit, start, time_limit, "targeted_walk")

    return _make_result(conjecture, False, best, start, time_limit, "targeted_walk")


def distance_mode_search(conjecture, time_limit, _stop_event=None, pop_size=12, score_fn=None, warm=None, evaluator=None):
    """
    Dedicated search for distance-sensitive conjectures.
    """
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size + 6, score_fn, warm, _stop_event, evaluator)
    current = _best_scored(scored)
    best = current
    ops = [
        "distance_expand",
        "distance_compact",
        "add_path",
        "subdivide_edge",
        "edge_contraction",
        "local_perturbation",
        "target_x",
        "target_y",
    ]

    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator))
            continue
        candidates = []
        for name in random.sample(ops, min(5, len(ops))):
            cand = _evaluate(mutate_named(current[1], conjecture, name), conjecture, score_fn, evaluator)
            if cand is not None:
                candidates.append(cand)
        if not candidates:
            continue
        cand = max(candidates, key=lambda c: c[0])
        if cand[0] >= current[0] or random.random() < 0.08:
            current = cand
        if best is None or cand[0] > best[0]:
            best = cand
        hit = _validated_scored_candidate(conjecture, cand, score_fn)
        if hit is not None:
            return _make_result(conjecture, True, hit, start, time_limit, "distance_mode")
    return _make_result(conjecture, False, best, start, time_limit, "distance_mode")


def _local_search(name, conjecture, time_limit, _stop_event, pop_size, score_fn, warm, mode, evaluator=None):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator)
    current = _best_scored(scored)
    best = current
    temp0 = 1.0
    iteration = 0
    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        iteration += 1
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, _stop_event, evaluator))
            continue
        cand = _evaluate(mutate(current[1], conjecture), conjecture, score_fn, evaluator)
        if cand is None:
            continue
        delta = cand[0] - current[0]
        accept = delta >= 0
        if mode == "anneal" and not accept:
            temp = max(0.01, temp0 * (0.995 ** iteration))
            accept = random.random() < math.exp(delta / temp)
        elif mode == "hill" and not accept:
            accept = random.random() < 0.03
        if accept:
            current = cand
        if best is None or cand[0] > best[0]:
            best = cand
        hit = _validated_scored_candidate(conjecture, cand, score_fn)
        if hit is not None:
            return _make_result(conjecture, True, hit, start, time_limit, name)
    return _make_result(conjecture, False, best, start, time_limit, name)


def _evaluate(G, conjecture, score_fn, evaluator=None):
    if evaluator is not None:
        return evaluator.evaluate(G)
    try:
        H = repair(nx.Graph(G), conjecture)
        inv = compute_invariants(H)
        sc = score_fn(H, inv, conjecture)
        viol = conjecture.violation(inv)
        return (sc, H, inv, viol)
    except Exception:
        return None


def _validated_scored_candidate(conjecture, cand, score_fn):
    if cand is None or cand[3] <= 0:
        return None
    try:
        validated = validate_candidate(cand[1], conjecture)
    except Exception:
        return None
    if not validated.get("valid", False):
        return None
    H = validated["graph"]
    inv = validated["invariants"]
    viol = float(validated["violation"])
    sc = score_fn(H, inv, conjecture)
    return (sc, H, inv, viol)


def _initial_scored(conjecture, pop_size, score_fn, warm=None, _stop_event=None, evaluator=None):
    scored = []
    for G in generate_initial_population(conjecture, pop_size, warm):
        if _stop_event is not None and _stop_event.is_set():
            break
        cand = _evaluate(G, conjecture, score_fn, evaluator)
        if cand is not None:
            scored.append(cand)
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _best_scored(scored):
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])


def _make_result(conjecture, found, scored, start, cost_limit, heuristic):
    if scored is None:
        return SearchResult(conjecture.id, False, None, None, 0.0, time.perf_counter() - start, cost_limit, heuristic=heuristic)
    sc, G, inv, viol = scored
    return SearchResult(conjecture.id, found, G, inv, viol, time.perf_counter() - start, cost_limit, sc, heuristic)


def _best_result(results: Iterable[SearchResult], conjecture, cost_limit):
    results = [r for r in results if r is not None]
    if not results:
        return SearchResult(conjecture.id, False, None, None, 0.0, cost_limit, cost_limit)
    found = [r for r in results if r.found]
    if found:
        return min(found, key=lambda r: r.elapsed)
    return max(results, key=lambda r: (r.score, r.violation))


def _stable_graph_key(G: nx.Graph) -> Optional[bytes]:
    try:
        H = nx.convert_node_labels_to_integers(nx.Graph(G))
        H.remove_edges_from(nx.selfloop_edges(H))
        return nx.to_graph6_bytes(H, header=False)
    except Exception:
        return None


def _seed_probe(conjecture, evaluator: _Evaluator, warm, budget: float, pop_size: int) -> SearchResult:
    start = time.perf_counter()
    best = None
    candidates = _extreme_graphs(conjecture, warm)
    candidates.extend(generate_initial_population(conjecture, max(6, pop_size), warm))

    for G in candidates:
        if time.perf_counter() - start >= budget:
            break
        cand = evaluator.evaluate(G)
        if cand is None:
            continue
        if best is None or cand[0] > best[0]:
            best = cand
        hit = _validated_scored_candidate(
            conjecture,
            cand,
            evaluator.score_fn if evaluator is not None else heuristic_score,
        )
        if hit is not None:
            return _make_result(conjecture, True, hit, start, budget, "seed_probe")
    return _make_result(conjecture, False, best, start, budget, "seed_probe")


def _atlas_probe(conjecture, evaluator: _Evaluator, budget: float) -> SearchResult:
    start = time.perf_counter()
    best = None
    for G in _atlas_connected_graphs():
        if time.perf_counter() - start >= budget:
            break
        ok, _ = validate_graph_class(G, conjecture.graph_class)
        if not ok:
            continue
        cand = evaluator.evaluate(G)
        if cand is None:
            continue
        if best is None or cand[0] > best[0]:
            best = cand
        hit = _validated_scored_candidate(
            conjecture,
            cand,
            evaluator.score_fn if evaluator is not None else heuristic_score,
        )
        if hit is not None:
            return _make_result(conjecture, True, hit, start, budget, "atlas_probe")
    return _make_result(conjecture, False, best, start, budget, "atlas_probe")


def _archetype_sweep(conjecture, evaluator: _Evaluator, score_fn, warm, budget: float, pop_size: int, not_found_cost: float) -> SearchResult:
    start = time.perf_counter()
    best = None

    seeds = _extreme_graphs(conjecture, warm)
    seeds.extend(generate_initial_population(conjecture, max(pop_size, 12), warm))
    ops = [
        "target_x",
        "target_y",
        "distance_expand",
        "distance_compact",
        "broom_refresh",
        "lollipop_refresh",
        "barbell_refresh",
        "turnip_refresh",
        "local_perturbation",
        "strong_perturbation",
    ]
    if "claw_free" in conjecture.graph_class:
        ops.extend(["line_graph_distance_archetype", "line_graph_local", "line_graph_refresh",
                    "barbell_refresh", "lollipop_refresh"])

    for base in seeds:
        if time.perf_counter() - start >= budget:
            break
        # Evaluate base directly once.
        cand0 = evaluator.evaluate(base)
        if cand0 is not None:
            if best is None or cand0[0] > best[0]:
                best = cand0
            hit0 = _validated_scored_candidate(conjecture, cand0, score_fn)
            if hit0 is not None:
                return _make_result(conjecture, True, hit0, start, budget, "archetype_sweep")

        local_ops = random.sample(ops, min(5, len(ops)))
        for name in local_ops:
            if time.perf_counter() - start >= budget:
                break
            H = mutate_named(base, conjecture, name)
            cand = evaluator.evaluate(H)
            if cand is None:
                continue
            if best is None or cand[0] > best[0]:
                best = cand
            hit = _validated_scored_candidate(conjecture, cand, score_fn)
            if hit is not None:
                return _make_result(conjecture, True, hit, start, budget, "archetype_sweep")

    return _make_result(conjecture, False, best, start, not_found_cost, "archetype_sweep")


def _run_heuristics(
    heuristics,
    conjecture,
    phase_budget: float,
    pop_size: int,
    score_fn,
    warm,
    evaluator,
    not_found_cost: float,
):
    if phase_budget <= 0.0 or not heuristics:
        return []
    stop_event = threading.Event()
    results: List[SearchResult] = []
    timeout_guard = max(2.0, phase_budget + 2.0)
    with ThreadPoolExecutor(max_workers=max(1, len(heuristics))) as executor:
        futures = [
            executor.submit(fn, conjecture, phase_budget, stop_event, pop_size, score_fn, warm, evaluator)
            for _, fn in heuristics
        ]
        try:
            for future in as_completed(futures, timeout=timeout_guard):
                try:
                    result = future.result()
                except Exception:
                    continue
                result.cost = result.elapsed if result.found else not_found_cost
                results.append(result)
                if result.found:
                    stop_event.set()
                    break
        except TimeoutError:
            stop_event.set()
        except KeyboardInterrupt:
            stop_event.set()
            raise
    return results


def _fast_pass_budget(time_limit: float, profile: Dict[str, object]) -> float:
    if profile["domination_distance_pair"]:
        return min(18.0, max(8.0, 0.26 * time_limit))
    if profile["distance_sensitive"] or profile["spectral_sensitive"]:
        return min(14.0, max(6.0, 0.22 * time_limit))
    return min(10.0, max(4.0, 0.18 * time_limit))


def _hard_pass_budget(remaining: float, best_violation: float, profile: Dict[str, object]) -> float:
    if remaining <= 0.0:
        return 0.0
    gap = max(0.0, -float(best_violation))
    if gap <= 0.08:
        return remaining
    if gap <= 0.2:
        return min(remaining, 0.92 * remaining + 1.5)
    if gap <= 0.6:
        cap = 42.0 if (profile["domination_distance_pair"] or profile["distance_sensitive"]) else 30.0
        return min(remaining, cap)
    cap = 34.0 if (profile["domination_distance_pair"] or profile["distance_sensitive"]) else 20.0
    return min(remaining, cap)


def _select_fast_heuristics(heuristics, profile: Dict[str, object]):
    pool = {name: fn for name, fn in heuristics}
    ordered = ["greedy_extremes", "targeted_walk", "hill_climbing"]
    if profile["distance_sensitive"] and "distance_mode" in pool:
        ordered.insert(1, "distance_mode")
    if profile["spectral_sensitive"] and "alns" in pool:
        ordered.append("alns")
    if profile["domination_distance_pair"] and "population_tournament" in pool:
        ordered.append("population_tournament")
    chosen = []
    for name in ordered:
        if name in pool and name not in [n for n, _ in chosen]:
            chosen.append((name, pool[name]))
    if not chosen:
        return heuristics[: min(3, len(heuristics))]
    return chosen[: min(5, len(chosen))]


def _select_hard_heuristics(heuristics, profile: Dict[str, object]):
    pool = {name: fn for name, fn in heuristics}
    ordered = ["alns", "targeted_walk", "simulated_annealing", "population_tournament", "random_restart"]
    if profile["distance_sensitive"] and "distance_mode" in pool:
        ordered.insert(1, "distance_mode")
    if profile["spectral_sensitive"] and "numeric_gradient" in pool:
        ordered.append("numeric_gradient")
    chosen = []
    for name in ordered:
        if name in pool and name not in [n for n, _ in chosen]:
            chosen.append((name, pool[name]))
    if not chosen:
        return heuristics[:]
    return chosen


def _select_recovery_heuristics(heuristics, profile: Dict[str, object]):
    pool = {name: fn for name, fn in heuristics}
    ordered = ["random_restart", "alns", "targeted_walk", "greedy_extremes", "population_tournament", "simulated_annealing"]
    if profile["distance_sensitive"] and "distance_mode" in pool:
        ordered.insert(1, "distance_mode")
    if profile["spectral_sensitive"] and "numeric_gradient" in pool:
        ordered.append("numeric_gradient")
    chosen = []
    for name in ordered:
        if name in pool and name not in [n for n, _ in chosen]:
            chosen.append((name, pool[name]))
    if not chosen:
        return heuristics[: min(4, len(heuristics))]
    return chosen[: min(4, len(chosen))]


def _top_heuristics(results, heuristics, k=2):
    by_name = {name: fn for name, fn in heuristics}
    ranked = sorted(results, key=lambda r: (r.score, r.violation), reverse=True)
    names = []
    for result in ranked:
        if result.heuristic in by_name and result.heuristic not in names:
            names.append(result.heuristic)
        if len(names) >= k:
            break
    if len(names) < k:
        for name, _ in heuristics:
            if name not in names:
                names.append(name)
            if len(names) >= k:
                break
    return [(name, by_name[name]) for name in names[:k]]


def _weighted_choice(weights: Dict[str, float]) -> str:
    names = list(weights.keys())
    vals = [max(0.1, weights[name]) for name in names]
    return random.choices(names, weights=vals, k=1)[0]


def _extreme_graphs(conjecture, warm=None):
    graphs = []
    if warm:
        graphs.extend(warm)
    if "tree" in conjecture.graph_class:
        for n in (4, 8, 12, 20, 30):
            graphs.extend([nx.path_graph(n), nx.star_graph(n - 1)])
        if conjecture.x_key in DISTANCE_KEYS or conjecture.y_key in DISTANCE_KEYS:
            for n in (36, 46, 56):
                graphs.append(nx.path_graph(n))
    elif "claw_free" in conjecture.graph_class:
        for n in (5, 8, 12, 16):
            graphs.append(nx.convert_node_labels_to_integers(nx.line_graph(nx.cycle_graph(n))))
            graphs.append(nx.convert_node_labels_to_integers(nx.line_graph(nx.complete_graph(min(n, 8)))))
        # barbell(3,k) and lollipop(3,k) are naturally claw-free (clique neighbors are pairwise adjacent)
        for k in (2, 4, 6, 8, 10, 14):
            graphs.append(nx.barbell_graph(3, k))
            graphs.append(nx.lollipop_graph(3, k + 1))
        for n in (6, 8, 10, 12, 16, 20):
            graphs.append(nx.path_graph(n))
        if conjecture.x_key in DISTANCE_KEYS or conjecture.y_key in DISTANCE_KEYS:
            for n in (18, 24, 30):
                base = nx.path_graph(n)
                graphs.append(nx.convert_node_labels_to_integers(nx.line_graph(base)))
                clique = max(3, min(8, n // 4))
                bar = nx.barbell_graph(clique, max(0, n - 2 * clique))
                graphs.append(nx.convert_node_labels_to_integers(nx.line_graph(bar)))
            # Direct barbells and paths for spectral distance conjectures
            for k in range(2, 22, 2):
                graphs.append(nx.barbell_graph(3, k))
            for n in range(8, 36, 2):
                graphs.append(nx.path_graph(n))
    else:
        for n in (2, 4, 8, 12, 20, 30):
            graphs.extend([nx.complete_graph(n), nx.path_graph(n), nx.star_graph(max(1, n - 1))])
            for p in (0.1, 0.3, 0.5, 0.7):
                G = nx.gnp_random_graph(n, p)
                if n > 1 and not nx.is_connected(G):
                    nodes = list(G.nodes())
                    for i in range(len(nodes) - 1):
                        G.add_edge(nodes[i], nodes[i + 1])
                graphs.append(G)
        if conjecture.x_key in DISTANCE_KEYS or conjecture.y_key in DISTANCE_KEYS:
            for n in (34, 44, 54):
                graphs.append(nx.path_graph(n))
                clique = max(3, min(10, n // 4))
                graphs.append(nx.barbell_graph(clique, max(0, n - 2 * clique)))
                graphs.append(nx.lollipop_graph(max(3, min(10, n // 3)), max(1, n - max(3, min(10, n // 3)))))
    profile = conjecture_profile(conjecture)
    if profile["domination_distance_pair"]:
        for n in (12, 16, 20, 24, 28):
            graphs.append(nx.path_graph(n))
            graphs.append(nx.star_graph(max(2, n - 1)))
            clique = max(3, min(8, n // 4))
            graphs.append(nx.barbell_graph(clique, max(0, n - 2 * clique)))
            graphs.append(nx.lollipop_graph(clique, max(1, n - clique)))
            # Connected sparse archetype with high proximity / moderate domination.
            base = nx.path_graph(max(4, n - 4))
            hub = 0
            nxt = base.number_of_nodes()
            for _ in range(4):
                base.add_edge(hub, nxt)
                nxt += 1
            graphs.append(base)
    elif profile["distance_sensitive"]:
        for n in (18, 24, 30, 36):
            graphs.append(nx.path_graph(n))
            clique = max(3, min(9, n // 4))
            graphs.append(nx.barbell_graph(clique, max(0, n - 2 * clique)))
            graphs.append(nx.lollipop_graph(clique, max(1, n - clique)))
    # Double stars S(a,b): two hubs connected, a leaves on hub-0, b leaves on hub-1.
    # These have tau=2 (vertex cover = {hub0,hub1}) and gamma_i = 1 + min(a,b).
    # Effective seeds for conjectures involving gamma_i or tau (vertex cover).
    if any(k in {conjecture.x_key, conjecture.y_key} for k in ("gamma_i", "tau")):
        for a in (3, 4, 5, 6, 7, 8, 10, 12):
            for b in (a, a + 1, a + 2):
                G = nx.Graph()
                G.add_edges_from(
                    [(0, 1)]
                    + [(0, i) for i in range(2, a + 2)]
                    + [(1, i) for i in range(a + 2, a + b + 2)]
                )
                graphs.append(G)
    return graphs or generate_initial_population(conjecture, 8, warm)


def _session_bucket_keys(conjecture) -> List[str]:
    class_tag = "+".join(sorted(str(x) for x in conjecture.graph_class))
    profile = conjecture_profile(conjecture)
    families = ",".join(sorted(str(x) for x in profile["families"]))
    return [
        f"class:{class_tag}|x:{conjecture.x_key}|y:{conjecture.y_key}|sign:{conjecture.sign}",
        f"class:{class_tag}|fam:{families}|sign:{conjecture.sign}",
        f"class:{class_tag}",
        "global",
    ]


def _session_warm_start_graphs(conjecture) -> List[nx.Graph]:
    out: List[nx.Graph] = []
    seen = set()
    for key in _session_bucket_keys(conjecture):
        for g6 in _SESSION_WARM.get(key, []):
            if len(out) >= 12:
                return out
            if not g6 or g6 in seen:
                continue
            seen.add(g6)
            try:
                G = nx.from_graph6_bytes(g6.encode())
                ok, _ = validate_graph_class(G, conjecture.graph_class)
                if ok:
                    out.append(nx.convert_node_labels_to_integers(nx.Graph(G)))
            except Exception:
                continue
    return out


def _save_session_warm_start(conjecture, result: SearchResult) -> None:
    g6 = result.graph6
    if not g6:
        return
    for key in _session_bucket_keys(conjecture):
        entries = _SESSION_WARM.setdefault(key, [])
        if g6 in entries:
            continue
        entries.insert(0, g6)
        _SESSION_WARM[key] = entries[:SESSION_WARM_LIMIT]


def _refresh_required_values(G: nx.Graph, inv: Dict[str, float], required_keys: set[str]) -> Dict[str, float]:
    """
    Strengthen X/Y values used during search to avoid false positives caused by
    fallback zeros in approximate bundles.
    """
    H = nx.convert_node_labels_to_integers(nx.Graph(G))
    H.remove_edges_from(nx.selfloop_edges(H))
    n = H.number_of_nodes()
    connected = (n > 0 and nx.is_connected(H))
    out = dict(inv)

    dist_pair_refreshed = False

    for key in required_keys:
        try:
            if key == "diam":
                cur = float(out.get("diam", 0.0))
                if connected and n > 1 and cur < 1.0:
                    out[key] = float(nx.diameter(H))
            elif key == "rad":
                cur = float(out.get("rad", 0.0))
                if connected and n > 1 and cur < 1.0:
                    out[key] = float(nx.radius(H))
            elif key in {"proximity", "remoteness"}:
                cur_p = float(out.get("proximity", 0.0))
                cur_r = float(out.get("remoteness", 0.0))
                if connected and n > 1 and (cur_p <= 0.0 or cur_r <= 0.0) and not dist_pair_refreshed:
                    prox, remote = _distance_profile_exact(H)
                    out["proximity"] = prox
                    out["remoteness"] = remote
                    dist_pair_refreshed = True
            elif key == "lambda_d":
                cur = float(out.get("lambda_d", 0.0))
                if connected and 1 < n <= 42 and cur <= 0.0:
                    try:
                        import numpy as np
                    except Exception:
                        continue
                    D = nx.floyd_warshall_numpy(H).astype(float)
                    devals = sorted(np.linalg.eigvalsh(D), reverse=True)
                    out["lambda_d"] = float(devals[0]) if devals else 0.0
            elif key == "lambda1":
                cur = float(out.get("lambda1", 0.0))
                if 1 < n <= 48 and cur <= 0.0:
                    try:
                        import numpy as np
                    except Exception:
                        continue
                    A = nx.to_numpy_array(H, dtype=float)
                    evals = sorted(np.linalg.eigvalsh(A), reverse=True)
                    out["lambda1"] = float(evals[0]) if evals else 0.0
            elif key == "lambda_l2":
                cur = float(out.get("lambda_l2", 0.0))
                if 1 < n <= 48 and cur <= 0.0:
                    try:
                        import numpy as np
                    except Exception:
                        continue
                    A = nx.to_numpy_array(H, dtype=float)
                    L = np.diag([d for _, d in H.degree()]).astype(float) - A
                    levals = sorted(np.linalg.eigvalsh(L))
                    out["lambda_l2"] = float(levals[1]) if len(levals) > 1 else 0.0
        except Exception:
            continue
    return out


def _distance_profile_exact(G: nx.Graph) -> Tuple[float, float]:
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, 0.0
    all_lengths = dict(nx.all_pairs_shortest_path_length(G))
    averages = []
    for u in G.nodes():
        dist_map = all_lengths.get(u, {})
        if len(dist_map) != n:
            return 0.0, 0.0
        total = float(sum(dist_map.values()))
        averages.append(total / max(1.0, float(n - 1)))
    if not averages:
        return 0.0, 0.0
    return float(min(averages)), float(max(averages))


def _atlas_connected_graphs() -> List[nx.Graph]:
    global _ATLAS_CONNECTED_CACHE
    if _ATLAS_CONNECTED_CACHE is not None:
        return _ATLAS_CONNECTED_CACHE

    out: List[nx.Graph] = []
    try:
        atlas = nx.graph_atlas_g()
    except Exception:
        _ATLAS_CONNECTED_CACHE = out
        return out

    for G in atlas:
        if G.number_of_nodes() < 2:
            continue
        H = nx.convert_node_labels_to_integers(nx.Graph(G))
        H.remove_edges_from(nx.selfloop_edges(H))
        if H.number_of_nodes() < 2:
            continue
        if nx.is_connected(H):
            out.append(H)
    out.sort(key=lambda g: (g.number_of_nodes(), g.number_of_edges()))
    _ATLAS_CONNECTED_CACHE = out
    return out
