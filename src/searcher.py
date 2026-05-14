"""
searcher.py
-----------
Parallel heuristic search engine for GraphBench conjectures.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from .generator import CONTINUOUS_KEYS, generate_initial_population
from .invariants import compute_invariants
from .mutations import mutate, mutate_named, mutation_names
from .repair import repair
from .score import heuristic_score


BEST_GRAPHS_PATH = os.path.join("results", "best_graphs.json")


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


def _state_key(G: nx.Graph) -> bytes:
    H = nx.convert_node_labels_to_integers(nx.Graph(G))
    H.remove_edges_from(nx.selfloop_edges(H))
    return nx.to_graph6_bytes(H, header=False)


def _required_keys_for_conjecture(conjecture) -> Set[str]:
    # Score + violation only require a compact subset of invariants.
    return {"n", "Delta", "diam", conjecture.x_key, conjecture.y_key}


def search(
    conjecture,
    time_limit: float = 60.0,
    pop_size: int = 12,
    score_fn=None,
    warm_start: bool = False,
) -> SearchResult:
    return auto_select(
        conjecture,
        time_limit=time_limit,
        pop_size=pop_size,
        score_fn=score_fn,
        warm_start=warm_start,
    )


def auto_select(
    conjecture,
    time_limit: float = 60.0,
    pop_size: int = 12,
    score_fn=None,
    warm_start: bool = False,
    not_found_cost: float = 120.0,
) -> SearchResult:
    if score_fn is None:
        score_fn = heuristic_score

    start = time.perf_counter()
    stop_event = threading.Event()
    required_keys = _required_keys_for_conjecture(conjecture)
    eval_cache: Dict[bytes, Tuple[float, nx.Graph, Dict[str, float], float]] = {}

    warm = _warm_start_graphs(conjecture) if warm_start else []
    phase_one = min(30.0, time_limit)
    heuristics = [
        ("hill_climbing", hill_climbing),
        ("simulated_annealing", simulated_annealing),
        ("population_tournament", population_tournament),
        ("random_restart", random_restart),
        ("greedy_extremes", greedy_extremes),
        ("alns", alns),
    ]

    if conjecture.x_key in CONTINUOUS_KEYS or conjecture.y_key in CONTINUOUS_KEYS:
        heuristics.append(("numeric_gradient", numeric_gradient_search))

    results: List[SearchResult] = []
    with ThreadPoolExecutor(max_workers=len(heuristics)) as executor:
        futures = [
            executor.submit(fn, conjecture, phase_one, stop_event, pop_size, score_fn, warm, required_keys, eval_cache)
            for _, fn in heuristics
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                result.cost = result.elapsed if result.found else not_found_cost
                results.append(result)
                if result.found:
                    stop_event.set()
            except Exception:
                continue

    elapsed = time.perf_counter() - start
    best = _best_result(results, conjecture, not_found_cost)
    if best.found or elapsed >= time_limit:
        best.elapsed = min(time.perf_counter() - start, time_limit)
        best.cost = best.elapsed if best.found else not_found_cost
        if warm_start:
            _save_warm_start(conjecture, best)
        return best

    remaining = max(0.0, time_limit - elapsed)
    leaders = _top_heuristics(results, heuristics, k=2)
    stop_event = threading.Event()
    second_results: List[SearchResult] = []
    with ThreadPoolExecutor(max_workers=max(1, len(leaders))) as executor:
        futures = [
            executor.submit(fn, conjecture, remaining, stop_event, pop_size + 6, score_fn, warm, required_keys, eval_cache)
            for _, fn in leaders
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                result.cost = result.elapsed if result.found else not_found_cost
                second_results.append(result)
                if result.found:
                    stop_event.set()
            except Exception:
                continue

    all_results = results + second_results
    best = _best_result(all_results, conjecture, not_found_cost)
    best.elapsed = min(time.perf_counter() - start, time_limit)
    best.cost = best.elapsed if best.found else not_found_cost
    if warm_start:
        _save_warm_start(conjecture, best)
    return best


def hill_climbing(
    conjecture,
    time_limit,
    _stop_event=None,
    pop_size=12,
    score_fn=None,
    warm=None,
    required_keys=None,
    eval_cache=None,
):
    return _local_search(
        "hill_climbing",
        conjecture,
        time_limit,
        _stop_event,
        pop_size,
        score_fn,
        warm,
        "hill",
        required_keys,
        eval_cache,
    )


def simulated_annealing(
    conjecture,
    time_limit,
    _stop_event=None,
    pop_size=12,
    score_fn=None,
    warm=None,
    required_keys=None,
    eval_cache=None,
):
    return _local_search(
        "simulated_annealing",
        conjecture,
        time_limit,
        _stop_event,
        pop_size,
        score_fn,
        warm,
        "anneal",
        required_keys,
        eval_cache,
    )


def population_tournament(
    conjecture,
    time_limit,
    _stop_event=None,
    pop_size=12,
    score_fn=None,
    warm=None,
    required_keys=None,
    eval_cache=None,
):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size, score_fn, warm, required_keys, eval_cache)
    best = _best_scored(scored)
    if best and best[3] > 0:
        return _make_result(conjecture, True, best, start, time_limit, "population_tournament")

    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        if not scored:
            scored = _initial_scored(conjecture, pop_size, score_fn, warm, required_keys, eval_cache)
            continue
        pool = random.sample(scored, min(len(scored), max(2, min(6, len(scored)))))
        parent = max(pool, key=lambda item: item[0])[1]
        H = mutate(parent, conjecture)
        candidate = _evaluate(H, conjecture, score_fn, required_keys, eval_cache)
        if candidate is None:
            continue
        scored.append(candidate)
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = scored[: max(pop_size * 3, pop_size)]
        if candidate[3] > 0:
            return _make_result(conjecture, True, candidate, start, time_limit, "population_tournament")
        if best is None or candidate[0] > best[0]:
            best = candidate
    return _make_result(conjecture, False, best, start, time_limit, "population_tournament")


def random_restart(
    conjecture,
    time_limit,
    _stop_event=None,
    pop_size=12,
    score_fn=None,
    warm=None,
    required_keys=None,
    eval_cache=None,
):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    best = None
    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        scored = _initial_scored(conjecture, max(4, pop_size // 2), score_fn, warm, required_keys, eval_cache)
        local = _best_scored(scored)
        if local and (best is None or local[0] > best[0]):
            best = local
        if local and local[3] > 0:
            return _make_result(conjecture, True, local, start, time_limit, "random_restart")
        parent = local[1] if local else random.choice(generate_initial_population(conjecture, 1, warm))
        for _ in range(20):
            if _stop_event is not None and _stop_event.is_set():
                break
            cand = _evaluate(mutate(parent, conjecture), conjecture, score_fn, required_keys, eval_cache)
            if cand is None:
                continue
            parent = cand[1]
            if best is None or cand[0] > best[0]:
                best = cand
            if cand[3] > 0:
                return _make_result(conjecture, True, cand, start, time_limit, "random_restart")
    return _make_result(conjecture, False, best, start, time_limit, "random_restart")


def greedy_extremes(
    conjecture,
    time_limit,
    _stop_event=None,
    pop_size=12,
    score_fn=None,
    warm=None,
    required_keys=None,
    eval_cache=None,
):
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
            cand = _evaluate(mutate_named(G, conjecture, name), conjecture, score_fn, required_keys, eval_cache)
            if cand is None:
                continue
            if best is None or cand[0] > best[0]:
                best = cand
            if cand[3] > 0:
                return _make_result(conjecture, True, cand, start, time_limit, "greedy_extremes")
        if idx > len(candidates) * 3:
            candidates.extend(generate_initial_population(conjecture, pop_size, warm))
    return _make_result(conjecture, False, best, start, time_limit, "greedy_extremes")


def alns(
    conjecture,
    time_limit,
    _stop_event=None,
    pop_size=12,
    score_fn=None,
    warm=None,
    required_keys=None,
    eval_cache=None,
):
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
    scored = _initial_scored(conjecture, pop_size, score_fn, warm, required_keys, eval_cache)
    current = _best_scored(scored)
    best = current
    iteration = 0

    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        iteration += 1
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, required_keys, eval_cache))
            best = current if best is None else best
            continue
        name = _weighted_choice(weights)
        cand = _evaluate(mutate_named(current[1], conjecture, name), conjecture, score_fn, required_keys, eval_cache)
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
        if cand[3] > 0:
            return _make_result(conjecture, True, cand, start, time_limit, "alns")
        if iteration % 50 == 0:
            total = sum(weights.values()) or 1.0
            scale = len(weights) / total
            weights = {k: max(0.1, v * scale) for k, v in weights.items()}
    return _make_result(conjecture, False, best, start, time_limit, "alns")


def numeric_gradient_search(
    conjecture,
    time_limit,
    _stop_event=None,
    pop_size=12,
    score_fn=None,
    warm=None,
    required_keys=None,
    eval_cache=None,
):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size + 8, score_fn, warm, required_keys, eval_cache)
    current = _best_scored(scored)
    best = current
    probes = ["add_edge", "remove_edge", "subdivide_edge", "edge_contraction", "clique_expansion", "graph_product"]
    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, required_keys, eval_cache))
            continue
        probe_results = []
        for name in probes:
            cand = _evaluate(mutate_named(current[1], conjecture, name), conjecture, score_fn, required_keys, eval_cache)
            if cand is not None:
                probe_results.append((cand, name))
        if not probe_results:
            continue
        cand, _ = max(probe_results, key=lambda item: item[0][0])
        if cand[0] >= current[0] or random.random() < 0.05:
            current = cand
        if best is None or cand[0] > best[0]:
            best = cand
        if cand[3] > 0:
            return _make_result(conjecture, True, cand, start, time_limit, "numeric_gradient")
    return _make_result(conjecture, False, best, start, time_limit, "numeric_gradient")


def _local_search(
    name,
    conjecture,
    time_limit,
    _stop_event,
    pop_size,
    score_fn,
    warm,
    mode,
    required_keys=None,
    eval_cache=None,
):
    if score_fn is None:
        score_fn = heuristic_score
    start = time.perf_counter()
    scored = _initial_scored(conjecture, pop_size, score_fn, warm, required_keys, eval_cache)
    current = _best_scored(scored)
    best = current
    temp0 = 1.0
    iteration = 0
    while time.perf_counter() - start < time_limit:
        if _stop_event is not None and _stop_event.is_set():
            break
        iteration += 1
        if current is None:
            current = _best_scored(_initial_scored(conjecture, pop_size, score_fn, warm, required_keys, eval_cache))
            continue
        cand = _evaluate(mutate(current[1], conjecture), conjecture, score_fn, required_keys, eval_cache)
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
        if cand[3] > 0:
            return _make_result(conjecture, True, cand, start, time_limit, name)
    return _make_result(conjecture, False, best, start, time_limit, name)


def _evaluate(G, conjecture, score_fn, required_keys=None, eval_cache=None):
    try:
        H = repair(nx.Graph(G), conjecture)
        key = _state_key(H)
        if eval_cache is not None:
            cached = eval_cache.get(key)
            if cached is not None:
                return cached
        inv = compute_invariants(H, required_keys=required_keys)
        sc = score_fn(H, inv, conjecture)
        viol = conjecture.violation(inv)
        cand = (sc, H, inv, viol)
        if eval_cache is not None:
            eval_cache[key] = cand
        return cand
    except Exception:
        return None


def _initial_scored(conjecture, pop_size, score_fn, warm=None, required_keys=None, eval_cache=None):
    scored = []
    for G in generate_initial_population(conjecture, pop_size, warm):
        cand = _evaluate(G, conjecture, score_fn, required_keys, eval_cache)
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
    elif "claw_free" in conjecture.graph_class:
        for n in (5, 8, 12, 16):
            graphs.append(nx.convert_node_labels_to_integers(nx.line_graph(nx.cycle_graph(n))))
            graphs.append(nx.convert_node_labels_to_integers(nx.line_graph(nx.complete_graph(min(n, 8)))))
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
    return graphs or generate_initial_population(conjecture, 8, warm)


def _load_best_graphs():
    try:
        if not os.path.exists(BEST_GRAPHS_PATH):
            return {}
        with open(BEST_GRAPHS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _warm_start_graphs(conjecture) -> List[nx.Graph]:
    data = _load_best_graphs()
    entry = data.get(str(conjecture.id), {})
    graph6_values = entry.get("graphs", [])
    if isinstance(entry.get("graph6"), str):
        graph6_values.append(entry["graph6"])
    graphs = []
    for g6 in graph6_values[:5]:
        try:
            graphs.append(nx.from_graph6_bytes(str(g6).encode()))
        except Exception:
            continue
    return graphs


def _save_warm_start(conjecture, result: SearchResult) -> None:
    if result.graph is None:
        return
    try:
        os.makedirs(os.path.dirname(BEST_GRAPHS_PATH), exist_ok=True)
        data = _load_best_graphs()
        key = str(conjecture.id)
        g6 = result.graph6
        if not g6:
            return
        old = data.get(key, {})
        old_violation = float(old.get("violation", float("-inf")))
        old_graphs = old.get("graphs", [])
        if g6 not in old_graphs:
            old_graphs = [g6] + old_graphs
        if result.violation >= old_violation:
            data[key] = {
                "graph6": g6,
                "graphs": old_graphs[:5],
                "violation": float(result.violation),
                "score": float(result.score) if result.score != float("-inf") else 0.0,
                "found": bool(result.found),
                "heuristic": result.heuristic,
            }
        else:
            old["graphs"] = old_graphs[:5]
            data[key] = old
        with open(BEST_GRAPHS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
