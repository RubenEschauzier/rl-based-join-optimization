"""Cheap ground-truth certificates that a query's plan space is hard.

Instead of counting every connected subset, a candidate gets:

    the full result size              1 COUNT
    a few sampled left-deep plans      their prefixes only, ~ (n_plans) * (n - 1) COUNTs:
        `random_plans` uniformly random connected orders, plus one heuristic order (start
        from the smallest pattern, always add the smallest connected pattern next; exact
        single-pattern sizes from the data). The heuristic is only a CANDIDATE for a cheap
        plan, not an optimizer whose failures we select on.
    for hard-zero candidates           every connected pattern pair, 1 COUNT each

C_out of a plan = the sum of its prefix sizes (joins only; leaves cost 0, as in the p-error).
From the sampled plans:

    min_cost  <= C_out of the sample's cheapest plan, an UPPER bound on the optimum, so a
                 small min_cost certifies that the optimum is cheap;
    spread     = median sampled C_out / max(min_cost, 1): a conservative estimate of
                 median-plan / optimal-plan (the true optimum can only be cheaper).

Every plan scans every pattern once, so scan work is the same for all plans: it does not change
which plan is best, but it dilutes what a better plan saves. With scan_rows = sum of the
pattern sizes:

    best_plan_work     = scan_rows + min_cost: the work of the sample's cheapest plan;
    spread_with_scans  = (scan_rows + median_cost) / (scan_rows + min_cost): the spread of
                         the total work, near 1 when the scans dominate whatever the order.

Counts come from counting.ExactCounter (exact, in memory): directly for acyclic subsets (all of
them for star / path / tree / snowflake / path_star), by conditioning on a cycle-breaking
variable for cyclic ones. Cyclic subsets whose conditioning domain is too large, and subsets
with a pattern of more than `max_in_memory_rows` matches (message passing through a huge
predicate is slow and memory-hungry in numpy), are COUNTed on QLever, with a FORCED join order (smallest pattern first), so how long a count takes depends on
the data, not on QLever's own optimizer. A QLever count that fails or times out is unknown;
plans needing it are left out of the estimate (`n_unknown_plans`).
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import requests

from src.random_query_generation.statistics_based_generation.graph import pattern_terms
from src.supervised_value_estimation.amortized_dp.labels import mask_of
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import count_query


def pattern_adjacency(triple_patterns):
    """Pairs of patterns sharing a variable."""
    variables = [{t for t in pattern_terms(p) if t.startswith("?")} for p in triple_patterns]
    n = len(variables)
    return [[j for j in range(n) if j != i and variables[i] & variables[j]] for i in range(n)]


def random_order(adjacency, rng):
    n = len(adjacency)
    order = [int(rng.integers(0, n))]
    frontier = set(adjacency[order[0]])
    while len(order) < n:
        nxt = sorted(frontier - set(order))
        choice = nxt[int(rng.integers(0, len(nxt)))]
        order.append(choice)
        frontier |= set(adjacency[choice])
    return order


def smallest_first_order(adjacency, sizes, rng):
    n = len(adjacency)
    jitter = rng.random(n) * 1e-6
    order = [int(np.argmin(np.asarray(sizes, float) + jitter))]
    frontier = set(adjacency[order[0]])
    while len(order) < n:
        nxt = sorted(frontier - set(order))
        choice = min(nxt, key=lambda j: (sizes[j], jitter[j]))
        order.append(choice)
        frontier |= set(adjacency[choice])
    return order


def plan_orders(triple_patterns, sizes, n_random, rng):
    adjacency = pattern_adjacency(triple_patterns)
    orders = [smallest_first_order(adjacency, sizes, rng)]
    for _ in range(n_random):
        order = random_order(adjacency, rng)
        if order not in orders:
            orders.append(order)
    return orders


def connected_pairs(triple_patterns):
    adjacency = pattern_adjacency(triple_patterns)
    return sorted({mask_of((i, j)) for i in range(len(adjacency)) for j in adjacency[i]})


def certificate_masks(triple_patterns, orders, hard_zero=False):
    full = (1 << len(triple_patterns)) - 1
    masks = {full}
    for order in orders:
        masks.update(mask_of(order[:k]) for k in range(2, len(order) + 1))
    if hard_zero:
        masks.update(connected_pairs(triple_patterns))
    return sorted(masks)


def summarise(triple_patterns, orders, counts, hard_zero=False, sizes=None):
    """The certificate of one query from its counts {mask: rows or None} (and its pattern sizes,
    for the scan-inclusive fields)."""
    full = (1 << len(triple_patterns)) - 1
    costs, unknown, heuristic_cost, first_empty = [], 0, None, None
    for position, order in enumerate(orders):
        prefix = [counts.get(mask_of(order[:k])) for k in range(2, len(order) + 1)]
        empty_at = next((k for k, value in enumerate(prefix, start=2) if value == 0), None)
        if empty_at is not None:
            first_empty = empty_at if first_empty is None else min(first_empty, empty_at)
        if any(value is None for value in prefix):
            unknown += 1
            continue
        costs.append(int(sum(prefix)))
        if position == 0:                       # orders[0] is the smallest-first heuristic
            heuristic_cost = costs[-1]
    result = {"result": counts.get(full), "plan_costs": costs, "n_unknown_plans": unknown,
              "heuristic_cost": heuristic_cost,
              # smallest prefix size at which a sampled plan's intermediate result is empty
              "first_empty_prefix": first_empty}
    if costs:
        result["min_cost"] = int(min(costs))
        result["median_cost"] = float(np.median(costs))
        result["spread"] = float(result["median_cost"] / max(result["min_cost"], 1))
        if sizes is not None:
            scan = int(sum(sizes))
            result["scan_rows"] = scan
            result["best_plan_work"] = scan + result["min_cost"]
            result["spread_with_scans"] = float((scan + result["median_cost"]) / max(scan + result["min_cost"], 1))
        # > 1: some random order beat the smallest-first heuristic, i.e. the best plan is not
        # the obvious one. Recorded for analysis, never used to select.
        if heuristic_cost is not None:
            result["heuristic_gap"] = float(max(heuristic_cost, 1) / max(result["min_cost"], 1))
    if hard_zero:
        pairs = [counts.get(mask) for mask in connected_pairs(triple_patterns)]
        known = [value for value in pairs if value is not None]
        result["empty_pairs"] = sum(value == 0 for value in known) if len(known) == len(pairs) else None
        result["pairs_nonempty"] = result["empty_pairs"] == 0
    return result


def forced_count_query(triple_patterns, mask, sizes):
    """COUNT(*) of a connected subset, joined smallest pattern first (nested groups force
    QLever to join in that order, as QLeverOptimizerClient._apply_join_order does)."""
    members_ = [i for i in range(len(triple_patterns)) if mask >> i & 1]
    sub_patterns = [triple_patterns[i] for i in members_]
    order = smallest_first_order(pattern_adjacency(sub_patterns), [sizes[i] for i in members_],
                                 np.random.default_rng(0))

    def nest(indices):
        if len(indices) == 1:
            return f"{{ {sub_patterns[indices[0]]} . }}"
        return f"{{ {nest(indices[:-1])} {{ {sub_patterns[indices[-1]]} . }} }}"

    return f"SELECT (COUNT(*) AS ?count) WHERE {nest(order)}"


def certify_in_memory(candidate, structured, counter, n_random_plans, rng, max_empty_pairs=0,
                      max_in_memory_rows=None):
    """In the worker: orders, every count the in-memory counter can do (subsets whose patterns
    all match at most max_in_memory_rows triples), and the masks left for QLever. Returns False if the candidate is already rejected (a hard-zero candidate with more
    than max_empty_pairs empty pairs, or a non-empty acyclic result)."""
    patterns = candidate["triple_patterns"]
    orders = plan_orders(patterns, candidate["pattern_sizes"], n_random_plans, rng)
    hard_zero = candidate.get("hard_zero", False)
    counts, remaining = {}, []

    exact = getattr(counter, "count_exact", None)

    sizes = candidate["pattern_sizes"]

    def count(mask):
        subset = [i for i in range(len(patterns)) if mask >> i & 1]
        if max_in_memory_rows is not None and max(sizes[i] for i in subset) > max_in_memory_rows:
            remaining.append(mask)
            return
        if counter.is_acyclic(structured, subset):
            counts[mask] = int(round(counter.count(structured, subset)))
            return
        value = exact(structured, subset) if exact is not None else None
        if value is None:
            remaining.append(mask)            # too large to condition on: QLever counts it
        else:
            counts[mask] = int(round(value))

    if hard_zero:                     # cheap rejections first: pairs, then the full result
        empty = 0
        for mask in connected_pairs(patterns):
            count(mask)
            empty += counts.get(mask) == 0
            if empty > max_empty_pairs:
                return False
        full = (1 << len(patterns)) - 1
        count(full)
        if counts.get(full, 0) != 0:
            return False
    for mask in certificate_masks(patterns, orders, hard_zero):
        if mask not in counts and mask not in remaining:
            count(mask)
    candidate["orders"], candidate["counts"], candidate["qlever_masks"] = orders, counts, remaining
    return True


def finish_certificate(candidate, qlever_counts=None):
    """In the parent: summarise a candidate from its in-memory counts plus the QLever counts
    of its `qlever_masks` ({mask: rows or None}), and drop the raw counts."""
    qlever_counts = qlever_counts or {}
    values = {**candidate["counts"], **{mask: qlever_counts.get(mask) for mask in candidate["qlever_masks"]}}
    candidate["certificate"] = summarise(candidate["triple_patterns"], candidate["orders"], values,
                                         candidate.get("hard_zero", False), candidate["pattern_sizes"])
    candidate["certificate"]["qlever_counts"] = len(candidate["qlever_masks"])
    del candidate["counts"], candidate["qlever_masks"]
    return candidate


def certificate_tasks(candidate, timeout_s):
    """The candidate's subsets the in-memory counter left for QLever, with a forced order."""
    return [(mask, forced_count_query(candidate["triple_patterns"], mask, candidate["pattern_sizes"]), timeout_s)
            for mask in candidate["qlever_masks"]]


def final_count_task(candidate, timeout_s):
    """The plain COUNT(*) of the whole query, planned by QLever itself: the source of `y`."""
    full = (1 << len(candidate["triple_patterns"])) - 1
    return [("result", count_query(candidate["triple_patterns"], full), timeout_s)]


def count_with_reason(session, endpoint, query, timeout_s):
    """(rows, None) or (None, why): QLever answers a timeout with HTTP 429 "Operation timed out" and a query
    over its memory limit with HTTP 500 "Tried to allocate ...". Why is timeout / memory / client_timeout /
    connection / http_<status> / bad_reply."""
    try:
        response = session.post(endpoint, params={"timeout": f"{int(timeout_s)}s"}, data=query,
                                headers={"Accept": "application/sparql-results+json",
                                         "Content-Type": "application/sparql-query"},
                                timeout=timeout_s + 30)
    except requests.Timeout:
        return None, "client_timeout"
    except requests.RequestException:
        return None, "connection"
    if response.status_code != 200:
        text = response.text[:2000]
        if response.status_code == 429 or "timed out" in text:
            return None, "timeout"
        if "allocate" in text:
            return None, "memory"
        return None, f"http_{response.status_code}"
    try:
        bindings = response.json().get("results", {}).get("bindings", [])
        return (int(bindings[0]["count"]["value"]) if bindings else 0), None
    except (ValueError, KeyError, IndexError):
        return None, "bad_reply"


class BackgroundCounter:
    """COUNTs on QLever in background threads while the workers keep generating: one thread per
    endpoint with one request in flight, so each single-core instance runs one count at a time.
    `submit(candidate, tasks, tag)` counts tasks [(key, sparql, timeout_s)]; when all are done
    the candidate is put on `done` as (tag, candidate, {key: rows or None}), None for a count
    that failed or timed out. Each count's seconds, failure reason and lane go in the candidate's
    "count_log" {key: {"seconds", "error", "lane"}} (seconds summed over both lanes).

    Two lanes (deploy_isolated_qlever_instances.py, QLEVER_BIG_MEMORY_MB): `endpoints` are instances
    with a small memory limit, `big_endpoints` instances with a large one. A count that fails on
    memory in the small lane is retried in the big lane instead of being lost: the queries whose
    intermediate results do not fit are the hard ones, and dropping them would bias the workload. The
    instances' limits add up to less than the node's RAM, so no mix of queries can exhaust it. A big
    lane thread takes retries first and helps with the small lane's queue when it has none."""

    POLL_S = 0.05

    def __init__(self, endpoints, done, big_endpoints=()):
        self.done = done
        self.work, self.retry, self.lock = queue.Queue(), queue.Queue(), threading.Lock()
        self.pending, self.counted, self.seconds = 0, 0, 0.0
        self.escalated, self.escalated_ok = 0, 0
        self.stopping = threading.Event()
        big = list(dict.fromkeys(big_endpoints))
        small = [e for e in dict.fromkeys(endpoints) if e not in big]
        self.has_big_lane = bool(big)
        self.threads = ([threading.Thread(target=self._worker, args=(e,), daemon=True) for e in small]
                        + [threading.Thread(target=self._big_worker, args=(e,), daemon=True) for e in big])
        self.n_small = len(small)
        for thread in self.threads:
            thread.start()

    def submit(self, candidate, tasks, tag):
        state = {"candidate": candidate, "left": len(tasks), "counts": {}, "tag": tag}
        with self.lock:
            self.pending += len(tasks)
        for key, query, timeout_s in tasks:
            self.work.put((state, key, query, timeout_s))

    def _run(self, session, endpoint, item, lane):
        state, key, query, timeout_s = item
        start = time.perf_counter()
        value, error = count_with_reason(session, endpoint, query, timeout_s)
        seconds = time.perf_counter() - start
        escalate = error == "memory" and lane == "small" and self.has_big_lane
        with self.lock:
            self.counted += 1
            self.seconds += seconds
            log = state["candidate"].setdefault("count_log", {})
            previous = log.get(key, {}).get("seconds", 0.0)
            log[key] = {"seconds": round(previous + seconds, 3), "error": error, "lane": lane}
            if lane == "big" and key in state.setdefault("escalated", set()):
                self.escalated_ok += value is not None
            if escalate:
                self.escalated += 1
                state.setdefault("escalated", set()).add(key)
            else:
                self.pending -= 1
                state["counts"][key] = value
                state["left"] -= 1
            finished = state["left"] == 0
        if escalate:
            self.retry.put(item)
        elif finished:
            self.done.put((state["tag"], state["candidate"], state["counts"]))

    def _worker(self, endpoint):
        session = requests.Session()
        while True:
            item = self.work.get()
            if item is None:
                return
            self._run(session, endpoint, item, "small")

    def _big_worker(self, endpoint):
        session = requests.Session()
        while not self.stopping.is_set():
            try:
                item = self.retry.get_nowait()
            except queue.Empty:
                try:
                    item = self.work.get(timeout=self.POLL_S)
                except queue.Empty:
                    continue
                if item is None:             # a small thread's stop signal: leave it for them
                    self.work.put(None)
                    time.sleep(self.POLL_S)
                    continue
            self._run(session, endpoint, item, "big")

    def close(self):
        self.stopping.set()
        for _ in range(self.n_small):
            self.work.put(None)
