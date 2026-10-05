"""Plan quality against TRUE cardinalities: C_out, the optimal left-deep plan, and the p-error.

C_out follows the p-error definition (Moerkotte et al.): the cost of a join is its output
size plus the cost of its inputs, leaves (triple patterns) cost 0, so C_out is the sum of all
join output cardinalities. For a left-deep order o1..on that is
    C_out = sum over k = 2..n of |result of {o1, ..., ok}|.
(This differs from labels.plan_log_cost, which also counts the smaller first scan; that one
is the training objective, this one is the reported metric.)

    p-error(Q) = C_out(chosen plan, true cards) / C_out(optimal plan, true cards)

The optimum is the cheapest left-deep plan without cartesian products (exact DP over the
connected subsets). A bushy plan (QLever's own optimizer may choose one) can be cheaper than
every left-deep plan, so its p-error can be below 1. Empty results make C_out 0; both sides
are floored at 1, which leaves the ratio unchanged whenever the optimum is >= 1.

`rows` maps a subset bitmask to its exact row count (None = unknown, e.g. its count query
timed out); costs that need an unknown count are None.
"""
from __future__ import annotations

import math

from src.supervised_value_estimation.amortized_dp.labels import QueryLabels, mask_of
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import JOIN_PREFIXES, _finished


def join_cout(plan, rows):
    """C_out of a left-deep order: the sum of every prefix's row count (prefixes of 2+)."""
    plan = [int(p) for p in plan]
    total = 0
    for size in range(2, len(plan) + 1):
        value = rows.get(mask_of(plan[:size]))
        if value is None:
            return None
        total += value
    return total


def optimal_join_cout(labels: QueryLabels, rows):
    """Exact DP over connected subsets: the lowest C_out of any left-deep plan without
    cartesian products, and one plan achieving it. (None, None) if a needed count is unknown."""
    if labels.n_tp < 2:
        return 0, list(range(labels.n_tp))
    connected = [mask for mask in rows if mask & (mask - 1)]          # subsets of 2+ patterns
    if any(rows[mask] is None for mask in connected):
        return None, None
    remaining = {labels.full_mask: 0}                                 # min C_out still to pay
    best_child = {}
    for mask in sorted(connected, key=lambda m: -m.bit_count()):
        if mask == labels.full_mask:
            continue
        options = [(rows[child] + remaining[child], child) for child in labels.children(mask)
                   if child in remaining]
        if options:
            remaining[mask], best_child[mask] = min(options)
    pairs = [(rows[mask_of(p)] + remaining[mask_of(p)], p) for p in labels.connected_pairs()
             if mask_of(p) in remaining]
    if not pairs:
        return None, None
    cost, (i, j) = min(pairs)
    order, mask = [i, j], mask_of((i, j))
    while mask != labels.full_mask:
        child = best_child[mask]
        order.append((child ^ mask).bit_length() - 1)
        mask = child
    return cost, order


def tree_join_cout(tree):
    """C_out of whatever plan QLever executed (bushy or not), read from its runtime tree: the
    sum of result_rows over all join nodes. None if a join did not finish (e.g. timeout)."""
    total, stack, n_joins = 0, [tree], 0
    while stack:
        node = stack.pop()
        if not node:
            continue
        stack.extend(node.get("children", []) or [])
        if str(node.get("description", "")).startswith(JOIN_PREFIXES):
            if not _finished(node) or node.get("result_rows") is None:
                return None
            total += int(node["result_rows"])
            n_joins += 1
    return total if n_joins else None


def p_error(cost, optimum):
    if cost is None or optimum is None:
        return None
    return max(cost, 1) / max(optimum, 1)


def ratio_summary(values):
    """Distribution of ratios (p-errors or C_out ratios); None entries are skipped."""
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"n": 0}
    values = sorted(values)

    def quantile(q):
        return values[min(len(values) - 1, int(math.ceil(q * len(values))) - 1)]

    return {"n": len(values),
            "geomean": math.exp(sum(math.log(v) for v in values) / len(values)),
            "median": quantile(0.5), "p90": quantile(0.9), "p95": quantile(0.95), "p99": quantile(0.99),
            "max": values[-1], "mean": sum(values) / len(values),
            "frac_at_most_1": sum(v <= 1 + 1e-9 for v in values) / len(values),
            "frac_within_2x": sum(v <= 2 for v in values) / len(values)}
