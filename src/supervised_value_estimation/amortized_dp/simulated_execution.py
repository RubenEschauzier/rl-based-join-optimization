"""Drop-in execution strategy for multiprocess_validate_agent that needs no database.

It "executes" a plan by pricing it with the oracle cardinalities (C_out, as JoinPlan does)
and compares it with the exact left-deep DP optimum under the same cardinalities. That is
an exact regret, which a real engine run cannot give: nobody knows the optimal plan there.

Each result keeps the runner's contract ({"rl_metrics": {"is_error", "latency",
"total_cost"}}) -- with "latency" set to the cost ratio to the optimum -- and adds the
per-query fields summarize() uses.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.supervised_value_estimation.amortized_dp.labels import optimal_left_deep_plan, plan_log_cost


def _query_string(query):
    value = query.query
    return value if isinstance(value, str) else value[0]


class SimulatedCostExecutionStrategy:
    default_timeout_s = 0.0

    def __init__(self, labels_by_query):
        self.labels_by_query = labels_by_query
        self._optimal = {}

    def setup(self):
        pass

    def teardown(self):
        pass

    def optimal_log_cost(self, query):
        if query not in self._optimal:
            self._optimal[query] = optimal_left_deep_plan(self.labels_by_query[query])[1]
        return self._optimal[query]

    def score(self, query, order):
        labels = self.labels_by_query[query]
        complete = sorted(int(i) for i in order) == list(range(labels.n_tp))
        log_cost = plan_log_cost(order, labels) if complete else math.inf
        optimal = self.optimal_log_cost(query)
        log_ratio = log_cost - optimal
        return {
            "rl_metrics": {"is_error": not complete, "latency": math.exp(min(log_ratio, 700.0)),
                           "total_cost": log_cost},
            "query": query, "plan": [int(i) for i in order], "n_tp": labels.n_tp,
            "family": labels.family, "log_cost": log_cost, "log_optimal_cost": optimal,
            "log_ratio": log_ratio,
        }

    def execute(self, execution_plans):
        return [self.score(_query_string(item["query"]), item["plan"]["plan"]) for item in execution_plans]


def _ratio_statistics(log_ratios):
    log_ratios = np.asarray([r for r in log_ratios if math.isfinite(r)], dtype=float)
    if len(log_ratios) == 0:
        return {"n": 0}
    ratios = np.exp(log_ratios)
    return {
        "n": int(len(log_ratios)),
        "geomean_ratio": float(np.exp(log_ratios.mean())),
        "mean_ratio": float(ratios.mean()),
        "median_ratio": float(np.median(ratios)),
        "p90_ratio": float(np.percentile(ratios, 90)),
        "p95_ratio": float(np.percentile(ratios, 95)),
        "p99_ratio": float(np.percentile(ratios, 99)),
        "max_ratio": float(ratios.max()),
        "frac_optimal": float(np.mean(log_ratios <= 1e-9)),
        "frac_within_1.1x": float(np.mean(ratios <= 1.1)),
        "frac_within_2x": float(np.mean(ratios <= 2.0)),
    }


def summarize(results, planning_metrics=None):
    """Cost-ratio statistics overall, per query size and per query family."""
    summary = {"overall": _ratio_statistics([r["log_ratio"] for r in results]),
               "n_incomplete_plans": int(sum(r["rl_metrics"]["is_error"] for r in results))}
    for key in ("n_tp", "family"):
        groups = defaultdict(list)
        for r in results:
            groups[r[key]].append(r["log_ratio"])
        summary[f"by_{key}"] = {str(k): _ratio_statistics(v) for k, v in sorted(groups.items())}
    if planning_metrics:
        summary["planning"] = {k: float(v) for k, v in planning_metrics.items() if k.startswith("planning_time")}
    return summary
