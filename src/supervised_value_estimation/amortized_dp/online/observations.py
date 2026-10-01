"""Per-query store of everything real executions have revealed, and the targets built from it.

Only observations go in; model predictions never do. Per query:

    logcard      log1p(rows) of every intermediate and scan seen in any executed plan.
                 log1p, not log: real intermediates can be empty.
    step_ms      (parent set, child set) -> fastest observed time of the join step that built
                 the child from the parent (parent 0 = the first join, which includes both
                 scans). Order-dependent effects (sortedness) are ignored: a set-based
                 approximation, which is what makes latency stitchable at all.
    executions   every executed plan with its latency, whether it was censored, and the
                 timeout it ran under.

Targets:
    C_out cost-to-go    DP over observed cardinalities (labels.cost_to_go on this table)
    latency cost-to-go  DP over observed step times: the fastest remaining time assembled
                        from observed steps, possibly across different executions. Like the
                        C_out target it is an upper bound on the true cost-to-go.
"""
from __future__ import annotations

import math

from src.supervised_value_estimation.amortized_dp.labels import QueryLabels, cost_to_go, mask_of


class QueryObservations:
    def __init__(self, query: str, family: str, n_tp: int, neighbour_masks: list[int]):
        self.labels = QueryLabels(query=query, family=family, n_tp=n_tp, neighbour_masks=neighbour_masks, logcard={})
        self.step_ms: dict[tuple[int, int], float] = {}
        self.executions: list[dict] = []

    @property
    def logcard(self):
        return self.labels.logcard

    def add(self, plan, parsed, latency_s, censored, timeout_s, cached=False):
        """Record one execution of `plan` (parsed by runtime_tree.parse_execution)."""
        plan = [int(p) for p in plan]
        for pattern, rows in parsed["scan_rows"].items():
            self.logcard[1 << pattern] = math.log1p(rows)
        for size, rows in parsed["prefix_rows"].items():
            self.logcard[mask_of(plan[:size])] = math.log1p(rows)
        for size, milliseconds in parsed["step_ms"].items():
            parent = 0 if size == 2 else mask_of(plan[:size - 1])
            key = (parent, mask_of(plan[:size]))
            self.step_ms[key] = min(self.step_ms.get(key, math.inf), milliseconds)
        self.executions.append({"plan": plan, "latency_s": latency_s, "censored": censored,
                                "timeout_s": timeout_s, "cached": cached, "aligned": parsed["aligned"]})

    @property
    def best_latency_s(self):
        """Fastest uncensored, uncached execution so far (None if there is none)."""
        latencies = [e["latency_s"] for e in self.executions if not e["censored"] and not e["cached"]]
        return min(latencies) if latencies else None

    def cost_to_go(self):
        """log C_out cost-to-go from observed cardinalities (log1p rows)."""
        return cost_to_go(self.labels, self.logcard)

    def latency_to_go(self):
        """log1p of the fastest known remaining milliseconds, for every set with a known
        completion; the full set has 0 remaining (log1p(0) = 0)."""
        remaining = {self.labels.full_mask: 0.0}
        outgoing = {}
        for (parent, child), milliseconds in self.step_ms.items():
            if parent:
                outgoing.setdefault(parent, []).append((child, milliseconds))
        for mask in sorted(outgoing, key=lambda m: -m.bit_count()):
            options = [ms + remaining[child] for child, ms in outgoing[mask] if child in remaining]
            if options:
                remaining[mask] = min(options)
        return {mask: math.log1p(ms) for mask, ms in remaining.items()}

    def best_stitched_latency_ms(self):
        """Fastest complete plan assembled from observed steps (inf if none)."""
        remaining = self.latency_to_go()
        best = math.inf
        for (parent, pair), milliseconds in self.step_ms.items():
            if parent == 0 and pair in remaining:
                best = min(best, milliseconds + math.expm1(remaining[pair]))
        return best
