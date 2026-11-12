"""Count-free estimates of a candidate query, for generation.method = estimated (generate.py).

Instead of exact counts (tighten_to_budget, certify_in_memory), a candidate's result size and its sampled
plans' costs are ESTIMATED with the textbook independence estimate:

    est(S) = prod_i |tp_i| * prod_v ( min_i d_i(v) / prod_i d_i(v) )

over the patterns i of S and their join variables v, with |tp_i| the pattern's exact size and d_i(v) the
distinct values of v in pattern i: |tp_i| next to a constant (the graph has no duplicate triples), else the
predicate's distinct subjects / objects. The estimate only picks the spread class (bucket) and screens the
work budget; every kept query's `y` is still QLever's exact COUNT, which also decides hard zeros. Measured on
the certified WatDiv workload (experiments/experiment_outputs/statistics_based_watdiv_v1/work_2026-11-12):
a 25/35/40 target on estimated spread classes gives a true mix of about 19/37/44.
"""
from __future__ import annotations

import numpy as np

from src.random_query_generation.statistics_based_generation.instantiate import (
    BindingPlan, Instance, _natural_stratum, cycle_nodes, variable_connected,
)
from src.supervised_value_estimation.amortized_dp.labels import mask_of


class Estimator:
    def __init__(self, distinct_subjects, distinct_objects):
        self.distinct_subjects = np.asarray(distinct_subjects, dtype=np.float64)
        self.distinct_objects = np.asarray(distinct_objects, dtype=np.float64)

    @classmethod
    def from_index(cls, index):
        """From counting.PredicateIndex's distinct subjects / objects per predicate."""
        return cls(index.distinct[True], index.distinct[False])

    def cardinality(self, structured, sizes, members):
        """est of the patterns `members` of `structured` [(s, p, o)], terms ("v", node) / ("c", entity)."""
        value, seen = 1.0, {}
        for i in members:
            s, p, o = structured[i]
            size = float(sizes[i])
            value *= size
            if s[0] == "v":
                seen.setdefault(s[1], []).append(size if o[0] == "c" else self.distinct_subjects[p])
            if o[0] == "v":
                seen.setdefault(o[1], []).append(size if s[0] == "c" else self.distinct_objects[p])
        for ds in seen.values():
            if len(ds) > 1:
                ds = np.maximum(np.asarray(ds), 1.0)
                value *= ds.min() / np.prod(ds)
        return value

    def certificate(self, structured, sizes, orders):
        """The certificate's fields (certificate.summarise), estimated: C_out of each sampled order is the
        sum of its prefixes' estimates."""
        cache = {}

        def est(members):
            key = mask_of(members)
            if key not in cache:
                cache[key] = self.cardinality(structured, sizes, members)
            return cache[key]

        costs = [sum(est(order[:k]) for k in range(2, len(order) + 1)) for order in orders]
        scan = float(sum(sizes))
        low, median = min(costs), float(np.median(costs))
        return {"result": est(range(len(structured))), "plan_costs": costs, "min_cost": low,
                "median_cost": median, "spread": median / max(low, 1.0), "scan_rows": scan,
                "best_plan_work": scan + low, "spread_with_scans": (scan + median) / max(scan + low, 1.0)}


def tighten_estimated(instance, graph, statistics, estimator, budget, rng, tries_per_step=3):
    """instantiate.tighten_to_budget with the ESTIMATED result instead of exact counts: bind more nodes (with
    their witness entities, so the query stays non-empty) until the estimate is at most `budget`; each step
    tries a few bindable nodes (leaves first) and keeps the least selective one that fits, else the one
    closest from above. None if no bindable node is left. The exact result is QLever's final count."""
    on_cycle = cycle_nodes(instance.shape)
    degree = instance.shape.degree()

    def estimate(variant):
        structured = variant.structured_patterns()
        return estimator.cardinality(structured, variant.pattern_sizes(graph, statistics), range(len(structured)))

    bound = estimate(instance)
    while bound > budget:
        free = [node for node in range(instance.shape.n_nodes)
                if node not in instance.plan.constants and node not in on_cycle
                and variable_connected(instance.shape, {**instance.plan.constants, node: None})]
        if not free:
            return None
        leaves = [node for node in free if degree[node] == 1]
        pool = leaves or free
        tried = []
        for node in rng.choice(pool, min(tries_per_step, len(pool)), replace=False):
            node = int(node)
            plan = BindingPlan({**instance.plan.constants, node: _natural_stratum(instance, graph, statistics, node)})
            variant = Instance(instance.shape, instance.entities, instance.predicates, plan, instance.oriented)
            tried.append((estimate(variant), variant))
        under = [t for t in tried if t[0] <= budget]
        bound, instance = max(under, key=lambda t: t[0]) if under else min(tried, key=lambda t: t[0])
    return instance


class PairChecker:
    """Whether two patterns that share a variable have a match, without counting.PredicateIndex (the estimated
    method's memory saving: 3.5 GB less on WatDiv): the stand-in for the counter in HardZeroMaker's pair checks.

    count(patterns, (i, j)) is 0 for an empty pair and positive otherwise (not the size):
      one shared variable, no constant      the exact join size from the statistics (statistics.joins)
      one shared variable, a constant        the constant's neighbours checked one by one (early exit)
      two shared variables                   None (unknown; HardZeroMaker counts it as non-empty)."""

    def __init__(self, graph, statistics, max_checks=200_000):
        self.graph, self.statistics, self.max_checks = graph, statistics, max_checks

    def count(self, patterns, members):
        i, j = members
        ends = []
        for s, p, o in (patterns[i], patterns[j]):
            ends.append({"s": s, "o": o, "p": int(p)})
        shared = ({t[1] for t in (ends[0]["s"], ends[0]["o"]) if t[0] == "v"}
                  & {t[1] for t in (ends[1]["s"], ends[1]["o"]) if t[0] == "v"})
        if len(shared) != 1:
            return None
        x = ("v", shared.pop())
        role, other = "", []
        for e in ends:
            at_subject = e["s"] == x
            role += "S" if at_subject else "O"
            other.append(e["o"] if at_subject else e["s"])
        if other[0][0] == "v" and other[1][0] == "v":
            return float(self.statistics.joins[role][ends[0]["p"], ends[1]["p"]])
        k = 0 if other[0][0] == "c" else 1                         # the pattern with a constant end
        e, c = ends[k], other[k][1]
        # x's values under that pattern: subjects of (x p c) or objects of (c p x)
        domain = self.graph.subjects_of(c, e["p"]) if role[k] == "S" else self.graph.objects_of(c, e["p"])
        f, rest = ends[1 - k], other[1 - k]
        for value in domain[: self.max_checks]:
            value = int(value)
            if role[1 - k] == "S":
                hit = (self.graph.has_triple(value, f["p"], rest[1]) if rest[0] == "c"
                       else self.graph.count_bound_subject(value, f["p"]) > 0)
            else:
                hit = (self.graph.has_triple(rest[1], f["p"], value) if rest[0] == "c"
                       else self.graph.count_bound_object(value, f["p"]) > 0)
            if hit:
                return 1.0
        return 0.0 if len(domain) <= self.max_checks else None


def predicate_pools(graph):
    """(ptr, subjects, objects): every triple grouped by predicate (subjects / objects as int32), what
    HardZeroMaker samples from, built one predicate at a time (no argsort of all triples)."""
    P = graph.n_predicates
    ptr = np.zeros(P + 1, dtype=np.int64)
    np.cumsum(np.bincount(graph.out_p, minlength=P), out=ptr[1:])
    subject_of_edge = np.repeat(np.arange(graph.n_entities, dtype=np.int32), np.diff(graph.out_ptr))
    subjects = np.empty(len(graph.out_p), dtype=np.int32)
    objects = np.empty(len(graph.out_p), dtype=np.int32)
    for p in range(P):
        edges = np.flatnonzero(graph.out_p == p)
        subjects[ptr[p]:ptr[p + 1]] = subject_of_edge[edges]
        objects[ptr[p]:ptr[p + 1]] = graph.out_o[edges]
    return ptr, subjects, objects
