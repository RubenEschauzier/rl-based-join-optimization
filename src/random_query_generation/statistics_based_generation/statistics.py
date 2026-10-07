"""Exact data statistics that steer generation (no optimizer, no model).

Per predicate p:  count |p|, distinct subjects V_s(p), distinct objects V_o(p).

Per predicate pair and join role, the EXACT size of the two-pattern join with fresh variables
(the join variable at the given positions, the other ends free), from entity x predicate count
matrices A[e, p] = #(e, p, ?) and B[e, p] = #(?, p, e):

    SS[p1, p2] = sum_e A[e,p1] A[e,p2]   (?x p1 ?a . ?x p2 ?b)
    SO[p1, p2] = sum_e A[e,p1] B[e,p2]   (?x p1 ?a . ?b p2 ?x)
    OO[p1, p2] = sum_e B[e,p1] B[e,p2]   (?a p1 ?x . ?b p2 ?x)

and its LIFT against the textbook independence (uniformity) estimate
|p1| |p2| / max(V(p1.x), V(p2.x)) -- the containment assumption most optimizers make. Lift
>> 1: the predicates co-occur on the join entity far more than uniformity says (correlation);
lift << 1: far less (anti-correlation). This is a property of the data, used to steer which
predicates a query combines (JOB's point: correlations are what makes estimation hard).

Per predicate and role, the frequency strata of constants: for a subject constant c of p the
pattern (c p ?) matches A[c, p] triples; quantiles of those counts over all c split constants
into tail (rare, very selective), middle and head (frequent) strata. Same for object constants.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROLES = ("SS", "SO", "OS", "OO")      # join role: positions of the shared entity in (pattern1, pattern2)
STRATA = ("tail", "middle", "head")


class Statistics:
    def __init__(self, counts, distinct_subjects, distinct_objects, joins, lifts, strata, degree):
        self.counts = counts                          # (P,)
        self.distinct_subjects = distinct_subjects    # (P,)
        self.distinct_objects = distinct_objects      # (P,)
        self.joins = joins                            # {role: (P, P) exact join sizes}
        self.lifts = lifts                            # {role: (P, P)}
        self.strata = strata                          # {"subject"|"object": (P, 2) quantile cut points}
        self.degree = degree                          # (E,) total degree per entity

    def lift(self, role, p1, p2):
        return float(self.lifts[role][p1, p2])

    def stratum(self, role, predicate, frequency):
        """'tail' / 'middle' / 'head' of a constant whose pattern matches `frequency` triples."""
        low, high = self.strata[role][predicate]
        return "tail" if frequency <= low else ("middle" if frequency <= high else "head")

    def save(self, directory):
        directory = Path(directory)
        np.savez(directory / "statistics.npz", counts=self.counts, distinct_subjects=self.distinct_subjects,
                 distinct_objects=self.distinct_objects, degree=self.degree,
                 strata_subject=self.strata["subject"], strata_object=self.strata["object"],
                 **{f"join_{r}": self.joins[r] for r in ROLES}, **{f"lift_{r}": self.lifts[r] for r in ROLES})

    @classmethod
    def load(cls, directory):
        data = np.load(Path(directory) / "statistics.npz")
        return cls(data["counts"], data["distinct_subjects"], data["distinct_objects"],
                   {r: data[f"join_{r}"] for r in ROLES}, {r: data[f"lift_{r}"] for r in ROLES},
                   {"subject": data["strata_subject"], "object": data["strata_object"]}, data["degree"])


def _quantile_cuts(matrix, quantiles):
    """Per column (predicate), quantiles of the nonzero entries."""
    matrix = matrix.tocsc()
    cuts = np.zeros((matrix.shape[1], len(quantiles)))
    for p in range(matrix.shape[1]):
        values = matrix.data[matrix.indptr[p]:matrix.indptr[p + 1]]
        if len(values):
            cuts[p] = np.quantile(values, quantiles)
    return cuts


def compute_statistics(graph, strata_quantiles=(0.5, 0.9)):
    start = time.perf_counter()
    n_entities, n_predicates = graph.n_entities, graph.n_predicates
    subjects = np.repeat(np.arange(n_entities), np.diff(graph.out_ptr))
    objects = np.repeat(np.arange(n_entities), np.diff(graph.in_ptr))
    ones = np.ones(graph.n_triples, dtype=np.float64)
    a = sp.csr_matrix((ones, (subjects, graph.out_p)), shape=(n_entities, n_predicates))   # duplicates summed
    b = sp.csr_matrix((ones, (objects, graph.in_p)), shape=(n_entities, n_predicates))
    counts = np.asarray(a.sum(axis=0)).ravel()
    distinct_subjects = np.asarray((a > 0).sum(axis=0)).ravel()
    distinct_objects = np.asarray((b > 0).sum(axis=0)).ravel()
    joins = {"SS": (a.T @ a).toarray(), "SO": (a.T @ b).toarray(), "OO": (b.T @ b).toarray()}
    joins["OS"] = joins["SO"].T.copy()
    distinct = {"S": distinct_subjects, "O": distinct_objects}
    lifts = {}
    for role in ROLES:
        ndv = np.maximum(distinct[role[0]][:, None], distinct[role[1]][None, :])
        independent = counts[:, None] * counts[None, :] / np.maximum(ndv, 1)
        lifts[role] = joins[role] / np.maximum(independent, 1e-12)
    strata = {"subject": _quantile_cuts(a, strata_quantiles), "object": _quantile_cuts(b, strata_quantiles)}
    degree = np.diff(graph.out_ptr) + np.diff(graph.in_ptr)
    print(f"Statistics: {n_predicates} predicates, join sizes and lifts for {len(ROLES)} roles "
          f"({time.perf_counter() - start:.0f}s)")
    return Statistics(counts, distinct_subjects, distinct_objects, joins, lifts, strata, degree)


def load_statistics(graph, cache_dir):
    cache = Path(cache_dir)
    if (cache / "statistics.npz").exists():
        return Statistics.load(cache)
    statistics = compute_statistics(graph)
    statistics.save(cache)
    summary = {"predicates": int(graph.n_predicates), "triples": int(graph.n_triples),
               "lift_quantiles": {role: np.quantile(statistics.lifts[role][statistics.joins[role] > 0],
                                                    [0.01, 0.1, 0.5, 0.9, 0.99]).round(4).tolist()
                                  for role in ROLES}}
    with open(cache / "statistics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return statistics
