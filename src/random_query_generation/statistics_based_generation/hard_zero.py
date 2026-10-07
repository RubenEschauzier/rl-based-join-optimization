"""Hard zeros: queries with an empty result whose patterns, and connected pattern pairs, are
all non-empty, so the emptiness only shows up after joining (anti-correlation, not a
non-existent constant). A good plan reaches an empty intermediate early; a bad one builds
large intermediates first.

A hard-zero candidate is a non-empty instance with ONE perturbation:
    constant    a constant replaced by another value of the same predicate in the same
                position and frequency stratum (a sensible value: it occurs with that predicate);
    predicate   a predicate between two variables replaced by one that joins with every
                neighbouring predicate (exact pair join size > 0, statistics.py) but with low
                lift (weights lift^-gamma), within the pattern-size cap.
For a constant whose variable neighbour x has other patterns that are all leaves (stars, and
star-like parts), the replacement is CONSTRUCTED rather than drawn: with S_i the entities x can
take under each other pattern and T their intersection (non-empty, the instance has a solution),
the new constant c' must give a set that meets every S_i but misses T. Then every pair is
non-empty and the result empty by construction (three sets can intersect pairwise but not
jointly). Otherwise candidates are drawn and kept if at most max_empty_pairs pairs are empty.
Single patterns stay non-empty by construction; the certificate (certificate.py) then verifies
the empty result and the empty-pair count.
"""
from __future__ import annotations

import copy

import numpy as np


class HardZeroMaker:
    def __init__(self, graph, statistics, max_pattern_matches, anticorrelation_gamma=1.0, counter=None,
                 max_empty_pairs=0):
        self.graph, self.statistics = graph, statistics
        self.max_empty_pairs = max_empty_pairs
        self.counter = counter          # counting.AcyclicCounter: pair checks before a swap is kept
        self.max_pattern_matches = max_pattern_matches
        self.gamma = anticorrelation_gamma
        # every triple, grouped by predicate, to sample a predicate's subjects / objects
        subjects = np.repeat(np.arange(graph.n_entities, dtype=np.int32), np.diff(graph.out_ptr))
        order = np.argsort(graph.out_p, kind="stable")
        self.by_predicate_subject = subjects[order]
        self.by_predicate_object = graph.out_o[order]
        self.predicate_ptr = np.zeros(graph.n_predicates + 1, dtype=np.int64)
        np.cumsum(np.bincount(graph.out_p, minlength=graph.n_predicates), out=self.predicate_ptr[1:])

    def _incident(self, instance, node):
        return [(i, int(instance.predicates[i]), instance.oriented[i][0] == node)
                for i in range(len(instance.oriented)) if node in instance.oriented[i]]

    def _domain(self, instance, edge, variable):
        """Entities `variable` can take under pattern `edge` alone, if the pattern's other end is
        a constant or a variable that occurs in no other pattern (a leaf); else None."""
        u, v = instance.oriented[edge]
        other = v if u == variable else u
        p = int(instance.predicates[edge])
        variable_is_subject = u == variable
        if other in instance.plan.constants:
            c = int(instance.entities[other])
            found = self.graph.subjects_of(c, p) if variable_is_subject else self.graph.objects_of(c, p)
            return np.unique(found)
        if sum(other in e for e in instance.oriented) == 1:
            a, b = self.predicate_ptr[p], self.predicate_ptr[p + 1]
            pool = self.by_predicate_subject if variable_is_subject else self.by_predicate_object
            return np.unique(pool[a:b])
        return None

    def _split_constant(self, instance, node, rng, attempts=60):
        """Constructed hard zero at a star-like variable (see the module docstring), or None."""
        incident = self._incident(instance, node)
        if len(incident) != 1:
            return None
        edge, predicate, constant_is_subject = incident[0]
        u, v = instance.oriented[edge]
        x = v if u == node else u
        others = [i for i, e in enumerate(instance.oriented) if x in e and i != edge]
        if len(others) < 2:
            return None
        domains = [self._domain(instance, i, x) for i in others]
        if any(d is None or len(d) == 0 for d in domains):
            return None
        joint = domains[0]
        for d in domains[1:]:
            joint = np.intersect1d(joint, d, assume_unique=True)
        if len(joint) == 0:
            return None
        used = set(int(e) for e in instance.entities)
        for _ in range(attempts):
            # c' through an entity that satisfies some other pattern but not all of them
            d = domains[int(rng.integers(0, len(domains)))]
            outside = np.setdiff1d(d, joint, assume_unique=True)
            if len(outside) == 0:
                continue
            e = int(outside[int(rng.integers(0, len(outside)))])
            # (x p c'): c' is an object of e; (c' p x): c' is a subject of e
            options = (self.graph.objects_of(e, predicate) if not constant_is_subject
                       else self.graph.subjects_of(e, predicate))
            if len(options) == 0:
                continue
            candidate = int(options[int(rng.integers(0, len(options)))])
            if candidate in used:
                continue
            new_domain = np.unique(self.graph.objects_of(candidate, predicate) if constant_is_subject
                                   else self.graph.subjects_of(candidate, predicate))
            if not 0 < len(new_domain) <= self.max_pattern_matches:
                continue
            if len(np.intersect1d(new_domain, joint, assume_unique=True)):
                continue                      # some x satisfies everything: not empty
            if all(len(np.intersect1d(new_domain, d_i, assume_unique=True)) for d_i in domains):
                variant = copy.deepcopy(instance)
                variant.entities[node] = candidate
                return variant
        return None

    def _swap_constant(self, instance, node, rng, attempts=30):
        constructed = self._split_constant(instance, node, rng)
        if constructed is not None:
            return constructed
        incident = self._incident(instance, node)
        _, predicate, is_subject = incident[0]
        a, b = self.predicate_ptr[predicate], self.predicate_ptr[predicate + 1]
        pool = self.by_predicate_subject if is_subject else self.by_predicate_object
        stratum = instance.plan.constants[node]
        used = set(int(e) for e in instance.entities)
        for _ in range(attempts):
            candidate = int(pool[a + int(rng.integers(0, b - a))])
            if candidate in used:
                continue
            ok = True
            for _, p, subject in incident:                     # every incident pattern non-empty
                size = (self.graph.count_bound_subject(candidate, p) if subject
                        else self.graph.count_bound_object(candidate, p))
                ok &= 0 < size <= self.max_pattern_matches
            frequency = (self.graph.count_bound_subject(candidate, predicate) if is_subject
                         else self.graph.count_bound_object(candidate, predicate))
            role = "subject" if is_subject else "object"
            if ok and self.statistics.stratum(role, predicate, frequency) == stratum:
                variant = copy.deepcopy(instance)
                variant.entities[node] = candidate
                if self._empty_pairs(variant, [i for i, _, _ in incident]) <= self.max_empty_pairs:
                    return variant
        return None

    def _empty_pairs(self, instance, edges):
        """How many pairs of patterns sharing a variable with one of `edges` are empty."""
        if self.counter is None:
            return 0
        patterns = instance.structured_patterns()
        variables = [{t[1] for t in (s, o) if t[0] == "v"} for s, _, o in patterns]
        pairs = {tuple(sorted((i, j))) for i in edges for j in range(len(patterns))
                 if j != i and variables[i] & variables[j]}
        return sum(self.counter.count(patterns, list(pair)) == 0 for pair in pairs)

    def _swap_predicate(self, instance, edge, rng):
        u, v = instance.oriented[edge]
        allowed = (self.statistics.counts <= self.max_pattern_matches) & (self.statistics.counts > 0)
        allowed[instance.predicates[edge]] = False
        log_lift = np.zeros(self.graph.n_predicates)
        for node, node_is_subject in ((u, True), (v, False)):
            for other, p, other_subject in self._incident(instance, node):
                if other == edge:
                    continue
                role = ("S" if other_subject else "O") + ("S" if node_is_subject else "O")
                allowed &= self.statistics.joins[role][p] > 0          # the pair can be non-empty
                log_lift += np.log(np.clip(self.statistics.lifts[role][p], 1e-4, 1e4))
        if not allowed.any():
            return None
        candidates = np.flatnonzero(allowed)
        weights = np.exp(-self.gamma * (log_lift[candidates] - log_lift[candidates].min()))
        for _ in range(5):
            variant = copy.deepcopy(instance)
            variant.predicates[edge] = int(candidates[rng.choice(len(candidates), p=weights / weights.sum())])
            if self._empty_pairs(variant, [edge]) <= self.max_empty_pairs:
                return variant
        return None

    def perturb(self, instance, rng):
        """One hard-zero candidate from a non-empty instance, or None."""
        constants = list(instance.plan.constants)
        free_edges = [i for i, (u, v) in enumerate(instance.oriented)
                      if u not in instance.plan.constants and v not in instance.plan.constants]
        options = [("constant", node) for node in constants] + [("predicate", edge) for edge in free_edges]
        if not options:
            return None
        kind, target = options[int(rng.integers(0, len(options)))]
        variant = (self._swap_constant(instance, target, rng) if kind == "constant"
                   else self._swap_predicate(instance, target, rng))
        if variant is not None:
            variant.perturbation = kind
        return variant
