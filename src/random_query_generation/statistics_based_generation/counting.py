"""Exact result counts of ACYCLIC conjunctive queries, in memory, without materialising joins.

A connected set of triple patterns whose variables form a tree (every pattern has one or two
variables; no two patterns link the same two variables; no cycle) is counted by message
passing over that tree (the counting form of Yannakakis' algorithm): root the tree at a
variable; the message of a variable is, per entity it can take, the number of ways to extend
its subtree; a pattern with a constant end is an indicator on its variable; a child message is
pushed to the parent through the predicate's triples and summed; messages of different
children multiply. The count is the root message's sum. Cost is linear in the triples of the
predicates involved, independent of how large the result or any intermediate result is.

This is the number of SPARQL solutions (homomorphisms; variables may bind equal entities),
which is what COUNT(*) returns on a set of triples. Values are float64: exact below 2^53.
Subsets with a cycle are not counted here (`is_acyclic` is False); generate.py sends those to
QLever with a forced join order.
"""
from __future__ import annotations

import numpy as np


class PredicateIndex:
    """All triples grouped by predicate, with per-predicate subject/object groupings."""

    def __init__(self, graph):
        self.graph = graph
        subjects = np.repeat(np.arange(graph.n_entities, dtype=np.int32), np.diff(graph.out_ptr))
        order = np.argsort(graph.out_p, kind="stable")
        self.subjects = subjects[order]
        self.objects = graph.out_o[order]
        self.ptr = np.zeros(graph.n_predicates + 1, dtype=np.int64)
        np.cumsum(np.bincount(graph.out_p, minlength=graph.n_predicates), out=self.ptr[1:])

    def triples(self, predicate):
        a, b = self.ptr[predicate], self.ptr[predicate + 1]
        return self.subjects[a:b], self.objects[a:b]


class _Message:
    """Sparse vector over entities: sorted ids and values; None stands for 'all ones'."""
    __slots__ = ("ids", "values")

    def __init__(self, ids, values):
        self.ids, self.values = ids, values

    def lookup(self, entities):
        position = np.searchsorted(self.ids, entities)
        position = np.minimum(position, len(self.ids) - 1) if len(self.ids) else position
        hit = (self.ids[position] == entities) if len(self.ids) else np.zeros(len(entities), bool)
        return np.where(hit, self.values[position] if len(self.ids) else 0.0, 0.0)

    def multiply(self, other):
        if other is None:
            return self
        common, a, b = np.intersect1d(self.ids, other.ids, assume_unique=True, return_indices=True)
        return _Message(common, self.values[a] * other.values[b])

    def total(self):
        return float(self.values.sum())


def _multiply(a, b):
    if a is None:
        return b
    return a.multiply(b)


def _sum_by(keys, weights):
    if len(keys) == 0:
        return _Message(np.empty(0, np.int64), np.empty(0))
    ids, inverse = np.unique(keys, return_inverse=True)
    return _Message(ids.astype(np.int64), np.bincount(inverse, weights=weights))


class AcyclicCounter:
    """Counts of connected sub-BGPs of one query, given as patterns
    (subject term, predicate, object term) with a term ("v", node) or ("c", entity)."""

    def __init__(self, graph, predicate_index, small_support=2048, cache_entries=200_000):
        self.graph, self.index = graph, predicate_index
        self.small_support = small_support
        # Messages of unconstrained leaves are the same in every query: the per-predicate
        # degree vectors, cached for the whole run. Messages of subtrees are cached per query.
        self._degree_cache = {}
        self._subtree_cache = {}
        self._query_key = None
        self.cache_entries = cache_entries

    @staticmethod
    def is_acyclic(patterns, subset):
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in subset:
            s, _, o = patterns[i]
            if s[0] == "v" and o[0] == "v":
                if s[1] == o[1]:
                    return False
                a, b = find(s[1]), find(o[1])
                if a == b:
                    return False          # a cycle (or two patterns on the same variable pair)
                parent[a] = b
        return True

    def _indicator(self, pattern, variable_is_subject):
        s, p, o = pattern
        if variable_is_subject:
            entities = self.graph.subjects_of(o[1], p)
        else:
            entities = self.graph.objects_of(s[1], p)
        entities = np.unique(entities.astype(np.int64))
        return _Message(entities, np.ones(len(entities)))

    def _push(self, message, predicate, child_is_object):
        """sum over the child's entities e' of message(e') * [triple between e and e'] -> parent e.
        child_is_object: the child variable is the pattern's object (the parent its subject)."""
        if message is not None and len(message.ids) <= self.small_support:
            # few child entities: gather their own edges with this predicate
            parents, weights = [], []
            for entity, value in zip(message.ids, message.values):
                found = (self.graph.subjects_of(int(entity), predicate) if child_is_object
                         else self.graph.objects_of(int(entity), predicate))
                parents.append(found)
                weights.append(np.full(len(found), value))
            if not parents:
                return _Message(np.empty(0, np.int64), np.empty(0))
            return _sum_by(np.concatenate(parents).astype(np.int64), np.concatenate(weights))
        if message is None and (predicate, child_is_object) in self._degree_cache:
            return self._degree_cache[(predicate, child_is_object)]
        subjects, objects = self.index.triples(predicate)
        child, parent = (objects, subjects) if child_is_object else (subjects, objects)
        if message is None:
            result = _sum_by(parent.astype(np.int64), np.ones(len(child)))
            self._degree_cache[(predicate, child_is_object)] = result
            return result
        weights = np.ones(len(child)) if message is None else message.lookup(child.astype(np.int64))
        keep = weights != 0
        return _sum_by(parent[keep].astype(np.int64), weights[keep])

    def count(self, patterns, subset):
        """Exact number of solutions of the connected, acyclic sub-BGP `subset`."""
        subset = list(subset)
        key = tuple(patterns)
        if key != self._query_key or len(self._subtree_cache) > self.cache_entries:
            self._query_key, self._subtree_cache = key, {}
        if not self.is_acyclic(patterns, subset):
            raise ValueError("subset has a cycle")
        unary, binary = {}, {}
        for i in subset:
            s, p, o = patterns[i]
            if s[0] == "v" and o[0] == "v":
                binary.setdefault(s[1], []).append((o[1], p, True))       # child o is object
                binary.setdefault(o[1], []).append((s[1], p, False))      # child s is subject
            elif s[0] == "v":
                unary.setdefault(s[1], []).append(self._indicator(patterns[i], True))
            elif o[0] == "v":
                unary.setdefault(o[1], []).append(self._indicator(patterns[i], False))
            else:
                raise ValueError("pattern without a variable")
        variables = set(unary) | set(binary)
        root = next(iter(variables))

        members = set(subset)
        edges_of = {}
        for i in subset:
            s, _, o = patterns[i]
            for term in (s, o):
                if term[0] == "v":
                    edges_of.setdefault(term[1], []).append(i)

        def subtree_patterns(node, parent):
            """Pattern indices in the subtree below `node` (away from `parent`)."""
            seen, stack, found = {node}, [node], set()
            while stack:
                x = stack.pop()
                for i in edges_of.get(x, []):
                    if i in members and i not in found:
                        s, _, o = patterns[i]
                        ends = [t[1] for t in (s, o) if t[0] == "v"]
                        if parent in ends and x == node and len(ends) == 2 and node in ends:
                            continue                      # the edge to the parent itself
                        found.add(i)
                        for y in ends:
                            if y not in seen and y != parent:
                                seen.add(y)
                                stack.append(y)
            return frozenset(found)

        def message(node, parent):
            cache_key = (node, parent, subtree_patterns(node, parent))
            if cache_key in self._subtree_cache:
                return self._subtree_cache[cache_key]
            result = _compute(node, parent)
            self._subtree_cache[cache_key] = result
            return result

        def _compute(node, parent):
            current = None
            for indicator in unary.get(node, []):
                current = _multiply(current, indicator)
            for child, predicate, child_is_object in binary.get(node, []):
                if child == parent:
                    continue
                current = _multiply(current, self._push(message(child, node), predicate, child_is_object))
            return current

        result = message(root, None)
        if result is None:          # cannot happen for a connected subset with a pattern
            raise ValueError("empty subset")
        return result.total()


def _variables(pattern):
    return [t[1] for t in (pattern[0], pattern[2]) if t[0] == "v"]


def _components(patterns, subset):
    """Groups of `subset` connected through shared variables."""
    subset = list(subset)
    remaining, groups = set(subset), []
    while remaining:
        start = remaining.pop()
        group, frontier = [start], [start]
        while frontier:
            i = frontier.pop()
            vi = set(_variables(patterns[i]))
            for j in list(remaining):
                if vi & set(_variables(patterns[j])):
                    remaining.discard(j)
                    group.append(j)
                    frontier.append(j)
        groups.append(group)
    return groups


class ExactCounter(AcyclicCounter):
    """AcyclicCounter plus cyclic sub-BGPs by conditioning: bind a small set of variables that
    breaks every cycle, count the (now acyclic) rest exactly for each of their joint values,
    and sum. Returns None when the conditioning domain exceeds max_assignments."""

    def __init__(self, graph, predicate_index, max_assignments=2000, **kwargs):
        super().__init__(graph, predicate_index, **kwargs)
        self.max_assignments = max_assignments
        self._predicate_entities = {}

    def _entities_with(self, predicate, as_subject):
        key = (predicate, as_subject)
        if key not in self._predicate_entities:
            subjects, objects = self.index.triples(predicate)
            self._predicate_entities[key] = np.unique((subjects if as_subject else objects).astype(np.int64))
        return self._predicate_entities[key]

    def _domain(self, patterns, subset, variable):
        domain = None
        for i in subset:
            s, p, o = patterns[i]
            if s == ("v", variable) and o[0] == "c":
                values = np.unique(self.graph.subjects_of(o[1], p).astype(np.int64))
            elif o == ("v", variable) and s[0] == "c":
                values = np.unique(self.graph.objects_of(s[1], p).astype(np.int64))
            elif s == ("v", variable):
                values = self._entities_with(p, True)
            elif o == ("v", variable):
                values = self._entities_with(p, False)
            else:
                continue
            domain = values if domain is None else np.intersect1d(domain, values, assume_unique=True)
        return domain

    def _breaking_variable(self, patterns, subset):
        """The variable with most binary patterns in the cyclic part (greedy feedback vertex)."""
        degree = {}
        for i in subset:
            vs = _variables(patterns[i])
            if len(vs) == 2:
                for v in vs:
                    degree[v] = degree.get(v, 0) + 1
        return max(degree, key=degree.get)

    def _bind(self, patterns, subset, variable, entity):
        """Patterns with `variable` replaced by the constant `entity`; None if a pattern that
        becomes fully bound is not a triple (count 0); fully bound true patterns are dropped."""
        bound, keep = list(patterns), []
        for i in subset:
            s, p, o = patterns[i]
            s = ("c", entity) if s == ("v", variable) else s
            o = ("c", entity) if o == ("v", variable) else o
            if s[0] == "c" and o[0] == "c":
                if not self.graph.has_triple(s[1], p, o[1]):
                    return None, None
                continue
            bound[i] = (s, p, o)
            keep.append(i)
        return bound, keep

    def count_exact(self, patterns, subset, depth=0):
        """Exact count of any connected sub-BGP, or None if conditioning would be too large."""
        subset = list(subset)
        if not subset:
            return 1.0
        total = 1.0
        for group in _components(patterns, subset):
            if self.is_acyclic(patterns, group):
                value = self.count(patterns, group)
            else:
                if depth >= 3:
                    return None
                variable = self._breaking_variable(patterns, group)
                domain = self._domain(patterns, group, variable)
                if domain is None or len(domain) > self.max_assignments:
                    return None
                value = 0.0
                for entity in domain:
                    bound, keep = self._bind(patterns, group, variable, int(entity))
                    if bound is None:
                        continue
                    part = self.count_exact(bound, keep, depth + 1)
                    if part is None:
                        return None
                    value += part
            total *= value
            if total == 0:
                return 0.0
        return total
