"""Turn an abstract shape into a concrete, non-empty query on the data graph.

1. Binding plan: which shape nodes become constants, and in which frequency stratum (tail /
   middle / head of that predicate's constants). Never a plan that breaks the query apart:
   every pattern keeps a variable and the patterns stay connected through shared VARIABLES
   (a constant shared by two patterns is no join, it makes a cartesian product).
2. Embedding: a randomised homomorphism search that maps every shape node to an entity and
   every shape edge to a predicate such that each edge is a triple of the data (so the query
   is non-empty by construction). Nodes are assigned in BFS order; a node's candidates are the
   neighbours of an assigned neighbour (sampled for hubs), filtered by every other already
   assigned neighbour (that is how cycles, diamonds and flowers close), and drawn with weights
   from the choke-point policy:
       predicates  uniform, or by lift with the predicates already at the shared node
                   (correlated: lift^gamma, anticorrelated: lift^-gamma; statistics.py)
       entities    uniform, or by degree^gamma for join variables (hub skew), and for nodes
                   that will be constants, by whether their frequency hits the planned stratum
   Distinct nodes get distinct entities. A dead end restarts the search.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.random_query_generation.statistics_based_generation.statistics import STRATA


@dataclass(frozen=True)
class ChokePoint:
    """How to instantiate: a named policy (see generate.py's config for the catalogue)."""
    name: str
    predicate_mode: str = "uniform"        # uniform | correlated | anticorrelated
    predicate_gamma: float = 1.0
    hub_gamma: float = 0.0                  # > 0: join variables prefer high-degree entities
    n_constants: tuple = (1, 2)             # inclusive range
    strata_weights: tuple = (1.0, 1.0, 1.0) # tail, middle, head
    prefer_leaves: float = 0.8              # probability a constant goes on a leaf when one is free
    spread_constants: bool = False          # constants as far apart as possible (competing anchors)


@dataclass
class BindingPlan:
    constants: dict = field(default_factory=dict)       # node -> target stratum


@dataclass
class Instance:
    shape: object
    entities: np.ndarray            # (n_nodes,) entity per node
    predicates: np.ndarray          # (n_edges,) predicate id per edge
    plan: BindingPlan
    oriented: list                  # (subject node, object node) per edge, as embedded

    def triple_patterns(self, graph):
        """SPARQL triple patterns, constants as IRIs, other nodes as ?v<node>."""
        def term(node):
            return graph.iri(self.entities[node]) if node in self.plan.constants else f"?v{node}"
        return [f"{term(u)} {graph.predicate_iri(p)} {term(v)}"
                for (u, v), p in zip(self.oriented, self.predicates)]

    def structured_patterns(self):
        """(subject term, predicate, object term), a term ("c", entity) or ("v", node);
        the input of counting.AcyclicCounter."""
        def term(node):
            return ("c", int(self.entities[node])) if node in self.plan.constants else ("v", int(node))
        return [(term(u), int(p), term(v)) for (u, v), p in zip(self.oriented, self.predicates)]

    def pattern_sizes(self, graph, statistics):
        """Exact number of matches of every single pattern."""
        sizes = []
        for (u, v), p in zip(self.oriented, self.predicates):
            if u in self.plan.constants and v in self.plan.constants:
                sizes.append(1)
            elif u in self.plan.constants:
                sizes.append(graph.count_bound_subject(self.entities[u], p))
            elif v in self.plan.constants:
                sizes.append(graph.count_bound_object(self.entities[v], p))
            else:
                sizes.append(int(statistics.counts[p]))
        return sizes


# --- binding plans ------------------------------------------------------------------------

def variable_connected(shape, constants):
    """Every pattern has a variable, and the patterns are connected through shared variables."""
    if any(u in constants and v in constants for u, v in shape.edges):
        return False
    by_variable = {}
    for index, (u, v) in enumerate(shape.edges):
        for node in (u, v):
            if node not in constants:
                by_variable.setdefault(node, []).append(index)
    seen, frontier = {0}, [0]
    while frontier:
        u, v = shape.edges[frontier.pop()]
        for node in (u, v):
            for other in by_variable.get(node, []):
                if other not in seen:
                    seen.add(other)
                    frontier.append(other)
    return len(seen) == len(shape.edges)


def _distances(shape, source):
    adjacency = [[] for _ in range(shape.n_nodes)]
    for u, v in shape.edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    distance = [None] * shape.n_nodes
    distance[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if distance[v] is None:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance


def cycle_nodes(shape):
    """Nodes on a cycle (the 2-core of the undirected shape): never constants, or the cycle
    would no longer be a cycle of join variables."""
    degree = shape.degree().copy()
    alive = set(range(shape.n_nodes))
    changed = True
    while changed:
        changed = False
        for node in list(alive):
            if degree[node] <= 1:
                alive.discard(node)
                changed = True
                for u, v in shape.edges:
                    if node in (u, v):
                        other = v if u == node else u
                        if other in alive:
                            degree[other] -= 1
    return alive


def binding_plan(shape, choke_point, rng):
    """Constants on nodes that keep the query variable-connected and its cycles intact."""
    low, high = choke_point.n_constants
    target = int(rng.integers(low, high + 1))
    degree = shape.degree()
    strata_p = np.asarray(choke_point.strata_weights, dtype=float)
    strata_p = strata_p / strata_p.sum()
    constants = {}
    on_cycle = cycle_nodes(shape)
    for _ in range(target):
        free = [node for node in range(shape.n_nodes) if node not in constants and node not in on_cycle
                and variable_connected(shape, {**constants, node: None})]
        if not free:
            break
        leaves = [node for node in free if degree[node] == 1]
        pool = leaves if leaves and rng.random() < choke_point.prefer_leaves else free
        if choke_point.spread_constants and constants:
            # competing anchors: the free node farthest from every constant so far
            nearest = [min(_distances(shape, c)[node] for c in constants) for node in pool]
            best = max(nearest)
            pool = [node for node, d in zip(pool, nearest) if d == best]
        node = pool[int(rng.integers(0, len(pool)))]
        constants[node] = STRATA[int(rng.choice(len(STRATA), p=strata_p))]
    return BindingPlan(constants)


# --- embedding ----------------------------------------------------------------------------

class Instantiator:
    """Randomised homomorphism search; see the module docstring.

    Edge directions are decided here, not by the shape: a node's candidates are its neighbours
    in BOTH directions, the shape's direction only weighs (direction_preference : 1). Many
    entities only occur as subjects or only as objects, so fixed directions dead-end walks.
    Order: most-constrained-first (the next node has the most assigned neighbours, ties by
    shape degree), so cycles, diamonds and flowers close as early as possible.
    """

    def __init__(self, graph, statistics, max_pattern_matches=1_000_000, max_candidates=2000, max_restarts=30,
                 off_stratum_weight=0.02, direction_preference=3.0):
        self.graph, self.statistics = graph, statistics
        self.max_pattern_matches = max_pattern_matches
        self.max_candidates = max_candidates
        self.max_restarts = max_restarts
        self.off_stratum_weight = off_stratum_weight
        self.direction_preference = direction_preference
        # Predicates too large to appear between two variables (the guardrail against queries
        # that are hard only because a pattern matches a huge part of the data).
        self.small_predicate = statistics.counts <= max_pattern_matches
        self._seed_cdf = np.cumsum(statistics.degree.astype(np.float64))

    def _sample_range(self, a, b, rng):
        size = b - a
        if size > self.max_candidates:
            return a + rng.choice(size, self.max_candidates, replace=False)
        return np.arange(a, b)

    def _candidates(self, entity_u, rng):
        """Neighbours of entity_u in both directions: (predicates, entities, u_is_subject)."""
        g = self.graph
        out_index = self._sample_range(g.out_ptr[entity_u], g.out_ptr[entity_u + 1], rng)
        in_index = self._sample_range(g.in_ptr[entity_u], g.in_ptr[entity_u + 1], rng)
        preds = np.concatenate([g.out_p[out_index], g.in_p[in_index]]).astype(np.int64)
        ents = np.concatenate([g.out_o[out_index], g.in_s[in_index]]).astype(np.int64)
        u_subject = np.concatenate([np.ones(len(out_index), bool), np.zeros(len(in_index), bool)])
        return preds, ents, u_subject

    def _adjacent(self, entity_w, candidates):
        """Candidates adjacent to entity_w in either direction."""
        return (np.isin(candidates, self.graph.out_edges(entity_w)[1])
                | np.isin(candidates, self.graph.in_edges(entity_w)[1]))

    def _connections(self, entity_w, entity_v):
        """(predicate, w_is_subject) for every triple between entity_w and entity_v."""
        out_p, out_o = self.graph.out_edges(entity_w)
        in_p, in_s = self.graph.in_edges(entity_w)
        preds = np.concatenate([out_p[out_o == entity_v], in_p[in_s == entity_v]]).astype(np.int64)
        w_subject = np.concatenate([np.ones(int((out_o == entity_v).sum()), bool),
                                    np.zeros(int((in_s == entity_v).sum()), bool)])
        return preds, w_subject

    def _lift_weights(self, preds, node_is_subject, assigned_at_node, choke_point):
        """Weights of candidate predicates for a new edge at a node, from their lifts with the
        edges already at that node. assigned_at_node: [(predicate, node_is_subject)]."""
        if choke_point.predicate_mode == "uniform" or not assigned_at_node:
            return np.ones(len(preds))
        log_weight = np.zeros(len(preds))
        for predicate, existing_subject in assigned_at_node:
            for subject_flag in (True, False):
                mask = node_is_subject == subject_flag
                if not mask.any():
                    continue
                role = ("S" if existing_subject else "O") + ("S" if subject_flag else "O")
                lift = self.statistics.lifts[role][predicate, preds[mask]]
                log_weight[mask] += np.log(np.clip(lift, 1e-4, 1e4))
        sign = 1.0 if choke_point.predicate_mode == "correlated" else -1.0
        scaled = sign * choke_point.predicate_gamma * log_weight
        return np.exp(scaled - scaled.max())

    def _seed(self, shape, plan, choke_point, rng):
        n = 64
        draws = np.searchsorted(self._seed_cdf, rng.random(n) * self._seed_cdf[-1])
        draws = draws[self.statistics.degree[draws] >= shape.degree()[shape.root]]
        if len(draws) == 0:
            return None
        if shape.root in plan.constants or choke_point.hub_gamma <= 0:
            return int(draws[0])
        weights = self.statistics.degree[draws].astype(np.float64) ** (choke_point.hub_gamma - 1.0)
        return int(draws[rng.choice(len(draws), p=weights / weights.sum())])

    def _pattern_size(self, predicate, subject_entity, object_entity, subject_constant, object_constant):
        if subject_constant:
            return self.graph.count_bound_subject(subject_entity, predicate)
        if object_constant:
            return self.graph.count_bound_object(object_entity, predicate)
        return int(self.statistics.counts[predicate])

    def _stratum(self, predicate, entity, entity_is_subject):
        if entity_is_subject:
            return self.statistics.stratum("subject", predicate, self.graph.count_bound_subject(entity, predicate))
        return self.statistics.stratum("object", predicate, self.graph.count_bound_object(entity, predicate))

    @staticmethod
    def _order(shape, adjacency):
        degree = shape.degree()
        order, assigned = [shape.root], {shape.root}
        while len(order) < shape.n_nodes:
            best, best_key = None, None
            for node in range(shape.n_nodes):
                if node in assigned:
                    continue
                links = sum(1 for i in adjacency[node] if (set(shape.edges[i]) - {node}) & assigned)
                if links == 0:
                    continue
                key = (links, degree[node], -node)
                if best_key is None or key > best_key:
                    best, best_key = node, key
            order.append(best)
            assigned.add(best)
        return order

    def embed(self, shape, plan, choke_point, rng):
        """An Instance (all edges real triples, directions decided), or None."""
        adjacency = [[] for _ in range(shape.n_nodes)]
        for index, (u, v) in enumerate(shape.edges):
            adjacency[u].append(index)
            adjacency[v].append(index)
        order = self._order(shape, adjacency)
        for _ in range(self.max_restarts):
            instance = self._attempt(shape, plan, choke_point, rng, adjacency, order)
            if instance is not None:
                return instance
        return None

    def _attempt(self, shape, plan, choke_point, rng, adjacency, order):
        degree = shape.degree()
        entities = np.full(shape.n_nodes, -1, dtype=np.int64)
        predicates = np.full(shape.n_edges, -1, dtype=np.int64)
        oriented = [None] * shape.n_edges
        seed = self._seed(shape, plan, choke_point, rng)
        if seed is None:
            return None
        entities[order[0]] = seed
        used = {seed}

        def other_end(index, node):
            u, v = shape.edges[index]
            return v if u == node else u

        def at_node(node):
            return [(predicates[i], oriented[i][0] == node) for i in adjacency[node] if predicates[i] >= 0]

        for node in order[1:]:
            constraint = [i for i in adjacency[node] if entities[other_end(i, node)] >= 0]
            parent_edge = constraint[0]
            u = other_end(parent_edge, node)
            preds, ents, u_subject = self._candidates(int(entities[u]), rng)
            keep = ~np.isin(ents, np.fromiter(used, dtype=np.int64))
            keep &= self.statistics.degree[ents] >= degree[node]          # room for its other edges
            both_variables = u not in plan.constants and node not in plan.constants
            if both_variables:
                keep &= self.small_predicate[preds]
            for other_edge in constraint[1:]:
                keep &= self._adjacent(int(entities[other_end(other_edge, node)]), ents)
            preds, ents, u_subject = preds[keep], ents[keep], u_subject[keep]
            if len(ents) == 0:
                return None
            weights = self._lift_weights(preds, u_subject, at_node(u), choke_point)
            preferred = u_subject == (shape.edges[parent_edge][0] == u)
            weights = weights * np.where(preferred, self.direction_preference, 1.0)
            if node in plan.constants:
                ok = np.array([self._stratum(int(p), int(c), not us) == plan.constants[node]
                               for p, c, us in zip(preds, ents, u_subject)])
                weights = weights * np.where(ok, 1.0, self.off_stratum_weight)
            elif choke_point.hub_gamma > 0:
                weights = weights * self.statistics.degree[ents].astype(np.float64) ** choke_point.hub_gamma
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                weights = np.ones(len(ents))
            pick = int(rng.choice(len(ents), p=weights / weights.sum()))
            entities[node] = ents[pick]
            predicates[parent_edge] = preds[pick]
            oriented[parent_edge] = (u, node) if u_subject[pick] else (node, u)
            used.add(int(ents[pick]))
            # the other constraint edges close cycles: pick one of the existing connections
            for other_edge in constraint[1:]:
                w = other_end(other_edge, node)
                options, w_subject = self._connections(int(entities[w]), int(entities[node]))
                if w not in plan.constants and node not in plan.constants:
                    small = self.small_predicate[options]
                    options, w_subject = options[small], w_subject[small]
                if len(options) == 0:
                    return None
                option_weights = self._lift_weights(options, w_subject, at_node(w), choke_point)
                choice = int(rng.choice(len(options), p=option_weights / option_weights.sum()))
                predicates[other_edge] = options[choice]
                oriented[other_edge] = (w, node) if w_subject[choice] else (node, w)
        # duplicate triple patterns (same oriented node pair and predicate) are not allowed
        if len({(o, int(p)) for o, p in zip(oriented, predicates)}) < shape.n_edges:
            return None
        instance = Instance(shape, entities, predicates, plan, oriented)
        if max(instance.pattern_sizes(self.graph, self.statistics)) > self.max_pattern_matches:
            return None
        return instance


# --- result budget ------------------------------------------------------------------------

def _spanning_subset(structured):
    """Pattern indices of a spanning tree of the variable graph plus every pattern with a
    constant: dropping the closing patterns of cycles only loosens the query, so its count is
    an upper bound on the full query's count (exact for acyclic queries)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keep = []
    for i, (s, _, o) in enumerate(structured):
        if s[0] == "v" and o[0] == "v":
            a, b = find(s[1]), find(o[1])
            if a == b:
                continue
            parent[a] = b
        keep.append(i)
    return keep


def result_upper_bound(instance, counter):
    """The exact result size if the counter can give it (counting.ExactCounter conditions
    cyclic queries); otherwise the count of a spanning tree, an upper bound."""
    structured = instance.structured_patterns()
    exact = getattr(counter, "count_exact", None)
    if exact is not None:
        value = exact(structured, range(len(structured)))
        if value is not None:
            return value
    return counter.count(structured, _spanning_subset(structured))


def tighten_to_budget(instance, graph, statistics, counter, budget, rng, tries_per_step=3):
    """Bind more nodes (with their witness entities, so the query stays non-empty) until the
    result is at most `budget`; returns the tightened Instance or None if no bindable node is
    left. Each step tries a few random bindable nodes (leaves first) and keeps the one whose
    result lands closest to the budget, so constants do not all come from the rarest values.
    Pure data: exact counts in memory (counting.AcyclicCounter), no optimizer, no model."""
    on_cycle = cycle_nodes(instance.shape)
    degree = instance.shape.degree()
    bound = result_upper_bound(instance, counter)
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
            tried.append((result_upper_bound(variant, counter), variant))
        under = [t for t in tried if t[0] <= budget]
        # closest from above if none fits, else the least selective one that fits
        bound, instance = max(under, key=lambda t: t[0]) if under else min(tried, key=lambda t: t[0])
    return instance


def _natural_stratum(instance, graph, statistics, node):
    """Stratum of the witness entity as a constant of its first pattern."""
    for (u, v), p in zip(instance.oriented, instance.predicates):
        if node in (u, v):
            if node == u:
                return statistics.stratum("subject", int(p), graph.count_bound_subject(int(instance.entities[node]), int(p)))
            return statistics.stratum("object", int(p), graph.count_bound_object(int(instance.entities[node]), int(p)))
    return "middle"
