"""Bushy join plans: plan trees, bottom-up forest states, and the exact (constrained) bushy DP.

An option next to the left-deep machinery in labels.py, which stays unchanged.

Plan tree    a pattern index (leaf) or a pair (left, right) of trees, e.g. ((0, 1), (2, 3)). A
             left-deep order [a, b, c] is the tree ((a, b), c) (tree_from_order).
Forest       the state of bottom-up planning: a tuple of disjoint component bitmasks covering all
             patterns, e.g. (0b0011, 0b0100, 0b1000) after joining patterns 0 and 1. Planning
             starts from the singletons and merges two components that share a variable, until one
             component is left; a plan is the sequence of merges (tree_to_merges / merges_to_tree).
             Left-deep planning is the special case that always merges the one large component
             with a single pattern.
Cost         C_out = sum of the cardinalities of all join results (the internal nodes); scans cost
             nothing. (The left-deep training objective in labels.plan_log_cost also adds
             min(card a, card b) for the first pair; the C_out metric of plan_quality does not.)

Exact DP over connected subsets:
    B(U) = card(U) + min over splits U = U1 + U2 (both connected, adjacent) of B(U1) + B(U2),   B(leaf) = 0
enumerated with DPccp (only valid csg-cmp pairs, src.baselines.enumeration). For a forest F the
cost-to-go G(F) is the same DP on the join graph contracted by F: each component is one node that
costs 0, so G(F) = B over all nodes.

Cost: one pass over the csg-cmp pairs -- milliseconds for most v2 queries, seconds for a
16-pattern star (21M pairs); each built component shrinks the contracted graph, so forest DPs
become cheap quickly.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Union

from src.baselines.enumeration import _bits, connected_subgraphs, csg_cmp_pairs
from src.supervised_value_estimation.amortized_dp.labels import NEG_INF, QueryLabels, members

Tree = Union[int, tuple]


# --- plan trees ----------------------------------------------------------------------------

def is_leaf(tree: Tree) -> bool:
    return isinstance(tree, int)


def leaves(tree: Tree) -> list[int]:
    return [tree] if is_leaf(tree) else leaves(tree[0]) + leaves(tree[1])


def tree_mask(tree: Tree) -> int:
    return 1 << tree if is_leaf(tree) else tree_mask(tree[0]) | tree_mask(tree[1])


def internal_masks(tree: Tree) -> list[int]:
    """Masks of the join results, children before parents (the execution order bottom-up)."""
    if is_leaf(tree):
        return []
    return internal_masks(tree[0]) + internal_masks(tree[1]) + [tree_mask(tree)]


def tree_from_order(order: Sequence[int]) -> Tree:
    """The left-deep tree of a join order: [a, b, c, d] -> (((a, b), c), d)."""
    tree: Tree = int(order[0])
    for index in order[1:]:
        tree = (tree, int(index))
    return tree


def is_left_deep(tree: Tree) -> bool:
    """Linear: every join has a single pattern on one side (C_out is symmetric in the sides)."""
    if is_leaf(tree):
        return True
    left, right = tree
    return (is_leaf(right) and is_left_deep(left)) or (is_leaf(left) and is_left_deep(right))


def canonical(tree: Tree) -> Tree:
    """Children ordered by their smallest pattern, so equal plans compare equal."""
    if is_leaf(tree):
        return tree
    left, right = canonical(tree[0]), canonical(tree[1])
    return (left, right) if min(leaves(left)) < min(leaves(right)) else (right, left)


def to_json(tree: Tree):
    return tree if is_leaf(tree) else [to_json(tree[0]), to_json(tree[1])]


def from_json(value) -> Tree:
    return int(value) if isinstance(value, int) else (from_json(value[0]), from_json(value[1]))


def is_valid_tree(tree: Tree, n: int, neighbour_masks: Sequence[int]) -> bool:
    """Every pattern exactly once and every join between subtrees that share a variable."""
    if sorted(leaves(tree)) != list(range(n)):
        return False

    def ok(t):
        if is_leaf(t):
            return True
        left, right = tree_mask(t[0]), tree_mask(t[1])
        touching = any(neighbour_masks[i] & right for i in members(left))
        return touching and ok(t[0]) and ok(t[1])
    return ok(tree)


# --- bottom-up forest states ---------------------------------------------------------------

def initial_forest(n: int) -> tuple[int, ...]:
    return tuple(1 << i for i in range(n))


def merge(forest: Sequence[int], a: int, b: int) -> tuple[int, ...]:
    return tuple(sorted([c for c in forest if c not in (a, b)] + [a | b]))


def candidate_merges(forest: Sequence[int], neighbour_masks: Sequence[int]) -> list[tuple[int, int]]:
    """Pairs of components that share a variable (no cartesian products)."""
    reach = {c: _reach(c, neighbour_masks) for c in forest}
    return [(a, b) for i, a in enumerate(forest) for b in forest[i + 1:] if reach[a] & b]


def _reach(mask: int, neighbour_masks: Sequence[int]) -> int:
    reach = 0
    for i in members(mask):
        reach |= neighbour_masks[i]
    return reach & ~mask


def tree_to_merges(tree: Tree) -> list[tuple[int, int]]:
    """The plan as bottom-up merges (left component mask, right component mask)."""
    if is_leaf(tree):
        return []
    return tree_to_merges(tree[0]) + tree_to_merges(tree[1]) + [(tree_mask(tree[0]), tree_mask(tree[1]))]


def merges_to_tree(n: int, merges: Iterable[tuple[int, int]]) -> Tree:
    trees = {1 << i: i for i in range(n)}
    for a, b in merges:
        trees[a | b] = (trees.pop(a), trees.pop(b))
    if len(trees) != 1:
        raise ValueError("merges do not join all patterns into one tree")
    return next(iter(trees.values()))


def forest_after(tree: Tree, n: int, steps: int) -> tuple[int, ...]:
    """The forest after the first `steps` merges of the plan (bottom-up)."""
    forest = initial_forest(n)
    for a, b in tree_to_merges(tree)[:steps]:
        forest = merge(forest, a, b)
    return forest


# --- costs ---------------------------------------------------------------------------------

def _logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    peak = max(a, b)
    return peak + math.log(math.exp(a - peak) + math.exp(b - peak))


def tree_log_cost(tree: Tree, logcard: dict[int, float]) -> float:
    """log C_out of a plan tree: log of the sum of its join results' cardinalities."""
    total = NEG_INF
    for mask in internal_masks(tree):
        total = _logaddexp(total, logcard[mask])
    return total


def tree_join_cout(tree: Tree, rows: dict[int, int]):
    """C_out of a plan tree from row counts per subset; None if a count is missing."""
    total = 0
    for mask in internal_masks(tree):
        if rows.get(mask) is None:
            return None
        total += rows[mask]
    return total


# --- exact (constrained) bushy DP ----------------------------------------------------------

def contract(forest: Sequence[int], n: int, neighbour_masks: Sequence[int]):
    """The join graph with each forest component as one node: (component masks, node neighbour
    masks). Patterns not covered by the forest become their own nodes."""
    covered = 0
    for c in forest:
        covered |= c
    components = sorted(list(forest) + [1 << i for i in range(n) if not covered >> i & 1])
    reach = [_reach(c, neighbour_masks) for c in components]
    nbr = [sum(1 << j for j, d in enumerate(components) if j != i and reach[i] & d) for i in range(len(components))]
    return components, nbr


def bushy_dp(labels: QueryLabels, logcard: dict[int, float] | None = None, forest: Sequence[int] = ()):
    """DPccp (Moerkotte & Neumann 2006, src.baselines.enumeration) for the cheapest C_out plan,
    over the join graph contracted by `forest` (no forest: the singletons, i.e. the plain DP).
    Components are base relations that cost 0, so the result is the cost-to-go G(forest).

    Runs in linear space, scaled by the largest cardinality (sums compare as their logs would,
    without a log per pair). Returns (log cost, best split per contracted subset, components);
    log cost is inf when the graph is not connected."""
    logcard = labels.logcard if logcard is None else logcard
    components, nbr = contract(forest or initial_forest(labels.n_tp), labels.n_tp, labels.neighbour_masks)
    k = len(components)
    hood = connected_subgraphs(k, nbr)
    weight = {}                                        # contracted csg -> card (0 for one node)
    for m in hood:
        if m & (m - 1):
            original = 0
            for i in _bits(m):
                original |= components[i]
            weight[m] = logcard[original]
    peak = max(weight.values(), default=0.0)
    for m in hood:
        weight[m] = math.exp(weight[m] - peak) if m in weight else 0.0
    build = {1 << i: 0.0 for i in range(k)}            # B(U) = card(U) + best split, final per DP order
    split: dict[int, tuple[int, int]] = {}
    best: dict[int, float] = {}
    last = None
    for s1, s2 in csg_cmp_pairs(k, nbr, hood):
        if s1 != last:                                 # all pairs building s1 came before
            if s1 not in build:
                build[s1] = weight[s1] + best[s1]
            last = s1
        b2 = build.get(s2)
        if b2 is None:
            b2 = build[s2] = weight[s2] + best[s2]
        value = build[s1] + b2
        union = s1 | s2
        if value < best.get(union, math.inf):
            best[union], split[union] = value, (s1, s2)
    full = (1 << k) - 1
    if k == 1:
        return NEG_INF, split, components
    if full not in best:
        return math.inf, split, components
    return peak + math.log(weight[full] + best[full]), split, components


def _decode(mask: int, split: dict[int, tuple[int, int]], node_trees: list) -> Tree:
    if not mask & (mask - 1):
        return node_trees[mask.bit_length() - 1]
    left, right = split[mask]
    return (_decode(left, split, node_trees), _decode(right, split, node_trees))


def optimal_bushy_plan(labels: QueryLabels, logcard: dict[int, float] | None = None,
                       forest: Sequence[int] = (), built_trees: dict[int, Tree] | None = None):
    """The cheapest bushy plan (tree) and its log C_out under `logcard`. With a forest, the
    cheapest completion: the plan's cost counts only joins above the built components, i.e. it is
    the cost-to-go G(forest); `built_trees` gives the components' subtrees for the returned tree
    (otherwise each built component appears as the left-deep tree of its patterns)."""
    log_cost, split, components = bushy_dp(labels, logcard, forest)
    if log_cost == math.inf:
        return None, math.inf
    node_trees = [c.bit_length() - 1 if not c & (c - 1) else (built_trees or {}).get(c, tree_from_order(members(c)))
                  for c in components]
    return _decode((1 << len(components)) - 1, split, node_trees), log_cost


def forest_cost_to_go(labels: QueryLabels, forest: Sequence[int], logcard: dict[int, float] | None = None) -> float:
    """log G(forest): the cheapest cost of merging the forest's components into one plan."""
    return bushy_dp(labels, logcard, forest)[0]


# --- estimator-only planners (no cost-to-go) ----------------------------------------------

def estimator_beam_plan(n: int, neighbour_masks: Sequence[int], log_cards, width: int, all_trees: bool = False):
    """Bottom-up beam search on a cardinality estimator alone: keep the `width` forests with the
    smallest estimated C_out so far (the sum of their own merges' cards; there is no cost-to-go).
    Width 1 is greedy operator ordering (GOO: merge the pair with the smallest result); an
    unbounded width is the exact bushy DP over the estimates. `log_cards(masks)` estimates a
    list of subsets and is called once per step, for the subsets not estimated before.
    Returns the tree with the smallest estimated C_out (with all_trees: the beam's final trees,
    cheapest first) and the number of subsets estimated."""
    known: dict[int, float] = {}
    beams = [(initial_forest(n), NEG_INF, {1 << i: i for i in range(n)})]     # (forest, log C_out so far, trees)
    while len(beams[0][0]) > 1:
        candidates = [(index, a, b) for index, (forest, _, _) in enumerate(beams)
                      for a, b in candidate_merges(forest, neighbour_masks)]
        missing = list(dict.fromkeys(a | b for _, a, b in candidates if a | b not in known))
        if missing:
            known.update(zip(missing, log_cards(missing)))
        best: dict[tuple[int, ...], tuple[float, int]] = {}                 # child forest -> cheapest candidate
        for k, (index, a, b) in enumerate(candidates):
            child, so_far = merge(beams[index][0], a, b), _logaddexp(beams[index][1], known[a | b])
            if child not in best or so_far < best[child][0]:
                best[child] = (so_far, k)
        next_beams = []
        for child, (so_far, k) in sorted(best.items(), key=lambda item: item[1][0])[:width]:
            index, a, b = candidates[k]
            trees = dict(beams[index][2])
            trees[a | b] = (trees.pop(a), trees.pop(b))
            next_beams.append((child, so_far, trees))
        beams = next_beams
    if all_trees:
        return [next(iter(beam[2].values())) for beam in beams], len(known)
    return next(iter(beams[0][2].values())), len(known)


# --- partial tables: stitched upper bounds on the cost-to-go ----------------------------------

def repair_completion(tree: Tree, forest: Sequence[int]) -> list[tuple[int, int]]:
    """A completion of `forest` that follows `tree`: the tree's merges bottom-up, each one joining
    the current components that hold its two sides (skipped once they are one component). Every
    tree node processed so far lies inside one component, so two joined components share a
    variable whenever the tree's two sides do: always a valid plan from the forest, and the
    tree's own remaining merges when the forest is on the tree's path. Returns the merges."""
    owner = {}
    for component in forest:
        for p in members(component):
            owner[p] = component
    merges = []
    for a, b in tree_to_merges(tree):
        left, right = owner[(a & -a).bit_length() - 1], owner[(b & -b).bit_length() - 1]
        if left == right:
            continue
        merges.append((left, right))
        for p in members(left | right):
            owner[p] = left | right
    return merges


class StitchTable:
    """Stitched upper bounds on G(forest) from a partial table: the constrained bushy DP restricted
    to the join results in `sets` (each with a card in logcard), i.e. over the plans from a forest
    whose every join builds a set in `sets` out of two parts that are forest components or such
    sets. Any plan stitched together from scored subsets counts; with all connected subsets it is
    the exact forest_cost_to_go. The splits of a set into two parts from `sets` (or single
    patterns) do not depend on the forest: computed once, when first needed, and shared by all the
    forests of a query; the DP itself runs top-down from the full set, so it only visits sets that
    some stitched plan can use."""

    def __init__(self, sets: Iterable[int], logcard: dict[int, float]):
        import numpy as np
        self.np = np
        self.joins = {u for u in sets if u & (u - 1)}
        n = max(self.joins, default=1).bit_length()
        self.parts = np.array(sorted(self.joins | {1 << i for i in range(n)}), dtype=np.int64)
        self.weight = {}
        self.peak = max((logcard[u] for u in self.joins), default=0.0)   # linear space, scaled as in bushy_dp
        self.logcard = logcard
        self.splits_of: dict[int, list[tuple[int, int]]] = {}

    def splits(self, u: int) -> list[tuple[int, int]]:
        if u not in self.splits_of:
            np, parts = self.np, self.parts
            inside = parts[((parts & ~u) == 0) & (parts < (u ^ parts))]          # v < u ^ v (so v != u)
            rest = u ^ inside
            found = parts[np.minimum(np.searchsorted(parts, rest), len(parts) - 1)] == rest
            self.splits_of[u] = list(zip(inside[found].tolist(), rest[found].tolist()))
        return self.splits_of[u]

    def cost_to_go(self, forest: Sequence[int]) -> float:
        """log of the stitched upper bound on G(forest); -inf when complete, inf when no plan
        stitched from the sets completes the forest."""
        if len(forest) == 1:
            return NEG_INF
        components = set(forest)
        full = 0
        for c in forest:
            full |= c
        whole: dict[int, bool] = {}
        build: dict[int, float] = {}
        unlisted = [c for c in forest if c & (c - 1) and c not in self.joins]   # components not in `sets`

        def is_whole(u):                                 # a union of components (not cutting one)
            if u not in whole:
                whole[u] = all(u & c in (0, c) for c in forest)
            return whole[u]

        def cost(u):
            if u in components:
                return 0.0
            if u not in build:
                best = math.inf
                if u in self.joins:
                    for v, w in self.splits(u):
                        if is_whole(v) and is_whole(w):
                            best = min(best, cost(v) + cost(w))
                    for c in unlisted:
                        w = u ^ c
                        if c & ~u == 0 and w and (w in self.joins or w in components) and is_whole(w):
                            best = min(best, cost(w))
                if best < math.inf:
                    best += math.exp(self.logcard[u] - self.peak)
                build[u] = best
            return build[u]
        value = cost(full)
        return self.peak + math.log(value) if value < math.inf else math.inf


def stitched_cost_to_go(forest: Sequence[int], sets: Iterable[int], logcard: dict[int, float]) -> float:
    """log of the stitched upper bound on G(forest) over the join results in `sets` (StitchTable)."""
    return StitchTable(sets, logcard).cost_to_go(forest)
