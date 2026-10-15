import functools
import math
import random

import pytest

from src.baselines.enumeration import connected_subgraphs, csg_cmp_pairs
from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.labels import NEG_INF, QueryLabels, logsumexp, mask_of, members
from src.supervised_value_estimation.amortized_dp.test.test_amortized_dp import _all_left_deep_orders, _synthetic_labels


def _brute_force_cost_to_go(labels, forest):
    """Minimum over ALL merge sequences from `forest` of log(sum of the merged results' cards)."""
    @functools.lru_cache(maxsize=None)
    def best(state):
        if len(state) == 1:
            return NEG_INF
        options = [logsumexp([labels.logcard[a | b], best(bushy.merge(state, a, b))])
                   for a, b in bushy.candidate_merges(state, labels.neighbour_masks)]
        return min(options) if options else math.inf
    return best(tuple(sorted(forest)))


def _random_forest(labels, steps, rng):
    forest, trees = bushy.initial_forest(labels.n_tp), {1 << i: i for i in range(labels.n_tp)}
    for _ in range(steps):
        a, b = rng.choice(bushy.candidate_merges(forest, labels.neighbour_masks))
        forest = bushy.merge(forest, a, b)
        trees[a | b] = (trees.pop(a), trees.pop(b))
    return forest, trees


@pytest.mark.parametrize("seed", range(6))
def test_bushy_optimum_matches_brute_force(seed):
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=seed)
    tree, log_cost = bushy.optimal_bushy_plan(labels)
    assert log_cost == pytest.approx(_brute_force_cost_to_go(labels, bushy.initial_forest(6)), abs=1e-9)
    assert bushy.is_valid_tree(tree, 6, labels.neighbour_masks)
    assert bushy.tree_log_cost(tree, labels.logcard) == pytest.approx(log_cost, abs=1e-9)
    # Left-deep plans are bushy plans: the bushy optimum is never worse.
    best_left_deep = min(bushy.tree_log_cost(bushy.tree_from_order(o), labels.logcard)
                         for o in _all_left_deep_orders(labels))
    assert log_cost <= best_left_deep + 1e-9


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_forest_cost_to_go_matches_brute_force(seed, steps):
    labels, _ = _synthetic_labels(7, extra_edges=3, seed=seed)
    forest, trees = _random_forest(labels, steps, random.Random(seed * 10 + steps))
    expected = _brute_force_cost_to_go(labels, forest)
    assert bushy.forest_cost_to_go(labels, forest) == pytest.approx(expected, abs=1e-9)

    built = {c: t for c, t in trees.items() if c & (c - 1)}
    tree, log_cost = bushy.optimal_bushy_plan(labels, forest=forest, built_trees=built)
    assert log_cost == pytest.approx(expected, abs=1e-9)
    assert bushy.is_valid_tree(tree, 7, labels.neighbour_masks)
    # The completion keeps the built subtrees and only pays for the joins above them.
    above = [m for m in bushy.internal_masks(tree) if not any(m & c == m for c in built)]
    assert logsumexp([labels.logcard[m] for m in above]) == pytest.approx(log_cost, abs=1e-9)
    assert set(built.values()) <= set(_subtrees(tree))


def _subtrees(tree):
    yield tree
    if not bushy.is_leaf(tree):
        yield from _subtrees(tree[0])
        yield from _subtrees(tree[1])


@pytest.mark.parametrize("seed", range(4))
def test_bellman_consistency(seed):
    """G(F) = min over merges (a, b) of card(a | b) + G(F with a, b merged): the Q-target of a step."""
    labels, _ = _synthetic_labels(7, extra_edges=4, seed=seed)
    forest, _ = _random_forest(labels, 2, random.Random(seed))
    q_values = [logsumexp([labels.logcard[a | b], bushy.forest_cost_to_go(labels, bushy.merge(forest, a, b))])
                for a, b in bushy.candidate_merges(forest, labels.neighbour_masks)]
    assert bushy.forest_cost_to_go(labels, forest) == pytest.approx(min(q_values), abs=1e-9)


def test_bushy_beats_left_deep_when_the_middle_explodes():
    # Path 0-1-2-3: every left-deep plan builds a 3-pattern subset (huge); ((0,1),(2,3)) avoids them.
    neighbour_masks = [0b0010, 0b0101, 0b1010, 0b0100]
    logcard = {1: 5.0, 2: 5.0, 4: 5.0, 8: 5.0, 0b0011: 1.0, 0b0110: 9.0, 0b1100: 1.0,
               0b0111: 20.0, 0b1110: 20.0, 0b1111: 2.0}
    labels = QueryLabels(query="q", family="path", n_tp=4, neighbour_masks=neighbour_masks, logcard=logcard)
    tree, log_cost = bushy.optimal_bushy_plan(labels)
    assert bushy.canonical(tree) == ((0, 1), (2, 3))
    assert log_cost == pytest.approx(logsumexp([1.0, 1.0, 2.0]))
    assert not bushy.is_left_deep(tree)


def test_disconnected_query_has_no_plan():
    labels = QueryLabels(query="q", family="x", n_tp=2, neighbour_masks=[0, 0], logcard={1: 1.0, 2: 1.0})
    assert bushy.optimal_bushy_plan(labels) == (None, math.inf)


def test_tree_and_merge_conversions():
    tree = ((0, (1, 2)), (3, 4))
    merges = bushy.tree_to_merges(tree)
    assert merges == [(0b00010, 0b00100), (0b00001, 0b00110), (0b01000, 0b10000), (0b00111, 0b11000)]
    assert bushy.merges_to_tree(5, merges) == tree
    assert bushy.forest_after(tree, 5, 2) == (0b00111, 0b01000, 0b10000)
    assert bushy.internal_masks(tree) == [a | b for a, b in merges]
    assert bushy.from_json(bushy.to_json(tree)) == tree
    order = [2, 0, 1, 3]
    assert bushy.tree_from_order(order) == (((2, 0), 1), 3)
    assert bushy.is_left_deep(bushy.tree_from_order(order)) and bushy.is_left_deep((0, (1, 2)))
    assert not bushy.is_left_deep(tree)
    assert bushy.tree_mask(tree) == mask_of(range(5)) and members(bushy.tree_mask((3, 4))) == [3, 4]


def _dpsub_log_cost(labels):
    """Reference: DPsub over all submask splits of every connected subset (plain log space)."""
    b = {m: NEG_INF for m in labels.logcard if not m & (m - 1)}
    for mask in sorted((m for m in labels.logcard if m & (m - 1)), key=int.bit_count):
        low, rest = mask & -mask, mask ^ (mask & -mask)
        options, sub = [], rest
        while True:
            left = low | sub
            if left != mask and left in b and mask ^ left in b:
                options.append(logsumexp([b[left], b[mask ^ left]]))
            if sub == 0:
                break
            sub = (sub - 1) & rest
        b[mask] = logsumexp([labels.logcard[mask], min(options)])
    return b[labels.full_mask]


@pytest.mark.parametrize("seed", range(8))
def test_dpccp_matches_dpsub_on_larger_graphs(seed):
    labels, _ = _synthetic_labels(11, extra_edges=seed * 4, seed=seed)
    assert bushy.optimal_bushy_plan(labels)[1] == pytest.approx(_dpsub_log_cost(labels), abs=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_csg_cmp_pairs_are_every_valid_split_once_in_dp_order(seed):
    labels, _ = _synthetic_labels(8, extra_edges=seed * 2, seed=seed)
    n, nbr = labels.n_tp, labels.neighbour_masks
    pairs = list(csg_cmp_pairs(n, nbr))
    assert set(connected_subgraphs(n, nbr)) == set(labels.logcard)
    expected = {frozenset((left, mask ^ left)) for mask in labels.logcard for left in labels.logcard
                if left & mask == left and left != mask and mask ^ left in labels.logcard}
    assert len(pairs) == len(expected) and {frozenset(p) for p in pairs} == expected
    done = {}                                    # union -> index of the last pair that builds it
    for index, (a, b) in enumerate(pairs):
        done[a | b] = index
    for index, (a, b) in enumerate(pairs):       # both halves are complete before they are used
        assert done.get(a, -1) < index and done.get(b, -1) < index


def _goo(labels):
    """Greedy operator ordering, written out directly: merge the pair with the smallest result card."""
    forest, trees = bushy.initial_forest(labels.n_tp), {1 << i: i for i in range(labels.n_tp)}
    while len(forest) > 1:
        a, b = min(bushy.candidate_merges(forest, labels.neighbour_masks), key=lambda m: labels.logcard[m[0] | m[1]])
        trees[a | b] = (trees.pop(a), trees.pop(b))
        forest = bushy.merge(forest, a, b)
    return next(iter(trees.values()))


@pytest.mark.parametrize("seed", range(10))
def test_estimator_beam_is_goo_at_width_one_and_the_exact_dp_unbounded(seed):
    labels, _ = _synthetic_labels(7, extra_edges=seed % 4, seed=seed)
    calls = []
    def log_cards(masks):
        calls.append(masks)
        return [labels.logcard[m] for m in masks]
    tree, estimated = bushy.estimator_beam_plan(7, labels.neighbour_masks, log_cards, width=1)
    assert tree == _goo(labels)
    assert estimated == sum(len(c) for c in calls) and all(len(set(c)) == len(c) for c in calls)
    for width in (2, 4):
        assert bushy.is_valid_tree(bushy.estimator_beam_plan(7, labels.neighbour_masks, log_cards, width)[0], 7, labels.neighbour_masks)
    tree, _ = bushy.estimator_beam_plan(7, labels.neighbour_masks, log_cards, width=10 ** 6)
    assert abs(bushy.tree_log_cost(tree, labels.logcard) - bushy.optimal_bushy_plan(labels)[1]) < 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_repair_completion_is_a_valid_plan_and_follows_the_tree_on_its_path(seed):
    labels, _ = _synthetic_labels(7, extra_edges=seed % 4, seed=seed)
    rng = random.Random(seed)
    tree, _ = bushy.optimal_bushy_plan(labels)
    for steps in range(6):
        forest, trees = _random_forest(labels, steps, rng)
        merges = bushy.repair_completion(tree, forest)
        state = forest
        for a, b in merges:
            assert (a, b) in bushy.candidate_merges(state, labels.neighbour_masks) or (b, a) in bushy.candidate_merges(state, labels.neighbour_masks)
            state = bushy.merge(state, a, b)
        assert len(state) == 1
        on_path = bushy.forest_after(tree, 7, steps)
        assert bushy.repair_completion(tree, on_path) == bushy.tree_to_merges(tree)[steps:]


@pytest.mark.parametrize("seed", range(8))
def test_stitched_cost_to_go_is_exact_on_all_subsets_and_an_upper_bound_on_plans(seed):
    labels, _ = _synthetic_labels(7, extra_edges=seed % 4, seed=seed)
    rng = random.Random(seed)
    tree, _ = bushy.optimal_bushy_plan(labels)
    for steps in range(6):
        forest, _ = _random_forest(labels, steps, rng)
        exact = bushy.forest_cost_to_go(labels, forest)
        assert bushy.stitched_cost_to_go(forest, labels.logcard, labels.logcard) == pytest.approx(exact, abs=1e-9)
        repair = [a | b for a, b in bushy.repair_completion(tree, forest)]
        repair_cost = logsumexp([labels.logcard[m] for m in repair])
        stitched = bushy.stitched_cost_to_go(forest, set(bushy.internal_masks(tree)) | set(repair), labels.logcard)
        assert exact - 1e-9 <= stitched <= repair_cost + 1e-9
    assert bushy.stitched_cost_to_go(bushy.initial_forest(7), [labels.full_mask], labels.logcard) == math.inf
    assert bushy.stitched_cost_to_go((labels.full_mask,), [], labels.logcard) == NEG_INF
