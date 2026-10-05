import copy
import itertools
import random

from src.supervised_value_estimation.amortized_dp.labels import QueryLabels, mask_of
from src.supervised_value_estimation.amortized_dp.online.plan_quality import (
    join_cout, optimal_join_cout, p_error, ratio_summary, tree_join_cout,
)
from src.supervised_value_estimation.amortized_dp.online.test.test_runtime_tree import PLAN, PATTERNS, _real_tree
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import count_query


def _random_query(n_tp, seed):
    """A random connected join graph with random true row counts on its connected subsets."""
    rng = random.Random(seed)
    edges = {(i, rng.randrange(i)) for i in range(1, n_tp)}                  # spanning tree
    edges |= {tuple(sorted(rng.sample(range(n_tp), 2))) for _ in range(rng.randrange(n_tp))}
    neighbour_masks = [0] * n_tp
    for i, j in edges:
        neighbour_masks[i] |= 1 << j
        neighbour_masks[j] |= 1 << i

    def connected(mask):
        start = mask & -mask
        reached, frontier = start, start
        while frontier:
            grow = 0
            for i in range(n_tp):
                if frontier >> i & 1:
                    grow |= neighbour_masks[i]
            frontier = grow & mask & ~reached
            reached |= frontier
        return reached == mask

    masks = [m for m in range(1, 1 << n_tp) if connected(m)]
    rows = {m: (rng.choice([0, 1, 10, 1000, 10 ** 6]) if m.bit_count() > 1 else rng.randrange(1, 10 ** 6))
            for m in masks}
    labels = QueryLabels(query="q", family="test", n_tp=n_tp, neighbour_masks=neighbour_masks,
                         logcard={m: 0.0 for m in masks})
    return labels, rows


def _brute_force_optimum(labels, rows):
    best = None
    for order in itertools.permutations(range(labels.n_tp)):
        if all(mask_of(order[:k]) in rows for k in range(2, labels.n_tp + 1)):   # no cartesian product
            cost = join_cout(order, rows)
            best = cost if best is None else min(best, cost)
    return best


def test_optimal_join_cout_matches_brute_force_and_its_plan_achieves_it():
    for seed in range(40):
        labels, rows = _random_query(n_tp=random.Random(seed).randrange(2, 7), seed=seed)
        cost, plan = optimal_join_cout(labels, rows)
        assert cost == _brute_force_optimum(labels, rows)
        assert sorted(plan) == list(range(labels.n_tp)) and join_cout(plan, rows) == cost


def test_join_cout_sums_join_outputs_only_and_leaves_cost_nothing():
    rows = {0b001: 999, 0b010: 999, 0b100: 999, 0b011: 5, 0b111: 7}
    assert join_cout([0, 1, 2], rows) == 12
    assert join_cout([0, 2, 1], rows) is None                  # {0, 2} unknown


def test_unknown_counts_leave_the_optimum_unknown():
    labels, rows = _random_query(5, seed=3)
    rows[max(m for m in rows if m.bit_count() == 2)] = None
    assert optimal_join_cout(labels, rows) == (None, None)


def test_tree_join_cout_on_the_real_qlever_tree():
    # Prefix rows of that plan are 1, 1, 1, 12 (test_runtime_tree), so C_out = 15.
    assert tree_join_cout(_real_tree()) == 15
    unfinished = copy.deepcopy(_real_tree())
    unfinished["status"] = "cancelled"
    assert tree_join_cout(unfinished) is None


def test_p_error_floors_empty_results():
    assert p_error(30, 10) == 3.0
    assert p_error(0, 0) == 1.0 and p_error(5, 0) == 5.0
    assert p_error(None, 10) is None
    summary = ratio_summary([1.0, 1.0, 2.0, 4.0, None])
    assert summary["n"] == 4 and summary["frac_at_most_1"] == 0.5 and abs(summary["geomean"] - 2 ** 0.75) < 1e-9


def test_count_query_uses_only_the_subset():
    query = count_query(PATTERNS, mask_of([1, 3]))
    assert query == ("SELECT (COUNT(*) AS ?count) WHERE { ?s <http://example.com/13000080> "
                     "<http://example.com/10724425> . ?s <http://example.com/13000080> <http://example.com/8719681> . }")
    assert PLAN  # same fixture module as the runtime-tree tests
