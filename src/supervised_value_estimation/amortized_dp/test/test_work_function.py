import math
import random

import pytest

from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import neighbour_masks_of
from src.supervised_value_estimation.amortized_dp.work_function import (
    QLeverCost, QLeverParameters, WorkStitchTable, expected_rows_read, join_work, lazy_splits, pattern_variables,
    plan_internal_splits, plan_work, qlever_work, sort_rows, variable_masks, weighted,
)

PATTERNS = ["?a <p> ?b", "?b <q> ?c", "?c <r> <x>"]
VARIABLES = pattern_variables(PATTERNS)


def test_merge_join_reads_both_inputs_and_writes_the_result():
    work = qlever_work((0, 1), {0b011: 50}, [100, 200, 1], VARIABLES)
    assert work["scan_full"] == 300 and work["merge"] == 300 and work["out"] == 50
    assert work["sort"] == 0 and work["scan_lazy"] == 0 and "gallop" not in work
    assert weighted(work, {"out": 1}) == 50                # C_out


def test_no_galloping_inputs_of_very_different_sizes_are_merged():
    work = qlever_work((1, 2), {0b110: 3}, [100, 200_000, 2], VARIABLES)
    assert work["merge"] == 200_002


def test_a_large_scan_under_a_join_is_read_lazily():
    p = QLeverParameters(lazy_rows=1000, block_rows=100)
    work = qlever_work((1, 2), {0b110: 3}, [100, 10_000, 2], VARIABLES, p)
    assert work["scan_full"] == 2 and work["scan_lazy"] == pytest.approx(expected_rows_read(10_000, 2, 100))
    measured = qlever_work((1, 2), {0b110: 3}, [100, 10_000, 2], VARIABLES, p, read={1: 300})
    assert measured["scan_lazy"] == 300 and measured["merge"] == 302


def test_expected_rows_read():
    assert expected_rows_read(10_000, 1, 100) == pytest.approx(100)          # one key: one block
    assert expected_rows_read(10_000, 10 ** 9, 100) == pytest.approx(10_000)  # every block
    assert expected_rows_read(50, 3, 100) == 50                              # a single block


def test_a_join_result_on_other_join_variables_is_sorted():
    # ((0, 1), 2): (0, 1) joins on ?b, its result then joins pattern 2 on ?c -> sorted first
    work = qlever_work(((0, 1), 2), {0b011: 40, 0b111: 5}, [100, 200, 1], VARIABLES)
    assert work["sort"] == pytest.approx(40 * math.log2(40))
    assert qlever_work(((0, 1), 2), {0b011: 40, 0b111: 5}, [100, 200, 1], VARIABLES,
                       QLeverParameters(sorts=False))["sort"] == 0
    assert qlever_work(((0, 1), 2), {0b011: 40}, [100, 200, 1], VARIABLES) is None     # a count is missing


# --- the planning cost -------------------------------------------------------------------------------------------

def test_plan_work_by_hand():
    cost = QLeverCost(out=1.0, read=0.5, sort=0.25, lazy_rows=150)
    vm = variable_masks(PATTERNS)
    sizes = [100, 200, 1]                                # pattern 1 (200 rows) is read lazily
    counts = {0b011: 40, 0b111: 5, 0b110: 7}
    reads = {(0b001, 1): 60, (0b100, 1): 20}
    # ((0, 1), 2): {0,1} reads 60 of pattern 1; its result (on ?b) joins 2 on ?c: sorted first
    w, imputed = plan_work(((0, 1), 2), counts, sizes, vm, reads, cost)
    assert imputed == 0 and w == pytest.approx(41 + 0.5 * 60 + 6 + 0.25 * sort_rows(40))
    # (0, (1, 2)): {1,2} reads 20 of pattern 1; its result (on ?c) joins 0 on ?b: sorted first
    w, _ = plan_work((0, (1, 2)), counts, sizes, vm, reads, cost)
    assert w == pytest.approx(8 + 0.5 * 20 + 6 + 0.25 * sort_rows(7))
    assert plan_work(((0, 1), 2), counts, sizes, vm, {}, cost)[0] is None           # a read is missing
    assert plan_work(((0, 1), 2), {0b011: None, 0b111: 5}, sizes, vm, reads, cost, bound=1000) == (
        pytest.approx(1001 + 30 + 6 + 0.25 * sort_rows(1000)), 1)
    # C_out: out 1, nothing else
    plain = QLeverCost(out=1.0, read=0.0, sort=0.0, lazy_rows=10 ** 9)
    assert plan_work(((0, 1), 2), counts, sizes, vm, {}, plain)[0] == 41 + 6
    assert join_work(cost, 6, [(0b011, 40, True), (0b100, 1, False)], reads, sizes) == pytest.approx(6 + 0.25 * sort_rows(40))
    assert lazy_splits(plan_internal_splits(((0, 1), 2)), sizes, cost) == [(0b001, 0b010)]


def _all_trees(mask, nb):
    """Every bushy tree (no cartesian products) over the patterns of a connected mask."""
    members = [i for i in range(mask.bit_length()) if mask >> i & 1]
    if len(members) == 1:
        return [members[0]]
    out = []
    sub = mask
    while sub:
        sub = (sub - 1) & mask
        rest = mask ^ sub
        if not sub or sub > rest or not _connected(sub, nb) or not _connected(rest, nb):
            continue
        if not any(nb[i] & rest for i in range(mask.bit_length()) if sub >> i & 1):
            continue
        out += [(a, b) for a in _all_trees(sub, nb) for b in _all_trees(rest, nb)]
    return out


def _connected(mask, nb):
    start = mask & -mask
    seen, frontier = start, start
    while frontier:
        reach = 0
        for i in range(mask.bit_length()):
            if frontier >> i & 1:
                reach |= nb[i]
        frontier = reach & mask & ~seen
        seen |= frontier
    return seen == mask


@pytest.mark.parametrize("seed", range(4))
def test_work_stitch_table_is_the_cheapest_plan_with_everything_counted(seed):
    rng = random.Random(seed)
    patterns = ["?a <p0> ?b .", "?b <p1> ?c .", "?b <p2> ?d .", "?d <p3> ?e .", "?a <p4> ?e ."][:4 + seed % 2]
    n = len(patterns)
    nb = neighbour_masks_of(patterns)
    vm = variable_masks(patterns)
    cost = QLeverCost(out=1.0, read=0.7, sort=0.3, lazy_rows=500)
    sizes = [rng.choice([50, 800, 3000]) for _ in range(n)]
    full = (1 << n) - 1
    conn = [m for m in range(1, full + 1) if m & (m - 1) and _connected(m, nb)]
    counts = {m: rng.randint(0, 5000) for m in conn}
    reads = {(o, p): rng.randint(1, sizes[p]) for o in range(1, full + 1) for p in range(n) if not o >> p & 1}
    table = WorkStitchTable(conn, {m: math.log1p(c) for m, c in counts.items()}, sizes, vm, reads, cost)
    brute = min(plan_work(t, counts, sizes, vm, reads, cost)[0] for t in _all_trees(full, nb))
    assert math.exp(table.cost_to_go(bushy.initial_forest(n))) == pytest.approx(brute, rel=1e-9)
    # from a forest: built components with their sort keys, the rest of the plan stitched
    t = min(_all_trees(full, nb), key=lambda x: plan_work(x, counts, sizes, vm, reads, cost)[0])
    sub = t[0] if not bushy.is_leaf(t[0]) else t[1]
    if not bushy.is_leaf(sub):
        forest = tuple(sorted([bushy.tree_mask(sub)] + [1 << i for i in range(n) if not bushy.tree_mask(sub) >> i & 1]))
        jv = (lambda a, b: _vars(a, vm) & _vars(b, vm))(bushy.tree_mask(sub[0]), bushy.tree_mask(sub[1]))
        rest = [x for x in _all_trees(full, nb) if _has_subtree(x, sub)]
        built = plan_work(sub, counts, sizes, vm, reads, cost)[0]
        expect = min(plan_work(x, counts, sizes, vm, reads, cost)[0] for x in rest) - built
        assert math.exp(table.cost_to_go(forest, {bushy.tree_mask(sub): jv})) == pytest.approx(expect, rel=1e-9)


def _vars(mask, vm):
    v = 0
    for i, m in enumerate(vm):
        if mask >> i & 1:
            v |= m
    return v


def _has_subtree(tree, sub):
    if bushy.canonical(tree) == bushy.canonical(sub):
        return True
    return not bushy.is_leaf(tree) and (_has_subtree(tree[0], sub) or _has_subtree(tree[1], sub))


def test_work_stitch_table_skips_splits_without_a_measured_read():
    patterns = ["?a <p0> ?b .", "?b <p1> ?c ."]
    vm = variable_masks(patterns)
    cost = QLeverCost(lazy_rows=100)
    table = WorkStitchTable([0b11], {0b11: math.log1p(9)}, [5, 1000], vm, {}, cost)
    assert table.cost_to_go((0b01, 0b10)) == math.inf
    table = WorkStitchTable([0b11], {0b11: math.log1p(9)}, [5, 1000], vm, {(0b01, 1): 40}, cost)
    assert math.exp(table.cost_to_go((0b01, 0b10))) == pytest.approx(10 + 0.98 * 40)
