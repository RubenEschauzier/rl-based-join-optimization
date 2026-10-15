import math
import random

import pytest
import torch

from src.supervised_value_estimation.amortized_dp import bushy, soft_dp
from src.supervised_value_estimation.amortized_dp.online.train_plan_loss import failed_bound, multi_query_batch, plan_cost, query_loss
from src.supervised_value_estimation.amortized_dp.test.test_amortized_dp import _synthetic_labels
from src.supervised_value_estimation.amortized_dp.test.test_soft_dp import _synthetic_query


class _Q:
    def __init__(self, n, seed):
        self.data = _synthetic_query(n, seed)
        self.patterns = self.data.triple_patterns
        terms = [set(p.split()[0:3:2]) for p in self.patterns]
        self.neighbour_masks = [sum(1 << j for j in range(n) if j != i and terms[i] & terms[j]) for i in range(n)]


def test_multi_query_batch_is_the_per_query_batches_offset():
    a, b = _Q(5, 1), _Q(6, 2)
    sa, sb = soft_dp.SubsetStructure(a.neighbour_masks, "cpu"), soft_dp.SubsetStructure(b.neighbour_masks, "cpu")
    x, ei, ea, batch = multi_query_batch([(a, sa.masks), (b, sb.masks)], "cpu")
    xa, eia, eaa, ba = soft_dp.subset_batch(a.data, sa.masks, "cpu")
    xb, eib, eab, bb = soft_dp.subset_batch(b.data, sb.masks, "cpu")
    assert torch.equal(x, torch.cat([xa, xb])) and torch.equal(ea, torch.cat([eaa, eab]))
    assert torch.equal(ei, torch.cat([eia, eib + xa.shape[0]], 1))
    assert torch.equal(batch, torch.cat([ba, bb + len(sa)]))


@pytest.mark.parametrize("seed", range(3))
def test_plan_loss_is_a_regret_and_its_gradient_prefers_the_cheaper_plans(seed):
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=seed)
    rng = random.Random(seed)
    counts = {m: int(math.exp(x)) for m, x in labels.logcard.items() if m & (m - 1)}
    family = set(rng.sample(sorted(counts), len(counts) // 2)) | set(bushy.internal_masks(bushy.tree_from_order(range(6))))
    family |= set(bushy.internal_masks(bushy.optimal_bushy_plan(labels)[0]))
    s = soft_dp.SubsetStructure(labels.neighbour_masks, "cpu", family=family)
    masks = s.masks.tolist()
    est = torch.tensor([math.log1p(counts.get(m, 0)) for m in masks], dtype=torch.float64)
    est = (est + torch.randn(len(masks), dtype=torch.float64)).requires_grad_()
    l1, n_terms, plan, floor = query_loss(est, None, (s, masks), counts, 0.5, "cpu", None)
    assert n_terms == len(family) and float(plan) >= -1e-9          # E_p[cost] >= the stitched optimum
    assert 0.0 <= floor <= float(plan) + 1e-9 or floor >= 0.0          # the floor: true counts as estimates
    assert query_loss(torch.tensor([math.log1p(counts.get(m, 0)) for m in masks], dtype=torch.float64), None,
                      (s, masks), counts, 0.5, "cpu", None)[2] == pytest.approx(floor)
    before = float(plan)
    plan.backward()
    with torch.no_grad():                                             # a small step along -grad lowers the regret
        stepped = est - 1e-2 * est.grad
    assert float(query_loss(stepped, None, (s, masks), counts, 0.5, "cpu", None)[2]) < before


def test_plan_cost_needs_every_join_count_and_costs_a_failed_one_at_the_bound():
    tree = ((0, 1), 2)
    assert plan_cost(tree, {0b011: 5, 0b111: 7}) == (12, 0)
    assert plan_cost(tree, {0b111: 7}) == (None, 0)
    assert plan_cost(tree, {0b011: None, 0b111: 7}) == (None, 0)
    assert plan_cost(tree, {0b011: None, 0b111: 7}, bound=14) == (21, 1)


def test_failed_bound_is_twice_the_querys_largest_count():
    assert failed_bound({1: 10, 2: None, 3: 400}, fallback=10 ** 9) == 800
    assert failed_bound({2: None}, fallback=1000) == 2000
    assert failed_bound({1: 0, 2: None}, fallback=1000) == 2


def test_a_failed_count_is_one_sided_and_costs_the_bound():
    labels, _ = _synthetic_labels(4, extra_edges=1, seed=0)
    counts = {m: int(math.exp(x)) for m, x in labels.logcard.items() if m & (m - 1)}
    failed = max(m for m in counts if m != 0b1111 and m & (m - 1))
    counts[failed] = None
    s = soft_dp.SubsetStructure(labels.neighbour_masks, "cpu", family=list(counts))
    masks = s.masks.tolist()
    k = masks.index(failed)
    bound = failed_bound(counts, 0)
    low = torch.zeros(len(masks), dtype=torch.float64)
    high = low.clone()
    high[k] = math.log1p(bound) + 5                                   # above the bound: no L1 for it
    low[k] = math.log1p(bound) - 5                                    # below: penalised by the gap
    l1_high = query_loss(high, None, (s, masks), counts, 0.5, "cpu", bound)[0]
    l1_low = query_loss(low, None, (s, masks), counts, 0.5, "cpu", bound)[0]
    assert float(l1_low - l1_high) == pytest.approx(5.0)
