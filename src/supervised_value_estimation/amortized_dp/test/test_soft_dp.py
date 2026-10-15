import math
import random

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.baselines.enumeration import connected_subgraphs, csg_cmp_pairs
from src.query_environments.gym.query_gym_estimated_cost import QueryGymEstimatedCost
from src.supervised_value_estimation.amortized_dp import bushy, soft_dp
from src.supervised_value_estimation.amortized_dp.labels import members
from src.supervised_value_estimation.amortized_dp.test.test_amortized_dp import _synthetic_labels

DEVICE = "cpu"


def _logcards(structure, logcard, dtype=torch.float64):
    return torch.tensor([logcard.get(m, 0.0) for m in structure.masks.tolist()], dtype=dtype)


@pytest.mark.parametrize("seed", range(6))
def test_subsets_and_splits_match_dpccp(seed):
    labels, _ = _synthetic_labels(8, extra_edges=4, seed=seed)
    s = soft_dp.SubsetStructure(labels.neighbour_masks, DEVICE)
    assert s.masks.tolist() == sorted(connected_subgraphs(8, labels.neighbour_masks))
    masks = s.masks.tolist()
    mine = {(masks[a], masks[b]) for _, _, s1, s2 in s.levels for a, b in zip(s1.tolist(), s2.tolist())}
    low = lambda m: m & -m
    ref = {(min(a, b, key=low), max(a, b, key=low)) for a, b in csg_cmp_pairs(8, labels.neighbour_masks)}
    assert mine == ref


@pytest.mark.parametrize("seed", range(6))
def test_hard_dp_is_the_bushy_optimum(seed):
    labels, _ = _synthetic_labels(7, extra_edges=3, seed=seed)
    s = soft_dp.SubsetStructure(labels.neighbour_masks, DEVICE)
    logb, log_p = soft_dp.soft_dp(s, _logcards(s, labels.logcard), 0.0)
    _, best = bushy.optimal_bushy_plan(labels)
    assert float(logb[s.full]) == pytest.approx(best, abs=1e-9)
    tree = soft_dp.hard_plan(s, log_p)
    assert bushy.is_valid_tree(tree, 7, labels.neighbour_masks)
    assert bushy.tree_log_cost(tree, labels.logcard) == pytest.approx(best, abs=1e-9)
    mu = soft_dp.plan_marginals(s, log_p)       # one-hot: mu marks exactly the plan's subsets
    built = {m for m, x in zip(s.masks.tolist(), mu.tolist()) if x > 0.5 and m & (m - 1)}
    assert built == set(bushy.internal_masks(tree))


@pytest.mark.parametrize("seed", range(4))
def test_marginals_are_a_plan_distribution(seed):
    labels, _ = _synthetic_labels(7, extra_edges=3, seed=seed)
    s = soft_dp.SubsetStructure(labels.neighbour_masks, DEVICE)
    logcard = _logcards(s, labels.logcard)
    logb, log_p = soft_dp.soft_dp(s, logcard, 0.7)
    for (_, u, _, _), lp in zip(s.levels, log_p):     # p sums to one per subset
        total = torch.zeros(len(s), dtype=lp.dtype).index_add(0, u, lp.exp())
        assert torch.allclose(total[torch.unique(u)], torch.ones(1, dtype=lp.dtype))
    mu = soft_dp.plan_marginals(s, log_p)
    assert torch.allclose(mu[s.singleton], torch.ones(1, dtype=mu.dtype))
    assert float(mu[s.full]) == pytest.approx(1.0)
    assert float(mu[~s.singleton].sum()) == pytest.approx(6.0)        # a plan builds n - 1 joins
    # the expected cost from the marginals is the DP's own expected cost
    assert float(soft_dp.expected_log_cost(s, mu, logcard)) == pytest.approx(float(logb[s.full]), abs=1e-9)
    _, best = bushy.optimal_bushy_plan(labels)
    assert float(logb[s.full]) >= best - 1e-9


def test_soft_dp_tends_to_the_hard_dp():
    labels, _ = _synthetic_labels(7, extra_edges=3, seed=1)
    s = soft_dp.SubsetStructure(labels.neighbour_masks, DEVICE)
    logcard = _logcards(s, labels.logcard)
    hard = soft_dp.soft_dp(s, logcard, 0.0)[0][s.full]
    assert float(soft_dp.soft_dp(s, logcard, 1e-3)[0][s.full]) == pytest.approx(float(hard), abs=1e-3)


def test_gradients_are_exact():
    labels, _ = _synthetic_labels(5, extra_edges=2, seed=3)
    s = soft_dp.SubsetStructure(labels.neighbour_masks, DEVICE)
    true = _logcards(s, labels.logcard)
    estimate = (true + torch.randn(len(s), dtype=torch.float64)).requires_grad_()

    def loss(logcard):
        _, log_p = soft_dp.soft_dp(s, logcard, 0.5)
        return soft_dp.expected_log_cost(s, soft_dp.plan_marginals(s, log_p), true)
    assert torch.autograd.gradcheck(loss, (estimate,))
    assert torch.autograd.gradcheck(lambda c: soft_dp.soft_dp(s, c, 0.5)[0][s.full], (estimate,))


@pytest.mark.parametrize("seed", range(4))
def test_family_dp_is_the_stitched_optimum(seed):
    labels, _ = _synthetic_labels(7, extra_edges=3, seed=seed)
    rng = random.Random(seed)
    joins = [m for m in labels.logcard if m & (m - 1)]
    family = set(rng.sample(joins, len(joins) // 3)) | set(bushy.internal_masks(bushy.optimal_bushy_plan(labels)[0]))
    family |= set(bushy.internal_masks(bushy.tree_from_order(range(7)))) & set(joins)
    s = soft_dp.SubsetStructure(labels.neighbour_masks, DEVICE, family=family)
    assert set(s.masks.tolist()) == family | {1 << i for i in range(7)}
    logb, log_p = soft_dp.soft_dp(s, _logcards(s, labels.logcard), 0.0)
    stitched = bushy.StitchTable(family, labels.logcard).cost_to_go(bushy.initial_forest(7))
    assert float(logb[s.full]) == pytest.approx(stitched, abs=1e-9)
    assert set(bushy.internal_masks(soft_dp.hard_plan(s, log_p))) <= family
    mu = soft_dp.plan_marginals(s, soft_dp.soft_dp(s, _logcards(s, labels.logcard), 0.7)[1])
    assert torch.isfinite(mu).all() and float(mu[~s.singleton].sum()) == pytest.approx(6.0)


def _synthetic_query(n, seed):
    """A query graph as the datasets store it: term nodes, pattern i = edges 2i (s -> o) and 2i + 1."""
    rng = random.Random(seed)
    patterns, edges = [], []
    for i in range(n):
        a = 0 if i == 0 else rng.randrange(i + 1)           # connected: shares a variable with an earlier one
        b = i + 1
        if rng.random() < 0.5:
            a, b = b, a
        patterns.append(f"?v{a} <p{i}> ?v{b}")
        edges += [(a, b), (b, a)]
    nodes = n + 1
    x = torch.cat([torch.arange(nodes, dtype=torch.float32)[:, None], torch.randn(nodes, 3)], 1)
    return Data(x=x, edge_index=torch.tensor(edges).T, edge_attr=torch.randn(2 * n, 2),
                term_to_id={f"?v{i}": i for i in range(nodes)}, triple_patterns=patterns, query="q", type="star")


def _edge_sets(x, edge_index, edge_attr, batch):
    """Per graph: the set of (original source id, original target id, edge attribute) edges."""
    out = {}
    for e in range(edge_index.shape[1]):
        a, b = int(edge_index[0, e]), int(edge_index[1, e])
        out.setdefault(int(batch[a]), set()).add((int(x[a, 0]), int(x[b, 0]), tuple(edge_attr[e].tolist())))
    return out


def test_subset_batch_matches_reduced_form_queries():
    q = _synthetic_query(6, seed=2)
    nb = [0] * 6
    for i in range(6):
        for j in range(6):
            terms = lambda k: {q.triple_patterns[k].split()[0], q.triple_patterns[k].split()[2]}
            if i != j and terms(i) & terms(j):
                nb[i] |= 1 << j
    s = soft_dp.SubsetStructure(nb, DEVICE)
    mine = soft_dp.subset_batch(q, s.masks, DEVICE)
    ref = Batch.from_data_list([QueryGymEstimatedCost.reduced_form_query(q, members(m), m.bit_count())
                                for m in s.masks.tolist()])
    assert torch.equal(torch.bincount(mine[3]), torch.bincount(ref.batch))            # same nodes per subset
    assert _edge_sets(*mine) == _edge_sets(ref.x, ref.edge_index, ref.edge_attr, ref.batch)
    assert sorted(map(tuple, mine[0].tolist())) == sorted(map(tuple, ref.x.tolist()))
