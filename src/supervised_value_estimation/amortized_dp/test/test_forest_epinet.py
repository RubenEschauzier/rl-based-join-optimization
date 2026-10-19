import random

import pytest
import torch

from src.supervised_value_estimation.amortized_dp import bushy, forest_epinet as fe, forest_value as fv
from src.supervised_value_estimation.amortized_dp.test.test_forest_value import D, _counted_part, _encoder, _net, _query


def _epinet(seed=0, trained=True):
    torch.manual_seed(seed)
    epi = fe.ForestEpinet(phi_dim=D, value_dim=16, index_dim=4, hidden_dim=8, prior_hidden_dim=8)
    if trained:                                          # a learnable part that is not zero
        for head in (epi.card.learnable_epinet_heads["card"], epi.value.learnable_epinet_heads["value"]):
            torch.nn.init.normal_(head.weight, std=0.5)
    return epi


@pytest.mark.parametrize("seed", range(3))
def test_index_zero_is_the_plain_planner(seed):
    n = 8
    patterns, nb, degrees = _query(n, seed)
    encode, _ = _encoder(n, seed)
    net, epi = _net(seed, trained=True), _epinet(seed)
    plain, _ = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), [1, 4, 8], all_trees=[False, True, False])
    zero, _ = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), [1, 4, 8], all_trees=[False, True, False],
                            epinet=epi, indexes=torch.zeros(3, 4))
    assert zero == plain


def test_thompson_samples_are_coherent_plans():
    """A group's index is one model for its whole plan: the same index gives the same plan in any batch, other
    indexes give other (valid) plans."""
    n = 9
    patterns, nb, degrees = _query(n, 5)
    encode, _ = _encoder(n, 5)
    net, epi = _net(5, trained=True), _epinet(5)
    epi.scale_card.fill_(3.0), epi.scale_value.fill_(3.0)
    Z = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    together, _ = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), [8] * 6, epinet=epi, indexes=Z)
    for g in range(6):
        alone, _ = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), [8], epinet=epi, indexes=Z[g:g + 1])
        assert alone[0] == together[g]
    trees = [bushy.canonical(t[0]) for t in together]
    assert all(bushy.is_valid_tree(t, n, nb) for t in trees)
    plain, _ = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), [8])
    assert any(t != bushy.canonical(plain[0][0]) for t in trees), "samples never change the plan"


def test_losses_train_only_the_epinet():
    part, nb, encode = _counted_part(7, 4)
    part.update(fe.anchor_keys(part, "q"))
    net, epi = _net(4, trained=True), _epinet(4, trained=False)
    phi, l = encode(part["masks"].tolist())
    phi, l = phi.clone().requires_grad_(), l.clone().requires_grad_()
    out = fv.forward_batch(net, [part], phi, l, "cpu", 0.5, stop_gradient=True, hidden=True)
    res = fe.losses(epi, net, out, phi, sigma=0.05, n_indexes=16, max_rows=1000, generator=torch.Generator().manual_seed(0))
    assert float(res["card"]) > 0 and float(res["value"]) > 0
    (res["card"] + res["value"]).backward()
    assert phi.grad is None and l.grad is None
    assert all(p.grad is None for p in net.parameters())
    learnable = [p for p in epi.trainable() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert learnable, "no gradient reaches the learnable epinet"
    frozen = list(epi.card.prior_epinet_features.parameters()) + list(epi.value.prior_epinet_heads.parameters())
    assert all(not p.requires_grad and p.grad is None for p in frozen)


def test_anchor_keys_follow_the_forest_not_its_position():
    n = 7
    patterns, nb, degrees = _query(n, 3)
    encode, _ = _encoder(n, 3)
    trees = [fv.beam_plan(_net(s), fv.QueryContext(n, nb, patterns, degrees, encode), 2)[0] for s in (0, 1)]
    from src.supervised_value_estimation.amortized_dp.soft_dp import connected_masks
    conn = torch.nonzero(connected_masks(nb, "cpu")).squeeze(1).tolist()
    counts = {m: random.Random(m).randint(0, 1000) for m in conn if m & (m - 1)}
    keys = []
    for plans in (trees, trees[::-1]):
        part = fv.build_query_data(n, nb, patterns, degrees, plans, counts, 10 ** 9)
        k = fe.anchor_keys(part, "q")
        node_mask, masks = part["node_mask"].tolist(), part["masks"].tolist()
        forests = [tuple(sorted(masks[node_mask[i]] for i in row if i >= 0)) for row in part["forest_nodes"].tolist()]
        keys.append(dict(zip(forests, k["forest_keys"].tolist())))
    shared = set(keys[0]) & set(keys[1])
    assert shared and all(keys[0][f] == keys[1][f] for f in shared)


def test_query_priorities():
    net, epi = _net(0, trained=True), _epinet(0)
    items = []
    for seed, n in ((0, 5), (1, 8), (2, 2)):
        patterns, nb, _ = _query(n, seed)
        encode, _ = _encoder(n, seed)
        phi, l = encode([1 << i for i in range(n)] + [(1 << n) - 1])
        items.append((phi, l, nb))
    g = torch.Generator().manual_seed(0)
    p = fe.query_priorities(net, epi, items, "cpu", n_indexes=64, generator=g)
    assert p.shape == (3,) and bool((p >= 0).all()) and bool((p > 0).any())
    epi.scale_card.fill_(0.0), epi.scale_value.fill_(0.0)
    assert torch.allclose(fe.query_priorities(net, epi, items, "cpu", n_indexes=16), torch.zeros(3))


def test_calibration_of_a_calibrated_head():
    g = torch.Generator().manual_seed(0)
    N, K = 4000, 256
    base = torch.randn(N, generator=g)
    std = torch.rand(N, generator=g) * 2 + 0.1
    true = base + std * torch.randn(N, generator=g)
    samples = base + std * torch.randn(K, N, generator=g)
    cal = fe.calibration(samples, base, true)
    assert abs(cal["coverage90"] - 0.9) < 0.03 and abs(cal["fitted_scale"] - 1) < 0.1 and cal["spearman"] > 0.3
    cal = fe.calibration(base + 0.25 * std * torch.randn(K, N, generator=g), base, true)       # 4x too narrow
    assert cal["coverage90"] < 0.5 and abs(fe.fit_scale(1.0, cal, step=1.0) - 4) < 0.5 and abs(fe.fit_scale(1.0, cal) - 2) < 0.3


def test_value_samples_move_the_query_estimate_with_the_same_index():
    """V_z shifts l(Q) with the same z as W (as planning does): at zero scales it is the plain V, and with the subset
    head on it can fall below l(Q), which a shift of W alone never can."""
    part, nb, encode = _counted_part(7, 4)
    net, epi = _net(4, trained=True), _epinet(4)
    phi, l = encode(part["masks"].tolist())
    out = fv.forward_batch(net, [part], phi, l, "cpu", 0.5, stop_gradient=True, hidden=True)
    f = (out["forest_k"] >= 3).nonzero().squeeze(1)
    assert len(f)
    Z = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
    plain = fe.forest_value_samples(epi, net, out, phi, f, Z, card_scale=0.0, value_scale=0.0)
    assert torch.allclose(plain, out["V"][f].expand_as(plain), atol=1e-5)
    lq = out["l"][out["forest_q"][f]]
    w_only = fe.forest_value_samples(epi, net, out, phi, f, Z, card_scale=0.0, value_scale=5.0)
    assert bool((w_only >= lq - 1e-5).all())
    joint = fe.forest_value_samples(epi, net, out, phi, f, Z, card_scale=3.0, value_scale=1.0)
    assert bool((joint < lq - 0.1).any())


def test_value_head_on_the_w_target():
    """value_target "w": the value head is fitted to W* and gets a gradient; V's target is not involved."""
    part, nb, encode = _counted_part(8, 6)
    part.update(fe.anchor_keys(part, "q"))
    net, epi = _net(6, trained=True), _epinet(6, trained=False)
    epi.alpha_value.fill_(0.0)                                 # no prior and a zero last layer: the head adds exactly 0
    phi, l = encode(part["masks"].tolist())
    out = fv.forward_batch(net, [part], phi, l, "cpu", 0.5, stop_gradient=True, hidden=True)
    g = torch.Generator().manual_seed(0)
    res = fe.losses(epi, net, out, phi, 0.0, 16, 1000, g, value_target="w")
    ok = ~torch.isnan(out["w_true"])
    assert torch.isclose(res["value"], (out["W"][ok] - out["w_true"][ok]).abs().mean(), atol=1e-5)
    res["value"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in epi.value.get_learnable_epinet_params())
    out["g_true"].fill_(float("nan"))                          # V's target plays no part
    assert torch.isfinite(fe.losses(epi, net, out, phi, 0.0, 16, 1000, g, value_target="w")["value"])


@pytest.mark.parametrize("seed", range(3))
def test_plan_scores_match_the_beam_and_the_training_pass(seed):
    n = 8
    patterns, nb, degrees = _query(n, seed)
    encode, _ = _encoder(n, seed)
    net, epi = _net(seed, trained=True), _epinet(seed)
    epi.scale_card.fill_(2.0)
    ctx = fv.QueryContext(n, nb, patterns, degrees, encode)
    trees = fv.beam_plans(net, ctx, [8], all_trees=[True])[0][0]           # the beam's finished trees, cheapest first
    plain = fv.plan_scores(net, ctx, trees)[0]
    assert bool((plain[1:] >= plain[:-1] - 1e-4).all())
    # the deployed tree's score: logsumexp of l'' over its joins, as the training pass computes them
    part = fv.build_query_data(n, nb, patterns, degrees, [trees[0]], {}, 10 ** 9)
    phi, l = encode(part["masks"].tolist())
    out = fv.forward_batch(net, [part], phi, l, "cpu", 0.5)
    def joins(i):                                         # the plan's own joins: the root and its descendants
        return [i] + [j for c in (int(out["node_c1"][i]), int(out["node_c2"][i])) if c >= 0
                      and int(out["node_height"][c]) >= 1 for j in joins(c)]
    roots = [i for i in range(len(out["node_mask"])) if int(out["node_mask"][i]) == part["full"]]
    totals = torch.stack([torch.logsumexp(out["l2"][joins(r)], 0) for r in roots])
    assert float((totals - plain[0]).abs().min()) < 1e-4
    # under an index: the Thompson beam's finished trees are in order of their score under that index
    Z = torch.randn(3, 4, generator=torch.Generator().manual_seed(seed))
    for j in range(3):
        tz = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), [8], all_trees=[True],
                           epinet=epi, indexes=Z[j:j + 1])[0][0]
        sz = fv.plan_scores(net, ctx, tz, epi, Z[j:j + 1])[0]
        assert bool((sz[1:] >= sz[:-1] - 1e-4).all())
    rows = fv.plan_scores(net, ctx, trees, epi, Z)
    assert all(torch.allclose(rows[j], fv.plan_scores(net, ctx, trees, epi, Z[j:j + 1])[0], atol=1e-5) for j in range(3))
    assert torch.allclose(fv.plan_scores(net, ctx, trees, epi, torch.zeros(1, 4))[0], plain, atol=1e-5)


@pytest.mark.parametrize("stat", ["mean", "p90"])
def test_robust_beam(stat):
    n = 9
    patterns, nb, degrees = _query(n, 7)
    encode, _ = _encoder(n, 7)
    net, epi = _net(7, trained=True), _epinet(7)
    epi.scale_card.fill_(3.0), epi.scale_value.fill_(2.0)
    ctx = lambda: fv.QueryContext(n, nb, patterns, degrees, encode)
    plain = fv.beam_plans(net, ctx(), [1, 8], all_trees=[False, True])[0]
    same = fv.beam_plans(net, ctx(), [1, 8], all_trees=[False, True],
                         robust={"epinet": epi, "indexes": torch.zeros(5, 4), "stat": stat})[0]
    assert same == plain                                       # identical worlds: the plain planner
    Z = torch.randn(16, 4, generator=torch.Generator().manual_seed(1))
    trees = fv.beam_plans(net, ctx(), [8], all_trees=[True], robust={"epinet": epi, "indexes": Z, "stat": stat})[0][0]
    S = fv.plan_scores(net, ctx(), trees, epi, Z)               # the worlds carry each plan's own costs
    score = S.mean(0) if stat == "mean" else torch.quantile(S, 0.9, dim=0)
    assert bool((score[1:] >= score[:-1] - 1e-4).all())         # finished trees, cheapest first by the statistic
    assert all(bushy.is_valid_tree(bushy.canonical(t), n, nb) for t in trees)
