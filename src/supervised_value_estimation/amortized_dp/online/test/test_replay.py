import math

import numpy as np

from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.online.train_forest_value import (importance_weights, realized_regret,
                                                                                    replay_probabilities)


def test_realized_regret_from_counts():
    # 3 patterns in a path 0-1-2: plans ((0,1),2) and (0,(1,2)); counts of their joins
    counts = {0b011: 100, 0b110: 10, 0b111: 50}
    left, right = bushy.canonical(((0, 1), 2)), bushy.canonical((0, (1, 2)))
    assert math.isclose(realized_regret(left, [left, right], counts, 10 ** 6), math.log(150) - math.log(60))
    assert realized_regret(right, [left, right], counts, 10 ** 6) == 0.0
    assert realized_regret(left, [left], counts, 10 ** 6) == 0.0
    counts[0b110] = None                                       # a failed count costs the bound
    assert math.isclose(realized_regret(right, [left, right], counts, 1000), math.log(1050) - math.log(150))
    assert realized_regret(left, [left], {0b111: 5}, 10) is None   # a missing count: no regret


def test_replay_probabilities_and_weights():
    regrets = np.array([0.0, 0.0, 1.0, 10.0])
    p = replay_probabilities(regrets, alpha=0.6, floor=0.2, eps=0.05, cap=4.6)
    assert np.isclose(p.sum(), 1) and np.all(p >= 0.2 / 4 - 1e-12)
    assert p[3] > p[2] > p[0] == p[1]
    capped = replay_probabilities(np.array([0.0, 0.0, 1.0, 4.6]), alpha=0.6, floor=0.2, eps=0.05, cap=4.6)
    assert np.allclose(p, capped)                              # above the cap, regret no longer counts
    assert np.allclose(replay_probabilities(np.zeros(5), 0.6, 0.2, 0.05, 4.6), 0.2)
    w = importance_weights(p, beta=1.0)
    assert np.isclose(w.max(), 1) and np.allclose(w * p, (w * p)[0])   # beta 1: w ~ 1 / p, uniform in expectation
    assert np.allclose(importance_weights(p, beta=0.0), 1)


def test_decision_uncertainty():
    """D is 0 when every index prefers the deployed plan, and the mean log-cost gap to each index's best plan otherwise."""
    import torch
    from src.supervised_value_estimation.amortized_dp import forest_value as fv
    from src.supervised_value_estimation.amortized_dp.online import train_forest_value as t
    from src.supervised_value_estimation.amortized_dp.test.test_forest_epinet import _epinet
    from src.supervised_value_estimation.amortized_dp.test.test_forest_value import _encoder, _net, _query
    n = 8
    patterns, nb, degrees = _query(n, 1)
    encode, _ = _encoder(n, 1)
    net, epi = _net(1, trained=True), _epinet(1)
    ctx = fv.QueryContext(n, nb, patterns, degrees, encode)
    trees = fv.beam_plans(net, ctx, [8], all_trees=[True])[0][0]
    d, share = t.decision_uncertainty(net, epi, ctx, trees, torch.zeros(4, 4))   # z = 0: all agree with the deployed
    assert d == 0.0 and share == 0.0
    epi.scale_card.fill_(4.0)
    Z = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    d, share = t.decision_uncertainty(net, epi, ctx, trees, Z)
    S = fv.plan_scores(net, ctx, trees, epi, Z)
    assert math.isclose(d, float((S[:, 0] - S.min(1).values).mean()), rel_tol=1e-6) and d >= 0
    assert (d > 0) == (share > 0)
