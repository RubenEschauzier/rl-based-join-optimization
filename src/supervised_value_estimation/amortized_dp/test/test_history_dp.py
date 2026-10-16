import math

import pytest
import torch

from src.supervised_value_estimation.amortized_dp import history_dp, soft_dp
from src.supervised_value_estimation.amortized_dp.test.test_amortized_dp import _synthetic_labels

DEVICE = "cpu"


def _part(n, seed, family=None):
    labels, _ = _synthetic_labels(n, extra_edges=3, seed=seed)
    s = soft_dp.SubsetStructure(labels.neighbour_masks, DEVICE, family=family(labels) if family else None)
    g = torch.Generator().manual_seed(seed)
    S = len(s)
    true = torch.tensor([labels.logcard[m] for m in s.masks.tolist()], dtype=torch.float32)
    part = {"structure": s, "exact": torch.rand(S, generator=g) * 10, "true": true, "label": true.clone(),
            "failed": torch.zeros(S, dtype=torch.bool)}
    return part, torch.randn(S, 200, generator=g), torch.rand(S, generator=g) * 12


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("tau", [0.0, 0.3])
def test_init_is_the_cardinality_model(seed, tau):
    """delta = 0 at init: the estimates are the model's (exact single patterns) and the split distribution is soft_dp's."""
    part, phi, direct = _part(7, seed)
    s = part["structure"]
    b = history_dp.merge([part], DEVICE)
    L, logb, log_p = history_dp.HistoryDP()(b, phi, direct, tau)
    L0 = torch.where(s.singleton, part["exact"], direct)
    assert torch.allclose(L, L0)
    ref_b, ref_p = soft_dp.soft_dp(s, L0, tau)
    assert torch.allclose(logb, ref_b, atol=1e-4)
    for a, r in zip(log_p, ref_p):
        assert torch.allclose(torch.exp(a), torch.exp(r), atol=1e-5)
    if tau == 0.0:
        assert soft_dp.hard_plan(s, log_p) == soft_dp.hard_plan(s, ref_p)


def test_merged_queries_match_separate_ones():
    parts = [_part(6, 1), _part(8, 2), _part(5, 3)]
    net = history_dp.HistoryDP()
    torch.nn.init.normal_(net.delta[-1].weight, std=0.5)       # a non-trivial head
    b = history_dp.merge([p for p, _, _ in parts], DEVICE)
    L, _, log_p = net(b, torch.cat([phi for _, phi, _ in parts]), torch.cat([d for _, _, d in parts]), 0.3)
    mu = history_dp.marginals(b, log_p)
    offset = 0
    for part, phi, direct in parts:
        s = part["structure"]
        bi = history_dp.merge([part], DEVICE)
        Li, _, lpi = net(bi, phi, direct, 0.3)
        assert torch.allclose(L[offset:offset + len(s)], Li, atol=1e-5)
        assert torch.allclose(mu[offset:offset + len(s)], soft_dp.plan_marginals(s, lpi), atol=1e-5)
        offset += len(s)


def test_plan_loss_matches_soft_dp():
    part, phi, direct = _part(7, 5)
    s = part["structure"]
    b = history_dp.merge([part], DEVICE)
    _, _, log_p = history_dp.HistoryDP()(b, phi, direct, 0.3)
    best = soft_dp.soft_dp(s, part["true"], 0.0)[0][s.full]
    loss, ok = history_dp.plan_losses(b, log_p, best.reshape(1))
    ref = soft_dp.expected_log_cost(s, soft_dp.plan_marginals(s, log_p), part["true"]) - best
    assert bool(ok[0]) and math.isclose(float(loss[0]), float(ref), abs_tol=1e-4)


def test_plan_loss_on_a_counted_family():
    """Over a restricted family (counted subsets), as in training."""
    keep = lambda labels: [m for m in labels.logcard if bin(m).count("1") in (2, 4) or m == (1 << labels.n_tp) - 1]
    part, phi, direct = _part(6, 7, family=keep)
    s = part["structure"]
    b = history_dp.merge([part], DEVICE)
    _, _, log_p = history_dp.HistoryDP()(b, phi, direct, 0.3)
    best = soft_dp.soft_dp(s, part["true"], 0.0)[0][s.full]
    loss, ok = history_dp.plan_losses(b, log_p, best.reshape(1))
    ref = soft_dp.expected_log_cost(s, soft_dp.plan_marginals(s, log_p), part["true"]) - best
    assert bool(ok[0]) and math.isclose(float(loss[0]), float(ref), abs_tol=1e-4)


def test_degree_bound():
    patterns = ["?a <p> ?b .", "?b <q> ?c .", "?c <r> <C> .", "?a <q> ?d ."]
    from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import neighbour_masks_of
    s = soft_dp.SubsetStructure(neighbour_masks_of(patterns), DEVICE)
    est = torch.full((len(s),), 20.0, dtype=torch.float64)
    est[s.singleton] = 3.0
    loose = history_dp.degree_bound(patterns, s, est, {p: {"s": {"max": 1e12}, "o": {"max": 1e12}} for p in ("<p>", "<q>", "<r>")})
    masks = s.masks.tolist()
    with_constant = masks.index(0b0110)                         # ?b q ?c . ?c r <C>: each row matches <= 1 triple
    assert float(loose[with_constant]) == 3.0
    others = [i for i, m in enumerate(masks) if m != 0b0110 and not (m & 0b0100 and m & 0b0010)]
    assert torch.allclose(loose[others], est[others])           # huge degrees bound nothing else
    tight = history_dp.degree_bound(patterns, s, est, {p: {"s": {"max": 1.0}, "o": {"max": 1.0}} for p in ("<p>", "<q>", "<r>")})
    assert torch.allclose(tight[~s.singleton], torch.full_like(tight[~s.singleton], 3.0))   # fanout 1: no growth
    assert bool((tight <= est).all())
