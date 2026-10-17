import math
import random

import pytest
import torch

from src.supervised_value_estimation.amortized_dp import bushy, forest_value as fv
from src.supervised_value_estimation.amortized_dp.labels import QueryLabels
from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import neighbour_masks_of

D = 8


def _query(n, seed, constants=True, tree=False):
    """Random connected patterns (subject an existing variable, object new / existing / constant) and degrees."""
    rng = random.Random(seed)
    patterns, variables = [], ["?v0"]
    for i in range(n):
        s = rng.choice(variables)
        r = rng.random()
        o = f"?v{len(variables)}" if r < 0.6 or len(variables) < 2 or tree else (f"<c{i}>" if r < 0.8 and constants else rng.choice([v for v in variables if v != s]))
        if o.startswith("?") and o not in variables:
            variables.append(o)
        patterns.append(f"{s} <p{i % 3}> {o} .")
    degrees = {f"<p{j}>": {side: {"mean": rng.uniform(1, 3), "skew": rng.uniform(1, 4), "max": rng.uniform(5, 50)}
                           for side in ("s", "o")} for j in range(3)}
    return patterns, neighbour_masks_of(patterns), degrees


def _encoder(n, seed, multi=(0.0, 12.0), below_parts=False):
    g = torch.Generator().manual_seed(seed)
    table = {}

    def encode(masks):
        out_phi, out_l = [], []
        for m in masks:
            if m not in table:
                lo, hi = (3.0, 10.0) if not m & (m - 1) else multi
                value = torch.rand((), generator=g) * (hi - lo) + lo
                if below_parts and m & (m - 1):                              # l(U) < l(A) + l(B) for every split
                    value = bin(m).count("1") + 0.5 + 0.5 * torch.rand((), generator=g)
                table[m] = (torch.randn(D, generator=g), value)
            out_phi.append(table[m][0]), out_l.append(table[m][1])
        return torch.stack(out_phi), torch.stack(out_l)
    return encode, table


def _net(seed=0, trained=False):
    torch.manual_seed(seed)
    net = fv.ForestPlanner(phi_dim=D, d=16, heads=2, layers=2, bound_hidden=8)
    if trained:                                                      # a non-trivial value net and gate
        torch.nn.init.normal_(net.value.out[-1].weight, std=0.5)
        torch.nn.init.normal_(net.bound.gate[-1].weight, std=0.5)
        torch.nn.init.normal_(net.bound.mix[-1].weight, std=0.5)
    return net.eval()


def test_soft_bound_only_tightens_and_starts_at_the_hard_bound():
    torch.manual_seed(0)
    M = 500
    bound = fv.SoftBound(phi_dim=D, hidden=8)
    l, la, lb = torch.rand(M) * 20, torch.rand(M) * 10, torch.rand(M) * 10
    f = torch.sort(torch.rand(M, 3) * 5, dim=1).values                     # mean <= size-biased <= max
    fa = torch.where(torch.rand(M, 1) < 0.5, f, torch.full_like(f, fv.BIG))
    fb = torch.where(torch.rand(M, 1) < 0.5, f.flip(0), torch.full_like(f, fv.BIG))
    phi, size = torch.randn(M, D), torch.rand(M)
    l2, hard = bound(l, la, lb, fa, fb, phi, size)
    ref = torch.minimum(la + lb, torch.minimum(lb + fa[:, 2], la + fb[:, 2]))
    assert torch.allclose(hard, ref)
    assert bool((l2 <= hard + 1e-5).all()) and bool((l2 <= l + 1e-5).all())
    below = l < hard - 2                                                   # far below the bound: unchanged
    assert torch.allclose(l2[below], l[below], atol=0.02)
    torch.nn.init.normal_(bound.gate[-1].weight, std=1.0)                  # any gate / mix: never above the hard bound
    torch.nn.init.normal_(bound.mix[-1].weight, std=1.0)
    l2, _ = bound(l, la, lb, fa, fb, phi, size)
    assert bool((l2 <= hard + 1e-5).all())


def test_values_construction():
    net = _net(trained=True)
    B, K = 5, 4
    tokens, pad = torch.randn(B, K, D + 4), torch.zeros(B, K, dtype=torch.bool)
    adj, query = torch.zeros(B, K, K, dtype=torch.bool), torch.randn(B, D + 2)
    lq = torch.rand(B) * 10
    k = torch.tensor([1, 2, 3, 4, 4])
    V = net.values(tokens, pad, adj, query, lq, k)
    assert float(V[0]) == fv.NEG and float(V[1]) == float(lq[1])
    assert bool((V[2:] >= lq[2:]).all())                                   # the final result is always paid


@pytest.mark.parametrize("seed", range(4))
def test_untrained_planner_is_the_estimator_beam(seed):
    """W = 0 at init: V constant per query, so the beam ranks by the cost so far, as estimator_beam_plan (with
    estimates that no bound touches: joins below their parts, no degree statistics, gate off)."""
    n = 7
    patterns, nb, _ = _query(n, seed, constants=False, tree=True)  # a constant or a second shared end bounds the fan to 1
    encode, table = _encoder(n, seed, below_parts=True)
    ctx = fv.QueryContext(n, nb, patterns, {}, encode)
    net = _net(seed)
    torch.nn.init.constant_(net.bound.gate[-1].bias, -30.0)
    for w in (1, 4):
        tree, _ = fv.beam_plan(net, ctx, w)
        ref, _ = bushy.estimator_beam_plan(n, nb, lambda ms: [float(encode([m])[1][0]) for m in ms], w)
        assert bushy.canonical(tree) == bushy.canonical(ref)


@pytest.mark.parametrize("seed", range(3))
def test_training_forward_matches_the_planner(seed):
    """Along a greedy plan, forward_batch's l'' and V of every candidate's child forest equal the planner's."""
    n = 7
    patterns, nb, degrees = _query(n, seed)
    encode, table = _encoder(n, seed)
    ctx = fv.QueryContext(n, nb, patterns, degrees, encode)
    net = _net(seed, trained=True)
    trace = []
    tree, _ = fv.beam_plan(net, ctx, 1, trace=trace)
    counts = {m: int(math.exp(float(table[m][1]))) for m in table if m & (m - 1)}
    part = fv.build_query_data(n, nb, patterns, degrees, [tree], counts, 10 ** 9)
    phi, l = encode(part["masks"].tolist())
    with torch.no_grad():
        out = fv.forward_batch(net, [part], phi, l, "cpu", 0.3)
    masks = part["masks"]
    by_forest = {}
    for f in range(len(part["forest_nodes"])):
        ids = [i for i in part["forest_nodes"][f].tolist() if i >= 0]
        by_forest[tuple(sorted(int(masks[part["node_mask"][i]]) for i in ids))] = float(out["V"][f])
    assert len(trace) == len(part["cand"])
    for forest, l2, V in trace:
        assert math.isclose(by_forest[forest], V, rel_tol=1e-4, abs_tol=1e-4)
    cand_l2 = sorted(float(out["l2"][c]) for c in part["cand"][:, 1])
    assert all(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4) for a, b in zip(cand_l2, sorted(t[1] for t in trace)))


def test_stitched_g_is_the_exact_cost_to_go_with_all_subsets_counted():
    n = 6
    patterns, nb, degrees = _query(n, 3)
    encode, table = _encoder(n, 3)
    ctx = fv.QueryContext(n, nb, patterns, degrees, encode)
    tree, _ = fv.beam_plan(_net(), ctx, 2)
    labels = QueryLabels(query="q", family="x", n_tp=n, neighbour_masks=nb, logcard={})
    from src.supervised_value_estimation.amortized_dp.soft_dp import connected_masks
    conn = torch.nonzero(connected_masks(nb, "cpu")).squeeze(1).tolist()
    rng = random.Random(0)
    counts = {m: rng.randint(0, 10 ** 6) for m in conn if m & (m - 1)}
    labels.logcard = {m: math.log1p(counts.get(m, 0)) for m in conn}
    part = fv.build_query_data(n, nb, patterns, degrees, [tree], counts, 10 ** 9)
    masks = part["masks"]
    for f in part["state_forest"].tolist():
        ids = [i for i in part["forest_nodes"][f].tolist() if i >= 0]
        forest = [int(masks[part["node_mask"][i]]) for i in ids]
        exact = bushy.forest_cost_to_go(labels, forest)
        assert math.isclose(float(part["g_true"][f]), exact, rel_tol=1e-5)
    assert bool((~torch.isnan(part["q_true"])).all())                      # every candidate's Q is known


def test_plan_loss_by_hand():
    n = 5
    patterns, nb, degrees = _query(n, 1)
    encode, table = _encoder(n, 1)
    ctx = fv.QueryContext(n, nb, patterns, degrees, encode)
    net = _net(1, trained=True)
    tree, _ = fv.beam_plan(net, ctx, 1)
    from src.supervised_value_estimation.amortized_dp.soft_dp import connected_masks
    conn = torch.nonzero(connected_masks(nb, "cpu")).squeeze(1).tolist()
    counts = {m: random.Random(m).randint(0, 1000) for m in conn if m & (m - 1)}
    part = fv.build_query_data(n, nb, patterns, degrees, [tree], counts, 10 ** 9)
    phi, l = encode(part["masks"].tolist())
    tau = 0.5
    out = fv.forward_batch(net, [part], phi, l, "cpu", tau)
    res = fv.losses(out, tau)
    regrets = []
    for s in range(len(part["state_forest"])):
        rows = (part["cand"][:, 0] == s).nonzero().squeeze(1)
        qh, qt = out["q_hat"][rows], part["q_true"][rows]
        if len(rows) < 2:
            continue
        p = torch.softmax(-qh / tau, 0)
        regrets.append(torch.log((p * torch.exp(qt)).sum()) - qt.min())
    assert math.isclose(float(res["plan"]), float(torch.stack(regrets).mean()), rel_tol=1e-4)
    assert res["states_with_choice"] == len(regrets)
    res["plan"].backward()                                                 # reaches the bound and the value net
    assert net.value.out[-1].weight.grad is not None and float(net.value.out[-1].weight.grad.abs().sum()) > 0


@torch.no_grad()
def _reference_beam_plan(net, ctx, width, all_trees=False):
    """The per-candidate Python beam that beam_plans replaced (d29b0cb), kept as the reference."""
    n, nb = ctx.n, ctx.neighbour_masks
    full = (1 << n) - 1
    phi_q, lq = ctx.encoded([full])
    singles = [1 << i for i in range(n)]
    _, ls = ctx.encoded(singles)
    q_std = net.standardise(phi_q)
    t = lambda x, **kw: torch.tensor(x, device=lq.device, **kw)
    beams = [(bushy.initial_forest(n), fv.NEG, {m: (float(ls[i]), fv.NEG, i) for i, m in enumerate(singles)})]
    while len(beams[0][0]) > 1:
        cands = [(bi, a, b) for bi, (forest, _, _) in enumerate(beams) for a, b in bushy.candidate_merges(forest, nb)]
        U = [a | b for _, a, b in cands]
        phi_u, l_u = ctx.encoded(U)
        comps = [beams[bi][2] for bi, _, _ in cands]
        la = t([c[a][0] for c, (_, a, _) in zip(comps, cands)])
        lb = t([c[b][0] for c, (_, _, b) in zip(comps, cands)])
        fa = torch.stack([fv.merge_fan(a, b, ctx.touch_s, ctx.touch_o, ctx.fan) for _, a, b in cands]).to(lq.device)
        fb = torch.stack([fv.merge_fan(b, a, ctx.touch_s, ctx.touch_o, ctx.fan) for _, a, b in cands]).to(lq.device)
        size = t([bin(u).count("1") / n for u in U])
        l2, _ = net.bound(l_u, la, lb, fa, fb, net.standardise(phi_u), size)
        ca = t([c[a][1] for c, (_, a, _) in zip(comps, cands)])
        cb = t([c[b][1] for c, (_, _, b) in zip(comps, cands)])
        cost_u = torch.logaddexp(l2, torch.logaddexp(ca, cb))
        so_far = torch.logaddexp(t([beams[bi][1] for bi, _, _ in cands]), l2)
        children = [bushy.merge(beams[bi][0], a, b) for bi, a, b in cands]
        k = len(children[0])
        if k == 1:
            total = so_far.double()
        else:
            masks = [[c for c in child if c != u] + [u] for child, u in zip(children, U)]
            old = [m for ms in masks for m in ms[:-1]]
            phi_old, _ = ctx.encoded(old)
            l_old = t([comps[j][m][0] for j, ms in enumerate(masks) for m in ms[:-1]])
            c_old = t([comps[j][m][1] for j, ms in enumerate(masks) for m in ms[:-1]])
            sz_old = t([bin(m).count("1") / n for m in old])
            tok_old = fv.token_features(net.standardise(phi_old), l_old, c_old, sz_old * n < 1.5, sz_old).view(len(cands), k - 1, -1)
            tok_new = fv.token_features(net.standardise(phi_u), l2, cost_u, torch.zeros_like(l2, dtype=torch.bool), size)
            tokens = torch.cat([tok_old, tok_new[:, None]], 1)
            adj = torch.stack([fv.forest_adjacency(ms, nb, k, ctx.reach) for ms in masks]).to(lq.device)
            query = torch.cat([q_std.expand(len(cands), -1), (lq / 10).expand(len(cands))[:, None],
                               torch.full((len(cands), 1), k / n, device=lq.device)], 1)
            V = net.values(tokens, torch.zeros(len(cands), k, dtype=torch.bool, device=lq.device), adj, query,
                           lq.expand(len(cands)), torch.full((len(cands),), k, device=lq.device))
            total = torch.logaddexp(so_far.double(), V.double())
        best: dict = {}
        for j, child in enumerate(children):
            key = (j,) if all_trees and k == 1 else child
            score = (float(total[j]), float(so_far[j]))
            if key not in best or score < best[key][0]:
                best[key] = (score, j)
        next_beams = []
        for _, (_, j) in sorted(best.items(), key=lambda item: item[1][0])[:width]:
            bi, a, b = cands[j]
            info = dict(beams[bi][2])
            ta, tb = info.pop(a)[2], info.pop(b)[2]
            info[a | b] = (float(l2[j]), float(cost_u[j]), (ta, tb))
            next_beams.append((children[j], float(so_far[j]), info))
        beams = next_beams
    trees = [next(iter(beam[2].values()))[2] for beam in beams]
    return trees if all_trees else trees[0]


@pytest.mark.parametrize("seed", range(6))
def test_batched_beam_matches_the_reference(seed):
    """beam_plans equals the per-candidate Python beam it replaced (trained value net and gate, constants, cycles)."""
    n = 6 + seed % 4
    patterns, nb, degrees = _query(n, seed)
    encode, _ = _encoder(n, seed)
    net = _net(seed, trained=True)
    for w in (1, 3, 8):
        for all_trees in (False, True):
            ref = _reference_beam_plan(net, fv.QueryContext(n, nb, patterns, degrees, encode), w, all_trees)
            got, _ = fv.beam_plan(net, fv.QueryContext(n, nb, patterns, degrees, encode), w, all_trees)
            assert got == ref


@pytest.mark.parametrize("seed", range(3))
def test_groups_are_independent_beams(seed):
    """Several groups in one batch give each group's own beam; noisy groups give valid trees."""
    n = 8
    patterns, nb, degrees = _query(n, seed)
    encode, _ = _encoder(n, seed)
    net = _net(seed, trained=True)
    widths, alls = [1, 4, 8, 8], [False, True, True, False]
    groups, _ = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), widths, all_trees=alls)
    for w, a, trees in zip(widths, alls, groups):
        alone, _ = fv.beam_plan(net, fv.QueryContext(n, nb, patterns, degrees, encode), w, a)
        assert trees == (alone if a else [alone])
    g = torch.Generator().manual_seed(seed)
    groups, _ = fv.beam_plans(net, fv.QueryContext(n, nb, patterns, degrees, encode), [8, 8, 8], [0.0, 1.0, 1.0],
                              [True, False, False], g)
    assert groups[0] == fv.beam_plan(net, fv.QueryContext(n, nb, patterns, degrees, encode), 8, True)[0]
    assert all(bushy.is_valid_tree(t, n, nb) for ts in groups for t in ts)


def test_stop_gradient_leaves_only_the_raw_l1_on_the_encoder():
    n = 6
    patterns, nb, degrees = _query(n, 2)
    encode, table = _encoder(n, 2)
    net = _net(2, trained=True)
    tree, _ = fv.beam_plan(net, fv.QueryContext(n, nb, patterns, degrees, encode), 2)
    from src.supervised_value_estimation.amortized_dp.soft_dp import connected_masks
    conn = torch.nonzero(connected_masks(nb, "cpu")).squeeze(1).tolist()
    counts = {m: random.Random(m).randint(0, 1000) for m in conn if m & (m - 1)}
    part = fv.build_query_data(n, nb, patterns, degrees, [tree], counts, 10 ** 9)
    for stop in (False, True):
        phi, l = encode(part["masks"].tolist())
        phi, l = phi.clone().requires_grad_(), l.clone().requires_grad_()
        res = fv.losses(fv.forward_batch(net, [part], phi, l, "cpu", 0.5, stop_gradient=stop), 0.5)
        (res["plan"] + res["value"] + res["consistency"] + res["l1_bound"]).backward(retain_graph=True)
        planner_reaches_encoder = phi.grad is not None and float(phi.grad.abs().sum()) > 0
        assert planner_reaches_encoder != stop
        phi.grad = l.grad = None
        res["l1_raw"].backward()
        assert float(l.grad.abs().sum()) > 0


def _counted_part(n, seed):
    patterns, nb, degrees = _query(n, seed)
    encode, _ = _encoder(n, seed)
    tree, _ = fv.beam_plan(_net(seed), fv.QueryContext(n, nb, patterns, degrees, encode), 2)
    from src.supervised_value_estimation.amortized_dp.soft_dp import connected_masks
    conn = torch.nonzero(connected_masks(nb, "cpu")).squeeze(1).tolist()
    counts = {m: random.Random(m).randint(0, 1000) for m in conn if m & (m - 1)}
    part = fv.build_query_data(n, nb, patterns, degrees, [tree], counts, 10 ** 9)
    return part, nb, encode


def test_chunked_value_pass_is_the_same():
    part, nb, encode = _counted_part(7, 4)
    phi, l = encode(part["masks"].tolist())
    torch.manual_seed(0)
    net = fv.ForestPlanner(phi_dim=D, d=16, heads=2, layers=2, bound_hidden=8)
    torch.nn.init.normal_(net.value.out[-1].weight, std=0.5)
    grads = []
    for chunk in (None, 5):
        net.zero_grad()
        res = fv.losses(fv.forward_batch(net, [part], phi, l, "cpu", 0.5, value_chunk=chunk), 0.5)
        (res["value"] + res["plan"]).backward()
        grads.append(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]))
        grads.append(res["value"].detach())
    assert torch.allclose(grads[0], grads[2], atol=1e-6) and torch.allclose(grads[1], grads[3])
