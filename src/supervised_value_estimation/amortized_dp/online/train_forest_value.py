"""Online training of the forest value planner (forest_value.py) and the cardinality model together, with true counts.

    python -m src.supervised_value_estimation.amortized_dp.online.train_forest_value --config-name forest_value_mixed_yago_v2.yaml

The planner is a beam over forests ranked by log(cost so far + V(F)), with the cardinality model's per-subset
estimates bounded by the components already built (forest_value.py's module text). It starts from the pretrained
cardinality model and an untrained planner (W = 0, gate ~ off): round 0 is the estimator beam with the hard degree
bound. Each ROUND, as train_plan_loss:
  1. take `queries_per_round` training queries (a share `revisit_fraction` already seen, re-planned);
  2. plan each with the current model: the beam of width `deployed_width` (all its final trees with
     `count_beam_alternatives`) and `explore_plans` beams with N(0, explore_sigma) noise on the scores; with
     forest.count_candidates also the result of every candidate merge along the deployed plan (all beams of a
     query run as one batch, fv.beam_plans);
  3. COUNT the new subsets on QLever (a failed count costs the query's failed-count bound, 2 x its largest count),
     streamed: each query's subsets are counted while the next queries are planned (CountStream);
  4. train on the most recent `replay_queries` counted queries (up to forest.max_plans_per_query counted plans each):
         l1_weight * L1(l) + forest.bound_l1_weight * L1(l'') + forest.value_weight * |V - log G|
         + plan_weight * per-step plan regret + forest.consistency_weight * |V - min Q^|   (forest_value.losses)
     with gradients into the backbone (online.lr) and the planner (forest.lr). The plan regret is off for the first
     forest.plan_warmup_rounds rounds: the untrained V would otherwise push the estimates to compensate for it.
     With forest.stop_gradient the planner reads the backbone's phi and l detached: only the L1 on l trains the
     backbone (the plan loss pulled the estimates away from the counts: the references drifted in the run of 17-10).
With epinet.enabled (forest_epinet.py: one epistemic index z drives an epinet on the subset estimates and one on
the value; Osband et al. 2023, Wen et al. 2023):
  - Thompson sampling (epinet.thompson): each exploration beam draws one z and plans with l_z, V_z, instead of
    N(0, explore_sigma) noise on its scores; the deployed beam uses z = 0 (the plain planner);
  - which queries to count (epinet.select_pool_factor f > 1): the new queries of a round are the top 1/f of f times
    as many candidates by priority Var_z V_z(F0), the epistemic variance of the predicted cost of the whole query
    ("Fine-Tuning Language Models via Epistemic Neural Networks", the "variance" priority);
  - which queries to train on (epinet.train_select_factor f > 1): each batch is the top 1/f of f times as many
    window queries by the same priority (computed once per round);
  - training: the epinet losses (forest_epinet.losses, Gaussian bootstrap) next to the planner's, epinet only;
  - calibration: each round on its fresh counts (not yet trained on): coverage of the 90% interval and Spearman of
    std vs |error|, per head; with epinet.sample_scale auto the sample scale is refitted so 90% intervals cover 90%.
With online.replay.prioritized (regret-based prioritized experience replay, Schaul et al. 2016): after each round's
counts, every executed query's realized regret = log C_out(its deployed plan) - log min C_out(its counted plans), true
costs from the counts (the alternatives: beam alternatives, exploration / Thompson plans, earlier rounds' plans).
Training batches and revisits draw queries with p ~ (min(regret, cap) + eps)^alpha, mixed with a uniform floor;
the training losses weigh each query by its importance weight (N p)^-beta / max (beta annealed to beta_end). A
query's regret is updated whenever it is executed again. Common offsets of a query (its final result) cancel.
Validation (round 0, then every validation.every_rounds) on the same 2,000 queries as train_plan_loss: "value GOO /
beam-4 / beam-8" (this planner) next to GNCE GOO / beam-4 / beam-8, and with forest.reference_planners the current
cardinality model's plain estimator beam-8 and its lattice DP with the hard degree bound. True C_out, geomean vs
GNCE GOO and vs round 0, plus the queries more than 10x better / worse than round 0 (the tail decides the geomean),
and the tail: percentiles (PERCENTILES) and shares > 2x / > 10x of the per-query C_out over a FIXED reference, the
cheapest plan any planner (ours or GNCE) found for the query at round 0.

Outputs in online.output_directory/forest-value-<time>/: config.yaml, rounds.jsonl, validation.jsonl,
validation-<round>.json, model-<round>.pt (cardinality model), planner-<round>.pt, epinet-<round>.pt, counts.pkl.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.supervised_value_estimation.amortized_dp import bushy, forest_epinet as fe, forest_value as fv, history_dp, soft_dp
from src.supervised_value_estimation.amortized_dp.bushy_train import count_connected_subsets, load_backbone, load_split
from src.supervised_value_estimation.amortized_dp.labels import model_log_cardinalities
from src.supervised_value_estimation.amortized_dp.online.train_online import _endpoints
from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import connected, neighbour_masks_of
from src.supervised_value_estimation.amortized_dp.online.train_plan_loss import (BUCKETS, Query, failed_bound,
                                                                                 multi_query_batch, phi_and_direct,
                                                                                 plan_cost, subset_phi)
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import CountStream, count_query, count_subsets

PERCENTILES = (50, 90, 95, 99)


def encoder(model, q, device):
    """encode(masks) -> (phi, l) for forest_value.QueryContext: the cardinality model's embedding and estimate,
    exact counts for single patterns."""
    def encode(masks):
        m = torch.tensor(masks, dtype=torch.long, device=device)
        phi, l = subset_phi(model, q.data, m, device)
        single = (m & (m - 1)) == 0
        return phi, torch.where(single, history_dp.single_pattern_logcounts(q.data, m), l)
    return encode


VALUE_CHUNK = 32768        # forests per value-net pass: the fused attention kernel fails above 65,535 in one batch


def realized_regret(deployed, plans, counts, bound):
    """log C_out(deployed) - log min C_out(plans) (true costs from counts; plans include the deployed one), or None."""
    costs = [plan_cost(t, counts, bound)[0] for t in plans]
    costs = [c for c in costs if c is not None]
    mine = plan_cost(deployed, counts, bound)[0]
    if mine is None or not costs:
        return None
    return max(0.0, math.log(max(mine, 1)) - math.log(max(min(costs), 1)))


def replay_probabilities(regrets, alpha, floor, eps, cap):
    """p_i ~ (min(regret_i, cap) + eps)^alpha, mixed with a uniform share `floor`."""
    pr = (np.clip(np.asarray(regrets, dtype=np.float64), 0.0, cap) + eps) ** alpha
    return (1 - floor) * pr / pr.sum() + floor / len(pr)


def importance_weights(p, beta):
    """(N p_i)^-beta, normalised by its max (Schaul et al.): corrects non-uniform sampling, never scales up."""
    w = (len(p) * np.asarray(p)) ** -beta
    return w / w.max()


@torch.no_grad()
def encode_batch(model, queries, parts, device):
    """The encoder's (phi, l) for the concatenated masks of the parts (single patterns: exact counts)."""
    x = multi_query_batch([(q, p["masks"].to(device)) for q, p in zip(queries, parts)], device)
    phi, l = phi_and_direct(model, *x)
    m = torch.cat([p["masks"] for p in parts]).to(device)
    exact = torch.cat([history_dp.single_pattern_logcounts(q.data, p["masks"].to(device)) for q, p in zip(queries, parts)])
    return phi, torch.where((m & (m - 1)) == 0, exact, l)


@torch.no_grad()
def start_items(model, queries, device, chunk=256):
    """forest_epinet.start_forests items: per query (phi, l) of its single patterns and of the whole query."""
    items = []
    for s in range(0, len(queries), chunk):
        qs = queries[s:s + chunk]
        masks = [torch.tensor([1 << i for i in range(q.n)] + [(1 << q.n) - 1], device=device) for q in qs]
        phi, l = phi_and_direct(model, *multi_query_batch(list(zip(qs, masks)), device))
        off = 0
        for q, m in zip(qs, masks):
            exact = history_dp.single_pattern_logcounts(q.data, m)
            items.append((phi[off:off + q.n + 1], torch.cat([exact[:q.n], l[off + q.n:off + q.n + 1]]), q.neighbour_masks))
            off += q.n + 1
    return items


@torch.no_grad()
def card_calibration(model, net, epi, pairs, device, n_indexes=64, generator=None):
    """forest_epinet.calibration of the subset head on [(query, {mask: count})] (multi-pattern, counted)."""
    pairs = [(q, table) for q, table in pairs if table]
    if not pairs:
        return None
    queries = [q for q, _ in pairs]
    phi, l = encode_batch(model, queries, [{"masks": torch.tensor(list(t), dtype=torch.long)} for _, t in pairs], device)
    true = torch.tensor([math.log1p(c) for _, t in pairs for c in t.values()], device=device)
    Z = torch.randn(n_indexes, epi.index_dim, device=device, generator=generator)
    return fe.calibration(fe.scaled_card_samples(epi, net, phi, l, Z), l, true)


@torch.no_grad()
def value_calibration(model, net, epi, queries, parts, device, n_indexes=64, generator=None, chunk=32, target="v"):
    """forest_epinet.calibration of the value head on these parts' forests: target "v" V_z against log G (forests
    with a stitched G), "w" W + s e_value against W* (forests where W* is defined)."""
    samples, base, true = [], [], []
    Z = torch.randn(n_indexes, epi.index_dim, device=device, generator=generator)
    for s in range(0, len(parts), chunk):
        qs, ps = queries[s:s + chunk], parts[s:s + chunk]
        phi, l = encode_batch(model, qs, ps, device)
        out = fv.forward_batch(net, ps, phi, l, device, 1.0, True, value_chunk=VALUE_CHUNK, hidden=True)
        if target == "w":
            f = (~torch.isnan(out["w_true"])).nonzero().squeeze(1)
            samples.append(out["W"][f] + epi.scale_value * epi.value_samples(out["H"][f], Z))
            base.append(out["W"][f]), true.append(out["w_true"][f])
        else:
            f = (~torch.isnan(out["g_true"]) & (out["forest_k"] >= 3)).nonzero().squeeze(1)
            samples.append(fe.forest_value_samples(epi, net, out, phi, f, Z, float(epi.scale_card), float(epi.scale_value)))
            base.append(out["V"][f]), true.append(out["g_true"][f])
    if not samples:
        return None
    return fe.calibration(torch.cat(samples, 1), torch.cat(base), torch.cat(true))


@torch.no_grad()
def plan_query(model, net, q, degrees, device, widths, explore=0, sigma=0.0, generator=None, alternatives=False,
               reference=False, epinet=None, thompson=False):
    """{name: tree} for "beam-w" (with alternatives also "beam-w #i"), "explore-i" and, with reference, the plain
    estimator "est beam-8" and "est DP + bound"; plus the deployed beam's states' candidate merges (masks). With an
    epinet and thompson, each exploration beam plans with its own epistemic index instead of score noise."""
    ctx = fv.QueryContext(q.n, q.neighbour_masks, q.patterns, degrees, encoder(model, q, device))
    groups = [("GOO" if w == 1 else f"beam-{w}", w, 0.0, alternatives) for w in widths]
    groups += [(f"explore-{i}", widths[-1], 0.0 if thompson else sigma, False) for i in range(explore)]
    indexes = None
    if epinet is not None:
        indexes = torch.zeros(len(groups), epinet.index_dim, device=device)
        if thompson and explore:
            indexes[len(widths):] = torch.randn(explore, epinet.index_dim, device=device, generator=generator)
    out, _ = fv.beam_plans(net, ctx, [g[1] for g in groups], [g[2] for g in groups], [g[3] for g in groups], generator,
                           epinet=epinet, indexes=indexes)
    plans = {}
    for (name, *_), trees in zip(groups, out):                  # all beams of the query in one batch
        plans[name] = trees[0]
        plans.update({f"{name} #{i}": t for i, t in enumerate(trees[1:], 1)})
    if reference:
        s = soft_dp.SubsetStructure(q.neighbour_masks, device)
        est = soft_dp.model_subset_logcards(model, q.data, s.masks, device)
        table = dict(zip(s.masks.tolist(), est.tolist()))
        plans["est beam-8"], _ = bushy.estimator_beam_plan(q.n, q.neighbour_masks, lambda ms: [table[m] for m in ms], 8)
        exact = history_dp.single_pattern_logcounts(q.data, s.masks)
        bounded = history_dp.degree_bound(q.patterns, s, torch.where(s.singleton, exact, est).clamp(min=0), degrees)
        plans["est DP + bound"] = soft_dp.hard_plan(s, soft_dp.soft_dp(s, bounded, 0.0)[1])
    return plans


def candidate_masks(tree, q):
    """The result of every candidate merge at the states along `tree`."""
    out, forest = [], bushy.initial_forest(q.n)
    for a, b in bushy.tree_to_merges(tree):
        out += [x | y for x, y in bushy.candidate_merges(forest, q.neighbour_masks)]
        forest = bushy.merge(forest, a, b)
    return out


@hydra.main(version_base=None, config_path="../../../../experiments/experiment_configs/amortized_dp",
            config_name="forest_value_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    online, fcfg = cfg.online, cfg.forest
    rng = np.random.default_rng(int(online.seed))
    torch.manual_seed(int(online.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device.type).manual_seed(int(online.seed))
    run_directory = Path(online.output_directory) / f"forest-value-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    run_directory.mkdir(parents=True, exist_ok=False)
    with open(run_directory / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {run_directory} | device {device}", flush=True)

    model, epoch = load_backbone(cfg.estimator.config, cfg.estimator.run, cfg.estimator.select, device)
    with open(fcfg.degrees) as f:
        degrees = json.load(f)

    # --- queries: the same seeded selection as train_plan_loss ----------------------------------------
    train_dataset = load_split(cfg.data.train_dir)
    train_pool = [i for i in range(len(train_dataset)) if connected(neighbour_masks_of(train_dataset[i].triple_patterns))]
    rng.shuffle(train_pool)
    val_dataset = load_split(cfg.data.val_dir)
    gnce_dataset = load_split(cfg.gnce.dataset)
    gnce_index = {" ".join(gnce_dataset[i].query.split()): i for i in range(len(gnce_dataset))}
    candidates = [i for i in range(len(val_dataset))
                  if " ".join(val_dataset[i].query.split()) in gnce_index
                  and list(gnce_dataset[gnce_index[" ".join(val_dataset[i].query.split())]].triple_patterns)
                  == list(val_dataset[i].triple_patterns)
                  and connected(neighbour_masks_of(val_dataset[i].triple_patterns))]
    picked = sorted(rng.choice(len(candidates), min(int(online.validation.n_queries), len(candidates)), replace=False))
    validation = [Query(val_dataset[candidates[k]]) for k in picked]

    with torch.no_grad():                                # phi standardisation from a sample of training subsets
        phis = []
        for i in train_pool[:64]:
            q = Query(train_dataset[i])
            if count_connected_subsets(q.neighbour_masks, 1024) <= 1024:
                phis.append(subset_phi(model, q.data, soft_dp.SubsetStructure(q.neighbour_masks, device).masks, device)[0])
        phi = torch.cat(phis)
    net = fv.ForestPlanner(phi.shape[1], int(fcfg.d), int(fcfg.heads), int(fcfg.layers), int(fcfg.bound_hidden)).to(device)
    net.mean.copy_(phi.mean(0)), net.std.copy_(phi.std(0).clamp(min=1e-3))
    print(f"Estimator: {cfg.estimator.run} (epoch {epoch}), {sum(p.numel() for p in model.parameters()):,} parameters | "
          f"forest planner {sum(p.numel() for p in net.parameters()):,} parameters (untrained) | degrees of {len(degrees)} predicates",
          flush=True)
    ecfg = cfg.get("epinet") or {"enabled": False}
    epi = None
    if bool(ecfg["enabled"]):
        epi = fe.ForestEpinet(phi.shape[1], int(fcfg.d), int(ecfg.index_dim), int(ecfg.hidden_dim),
                                          int(ecfg.prior_hidden_dim)).to(device)
        sample = start_items(model, [Query(train_dataset[i]) for i in train_pool[:256]], device)
        epi.set_prior_scale(net.standardise(phi), fe.start_forests(net, sample, device)[4], float(ecfg.prior_scale_target))
        if str(ecfg.sample_scale) != "auto":
            epi.scale_card.fill_(float(ecfg.sample_scale)), epi.scale_value.fill_(float(ecfg.sample_scale))
        n_train = sum(p.numel() for p in epi.trainable())
        print(f"Epinet: {n_train:,} trainable parameters (+{sum(p.numel() for p in epi.parameters()) - n_train:,} frozen "
              f"prior) | index dim {ecfg.index_dim} | Thompson exploration {bool(ecfg.thompson)} | select queries "
              f"top 1/{ecfg.select_pool_factor} to count, top 1/{ecfg.train_select_factor} to train | sample scale "
              f"{ecfg.sample_scale}", flush=True)
    print(f"Train pool: {len(train_pool):,} connected queries | validation: {len(validation)} "
          f"(sizes {np.bincount([v.n for v in validation]).tolist()})", flush=True)

    widths = [int(w) for w in online.validation.widths]
    gnce, _ = load_backbone(cfg.gnce.config, cfg.gnce.run, cfg.gnce.select, device)
    gnce_plans = {}
    for v in validation:
        data = gnce_dataset[gnce_index[v.key]]
        est = lambda masks: model_log_cardinalities(gnce, data, masks, device)
        gnce_plans[v.key] = {f"GNCE {'GOO' if w == 1 else f'beam-{w}'}": bushy.estimator_beam_plan(v.n, v.neighbour_masks, est, w)[0]
                             for w in widths}
    del gnce

    endpoints = _endpoints(online.execution)
    print(f"{len(endpoints)} QLever endpoints: {endpoints[0]} ... {endpoints[-1]}", flush=True)
    counts: dict[str, dict[int, int | None]] = {}
    largest = [0]
    seed_cache = online.validation.get("true_counts_cache")
    if seed_cache and Path(seed_cache).exists():
        with open(seed_cache, "rb") as f:
            for key, table in pickle.load(f).items():
                counts[key] = {m: c for m, c in table.items() if c is not None}
                largest[0] = max([largest[0], *counts[key].values()])
        print(f"Seeded counts from {seed_cache}: {sum(len(t) for t in counts.values()):,}", flush=True)

    def count_missing(pairs):
        tasks = []
        for q, masks in pairs:
            known = counts.setdefault(q.key, {})
            tasks += [((q.key, m), count_query(q.patterns, m)) for m in dict.fromkeys(masks) if m not in known]
        tasks = list(dict(tasks).items())
        start = time.perf_counter()
        results = count_subsets(tasks, endpoints, float(online.count_timeout_s)) if tasks else {}
        return store_counts(results), sum(v is None for v in results.values()), time.perf_counter() - start

    def store_counts(results):
        for (key, m), value in results.items():
            counts[key][m] = value
            if value is not None:
                largest[0] = max(largest[0], value)
        return len(results)

    # --- validation --------------------------------------------------------------------------------
    reference = {}
    best0 = {}                                           # query -> cheapest plan of any planner at round 0 (fixed)

    def validate(round_index):
        model.eval(), net.eval()
        start = time.perf_counter()
        plans = {}
        for v in tqdm(validation, desc="Planning (validation)", file=sys.stdout, mininterval=30):
            ours = plan_query(model, net, v, degrees, device, widths, reference=bool(fcfg.reference_planners))
            plans[v.key] = {(k if k.startswith("est") else f"value {k}"): t for k, t in ours.items()} | gnce_plans[v.key]
        planned = time.perf_counter() - start
        new, failed, seconds = count_missing([(v, [m for t in plans[v.key].values() for m in bushy.internal_masks(t)])
                                              for v in validation])
        names = list(plans[validation[0].key])
        costed = {name: {v.key: plan_cost(plans[v.key][name], counts[v.key], failed_bound(counts[v.key], largest[0]))
                         for v in validation} for name in names}
        cost = {name: {k: c for k, (c, _) in table.items()} for name, table in costed.items()}
        imputed = {name: sum(i > 0 for _, i in table.values()) for name, table in costed.items()}
        if round_index == 0:
            reference.update({name: dict(c) for name, c in cost.items()})
            best0.update({v.key: max(min(c[v.key] for c in cost.values() if c[v.key] is not None), 1) for v in validation})
        summary = {"round": round_index, "counted": new, "failed": failed, "count_s": round(seconds, 1),
                   "plan_s": round(planned, 1), "planners": {}}

        def ratios(name, base, keep=lambda n: True):
            return np.array([math.log(max(cost[name][v.key], 1) / max(base[v.key], 1)) for v in validation if keep(v.n)
                             and cost[name][v.key] is not None and base.get(v.key) is not None])

        def geo(name, base):
            out = {}
            for b, keep in BUCKETS.items():
                r = ratios(name, base, keep)
                out[b] = round(float(np.exp(np.mean(r))), 4) if len(r) else None
            return out
        for name in names:
            ours = not name.startswith("GNCE")
            r0 = ratios(name, reference[name]) if ours else np.array([])
            summary["planners"][name] = {"vs_GNCE_GOO": geo(name, cost["GNCE GOO"]), "plans_with_failed_count": imputed[name],
                                         "vs_round_0": geo(name, reference[name]) if ours else None,
                                         "vs_round_0_10x": [int((r0 < -math.log(10)).sum()), int((r0 > math.log(10)).sum())] if ours else None}
            rb = np.array([max(cost[name][v.key], 1) / best0[v.key] for v in validation if cost[name][v.key] is not None])
            summary["planners"][name]["vs_best_round_0"] = {**{f"p{q}": round(float(np.percentile(rb, q)), 3) for q in PERCENTILES},
                                                            ">2x": round(float(np.mean(rb > 2)), 4), ">10x": round(float(np.mean(rb > 10)), 4)}
        with open(run_directory / "validation.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        with open(run_directory / f"validation-{round_index}.json", "w") as f:
            json.dump({v.key: {"n": v.n, "family": v.family, **{name: {"plan": bushy.to_json(plans[v.key][name]),
                       "true_cout": cost[name][v.key], "failed_counts": costed[name][v.key][1]} for name in names}}
                       for v in validation}, f)
        print(f"\n[validation round {round_index}] true C_out, geomean vs GNCE GOO ({new:,} new counts, "
              f"{failed} failed, {seconds:.0f}s; planning {planned:.0f}s):", flush=True)
        print(f"  {'planner':18s}" + "".join(f"{b:>8s}" for b in BUCKETS) + "  failed*   vs round 0 (all)   >10x better/worse",
              flush=True)
        for name in names:
            p = summary["planners"][name]
            line = f"  {name:18s}" + "".join(f"{x:8.3f}" if x is not None else f"{'-':>8s}" for x in p["vs_GNCE_GOO"].values())
            line += f"{imputed[name]:9d}"
            if p["vs_round_0"]:
                line += f"   {p['vs_round_0']['all']:.3f}            {p['vs_round_0_10x'][0]:4d} / {p['vs_round_0_10x'][1]:<4d}"
            print(line, flush=True)
        print("  * plans with a failed count, costed at the query's failed-count bound (2 x its largest count)", flush=True)
        print(f"  per-query C_out / the cheapest plan of any planner at round 0 (fixed reference; the tail):", flush=True)
        print(f"  {'planner':18s}" + "".join(f"{'p' + str(q):>8s}" for q in PERCENTILES) + f"{'> 2x':>8s}{'> 10x':>8s}", flush=True)
        for name in names:
            p = summary["planners"][name]["vs_best_round_0"]
            print(f"  {name:18s}" + "".join(f"{p['p' + str(q)]:8.2f}" for q in PERCENTILES)
                  + f"{p['>2x']:8.1%}{p['>10x']:8.1%}", flush=True)
        if epi is not None:
            pick = np.random.default_rng(round_index)
            pairs = []
            for v in validation:
                multi = [m for m, c in counts[v.key].items() if c is not None and m & (m - 1)]
                chosen = pick.choice(len(multi), min(8, len(multi)), replace=False) if multi else []
                pairs.append((v, {multi[i]: counts[v.key][multi[i]] for i in chosen}))
            cal = card_calibration(model, net, epi, pairs, device, generator=generator)
            with open(run_directory / "validation.jsonl", "a") as f:
                f.write(json.dumps({"round": round_index, "epinet_card_calibration": cal}) + "\n")
            print(f"  epinet, subset estimates of the validation plans (never trained on): {cal}", flush=True)
            epi.save(run_directory / f"epinet-{round_index}.pt")
        torch.save(model.state_dict(), run_directory / f"model-{round_index}.pt")
        net.save(run_directory / f"planner-{round_index}.pt")
        with open(run_directory / "counts.pkl", "wb") as f:
            pickle.dump(counts, f)

    # --- training ------------------------------------------------------------------------------------
    groups = [{"params": list(model.parameters()), "lr": float(online.lr)}, {"params": list(net.parameters()), "lr": float(fcfg.lr)}]
    if epi is not None:
        groups.append({"params": epi.trainable(), "lr": float(ecfg.lr)})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(online.weight_decay))
    trainable = list(model.parameters()) + list(net.parameters())   # the epinet is clipped on its own
    seen: dict[str, Query] = {}
    plans_of: dict[str, list] = {}                       # query -> counted plans (canonical, most recent last)
    parts: dict[str, tuple] = {}                         # query -> (build_query_data part, counts seen, plans seen)
    weights = {"l1_raw": float(online.l1_weight), "l1_bound": float(fcfg.bound_l1_weight), "value": float(fcfg.value_weight),
               "w": float(fcfg.w_loss_weight), "plan": float(online.plan_weight), "consistency": float(fcfg.consistency_weight)}
    value_target = "v"
    if epi is not None:
        value_target = ("w" if float(fcfg.w_loss_weight) > 0 else "v") if str(ecfg.value_target) == "auto" else str(ecfg.value_target)
        print(f"Epinet value head target: {'W against W*' if value_target == 'w' else 'V against log G'}", flush=True)
    rcfg = online.replay
    prioritized = bool(rcfg.prioritized)
    if prioritized and epi is not None and int(ecfg.train_select_factor) > 1:
        raise ValueError("online.replay.prioritized and epinet.train_select_factor > 1 both choose the training batches")
    regret_of: dict[str, float] = {}                    # query -> realized regret of its last execution
    deployed_of: dict[str, tuple] = {}                  # query -> its deployed plan in its last execution

    def probabilities(keys):
        top = max(regret_of.values(), default=0.0)       # queries without a regret yet: the largest (seen once)
        return replay_probabilities([regret_of.get(k, top) for k in keys], float(rcfg.alpha), float(rcfg.floor),
                                    float(rcfg.eps), float(rcfg.cap))
    tau = float(online.plan_tau)
    next_index = 0
    if online.validation.at_start:
        validate(0)
    for round_index in range(1, int(online.rounds) + 1):
        t0 = time.perf_counter()
        n_revisit = min(int(round(float(online.revisit_fraction) * int(online.queries_per_round))), len(seen))
        keys = list(seen)
        p_revisit = probabilities(keys) if prioritized and bool(rcfg.revisits) and keys else None
        batch = [seen[k] for k in rng.choice(keys, n_revisit, replace=False, p=p_revisit)] if n_revisit else []
        model.eval(), net.eval()
        n_new = int(online.queries_per_round) - len(batch)
        factor = int(ecfg.select_pool_factor) if epi is not None else 1
        candidates = []
        while len(candidates) < n_new * factor:
            candidates.append(Query(train_dataset[train_pool[next_index % len(train_pool)]]))
            next_index += 1
        selection = None
        if factor > 1 and n_new:
            priority = fe.query_priorities(net, epi, start_items(model, candidates, device), device,
                                           int(ecfg.n_indexes_priority), generator)
            top = sorted(torch.topk(priority, n_new).indices.tolist())
            selection = {"pool": len(candidates), "pool_mean_n": round(float(np.mean([q.n for q in candidates])), 2),
                         "selected_mean_n": round(float(np.mean([candidates[i].n for i in top])), 2),
                         "pool_mean_priority": round(float(priority.mean()), 4),
                         "selected_mean_priority": round(float(priority[top].mean()), 4)}
            candidates = [candidates[i] for i in top]
        new_keys = {q.key for q in candidates}
        batch += candidates
        deployed = f"beam-{int(online.deployed_width)}" if int(online.deployed_width) > 1 else "GOO"
        stream = CountStream(endpoints, float(online.count_timeout_s))
        queued = set()
        for q in tqdm(batch, desc="Planning", file=sys.stdout, mininterval=30):
            plans = plan_query(model, net, q, degrees, device, [int(online.deployed_width)], int(online.explore_plans),
                               float(online.explore_sigma), generator, bool(online.count_beam_alternatives),
                               epinet=epi, thompson=epi is not None and bool(ecfg.thompson))
            masks = [m for t in plans.values() for m in bushy.internal_masks(t)]
            if bool(fcfg.count_candidates):
                masks += candidate_masks(plans[deployed], q)
            known = counts.setdefault(q.key, {})
            tasks = [((q.key, m), count_query(q.patterns, m)) for m in dict.fromkeys(masks)
                     if m not in known and (q.key, m) not in queued]
            queued.update(key for key, _ in tasks)
            stream.add(tasks)                                    # counted while the next queries are planned
            deployed_of[q.key] = bushy.canonical(plans[deployed])
            known = plans_of.setdefault(q.key, [])
            for t in plans.values():
                c = bushy.canonical(t)
                if c in known:
                    known.remove(c)
                known.append(c)
            del known[:-int(fcfg.max_plans_per_query)]
        t1 = time.perf_counter()
        results = stream.finish()
        new, failed, count_s = store_counts(results), sum(v is None for v in results.values()), stream.seconds
        count_wait = time.perf_counter() - t1
        for q in batch:                                  # realized regret of this execution (true costs)
            r = realized_regret(deployed_of[q.key], plans_of[q.key], counts[q.key], failed_bound(counts[q.key], largest[0]))
            if r is not None:
                regret_of[q.key] = r
        executed = [regret_of[q.key] for q in batch if q.key in regret_of]
        cal_card = cal_value = None
        if epi is not None:                              # this round's counts: not trained on yet
            fresh: dict[str, dict] = {}
            for (key, m), c in results.items():
                if c is not None and m & (m - 1) and len(fresh.setdefault(key, {})) < 16:
                    fresh[key][m] = c
            cal_card = card_calibration(model, net, epi, [(q, fresh.get(q.key, {})) for q in batch], device, generator=generator)
            if str(ecfg.sample_scale) == "auto":
                epi.scale_card.fill_(fe.fit_scale(float(epi.scale_card), cal_card))
        for q in batch:
            seen.pop(q.key, None)
            seen[q.key] = q
        while len(seen) > int(online.replay_queries):
            gone = next(iter(seen))
            seen.pop(gone), plans_of.pop(gone, None), parts.pop(gone, None), regret_of.pop(gone, None), deployed_of.pop(gone, None)

        t2 = time.perf_counter()
        window = list(seen.values())
        for q in window:
            state = (len(counts[q.key]), tuple(plans_of[q.key]))
            if q.key not in parts or parts[q.key][1] != state:
                parts[q.key] = (fv.build_query_data(q.n, q.neighbour_masks, q.patterns, degrees, plans_of[q.key], counts[q.key],
                                                    failed_bound(counts[q.key], largest[0])), state)
                if epi is not None:
                    parts[q.key][0].update(fe.anchor_keys(parts[q.key][0], q.key))
        priority_w = None
        if epi is not None:
            fresh_q = [q for q in batch if q.key in new_keys][:256]
            cal_value = value_calibration(model, net, epi, fresh_q, [parts[q.key][0] for q in fresh_q], device,
                                          generator=generator, target=value_target)
            if str(ecfg.sample_scale) == "auto":
                epi.scale_value.fill_(fe.fit_scale(float(epi.scale_value), cal_value))
            if int(ecfg.train_select_factor) > 1:
                priority_w = fe.query_priorities(net, epi, start_items(model, window, device), device,
                                                 int(ecfg.n_indexes_priority), generator).cpu().numpy()
        t3 = time.perf_counter()
        model.train(), net.train()
        steps = max(1, int(float(online.gradient_passes_per_round) * len(window) / int(online.queries_per_batch)))
        stats = {k: [] for k in ("loss", "l1_raw", "l1_bound", "value", "w", "plan", "floor", "consistency", "choice",
                                 "epinet_card", "epinet_value", "trained_regret", "is_weight")}
        B = min(int(online.queries_per_batch), len(window))
        p_train = iw = None
        if prioritized:
            p_train = probabilities([q.key for q in window])
            frac = (round_index - 1) / max(int(online.rounds) - 1, 1)
            beta = float(rcfg.beta_start) + (float(rcfg.beta_end) - float(rcfg.beta_start)) * frac
            iw = importance_weights(p_train, beta)
        for _ in range(steps):
            if priority_w is not None:                   # the top B of f x B window queries by priority
                cand = rng.choice(len(window), min(int(ecfg.train_select_factor) * B, len(window)), replace=False)
                picked = cand[np.argsort(-priority_w[cand], kind="stable")[:B]]
            else:
                picked = rng.choice(len(window), B, replace=False, p=p_train)
            chosen = [window[k] for k in picked]
            batch_parts = [parts[q.key][0] for q in chosen]
            x = multi_query_batch([(q, p["masks"].to(device)) for q, p in zip(chosen, batch_parts)], device)
            phi_b, l_b = phi_and_direct(model, *x)
            m_all = torch.cat([p["masks"] for p in batch_parts]).to(device)
            exact = torch.cat([history_dp.single_pattern_logcounts(q.data, p["masks"].to(device)) for q, p in zip(chosen, batch_parts)])
            l_b = torch.where((m_all & (m_all - 1)) == 0, exact, l_b)
            out = fv.forward_batch(net, batch_parts, phi_b, l_b, device, tau, bool(fcfg.stop_gradient),
                                   value_chunk=VALUE_CHUNK, hidden=epi is not None)
            qw = torch.tensor(iw[picked], dtype=torch.float32, device=device) if iw is not None else None
            res = fv.losses(out, tau, qw)
            if prioritized:
                stats["trained_regret"].append(float(np.mean([regret_of.get(q.key, 0.0) for q in chosen])))
                stats["is_weight"].append(float(iw[picked].mean()))
            warm = round_index <= int(fcfg.plan_warmup_rounds)      # V learns from its own targets before the plan loss
            loss = sum(w * res[k] for k, w in weights.items() if not (warm and k == "plan"))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epi is not None:                          # its inputs are detached: a separate, smaller graph
                eres = fe.losses(epi, net, out, phi_b, float(ecfg.sigma), int(ecfg.n_indexes_train), int(ecfg.max_rows), generator,
                                 value_target=value_target)
                (float(ecfg.weight) * (eres["card"] + eres["value"])).backward()
                stats["epinet_card"].append(float(eres["card"])), stats["epinet_value"].append(float(eres["value"]))
            torch.nn.utils.clip_grad_norm_(trainable, float(online.grad_clip))
            if epi is not None:
                torch.nn.utils.clip_grad_norm_(epi.trainable(), float(online.grad_clip))
            optimizer.step()
            stats["loss"].append(float(loss))
            for k in ("l1_raw", "l1_bound", "value", "w", "plan", "floor", "consistency"):
                stats[k].append(float(res[k]))
            stats["choice"].append(res["states_with_choice"] / max(res["states"], 1))
        record = {"round": round_index, "queries": len(batch), "revisited": n_revisit, "counted": new, "failed": failed,
                  "plan_s": round(t1 - t0, 1), "count_s": round(count_s, 1), "count_wait_s": round(count_wait, 1),
                  "build_s": round(t3 - t2, 1),
                  "train_s": round(time.perf_counter() - t3, 1), "window": len(window), "steps": steps,
                  **{k: round(float(np.mean(v)), 4) for k, v in stats.items() if v}}
        record["executed_regret"] = round(float(np.mean(executed)), 4) if executed else None
        record["executed_regret>log2"] = round(float(np.mean(np.array(executed) > math.log(2))), 4) if executed else None
        if epi is not None:
            record["epinet"] = {"scale_card": round(float(epi.scale_card), 4), "scale_value": round(float(epi.scale_value), 4),
                                "fresh_card": cal_card, "fresh_value": cal_value, "selection": selection}
        with open(run_directory / "rounds.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[round {round_index}] {new:,} counts ({failed} failed) in {count_s:.0f}s | {steps} steps: L1 {record['l1_raw']:.3f} "
              f"(bounded {record['l1_bound']:.3f}), value {record['value']:.3f} (W {record['w']:.3f}), plan regret {record['plan']:.3f} (floor "
              f"{record['floor']:.3f}), consistency {record['consistency']:.3f}, states with a counted choice "
              f"{record['choice']:.0%} | plan {record['plan_s']:.0f}s (+{count_wait:.0f}s waiting for counts), build "
              f"{record['build_s']:.0f}s, train "
              f"{record['train_s']:.0f}s | window {len(window):,}", flush=True)
        if executed:
            replay = (f" | trained on regret {record['trained_regret']:.3f}, importance weight {record['is_weight']:.2f}"
                      if prioritized else "")
            print(f"[round {round_index} regret] executed queries: realized regret {record['executed_regret']:.3f}, "
                  f"> 2x for {record['executed_regret>log2']:.1%}{replay}", flush=True)
        if epi is not None:
            show = lambda c: "-" if c is None else (f"coverage90 {c['coverage90']:.2f}, Spearman(std, |err|) "
                                                    f"{c['spearman'] if c['spearman'] is not None else float('nan'):.2f}, "
                                                    f"mean std {c['mean_std']:.2f} vs |err| {c['mean_abs_error']:.2f}")
            sel = "" if selection is None else (f" | counted queries: priority {selection['selected_mean_priority']:.3f} "
                                                f"vs pool {selection['pool_mean_priority']:.3f}, n {selection['selected_mean_n']:.1f} "
                                                f"vs {selection['pool_mean_n']:.1f}")
            print(f"[round {round_index} epinet] loss card {record['epinet_card']:.3f} value {record['epinet_value']:.3f} | "
                  f"fresh subsets: {show(cal_card)} | fresh forests: {show(cal_value)} | sample scale card "
                  f"{float(epi.scale_card):.2f} value {float(epi.scale_value):.2f}{sel}", flush=True)
        if round_index % int(online.validation.every_rounds) == 0 or round_index == int(online.rounds):
            validate(round_index)


if __name__ == "__main__":
    main()
