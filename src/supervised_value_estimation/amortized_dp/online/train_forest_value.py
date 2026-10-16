"""Online training of the forest value planner (forest_value.py) and the cardinality model together, with true counts.

    python -m src.supervised_value_estimation.amortized_dp.online.train_forest_value --config-name forest_value_mixed_yago_v2.yaml

The planner is a beam over forests ranked by log(cost so far + V(F)), with the cardinality model's per-subset
estimates bounded by the components already built (forest_value.py's module text). It starts from the pretrained
cardinality model and an untrained planner (W = 0, gate ~ off): round 0 is the estimator beam with the hard degree
bound. Each ROUND, as train_plan_loss:
  1. take `queries_per_round` training queries (a share `revisit_fraction` already seen, re-planned);
  2. plan each with the current model: the beam of width `deployed_width` (all its final trees with
     `count_beam_alternatives`) and `explore_plans` beams with N(0, explore_sigma) noise on the scores; with
     forest.count_candidates also the result of every candidate merge along the deployed plan;
  3. COUNT the new subsets on QLever (a failed count costs the query's failed-count bound, 2 x its largest count);
  4. train on the most recent `replay_queries` counted queries (up to forest.max_plans_per_query counted plans each):
         l1_weight * L1(l) + forest.bound_l1_weight * L1(l'') + forest.value_weight * |V - log G|
         + plan_weight * per-step plan regret + forest.consistency_weight * |V - min Q^|   (forest_value.losses)
     with gradients into the backbone (online.lr) and the planner (forest.lr). The plan regret is off for the first
     forest.plan_warmup_rounds rounds: the untrained V would otherwise push the estimates to compensate for it.
Validation (round 0, then every validation.every_rounds) on the same 2,000 queries as train_plan_loss: "value GOO /
beam-4 / beam-8" (this planner) next to GNCE GOO / beam-4 / beam-8, and with forest.reference_planners the current
cardinality model's plain estimator beam-8 and its lattice DP with the hard degree bound. True C_out, geomean vs
GNCE GOO and vs round 0, plus the queries more than 10x better / worse than round 0 (the tail decides the geomean),
and the tail: percentiles (PERCENTILES) and shares > 2x / > 10x of the per-query C_out over a FIXED reference, the
cheapest plan any planner (ours or GNCE) found for the query at round 0.

Outputs in online.output_directory/forest-value-<time>/: config.yaml, rounds.jsonl, validation.jsonl,
validation-<round>.json, model-<round>.pt (cardinality model), planner-<round>.pt, counts.pkl.
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

from src.supervised_value_estimation.amortized_dp import bushy, forest_value as fv, history_dp, soft_dp
from src.supervised_value_estimation.amortized_dp.bushy_train import count_connected_subsets, load_backbone, load_split
from src.supervised_value_estimation.amortized_dp.labels import model_log_cardinalities
from src.supervised_value_estimation.amortized_dp.online.train_online import _endpoints
from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import connected, neighbour_masks_of
from src.supervised_value_estimation.amortized_dp.online.train_plan_loss import (BUCKETS, Query, failed_bound,
                                                                                 multi_query_batch, phi_and_direct,
                                                                                 plan_cost, subset_phi)
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import count_query, count_subsets

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


@torch.no_grad()
def plan_query(model, net, q, degrees, device, widths, explore=0, sigma=0.0, generator=None, alternatives=False,
               reference=False):
    """{name: tree} for "beam-w" (with alternatives also "beam-w #i"), "explore-i" and, with reference, the plain
    estimator "est beam-8" and "est DP + bound"; plus the deployed beam's states' candidate merges (masks)."""
    ctx = fv.QueryContext(q.n, q.neighbour_masks, q.patterns, degrees, encoder(model, q, device))
    plans = {}
    for w in widths:
        name = "GOO" if w == 1 else f"beam-{w}"
        trees, _ = fv.beam_plan(net, ctx, w, all_trees=True)
        plans[name] = trees[0]
        if alternatives:
            plans.update({f"{name} #{i}": t for i, t in enumerate(trees[1:], 1)})
    for i in range(explore):
        plans[f"explore-{i}"], _ = fv.beam_plan(net, ctx, widths[-1], noise=sigma, generator=generator)
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
        for (key, m), value in results.items():
            counts[key][m] = value
            if value is not None:
                largest[0] = max(largest[0], value)
        return len(tasks), sum(v is None for v in results.values()), time.perf_counter() - start

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
        torch.save(model.state_dict(), run_directory / f"model-{round_index}.pt")
        net.save(run_directory / f"planner-{round_index}.pt")
        with open(run_directory / "counts.pkl", "wb") as f:
            pickle.dump(counts, f)

    # --- training ------------------------------------------------------------------------------------
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": float(online.lr)},
                                   {"params": list(net.parameters()), "lr": float(fcfg.lr)}],
                                  weight_decay=float(online.weight_decay))
    trainable = list(model.parameters()) + list(net.parameters())
    seen: dict[str, Query] = {}
    plans_of: dict[str, list] = {}                       # query -> counted plans (canonical, most recent last)
    parts: dict[str, tuple] = {}                         # query -> (build_query_data part, counts seen, plans seen)
    weights = {"l1_raw": float(online.l1_weight), "l1_bound": float(fcfg.bound_l1_weight), "value": float(fcfg.value_weight),
               "plan": float(online.plan_weight), "consistency": float(fcfg.consistency_weight)}
    tau = float(online.plan_tau)
    next_index = 0
    if online.validation.at_start:
        validate(0)
    for round_index in range(1, int(online.rounds) + 1):
        t0 = time.perf_counter()
        n_revisit = min(int(round(float(online.revisit_fraction) * int(online.queries_per_round))), len(seen))
        batch = [seen[k] for k in rng.choice(list(seen), n_revisit, replace=False)] if n_revisit else []
        while len(batch) < int(online.queries_per_round):
            batch.append(Query(train_dataset[train_pool[next_index % len(train_pool)]]))
            next_index += 1
        model.eval(), net.eval()
        to_count = []
        deployed = f"beam-{int(online.deployed_width)}" if int(online.deployed_width) > 1 else "GOO"
        for q in tqdm(batch, desc="Planning", file=sys.stdout, mininterval=30):
            plans = plan_query(model, net, q, degrees, device, [int(online.deployed_width)], int(online.explore_plans),
                               float(online.explore_sigma), generator, bool(online.count_beam_alternatives))
            masks = [m for t in plans.values() for m in bushy.internal_masks(t)]
            if bool(fcfg.count_candidates):
                masks += candidate_masks(plans[deployed], q)
            to_count.append((q, masks))
            known = plans_of.setdefault(q.key, [])
            for t in plans.values():
                c = bushy.canonical(t)
                if c in known:
                    known.remove(c)
                known.append(c)
            del known[:-int(fcfg.max_plans_per_query)]
        t1 = time.perf_counter()
        new, failed, count_s = count_missing(to_count)
        for q in batch:
            seen.pop(q.key, None)
            seen[q.key] = q
        while len(seen) > int(online.replay_queries):
            gone = next(iter(seen))
            seen.pop(gone), plans_of.pop(gone, None), parts.pop(gone, None)

        t2 = time.perf_counter()
        window = list(seen.values())
        for q in window:
            state = (len(counts[q.key]), tuple(plans_of[q.key]))
            if q.key not in parts or parts[q.key][1] != state:
                parts[q.key] = (fv.build_query_data(q.n, q.neighbour_masks, q.patterns, degrees, plans_of[q.key], counts[q.key],
                                                    failed_bound(counts[q.key], largest[0])), state)
        t3 = time.perf_counter()
        model.train(), net.train()
        steps = max(1, int(float(online.gradient_passes_per_round) * len(window) / int(online.queries_per_batch)))
        stats = {k: [] for k in ("loss", "l1_raw", "l1_bound", "value", "plan", "floor", "consistency", "choice")}
        for _ in range(steps):
            chosen = [window[k] for k in rng.choice(len(window), min(int(online.queries_per_batch), len(window)), replace=False)]
            batch_parts = [parts[q.key][0] for q in chosen]
            x = multi_query_batch([(q, p["masks"].to(device)) for q, p in zip(chosen, batch_parts)], device)
            phi_b, l_b = phi_and_direct(model, *x)
            m_all = torch.cat([p["masks"] for p in batch_parts]).to(device)
            exact = torch.cat([history_dp.single_pattern_logcounts(q.data, p["masks"].to(device)) for q, p in zip(chosen, batch_parts)])
            l_b = torch.where((m_all & (m_all - 1)) == 0, exact, l_b)
            out = fv.forward_batch(net, batch_parts, phi_b, l_b, device, tau)
            res = fv.losses(out, tau)
            warm = round_index <= int(fcfg.plan_warmup_rounds)      # V learns from its own targets before the plan loss
            loss = sum(w * res[k] for k, w in weights.items() if not (warm and k == "plan"))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(online.grad_clip))
            optimizer.step()
            stats["loss"].append(float(loss))
            for k in ("l1_raw", "l1_bound", "value", "plan", "floor", "consistency"):
                stats[k].append(float(res[k]))
            stats["choice"].append(res["states_with_choice"] / max(res["states"], 1))
        record = {"round": round_index, "queries": len(batch), "revisited": n_revisit, "counted": new, "failed": failed,
                  "plan_s": round(t1 - t0, 1), "count_s": round(count_s, 1), "build_s": round(t3 - t2, 1),
                  "train_s": round(time.perf_counter() - t3, 1), "window": len(window), "steps": steps,
                  **{k: round(float(np.mean(v)), 4) for k, v in stats.items()}}
        with open(run_directory / "rounds.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[round {round_index}] {new:,} counts ({failed} failed) in {count_s:.0f}s | {steps} steps: L1 {record['l1_raw']:.3f} "
              f"(bounded {record['l1_bound']:.3f}), value {record['value']:.3f}, plan regret {record['plan']:.3f} (floor "
              f"{record['floor']:.3f}), consistency {record['consistency']:.3f}, states with a counted choice "
              f"{record['choice']:.0%} | plan {record['plan_s']:.0f}s, build {record['build_s']:.0f}s, train "
              f"{record['train_s']:.0f}s | window {len(window):,}", flush=True)
        if round_index % int(online.validation.every_rounds) == 0 or round_index == int(online.rounds):
            validate(round_index)


if __name__ == "__main__":
    main()
