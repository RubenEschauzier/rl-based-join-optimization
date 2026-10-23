"""Online plan-loss fine-tuning of a cardinality model for bushy C_out planning (soft_dp.py, "M1").

    python -m src.supervised_value_estimation.amortized_dp.online.train_plan_loss [--config-name ...] [overrides]

The planner is the cardinality model plus a search over its estimates (the exact bushy DP, or
bushy.estimator_beam_plan); nothing predicts a cost-to-go. Training makes the estimates good where
they decide the plan, with TRUE counts only (COUNT(*) of subsets on QLever), no oracle. Each ROUND:
  1. take `queries_per_round` training queries (a share `revisit_fraction` already seen, re-planned);
  2. estimate every connected subset (one batched forward), plan the hard DP plan, the beam plan of
     width `deployed_width` (with `count_beam_alternatives`, all the beam's final trees: the
     near-optimal alternatives the model chooses between) and `explore_plans` DP plans over estimates
     perturbed by N(0, explore_sigma) per subset (log scale);
  3. COUNT the plans' join results not counted before, on all endpoints in parallel; a failed or
     timed-out count is recorded and never retried. It stands for a large result: it gets the
     query's FAILED-COUNT BOUND, 2 x the largest successful count of that query (2 x the run's
     largest if the query has none), recomputed as counts arrive -- an assumption, not a measurement;
  4. train on the most recent `replay_queries` counted queries, per query over its counted subsets:
         L1   |estimate - log1p(count)|, as the pretraining loss; for a failed count one-sided,
              max(0, log1p(bound) - estimate) (right-censored, as the pretraining loss's censoring);
         plan log E_p[C_out true] - log min C_out true, over the plans stitched from counted subsets
              (soft_dp restricted to them, SubsetStructure(family=...)): p = the soft DP's plan
              distribution under the estimates at temperature `plan_tau`. Its gradient moves the
              estimates only where an error moves probability onto a truly dearer counted plan.
              A failed count is in the family with cost = the bound. Logged next to it: the FLOOR, the
              same regret with the true counts as estimates (the soft distribution's own spread, which
              no estimate removes), and the excess = plan regret - floor.
Validation (at the start = the pretrained model, then every `validation.every_rounds`): the same
seeded 500-query selection as train_online_bushy (and why_not_better.py), planned by
    ours bushy DP, ours GOO / beam-4 / beam-8 (current model), GNCE GOO / beam-4 / beam-8 (fixed),
with TRUE C_out (sum of the plans' join result counts, cached across validations; a failed count
costs the bound, and the number of plans with one is reported per planner) as the geomean ratio to
GNCE GOO and to the same planner at round 0, by query size.

With `history.enabled` (history_dp.py) the estimator is the cardinality model plus a history head inside the DP:
planning uses one history-DP pass at tau = 0 over all connected subsets, clamped by the degree bound; training
takes the same L1 and plan loss through the history DP (the plan loss on the counted family, the L1 on the full
lattice of queries with at most history.full_lattice_cap connected subsets, on the family otherwise), with
gradients into the backbone (online.lr) and the head (history.lr). history-<round>.pt holds the head.

Outputs in online.output_directory/plan-loss-<time>/: config.yaml, rounds.jsonl, validation.jsonl,
validation-<round>.json (per query plans and costs), model-<round>.pt (the estimator's state_dict),
counts.pkl (all counts: training and validation).
"""
from __future__ import annotations

import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.supervised_value_estimation.amortized_dp import bushy, history_dp, soft_dp
from src.supervised_value_estimation.amortized_dp.bushy_train import count_connected_subsets, load_backbone, load_split
from src.supervised_value_estimation.amortized_dp.labels import family_of, model_log_cardinalities
from src.supervised_value_estimation.amortized_dp.online.train_online import _endpoints
from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import connected, neighbour_masks_of
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import count_query, count_subsets

BUCKETS = {"all": lambda n: True, "<=6": lambda n: n <= 6, "7-12": lambda n: 7 <= n <= 12, "13-16": lambda n: n >= 13}


class Query:
    def __init__(self, data):
        self.data = data
        self.key = " ".join(data.query.split())
        self.patterns = list(data.triple_patterns)
        self.n = len(self.patterns)
        self.family = family_of(data.type if isinstance(data.type, str) else data.type[0])
        self.neighbour_masks = neighbour_masks_of(self.patterns)


def multi_query_batch(queries_and_masks, device):
    """One GNN input for the subsets of several queries: the soft_dp.subset_batch of each, with node
    and graph indices offset. Graphs are in order (query by query, mask order within)."""
    xs, edges, attrs, batches, nodes, graphs = [], [], [], [], 0, 0
    for q, masks in queries_and_masks:
        x, edge_index, edge_attr, batch = soft_dp.subset_batch(q.data, masks, device)
        xs.append(x), edges.append(edge_index + nodes), attrs.append(edge_attr), batches.append(batch + graphs)
        nodes += x.shape[0]
        graphs += len(masks)
    return torch.cat(xs), torch.cat(edges, 1), torch.cat(attrs), torch.cat(batches)


def log_cards(model, x, edge_index, edge_attr, batch):
    heads = model.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    return next(h["output"] for h in heads if h["output_type"] == "cardinality").reshape(-1)


def phi_and_direct(model, x, edge_index, edge_attr, batch, readout=None, masks=None):
    """(pooled embedding phi, log-card estimate) per graph: the cardinality head's input and output. readout: a
    row_readout.QueryReadout of the query in place of the backbone's RowReadout layer (masks: the subsets of the batch)."""
    if readout is None:
        phi = model.embedding_model.forward(x, edge_index, edge_attr, batch)
    else:
        from torch_geometric.nn.conv import MessagePassing
        x, edge_attr = readout(masks, x, edge_attr)
        for layer in list(model.embedding_model.children())[1:]:
            x = layer(x, edge_index, edge_attr) if isinstance(layer, MessagePassing) else layer(x, batch)
        phi = x
    return phi, model.heads[model.head_types.index("cardinality")](phi).reshape(-1)


def query_readout(model, data, device):
    """A row_readout.QueryReadout of the query when the backbone starts with a RowReadout layer, else None."""
    from src.models.model_layers.row_readout import QueryReadout, RowReadout
    first = next(iter(model.embedding_model.children()))
    return QueryReadout(first, data, device) if isinstance(first, RowReadout) else None


def subset_phi(model, data, masks, device, chunk=16384, readout=None):
    out = [phi_and_direct(model, *soft_dp.subset_batch(data, masks[i:i + chunk], device), readout=readout,
                          masks=masks[i:i + chunk]) for i in range(0, len(masks), chunk)]
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


@torch.no_grad()
def history_estimates(model, history, q, structure, device):
    """The history head's log1p estimates (hard DP histories, tau = 0) of every subset of `structure`, clamped by
    the degree bound: what the history planners plan with."""
    head, degrees = history
    phi, direct = subset_phi(model, q.data, structure.masks, device)
    b = history_dp.merge([{"structure": structure, "exact": history_dp.single_pattern_logcounts(q.data, structure.masks)}], device)
    L, _, _ = head(b, phi, direct, 0.0)
    return history_dp.degree_bound(q.patterns, structure, L.clamp(min=0), degrees)


def history_part(q, structure, counts, bound, device, l1=True):
    """history_dp.merge's part of one query: labels, failed counts and true log1p counts (failed = the bound)."""
    masks = structure.masks.tolist()
    label = torch.tensor([math.log1p(counts[m]) if counts.get(m) is not None else math.nan for m in masks], device=device)
    failed = torch.tensor([m in counts and counts[m] is None for m in masks], device=device)
    true = torch.tensor([math.log1p(bound if counts.get(m, 0) is None else counts.get(m, 0)) for m in masks], device=device)
    return {"structure": structure, "exact": history_dp.single_pattern_logcounts(q.data, structure.masks), "label": label,
            "failed": failed, "true": true, "l1": torch.full((len(masks),), l1, dtype=torch.bool, device=device)}


def failed_bound(counts, fallback):
    """The count a failed count of this query stands for: 2 x its largest successful count (2 x
    `fallback`, the run's largest, if it has none); at least 2."""
    done = [c for c in counts.values() if c is not None]
    return 2 * max(max(done) if done else fallback, 1)


def plan_cost(tree, counts, bound=None):
    """(True C_out of a plan: sum of its join results' counts, a failed (None) count costing `bound`;
    number of failed counts used). Cost None if a count is missing, or failed without a bound."""
    total, imputed = 0, 0
    for m in bushy.internal_masks(tree):
        if m not in counts or (counts[m] is None and bound is None):
            return None, 0
        if counts[m] is None:
            total, imputed = total + bound, imputed + 1
        else:
            total += counts[m]
    return total, imputed


@torch.no_grad()
def estimate_and_plan(model, q, device, widths, explore_plans=0, explore_sigma=0.0, generator=None, alternatives=False,
                      history=None):
    """Estimate every connected subset, then plan: {name: tree} for "DP", widths ("GOO" / "beam-w";
    with `alternatives` also the beam's other final trees, "beam-w #i") and the perturbed DP plans
    "explore-i"; also the number of subsets each beam estimated. With `history` ((head, degrees)) the
    estimates are history_estimates'."""
    s = soft_dp.SubsetStructure(q.neighbour_masks, device)
    if history is not None:
        est = history_estimates(model, history, q, s, device)
    else:
        est = soft_dp.model_subset_logcards(model, q.data, s.masks, device)
    plans = {"DP": soft_dp.hard_plan(s, soft_dp.soft_dp(s, est, 0.0)[1])}
    for i in range(explore_plans):
        noisy = est + explore_sigma * torch.randn(est.shape, device=est.device, generator=generator)
        plans[f"explore-{i}"] = soft_dp.hard_plan(s, soft_dp.soft_dp(s, noisy, 0.0)[1])
    table = dict(zip(s.masks.tolist(), est.tolist()))
    evaluations = {"DP": len(s)}
    for w in widths:
        name = "GOO" if w == 1 else f"beam-{w}"
        trees, evaluations[name] = bushy.estimator_beam_plan(q.n, q.neighbour_masks, lambda ms: [table[m] for m in ms], w,
                                                             all_trees=True)
        plans[name] = trees[0]
        if alternatives:
            plans.update({f"{name} #{i}": t for i, t in enumerate(trees[1:], 1)})
    del s
    return plans, evaluations


def query_loss(model_out, q, family, counts, tau, device, bound):
    """(L1 term summed, number of L1 terms, plan loss or None, its floor or None) of one query, from the
    model's estimates `model_out` (aligned with the family structure's masks). A failed (None) count is
    `bound`: a one-sided L1 (only estimates below it) and its cost in the plans. The floor is the plan
    loss with the true counts as the estimates (no gradient)."""
    structure, masks = family
    failed = torch.tensor([counts.get(m, 0) is None for m in masks], device=device)
    true = torch.tensor([math.log1p(bound if counts.get(m, 0) is None else counts.get(m, 0)) for m in masks],
                        device=device, dtype=model_out.dtype)
    joins = ~structure.singleton
    error = torch.where(failed, torch.relu(true - model_out), (model_out - true).abs())
    l1 = error[joins].sum()
    if structure.full < 0:
        return l1, int(joins.sum()), None, None
    _, log_p = soft_dp.soft_dp(structure, model_out, tau)
    expected = soft_dp.expected_log_cost(structure, soft_dp.plan_marginals(structure, log_p), true)
    with torch.no_grad():
        best = soft_dp.soft_dp(structure, true, 0.0)[0][structure.full]
        floor = soft_dp.expected_log_cost(structure, soft_dp.plan_marginals(structure, soft_dp.soft_dp(structure, true, tau)[1]),
                                          true) - best
    if float(best) >= soft_dp.UNBUILT / 2:              # no complete plan from counted subsets
        return l1, int(joins.sum()), None, None
    return l1, int(joins.sum()), expected - best, float(floor)


def history_entry(q, structure, counts, bound, tau, full_cap, device):
    """(family part, whether the query's L1 is taken on its full lattice, log optimum C_out over the family or inf,
    plan-loss floor) of a counted query: what history_step needs, rebuilt when the query gets new counts."""
    full = count_connected_subsets(q.neighbour_masks, full_cap) <= full_cap
    part = history_part(q, structure, counts, bound, device, l1=not full)
    best, floor = math.inf, 0.0
    if structure.full >= 0:
        with torch.no_grad():
            b0 = float(soft_dp.soft_dp(structure, part["true"], 0.0)[0][structure.full])
            if b0 < soft_dp.UNBUILT / 2:
                best = b0
                mu = soft_dp.plan_marginals(structure, soft_dp.soft_dp(structure, part["true"], tau)[1])
                floor = float(soft_dp.expected_log_cost(structure, mu, part["true"])) - b0
    return part, full, best, floor


def history_step(model, head, chosen, families, cache, counts, largest, tau, device):
    """(L1 mean, plan loss mean, floors of the queries with a plan loss) of one batch through the history DP, with
    gradients into the backbone and the head. The plan loss runs on the counted families; the L1 on the full lattice
    of the queries that have one in the cache (built here, they are not kept), on the family for the others."""
    entries = [cache[q.key] for q in chosen]
    phi, direct = phi_and_direct(model, *multi_query_batch([(q, families[q.key][0].masks) for q in chosen], device))
    b = history_dp.merge([e[0] for e in entries], device)
    L, _, log_p = head(b, phi, direct, tau)
    plan, ok = history_dp.plan_losses(b, log_p, torch.tensor([e[2] for e in entries], device=device))
    err, use = history_dp.l1_terms(L, b)
    terms = [err[use]]
    full = [q for q, e in zip(chosen, entries) if e[1]]
    if full:
        structures = [soft_dp.SubsetStructure(q.neighbour_masks, device) for q in full]
        phi_f, direct_f = phi_and_direct(model, *multi_query_batch([(q, s.masks) for q, s in zip(full, structures)], device))
        bf = history_dp.merge([history_part(q, s, counts[q.key], failed_bound(counts[q.key], largest), device)
                               for q, s in zip(full, structures)], device)
        Lf, _, _ = head(bf, phi_f, direct_f, tau)
        ef, uf = history_dp.l1_terms(Lf, bf)
        terms.append(ef[uf])
    terms = torch.cat(terms)
    zero = torch.zeros((), device=device)
    floors = [e[3] for e, o in zip(entries, ok.tolist()) if o]
    return (terms.mean() if len(terms) else zero), (plan[ok].mean() if bool(ok.any()) else zero), floors


@hydra.main(version_base=None, config_path="../../../../experiments/experiment_configs/amortized_dp",
            config_name="plan_loss_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    online = cfg.online
    rng = np.random.default_rng(int(online.seed))
    torch.manual_seed(int(online.seed))
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(online.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_directory = Path(online.output_directory) / f"plan-loss-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    run_directory.mkdir(parents=True, exist_ok=False)
    with open(run_directory / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {run_directory} | device {device}", flush=True)

    model, epoch = load_backbone(cfg.estimator.config, cfg.estimator.run, cfg.estimator.select, device)
    print(f"Estimator: {cfg.estimator.run} (epoch {epoch}), {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    history = None                                       # (history head, predicate degree statistics)
    if cfg.get("history") and bool(cfg.history.enabled):
        with open(cfg.history.degrees) as f:
            degrees = json.load(f)
        head = history_dp.HistoryDP.load(cfg.history.init, device) if cfg.history.get("init") else history_dp.HistoryDP().to(device)
        history = (head, degrees)
        print(f"History head: {cfg.history.get('init') or 'new (delta = 0: starts as the estimator)'}, "
              f"{sum(p.numel() for p in head.parameters()):,} parameters; degrees of {len(degrees)} predicates", flush=True)

    # --- queries: the same seeded selection as train_online_bushy ------------------------------------
    train_dataset = load_split(cfg.data.train_dir)
    train_pool = [i for i in range(len(train_dataset)) if connected(neighbour_masks_of(train_dataset[i].triple_patterns))]
    rng.shuffle(train_pool)
    if history is not None and not cfg.history.get("init"):
        with torch.no_grad():                            # the head's phi standardisation, from a sample
            phis = []
            for i in train_pool[:64]:
                q = Query(train_dataset[i])
                if count_connected_subsets(q.neighbour_masks, 1024) <= 1024:
                    phis.append(subset_phi(model, q.data, soft_dp.SubsetStructure(q.neighbour_masks, device).masks, device)[0])
            phi = torch.cat(phis)
            history[0].mean.copy_(phi.mean(0)), history[0].std.copy_(phi.std(0).clamp(min=1e-3))
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
    count_timeout = float(online.count_timeout_s)
    counts: dict[str, dict[int, int | None]] = {}      # query -> {subset mask: COUNT(*), None if it failed}
    largest = [0]                                        # the run's largest successful count (failed_bound fallback)
    seed_cache = online.validation.get("true_counts_cache")
    if seed_cache and Path(seed_cache).exists():
        with open(seed_cache, "rb") as f:
            for key, table in pickle.load(f).items():
                counts[key] = {m: c for m, c in table.items() if c is not None}
                largest[0] = max([largest[0], *counts[key].values()])
        print(f"Seeded counts from {seed_cache}: {sum(len(t) for t in counts.values()):,}", flush=True)

    def count_missing(pairs):
        """pairs: [(Query, masks)] -> counts the masks not counted before; returns (new, failed, seconds)."""
        tasks = []
        for q, masks in pairs:
            known = counts.setdefault(q.key, {})
            tasks += [((q.key, m), count_query(q.patterns, m)) for m in dict.fromkeys(masks) if m not in known]
        tasks = list(dict(tasks).items())
        start = time.perf_counter()
        results = count_subsets(tasks, endpoints, count_timeout) if tasks else {}
        for (key, m), value in results.items():
            counts[key][m] = value
            if value is not None:
                largest[0] = max(largest[0], value)
        return len(tasks), sum(v is None for v in results.values()), time.perf_counter() - start

    # --- validation --------------------------------------------------------------------------------
    reference = {}                                       # planner -> {query: true C_out at round 0}

    def validate(round_index):
        model.eval()
        start = time.perf_counter()
        plans, evaluations = {}, {}
        for v in validation:
            ours, evals = estimate_and_plan(model, v, device, widths, history=history)
            plans[v.key] = {f"ours {'bushy DP' if k == 'DP' else k}": t for k, t in ours.items()} | gnce_plans[v.key]
            evaluations[v.key] = {f"ours {'bushy DP' if k == 'DP' else k}": e for k, e in evals.items()}
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
        summary = {"round": round_index, "counted": new, "failed": failed, "count_s": round(seconds, 1),
                   "plan_s": round(planned, 1), "planners": {}}

        def geo(name, base):
            out = {}
            for b, keep in BUCKETS.items():
                r = [math.log(max(cost[name][v.key], 1) / max(base[v.key], 1)) for v in validation if keep(v.n)
                     and cost[name][v.key] is not None and base.get(v.key) is not None]
                out[b] = round(float(np.exp(np.mean(r))), 4) if r else None
            return out
        for name in names:
            summary["planners"][name] = {"vs_GNCE_GOO": geo(name, cost["GNCE GOO"]),
                                         "plans_with_failed_count": imputed[name],
                                         "vs_round_0": geo(name, reference[name]) if name.startswith("ours") else None,
                                         "subsets_estimated_median": float(np.median([evaluations[v.key][name] for v in validation]))
                                         if name.startswith("ours") else None}
        with open(run_directory / "validation.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        with open(run_directory / f"validation-{round_index}.json", "w") as f:
            json.dump({v.key: {"n": v.n, "family": v.family, **{name: {"plan": bushy.to_json(plans[v.key][name]),
                       "true_cout": cost[name][v.key], "failed_counts": costed[name][v.key][1]} for name in names}}
                       for v in validation}, f)
        print(f"\n[validation round {round_index}] true C_out, geomean vs GNCE GOO ({new:,} new counts, "
              f"{failed} failed, {seconds:.0f}s):", flush=True)
        print(f"  {'planner':18s}" + "".join(f"{b:>8s}" for b in BUCKETS) + "  failed*   vs round 0 (all)", flush=True)
        for name in names:
            p = summary["planners"][name]
            line = f"  {name:18s}" + "".join(f"{x:8.3f}" if x is not None else f"{'-':>8s}" for x in p["vs_GNCE_GOO"].values())
            line += f"{imputed[name]:9d}"
            if p["vs_round_0"]:
                line += f"   {p['vs_round_0']['all']:.3f}"
            print(line, flush=True)
        print("  * plans with a failed count, costed at the query's failed-count bound (2 x its largest count)", flush=True)
        torch.save(model.state_dict(), run_directory / f"model-{round_index}.pt")
        if history is not None:
            history[0].save(run_directory / f"history-{round_index}.pt")
        with open(run_directory / "counts.pkl", "wb") as f:
            pickle.dump(counts, f)

    # --- training ------------------------------------------------------------------------------------
    trainable = list(model.parameters()) + (list(history[0].parameters()) if history is not None else [])
    groups = [{"params": list(model.parameters()), "lr": float(online.lr)}]
    if history is not None:
        groups.append({"params": list(history[0].parameters()), "lr": float(cfg.history.lr)})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(online.weight_decay))
    history_cache: dict[str, tuple] = {}                 # query -> (family part, full lattice?, log optimum, floor)
    full_cap = int(cfg.history.full_lattice_cap) if history is not None else 0
    seen: dict[str, Query] = {}                          # counted training queries, oldest first
    families: dict[str, tuple] = {}                      # query -> (family structure, its masks, counts seen)
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
        model.eval()
        to_count = []
        for q in batch:
            plans, _ = estimate_and_plan(model, q, device, [int(online.deployed_width)], int(online.explore_plans),
                                         float(online.explore_sigma), generator, bool(online.count_beam_alternatives),
                                         history=history)
            to_count.append((q, [m for t in plans.values() for m in bushy.internal_masks(t)]))
        t1 = time.perf_counter()
        new, failed, count_s = count_missing(to_count)
        for q in batch:
            seen.pop(q.key, None)
            seen[q.key] = q                               # most recent last
        while len(seen) > int(online.replay_queries):
            seen.pop(next(iter(seen)))

        # one structure per replayed query over its counted subsets (tens of subsets), rebuilt only when
        # the query got new counts
        model.train()
        window = list(seen.values())
        for key in [k for k in families if k not in seen]:
            del families[key]
            history_cache.pop(key, None)
        for q in window:
            if q.key not in families or families[q.key][2] != len(counts[q.key]):
                s = soft_dp.SubsetStructure(q.neighbour_masks, device, family=list(counts[q.key]))   # failed ones too
                families[q.key] = (s, s.masks.tolist(), len(counts[q.key]))
                if history is not None:
                    history_cache[q.key] = history_entry(q, s, counts[q.key], failed_bound(counts[q.key], largest[0]),
                                                         float(online.plan_tau), full_cap, device)
        steps = max(1, int(float(online.gradient_passes_per_round) * len(window) / int(online.queries_per_batch)))
        stats = {"l1": [], "plan": [], "floor": [], "loss": []}
        for _ in range(steps):
            chosen = [window[k] for k in rng.choice(len(window), min(int(online.queries_per_batch), len(window)), replace=False)]
            if history is not None:
                l1_mean, plan_mean, floors = history_step(model, history[0], chosen, families, history_cache, counts,
                                                          largest[0], float(online.plan_tau), device)
            else:
                out = log_cards(model, *multi_query_batch([(q, families[q.key][0].masks) for q in chosen], device))
                l1_sum, l1_n, plan_terms, floors, offset = 0.0, 0, [], [], 0
                for q in chosen:
                    s, masks, _ = families[q.key]
                    l1, n_terms, plan, floor = query_loss(out[offset:offset + len(s)], q, (s, masks), counts[q.key],
                                                          float(online.plan_tau), device, failed_bound(counts[q.key], largest[0]))
                    offset += len(s)
                    l1_sum, l1_n = l1_sum + l1, l1_n + n_terms
                    if plan is not None:
                        plan_terms.append(plan)
                        floors.append(floor)
                l1_mean = l1_sum / max(l1_n, 1)
                plan_mean = torch.stack(plan_terms).mean() if plan_terms else torch.zeros((), device=device)
            loss = float(online.l1_weight) * l1_mean + float(online.plan_weight) * plan_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(online.grad_clip))
            optimizer.step()
            stats["l1"].append(float(l1_mean)), stats["plan"].append(float(plan_mean)), stats["loss"].append(float(loss))
            stats["floor"].append(float(np.mean(floors)) if floors else 0.0)
        record = {"round": round_index, "queries": len(batch), "revisited": n_revisit, "counted": new, "failed": failed,
                  "plan_s": round(t1 - t0, 1), "count_s": round(count_s, 1), "train_s": round(time.perf_counter() - t1 - count_s, 1),
                  "window": len(window), "steps": steps, **{k: round(float(np.mean(v)), 4) for k, v in stats.items()}}
        record["excess"] = round(record["plan"] - record["floor"], 4)
        with open(run_directory / "rounds.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[round {round_index}] {new:,} counts ({failed} failed) in {count_s:.0f}s | {steps} steps: "
              f"L1 {record['l1']:.3f}, plan regret {record['plan']:.3f} = floor {record['floor']:.3f} + excess "
              f"{record['excess']:.3f} (log) | window {len(window):,} queries | "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        if round_index % int(online.validation.every_rounds) == 0 or round_index == int(online.rounds):
            validate(round_index)


if __name__ == "__main__":
    main()
