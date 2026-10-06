"""Online training of the amortised-DP planner on real executions (no epinet yet).

    python -m src.supervised_value_estimation.amortized_dp.online.train_online
    python -m src.supervised_value_estimation.amortized_dp.online.train_online online.execution.mode=simulated

Starts from an offline-trained value net and adds a latency cost-to-go head. Each ROUND:
  1. take the next `queries_per_round` training queries;
  2. plan K plans per query with the current model: its greedy plan plus K-1 one-step
     deviations at the closest calls (partial_observation.deviation_plans), ranked by
     predicted C_out cost-to-go;
  3. execute them, each with a timeout fixed before the round:
         min(default, max(min, multiplier * fastest known latency of that query));
  4. add what each runtime tree reveals to the query's observation store: exact
     cardinalities of every intermediate and scan, and per-step times;
  5. train on the most recent `replay_queries` observed queries: card, stitched C_out
     cost-to-go and stitched latency cost-to-go (both upper bounds: one-sided losses), and the
     listwise loss over observed children; `gradient_passes_per_round` passes over the
     window's states (or a fixed `gradient_steps_per_round`).
Every `validation.every_rounds`, the greedy plan of each validation query is executed once,
back to back with QLever's own plan (join order left to QLever), and the
latency, timeout rate and speedup are logged. Plans are chosen by C_out in this version; the
latency head is trained alongside but not yet used for decisions.

Outputs in online.output_directory/online-<mode>-<time>/: config.yaml, rounds.jsonl,
validation.jsonl, model-<round>.pt, and state.pkl (observations, for resuming).
"""
from __future__ import annotations

import json
import math
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from src.supervised_value_estimation.amortized_dp.agents import (
    AmortizedDPAgent, EpinetAmortizedDPAgent, load_cardinality_gnn,
)
from src.supervised_value_estimation.amortized_dp.compare_agents import _resolve_checkpoint
from src.supervised_value_estimation.amortized_dp.data import (
    compute_embeddings, load_datasets, load_split, query_index,
)
from src.supervised_value_estimation.amortized_dp.labels import (
    QueryLabels, connected_subset_masks, family_of, model_log_cardinalities, optimal_left_deep_plan,
)
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet
from src.supervised_value_estimation.amortized_dp.online.executor import RayPlanExecutor, SimulatedExecutor
from src.supervised_value_estimation.amortized_dp.online.exploration import thompson_plans
from src.supervised_value_estimation.amortized_dp.online.observations import QueryObservations
from src.supervised_value_estimation.amortized_dp.online.plan_quality import (
    join_cout, optimal_join_cout, p_error, ratio_summary, tree_join_cout,
)
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import load_or_count
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import parse_execution
from src.supervised_value_estimation.amortized_dp.partial_observation import _TrainingSettings, deviation_plans
from src.supervised_value_estimation.amortized_dp.train_amortized_dp import (
    StateTables, calibrate_epinet_prior, training_step,
)
from src.supervised_value_estimation.optuna_epinet_sweep import _resolve_model_paths
from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search


def _is_connected(neighbour_masks):
    reached, frontier = 1, 1
    while frontier:
        new = 0
        for i in range(len(neighbour_masks)):
            if frontier >> i & 1:
                new |= neighbour_masks[i]
        frontier = new & ~reached
        reached |= new
    return reached == (1 << len(neighbour_masks)) - 1


def _observations_for(data):
    masks, neighbour_masks = connected_subset_masks(data)
    observations = QueryObservations(data.query, family_of(data.type), len(data.triple_patterns), neighbour_masks)
    observations.triple_patterns = list(data.triple_patterns)
    return observations


def _timeout(observations, timeouts):
    best = observations.best_latency_s
    if best is None:
        return float(timeouts.default_s)
    return float(min(timeouts.default_s, max(timeouts.min_s, timeouts.multiplier * best)))


def _load_model(checkpoint, device, epinet_config=None):
    """The offline value net plus the online heads (latency-to-go, step latency), which start
    fresh. The epinet comes from the checkpoint if it has one (trained offline on card and G;
    its latency/step heads start fresh), otherwise from `epinet_config` (None: no epinet)."""
    saved = torch.load(checkpoint, map_location=device)
    kwargs = {**saved["model_kwargs"], "latency_head": True, "step_latency_head": True}
    kwargs["epinet"] = saved["model_kwargs"].get("epinet") or epinet_config
    model = ContractedJoinGraphValueNet(**kwargs)
    missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
    fresh = ("latency", "step", "first_join_node", "epinet_step")
    new = sorted(k for k in missing if not k.startswith(fresh) and ".latency." not in k
                 and not (k.startswith("epinet_") and not saved["model_kwargs"].get("epinet")))
    if new or unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: missing {new}, unexpected {unexpected}")
    return model.to(device), kwargs


def _endpoints(execution):
    """Endpoint URLs: `endpoints` if given, otherwise every host in `endpoint_hosts` with
    `ports_per_host` consecutive ports from `base_port` (deploy_isolated_qlever_instances.py
    starts one instance per port, from 7000 up)."""
    if execution.get("endpoints"):
        return list(execution.endpoints)
    # Bare addresses; a scheme or trailing slash ("http://10.2.32.224/") is tolerated.
    hosts = [str(host).split("://", 1)[-1].rstrip("/") for host in (execution.get("endpoint_hosts") or [])]
    if not hosts:
        raise ValueError("Set online.execution.endpoint_hosts (the QLever nodes' addresses) or endpoints.")
    return [f"http://{host}:{port}" for host in hosts
            for port in range(int(execution.base_port), int(execution.base_port) + int(execution.ports_per_host))]


def _geomean_speedup(baseline, ours):
    """Geometric mean over queries of baseline latency / our latency (1 ms floor)."""
    if not ours:
        return None
    ratio = np.maximum(np.asarray(baseline, dtype=float), 0.001) / np.maximum(np.asarray(ours, dtype=float), 0.001)
    return float(np.exp(np.mean(np.log(ratio))))


def _time_summary(milliseconds):
    values = np.asarray(milliseconds, dtype=float)
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)), "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(values.max())}


def _plan_cout(plan, rows, result, triple_patterns):
    """C_out (sum of join outputs) of an executed left-deep plan: from the true table, or else
    from the prefix row counts its own runtime tree revealed."""
    cost = join_cout(plan, rows) if rows else None
    if cost is None:
        parsed = parse_execution(result["tree"], plan, triple_patterns)
        if parsed["aligned"] and len(parsed["prefix_rows"]) == len(plan) - 1:
            cost = sum(parsed["prefix_rows"].values())
    return cost


def _epinet_card_diagnostics(model, val_dataset, val_by_query, true_rows, embeddings, indexes, device):
    """Does the epinet's uncertainty point at its errors? Over every connected subset of the
    validation queries with a known true count: the spread (std over the sampled indices) of
    the predicted log cardinality against the base prediction's absolute error in log1p rows
    (Spearman rank correlation), and how often the truth lies inside the samples' 5-95% range."""
    from scipy.stats import spearmanr
    spreads, errors, covered = [], [], []
    for query, rows in true_rows.items():
        known = [mask for mask, value in rows.items() if value is not None]
        if not known:
            continue
        agent = EpinetAmortizedDPAgent(model, embed_fn=None, epistemic_indexes=indexes, device=device,
                                       embedding_cache=embeddings)
        episode = agent.setup_episode(Batch.from_data_list([val_dataset[val_by_query[query]]]))
        agent._evaluate_states(episode, known)
        base = model.unstandardise_head("card", torch.stack([episode["base"]["state"][m]["card"] for m in known]))
        for mask, prediction in zip(known, base.tolist()):
            samples, truth = episode["values"]["card"][mask], math.log1p(rows[mask])
            spreads.append(float(np.std(samples)))
            errors.append(abs(prediction - truth))
            low, high = np.percentile(samples, [5, 95])
            covered.append(low <= truth <= high)
    if len(spreads) < 3:
        return {"n": len(spreads)}
    return {"n": len(spreads), "spearman_std_vs_error": float(spearmanr(spreads, errors).statistic),
            "mean_std": float(np.mean(spreads)), "mean_abs_error": float(np.mean(errors)),
            "coverage_90": float(np.mean(covered))}


def _summarise_latencies(latencies, censored, native=None):
    # QLever reports whole milliseconds, so ratios of sub-millisecond latencies are noise.
    latencies = np.maximum(np.asarray(latencies, dtype=float), 0.001)
    summary = {"n": int(len(latencies)), "mean_s": float(latencies.mean()), "median_s": float(np.median(latencies)),
               "p90_s": float(np.percentile(latencies, 90)), "p99_s": float(np.percentile(latencies, 99)),
               "timeout_rate": float(np.mean(censored))}
    if native is not None:
        native = np.maximum(np.asarray(native, dtype=float), 0.001)
        ratio = native / latencies
        summary.update({"native_mean_s": float(native.mean()), "native_median_s": float(np.median(native)),
                        "geomean_speedup_vs_native": float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-6))))),
                        "frac_faster_than_native": float(np.mean(latencies < native)),
                        "frac_slower_than_2x_native": float(np.mean(latencies > 2 * native))})
    return summary


@hydra.main(version_base=None,
            config_path="../../../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
            config_name="online_amortized_dp_mixed_yago.yaml")
def main(cfg: DictConfig):
    _resolve_model_paths(cfg)
    online, cfg_training = cfg.online, cfg.amortized_dp.training
    rng = np.random.default_rng(online.seed)
    torch.manual_seed(online.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_directory = Path(online.output_directory) / (
        f"online-{online.execution.mode}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    run_directory.mkdir(parents=True, exist_ok=False)
    with open(run_directory / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {run_directory} | device {device}")

    train_dataset, val_dataset = load_datasets(cfg)
    val_queries, _ = load_split(cfg)
    train_pool = [i for i in range(len(train_dataset))
                  if _is_connected(connected_subset_masks(train_dataset[i])[1])]
    rng.shuffle(train_pool)
    val_index = query_index(val_dataset)
    validation_pool = [val_index[q] for q in val_queries if q in val_index
                       and _is_connected(connected_subset_masks(val_dataset[val_index[q]])[1])]
    validation_pool = [validation_pool[k] for k in
                       sorted(rng.choice(len(validation_pool), min(online.validation.n_queries, len(validation_pool)),
                                         replace=False))]
    print(f"Train pool: {len(train_pool)} variable-connected queries | validation: {len(validation_pool)}")

    checkpoint = _resolve_checkpoint(online.init_checkpoint, cfg.amortized_dp.output_directory, cfg)
    epinet_cfg = cfg.amortized_dp.get("epinet")
    epinet_config = (OmegaConf.to_container(epinet_cfg.model, resolve=True)
                     if epinet_cfg is not None and epinet_cfg.enabled else None)
    model, model_kwargs = _load_model(checkpoint, device, epinet_config)
    # Epinet training settings (sigma, indices per step); None trains no epinet term.
    epinet_settings = epinet_cfg if model.has_epinet and epinet_cfg is not None else None
    plan_selection = online.get("plan_selection", "deviation")
    if plan_selection == "thompson" and not model.has_epinet:
        raise ValueError("online.plan_selection=thompson needs an epinet (amortized_dp.epinet.enabled).")
    print(f"Epinet: {model.epinet_config if model.has_epinet else 'none'} | plan selection: {plan_selection}, "
          f"objective {online.get('decision_objective', 'cout')}")
    thompson_generator = torch.Generator(device=device)
    thompson_generator.manual_seed(int(online.seed))
    print(f"Initialised from {checkpoint} (+ new latency head)")
    # config.yaml only holds the spec ("latest:gine"); record which offline run it resolved to.
    with open(run_directory / "init_checkpoint.json", "w", encoding="utf-8") as f:
        json.dump({"spec": str(online.init_checkpoint), "checkpoint": str(checkpoint)}, f, indent=2)
    embedder = load_cardinality_gnn(cfg.models.embedder.config, cfg.models.embedder.dir, device)

    if online.execution.mode == "ray":
        endpoints = _endpoints(online.execution)
        print(f"{len(endpoints)} QLever endpoints: {endpoints[0]} ... {endpoints[-1]}")
        executor = RayPlanExecutor(endpoints, n_actors=online.execution.n_actors,
                                   in_flight_per_endpoint=online.execution.in_flight_per_endpoint,
                                   ray_address=online.execution.get("ray_address"))
    elif online.execution.mode == "simulated":
        # No database: the oracle's cardinalities stand in for execution. Needs oracle labels.
        from src.supervised_value_estimation.amortized_dp.data import load_or_build_labels
        oracle = load_cardinality_gnn(cfg.models.oracle.config, cfg.models.oracle.dir, device)
        wanted = train_pool[:online.rounds * online.queries_per_round]
        labels = load_or_build_labels(Path(cfg.amortized_dp.label_directory) / "online_simulated_train.pkl",
                                      train_dataset, wanted, oracle, device)
        labels.update(load_or_build_labels(Path(cfg.amortized_dp.label_directory) / "online_simulated_val.pkl",
                                           val_dataset, validation_pool, oracle, device))
        del oracle
        executor = SimulatedExecutor(labels, seed=online.seed)
    else:
        raise ValueError(f"online.execution.mode must be ray or simulated, got {online.execution.mode}")

    # --- Validation baselines and ground truth, fixed for the whole run ---------------------
    validation = online.validation
    val_by_query = {val_dataset[i].query: i for i in validation_pool}
    skeletons = {}                      # query -> (QueryLabels with its join graph, connected subsets)
    for query, i in val_by_query.items():
        data = val_dataset[i]
        masks, neighbour_masks = connected_subset_masks(data)
        skeletons[query] = (QueryLabels(query=query, family=family_of(data.type), n_tp=len(data.triple_patterns),
                                        neighbour_masks=neighbour_masks, logcard={m: 0.0 for m in masks}), masks)

    # Exact DP over the learned cardinality model (the cardinality GNN, models.embedder): its
    # plans never change, so they are planned once; the planning time covers estimating every
    # connected subset plus the DP, on this process's device.
    dp_plans, dp_planning_ms = {}, []
    if validation.get("include_dp_learned_cardinality", True):
        for query, i in val_by_query.items():
            data = val_dataset[i]
            skeleton, masks = skeletons[query]
            start = time.perf_counter()
            estimates = model_log_cardinalities(embedder, data, masks, device)
            estimated = QueryLabels(query=query, family=skeleton.family, n_tp=skeleton.n_tp,
                                    neighbour_masks=skeleton.neighbour_masks, logcard=dict(zip(masks, estimates)))
            dp_plans[query] = optimal_left_deep_plan(estimated)[0]
            dp_planning_ms.append(1000 * (time.perf_counter() - start))
        print(f"DP over learned cardinalities: planned {len(dp_plans)} validation queries, "
              f"median {np.median(dp_planning_ms):.1f} ms per query")

    # True row counts of every connected subset -> the optimal plan for the p-error.
    true_rows = {}
    if validation.true_cardinalities.enabled:
        if online.execution.mode == "ray":
            path = (validation.true_cardinalities.get("path")
                    or Path(cfg.amortized_dp.label_directory) / "validation_true_cardinalities.pkl")
            true_rows = load_or_count(path, {q: (list(val_dataset[i].triple_patterns), skeletons[q][1])
                                             for q, i in val_by_query.items()},
                                      endpoints, float(validation.true_cardinalities.timeout_s))
        else:                           # simulated: the oracle's cardinalities are the truth
            true_rows = {q: {m: int(round(math.exp(labels[q].logcard[m]))) for m in skeletons[q][1]}
                         for q in val_by_query}
    optimum = {q: optimal_join_cout(skeletons[q][0], rows)[0] for q, rows in true_rows.items()}
    print(f"Optimal C_out known for {sum(v is not None for v in optimum.values())}/{len(val_by_query)} "
          f"validation queries")
    # Robust planner: CVaR over a FIXED set of sampled indices, so validations stay comparable.
    robust = validation.get("robust")
    robust_indexes = None
    if robust is not None and robust.enabled and model.has_epinet:
        robust_generator = torch.Generator(device=device)
        robust_generator.manual_seed(int(robust.seed))
        robust_indexes = model.epinet_state.sample_epistemic_indexes_batched(int(robust.n_samples),
                                                                             generator=robust_generator)
    native_cout = {}                    # QLever's own plan never changes: its C_out, once known
    # Baselines whose plans are fixed are re-executed at every validation; their latency is
    # reported both for this validation and as a per-query average over all validations so far.
    history = {"native": defaultdict(list), "dp": defaultdict(list)}

    observations: dict[str, QueryObservations] = {}
    embeddings: dict[str, torch.Tensor] = {}
    # Queries by their most recent execution (dict: insertion-ordered, O(1) move to the end).
    # The replay window is the last `replay_queries` of these, so a query executed again in a
    # later pass over the pool re-enters the window with its new observations.
    order_seen: dict[str, None] = {}
    loss_settings = _TrainingSettings(cfg_training, online.upper_bound_under_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=online.lr, weight_decay=cfg_training.weight_decay)
    latency_statistics_set = False

    def ensure_embedded(dataset, indices):
        missing = [i for i in indices if dataset[i].query not in embeddings]
        if missing:
            embeddings.update(compute_embeddings(embedder, dataset, missing, device))

    def validate(round_index):
        ensure_embedded(val_dataset, validation_pool)
        model.eval()
        agent = AmortizedDPAgent(model, embed_fn=None, device=device, embedding_cache=embeddings)
        # Every planner's plan for a query is executed back to back, with the order rotated per
        # query, at EVERY validation: reusing old measurements of the fixed baselines would
        # compare cold first runs against later warm (page-cached) ones.
        kinds = (["ours"] + (["robust"] if robust_indexes is not None else [])
                 + (["native"] if validation.include_native else []) + (["dp"] if dp_plans else []))
        requests, tags, our_planning_ms, robust_planning_ms = [], [], [], []
        shared = {}
        for k, i in enumerate(validation_pool):
            data = val_dataset[i]
            start = time.perf_counter()
            plan = beam_search(Batch.from_data_list([data]), agent, 1)[0]["plan"]
            our_planning_ms.append(1000 * (time.perf_counter() - start))
            plans = {"ours": [int(p) for p in plan], "native": None, "dp": dp_plans.get(data.query)}
            if robust_indexes is not None:
                start = time.perf_counter()
                robust_agent = EpinetAmortizedDPAgent(model, embed_fn=None, epistemic_indexes=robust_indexes,
                                                      device=device, embedding_cache=embeddings,
                                                      objective=robust.objective, alpha_cvar=float(robust.alpha_cvar),
                                                      shared=shared)
                plans["robust"] = [int(p) for p in beam_search(Batch.from_data_list([data]), robust_agent, 1)[0]["plan"]]
                robust_planning_ms.append(1000 * (time.perf_counter() - start))
            rotation = k % len(kinds)
            for kind in kinds[rotation:] + kinds[:rotation]:
                requests.append({"query": data.query, "triple_patterns": list(data.triple_patterns),
                                 "plan": plans[kind], "timeout_s": float(online.timeouts.default_s)})
                tags.append((kind, data.query, plans[kind]))
        results = executor.execute(requests)
        by_kind = {kind: {} for kind in kinds}
        for (kind, query, plan), result in zip(tags, results):
            by_kind[kind][query] = (plan, result)
        queries = list(by_kind["ours"])

        def latencies(kind):
            return [by_kind[kind][q][1]["latency_s"] for q in queries]

        def censored(kind):
            return [by_kind[kind][q][1]["censored"] for q in queries]

        def cout(kind, query):
            plan, result = by_kind[kind][query]
            if kind == "native":
                if query not in native_cout:
                    value = tree_join_cout(result["tree"]) if not result["censored"] else None
                    if value is not None:
                        native_cout[query] = value
                return native_cout.get(query)
            return _plan_cout(plan, true_rows.get(query), result,
                              list(val_dataset[val_by_query[query]].triple_patterns))

        costs = {kind: {q: cout(kind, q) for q in queries} for kind in kinds}
        planners = {}
        for kind in kinds:
            planners[kind] = {
                "latency": _summarise_latencies(latencies(kind), censored(kind)),
                "p_error": ratio_summary([p_error(costs[kind][q], optimum.get(q)) for q in queries]),
            }
            if kind in history:
                for q in queries:
                    history[kind][q].append((by_kind[kind][q][1]["latency_s"], by_kind[kind][q][1]["censored"]))
                rolling = [np.mean([lat for lat, _ in history[kind][q]]) for q in queries]
                planners[kind]["latency_rolling"] = {
                    **_summarise_latencies(rolling, [np.mean([c for _, c in history[kind][q]]) for q in queries]),
                    "validations_averaged": len(history[kind][queries[0]]) if queries else 0}
        planners["ours"]["planning"] = _time_summary(our_planning_ms)
        if "robust" in planners:
            planners["robust"]["planning"] = _time_summary(robust_planning_ms)
        if "dp" in planners:
            planners["dp"]["planning"] = _time_summary(dp_planning_ms)

        ours_latency = latencies("ours")
        speedup, cout_ratio = {}, {}
        for mine in ("ours", "robust"):
            if mine not in kinds:
                continue
            prefix = "" if mine == "ours" else "robust_"
            for kind, name in (("native", "qlever"), ("dp", "dp_learned_cardinality")):
                if kind not in kinds:
                    continue
                rolling = [np.mean([lat for lat, _ in history[kind][q]]) for q in queries]
                speedup[f"{prefix}vs_{name}"] = _geomean_speedup(latencies(kind), latencies(mine))
                speedup[f"{prefix}vs_{name}_rolling"] = _geomean_speedup(rolling, latencies(mine))
                cout_ratio[f"{mine}_vs_{name}"] = ratio_summary(
                    [max(costs[mine][q], 1) / max(costs[kind][q], 1)
                     if costs[mine][q] is not None and costs[kind][q] is not None else None for q in queries])

        summary = {"round": round_index,
                   "executed_plans_total": sum(len(o.executions) for o in observations.values()),
                   # Flat fields as before (our plan vs QLever's plan in this validation).
                   **_summarise_latencies(ours_latency, censored("ours"),
                                          latencies("native") if "native" in kinds else None),
                   "planners": planners, "speedup": speedup, "cout_ratio": cout_ratio,
                   "optimum_known": sum(optimum.get(q) is not None for q in queries)}
        if robust_indexes is not None and true_rows:
            summary["epinet_card"] = _epinet_card_diagnostics(model, val_dataset, val_by_query, true_rows,
                                                              embeddings, robust_indexes, device)
        with open(run_directory / "validation.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        def fmt(value, spec=".3f"):
            return "n/a" if value is None else format(value, spec)

        print(f"[validation round {round_index}] {len(queries)} queries, optimum known for "
              f"{summary['optimum_known']}")
        for kind, label in (("ours", "ours  "), ("robust", "robust"), ("native", "QLever"), ("dp", "DP-LC ")):
            if kind not in planners:
                continue
            latency, perror = planners[kind]["latency"], planners[kind]["p_error"]
            rolling = planners[kind].get("latency_rolling")
            line = (f"  {label} latency mean {latency['mean_s']*1000:.1f} ms, median {latency['median_s']*1000:.1f} ms, "
                    f"p90 {latency['p90_s']*1000:.1f} ms, timeouts {latency['timeout_rate']:.1%}")
            if rolling:
                line += f" (rolling over {rolling['validations_averaged']}: mean {rolling['mean_s']*1000:.1f} ms)"
            line += (f" | p-error median {fmt(perror.get('median'))}, p90 {fmt(perror.get('p90'))}, "
                     f"geomean {fmt(perror.get('geomean'))}")
            if "planning" in planners[kind]:
                line += f" | planning median {planners[kind]['planning']['median_ms']:.1f} ms"
            print(line)
        for mine, prefix in (("ours", ""), ("robust", "robust_")):
            for name in ("qlever", "dp_learned_cardinality"):
                if f"{prefix}vs_{name}" in speedup:
                    ratio = cout_ratio[f"{mine}_vs_{name}"]
                    print(f"  {mine} vs {name}: speedup {fmt(speedup[f'{prefix}vs_{name}'])} "
                          f"(rolling {fmt(speedup[f'{prefix}vs_{name}_rolling'])}), C_out {mine}/{name} geomean "
                          f"{fmt(ratio.get('geomean'))}, {mine} <= on {fmt(ratio.get('frac_at_most_1'), '.1%')}")
        if "epinet_card" in summary and summary["epinet_card"].get("n", 0) >= 3:
            diag = summary["epinet_card"]
            print(f"  epinet card uncertainty: Spearman(std, |error|) {diag['spearman_std_vs_error']:.3f}, "
                  f"mean std {diag['mean_std']:.3f} vs mean |error| {diag['mean_abs_error']:.3f}, "
                  f"90% coverage {diag['coverage_90']:.1%} ({diag['n']} subsets)")
        torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "round": round_index},
                   run_directory / f"model-{round_index}.pt")
        with open(run_directory / "state.pkl", "wb") as f:
            pickle.dump({"observations": observations, "order_seen": order_seen, "round": round_index}, f)
        model.train()

    if online.validation.at_start:
        validate(0)

    cursor = 0
    for round_index in range(1, online.rounds + 1):
        start = time.perf_counter()
        batch = [train_pool[(cursor + k) % len(train_pool)] for k in range(online.queries_per_round)]
        cursor += online.queries_per_round
        ensure_embedded(train_dataset, batch)
        model.eval()
        agent = AmortizedDPAgent(model, embed_fn=None, device=device, embedding_cache=embeddings)
        shared = {}                     # value-net outputs shared by the Thompson samples of a query
        requests, is_greedy = [], []
        for i in batch:
            data = train_dataset[i]
            if data.query not in observations:
                observations[data.query] = _observations_for(data)
            order_seen.pop(data.query, None)
            order_seen[data.query] = None
            obs = observations[data.query]
            timeout_s = _timeout(obs, online.timeouts)
            if plan_selection == "thompson":
                plans = thompson_plans(model, data, online.plans_per_query, embeddings, device, thompson_generator,
                                       objective=online.get("decision_objective", "cout"), shared=shared)
            else:
                plans = deviation_plans(agent, Batch.from_data_list([data]), obs.labels, online.plans_per_query,
                                        online.deviation_steps, rng)
            for k, plan in enumerate(plans):
                requests.append({"query": data.query, "triple_patterns": list(data.triple_patterns),
                                 "plan": plan, "timeout_s": timeout_s})
                is_greedy.append(k == 0)
        planning_seconds = time.perf_counter() - start

        results = executor.execute(requests)
        execution_seconds = time.perf_counter() - start - planning_seconds
        aligned = cached = 0
        for request, result in zip(requests, results):
            parsed = parse_execution(result["tree"], request["plan"], request["triple_patterns"])
            aligned += parsed["aligned"]
            cached += parsed["any_cached"]
            observations[request["query"]].add(request["plan"], parsed, result["latency_s"], result["censored"],
                                               result["timeout_s"], cached=parsed["any_cached"])

        # Train on the most recently executed queries.
        window = list(order_seen)[-online.replay_queries:]
        window_obs = [observations[q] for q in window]
        tables = StateTables([o.labels for o in window_obs], [embeddings[q] for q in window], device,
                             patterns_list=[o.triple_patterns for o in window_obs],
                             observed_logcards=[o.logcard for o in window_obs], cost_to_go_upper_bound=True,
                             latency_to_go=[o.latency_to_go() for o in window_obs],
                             step_ms=[o.step_ms for o in window_obs])
        if not latency_statistics_set:
            model.set_latency_statistics(*tables.latency_statistics())
            model.set_step_statistics(*tables.step_statistics())
            latency_statistics_set = True
            if epinet_settings is not None:
                # Prior weights for the heads the offline run did not calibrate (latency, step).
                calibrated = calibrate_epinet_prior(model, tables, ["card", "cost_to_go", "latency", "step"],
                                                    epinet_settings)
                for head, (scale, alpha) in calibrated.items():
                    print(f"Epinet prior {head}: std {scale:.3f} -> alpha {alpha:.4f} "
                          f"(target {epinet_settings.prior_scale_target})")
                model_kwargs["epinet"] = model.epinet_config     # saved with every checkpoint
        model.train()
        running = []
        if online.get("gradient_steps_per_round") is not None:
            gradient_steps = int(online.gradient_steps_per_round)
        else:
            gradient_steps = max(1, math.ceil(online.gradient_passes_per_round * tables.n_states
                                              / cfg_training.states_per_batch))
        for _ in range(gradient_steps):
            state_ids = torch.randint(0, tables.n_states, (cfg_training.states_per_batch,), device=device)
            loss, parts = training_step(model, tables, state_ids, loss_settings, epinet_settings=epinet_settings)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.grad_clip)
            optimizer.step()
            running.append(parts)
        train_seconds = time.perf_counter() - start - planning_seconds - execution_seconds

        greedy = [r for r, first in zip(results, is_greedy) if first]
        summary = {
            "round": round_index, "queries_seen": len(order_seen), "plans_executed": len(requests),
            "timeout_rate": float(np.mean([r["censored"] for r in results])),
            "failed_rate": float(np.mean([bool(r.get("failed")) for r in results])),
            "aligned_rate": aligned / len(results), "cached_rate": cached / len(results),
            "greedy_mean_latency_s": float(np.mean([r["latency_s"] for r in greedy])),
            "greedy_timeout_rate": float(np.mean([r["censored"] for r in greedy])),
            "all_plans_mean_latency_s": float(np.mean([r["latency_s"] for r in results])),
            **{key: float(np.mean([p[key] for p in running if key in p])) for key in sorted(set().union(*running))},
            "plan_selection": plan_selection,
            "planning_seconds": planning_seconds, "execution_seconds": execution_seconds,
            "train_seconds": train_seconds, "window_states": tables.n_states, "gradient_steps": gradient_steps,
        }
        with open(run_directory / "rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        heads = ", ".join(f"{key[5:]} {summary[key]:.3f}" for key in ("loss_card", "loss_g", "loss_rank", "loss_latency",
                                                                      "loss_step") if key in summary)
        epinet_terms = sum(value for key, value in summary.items() if key.startswith("loss_epinet_"))
        if any(key.startswith("loss_epinet_") for key in summary):
            heads += f", epinet {epinet_terms:.3f}"
        print(f"Round {round_index}: {len(requests)} plans ({summary['timeout_rate']:.1%} timeouts, "
              f"{summary['aligned_rate']:.1%} aligned, {summary['cached_rate']:.1%} cached), loss {summary['loss']:.3f} "
              f"({heads}) | plan {planning_seconds:.0f}s, execute {execution_seconds:.0f}s, "
              f"train {train_seconds:.0f}s ({gradient_steps} steps over {tables.n_states} states)")
        if round_index % online.validation.every_rounds == 0 or round_index == online.rounds:
            validate(round_index)
    executor.close()


if __name__ == "__main__":
    main()
